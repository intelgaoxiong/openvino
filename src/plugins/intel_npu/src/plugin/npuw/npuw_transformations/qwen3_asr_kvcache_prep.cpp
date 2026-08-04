// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen3_asr_kvcache_prep.hpp"

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"

namespace opp = ov::pass::pattern;

namespace {

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

// ---------------------------------------------------------------------------
// Replaces:
//   Range_k(0, stop, 1) -> Unsqueeze -> Unsqueeze -> LessEqual(..., Q_pos)
// with:
//   attention_mask [1, capacity] -> Unsqueeze(1) -> Unsqueeze(2) -> Equal(_, 0)
//
// The Equal output has the same boolean semantics as LessEqual:
//   attention_mask[k] == 0 -> True  -> attend
//   attention_mask[k] != 0 -> False -> mask out
// ---------------------------------------------------------------------------
class Qwen3ASRAttentionMaskMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::Qwen3ASRAttentionMaskMatcher");

    explicit Qwen3ASRAttentionMaskMatcher(std::shared_ptr<ov::Model> model) {
        // Match: Range -> Unsqueeze x3 -> (optional Convert) -> LessEqual
        // 1D Range [N] needs 3 unsqueezes to reach [1,1,1,N] for the 4D causal mask.
        // An optional Convert may appear between the last Unsqueeze and LessEqual.
        auto range_k     = opp::wrap_type<ov::op::v4::Range>();
        auto unsq1       = opp::wrap_type<ov::op::v0::Unsqueeze>({range_k, opp::any_input()});
        auto unsq2       = opp::wrap_type<ov::op::v0::Unsqueeze>({unsq1,   opp::any_input()});
        auto unsq3       = opp::wrap_type<ov::op::v0::Unsqueeze>({unsq2,   opp::any_input()});
        auto opt_convert = opp::optional<ov::op::v0::Convert>({unsq3->output(0)});
        auto le          = opp::wrap_type<ov::op::v1::LessEqual>({opt_convert, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(le, this->get_type_info().name),
            [model, range_k](opp::Matcher& m) {
                auto le_node = m.get_match_root();

                // Guard: the Range start must be a Constant with value 0 (= Range_k, K positions)
                auto& pmap = m.get_pattern_value_map();
                auto range_node = pmap.at(range_k).get_node_shared_ptr();
                auto start_node = range_node->get_input_node_shared_ptr(0);
                if (start_node->get_type_name() != std::string("Constant")) {
                    return false;
                }
                auto start_const = ov::as_type_ptr<ov::op::v0::Constant>(start_node);
                if (!start_const)
                    return false;
                auto start_vals = start_const->cast_vector<int64_t>();
                if (start_vals.size() != 1 || start_vals[0] != 0) {
                    return false;
                }

                std::cout << "[Qwen3-ASR] Inject attention_mask: " << range_node->get_friendly_name() << std::endl;

                // Inject attention_mask [1, -1] parameter
                auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
                    ov::element::i64, ov::PartialShape{1, -1});
                attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                attention_mask->set_friendly_name("attention_mask");
                model->add_parameters({attention_mask});

                // Build replacement: Equal(Unsqueeze(Unsqueeze(attention_mask,1),2), 0)
                // attention_mask is [1,N] (2D); two unsqueezes give [1,1,1,N] (4D),
                // matching the original LessEqual output shape and polarity (True=attend).
                auto c0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                auto c1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
                auto c2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

                auto am_unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(attention_mask->output(0), c1);
                auto am_unsq2 = std::make_shared<ov::op::v0::Unsqueeze>(am_unsq1->output(0), c2);
                auto equal    = std::make_shared<ov::op::v1::Equal>(am_unsq2->output(0), c0);

                ov::replace_node(le_node, equal);
                return false;  // allow other matchers to continue
            });
    }
};

// ---------------------------------------------------------------------------
// Replaces the RoPE position path:
//   Gather(ShapeOf(param)) -> Range_q(Gather, Gather+N, 1) -> Reshape([1,1,-1])
// with:
//   position_ids [1] -> Reshape([1,1,-1])
//
// This fixes the frozen "position = capacity - 1" problem after ReshapeToStatic.
// ---------------------------------------------------------------------------
class Qwen3ASRPositionIdsMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::Qwen3ASRPositionIdsMatcher");

    explicit Qwen3ASRPositionIdsMatcher(std::shared_ptr<ov::Model> model) {
        // Match: Gather(ShapeOf(param)) as Range start, then Range -> Reshape
        auto gather   = opp::wrap_type<ov::op::v8::Gather>(
            {opp::any_input(), opp::any_input(), opp::any_input()});
        auto range_q  = opp::wrap_type<ov::op::v4::Range>(
            {gather, opp::any_input(), opp::any_input()});
        auto reshape  = opp::wrap_type<ov::op::v1::Reshape>({range_q, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(reshape, this->get_type_info().name),
            [model, gather](opp::Matcher& m) {
                auto reshape_node = m.get_match_root();

                // Guard: the Gather must consume a ShapeOf (= derives pos from tensor shape)
                auto& pmap = m.get_pattern_value_map();
                auto gather_node = pmap.at(gather).get_node_shared_ptr();
                auto gather_src = gather_node->get_input_node_shared_ptr(0);
                if (gather_src->get_type_name() != std::string("ShapeOf")) {
                    return false;
                }

                std::cout << "[Qwen3-ASR] Inject position_ids: " << reshape_node->get_friendly_name() << std::endl;

                // Inject position_ids [1] parameter
                auto position_ids = std::make_shared<ov::op::v0::Parameter>(
                    ov::element::i64, ov::Shape{1});
                position_ids->get_output_tensor(0).set_names({"position_ids"});
                position_ids->set_friendly_name("position_ids");
                model->add_parameters({position_ids});

                // Redirect: position_ids -> existing Reshape (preserves shape constant)
                reshape_node->input(0).replace_source_output(position_ids->output(0));
                return false;
            });
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // anonymous namespace

bool ov::npuw::Qwen3ASRAttentionMaskInput::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // Diagnostic: walk all LessEqual nodes and print their input chain so we can
    // verify the expected Range -> Unsqueeze x3 -> (Convert?) -> LessEqual pattern.
    std::cout << "[Qwen3-ASR][DBG] LessEqual nodes in kvcache model:" << std::endl;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_name() != std::string("LessEqual"))
            continue;
        std::cout << "  LessEqual: " << op->get_friendly_name() << std::endl;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            auto* src = op->get_input_node_ptr(i);
            std::cout << "    input[" << i << "] = " << src->get_type_name()
                      << " '" << src->get_friendly_name()
                      << "' shape=" << src->get_output_partial_shape(0) << std::endl;
            // Walk one more level up
            for (size_t j = 0; j < src->get_input_size(); ++j) {
                auto* src2 = src->get_input_node_ptr(j);
                std::cout << "      [" << j << "] " << src2->get_type_name()
                          << " '" << src2->get_friendly_name()
                          << "' shape=" << src2->get_output_partial_shape(0) << std::endl;
                for (size_t k = 0; k < src2->get_input_size(); ++k) {
                    auto* src3 = src2->get_input_node_ptr(k);
                    std::cout << "        [" << k << "] " << src3->get_type_name()
                              << " '" << src3->get_friendly_name()
                              << "' shape=" << src3->get_output_partial_shape(0) << std::endl;
                    for (size_t l = 0; l < src3->get_input_size(); ++l) {
                        auto* src4 = src3->get_input_node_ptr(l);
                        std::cout << "          [" << l << "] " << src4->get_type_name()
                                  << " '" << src4->get_friendly_name()
                                  << "' shape=" << src4->get_output_partial_shape(0) << std::endl;
                    }
                }
            }
        }
    }

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<Qwen3ASRAttentionMaskMatcher>(model);
    rewr.run_on_model(model);

    // Check whether the pass fired (attention_mask parameter should now exist).
    bool fired = false;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == "attention_mask") { fired = true; break; }
    }
    std::cout << "[Qwen3-ASR][DBG] Qwen3ASRAttentionMaskInput: "
              << (fired ? "MATCHED and replaced" : "NO MATCH - LessEqual pattern not found") << std::endl;

    if (fired)
        ov::pass::Validate().run_on_model(model);
    return true;
}

bool ov::npuw::Qwen3ASRPositionIdsInput::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<Qwen3ASRPositionIdsMatcher>(model);
    rewr.run_on_model(model);
    ov::pass::Validate().run_on_model(model);
    return true;
}
