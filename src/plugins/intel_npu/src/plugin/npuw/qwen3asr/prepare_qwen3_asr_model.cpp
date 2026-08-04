// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_qwen3_asr_model.hpp"

#include "../logging.hpp"
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

// ---------------------------------------------------------------------------
// Replaces:
//   Range_k(0, stop, 1) -> Unsqueeze x3 -> (optional Convert) -> LessEqual
// with:
//   attention_mask [1, capacity] -> Unsqueeze(1) -> Unsqueeze(2) -> Equal(_, 0)
//
// Boolean semantics preserved:
//   attention_mask[k] == 0 -> True  -> attend
//   attention_mask[k] != 0 -> False -> mask out
// ---------------------------------------------------------------------------
class Qwen3ASRAttentionMaskMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::Qwen3ASRAttentionMaskMatcher");

    explicit Qwen3ASRAttentionMaskMatcher(std::shared_ptr<ov::Model> model) {
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

                // Guard: Range start must be Constant 0 (= K-side positions)
                auto& pmap = m.get_pattern_value_map();
                auto range_node = pmap.at(range_k).get_node_shared_ptr();
                auto start_node = range_node->get_input_node_shared_ptr(0);
                auto start_const = ov::as_type_ptr<ov::op::v0::Constant>(start_node);
                if (!start_const)
                    return false;
                auto start_vals = start_const->cast_vector<int64_t>();
                if (start_vals.size() != 1 || start_vals[0] != 0)
                    return false;

                // Inject attention_mask [1, -1] parameter
                auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
                    ov::element::i64, ov::PartialShape{1, -1});
                attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                attention_mask->set_friendly_name("attention_mask");
                model->add_parameters({attention_mask});

                // Build Equal(Unsqueeze(Unsqueeze(attention_mask, 1), 2), 0)
                // [1,N] -> [1,1,1,N] to match original 4D causal mask shape.
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
// ---------------------------------------------------------------------------
class Qwen3ASRPositionIdsMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::Qwen3ASRPositionIdsMatcher");

    explicit Qwen3ASRPositionIdsMatcher(std::shared_ptr<ov::Model> model) {
        auto gather  = opp::wrap_type<ov::op::v8::Gather>(
            {opp::any_input(), opp::any_input(), opp::any_input()});
        auto range_q = opp::wrap_type<ov::op::v4::Range>(
            {gather, opp::any_input(), opp::any_input()});
        auto reshape = opp::wrap_type<ov::op::v1::Reshape>({range_q, opp::any_input()});

        register_matcher(
            std::make_shared<opp::Matcher>(reshape, this->get_type_info().name),
            [model, gather](opp::Matcher& m) {
                auto reshape_node = m.get_match_root();

                // Guard: Gather must consume a ShapeOf (= derives position from tensor shape)
                auto& pmap = m.get_pattern_value_map();
                auto gather_node = pmap.at(gather).get_node_shared_ptr();
                auto gather_src = gather_node->get_input_node_shared_ptr(0);
                if (gather_src->get_type_name() != std::string("ShapeOf"))
                    return false;

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

}  // anonymous namespace

// ---------------------------------------------------------------------------
// PrepareQwen3ASRModel — Step 1 (pre-clone)
// Removes residual ReadValue/Assign state nodes left by StatefulToStateless
// for states whose variable id does not match the standard naming convention
// (e.g. encoder_hidden_states state in cross-attention layers).
// ---------------------------------------------------------------------------
bool ov::npuw::PrepareQwen3ASRModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("[Qwen3-ASR] Removing residual state tensors.");

    const auto ops_snapshot = model->get_ops();
    for (const auto& op : ops_snapshot) {
        auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(op);
        if (!read_value)
            continue;
        const auto var_id = read_value->get_variable_id();
        std::shared_ptr<ov::Node> replacement;

        if (read_value->get_input_size() > 0) {
            auto init_node = read_value->get_input_node_shared_ptr(0);
            if (ov::as_type_ptr<ov::op::v0::Parameter>(init_node)) {
                // Initial value is already a model Parameter — reuse it directly.
                replacement = init_node;
                LOG_DEBUG("[Qwen3-ASR]   Replaced ReadValue '" << var_id << "' with existing Parameter.");
            }
        }

        if (!replacement) {
            // Initial value is a Constant or missing — create a new Parameter.
            auto param = std::make_shared<ov::op::v0::Parameter>(read_value->get_output_element_type(0),
                                                                  read_value->get_output_partial_shape(0));
            param->get_output_tensor(0).set_names({var_id});
            param->set_friendly_name(var_id);
            model->add_parameters({param});
            replacement = param;
            LOG_DEBUG("[Qwen3-ASR]   Replaced ReadValue '" << var_id << "' with new Parameter.");
        }

        ov::replace_node(read_value, replacement);
    }

    // Remove remaining Assign sinks
    std::vector<std::shared_ptr<ov::Node>> sinks_to_remove;
    for (const auto& sink : model->get_sinks()) {
        if (ov::as_type_ptr<ov::op::util::AssignBase>(sink))
            sinks_to_remove.push_back(sink);
    }
    for (auto& s : sinks_to_remove)
        model->remove_sink(ov::as_type_ptr<ov::op::Sink>(s));

    // Remove variable metadata
    for (const auto& var : model->get_variables())
        model->remove_variable(var);

    return true;
}

// ---------------------------------------------------------------------------
// PrepareQwen3ASRKVCacheModel — Step 2 (post-clone, kvcache model only)
// Injects attention_mask and position_ids Parameters for O(1) generation.
// ---------------------------------------------------------------------------
bool ov::npuw::PrepareQwen3ASRKVCacheModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    LOG_DEBUG("[Qwen3-ASR] Injecting attention_mask and position_ids into kvcache model.");

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<Qwen3ASRAttentionMaskMatcher>(model);
    rewr.add_matcher<Qwen3ASRPositionIdsMatcher>(model);
    rewr.run_on_model(model);

    ov::pass::Validate().run_on_model(model);
    return true;
}
