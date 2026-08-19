// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detect_causal_mask.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <queue>
#include <regex>
#include <string>
#include <unordered_set>

#include "../logging.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

// Matches e.g. "...layers.5.self_attn..." -> layer index 5. Same convention as
// patch_sliding_window_kvcache.cpp's layer_id_regex()/try_parse_layer_idx().
const std::regex& layer_id_regex() {
    static const std::regex re(R"(layers\.(\d+)\.self_attn)");
    return re;
}

bool try_parse_layer_idx(const std::string& text, size_t& out_idx) {
    std::smatch m;
    if (!std::regex_search(text, m, layer_id_regex())) {
        return false;
    }
    out_idx = static_cast<size_t>(std::stoul(m[1].str()));
    return true;
}

// Builds the Range chain shared by causal/sliding window mask pattern below:
//
//   Range(start, stop, step)
//     -> opt Add(range, offset)
//     -> opt Unsqueeze (up to 3x)
//     -> opt Reshape
//     -> opt Convert
//
// Real model shapes covered by this single chain:
//   Range -> Unsqueeze x(0-3) -> opt Convert          (Llama, Tril, Whisper)
//   Range -> Add(range, offset) -> Reshape            (MiniCPM, Q side)
//   Range -> Add(range, offset) -> Unsqueeze x3       (Gemma-4 cache_position)
std::shared_ptr<ov::Node> make_range_chain() {
    auto range = opp::wrap_type<ov::op::v4::Range>({opp::any_input(), opp::any_input(), opp::any_input()});
    auto add = opp::optional<ov::op::v1::Add>({range, opp::any_input()});
    auto unsqueeze1 = opp::optional<ov::op::v0::Unsqueeze>({add, opp::any_input()});
    auto unsqueeze2 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze1, opp::any_input()});
    auto unsqueeze3 = opp::optional<ov::op::v0::Unsqueeze>({unsqueeze2, opp::any_input()});
    auto reshape = opp::optional<ov::op::v1::Reshape>({unsqueeze3, opp::any_input()});
    auto convert = opp::optional<ov::op::v0::Convert>({reshape});
    return convert;
}

int64_t get_window_size(const std::shared_ptr<ov::Node>& node) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant)
        return 0;
    const auto vals = constant->cast_vector<int64_t>();
    return vals.empty() ? 0 : std::llabs(vals.front());
}

// Writes `encoded_value` (see NPUW_SDPA_MASK_RT_KEY for the encoding) onto `sdpa`'s
// rt_info, as a best-effort side annotation consumed by log_detected_masks() (and
// any future per-SDPA-aware code). Follows the same never-downgrade-to-Causal
// precedence rule as record_per_layer_mask_info() below: an already-recorded
// SlidingWindow is never overwritten with Causal for the same node. This is a
// convenience annotation only - m_mask_info/m_layer_mask_info (the source of
// truth for existing consumers) are populated directly by the matcher callbacks,
// not read back from rt_info.
void assign_mask_rt_info(const std::shared_ptr<ov::Node>& sdpa, int64_t encoded_value) {
    auto& rt_info = sdpa->get_rt_info();
    const auto it = rt_info.find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    if (it != rt_info.end() && it->second.as<int64_t>() >= 0) {
        return;  // already SlidingWindow - never downgrade
    }
    rt_info[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = encoded_value;
}

// Walks forward from `start` through consumer edges (BFS, stopping the moment an
// SDPA node is reached on a given path) to find every ScaledDotProductAttention
// node that transitively consumes the mask value produced at `start`. A mask
// subgraph can be CSE-shared and therefore feed more than one layer's SDPA -
// all of them are returned. Bounded by the (small) forward cone between a mask
// comparison chain and its consuming SDPA node(s), not by overall graph size.
std::vector<std::shared_ptr<ov::Node>> find_consuming_sdpas(const ov::Output<ov::Node>& start) {
    std::vector<std::shared_ptr<ov::Node>> found;
    std::unordered_set<ov::Node*> visited;
    std::queue<ov::Output<ov::Node>> to_visit;
    to_visit.push(start);
    while (!to_visit.empty()) {
        const auto out = to_visit.front();
        to_visit.pop();
        for (const auto& input : out.get_target_inputs()) {
            ov::Node* consumer = input.get_node();
            if (!visited.insert(consumer).second) {
                continue;
            }
            if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(consumer)) {
                found.push_back(consumer->shared_from_this());
                continue;  // don't traverse past SDPA
            }
            for (auto& consumer_out : consumer->outputs()) {
                to_visit.push(consumer_out);
            }
        }
    }
    return found;
}

// Records `info` for every layer whose SDPA (transitively) consumes the mask
// value produced at `anchor`, into `layer_info`. Mirrors the whole-model
// aggregate's precedence rule: never downgrade an already-recorded
// SlidingWindow to Causal for the same layer. Also writes the same info as a
// best-effort rt_info annotation directly on the consuming SDPA node(s) - see
// assign_mask_rt_info() above.
void record_per_layer_mask_info(const ov::Output<ov::Node>& anchor,
                                 const ov::npuw::MaskInfo& info,
                                 std::map<size_t, ov::npuw::MaskInfo>& layer_info) {
    const int64_t encoded = (info.mask_type == ov::npuw::MaskInfo::MaskType::SlidingWindow)
                                 ? info.window_size
                                 : ov::npuw::NPUW_SDPA_MASK_CAUSAL;
    for (const auto& sdpa : find_consuming_sdpas(anchor)) {
        assign_mask_rt_info(sdpa, encoded);
        size_t layer_idx = 0;
        if (!try_parse_layer_idx(sdpa->get_friendly_name(), layer_idx)) {
            continue;
        }
        auto& existing = layer_info[layer_idx];
        if (existing.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow) {
            existing = info;
        }
    }
}

// True when `node` is (or, through single-input passthrough ops like Unsqueeze/
// Reshape/Convert/Broadcast, transitively wraps) a Gemma-4-12B-style decomposed
// sliding-window bound check: GreaterEqual(Subtract(...), window_const). Used by
// TriuCausalMatcher below as a negative guard so it doesn't also fire on
// TriuSlidingMatcher's own GreaterEqual/Select anchor (Gemma-4-12B's traced
// torch.triu()/masked_fill() decomposition uses GreaterEqual + Select instead of
// LessEqual/Greater + BitwiseAnd, so both matchers share the same "Select(
// GreaterEqual(...), any, any)" pattern shape and must be told apart by what
// feeds the GreaterEqual).
bool contains_triu_window_check(const std::shared_ptr<ov::Node>& node) {
    if (auto ge = ov::as_type_ptr<ov::op::v1::GreaterEqual>(node)) {
        return ov::is_type<ov::op::v1::Subtract>(ge->get_input_node_shared_ptr(0)) ||
               ov::is_type<ov::op::v1::Subtract>(ge->get_input_node_shared_ptr(1));
    }
    if (ov::is_type<ov::op::v0::Unsqueeze>(node) || ov::is_type<ov::op::v1::Reshape>(node) ||
        ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v3::Broadcast>(node)) {
        return contains_triu_window_check(node->get_input_node_shared_ptr(0));
    }
    return false;
}

// True when `node` is (or, transitively through single-input/pass-through ops -
// Add, Unsqueeze, Reshape, Convert, Broadcast - within `depth` steps) derived
// from a Range op. Used by TriuCausalMatcher below as a post-match guard: unlike
// make_range_chain() (which anchors the LessEqual/Less family to an *exact*
// Range->Add->Unsqueeze->Reshape->Convert op ordering), this walks the graph
// looking for *any* Range reachable within a bounded number of hops, so it still
// rejects arbitrary/unrelated GreaterEqual(any_input, any_input) matches without
// having to match Gemma-4-12B's specific chain shape one op at a time. `depth` is
// capped to keep the walk local to the comparison's immediate operands.
bool traces_to_range(const std::shared_ptr<ov::Node>& node, int depth = 8) {
    if (!node || depth <= 0)
        return false;
    if (ov::is_type<ov::op::v4::Range>(node))
        return true;
    if (ov::is_type<ov::op::v1::Add>(node) || ov::is_type<ov::op::v0::Unsqueeze>(node) ||
        ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v0::Convert>(node) ||
        ov::is_type<ov::op::v3::Broadcast>(node)) {
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            if (traces_to_range(node->get_input_node_shared_ptr(i), depth - 1))
                return true;
        }
    }
    return false;
}

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

// ============================================================================
// Matches: ScaledDotProductAttention(is_causal=true)
//
// The case when causality is an SDPA attribute, not an explicit mask
// subgraph.
// ============================================================================
class SDPACausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::SDPACausalMatcher");
    explicit SDPACausalMatcher(ov::npuw::MaskInfo& mask_info, std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>();
        auto callback = [&mask_info, &layer_mask_info](opp::Matcher& m) {
            auto node = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(m.get_match_root());
            if (node && node->get_causal()) {
                if (mask_info.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                    mask_info = {ov::npuw::MaskInfo::MaskType::Causal, 0};
                // The anchor IS the SDPA node itself here (is_causal is its own
                // attribute, not a fed-in mask value) - annotate it directly and
                // parse the layer index straight from it instead of walking
                // forward.
                assign_mask_rt_info(node, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
                size_t layer_idx = 0;
                if (try_parse_layer_idx(node->get_friendly_name(), layer_idx)) {
                    auto& existing = layer_mask_info[layer_idx];
                    if (existing.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                        existing = {ov::npuw::MaskInfo::MaskType::Causal, 0};
                }
            }
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(sdpa, "DetectSDPACausal"), callback);
    }
};

// ============================================================================
// Matches: LessEqual|Less(K = range_chain, Q = range_chain)
// Two Range chains compared directly,
// with no extra offset between K and Q. This is the most common causal-mask
// shape, seen (with minor chain variations) in:
//   - Llama    : Range -> Unsqueeze x3
//   - Tril
//   - MiniCPM  : Less(Range, Reshape(Add(Range, offset)))
//   - Whisper  : Range -> Unsqueeze x3
//
// ============================================================================
class StandardCausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::StandardCausalMatcher");
    explicit StandardCausalMatcher(ov::npuw::MaskInfo& mask_info,
                                    std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto cmp = opp::wrap_type<ov::op::v1::LessEqual, ov::op::v1::Less>({make_range_chain(), make_range_chain()});
        auto callback = [&mask_info, &layer_mask_info](opp::Matcher& m) {
            if (mask_info.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                mask_info = {ov::npuw::MaskInfo::MaskType::Causal, 0};
            record_per_layer_mask_info(m.get_match_value(), {ov::npuw::MaskInfo::MaskType::Causal, 0}, layer_mask_info);
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(cmp, "StandardCausal"), callback);
    }
};

// ============================================================================
// Matches: LessEqual|Less(K = range_chain, Q = Add(any, range_chain))
//
// StandardCausalMatcher with extra Add: Add(cache_len, range_chain), with the range chain as the
// Add's 2nd input.
// ============================================================================
class Qwen3CausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::Qwen3CausalMatcher");
    explicit Qwen3CausalMatcher(ov::npuw::MaskInfo& mask_info, std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto add = opp::wrap_type<ov::op::v1::Add>({opp::any_input(), make_range_chain()});
        auto cmp = opp::wrap_type<ov::op::v1::LessEqual, ov::op::v1::Less>({make_range_chain(), add});
        auto callback = [&mask_info, &layer_mask_info](opp::Matcher& m) {
            if (mask_info.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                mask_info = {ov::npuw::MaskInfo::MaskType::Causal, 0};
            record_per_layer_mask_info(m.get_match_value(), {ov::npuw::MaskInfo::MaskType::Causal, 0}, layer_mask_info);
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(cmp, "Qwen3Causal"), callback);
    }
};

// ============================================================================
// Matches a generic sliding-window mask, built from two comparisons ANDed
// together:
//
//   window_check = Greater(K, Add(Q, neg_window))
//   causal_check = LessEqual(K, Q)
//   mask         = BitwiseAnd(BitwiseAnd(any, window_check), causal_check)
//
// Covers: Phi-3 / Gemma-2 / Gemma-3 / Gemma-4 models.
// ============================================================================
class BitwiseAndSlidingMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::BitwiseAndSlidingMatcher");
    explicit BitwiseAndSlidingMatcher(ov::npuw::MaskInfo& mask_info,
                                       std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto q_chain = make_range_chain();
        auto k_chain = make_range_chain();
        auto window_constant = opp::wrap_type<ov::op::v0::Constant>();
        auto add = opp::wrap_type<ov::op::v1::Add>({q_chain, window_constant});
        auto greater = opp::wrap_type<ov::op::v1::Greater>({k_chain, add});
        auto and_win = opp::wrap_type<ov::op::v13::BitwiseAnd>({opp::any_input(), greater});
        auto causal = opp::wrap_type<ov::op::v1::LessEqual>({k_chain, q_chain});
        auto anchor = opp::wrap_type<ov::op::v13::BitwiseAnd>({and_win, causal});
        auto callback = [&mask_info, &layer_mask_info, window_constant](opp::Matcher& m) {
            const int64_t window_size =
                get_window_size(m.get_pattern_value_map().at(window_constant).get_node_shared_ptr());
            if (window_size > 0) {
                const ov::npuw::MaskInfo info{ov::npuw::MaskInfo::MaskType::SlidingWindow, window_size};
                mask_info = info;
                record_per_layer_mask_info(m.get_match_value(), info, layer_mask_info);
            }
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(anchor, "BitwiseAndSliding"), callback);
    }
};

// ============================================================================
// Matches the legacy Phi-3 inverted sliding-window mask:
//
//   K = Convert(Convert(Range(0, atten_mask_len, step)))   // K_f32
//   Q = Reshape(Range(past, full_ctx, step), [-1, 1])      // Q_col
//
//   causal_check  = Greater(K, Q)
//   sliding_check = LessEqual(K, Add(Q, neg_window))
//   mask          = BitwiseOr(causal_check, sliding_check)
//
// ============================================================================
class OldPhi3SlidingMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::OldPhi3SlidingMatcher");
    explicit OldPhi3SlidingMatcher(ov::npuw::MaskInfo& mask_info,
                                    std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto k_constant = opp::wrap_type<ov::op::v0::Constant>();
        auto gather = opp::wrap_type<ov::op::v8::Gather>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto k_range = opp::wrap_type<ov::op::v4::Range>({k_constant, gather, opp::any_input()});
        auto k_convert = opp::wrap_type<ov::op::v0::Convert>({k_range});
        auto k_f32 = opp::wrap_type<ov::op::v0::Convert>({k_convert});
        auto q_range = opp::wrap_type<ov::op::v4::Range>({opp::any_input(), opp::any_input(), opp::any_input()});
        auto q_reshape = opp::wrap_type<ov::op::v1::Reshape>({q_range, opp::any_input()});
        auto q_constant = opp::wrap_type<ov::op::v0::Constant>();
        auto q_add = opp::wrap_type<ov::op::v1::Add>({q_reshape, q_constant});
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({k_f32, q_reshape});
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({k_f32, q_add});
        auto anchor = opp::wrap_type<ov::op::v13::BitwiseOr>({sliding_mask, causal_mask});

        auto callback = [=, &mask_info, &layer_mask_info](opp::Matcher& m) {
            const int64_t w = get_window_size(m.get_pattern_value_map().at(q_constant).get_node_shared_ptr());
            if (w > 0) {
                const ov::npuw::MaskInfo info{ov::npuw::MaskInfo::MaskType::SlidingWindow, w};
                mask_info = info;
                record_per_layer_mask_info(m.get_match_value(), info, layer_mask_info);
            }
            return false;
        };

        register_matcher(std::make_shared<opp::Matcher>(anchor, "OldPhi3Sliding"), callback);
    }
};

// ============================================================================
// Matches the default float sliding-window mask:
//
//   causal_check  = LessEqual(K, Q)
//   sliding_check = Greater(K, Subtract(Q, window))
//   mask          = LogicalAnd(causal_check, sliding_check)
//
// ============================================================================
class DefaultSWAMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::DefaultSWAMatcher");
    explicit DefaultSWAMatcher(ov::npuw::MaskInfo& mask_info, std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto k_chain = make_range_chain();
        auto q_chain = make_range_chain();
        auto window_const = opp::wrap_type<ov::op::v0::Constant>();
        auto causal_mask = opp::wrap_type<ov::op::v1::LessEqual>({k_chain, q_chain});
        auto subtract = opp::wrap_type<ov::op::v1::Subtract>({q_chain, window_const});
        auto sliding_mask = opp::wrap_type<ov::op::v1::Greater>({k_chain, subtract});
        auto anchor = opp::wrap_type<ov::op::v1::LogicalAnd>({causal_mask, sliding_mask});
        auto callback = [=, &mask_info, &layer_mask_info](opp::Matcher& m) {
            const int64_t w = get_window_size(m.get_pattern_value_map().at(window_const).get_node_shared_ptr());
            if (w > 0) {
                const ov::npuw::MaskInfo info{ov::npuw::MaskInfo::MaskType::SlidingWindow, w};
                mask_info = info;
                record_per_layer_mask_info(m.get_match_value(), info, layer_mask_info);
            }
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(anchor, "DefaultSWA"), callback);
    }
};

// ============================================================================
// Matches Gemma-4-12B's decomposed torch.triu(...)-style causal mask:
//
//   Select(GreaterEqual(row, col), any_input, any_input)
//
// Unlike every other causal matcher above (LessEqual/Less feeding a boolean
// combine op), Gemma-4-12B's trace decomposes torch.triu()/masked_fill() into a
// GreaterEqual boolean feeding a Select that directly picks between the
// unmasked (0) and masked (-inf) float fill values - there is no
// BitwiseAnd/BitwiseOr/LogicalAnd anywhere in this family. Row/col operands are
// left as any_input() rather than make_range_chain(), since this export's
// Range/Unsqueeze/Add ordering doesn't match that chain's grammar.
//
// The GreaterEqual/Select shape alone is too permissive (it also matches
// TriuSlidingMatcher's own anchor below) - nothing here ties `ge` to actual
// position indices the way make_range_chain() does for the other matchers. The
// callback additionally requires: the Select's output to be a floating-point
// tensor (a real mask always selects between float fill values, 0 / -inf); that
// at least one of `ge`'s operands (transitively) derives from a Range op (see
// traces_to_range above), i.e. is a genuine row/col position-index computation;
// and that `ge` is NOT itself a sliding-window bound check (see
// contains_triu_window_check above) - which rules out re-matching
// TriuSlidingMatcher's GreaterEqual(Subtract(row, col), window_const) anchor
// here.
// ============================================================================
class TriuCausalMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::TriuCausalMatcher");
    explicit TriuCausalMatcher(ov::npuw::MaskInfo& mask_info, std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto ge = opp::wrap_type<ov::op::v1::GreaterEqual>({opp::any_input(), opp::any_input()});
        auto sel = opp::wrap_type<ov::op::v1::Select>({ge, opp::any_input(), opp::any_input()});
        auto callback = [&mask_info, &layer_mask_info, ge](opp::Matcher& m) {
            auto root = m.get_match_root();
            if (!root->get_output_element_type(0).is_real())
                return false;
            auto ge_node = m.get_pattern_value_map().at(ge).get_node_shared_ptr();
            if (contains_triu_window_check(ge_node))
                return false;  // owned by TriuSlidingMatcher, not a plain causal mask
            if (!traces_to_range(ge_node->get_input_node_shared_ptr(0)) &&
                !traces_to_range(ge_node->get_input_node_shared_ptr(1)))
                return false;
            if (mask_info.mask_type != ov::npuw::MaskInfo::MaskType::SlidingWindow)
                mask_info = {ov::npuw::MaskInfo::MaskType::Causal, 0};
            record_per_layer_mask_info(root->output(0), {ov::npuw::MaskInfo::MaskType::Causal, 0}, layer_mask_info);
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(sel, "TriuCausal"), callback);
    }
};

// ============================================================================
// Matches Gemma-4-12B's decomposed sliding-window "beyond window" overwrite:
//
//   diff          = Subtract(row, col)
//   beyond_window = GreaterEqual(diff, window_const)
//   windowed      = Select(opt Unsqueeze x2(beyond_window), any_input, any_input)
//
// `windowed`'s data operands are the plain triu-causal mask matched by
// TriuCausalMatcher above (as either its then or else operand - Gemma-4-12B
// puts it in the "else" slot, but that's not load-bearing here) and a fill
// constant; this Select overwrites the causal mask with the fill value
// wherever the key is further back than `window_const` positions. Anchored
// independently of TriuCausalMatcher, same as BitwiseAndSlidingMatcher/
// DefaultSWAMatcher above vs. the LessEqual family.
// ============================================================================
class TriuSlidingMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::TriuSlidingMatcher");
    explicit TriuSlidingMatcher(ov::npuw::MaskInfo& mask_info, std::map<size_t, ov::npuw::MaskInfo>& layer_mask_info) {
        auto diff = opp::wrap_type<ov::op::v1::Subtract>({opp::any_input(), opp::any_input()});
        auto window_const = opp::wrap_type<ov::op::v0::Constant>();
        auto beyond_window = opp::wrap_type<ov::op::v1::GreaterEqual>({diff, window_const});
        auto beyond_window_unsq1 = opp::optional<ov::op::v0::Unsqueeze>({beyond_window, opp::any_input()});
        auto beyond_window_unsq2 = opp::optional<ov::op::v0::Unsqueeze>({beyond_window_unsq1, opp::any_input()});
        auto windowed = opp::wrap_type<ov::op::v1::Select>({beyond_window_unsq2, opp::any_input(), opp::any_input()});
        auto callback = [&mask_info, &layer_mask_info, window_const](opp::Matcher& m) {
            if (!m.get_match_root()->get_output_element_type(0).is_real())
                return false;
            const int64_t w = get_window_size(m.get_pattern_value_map().at(window_const).get_node_shared_ptr());
            if (w > 0) {
                const ov::npuw::MaskInfo info{ov::npuw::MaskInfo::MaskType::SlidingWindow, w};
                mask_info = info;
                record_per_layer_mask_info(m.get_match_value(), info, layer_mask_info);
            }
            return false;
        };
        register_matcher(std::make_shared<opp::Matcher>(windowed, "TriuSliding"), callback);
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

namespace ov::npuw {

bool DetectAttentionMask::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_mask_info = MaskInfo{};
    m_layer_mask_info.clear();

    // Sliding-window matchers are registered before the causal ones so that,
    // when a mask subgraph is CSE-shared between a genuine SWA anchor and a
    // looser causal comparison feeding the same SDPA, the never-downgrade
    // precedence rule in record_per_layer_mask_info()/the direct mask_info
    // writes above already has the correct SlidingWindow verdict recorded
    // before a causal matcher could (incorrectly) downgrade it.
    ov::pass::GraphRewrite detector;
    detector.add_matcher<BitwiseAndSlidingMatcher>(m_mask_info, m_layer_mask_info);
    detector.add_matcher<OldPhi3SlidingMatcher>(m_mask_info, m_layer_mask_info);
    detector.add_matcher<DefaultSWAMatcher>(m_mask_info, m_layer_mask_info);
    detector.add_matcher<TriuSlidingMatcher>(m_mask_info, m_layer_mask_info);
    detector.add_matcher<SDPACausalMatcher>(m_mask_info, m_layer_mask_info);
    detector.add_matcher<StandardCausalMatcher>(m_mask_info, m_layer_mask_info);
    detector.add_matcher<Qwen3CausalMatcher>(m_mask_info, m_layer_mask_info);
    detector.add_matcher<TriuCausalMatcher>(m_mask_info, m_layer_mask_info);
    detector.run_on_model(model);

    return false;
}

void log_detected_masks(const std::shared_ptr<ov::Model>& model) {
    if (ov::npuw::get_log_level() < ov::npuw::LogLevel::Debug) {
        return;
    }

    // Best-effort: pull a transformer layer index out of the SDPA node's friendly
    // name (HF/ONNX exports commonly retain the originating module's scope, e.g.
    // "__module.model.layers.4.self_attn/aten::scaled_dot_product_attention").
    // Falls back to topological position when no such index can be found (e.g. in
    // standalone/synthetic test graphs).
    static const std::regex layer_idx_re(R"([Ll]ayers?[._/]([0-9]+))");

    struct Entry {
        std::string name;
        std::string type;
        int64_t sort_key;
    };
    std::vector<Entry> entries;

    int64_t position = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
        if (!sdpa)
            continue;

        std::string type;
        const auto& rt_info = sdpa->get_rt_info();
        const auto it = rt_info.find(NPUW_SDPA_MASK_RT_KEY);
        if (it == rt_info.end()) {
            type = "Unknown";
        } else {
            const auto encoded = it->second.as<int64_t>();
            type = (encoded < 0) ? "Causal" : ("SlidingWindow(" + std::to_string(encoded) + ")");
        }

        const auto& name = sdpa->get_friendly_name();
        std::smatch match;
        int64_t sort_key = position;
        if (std::regex_search(name, match, layer_idx_re) && match.size() > 1) {
            const auto& digits = match[1].str();
            int64_t parsed = 0;
            const auto res = std::from_chars(digits.data(), digits.data() + digits.size(), parsed);
            if (res.ec == std::errc{}) {
                sort_key = parsed;
            }
        }
        entries.push_back({name, std::move(type), sort_key});
        ++position;
    }

    std::stable_sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
        return lhs.sort_key < rhs.sort_key;
    });
    for (const auto& entry : entries) {
        LOG_DEBUG("layer " << entry.sort_key << " (" << entry.name << "): " << entry.type);
    }
}

}  // namespace ov::npuw

