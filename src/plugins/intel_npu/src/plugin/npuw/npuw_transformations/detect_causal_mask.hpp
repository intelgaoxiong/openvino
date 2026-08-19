// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Detected attention mask type and (for sliding window) its window size.
struct MaskInfo {
    // No recognized mask pattern (e.g. full attention), plain causal, or
    // causal + sliding-window (local attention).
    enum class MaskType : int { Unknown = 0, Causal, SlidingWindow };

    // Value-initialized to MaskType::Unknown (== 0).
    MaskType mask_type{};
    // Valid only when mask_type == SlidingWindow.
    int64_t window_size = 0;
};

// Analysis pass: detects the attention mask type of each SDPA node in the model by
// inspecting its mask-construction subgraph (Range/LessEqual/Greater/BitwiseAnd, or
// the Select-based "triu" family) or its is_causal attribute, and annotates the
// node's rt_info[NPUW_SDPA_MASK_RT_KEY] accordingly (see the encoding documented
// below). A node may only be annotated once - a matcher that would overwrite an
// already-annotated node with a conflicting kind asserts instead, since that
// indicates genuinely contradictory evidence (e.g. an is_causal=true SDPA fed an
// explicit sliding-window mask), not just two matchers agreeing.
//
// run_on_model() never modifies the model (always returns false); after running,
// it also derives two convenience views from the per-SDPA rt_info it just wrote,
// retained for pre-existing consumers - get_mask_info() (whole-model aggregate)
// and get_layer_mask_info() (per-layer map, see below).
class DetectAttentionMask : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::DetectAttentionMask");
    DetectAttentionMask() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    // Whole-model aggregate result. Preserves the pre-existing behavior (last
    // matching matcher wins, never downgrading an already-found SlidingWindow to
    // Causal) so existing consumers (e.g. HFA mask-skipping) are unaffected.
    const MaskInfo& get_mask_info() const {
        return m_mask_info;
    }

    // Per-layer detection result, keyed by decoder layer index. The layer index
    // is parsed from the friendly_name of the ScaledDotProductAttention node(s)
    // that (transitively) consume each matched mask subgraph - not from the
    // matched subgraph's own name, since a mask subgraph can be CSE-shared across
    // multiple structurally-identical layers and therefore feed more than one
    // SDPA. Layers whose mask subgraph wasn't recognized by any matcher, or whose
    // consuming SDPA's name doesn't carry a parseable layer index, are absent
    // from the map (equivalent to MaskType::Unknown).
    const std::map<size_t, MaskInfo>& get_layer_mask_info() const {
        return m_layer_mask_info;
    }

private:
    MaskInfo m_mask_info;
    std::map<size_t, MaskInfo> m_layer_mask_info;
};

void log_detected_masks(const std::shared_ptr<ov::Model>& model);

// rt_info key written by DetectAttentionMask onto each ScaledDotProductAttention
// node. Value type: int64_t, encoding both the mask kind and (for sliding window)
// its window size in a single slot:
//   * key absent   -> Unknown (no recognized mask pattern, e.g. full/bidirectional
//                     attention)
//   * value <  0   -> Causal (equivalent to a sliding window whose size covers the
//                     whole context, i.e. "infinite window")
//   * value >= 0   -> SlidingWindow, value is the window size
static constexpr const char* NPUW_SDPA_MASK_RT_KEY = "npuw_sdpa_mask_type";

// Sentinel value written for the Causal case - see NPUW_SDPA_MASK_RT_KEY above.
static constexpr int64_t NPUW_SDPA_MASK_CAUSAL = -1;

}  // namespace ov::npuw
