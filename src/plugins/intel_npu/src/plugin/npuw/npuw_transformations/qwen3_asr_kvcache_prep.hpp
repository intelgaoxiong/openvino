// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {

// Injects `attention_mask [1, kv_capacity]` Parameter into the Qwen3-ASR generate
// (KV-cache) model by replacing the internal LessEqual-based causal mask.
//
// The original graph computes:
//   Range_k(0, past_kv_size+1, 1) -> Unsqueeze -> Unsqueeze [1,1,1,N]
//   Range_q(past_kv_size, past_kv_size+1, 1) -> Unsqueeze x2 [1,1,1,1]
//   LessEqual(K_positions, Q_position) -> Broadcast -> Select -> SDPA mask
//
// After transformation:
//   attention_mask [1, kv_capacity] -> Unsqueeze(1) -> Unsqueeze(2) [1,1,1,N]
//   Equal(attention_mask_4d, 0)  [True=attend, False=mask] replaces LessEqual
//
// Convention (same polarity as LessEqual True=attend):
//   attention_mask[k] = 0  => attend to position k
//   attention_mask[k] = 1  => mask out position k
class Qwen3ASRAttentionMaskInput : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::Qwen3ASRAttentionMaskInput");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Injects `position_ids [1]` Parameter into the Qwen3-ASR generate (KV-cache)
// model for the RoPE computation path.
//
// The original graph derives current token position from:
//   Gather(ShapeOf(past_kv), dim_const, axis_const) [= past_kv_size, static after reshape]
//   Range(Gather, Gather+1, 1) -> Reshape([1,1,-1]) -> Broadcast -> RoPE ScatterNDUpdate
//
// After transformation:
//   position_ids [1] -> Reshape([1,1,-1]) -> Broadcast -> RoPE ScatterNDUpdate
class Qwen3ASRPositionIdsInput : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::Qwen3ASRPositionIdsInput");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace npuw
}  // namespace ov
