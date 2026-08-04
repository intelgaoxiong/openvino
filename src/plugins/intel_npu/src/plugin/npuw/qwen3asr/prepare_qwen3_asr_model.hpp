// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {

// Step 1 — applied BEFORE cloning the prefill model (to kvcache_model while it is still the shared base).
// Removes residual ReadValue/Assign state nodes that StatefulToStateless does not convert
// (e.g. the encoder_hidden_states state whose variable id does not match "past_key_values.*").
// Each ReadValue is replaced with either its existing Parameter initial-value, or a fresh Parameter.
class PrepareQwen3ASRModel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareQwen3ASRModel");
    PrepareQwen3ASRModel() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Step 2 — applied AFTER cloning the prefill model (to kvcache model only).
// Injects an explicit attention_mask [1, kv_capacity] Parameter to replace the frozen causal mask,
// and an explicit position_ids [1] Parameter to replace the frozen RoPE position scalar.
// Must run BEFORE ReshapeToStatic so that the dynamic Range/Gather nodes are still present.
class PrepareQwen3ASRKVCacheModel : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PrepareQwen3ASRKVCacheModel");
    PrepareQwen3ASRKVCacheModel() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace npuw
}  // namespace ov
