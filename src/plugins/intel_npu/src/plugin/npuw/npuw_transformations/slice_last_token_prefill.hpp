// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Optimization pass for the prefill model: instead of running the full last
// transformer layer for all N query tokens and slicing the output after the
// LM head (SliceOutEmbeds), slice Q to the single last token *before* the
// last layer's ScaledDotProductAttention.
//
// This saves the attention computation for N-1 query positions in the last
// layer as well as the subsequent output-projection and FFN for those N-1
// positions.  K and V are still computed for all N tokens (needed for
// full causal attention).
//
// Must be called BEFORE DecomposeGQA (which replaces the native SDPA op).
// Works on the static-shape prefill model after ReshapeToStatic.
class SliceLastTokenPrefill : public ov::pass::ModelPass {
    uint32_t m_batch_dim;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::SliceLastTokenPrefill");
    explicit SliceLastTokenPrefill(uint32_t batch_dim);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
