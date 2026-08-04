// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../llm_infer_request.hpp"

namespace ov {
namespace npuw {

// Dedicated infer request for Qwen3-ASR encoder-decoder LLM models.
// Extends LLMInferRequest with:
//   - Qwen3-ASR-specific infer() dispatch (seq_len > 1 → prefill, seq_len == 1 → generate)
//   - infer_prefill(): left-aligned token copy + encoder_hidden_states injection + LM head last-token slice
//   - infer_generate(): O(1) per-step decoding with static KV cache, attention_mask, position_ids
//   - get_prefill_kv_range(): left-aligned [0, num_stored_tokens) KV slice for copy_kvcache()
class Qwen3ASRInferRequest final : public LLMInferRequest {
public:
    explicit Qwen3ASRInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model);

    void infer() override;

protected:
    void prepare_for_new_conversation() override;

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> enc_hidden_states);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids);

    // Qwen3-ASR uses left-aligned KV (tokens at [0, N)), vs standard right-aligned.
    std::pair<uint32_t, uint32_t> get_prefill_kv_range() const override;
};

}  // namespace npuw
}  // namespace ov
