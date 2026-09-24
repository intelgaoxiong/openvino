// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "just_sync_infer_request.hpp"

namespace ov {
namespace npuw {

// A specialization of JustInferRequest for the layer-major chunked-prefill MoE driver
// (NPUW_LLM_MOE_BATCH_CHUNKS > 1). Reuses JustInferRequest's dispatch machinery (closures,
// spatial, function pipelining, subgraph behaviors) unchanged, and adds only the extra
// capability the layer-major driver needs: peeking at a repeated MoE function's per-chunk
// input tensors (expert_input/router_scores) without dispatching the MoE call itself, so
// LLMInferRequest can accumulate them across a chunk window before issuing a single batched
// MoE call (see MoEExecutor::run_batched).
class ChunkMajorMoEInferRequest : public JustInferRequest {
public:
    explicit ChunkMajorMoEInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    // Peeks at the tensor currently bound to submodel idx's param_idx-th input by following
    // the producer link, without running idx. Empty if there's no such link or the producer
    // hasn't run yet in this infer_range() call.
    ov::SoPtr<ov::ITensor> peek_prev_output(std::size_t idx, std::size_t param_idx) const;
};

}  // namespace npuw
}  // namespace ov
