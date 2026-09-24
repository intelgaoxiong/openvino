// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "chunk_major_moe_infer_request.hpp"

ov::npuw::ChunkMajorMoEInferRequest::ChunkMajorMoEInferRequest(
    const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : JustInferRequest(compiled_model) {}

ov::SoPtr<ov::ITensor> ov::npuw::ChunkMajorMoEInferRequest::peek_prev_output(std::size_t idx,
                                                                             std::size_t param_idx) const {
    return resolve_prev_output(idx, param_idx);
}
