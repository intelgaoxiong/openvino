// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace npuw {
namespace pass {

/**
 * @brief Duplicate shared KV broadcast chains to enable per-SDPA ATTN isolation.
 *
 * Some models (e.g. Gemma-4) share a single accumulated KV tensor across multiple
 * SDPA nodes by fanning out through a Reshape node:
 *
 *   Concat(past_K_blocks…, current_K)
 *       → [Unsqueeze] → [Broadcast] → Reshape ─┬─→ SDPA_L13
 *                                               ├─→ SDPA_L15
 *                                               ├─→ SDPA_L16
 *                                               └─→ …
 *
 * The high fan-out prevents NPUW's ATTN isolation patterns (TagSDPA /
 * SDPADecomposed) from tagging each SDPA independently, because every tagging
 * attempt would claim the shared Reshape (and its ancestors) as part of its
 * own partition, creating conflicting assignments.
 *
 * This pass duplicates the Concat → [Unsqueeze] → [Broadcast] → Reshape chain
 * for every extra consumer beyond the first, giving each SDPA its own local
 * chain:
 *
 *   same past_K_blocks…, current_K → new_Concat_L15 → … → new_Reshape_L15 → SDPA_L15
 *   same past_K_blocks…, current_K → new_Concat_L16 → … → new_Reshape_L16 → SDPA_L16
 *   …
 *
 * All duplicate chains reference the same Parameter (block) nodes → zero
 * additional memory allocation.  The same current_K computation node is also
 * shared → zero extra compute.
 *
 * Must be run AFTER SplitKVCacheIntoBlocks so that the block Parameters are
 * already present and will be shared across all duplicate chains.
 */
class DuplicateSharedKVConcat : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("npuw::pass::DuplicateSharedKVConcat");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace npuw
}  // namespace ov
