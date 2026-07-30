// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// RT info key used on merged weight/scale constants to record their source
// constants (which carry valid WeightlessCacheAttribute bin offsets).
// Partitioning reads this to build a LazyTensor::Concat backed by bin offsets
// instead of creating a synthesized (no-bin-offset) LazyTensor.
static constexpr char kMergedSources[] = "npuw_merged_sources";

// Stored under kMergedSources in the RT info of every constant produced by
// MergeParallelDQMatMuls (i.e. new_w and new_s).
struct MergedConstSources {
    std::vector<std::shared_ptr<ov::op::v0::Constant>> consts;  // original src constants
    std::size_t axis = 0;                                        // concat axis
};

// Merges parallel DQ (Dequantize) MatMul pairs that share the same activation
// into a single wider MatMul followed by Slice ops for each original consumer.
//
// Target pattern (two or more MatMuls sharing activation `Act`):
//
//   Act ──────────────────────────────────────────────────► MatMul_0
//   Const(W_0)[u4/i4] → Convert(f16) → Multiply(Const(S_0)) → Reshape →
//
//   Act ──────────────────────────────────────────────────► MatMul_1
//   Const(W_1)[u4/i4] → Convert(f16) → Multiply(Const(S_1)) → Reshape →
//
// Becomes:
//
//   Act ───────────────────────────────────────────────────────► MatMul_merged
//   Concat(W_0,W_1,axis) → Convert → Multiply(Concat(S_0,S_1,axis)) → Reshape →
//         └─ Slice_0 ──► (original MatMul_0 consumers)
//         └─ Slice_1 ──► (original MatMul_1 consumers)
//
// Constraints:
//   - W and S must be ov::op::v0::Constant (not Parameter), i.e. pre-partition
//   - MatMul output must be 3-D with batch == 1: [1, seq, out]
//   - !transpose_a; either value of transpose_b is supported
//   - W and S must be 3-D constants; non-concat dims must be equal across candidates
//
// This pass is intended to run in LLMCompiledModel BEFORE partitioning so that:
//   1. BEST_PERF models (NPUW_ONLINE_PIPELINE=NONE, Partitioner::optimize never called)
//      also benefit from the merge.
//   2. Weights bank deduplication works: both prefill and generate models produce the
//      same LazyTensor(Concat(...)) key, enabling the bank to deduplicate.
//   3. The closure-level bookkeeping in Partitioner::optimize is bypassed entirely.

class MergeParallelDQMatMuls : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::MergeParallelDQMatMuls");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
