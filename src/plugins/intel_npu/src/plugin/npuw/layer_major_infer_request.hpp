// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>

#include "hidden_state_bridge.hpp"
#include "just_sync_infer_request.hpp"

namespace ov {
namespace npuw {

// A specialization of JustInferRequest for the layer-major chunked-prefill MoE driver
// (NPUW_LLM_MOE_BATCH_CHUNKS > 1). Reuses JustInferRequest's dispatch machinery (closures,
// spatial, function pipelining, subgraph behaviors) unchanged, and adds only the extra
// capability the layer-major driver needs: peeking at a repeated function's per-chunk input
// tensors without dispatching the call itself, plus a generic, graph-discovered set of
// HiddenStateBridge instances (see design doc §4.4.2) that let the driver buffer any
// funcall-folded producer's output, or any per-chunk global input, across a chunk-batch
// window before/after a single batched call (e.g. MoEExecutor::run_batched).
class LayerMajorInferRequest : public JustInferRequest {
public:
    explicit LayerMajorInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    // Peeks at the tensor currently bound to submodel idx's param_idx-th input by following
    // the producer link, without running idx. Empty if there's no such link or the producer
    // hasn't run yet in this infer_range() call.
    ov::SoPtr<ov::ITensor> peek_prev_output(std::size_t idx, std::size_t param_idx) const;

    // Category A (§4.4.2): one HiddenStateBridge per funcall-folded producer submodel output,
    // shared by every consumer of that output across every layer boundary - hidden state,
    // MoE expert_input/router_scores, downstream, MoE output, and any future signal, with no
    // per-role hand-naming required. Looked up by the *consumer's* (idx, param_idx); resolves
    // through m_submodels_input_to_prev_output to the producer-keyed shared instance. Returns
    // nullptr if that link isn't fed from a funcall-folded producer (no bridge needed).
    HiddenStateBridge* bridge_for_link(std::size_t consumer_idx, std::size_t consumer_param_idx);

    // Category B (§4.4.2): one HiddenStateBridge per top-level compiled-model input consumed
    // by more than one submodel occurrence (e.g. attention_mask, position_ids,
    // sliding_window_attention_mask) - re-filled once per chunk at group start, then read many
    // times (once per layer) as the group's layer loop advances. Looked up by the compiled
    // model's own input port index. Returns nullptr if global_input_idx has at most one reader
    // (no caching needed - it's consumed exactly once, right when produced).
    HiddenStateBridge* global_cache_for(std::size_t global_input_idx);

private:
    // One-time constructor walk that discovers which links/global-inputs need bridging (keys
    // only - no tensor memory is allocated here, see HiddenStateBridge::configure()).
    void register_bridges();

    std::map<LinkFrom, HiddenStateBridge> m_bridges;
    std::map<std::size_t, HiddenStateBridge> m_global_caches;
};

}  // namespace npuw
}  // namespace ov
