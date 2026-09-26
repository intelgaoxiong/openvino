// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "hidden_state_bridge.hpp"
#include "just_sync_infer_request.hpp"

namespace ov {
namespace npuw {

// A specialization of JustInferRequest for the layer-major chunked-prefill MoE driver
// (NPUW_LLM_MOE_BATCH_CHUNKS >= 1). Reuses JustInferRequest's dispatch machinery (closures,
// spatial, subgraph behaviors) unchanged, but forces function-call pipelining OFF (see ctor):
// its primary/reserve buffer swap assumes every funcall-folded occurrence is visited at most
// once before the pipeline moves on to `next_idx`, which this driver violates by re-dispatching
// the SAME occurrence once per chunk within a group via infer_range(). Adds only the extra
// capability the layer-major driver needs: peeking at a repeated function's per-chunk input
// tensors without dispatching the call itself, plus a generic, graph-discovered set of
// HiddenStateBridge instances (see design doc §4.4.2) that let the driver buffer any
// funcall-folded producer's output, or any per-chunk global input, across a chunk-batch
// window before/after a single batched call (e.g. MoEExecutor::run_batched).
class LayerMajorInferRequest : public JustInferRequest {
public:
    explicit LayerMajorInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    // One transformer layer's submodel range, in infer_range()'s [0, total_subrequests())
    // index space. [seg_start, moe_idx) is this layer's own pre-MoE (attention+router) part,
    // moe_idx is the MoE occurrence itself (driver calls this via moe_executor()/run_batched()
    // instead of infer_range()), and [moe_idx, seg_end) is everything up to (not including) the
    // NEXT layer's own MoE occurrence - i.e. this layer's downstream-combine FUSED with the next
    // layer's own pre-MoE part (seg_end == next entry's moe_idx, or total_subrequests() for the
    // last layer). This boundary is deliberately NOT drawn at "where downstream ends": running
    // the fused span as a single infer_range() call lets NPUW's own producer/consumer tensor
    // resolution (resolve_prev_output()/function_prologue()) wire up every value crossing that
    // internal boundary automatically, the same way it does within any single infer_range() span
    // - no manual bridging needed there. Manual bridging (see bridge_for_link()) is only required
    // for values crossing INTO/OUT OF this fused span: the batched MoE call itself, and any
    // top-level input reused by many occurrences (see global_cache_for()).
    struct LayerGroup {
        std::size_t seg_start;
        std::size_t moe_idx;
        std::size_t seg_end;
    };

    // One entry per MoE-bearing transformer layer, in ascending (layer) order - see
    // LayerGroup's declaration comment for how boundaries are derived (from
    // moe_occurrence_idxs() alone). Empty if the model has no MoE layers.
    const std::vector<LayerGroup>& layer_groups() const;

    // Peeks at the tensor currently bound to submodel idx's param_idx-th input by following
    // the producer link, without running idx. Empty if there's no such link or the producer
    // hasn't run yet in this infer_range() call.
    ov::SoPtr<ov::ITensor> peek_prev_output(std::size_t idx, std::size_t param_idx) const;

    // Re-binds submodel idx's global parameters (host_gather/quant_unpack_gather included),
    // reading whatever is currently written into its global inputs. prepare_for_infer() only
    // does this once, for next(0), before this driver has written any chunk's own front
    // inputs - a submodel fed from a per-chunk global input (e.g. the embedding-gather
    // submodel, reading input_ids) needs this called again after each chunk's own
    // prepare_prefill_chunk_front_inputs(), or it keeps gathering with stale/zero-initialized
    // data for every chunk.
    void rebind_global_parameters(std::size_t idx);

    // Category A (§4.4.2): one HiddenStateBridge per funcall-folded producer submodel output,
    // shared by every consumer of that output across every layer boundary - hidden state,
    // MoE expert_input/router_scores, downstream, MoE output, and any future signal, with no
    // per-role hand-naming required. Looked up by the *consumer's* (idx, param_idx); resolves
    // through m_submodels_input_to_prev_output to the producer-keyed shared instance. Returns
    // nullptr if that link isn't fed from a funcall-folded producer (no bridge needed).
    HiddenStateBridge* bridge_for_link(std::size_t consumer_idx, std::size_t consumer_param_idx);

    // Which occurrence (and its output port) feeds consumer_idx's consumer_param_idx-th input,
    // if any - lets a driver filter bridge_for_link() candidates by producer identity (e.g. "only
    // links produced within this specific infer_range() span") without duplicating
    // bridge_for_link()'s own m_submodels_input_to_prev_output lookup. Empty if param_idx isn't
    // fed from another submodel's output (e.g. a global input).
    std::optional<std::pair<std::size_t, std::size_t>> producer_of(std::size_t consumer_idx,
                                                                    std::size_t consumer_param_idx) const;

    // Number of inputs submodel idx's compiled model accepts - lets a driver generically loop
    // over every param of a submodel to discover which ones need bridging (bridge_for_link),
    // without hand-naming any particular role (e.g. "hidden state").
    std::size_t num_params(std::size_t idx) const;

    // Category B (§4.4.2): one HiddenStateBridge per top-level compiled-model input consumed
    // by more than one submodel occurrence (e.g. attention_mask, position_ids,
    // sliding_window_attention_mask) - re-filled once per chunk at group start, then read many
    // times (once per layer) as the group's layer loop advances. Looked up by the compiled
    // model's own input port index. Returns nullptr if global_input_idx has at most one reader
    // (no caching needed - it's consumed exactly once, right when produced).
    HiddenStateBridge* global_cache_for(std::size_t global_input_idx);

    // Convenience overload: resolves global_input_name (e.g. "attention_mask") to its
    // compiled-model input index via get_compiled_model()->inputs(), then delegates to the
    // index-based overload above. Returns nullptr if the name isn't a top-level input.
    HiddenStateBridge* global_cache_for(const std::string& global_input_name);

private:
    // One-time constructor walk that discovers which links/global-inputs need bridging (keys
    // only - no tensor memory is allocated here, see HiddenStateBridge::configure()).
    void register_bridges();

    // One-time constructor walk that computes m_layer_groups from moe_occurrence_idxs().
    void discover_layer_groups();

    std::map<LinkFrom, HiddenStateBridge> m_bridges;
    std::map<std::size_t, HiddenStateBridge> m_global_caches;
    std::vector<LayerGroup> m_layer_groups;
};

}  // namespace npuw
}  // namespace ov
