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
// spatial, subgraph behaviors) unchanged. Function-call pipelining's primary/reserve buffer
// swap (see JustInferRequest::run_subrequest_for_success()) assumes every funcall-folded
// occurrence is visited at most once before the pipeline moves on to `next_idx` - this
// driver's group loop instead calls infer_range() on the SAME span once per chunk within a
// group, which would corrupt that assumption for every chunk except the group's last one.
// set_function_pipelining() lets the driver selectively disable pipelining around exactly
// those repeated dispatches (see design doc §7.1) while leaving it enabled for the (safe)
// last-chunk-of-group dispatch and letting prepare_for_infer()'s own per-group head-unpack
// keep every funcall-folded occurrence's "head" (first-ever occurrence) correct regardless.
// Adds only the extra capability the layer-major driver needs: peeking at a repeated
// function's per-chunk input tensors without dispatching the call itself, plus a generic,
// graph-discovered set of HiddenStateBridge instances (see design doc §4.4.2) that let the
// driver buffer any funcall-folded producer's output, or any per-chunk global input, across a
// chunk-batch window before/after a single batched call (e.g. MoEExecutor::run_batched).
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

    // True if producer_idx's producer_port-th output was confirmed, at compile time, to
    // support NPU_ENABLE_STRIDES_FOR (see CompiledModel's layer-major bridge pre-pass), i.e. it
    // is safe to call bind_producer_output() for this link instead of a copy-based capture().
    bool supports_strided_producer_output(std::size_t producer_idx, std::size_t producer_port) const;

    // True if consumer_idx's consumer_param_idx-th input was confirmed, at compile time, to
    // support NPU_ENABLE_STRIDES_FOR (see CompiledModel's layer-major bridge pre-pass) - the
    // symmetric counterpart of supports_strided_producer_output() above, checked against
    // m_strided_consumer_ports instead. Used by the general bridge_inject loop to decide
    // whether a Category A link's consumer can read a bridge slot directly (skipping
    // copy_to()) instead of the producer side doing the zero-copy write.
    bool supports_strided_consumer_input(std::size_t consumer_idx, std::size_t consumer_param_idx) const;

    // producer_idx's producer_port-th output port, resolved via its function body's compiled
    // model - unlike a submodel's ov::Model (which detach_memory() may release), this is
    // always available at runtime. Used to read a producer's static shape/type before it has
    // ever been dispatched, so a HiddenStateBridge can be configured ahead of the first
    // bind_producer_output() call.
    const ov::Output<const ov::Node>& producer_output_port(std::size_t producer_idx, std::size_t producer_port) const;

    // Pre-binds producer_idx's producer_port-th output tensor to tensor, so the NEXT
    // infer_range() call that dispatches producer_idx (a funcall occurrence) writes its
    // result directly into tensor via function_prologue()'s existing m_funcall_result lookup,
    // instead of the request's own default output allocation. Must be called BEFORE that
    // infer_range() call, and only when supports_strided_producer_output() is true for the
    // same link - otherwise the NPU may write into non-contiguous memory it wasn't compiled
    // to support.
    void bind_producer_output(std::size_t producer_idx, std::size_t producer_port, const TensorPtr& tensor);

    // Allocates an NPU-resident tensor of the given type/shape, for use as a
    // HiddenStateBridge's backing buffer when zero-copy producer binding is used (see
    // bind_producer_output() above) - the NPU compiled model can only write into
    // device-resident memory. Wraps the base class's protected allocMem() (which needs
    // m_npuw_model's plugin to create the NPU remote context), since callers outside the
    // infer-request hierarchy (llm_infer_request.cpp) cannot call it directly.
    ov::SoPtr<ov::ITensor> alloc_strided_mem(const ov::element::Type& type, const ov::Shape& shape) const;

    // Toggles function-call pipelining (m_use_function_pipelining, protected in the base
    // class) for subsequent infer_range() calls. The driver must disable this immediately
    // before any infer_range() call that repeats a span it already dispatched earlier in the
    // SAME group (i.e. whenever this isn't the group's last chunk for that span), and may
    // leave/restore it enabled otherwise - see this class's declaration comment and design doc
    // §7.1. Returns the previous value, so callers can restore it without tracking it
    // separately.
    bool set_function_pipelining(bool enable);

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
