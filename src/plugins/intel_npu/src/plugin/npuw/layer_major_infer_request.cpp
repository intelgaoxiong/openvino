// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_major_infer_request.hpp"

#include <algorithm>
#include <set>

#include "compiled_model.hpp"
#include "moe/moe_subgraph.hpp"

ov::npuw::LayerMajorInferRequest::LayerMajorInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : JustInferRequest(compiled_model) {
    // Leave m_use_function_pipelining at whatever the base ctor derived from
    // NPUW_FUNCALL_ASYNC (the LLM prefill sub-config forces this to YES - see
    // llm_compiled_model.cpp). The driver (llm_infer_request.cpp) is responsible for
    // disabling it via set_function_pipelining() immediately around any infer_range() call
    // that repeats a span already dispatched earlier in the SAME group - see this class's
    // declaration comment and design doc §7.1 for why that's the only unsafe case.
    register_bridges();
    discover_layer_groups();
}

bool ov::npuw::LayerMajorInferRequest::set_function_pipelining(bool enable) {
    const bool prev = m_use_function_pipelining;
    m_use_function_pipelining = enable;
    return prev;
}

const std::vector<ov::npuw::LayerMajorInferRequest::LayerGroup>& ov::npuw::LayerMajorInferRequest::layer_groups()
    const {
    return m_layer_groups;
}

void ov::npuw::LayerMajorInferRequest::discover_layer_groups() {
    const auto& moe_idxs = moe_occurrence_idxs();
    const auto num_layers = moe_idxs.size();
    const auto num_submodels = total_subrequests();
    m_layer_groups.reserve(num_layers);
    for (std::size_t L = 0; L < num_layers; ++L) {
        // seg_end deliberately reaches the NEXT layer's own MoE occurrence (fusing this layer's
        // downstream-combine with the next layer's pre-MoE part into one infer_range() span -
        // see LayerGroup's declaration comment), not just "past this layer's own downstream".
        const auto seg_start = (L == 0) ? std::size_t{0} : moe_idxs[L - 1] + 1;
        const auto seg_end = (L + 1 < num_layers) ? moe_idxs[L + 1] : num_submodels;
        m_layer_groups.push_back(LayerGroup{seg_start, moe_idxs[L], seg_end});
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::LayerMajorInferRequest::peek_prev_output(std::size_t idx,
                                                                           std::size_t param_idx) const {
    return resolve_prev_output(idx, param_idx);
}

void ov::npuw::LayerMajorInferRequest::rebind_global_parameters(std::size_t idx) {
    // Deliberately bypass JustInferRequest::bind_global_parameters(): its now_idx()-based
    // primary/reserve subrequest selection assumes idx is the occurrence currently
    // in-flight inside unsafe_during()'s async window. now_idx() only compares real_idx
    // (the shared funcall-folded function), and infer_range() never resets it on return -
    // so it can still match real_idx from any earlier same-function dispatch (a previous
    // chunk, layer, or even a prior generate() call on a reused request), misrouting our
    // synchronous rebind-then-dispatch onto the reserve m_funcall_pipeline subrequest
    // instead of the primary one that actually gets executed (and dumped). Always bind
    // the primary subrequest directly.
    const auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    bind_global_params(idx, m_subrequests[real_idx]);
}

void ov::npuw::LayerMajorInferRequest::register_bridges() {
    // Category A: one bridge per funcall-folded producer, shared by every consumer of that
    // producer's output regardless of role (hidden state, MoE expert_input/router_scores,
    // downstream, MoE output, ...).
    for (const auto& link : m_npuw_model->m_submodels_input_to_prev_output) {
        const auto prod_idx = link.second.first;
        const auto prod_port = link.second.second;
        if (m_npuw_model->m_compiled_submodels[prod_idx].replaced_by.has_value()) {
            m_bridges[LinkFrom{prod_idx, prod_port}];  // default-construct if not present yet
        }
    }

    // Category B: one cache per top-level input read by more than one submodel occurrence
    // (e.g. attention_mask/position_ids/sliding_window_attention_mask - re-filled once per
    // chunk by the driver, then re-read by every layer as the group's layer loop advances).
    //
    // attention_mask/position_ids/sliding_window_attention_mask are ALWAYS bridged below,
    // regardless of num_readers: num_readers only counts distinct submodel *occurrences*
    // wired to this global by the graph-level input/subscriber maps, which - because every
    // attention layer's own pre-MoE part is funcall-folded onto the same shared real_idx -
    // can come out <= 1 even though the driver's Stage 2 dispatches that SAME occurrence
    // group_size times per group (once per chunk), each needing that chunk's own content.
    // num_readers is the wrong proxy for "does this need a per-chunk bridge" for these
    // specific globals; name-match them explicitly instead.
    static const std::set<std::string> kAlwaysBridgedGlobalNames = {"attention_mask",
                                                                     "position_ids",
                                                                     "sliding_window_attention_mask"};
    const auto num_global_inputs = m_npuw_model->m_inputs_to_submodels_inputs.size();
    for (std::size_t global_idx = 0; global_idx < num_global_inputs; ++global_idx) {
        std::size_t num_readers = 0;
        if (m_npuw_model->m_inputs_to_submodels_inputs[global_idx] != ov::npuw::CompiledModel::NO_LINK) {
            ++num_readers;
        }
        const auto extra = m_npuw_model->m_param_subscribers.find(global_idx);
        if (extra != m_npuw_model->m_param_subscribers.end()) {
            num_readers += extra->second.size();
        }
        const auto& g_names = m_npuw_model->inputs().at(global_idx).get_names();
        const bool always_bridged =
            std::any_of(g_names.begin(), g_names.end(), [](const std::string& n) {
                return kAlwaysBridgedGlobalNames.count(n) != 0;
            });
        if (num_readers > 1 || always_bridged) {
            m_global_caches[global_idx];  // default-construct if not present yet
        }
    }
}

std::size_t ov::npuw::LayerMajorInferRequest::num_params(std::size_t idx) const {
    // A funcall-folded occurrence's own compiled_model is null - the real one (with the real
    // input count) lives at replaced_by (see function_prologue()'s identical resolution).
    const auto& desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = desc.replaced_by.value_or(idx);
    return m_npuw_model->m_compiled_submodels[real_idx].compiled_model->inputs().size();
}

std::optional<std::pair<std::size_t, std::size_t>> ov::npuw::LayerMajorInferRequest::producer_of(
    std::size_t consumer_idx,
    std::size_t consumer_param_idx) const {
    const auto link_iter = m_npuw_model->m_submodels_input_to_prev_output.find({consumer_idx, consumer_param_idx});
    if (link_iter == m_npuw_model->m_submodels_input_to_prev_output.end()) {
        return std::nullopt;
    }
    return link_iter->second;
}

ov::npuw::HiddenStateBridge* ov::npuw::LayerMajorInferRequest::bridge_for_link(std::size_t consumer_idx,
                                                                               std::size_t consumer_param_idx) {
    const auto link_iter = m_npuw_model->m_submodels_input_to_prev_output.find({consumer_idx, consumer_param_idx});
    if (link_iter == m_npuw_model->m_submodels_input_to_prev_output.end()) {
        return nullptr;  // not fed from another submodel's output (e.g. a global input)
    }
    const auto bridge_iter = m_bridges.find(LinkFrom{link_iter->second.first, link_iter->second.second});
    return bridge_iter == m_bridges.end() ? nullptr : &bridge_iter->second;
}

ov::npuw::HiddenStateBridge* ov::npuw::LayerMajorInferRequest::global_cache_for(std::size_t global_input_idx) {
    const auto it = m_global_caches.find(global_input_idx);
    return it == m_global_caches.end() ? nullptr : &it->second;
}

ov::npuw::HiddenStateBridge* ov::npuw::LayerMajorInferRequest::global_cache_for(const std::string& global_input_name) {
    const auto& compiled_inputs = get_compiled_model()->inputs();
    for (std::size_t i = 0; i < compiled_inputs.size(); ++i) {
        if (compiled_inputs[i].get_names().count(global_input_name) != 0) {
            return global_cache_for(i);
        }
    }
    return nullptr;
}

bool ov::npuw::LayerMajorInferRequest::supports_strided_producer_output(std::size_t producer_idx,
                                                                         std::size_t producer_port) const {
    const auto& comp_model_desc = m_npuw_model->m_compiled_submodels[producer_idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(producer_idx);
    return m_npuw_model->m_strided_producer_ports.count({real_idx, producer_port}) != 0;
}

bool ov::npuw::LayerMajorInferRequest::supports_strided_consumer_input(std::size_t consumer_idx,
                                                                        std::size_t consumer_param_idx) const {
    const auto& comp_model_desc = m_npuw_model->m_compiled_submodels[consumer_idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(consumer_idx);
    return m_npuw_model->m_strided_consumer_ports.count({real_idx, consumer_param_idx}) != 0;
}

std::optional<std::size_t> ov::npuw::LayerMajorInferRequest::moe_downstream_expert_output_param_idx(
    std::size_t idx) const {
    const auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    if (const auto* moe_downstream = ov::npuw::moe::get_compiled_downstream(
            m_npuw_model->m_compiled_submodels[real_idx].pipeline.context)) {
        return moe_downstream->expert_output_param_idx;
    }
    return std::nullopt;
}


const ov::Output<const ov::Node>& ov::npuw::LayerMajorInferRequest::producer_output_port(
    std::size_t producer_idx,
    std::size_t producer_port) const {
    const auto& comp_model_desc = m_npuw_model->m_compiled_submodels[producer_idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(producer_idx);
    return m_npuw_model->m_compiled_submodels[real_idx].compiled_model->outputs()[producer_port];
}

void ov::npuw::LayerMajorInferRequest::bind_producer_output(std::size_t producer_idx,
                                                             std::size_t producer_port,
                                                             const TensorPtr& tensor) {
    m_funcall_result[LinkFrom{producer_idx, producer_port}] = tensor;
}

ov::SoPtr<ov::ITensor> ov::npuw::LayerMajorInferRequest::alloc_strided_mem(const ov::element::Type& type,
                                                                           const ov::Shape& shape) const {
    return allocMem(type, shape, "NPU");
}

