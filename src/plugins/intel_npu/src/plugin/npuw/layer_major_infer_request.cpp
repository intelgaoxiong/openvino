// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_major_infer_request.hpp"

#include "compiled_model.hpp"

ov::npuw::LayerMajorInferRequest::LayerMajorInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : JustInferRequest(compiled_model) {
    register_bridges();
}

ov::SoPtr<ov::ITensor> ov::npuw::LayerMajorInferRequest::peek_prev_output(std::size_t idx,
                                                                           std::size_t param_idx) const {
    return resolve_prev_output(idx, param_idx);
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
        if (num_readers > 1) {
            m_global_caches[global_idx];  // default-construct if not present yet
        }
    }
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
