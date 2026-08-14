// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unfold_sync_infer_request.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

#include "attn/attn_subgraph.hpp"
#include "compiled_model.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"

ov::npuw::UnfoldInferRequest::UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : ov::npuw::IBaseInferRequest(compiled_model) {
    // Create infer requests
    // Preallocate funcall tensors & substitute function call requests
    for (std::size_t i = 0; i < m_num_submodels; i++) {
        LOG_INFO("Creating infer request for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];

        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            // no model & no funcall - optimized out, do nothing
            LOG_INFO("OPTIMIZED OUT");
            continue;
        }

        if (comp_model_desc.replaced_by) {
            // Pre-allocate output tensors for this function call
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            if (proto_comp_model_desc.spatial) {
                NPUW_ASSERT(false && "Spatial is not supported in unfold");
            }
            if (ov::npuw::attn::get_compiled_dynamic(proto_comp_model_desc.pipeline.context) != nullptr) {
                NPUW_ASSERT(false && "Dynamic is not supported in unfold");
            }
        }  // if(replaced_by)

        const auto real_idx = comp_model_desc.replaced_by.value_or(i);
        auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
        // NB: UnfoldInferRequest is _NOT_ fail-safe! Fail means fail here
        m_subrequests[i] = proto_comp_model_desc.compiled_model->create_infer_request();
        LOG_INFO("DONE");
    }  // for(submodels)

    alloc_quant_gather();

    LOG_INFO("Connecting subrequests...");
    LOG_BLOCK();
    for (const auto& kvp : m_npuw_model->m_submodels_input_to_prev_output) {
        const auto& subm_idx_to = kvp.first.first;
        const auto& port_idx_to = kvp.first.second;
        const auto& subm_idx_from = kvp.second.first;
        const auto& port_idx_from = kvp.second.second;

        LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> " << "Subgraph[" << subm_idx_to << "]/"
                              << port_idx_to);
        NPUW_ASSERT(m_subrequests[subm_idx_from]);  // prod request is created
        NPUW_ASSERT(m_subrequests[subm_idx_to]);    // cons request is created
        NPUW_ASSERT(m_subrequests[subm_idx_from]._ptr != m_subrequests[subm_idx_to]._ptr);

        const auto& iport = m_subrequests[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
        const auto& oport = m_subrequests[subm_idx_from]->get_compiled_model()->outputs()[port_idx_from];
        const auto& tensor = m_subrequests[subm_idx_from]->get_tensor(oport);
        LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to Subgraph[" << subm_idx_from << "]/" << oport);
        m_subrequests[subm_idx_to]->set_tensor(iport, tensor);
    }  // for(map)
    LOG_INFO("Done");

    init_gio();

    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_VERB("Trying to preemptively set tensors for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;  // Optimized out
        }
        if (comp_model_desc.replaced_by) {
            unpack_closure(i, m_subrequests[i]);
        }
        LOG_VERB("Done");
    }
}

static void print_generate_sublayer_timing_report(
    const std::map<std::size_t, float>& timings_ms,
    const std::map<std::size_t, std::size_t>& call_counts,
    std::size_t infer_count,
    const ov::npuw::IBaseInferRequest& req) {
    if (timings_ms.empty())
        return;

    const int cw = 12;
    std::cerr << "\n=== Generate Sublayer Timing (accumulated over " << infer_count << " decode steps) ===\n";
    std::cerr << std::left << std::setw(6) << " real" << std::setw(8) << " device" << std::setw(10) << " calls"
              << std::setw(cw) << " total(ms)" << std::setw(cw) << " avg/call" << std::setw(cw) << " avg/step"
              << "\n";
    const std::size_t sep_len = static_cast<std::size_t>(6 + 8 + 10) + 3u * static_cast<std::size_t>(cw);
    std::cerr << std::string(sep_len, '-') << "\n";

    float grand_total = 0.f;
    std::size_t grand_calls = 0u;
    for (const auto& kv : timings_ms) {
        const std::size_t ridx = kv.first;
        const float total_ms = kv.second;
        const auto it_cnt = call_counts.find(ridx);
        const std::size_t calls = it_cnt != call_counts.end() ? it_cnt->second : 0u;
        const float avg_call = calls > 0u ? total_ms / static_cast<float>(calls) : 0.f;
        const float avg_step = infer_count > 0u ? total_ms / static_cast<float>(infer_count) : 0.f;
        const auto device = req.sublayer_device(ridx);
        std::cerr << std::left << std::setw(6) << ridx << std::setw(8) << device << std::setw(10) << calls
                  << std::fixed << std::setprecision(2) << std::setw(cw) << total_ms << std::setw(cw) << avg_call
                  << std::setw(cw) << avg_step << "\n";
        grand_total += total_ms;
        grand_calls += calls;
    }
    std::cerr << std::string(sep_len, '-') << "\n";
    const float grand_avg_step = infer_count > 0u ? grand_total / static_cast<float>(infer_count) : 0.f;
    std::cerr << std::left << std::setw(6) << "Total" << std::setw(8) << "" << std::setw(10) << grand_calls
              << std::fixed << std::setprecision(2) << std::setw(cw) << grand_total << std::setw(cw) << ""
              << std::setw(cw) << grand_avg_step << "\n\n";
}

ov::npuw::UnfoldInferRequest::~UnfoldInferRequest() {
    if (!m_generate_timings_ms.empty()) {
        print_generate_sublayer_timing_report(m_generate_timings_ms,
                                              m_generate_call_counts,
                                              m_generate_infer_count,
                                              *this);
    }
}

bool ov::npuw::UnfoldInferRequest::valid_subrequest(std::size_t idx) const {
    return m_subrequests.at(idx) != nullptr;
}

void ov::npuw::UnfoldInferRequest::infer() {
    using Clock = std::chrono::steady_clock;
    const bool do_async = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();

    auto prepare = [&](std::size_t idx) {
        if (idx >= m_subrequests.size()) {
            return;
        }
        bind_global_params(idx, m_subrequests[idx]);
        bind_global_results(idx, m_subrequests[idx]);
    };

    if (do_async) {
        std::size_t past_repl_id = 0u;

        struct PendingReq {
            RqPtr rq;
            std::size_t real_idx;
            Clock::time_point t_start;
        };
        std::vector<PendingReq> pending;

        auto wait_and_accum = [&]() {
            for (auto& p : pending) {
                p.rq->wait();
                const float elapsed_ms = std::chrono::duration<float, std::milli>(Clock::now() - p.t_start).count();
                m_generate_timings_ms[p.real_idx] += elapsed_ms;
                m_generate_call_counts[p.real_idx]++;
            }
            pending.clear();
        };

        prepare(0);
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                prepare(idx + 1);
                continue;
            }
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
            const auto this_repl_id = comp_model_desc.replaced_by.value_or(idx);
            if (this_repl_id != past_repl_id) {
                // Barrier: wait for all requests in the previous group before starting a new one
                wait_and_accum();
                past_repl_id = this_repl_id;
            }
            auto t_start = Clock::now();
            subr->start_async();
            pending.push_back({subr, this_repl_id, t_start});
            prepare(idx + 1);
        }
        wait_and_accum();
    } else {
        prepare(0);
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                prepare(idx + 1);
                continue;
            }
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
            const auto this_real_idx = comp_model_desc.replaced_by.value_or(idx);
            const auto t_start = Clock::now();
            subr->start_async();
            prepare(idx + 1);
            subr->wait();
            const float elapsed_ms = std::chrono::duration<float, std::milli>(Clock::now() - t_start).count();
            m_generate_timings_ms[this_real_idx] += elapsed_ms;
            m_generate_call_counts[this_real_idx]++;
        }
    }  // (async)
    m_generate_infer_count++;
}
