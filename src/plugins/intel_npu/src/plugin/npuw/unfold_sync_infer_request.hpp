// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "base_sync_infer_request.hpp"

namespace ov {
namespace npuw {

class UnfoldInferRequest final : public IBaseInferRequest {
public:
    explicit UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);
    ~UnfoldInferRequest();

    ////////////////////////////////////
    // implement IBaseInferRequest - nether of these are required here
    // this hierarchy needs revew
    void prepare_for_infer() override {}
    bool valid_subrequest(std::size_t idx) const override;
    void start_subrequest(std::size_t) override {}
    void run_subrequest_for_success(std::size_t) override {}
    void subscribe_subrequest(std::size_t, Completed cb) override {}
    void complete_subrequest(std::size_t) override {}
    void cancel_subrequest(std::size_t) override {}
    bool supports_async_pipeline() const override {
        return false;
    }
    void update_subrequest_links(std::size_t) override {}

private:
    void infer() override;

    // Per-real-subgraph timing accumulated across all generate decode steps.
    std::map<std::size_t, float> m_generate_timings_ms;   // real_idx -> total ms
    std::map<std::size_t, std::size_t> m_generate_call_counts;  // real_idx -> total calls
    std::size_t m_generate_infer_count = 0u;               // number of infer() invocations
};

}  // namespace npuw
}  // namespace ov
