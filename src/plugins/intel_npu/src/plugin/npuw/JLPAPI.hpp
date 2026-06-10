#pragma once
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "visibility.hpp"

namespace npu {
enum class JLPOption {
    none,
    dma_benchmark,
    inference_map,
    barrier_safety,
    smart_dma,
    add_performance_metrics,
    use_all_shaves,
    balance_shave_variants,
    lnl,
    lnl_production,
    verbose,
    reset_test,
    full_workload_management,
    npu7_tr,
    vertical_layout,
    dpu_nop,
    barrier4k,
    barrier4k_npu7,
    dpu_testbench,
    shv_testbench,
    barrier_dma,
    simics_autofetch,
    dataflow,
    make_persistent,
    make_init,
    make_realtime,
    fuse_elf,
    cmx_size_1_5,
    dcim_enable,
    set_fw_version,
    set_elf_version,
    set_npu,
    dpu_skip_all,
    dpu_skip_unsupported,
    dma_skip_all,
    summary_only,
    silent,
    ddr_shv_tasks,
    no_wlm,
    dma_skip_custom,
    dma_compression,
    dma_cmx_descriptor,
    dma_compression_ping_pong,
    shv_management_kernel,
    csv_only,
    output_xml,
    cmx_memory_alloc,
    matmul_hwop,
    fp4,
    dpu_wcb_bypass,
    conditional_hmo,
    conditional_hmo_demo,
    set_conditional_entry,
    memset_graph,
    io_reuse,
    io_iterate,
    io_consolidate,
    realtime_pipeline,
    output_directory,
    parse,
    pipeline_view,
    io_global
};

using blob = std::vector<uint8_t>;
using io_map = std::map<uint32_t, uint32_t>;
using parameter_map = std::map<std::string, std::vector<uint32_t>>;
class JLP_CORE_EXPORTS JLPAPI {
public:
    JLPAPI() = default;
    bool proceed(std::string elf_blob, blob& blob_data);
    bool proceed(blob& blob_data);
    bool proceed(std::vector<blob>& blob_data);
    bool add_option(JLPOption option, std::string option_parameter);
    bool add_option(JLPOption option);
    bool save_blob(const std::string& filename);
    blob load_blob(const std::string& filename);
    void clear_options();
    bool get_global_io_mapping(blob& blob_data, io_map& inputs, io_map& outputs, parameter_map& parameters);

private:
    blob output_blob_data_{};
    bool load_elf(blob& blob_data, bool get_global_io = false);
    std::deque<JLPOption> options_;
    std::map<JLPOption, uint32_t> option_usage_tracking_;
    std::map<std::string, std::string> option_parameters_;
    io_map global_inputs_{};
    io_map global_outputs_{};
    parameter_map global_paramaters_{};
    uint32_t global_parameter_count_{};
};
}  // namespace npu