// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/online/snapshot.hpp"
#include "partitioning/patterns/moe.hpp"

namespace {

using namespace ov;

// ============================================================================
// Minimal graph builders
// ============================================================================

// Build a minimal GPT-OSS Router subgraph that matches the GPTOSSRouter pattern:
//   Multiply(Const_i4→fp16, scale) → Convert(→f32) → MatMul
//   MatMul + bias → Add → TopK(K) → Softmax(values) → Slice
//
// The TopK friendly name contains ".mlp.router" so the name check in the
// callback passes.  k_param_as_input=true replaces the K constant with a
// Parameter to exercise the "non-constant K" branch.
std::shared_ptr<Model> build_gptoss_router_graph(int64_t k_value,
                                                 bool k_param_as_input = false,
                                                 size_t hidden_dim = 16,
                                                 size_t num_experts = 8) {
    ParameterVector params;

    // Router input: [1, hidden_dim]
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");
    params.push_back(router_input);

    // Quantized weight path: Const(i4) → Convert(fp16) → Multiply(scale) → Convert(f32)
    auto w_i4 = op::v0::Constant::create(element::i4,
                                         Shape{num_experts, hidden_dim},
                                         std::vector<int8_t>(num_experts * hidden_dim, 1));
    auto w_fp16 = std::make_shared<op::v0::Convert>(w_i4, element::f16);
    auto scale = op::v0::Constant::create(element::f16, Shape{num_experts, 1}, std::vector<float>(num_experts, 1.0f));
    auto w_scaled = std::make_shared<op::v1::Multiply>(w_fp16, scale);
    auto w_fp32 = std::make_shared<op::v0::Convert>(w_scaled, element::f32);

    // MatMul: [1, hidden_dim] x [num_experts, hidden_dim]^T → [1, num_experts]
    auto matmul = std::make_shared<op::v0::MatMul>(router_input, w_fp32, false, true);
    matmul->set_friendly_name("__module.model.layer0.mlp.router/aten::linear/MatMul");

    // Bias Add
    auto bias = op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto add = std::make_shared<op::v1::Add>(matmul, bias);
    add->set_friendly_name("__module.model.layer0.mlp.router/aten::linear/Add");

    // TopK
    Output<Node> k_input;
    if (k_param_as_input) {
        auto k_param = std::make_shared<op::v0::Parameter>(element::i64, Shape{});
        k_param->set_friendly_name("k_param");
        params.push_back(k_param);
        k_input = k_param->output(0);
    } else {
        k_input = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k_value})->output(0);
    }
    auto topk =
        std::make_shared<op::v11::TopK>(add, k_input, -1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::NONE);
    topk->set_friendly_name("__module.model.layer0.mlp.router/aten::topk/TopK");

    // Softmax on TopK values (output 0)
    auto softmax = std::make_shared<op::v8::Softmax>(topk->output(0), 1);

    // Slice: required as pattern root
    auto begin_c = op::v0::Constant::create(element::i64, Shape{1}, {0LL});
    auto end_c = op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{k_value});
    auto step_c = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
    auto axes_c = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
    auto slice = std::make_shared<op::v8::Slice>(softmax, begin_c, end_c, step_c, axes_c);

    auto result = std::make_shared<op::v0::Result>(slice);
    return std::make_shared<Model>(ResultVector{result}, params);
}

// Build a minimal Qwen3 Router subgraph that matches the Qwen3Router pattern:
//   Convert(weight) → Multiply(weight, scale) → Convert → MatMul → Softmax
//   → TopK(K, MAX) → (values→ReduceSum, values/ReduceSum→Divide)
//   ScatterElementsUpdate(base, indices, Divide, axis)
//   → Transpose → Reshape → Unsqueeze
std::shared_ptr<Model> build_qwen3_router_graph(int64_t k_value, size_t hidden_dim = 16, size_t num_experts = 8) {
    // Router input: [1, hidden_dim]
    auto router_input = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, hidden_dim});
    router_input->set_friendly_name("router_input");

    // Weight path: Convert(Const) → Multiply → Convert → MatMul
    auto w_const = op::v0::Constant::create(element::f16,
                                            Shape{num_experts, hidden_dim},
                                            std::vector<float>(num_experts * hidden_dim, 1.0f));
    auto w_convert_in = std::make_shared<op::v0::Convert>(w_const, element::f32);
    auto scale = op::v0::Constant::create(element::f32, Shape{num_experts, 1}, std::vector<float>(num_experts, 1.0f));
    auto w_multiply = std::make_shared<op::v1::Multiply>(w_convert_in, scale);
    auto w_convert_out = std::make_shared<op::v0::Convert>(w_multiply, element::f32);

    auto matmul = std::make_shared<op::v0::MatMul>(router_input, w_convert_out, false, true);
    matmul->set_friendly_name("__module.model.layer0.mlp.router/MatMul");

    // Softmax before TopK (Qwen3 style: softmax → topk)
    auto softmax = std::make_shared<op::v8::Softmax>(matmul, 1);

    // K constant and TopK
    auto k_const = op::v0::Constant::create(element::i64, Shape{}, std::vector<int64_t>{k_value});
    auto topk =
        std::make_shared<op::v11::TopK>(softmax, k_const, -1, op::v11::TopK::Mode::MAX, op::v11::TopK::SortType::NONE);
    topk->set_friendly_name("__module.model.layer0.mlp.router/TopK");

    // Renormalization: values / ReduceSum(values)
    auto reduce_axes = op::v0::Constant::create(element::i64, Shape{1}, {1LL});
    auto reduce_sum = std::make_shared<op::v1::ReduceSum>(topk->output(0), reduce_axes, true);
    auto divide = std::make_shared<op::v1::Divide>(topk->output(0), reduce_sum);

    // ScatterElementsUpdate: scatter normalized scores back to [1, num_experts]
    auto base = op::v0::Constant::create(element::f32, Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    auto scatter_axis = op::v0::Constant::create(element::i64, Shape{}, {1LL});
    auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(base, topk->output(1), divide, scatter_axis);

    // Transpose: [1, num_experts] → [num_experts, 1]
    auto t_order = op::v0::Constant::create(element::i32, Shape{2}, std::vector<int32_t>{1, 0});
    auto transpose = std::make_shared<op::v1::Transpose>(scatter, t_order);

    // Reshape: [num_experts, 1] → [num_experts, 1, 1]
    auto reshape_shape =
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{static_cast<int64_t>(num_experts), 1, 1});
    auto reshape = std::make_shared<op::v1::Reshape>(transpose, reshape_shape, false);

    // Unsqueeze: pattern root
    auto unsqueeze_axis = op::v0::Constant::create(element::i64, Shape{}, {3LL});
    auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(reshape, unsqueeze_axis);

    auto result = std::make_shared<op::v0::Result>(unsqueeze);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{router_input});
}

// Helper: find the first TopK node in a model
std::shared_ptr<op::v11::TopK> find_topk(const std::shared_ptr<Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (auto topk = std::dynamic_pointer_cast<op::v11::TopK>(node)) {
            return topk;
        }
    }
    return nullptr;
}

// ============================================================================
// GPTOSSRouter tests
// ============================================================================

class GPTOSSRouterTagK : public ::testing::Test {
protected:
    // Run GPTOSSRouter on the given model and return the model.
    void run_pass(const std::shared_ptr<Model>& model) {
        auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::moe::GPTOSSRouter>(snapshot, "router");
        rewr.run_on_model(model);
    }
};

TEST_F(GPTOSSRouterTagK, TagsTopKWithCorrectK) {
    constexpr int64_t K = 4;
    auto model = build_gptoss_router_graph(K);
    run_pass(model);

    auto topk = find_topk(model);
    ASSERT_NE(topk, nullptr) << "TopK node not found in model";

    const auto& rt = topk->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
        << "rt_info[\"" << ov::npuw::patterns::moe::RT_INFO_MOE_K << "\"] was not set by GPTOSSRouter";
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

TEST_F(GPTOSSRouterTagK, DoesNotIsolateNodes) {
    // GPTOSSRouter callback must return false without calling isolate_node().
    // Verify by building a Snapshot with populated groups and checking that
    // no node ends up with the "router" isolation tag after the pass runs.
    constexpr int64_t K = 4;
    auto model = build_gptoss_router_graph(K);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    snapshot->singleGroup();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::GPTOSSRouter>(snapshot, "router");
    rewr.run_on_model(model);

    const auto& node_to_gr = snapshot->getNodeToGroupMap();
    for (const auto& [node, group] : *node_to_gr) {
        EXPECT_NE(group->isolatedTag(), "router")
            << "Node \"" << node->get_friendly_name() << "\" was unexpectedly isolated with tag \"router\"";
    }
}

TEST_F(GPTOSSRouterTagK, NonConstKNotTagged) {
    // When K comes from a runtime input (Parameter), tag_topk_k() must return
    // false and leave rt_info untouched.
    constexpr int64_t K = 4;
    auto model = build_gptoss_router_graph(K, /*k_param_as_input=*/true);
    run_pass(model);

    auto topk = find_topk(model);
    ASSERT_NE(topk, nullptr);
    const auto& rt = topk->get_rt_info();
    EXPECT_EQ(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
        << "rt_info K should NOT be set when K input is not a Constant";
}

TEST_F(GPTOSSRouterTagK, ZeroKNotTagged) {
    // K=0 is invalid; tag_topk_k() must return false.
    constexpr int64_t K = 0;
    auto model = build_gptoss_router_graph(K);
    // Shape inference will fail for K=0 TopK, so skip validation.
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::GPTOSSRouter>(snapshot, "router");
    rewr.run_on_model(model);

    auto topk = find_topk(model);
    ASSERT_NE(topk, nullptr);
    const auto& rt = topk->get_rt_info();
    EXPECT_EQ(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end()) << "rt_info K should NOT be set when K=0";
}

// ============================================================================
// Qwen3Router tests
// ============================================================================

class Qwen3RouterTagK : public ::testing::Test {
protected:
    void run_pass(const std::shared_ptr<Model>& model) {
        auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::moe::Qwen3Router>(snapshot, "router");
        rewr.run_on_model(model);
    }
};

TEST_F(Qwen3RouterTagK, TagsTopKWithCorrectK) {
    constexpr int64_t K = 2;
    auto model = build_qwen3_router_graph(K);
    run_pass(model);

    auto topk = find_topk(model);
    ASSERT_NE(topk, nullptr) << "TopK node not found in model";

    const auto& rt = topk->get_rt_info();
    ASSERT_NE(rt.find(ov::npuw::patterns::moe::RT_INFO_MOE_K), rt.end())
        << "rt_info[\"" << ov::npuw::patterns::moe::RT_INFO_MOE_K << "\"] was not set by Qwen3Router";
    EXPECT_EQ(rt.at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), static_cast<size_t>(K));
}

TEST_F(Qwen3RouterTagK, DoesNotIsolateNodes) {
    constexpr int64_t K = 2;
    auto model = build_qwen3_router_graph(K);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
    snapshot->singleGroup();

    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::moe::Qwen3Router>(snapshot, "router");
    rewr.run_on_model(model);

    const auto& node_to_gr = snapshot->getNodeToGroupMap();
    for (const auto& [node, group] : *node_to_gr) {
        EXPECT_NE(group->isolatedTag(), "router")
            << "Node \"" << node->get_friendly_name() << "\" was unexpectedly isolated with tag \"router\"";
    }
}

}  // namespace
