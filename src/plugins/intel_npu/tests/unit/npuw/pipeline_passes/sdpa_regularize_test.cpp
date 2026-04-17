// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for ov::npuw::patterns::regularize::SoftmaxScalarShiftElimination.
//
// Background: Gemma4 exports an attention sub-graph of the form
//   MatMul → Add(mask) → Add(1e-7) → Softmax → MatMul
// The inner Add adds a tiny scalar epsilon before Softmax for numerical
// stability.  Because softmax(x + c) == softmax(x) for any scalar constant c,
// that Add is mathematically redundant and can be safely removed.
//
// SoftmaxScalarShiftElimination detects this pattern and bypasses the Add,
// wiring its non-constant input directly into Softmax.  It also handles the
// case where the constant is wrapped in a Convert node (the Gemma4 export
// produces a f64 constant that is then converted to f32).

#include <gtest/gtest.h>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/patterns/sdpa.hpp"

namespace {

using ov::npuw::patterns::regularize::SoftmaxScalarShiftElimination;

template <class Op>
static std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<Op>(op);
    });
}

// Run the pass as part of a GraphRewrite (matching the registration context in
// RegularizeSDPA::run_on_model) and return whether the graph changed.
static bool run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<SoftmaxScalarShiftElimination>();
    return rewr.run_on_model(model);
}

// ---------------------------------------------------------------------------
// Positive tests – the Add must be eliminated
// ---------------------------------------------------------------------------

// softmax(x + scalar_const) → softmax(x)   (scalar on RHS)
TEST(SoftmaxScalarShiftEliminationTest, EliminatesScalarConstOnRhs) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8});
    auto eps = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1e-7f});
    auto add = std::make_shared<ov::op::v1::Add>(param, eps);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, -1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    EXPECT_TRUE(run_pass(model));

    // Add must be removed from the graph.
    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), 0u);
    // Softmax now reads directly from the parameter.
    EXPECT_EQ(softmax->input_value(0).get_node_shared_ptr(), param);
}

// softmax(scalar_const + x) → softmax(x)   (scalar on LHS)
TEST(SoftmaxScalarShiftEliminationTest, EliminatesScalarConstOnLhs) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8});
    auto eps = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1e-7f});
    auto add = std::make_shared<ov::op::v1::Add>(eps, param);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, -1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    EXPECT_TRUE(run_pass(model));

    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), 0u);
    EXPECT_EQ(softmax->input_value(0).get_node_shared_ptr(), param);
}

// Gemma4 actual pattern: the eps constant is wrapped in a Convert node
// (f64 Constant → Convert(f32) → Add → Softmax).
TEST(SoftmaxScalarShiftEliminationTest, EliminatesConvertedScalarConst) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8});
    auto eps_f64 = ov::op::v0::Constant::create(ov::element::f64, ov::Shape{1}, {1.0000000116860974e-7});
    auto eps_f32 = std::make_shared<ov::op::v0::Convert>(eps_f64, ov::element::f32);
    auto add = std::make_shared<ov::op::v1::Add>(param, eps_f32);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, -1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    EXPECT_TRUE(run_pass(model));

    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), 0u);
    EXPECT_EQ(softmax->input_value(0).get_node_shared_ptr(), param);
}

// A true scalar constant (empty shape {}) whose shape_size is 1.
TEST(SoftmaxScalarShiftEliminationTest, EliminatesEmptyShapeScalarConst) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8});
    auto eps = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-7f});
    auto add = std::make_shared<ov::op::v1::Add>(param, eps);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, -1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    EXPECT_TRUE(run_pass(model));

    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), 0u);
    EXPECT_EQ(softmax->input_value(0).get_node_shared_ptr(), param);
}

// ---------------------------------------------------------------------------
// Negative tests – the Add must NOT be eliminated
// ---------------------------------------------------------------------------

// softmax(x + attention_mask) must be preserved: mask is a multi-element tensor,
// not a scalar shift.
TEST(SoftmaxScalarShiftEliminationTest, DoesNotEliminateNonScalarTensorAdd) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8});
    // mask has 32 elements — shape_size != 1
    auto mask = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4, 8}, std::vector<float>(32, 0.0f));
    auto add = std::make_shared<ov::op::v1::Add>(param, mask);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, -1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    EXPECT_FALSE(run_pass(model));

    // Graph must be unchanged.
    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), 1u);
    EXPECT_EQ(softmax->input_value(0).get_node_shared_ptr().get(), add.get());
}

// softmax(param1 + param2) must be preserved: neither input is a constant.
TEST(SoftmaxScalarShiftEliminationTest, DoesNotEliminateTwoNonConstantInputs) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8});
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4, 8});
    auto add = std::make_shared<ov::op::v1::Add>(param1, param2);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, -1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param1, param2});

    EXPECT_FALSE(run_pass(model));

    EXPECT_EQ(count_ops<ov::op::v1::Add>(model), 1u);
    EXPECT_EQ(softmax->input_value(0).get_node_shared_ptr().get(), add.get());
}

}  // namespace
