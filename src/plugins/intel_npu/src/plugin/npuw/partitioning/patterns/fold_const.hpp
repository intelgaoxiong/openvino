// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

// Lightweight constant-folding passes for shape-compute chains that appear in
// extracted MoE expert subgraphs after RegularizeSDPA folds ShapeOf(param)
// into a Constant.
//
// Target chain (each step enables the next):
//   ShapeOf(param)  --[RegularizeSDPA/ShapeOfParameter]--> Const
//   ShapeOf(any)    --[FoldShapeOf]--> Const   (bound-based, no input constraint)
//   Gather(C,C,C)   --[FoldGatherOfConst]--> Const
//   Unsqueeze(C,C)  --[FoldUnsqueezeOfConst]--> Const
//   Concat(C...)    --[FoldConcatOfConsts]--> Const
//
// NOTE: Convert and Broadcast are intentionally NOT folded here to avoid
// inadvertently dequantizing compressed (nf4/int4) weight constants.
//
// Usage:
//   ov::pass::GraphRewrite rewr;
//   rewr.add_matcher<FoldShapeOf>();
//   rewr.add_matcher<FoldGatherOfConst>();
//   rewr.add_matcher<FoldUnsqueezeOfConst>();
//   rewr.add_matcher<FoldConcatOfConsts>();
//   rewr.run_on_model(model);

namespace ov {
namespace npuw {
namespace patterns {
namespace util {

// Fold ShapeOf(any) → Constant when the output tensor has a known upper bound.
// More general than RegularizeSDPA::ShapeOfParameter: no constraint on input type.
class FoldShapeOf : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldShapeOf");
    FoldShapeOf();
};

// Fold Gather(Constant, Constant, Constant) → Constant.
class FoldGatherOfConst : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldGatherOfConst");
    FoldGatherOfConst();
};

// Fold Unsqueeze(Constant, Constant) → Constant.
class FoldUnsqueezeOfConst : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldUnsqueezeOfConst");
    FoldUnsqueezeOfConst();
};

// Fold Concat whose every input is a Constant → Constant.
// The pattern matches any Concat; the callback enforces the all-constant
// precondition so no variant for each arity is needed.
class FoldConcatOfConsts : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::util::FoldConcatOfConsts");
    FoldConcatOfConsts();
};

}  // namespace util
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
