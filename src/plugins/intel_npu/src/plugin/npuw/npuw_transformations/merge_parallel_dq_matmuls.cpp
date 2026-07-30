// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "merge_parallel_dq_matmuls.hpp"

#include <cstring>
#include <iostream>
#include <limits>

#include "../logging.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/slice.hpp"

namespace ov::npuw {

namespace {

// Represents a matched DQ MatMul.  Currently only 3-D BMM weights are supported:
//
//   3-D weights (BMM, e.g. MoE gate+up experts):
//     Const(W)[batch,out,hidden] → Convert → Multiply(_, Scale[batch,out,1])
//                                → [Convert] → MatMul(act[batch,seq,hidden], -, false, transpose_b)
//
struct DQMatMulCandidate {
    std::shared_ptr<ov::op::v0::Constant> w;              // quantized weight constant
    std::shared_ptr<ov::op::v0::Constant> s;              // scale constant
    bool                                  s_at_mul_idx1;  // is 's' at Multiply input index 1?
    ov::element::Type                     cvt_w_dst;      // Convert(w) destination type
    bool                                  has_post_cvt;   // optional Convert after Multiply
    ov::element::Type                     post_cvt_dst;
    std::shared_ptr<ov::op::v0::MatMul>   matmul;
    bool                                  transpose_b;
    std::size_t                           axis;           // concat axis on W and S (output dim)
    ov::Output<ov::Node>                  act_output;     // activation output (key for grouping)
};

// Match the DQ weight chain feeding a 3-D BMM MatMul (e.g. MoE experts):
//
//    MatMul input 1:  [Convert] ← Multiply ← Convert ← Const(W)[batch,out,hidden]
//                              ↑ Scale[batch,out,1]
//
// Returns false if the pattern does not match.
static bool try_match(const std::shared_ptr<ov::op::v0::MatMul>& mm, DQMatMulCandidate& out) {
    // Output must be 3-D (shapes may be dynamic in the pre-ReshapeToStatic template model;
    // only the rank is required here).
    const auto out_pshape = mm->output(0).get_partial_shape();
    if (!out_pshape.rank().is_static() || out_pshape.rank().get_length() != 3)
        return false;

    // transpose_a must be false
    if (mm->get_transpose_a())
        return false;
    const bool transpose_b = mm->get_transpose_b();

    // --- Weight side (input 1) ---
    // Peel optional final Convert (the "decompression" fp16→fp32 cast)
    ov::Output<ov::Node> w_in = mm->input_value(1);
    bool has_post_cvt = false;
    ov::element::Type post_cvt_dst = ov::element::dynamic;
    if (auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(w_in.get_node_shared_ptr())) {
        has_post_cvt = true;
        post_cvt_dst = cvt->get_destination_type();
        w_in = cvt->input_value(0);
    }

    // w_in must be a Multiply
    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(w_in.get_node_shared_ptr());
    if (!mul)
        return false;

    // One Multiply input: Convert(Const(W))  — the other: Const(S)
    std::shared_ptr<ov::op::v0::Constant> w_const, s_const;
    ov::element::Type cvt_w_dst = ov::element::dynamic;
    bool s_at_idx1 = false;
    for (int i = 0; i < 2; ++i) {
        auto inp = mul->input_value(i).get_node_shared_ptr();
        auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(inp);
        if (!cvt)
            continue;
        auto wc = ov::as_type_ptr<ov::op::v0::Constant>(cvt->input_value(0).get_node_shared_ptr());
        if (!wc)
            continue;
        auto sc = ov::as_type_ptr<ov::op::v0::Constant>(mul->input_value(1 - i).get_node_shared_ptr());
        if (!sc)
            continue;
        w_const = wc;
        s_const = sc;
        cvt_w_dst = cvt->get_destination_type();
        s_at_idx1 = (1 - i == 1);
        break;
    }
    if (!w_const || !s_const)
        return false;

    const auto& w_sh = w_const->get_shape();
    const auto& s_sh = s_const->get_shape();

    // Determine concat axis (the output-feature dimension of W)
    std::size_t axis = 0;
    if (w_sh.size() == 3) {
        // 3-D BMM (e.g. MoE): W[batch, out, hidden] or W[batch, hidden, out]
        // batch dim (axis 0) is shared; merge along the output-feature dim.
        if (s_sh.size() != 3)
            return false;
        // transpose_b=true  → W is [batch, out, hidden]; output-dim is axis 1
        // transpose_b=false → W is [batch, hidden, out]; output-dim is axis 2
        axis = transpose_b ? 1u : 2u;
    } else {
        // Only 3-D BMM is supported for now; skip 2-D weights.
        return false;
    }

    // --- Activation side (input 0): peel optional Convert for grouping key ---
    ov::Output<ov::Node> act = mm->input_value(0);
    if (auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(act.get_node_shared_ptr()))
        act = cvt->input_value(0);

    out.w            = w_const;
    out.s            = s_const;
    out.s_at_mul_idx1 = s_at_idx1;
    out.cvt_w_dst    = cvt_w_dst;
    out.has_post_cvt = has_post_cvt;
    out.post_cvt_dst = post_cvt_dst;
    out.matmul       = mm;
    out.transpose_b  = transpose_b;
    out.axis         = axis;
    out.act_output   = act;
    return true;
}

// Evaluate Concat of Constants eagerly, returning a new folded Constant.
// Equivalent to the fold_if_all_const helper in fold_const.cpp but restricted to Concat.
// Concatenate constants along the given axis.
// For sub-byte types (NF4, INT4, etc.) ov::op::v0::Concat::evaluate() is not
// supported, so we fall back to a manual byte-level memcpy.  This is valid as
// long as the number of elements along axes *after* the concat axis is even
// (i.e., row boundaries are byte-aligned in the packed layout).
//
// Memory note: output is written directly into an ov::Tensor whose buffer is
// then *shared* (zero-copy) by the returned Constant via Constant(Tensor).
// The (type, shape, void*) constructor always copies, so using Tensor avoids
// a peak-memory spike of `total_bytes` during large NF4 weight merges.
static std::shared_ptr<ov::op::v0::Constant> eval_concat(
    const std::vector<std::shared_ptr<ov::op::v0::Constant>>& consts,
    int64_t axis) {
    const auto& elem_type = consts[0]->get_element_type();

    // Compute output shape
    auto out_shape = consts[0]->get_shape();
    for (std::size_t i = 1; i < consts.size(); ++i)
        out_shape[axis] += consts[i]->get_shape()[axis];

    if (elem_type.bitwidth() < 8) {
        // --- packed sub-byte path (NF4, INT4, ...) ---
        // Elements trailing the concat axis (must be even for byte alignment).
        std::size_t trailing_elems = 1;
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < out_shape.size(); ++d)
            trailing_elems *= out_shape[d];

        std::size_t num_outer = 1;
        for (std::size_t d = 0; d < static_cast<std::size_t>(axis); ++d)
            num_outer *= out_shape[d];

        // Each packed byte holds (8 / bitwidth) elements.
        const std::size_t elems_per_byte = 8u / elem_type.bitwidth();

        // bytes per outer slice in the output
        const std::size_t out_slice_bytes = out_shape[axis] * trailing_elems / elems_per_byte;

        OPENVINO_ASSERT((out_shape[axis] * trailing_elems) % elems_per_byte == 0,
                        "eval_concat: axis slice is not byte-aligned for packed type ", elem_type);

        // Precompute per-input slice sizes and cumulative destination offsets within
        // one outer row, so the inner loop avoids redundant shape lookups.
        const std::size_t num_consts = consts.size();
        std::vector<std::size_t> c_slice_bytes(num_consts);
        std::vector<std::size_t> dest_off(num_consts);
        for (std::size_t ci = 0; ci < num_consts; ++ci) {
            c_slice_bytes[ci] = consts[ci]->get_shape()[static_cast<std::size_t>(axis)] * trailing_elems / elems_per_byte;
            dest_off[ci] = (ci == 0) ? 0 : dest_off[ci - 1] + c_slice_bytes[ci - 1];
        }

        // Allocate output tensor directly; Constant(Tensor) shares this buffer
        // (zero-copy via SharedBuffer<Tensor>), avoiding a redundant memcpy.
        ov::Tensor out_tensor(elem_type, out_shape);
        auto* dst = static_cast<uint8_t*>(out_tensor.data());

        for (std::size_t outer = 0; outer < num_outer; ++outer) {
            const std::size_t out_base = outer * out_slice_bytes;
            for (std::size_t ci = 0; ci < num_consts; ++ci) {
                const auto* src = static_cast<const uint8_t*>(consts[ci]->get_data_ptr()) + outer * c_slice_bytes[ci];
                std::memcpy(dst + out_base + dest_off[ci], src, c_slice_bytes[ci]);
            }
        }
        return std::make_shared<ov::op::v0::Constant>(out_tensor);
    }

    // --- full-byte path: use OV evaluate() ---
    // Constant(Tensor) shares the tensor buffer (SharedBuffer<Tensor>, zero-copy).
    ov::OutputVector inputs;
    inputs.reserve(consts.size());
    for (auto& c : consts)
        inputs.push_back(c->output(0));
    auto concat_op = std::make_shared<ov::op::v0::Concat>(inputs, axis);

    ov::TensorVector in_tensors, out_tensors;
    in_tensors.reserve(consts.size());
    for (auto& c : consts)
        in_tensors.emplace_back(c->get_element_type(), c->get_shape(), const_cast<void*>(c->get_data_ptr()));
    out_tensors.emplace_back(concat_op->get_output_element_type(0), concat_op->get_output_shape(0));

    if (!concat_op->evaluate(out_tensors, in_tensors)) {
        return nullptr;
    }
    return std::make_shared<ov::op::v0::Constant>(out_tensors[0]);
}

// Grouping key: same activation output + same concat axis
struct GroupKey {
    ov::Node* act_node;
    std::size_t act_port;
    std::size_t axis;
    bool operator<(const GroupKey& o) const {
        if (act_node != o.act_node)
            return act_node < o.act_node;
        if (act_port != o.act_port)
            return act_port < o.act_port;
        return axis < o.axis;
    }
};

}  // anonymous namespace

bool MergeParallelDQMatMuls::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // --- Step 1: collect all matching MatMul candidates ---
    std::map<GroupKey, std::vector<DQMatMulCandidate>> groups;

    for (auto&& node : model->get_ordered_ops()) {
        auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(node);
        if (!mm)
            continue;
        DQMatMulCandidate cand;
        if (!try_match(mm, cand))
            continue;
        GroupKey k{cand.act_output.get_node(), cand.act_output.get_index(), cand.axis};
        groups[k].push_back(std::move(cand));
    }

    bool changed = false;

    // --- Step 2: merge each group that has >= 2 candidates ---
    for (auto& [key, cands] : groups) {
        if (cands.size() < 2)
            continue;

        const auto& first = cands[0];
        const auto& w0_sh = first.w->get_shape();
        const auto& s0_sh = first.s->get_shape();

        // Validate compatibility: same topology flags and same non-concat dims
        bool ok = true;
        for (const auto& c : cands) {
            if (c.cvt_w_dst    != first.cvt_w_dst    ||
                c.transpose_b  != first.transpose_b  ||
                c.has_post_cvt != first.has_post_cvt ||
                (c.has_post_cvt && c.post_cvt_dst != first.post_cvt_dst)) {
                ok = false;
                break;
            }
            const auto& c_w_sh = c.w->get_shape();
            const auto& c_s_sh = c.s->get_shape();
            if (c_w_sh.size() != w0_sh.size() || c_s_sh.size() != s0_sh.size()) {
                ok = false;
                break;
            }
            for (std::size_t d = 0; d < w0_sh.size(); ++d) {
                if (d == key.axis)
                    continue;
                if (c_w_sh[d] != w0_sh[d] || c_s_sh[d] != s0_sh[d]) {
                    ok = false;
                    break;
                }
            }
            if (!ok)
                break;
        }
        if (!ok) {
            LOG_VERB("MergeParallelDQMatMuls: skipping incompatible group (axis=" << key.axis << ")");
            continue;
        }

        // --- Step 3: concatenate W and S constants ---
        std::vector<std::shared_ptr<ov::op::v0::Constant>> w_consts, s_consts;
        for (const auto& c : cands) {
            w_consts.push_back(c.w);
            s_consts.push_back(c.s);
        }

        auto new_w = eval_concat(w_consts, static_cast<int64_t>(key.axis));
        auto new_s = eval_concat(s_consts, static_cast<int64_t>(key.axis));
        if (!new_w || !new_s) {
            LOG_VERB("MergeParallelDQMatMuls: constant eval failed, skipping group");
            continue;
        }
        new_w->set_friendly_name("NPUW/PMM/w_merged");
        new_s->set_friendly_name("NPUW/PMM/s_merged");

        // --- Step 4: rebuild DQ chain ---
        // Convert(new_w) → Multiply(_, new_s) → [Convert] → MatMul
        auto new_cvt_w = std::make_shared<ov::op::v0::Convert>(new_w, first.cvt_w_dst);

        std::shared_ptr<ov::Node> new_mul;
        if (first.s_at_mul_idx1)
            new_mul = std::make_shared<ov::op::v1::Multiply>(new_cvt_w->output(0), new_s->output(0));
        else
            new_mul = std::make_shared<ov::op::v1::Multiply>(new_s->output(0), new_cvt_w->output(0));

        std::shared_ptr<ov::Node> weight_into_mm = new_mul;

        if (first.has_post_cvt)
            weight_into_mm = std::make_shared<ov::op::v0::Convert>(weight_into_mm->output(0), first.post_cvt_dst);

        // New MatMul: reuse the activation input of the first candidate as-is
        // (it already includes any optional Convert on the activation side)
        auto new_mm = std::make_shared<ov::op::v0::MatMul>(cands[0].matmul->input_value(0),
                                                            weight_into_mm->output(0),
                                                            false,
                                                            first.transpose_b);
        new_mm->set_friendly_name("NPUW/PMM/mm_merged");

        // --- Step 5: slice output for each original MatMul consumer ---
        // MatMul output is [batch, seq, total_out_dim]; slice along dim 2.
        // batch=1 for dense FFN, batch=num_experts for MoE BMM.
        // Use INT32_MAX for batch/seq in Slice end so the pass also works on the
        // dynamic-shape template model (before ReshapeToStatic).  The Slice op
        // clamps end to the actual dimension size at runtime.
        constexpr int32_t kSliceEnd = std::numeric_limits<int32_t>::max();

        using ISz = std::vector<int32_t>;
        std::size_t offset = 0;
        for (const auto& cand : cands) {
            const std::size_t out_size = cand.w->get_shape()[key.axis];

            auto s_start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3},
                               ISz{0, 0, static_cast<int32_t>(offset)});
            auto s_end = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3},
                             ISz{kSliceEnd, kSliceEnd, static_cast<int32_t>(offset + out_size)});
            auto s_step = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, ISz{1, 1, 1});
            auto slice = std::make_shared<ov::op::v8::Slice>(new_mm->output(0), s_start, s_end, s_step);
            slice->set_friendly_name("NPUW/PMM/slice_" + std::to_string(offset));

            for (auto& ti : cand.matmul->output(0).get_target_inputs())
                ti.replace_source_output(slice->output(0));

            offset += out_size;
        }

        LOG_VERB("MergeParallelDQMatMuls: merged "
                 << cands.size() << " MatMuls"
                 << " (axis=" << key.axis << ", transpose_b=" << first.transpose_b << ")"
                 << " total_out=" << offset);
        changed = true;
    }

    return changed;
}

}  // namespace ov::npuw
