// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "duplicate_shared_kv_concat.hpp"

#include <optional>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "../logging.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace {

// Returns true when all inputs of `concat` except the last are Parameters or
// Convert(Parameter).  These are the "past KV block" inputs; the final input
// is the current-chunk KV projection.
bool has_param_past_inputs(const std::shared_ptr<ov::op::v0::Concat>& concat) {
    const size_t n = concat->get_input_size();
    if (n < 2)
        return false;
    for (size_t i = 0; i + 1 < n; ++i) {
        auto inp = concat->get_input_node_shared_ptr(i);
        if (ov::as_type_ptr<ov::op::v0::Parameter>(inp))
            continue;
        if (auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(inp)) {
            if (ov::as_type_ptr<ov::op::v0::Parameter>(cvt->get_input_node_shared_ptr(0)))
                continue;
        }
        return false;
    }
    return true;
}

struct KVBroadcastChain {
    std::shared_ptr<ov::op::v0::Concat>    concat;
    std::shared_ptr<ov::op::v0::Unsqueeze> unsqueeze;  // may be nullptr
    std::shared_ptr<ov::Node>              broadcast;  // v1 or v3 Broadcast; may be nullptr
    std::shared_ptr<ov::op::v1::Reshape>   reshape;    // fan-out root
};

// Try to match the pattern ending at `node`:
//
//   Concat(Parameters…, current_KV)
//       → [Unsqueeze] → [Broadcast] → Reshape  [fan-out > 1]
//
// Both Unsqueeze and Broadcast are optional (present in GQA models to expand
// K/V heads; absent in MHA models where Q/K/V head counts already match).
std::optional<KVBroadcastChain> try_match(const std::shared_ptr<ov::Node>& node) {
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node);
    if (!reshape || reshape->output(0).get_target_inputs().size() <= 1)
        return std::nullopt;

    KVBroadcastChain chain;
    chain.reshape = reshape;

    // Walk backwards: Reshape ← [Broadcast] ← [Unsqueeze] ← Concat
    std::shared_ptr<ov::Node> cur = reshape->get_input_node_shared_ptr(0);

    if (ov::is_type<ov::op::v1::Broadcast>(cur) || ov::is_type<ov::op::v3::Broadcast>(cur)) {
        chain.broadcast = cur;
        cur = cur->get_input_node_shared_ptr(0);
    }

    if (auto unsq = ov::as_type_ptr<ov::op::v0::Unsqueeze>(cur)) {
        chain.unsqueeze = unsq;
        cur = cur->get_input_node_shared_ptr(0);
    }

    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(cur);
    if (!concat || !has_param_past_inputs(concat))
        return std::nullopt;

    chain.concat = concat;
    return chain;
}

// Clone the chain for every consumer beyond the first.
// The first consumer keeps the original (unmodified) chain.
void duplicate_for_extra_consumers(const KVBroadcastChain& chain) {
    // Snapshot consumers before touching the graph to avoid iterator invalidation.
    std::vector<ov::Input<ov::Node>> consumers;
    for (auto& ti : chain.reshape->output(0).get_target_inputs())
        consumers.push_back(ti);

    bool first = true;
    for (auto& consumer_input : consumers) {
        if (first) {
            first = false;
            continue;  // leave this one on the original chain
        }

        // --- Concat ---
        // Same inputs as the original: shared block Parameters + shared current_KV.
        auto new_concat = chain.concat->clone_with_new_inputs(chain.concat->input_values());
        ov::Output<ov::Node> data = new_concat->output(0);

        // --- Unsqueeze (optional) ---
        if (chain.unsqueeze) {
            ov::OutputVector inputs{data};
            for (size_t i = 1; i < chain.unsqueeze->get_input_size(); ++i)
                inputs.push_back(chain.unsqueeze->input_value(i));
            data = chain.unsqueeze->clone_with_new_inputs(inputs)->output(0);
        }

        // --- Broadcast (optional) ---
        if (chain.broadcast) {
            ov::OutputVector inputs{data};
            for (size_t i = 1; i < chain.broadcast->get_input_size(); ++i)
                inputs.push_back(chain.broadcast->input_value(i));
            data = chain.broadcast->clone_with_new_inputs(inputs)->output(0);
        }

        // --- Reshape ---
        {
            ov::OutputVector inputs{data};
            for (size_t i = 1; i < chain.reshape->get_input_size(); ++i)
                inputs.push_back(chain.reshape->input_value(i));
            data = chain.reshape->clone_with_new_inputs(inputs)->output(0);
        }

        consumer_input.replace_source_output(data);
    }
}

}  // namespace

bool DuplicateSharedKVConcat::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    // Snapshot the current node list so that newly inserted nodes are not
    // visited during the same pass iteration.
    const auto ops = model->get_ordered_ops();

    for (const auto& node : ops) {
        auto chain_opt = try_match(node);
        if (!chain_opt)
            continue;

        const size_t fan_out = chain_opt->reshape->output(0).get_target_inputs().size();
        LOG_DEBUG("DuplicateSharedKVConcat: Reshape \""
                  << node->get_friendly_name()
                  << "\" fan-out=" << fan_out
                  << " — duplicating KV broadcast chain for "
                  << (fan_out - 1) << " extra consumer(s)");

        duplicate_for_extra_consumers(*chain_opt);
        changed = true;
    }

    return changed;
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
