// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

// Buffers one transformer layer boundary's hidden-state output across a window of
// chunks, so a layer-major chunked-prefill driver can accumulate every chunk's
// per-layer output before invoking a single chunk-batched MoE call, then scatter
// MoE's output back to each chunk's slot. One instance is created per MoE-bearing
// layer boundary, and reused (not reallocated) across layers once configured with
// the same window size/shape.
//
// All chunks share ONE contiguous backing tensor (sized num_chunks * chunk_shape
// along chunk_dim); per-chunk slots are zero-copy ROI views into it (see
// util::make_tensor_slice). This lets MoE's batched call consume/produce the whole
// window in one shot with no extra concat/copy step, at the cost of exactly
// num_chunks * chunk_shape memory - the inherent price of batching, controlled by
// the num_chunks (batch window) knob.
class HiddenStateBridge {
public:
    using AllocFcn = std::function<util::TensorPtr(const ov::element::Type&, const ov::Shape&)>;

    // (Re)allocates a single buffer holding num_chunks copies of chunk_shape,
    // concatenated along chunk_dim, via alloc. No-op if already configured with
    // the same parameters.
    void configure(std::size_t num_chunks,
                  uint32_t chunk_dim,
                  const ov::element::Type& type,
                  const ov::Shape& chunk_shape,
                  const AllocFcn& alloc);

    std::size_t num_chunks() const;

    // Zero-copy view of chunk_id's region within the shared buffer.
    util::TensorPtr slot(std::size_t chunk_id) const;

    // The whole contiguous buffer spanning all chunks - what a batched MoE call consumes/produces.
    const util::TensorPtr& whole() const;

    // Bytes currently backing this bridge, 0 if never configure()'d - lets a driver report its
    // own peak buffer memory overhead without needing to know the buffer's shape/type itself.
    std::size_t byte_size() const;

private:
    util::TensorPtr m_buffer;
    uint32_t m_chunk_dim = 0;
    std::size_t m_num_chunks = 0;
    ov::Shape m_chunk_shape;
    ov::element::Type m_type;
};

}  // namespace npuw
}  // namespace ov
