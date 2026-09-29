// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hidden_state_bridge.hpp"

#include "infer_request_utils.hpp"  // util::make_tensor_slice
#include "openvino/core/except.hpp"

void ov::npuw::HiddenStateBridge::configure(std::size_t num_chunks,
                                             uint32_t chunk_dim,
                                             const ov::element::Type& type,
                                             const ov::Shape& chunk_shape,
                                             const AllocFcn& alloc) {
    if (num_chunks == m_num_chunks && chunk_dim == m_chunk_dim && type == m_type && chunk_shape == m_chunk_shape) {
        return;  // already configured for this window
    }
    ov::Shape whole_shape = chunk_shape;
    whole_shape[chunk_dim] *= num_chunks;
    m_buffer = alloc(type, whole_shape);
    m_num_chunks = num_chunks;
    m_chunk_dim = chunk_dim;
    m_chunk_shape = chunk_shape;
    m_type = type;
}

std::size_t ov::npuw::HiddenStateBridge::num_chunks() const {
    return m_num_chunks;
}

ov::npuw::util::TensorPtr ov::npuw::HiddenStateBridge::slot(std::size_t chunk_id) const {
    OPENVINO_ASSERT(chunk_id < m_num_chunks, "HiddenStateBridge: chunk_id out of range");
    const uint32_t chunk_len = static_cast<uint32_t>(m_chunk_shape[m_chunk_dim]);
    const uint32_t start = static_cast<uint32_t>(chunk_id) * chunk_len;
    return util::make_tensor_slice(m_buffer, m_chunk_dim, start, start + chunk_len);
}

const ov::npuw::util::TensorPtr& ov::npuw::HiddenStateBridge::whole() const {
    return m_buffer;
}

std::size_t ov::npuw::HiddenStateBridge::byte_size() const {
    return m_buffer ? m_buffer->get_byte_size() : 0u;
}
