// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {

// Physical layout maintained by write_kv_slice_sliding() for a saturated (capacity-C,
// total_tokens > C) sliding-window buffer:
//   - LeftAligned: the most recent C tokens are kept left-aligned at [0, C), in strict
//     chronological order. Reaching this state requires shifting the surviving tail
//     forward every time the window (re)saturates.
//   - Circular: token at absolute position p always lives at physical index (p % C) -
//     no data is ever moved, only overwritten in place. This is mathematically
//     equivalent for callers whose only consumers are (a) the compiled model's mask,
//     which gates the past-KV region with an unconditional "structural index <
//     past_kv_len" check (no per-column temporal identity check - see
//     rebuild_sliding_window_mask() in sliding_window_mask.cpp) and (b) attention
//     itself, which is a permutation-invariant reduction over the visible K/V columns.
//     RoPE is baked into K once at write time (using the token's own true absolute
//     position), never recomputed from buffer physical position at read time, so
//     reordering physical slots doesn't affect correctness either.
//     Circular MUST NOT be used for a buffer that is ever read back elsewhere as a
//     plain contiguous/left-aligned source (e.g. Continuous strategy's variant-switch
//     migration or chunked-prefill past-KV reuse) unless that reader is updated to
//     unwrap the circular layout first.
enum class SlidingBufferLayout { LeftAligned, Circular };

// Writes `src_new_kv` (holding the freshly-produced KV content, of which the last
// `num_new_tokens` entries along `src_kv_dim` are meaningful) into `dst_tensor`'s
// past-KV buffer along `dst_kv_dim`, honoring `dst_tensor`'s own capacity
// (dst_tensor->get_shape()[dst_kv_dim]) - which may be smaller than the logical
// "total tokens seen so far" for Sliding Window Attention (SWA) layers, since their
// past_key_values Parameter is reshaped to the (smaller) window size at compile time.
//
// `layout` selects the physical arrangement used once the window (re)saturates - see
// SlidingBufferLayout above. Defaults to LeftAligned, matching all pre-existing
// callers' expectations.
//
// For non-SWA layers (capacity >= total tokens ever seen), this is equivalent to the
// original unconditional "write at [old_total, new_total)" behavior in both layouts -
// no shifting/wrapping ever occurs and this call is a drop-in replacement.
//
// NB: `src_new_kv` is expected to follow the "present/output" convention - its
// meaningful content is right-aligned at the tail (mirrors update_kvcache_for's
// existing src_seq_len > num_tokens handling). Persistent *past* buffers reused as a
// source (e.g. chunked-prefill's own past_key_values) are LEFT-aligned instead and
// must be pre-sliced by the caller to their valid prefix before being passed in here.
void write_kv_slice_sliding(ov::SoPtr<ov::ITensor> dst_tensor,
                            ov::SoPtr<ov::ITensor> src_new_kv,
                            uint32_t dst_kv_dim,
                            uint32_t src_kv_dim,
                            uint32_t num_stored_tokens_before,
                            uint32_t num_new_tokens,
                            SlidingBufferLayout layout = SlidingBufferLayout::LeftAligned);

}  // namespace util

/**
 * @brief Manages a single persistent sliding-window KV buffer for one key or value stream
 * of one Sliding-Window-Attention (SWA) transformer layer.
 *
 * Unlike KVCacheBlockManager (a growable pool of interchangeable blocks for full-attention
 * layers), a sliding-window layer needs exactly one fixed-size, never-reallocated buffer:
 * a per-token-exact window cannot be represented by whole-block eviction/rotation, since
 * block-granularity eviction can only keep/evict in units of block_size, which cannot
 * reproduce an exact window_size cutoff whenever it doesn't land on a block boundary.
 *
 * The buffer is exposed to the compiled model as `window_size / block_size` numbered
 * ports, each a fixed-size adjacent VIEW into the buffer (see get_slot_view()). Callers
 * must bind each view to its port with set_tensor() exactly ONCE and never rebind it
 * again - the NPU zero-copy remote-tensor backend requires every bound port to stay at a
 * fixed, stable address for its entire lifetime (rebinding to a shifting offset every
 * step triggers "Strided remote tensor is not supported for this port!" and can silently
 * produce wrong results, not just run slower). Only the buffer's CONTENT changes, via
 * update(), which writes using SlidingBufferLayout::Circular so no existing content is
 * ever shifted or copied elsewhere - see write_kv_slice_sliding()'s doc comment above for
 * why this is correctness-safe for this buffer's only consumer (the compiled model
 * itself, through its fixed port views).
 *
 * Example usage:
 *   KVCacheSlidingWindowManager mgr(block_size, window_size, base_shape, elem_type, "NPU", plugin);
 *   for (uint32_t slot = 0; slot < mgr.num_slots(); ++slot) {
 *       request->set_tensor(numbered_port[slot], mgr.get_slot_view(slot));  // bind ONCE
 *   }
 *   mgr.update(new_kv_output, src_kv_dim, num_stored_tokens_before, num_new_tokens);
 */
class KVCacheSlidingWindowManager {
public:
    /**
     * @brief Construct a new KV Cache Sliding Window Manager
     *
     * @param block_size Number of tokens per numbered port slot (matches the model's
     *                   compiled block granularity)
     * @param window_size Attention window size in tokens. Must be a multiple of
     *                    block_size (a partial trailing block is not yet supported).
     * @param base_shape Base shape for one block_size-wide slot
     *                   [batch, num_heads, seq_len, head_dim] (seq_len == block_size in
     *                   dim 2 or dim 3 - the sequence dim is auto-detected)
     * @param elem_type Element type (e.g., fp16, fp32)
     * @param device Target device for memory allocation ("NPU", "CPU")
     * @param plugin Plugin instance for memory allocation
     */
    KVCacheSlidingWindowManager(uint32_t block_size,
                                uint32_t window_size,
                                const ov::Shape& base_shape,
                                ov::element::Type elem_type,
                                const std::string& device,
                                const std::shared_ptr<const ov::IPlugin>& plugin);

    ~KVCacheSlidingWindowManager() = default;

    // Disable copy
    KVCacheSlidingWindowManager(const KVCacheSlidingWindowManager&) = delete;
    KVCacheSlidingWindowManager& operator=(const KVCacheSlidingWindowManager&) = delete;

    // Allow move
    KVCacheSlidingWindowManager(KVCacheSlidingWindowManager&&) = default;
    KVCacheSlidingWindowManager& operator=(KVCacheSlidingWindowManager&&) = default;

    /**
     * @brief Number of block_size-wide numbered port slots the window buffer is divided
     * into (window_size / block_size).
     */
    uint32_t num_slots() const {
        return num_slots_;
    }

    /**
     * @brief Fixed, block_size-wide view into the window buffer at the given slot.
     * Intended to be bound to a numbered port with set_tensor() exactly once - the
     * returned view always aliases the same underlying memory, so it observes every
     * subsequent update()/reset() automatically without needing to be re-bound.
     *
     * @param slot_idx Slot index, must be < num_slots()
     */
    ov::SoPtr<ov::ITensor> get_slot_view(uint32_t slot_idx) const;

    /**
     * @brief Write this call's freshly produced KV content into the window buffer using
     * SlidingBufferLayout::Circular (see write_kv_slice_sliding()).
     *
     * @param src_new_kv Freshly produced KV tensor (e.g. a "present.N.key/value" output),
     *                   whose last `num_new_tokens` entries along `src_kv_dim` are meaningful
     * @param src_kv_dim Sequence dimension of `src_new_kv`
     * @param num_stored_tokens_before Total tokens seen by this layer before this call
     * @param num_new_tokens Number of new tokens produced by this call
     */
    void update(ov::SoPtr<ov::ITensor> src_new_kv,
               uint32_t src_kv_dim,
               uint32_t num_stored_tokens_before,
               uint32_t num_new_tokens);

    /**
     * @brief Zero-fill the window buffer's bytes (content only - the buffer itself, and
     * any port views already bound to it via get_slot_view(), remain valid and do not
     * need to be re-bound).
     */
    void reset();

    /**
     * @brief Get window size (tokens); equals num_slots() * block_size.
     */
    uint32_t get_window_size() const {
        return window_size_;
    }

    /**
     * @brief Sequence dimension (2 or 3) auto-detected in the constructor.
     */
    uint32_t get_seq_dim() const {
        return seq_dim_;
    }

private:
    uint32_t block_size_;
    uint32_t window_size_;
    uint32_t num_slots_;
    uint32_t seq_dim_;
    ov::SoPtr<ov::ITensor> window_;
};

}  // namespace npuw
}  // namespace ov
