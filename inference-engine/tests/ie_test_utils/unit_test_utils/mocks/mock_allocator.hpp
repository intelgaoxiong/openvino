// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Created by user on 20.10.16.
//

#pragma once

#include <gmock/gmock.h>

#include "ie_allocator.hpp"

class MockAllocator : public InferenceEngine::IAllocator {
public:
    MOCK_QUALIFIED_METHOD2(lock, noexcept, void*(void*, InferenceEngine::LockOp));
    MOCK_QUALIFIED_METHOD1(unlock, noexcept, void(void *));
    MOCK_QUALIFIED_METHOD1(alloc, noexcept, void*(size_t));
    // MOCK_QUALIFIED_METHOD1(alloc_cacheable, noexcept, void*(size_t));
    // MOCK_QUALIFIED_METHOD1(flush, noexcept, bool(void*, size_t));
    //MOCK_QUALIFIED_METHOD1(invalidate, noexcept, bool(void*, size_t));
    void* alloc_cacheable(size_t size) noexcept override { return nullptr; }
    bool flush(void* handle, const size_t size) noexcept override { return false; }
    bool invalidate(void* handle, const size_t size) noexcept override { return false; }
    MOCK_QUALIFIED_METHOD1(free, noexcept, bool(void*));  // NOLINT(readability/casting)
};
