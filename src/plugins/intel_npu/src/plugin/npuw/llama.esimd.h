// Copyright (C) 2024 Intel Corporation
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you ("License"). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit
// this software or the related documents without Intel's prior written
// permission.

// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly stated
// in the License.

#pragma once

#include <stdint.h>
#include <stdio.h>


#ifdef BUILD_WRAPPER_DLL // build a dll
#define LLAMA_ESIMD_API __declspec(dllexport)
#else
#define LLAMA_ESIMD_API __declspec(dllimport)
#endif


extern "C" bool LLAMA_ESIMD_API SetupLlamaEsimdEnvironment(); // optional call
extern "C" bool LLAMA_ESIMD_API InitializeLlamaEsimd(void *q); // optional call


extern "C" bool LLAMA_ESIMD_API RunGQA(void* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t context_len, uint32_t kv_head, uint32_t q_head, unsigned q_precision, unsigned o_precision);

