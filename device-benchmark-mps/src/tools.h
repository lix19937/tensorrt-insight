/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#include "NvInfer.h"

void checkCudaErrors(cudaError_t err);

constexpr int divUp(int a, int b) { return (a + b - 1) / b; }

// RAII helpers to automatically manage memory resource and TensorRT objects.
template <typename T> struct TrtDeleter
{
    void operator()(T *p) noexcept
    {
        if (p != nullptr)
            delete p;
    }
};

template <typename T> struct CuMemDeleter
{
    void operator()(T *p) noexcept { checkCudaErrors(cudaFree(p)); }
};

template <typename T, template <typename> typename DeleterType = TrtDeleter>
using UniqPtr = std::unique_ptr<T, DeleterType<T>>;

template <typename T> UniqPtr<T, CuMemDeleter> mallocCudaMem(size_t nbElems)
{
    T *ptr = nullptr;
    checkCudaErrors(cudaMalloc((void **)&ptr, sizeof(T) * nbElems));
    return UniqPtr<T, CuMemDeleter>{ptr};
}

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
  public:
    Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

// extern Logger gLogger;

struct StreamDeleter
{
    void operator()(CUstream_st *stream) noexcept { checkCudaErrors(cudaStreamDestroy(stream)); }
};

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStream(int flags = cudaStreamDefault, int priority = 0);

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStreamNew();
struct EventDeleter
{
    void operator()(CUevent_st *event) noexcept { checkCudaErrors(cudaEventDestroy(event)); }
};

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEvent(int flags = cudaEventDefault);

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEventNew(void);

#ifndef checkCudaErrorsDRV
#define checkCudaErrorsDRV(err) __checkCudaErrorsDRV(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrorsDRV(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        const char *errorStr = NULL;
        cuGetErrorString(err, &errorStr);
        fprintf(stderr,
                "checkCudaErrorsDRV() Driver API error = %04d \"%s\" from file <%s>, "
                "line %i.\n",
                err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

#endif
