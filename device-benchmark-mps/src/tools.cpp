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

#include <tools.h>

void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorName(err));
}

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStream(int flags, int priority)
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithPriority(&stream, flags, priority));
    return std::unique_ptr<CUstream_st, StreamDeleter>{stream};
}

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStreamNew()
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return std::unique_ptr<CUstream_st, StreamDeleter>{stream};
}

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEvent(int flags)
{
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreateWithFlags(&event, flags));
    return std::unique_ptr<CUevent_st, EventDeleter>{event};
}

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEventNew(void)
{
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    return std::unique_ptr<CUevent_st, EventDeleter>{event};
}