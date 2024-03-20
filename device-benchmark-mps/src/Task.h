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

#pragma once
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <tools.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "NvInfer.h"

//! 4 kinds of sync type
enum SyncType
{
    EventDefault   = 0,
    EventBlock     = 1,
    StreamDefault  = 2,
    StreamNonBlock = 3,
};

//! \class Task
//!
//! \brief Base class Task, which include some basic smart pointers: stream event etc.
//!
class Task
{
  public:
    void TaskInit(SyncType syncType);
    virtual ~Task()                   = default;
    virtual bool        Run()         = 0;
    virtual const char *getTaskType() = 0;
    void                ReportSyncType();
    void                Sync();
    //!
    //! get how many imgs has been totally processed
    //!
    int imgProcessed() { return mInference_count; };

  protected:
    std::unique_ptr<CUstream_st, StreamDeleter> mStream;
    std::unique_ptr<CUevent_st, EventDeleter>   mEvent;
    unsigned long long                          mLast_inference_time;
    unsigned long long                          mTotal_inference_time;
    int                                         mInference_count;
    std::string                                 mTaskName;
    SyncType                                    mSyncType;
};

//! \class TrtBenchTask
//!
//! \brief Task which can load a trt engine and run inference with random data.
//!
class TrtBenchTask : public Task
{
  public:
    TrtBenchTask() = delete;

    //!
    //! \brief init TrtBenchTask class object
    //!
    //! \param engineFilePath The path of trt engine file
    //! \param DLACore -1 : GPU, 0 : DLA0, 1 : DLA1
    //! \param streamCreateFlag create mStream use flag: cudaStreamDefault/cudaStreamNonBlocking
    //! \param eventCreateFlag create mEvent use flag: cudaEventDefault/cudaEventBlockingSync
    //! \param IsStreamSync true will call cudaStreamSynchronize() & false will call cudaEventSynchronize() everytime
    //! after inference
    //!
    //! 1. load engine from file
    //! 2. set DLA core & create cudastream with streamCreateFlag & create cudaevent with eventCreateFlag
    //! 3. allocate input/ouput cuda buffer for trt model
    //! 4. fill input buffer with random input
    //!
    TrtBenchTask(const char *engineFilePath, int DLACore, SyncType syncType, bool enableCudaGraph);

    //!
    //! \brief Run one time inference and call Synchronize function base on user's flag
    //!
    virtual bool Run();

    //!
    //! get Task Type: "GPU"/"DLA0"/"DLA1"
    //!
    const char *getTaskType();

    //!
    //! all the pointers is managed using smart point, So nothing todo here
    //!
    ~TrtBenchTask() {}

  private:
    // trt objects
    UniqPtr<nvinfer1::IRuntime>          mRuntime;
    UniqPtr<nvinfer1::ICudaEngine>       mEngine;
    UniqPtr<nvinfer1::IExecutionContext> mContext;

    std::vector<UniqPtr<char, CuMemDeleter>> mBindings;
    std::vector<void *>                      mBindingArray; // same content as bindings
    std::unique_ptr<Logger>                  mLogger;
    std::string                              mEnginePath;
    int                                      mDlaCore;

    // cudaGraph objects
    cudaGraph_t     mGraph{};
    cudaGraphExec_t mGraphExec{};
    bool            mCudaGraphEnabled;
    //!
    //! Report Args for visilization
    //!
    void ReportArgs();

    //!
    //! init Random input for the network which will be uesd for inference
    //!
    void initRandomInput(int bindingNum);
    // cudaGraph related
    void CudaGraphEndCapture(cudaStream_t stream);

    void CudaGraphBeginCapture(cudaStream_t stream);

    bool CudaGraphLaunch(cudaStream_t stream);

    bool enableCudaGraph();
};
