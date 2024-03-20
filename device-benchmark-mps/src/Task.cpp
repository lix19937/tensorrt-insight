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

#include <Task.h>

#include <cassert>

void Task::ReportSyncType()
{
    std::cout << "Sync Type:";
    switch (mSyncType)
    {
    case SyncType::StreamDefault:
        std::cout << "cudaStreamDefault" << std::endl;
        break;
    case SyncType::StreamNonBlock:
        std::cout << "cudaStreamNonBlocking" << std::endl;
        break;
    case SyncType::EventDefault:
        std::cout << "cudaEventDefault " << std::endl;
        break;
    case SyncType::EventBlock:
        std::cout << "cudaEventBlockingSync " << std::endl;
        break;
    default:
        assert(false && "INVALID SYNC TYPE");
    }
}
void Task::TaskInit(SyncType syncType)
{
    this->mInference_count      = 0;
    this->mTotal_inference_time = 0;
    this->mSyncType             = syncType;
    switch (syncType)
    {
    case SyncType::StreamDefault:
        this->mStream = makeCudaStream(cudaStreamDefault, /*priority*/ 0);
        this->mEvent = makeCudaEvent(cudaEventDefault);
        break;
    case SyncType::StreamNonBlock:
        this->mStream = makeCudaStream(cudaStreamNonBlocking, /*priority*/ 0);
        this->mEvent = makeCudaEvent(cudaEventDefault);
        break;
    case SyncType::EventDefault:
        this->mEvent = makeCudaEvent(cudaEventDefault);
        this->mStream = makeCudaStream(cudaStreamDefault, /*priority*/ 0);
        break;
    case SyncType::EventBlock:
        this->mEvent = makeCudaEvent(cudaEventBlockingSync);
        this->mStream = makeCudaStream(cudaStreamDefault, /*priority*/ 0);
        break;
    default:
        assert(false && "INVALID SYNC TYPE");
    }
}

void Task::Sync()
{
    /*
        Call synchronize depends on user's flag
    */
    if (mSyncType == SyncType::StreamDefault || mSyncType == SyncType::StreamNonBlock)
    {
        if (cudaSuccess != cudaStreamSynchronize(mStream.get()))
        {
            std::cout << "cudaStreamSynchronize failed" << std::endl;
        }
    }
    else if (mSyncType == SyncType::EventDefault || mSyncType == SyncType::EventBlock)
    {
        cudaEventRecord(mEvent.get(), mStream.get());
        if (cudaSuccess != cudaEventSynchronize(mEvent.get()))
        {
            std::cout << "cudaEventSynchronize failed" << std::endl;
        }
    }
    else
    {
        assert(false && "INVALID SYNC TYPE");
    }
    return;
}
/*
    the detail function & param introduction can be found in Task.h
*/
TrtBenchTask::TrtBenchTask(const char *engineFilePath, int DLACore, SyncType syncType, bool enableCudaGraph)
{
    // always call this first for all sub-class of Task
    this->TaskInit(syncType);

    /*
        init trt objects
    */
    this->mEnginePath = engineFilePath;
    Logger mLoggern;
    initLibNvInferPlugins(&mLoggern, "");
    this->mRuntime          = UniqPtr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(mLoggern)};
    this->mCudaGraphEnabled = enableCudaGraph && DLACore == -1; // DLA not support cudagraph now
    /*
        set DLA core for trt runtime
    */
    this->mDlaCore = DLACore;
    switch (DLACore)
    {
    case 0:
    case 1:
        this->mRuntime->setDLACore(DLACore);
        this->mTaskName = std::string("DLA") + std::to_string(DLACore);
        break;
    case -1:
        this->mTaskName = std::string("GPU");
        break;
    default:
        std::cout << "[ERROR]" << __FILE__ << __LINE__ << " DLACore==" << DLACore << "should not be here!" << std::endl;
        break;
    }

    /*
        load engine from file
    */
    std::ifstream     fin(engineFilePath, std::ios::binary);
    std::vector<char> inBuffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    fin.close();
    mEngine.reset(mRuntime->deserializeCudaEngine(inBuffer.data(), inBuffer.size()));
    mContext.reset(mEngine->createExecutionContext());

    /*
       1. malloc cuda memory for binding(in TensorRT, a binding is a network input or output, we can use
       bindingIsInput(i) to test if the i-th binding is input)
       2. fill random data in input-buffers
    */
    const int nbBindings = this->mEngine->getNbBindings();
    for (int i = 0; i < nbBindings; i++)
    {
        const auto dataType = this->mEngine->getBindingDataType(i);
        const int  elemSize = [&]() -> int {
            switch (dataType)
            {
            case nvinfer1::DataType::kFLOAT:
                return 4;
            case nvinfer1::DataType::kHALF:
                return 2;
            case nvinfer1::DataType::kINT8:
                return 1;
            case nvinfer1::DataType::kINT32:
                return 4;
            case nvinfer1::DataType::kBOOL:
                return 1;
            default:
                throw std::runtime_error("invalid data type");
            }
        }();
        const auto dims = this->mEngine->getBindingDimensions(i);

        const int bindingSize = elemSize * std::accumulate(dims.d, &dims.d[dims.nbDims], 1, std::multiplies<int>{});
        this->mBindings.emplace_back(mallocCudaMem<char>(bindingSize));
        this->mBindingArray.emplace_back(mBindings.back().get());
        /*
            fill random data in input-buffers
        */
        if (mEngine->bindingIsInput(i))
            initRandomInput(i);
    }
    if (this->mCudaGraphEnabled && this->mDlaCore == -1)
    {
        this->enableCudaGraph();
    }
    ReportArgs();
    return;
}
bool TrtBenchTask::enableCudaGraph()
{
    cudaStreamSynchronize(this->mStream.get());
    CudaGraphBeginCapture(this->mStream.get());
    // this->mCudaGraphEnabled = false;
    if (!this->mContext->enqueueV2(mBindingArray.data(), mStream.get(), NULL))
    {
        printf("ERROR!\n");
    }
    CudaGraphEndCapture(this->mStream.get());
    // this->mCudaGraphEnabled = true;
    return true;
}
bool TrtBenchTask::CudaGraphLaunch(cudaStream_t stream) { return cudaGraphLaunch(mGraphExec, stream) == cudaSuccess; }
void TrtBenchTask::CudaGraphBeginCapture(cudaStream_t stream)
{
    // blank enqueue
    if (!this->mContext->enqueueV2(mBindingArray.data(), mStream.get(), NULL))
    {
        printf("ERROR!\n");
    }
    checkCudaErrors(cudaGraphCreate(&mGraph, 0));
    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}
void TrtBenchTask::CudaGraphEndCapture(cudaStream_t stream)
{
    checkCudaErrors(cudaStreamEndCapture(stream, &mGraph));
    checkCudaErrors(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
    // checkCudaErrors(cudaGraphDestroy(mGraph));
}

void TrtBenchTask::ReportArgs()
{
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Engine Path: " << mEnginePath << std::endl;
    std::cout << "Task Type :" << getTaskType() << std::endl;
    const int nbBindings = mEngine->getNbBindings();
    for (int i = 0; i < nbBindings; i++)
    {
        const auto dims = mEngine->getBindingDimensions(i);
        if (mEngine->bindingIsInput(i))
            std::cout << "input : " << mEngine->getBindingName(i);
        else
            std::cout << "output : " << mEngine->getBindingName(i);
        std::cout << " , shape : [ ";
        for (int j = 0; j < dims.nbDims; j++)
            std::cout << dims.d[j] << ",";
        std::cout << "]" << std::endl;
    }
    // always call this
    ReportSyncType();
    std::cout << "--------------------------------------------------------" << std::endl;
}

bool TrtBenchTask::Run()
{
    /*
        In tensorRT, enqueueV2 will run one-time inference
    */
    if (!this->mCudaGraphEnabled)
    {
        if (!mContext->enqueueV2(mBindingArray.data(), mStream.get(), NULL))
        {
            std::cout << "failed to enqueue TensorRT context on device " << mDlaCore << std::endl;
            return false;
        }
    }
    else
    {
        if (!this->CudaGraphLaunch(this->mStream.get()))
        {
            std::cout << "failed to enqueue TensorRT context with cudaGraph on device " << mDlaCore << std::endl;
            return false;
        }
    }
    this->Sync();
    mInference_count++;
    return true;
}

const char *TrtBenchTask::getTaskType()
{
    switch (mDlaCore)
    {
    case -1:
        return "GPU";
        break;
    case 0:
        return "DLA0";
        break;
    case 1:
        return "DLA1";
        break;
    default:
        return NULL;
        break;
    }
}

/*
    helper function which fill random data to the binding-th input buffer
*/
void TrtBenchTask::initRandomInput(int binding)
{
    const auto dims        = this->mEngine->getBindingDimensions(binding);
    int        bindingSize = std::accumulate(dims.d, &dims.d[dims.nbDims], 1, std::multiplies<int>{});

    std::cout << "init random input for : " << mEngine->getBindingName(binding) << ", buffer size = " << bindingSize
              << std::endl;
    std::vector<float>                    h_buffer(bindingSize);
    std::default_random_engine            engine;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    auto generator = [&engine, &distribution]() { return static_cast<float>(distribution(engine)); };
    std::generate(h_buffer.data(), h_buffer.data() + bindingSize, generator);
    checkCudaErrors(cudaMemcpy(this->mBindings[binding].get(), h_buffer.data(), sizeof(float) * bindingSize,
                               cudaMemcpyHostToDevice));
    return;
}
