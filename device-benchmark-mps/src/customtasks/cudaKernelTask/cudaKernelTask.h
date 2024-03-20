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
#include <customTask.h>
// task for cuda kernel
#include "cuda.h"
#include "cuda_runtime.h"
#define MATRIXDIM 256
#define LOOPCOUNT 1

#define UNUSED(x) (void)(x)

void initMatrix(double **hptrA, double **hptrB, double **hptrC, CUdeviceptr *dptrA, CUdeviceptr *dptrB,
                CUdeviceptr *dptrC, int matrixDim);
void runMultiplyMatrix(CUdeviceptr dptrA, CUdeviceptr dptrB, CUdeviceptr dptrC, CUstream stream, int matrixDim,
                       int loopCount);
void FreeMatrix(double *hptrA, double *hptrB, double *hptrC, CUdeviceptr dptrA, CUdeviceptr dptrB, CUdeviceptr dptrC);

class CudaKernelTask : public CustomTask
{
    DECLARE_CLASS()
  public:
    CudaKernelTask() = delete;

    CudaKernelTask(std::string name, SyncType syncType)
    {
        UNUSED(name);
        // the same to all the task
        this->TaskInit(syncType);
        this->mTaskName = std::string("Cuda kernel task");
        ReportArgs();

        // the privite operation for the CudaKernel Task
        initMatrix(&hptrA, &hptrB, &hptrC, &dptrA, &dptrB, &dptrC, MATRIXDIM);
    }

    void ReportArgs()
    {
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "Cuda kernel Task... " << std::endl;
        // the same to all the task
        ReportSyncType();
        std::cout << "--------------------------------------------------------" << std::endl;
    }
    virtual bool Run()
    {
        // run cuda kernel here
        runMultiplyMatrix(dptrA, dptrB, dptrC, this->mStream.get(), MATRIXDIM, LOOPCOUNT);
        //
        this->Sync();
        mInference_count++;
        return true;
    }

    const char *getTaskType() { return "CudaKernelTask"; }

    ~CudaKernelTask() { FreeMatrix(hptrA, hptrB, hptrC, dptrA, dptrB, dptrC); }

  private:
    double *    hptrA;
    double *    hptrB;
    double *    hptrC;
    CUdeviceptr dptrA;
    CUdeviceptr dptrB;
    CUdeviceptr dptrC;
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                                            \
    {                                                                                                                  \
        cudaError_t error_code = callstr;                                                                              \
        if (error_code != cudaSuccess)                                                                                 \
        {                                                                                                              \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl;            \
            exit(0);                                                                                                   \
        }                                                                                                              \
    }
#endif

class H2DCopy : public CustomTask
{
    DECLARE_CLASS()
  public:
    H2DCopy() = delete;

    H2DCopy(std::string name, SyncType syncType)
    {
        UNUSED(name);
        this->TaskInit(syncType);
        this->mTaskName = "H2DCopy_Task";
        CUDA_CHECK(cudaMalloc((void **)&dptrA, byteSize));
        CUDA_CHECK(cudaMalloc((void **)&dptrB, byteSize));
        CUDA_CHECK(cudaHostAlloc((void **)&hptrA, byteSize, cudaHostAllocDefault));
        ReportArgs();
    }

    void ReportArgs()
    {
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "Cuda kernel Task... " << std::endl;
        // the same to all the task
        ReportSyncType();
        std::cout << "--------------------------------------------------------" << std::endl;
    }
    virtual bool Run()
    {
        unsigned int matrixSize = MATRIXDIM * MATRIXDIM;
        size_t       temp       = matrixSize * sizeof(double);
        CUDA_CHECK(cudaMemcpyAsync(dptrA, hptrA, temp, cudaMemcpyHostToDevice, mStream.get()));
        this->Sync();
        mInference_count++;
        return true;
    }

    const char *getTaskType() { return mTaskType.c_str(); }

  private:
    std::string mTaskType{"H2DCopy"};
    double *    hptrA;
    double *    dptrA;
    double *    dptrB;
    size_t      byteSize = 1 * MATRIXDIM * MATRIXDIM * sizeof(double);
};
IMPLEMENT_CLASS("H2DCopy", H2DCopy);

class D2HCopy : public CustomTask
{
    DECLARE_CLASS()
  public:
    D2HCopy() = delete;

    D2HCopy(std::string name, SyncType syncType)
    {
        UNUSED(name);
        this->TaskInit(syncType);
        this->mTaskName = "D2HCopy_Task";
        CUDA_CHECK(cudaMalloc((void **)&dptrA, byteSize));
        CUDA_CHECK(cudaMalloc((void **)&dptrB, byteSize));
        CUDA_CHECK(cudaHostAlloc((void **)&hptrA, byteSize, cudaHostAllocDefault));
        ReportArgs();
    }

    void ReportArgs()
    {
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "Cuda kernel Task... " << std::endl;
        // the same to all the task
        ReportSyncType();
        std::cout << "--------------------------------------------------------" << std::endl;
    }
    virtual bool Run()
    {
        CUDA_CHECK(cudaMemcpyAsync((void *)hptrA, (void *)dptrA, byteSize, cudaMemcpyDeviceToDevice, mStream.get()));
        this->Sync();
        mInference_count++;
        return true;
    }

    const char *getTaskType() { return mTaskType.c_str(); }

  private:
    std::string mTaskType{"D2HCopy"};
    double *    hptrA;
    double *    dptrA;
    double *    dptrB;
    size_t      byteSize = 1 * MATRIXDIM * MATRIXDIM * sizeof(double);
};
IMPLEMENT_CLASS("D2HCopy", D2HCopy);

class D2DCopy : public CustomTask
{
    DECLARE_CLASS()
  public:
    D2DCopy() = delete;

    D2DCopy(std::string name, SyncType syncType)
    {
        UNUSED(name);
        this->TaskInit(syncType);
        this->mTaskName = "D2DCopy_Task";
        CUDA_CHECK(cudaMalloc((void **)&dptrA, byteSize));
        CUDA_CHECK(cudaMalloc((void **)&dptrB, byteSize));
        CUDA_CHECK(cudaHostAlloc((void **)&hptrA, byteSize, cudaHostAllocDefault));
        ReportArgs();
    }

    void ReportArgs()
    {
        std::cout << "--------------------------------------------------------" << std::endl;
        std::cout << "Cuda kernel Task... " << std::endl;
        // the same to all the task
        ReportSyncType();
        std::cout << "--------------------------------------------------------" << std::endl;
    }
    virtual bool Run()
    {
        CUDA_CHECK(cudaMemcpyAsync(dptrB, dptrA, byteSize, cudaMemcpyDeviceToHost, mStream.get()));
        this->Sync();
        mInference_count++;
        return true;
    }

    const char *getTaskType() { return mTaskType.c_str(); }

  private:
    std::string mTaskType{"D2DCopy"};
    double *    hptrA;
    double *    dptrA;
    double *    dptrB;
    size_t      byteSize = 1 * MATRIXDIM * MATRIXDIM * sizeof(double);
};
IMPLEMENT_CLASS("D2DCopy", D2DCopy);
