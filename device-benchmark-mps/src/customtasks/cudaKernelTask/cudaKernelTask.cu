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

#include <time.h>

#include "cudaKernelTask.h"

#ifndef checkCudaErrorsKernel
#define checkCudaErrorsKernel(err) __checkCudaErrors(err, __FILE__, __LINE__)
// TODO: Do we need replace all the drv api to cudaruntime api?

inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        const char *errorStr = NULL;
        cuGetErrorString(err, &errorStr);
        fprintf(stderr,
                "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
                "line %i.\n",
                err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

__global__ void matrixMultiplyKernel(double *dataC, double *dataA, double *dataB, unsigned int matrixDim) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    double value = 0.0f;
    if (i > matrixDim) {
        i = i % matrixDim;
    }

    if (j > matrixDim) {
        j = j % matrixDim;
    }

    for (unsigned int k = 0; k < matrixDim; ++k) {
        value += dataA[k + i * matrixDim] * dataB[j + k * matrixDim];
    }
    dataC[j + i * matrixDim] = value;
}

void fillMatrixWithRandomValues(double *matrix, unsigned int matrixDim) {
    time_t t;
    unsigned int i, j;

    srand((unsigned int)time(&t));

    for (i = 0; i < matrixDim; ++i) {
        for (j = 0; j < matrixDim; ++j) {
            matrix[j + i * matrixDim] = (float)(rand() & 0xFF) / 10.0f;
            ;
        }
    }
}

void initMatrix(double **hptrA, double **hptrB, double **hptrC, CUdeviceptr *dptrA, CUdeviceptr *dptrB,
                CUdeviceptr *dptrC, int matrixDim) {
    unsigned int matrixSize = matrixDim * matrixDim;
    size_t byteSize = matrixSize * sizeof(double);

    checkCudaErrorsKernel(cuMemAlloc(dptrA, byteSize));
    checkCudaErrorsKernel(cuMemAlloc(dptrB, byteSize));
    checkCudaErrorsKernel(cuMemAlloc(dptrC, byteSize));
    checkCudaErrorsKernel(cuMemsetD8(*dptrC, 0, byteSize));

    *hptrA = new double[matrixSize];
    *hptrB = new double[matrixSize];
    *hptrC = new double[matrixSize];

    fillMatrixWithRandomValues(*hptrA, matrixDim);
    fillMatrixWithRandomValues(*hptrB, matrixDim);

    checkCudaErrorsKernel(cuMemcpyHtoD(*dptrA, *hptrA, byteSize));
    checkCudaErrorsKernel(cuMemcpyHtoD(*dptrB, *hptrB, byteSize));
}

void runMultiplyMatrix(CUdeviceptr dptrA, CUdeviceptr dptrB, CUdeviceptr dptrC, CUstream stream, int matrixDim,
                       int loopCount) {
    dim3 threads(32, 32);
    dim3 grid(matrixDim / threads.x, matrixDim / threads.y);

    for (int i = 0; i < loopCount; i++) {
        matrixMultiplyKernel<<<grid, threads, 0, stream>>>(reinterpret_cast<double *>(dptrC),
                                                           reinterpret_cast<double *>(dptrA),
                                                           reinterpret_cast<double *>(dptrB), matrixDim);
    }
}

void FreeMatrix(double *hptrA, double *hptrB, double *hptrC, CUdeviceptr dptrA, CUdeviceptr dptrB, CUdeviceptr dptrC) {
    checkCudaErrorsKernel(cuMemFree(dptrA));
    checkCudaErrorsKernel(cuMemFree(dptrB));
    checkCudaErrorsKernel(cuMemFree(dptrC));
    free(hptrA);
    free(hptrB);
    free(hptrC);
    return;
}