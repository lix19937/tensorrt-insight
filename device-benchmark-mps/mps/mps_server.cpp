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

#include <commandLine.h>
#include <stdio.h>
#include <sys/types.h>
#include <tools.h>
#include <unistd.h>

#include <cuda.h>
#include <iostream>
#include <map>
#include <mps_utils.h>
#include <sharedCtx.h>
#include <string>
#include <zmq_u.hpp>

inline void timeDiff(const timespec &start, const timespec &end, timespec *result)
{
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        result->tv_sec  = end.tv_sec - start.tv_sec - 1;
        result->tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        result->tv_sec  = end.tv_sec - start.tv_sec;
        result->tv_nsec = end.tv_nsec - start.tv_nsec;
    }
}

//!
//! help function for time benchmark
//!
inline timespec timeDiff(const timespec &start, const timespec &end)
{
    timespec result;
    timeDiff(start, end, &result);
    return result;
}

struct pid_and_perf
{
    pid_t  pid;
    int    taskID;
    double perf;
};

bool print_help()
{
    printf(
        "--------------------------------------------------------------------------------------------------------\n");
    printf("---------------------------- multi_process_server ---------------------------------------------\n");
    printf(" '--help': print help information \n");
    printf(" '--mps' : enable mps, otherwise will use multi-cuda-context  \n");
    return true;
}

int main(int argc, char **argv)
{
    char *CUDA_DEVICE_MAX_CONNECTIONS = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    printf("CUDA_DEVICE_MAX_CONNECTIONS=%s\n", CUDA_DEVICE_MAX_CONNECTIONS);
    if (CUDA_DEVICE_MAX_CONNECTIONS == NULL || 0 == strlen(CUDA_DEVICE_MAX_CONNECTIONS))
    {
        int ret = setenv("CUDA_DEVICE_MAX_CONNECTIONS", "4", 1);
        printf("set CUDA_DEVICE_MAX_CONNECTIONS to %d ret = %d\n", 4, ret);
    }
    commandLine cmdLine(argc, argv);
    if (cmdLine.GetFlag("help"))
    {
        print_help();
        return 0;
    }
    bool use_mps = cmdLine.GetFlag("mps");

    std::cout << "multi process server" << std::endl;
    zmq::context_t context;
    zmq::socket_t  socket(context, zmq::socket_type::rep);
    socket.bind("tcp://0.0.0.0:9999");

    checkCudaErrorsDRV(cuInit(0));
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    MPS_resource mps_resource;
    CUcontext    cuda_context;

    // Create cuda-contexts
    if (!use_mps)
    { // no mps
        // Create primary contexts
        checkCudaErrors(cudaSetDevice(0));
        unsigned int flags;
        checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        // verify if SetFlag is success
        checkCudaErrors(cudaGetDeviceFlags(&flags));
        printf("cuda Device flags: %d\n", flags);

    }
    else
    { // mps
        std::cout << "mps was enabled...." << std::endl;
        if (!init_mps_server(cuda_context, mps_resource))
        {
            printf("init mps server failed!");
            return -1;
        }
    }

    pid_and_perf              pf;
    std::vector<pid_and_perf> pid_perf_v;

    timespec timeBegin;
    timespec timeNow;
    clock_gettime(CLOCK_REALTIME, &timeBegin);
    double seconds;
    double total_perf;

    while (true)
    {
        zmq::message_t request_message;
        socket.recv(&request_message);

        // judge if the message is the device_key
        if (request_message.size() == sizeof(CUetblSharedCtx_deviceKey))
        { // recive mps device key
            // if not use mps, but recive this message, we must
            if (!use_mps)
            {
                zmq::message_t reply_message("server mps disabled!", 20);
                socket.send(reply_message);
                continue;
            }
            // send shareKey to client to help client start sub-context
            send_shareKey(socket, request_message, mps_resource, cuda_context);
            continue;
        }
        memcpy(&pf, (void *)request_message.data(), sizeof(pid_and_perf));
        zmq::message_t reply_message("message from server!", 20);
        socket.send(reply_message);

        pid_perf_v.push_back(pf);

        timespec timeElapsed;
        clock_gettime(CLOCK_REALTIME, &timeNow);
        timeElapsed = timeDiff(timeBegin, timeNow);

        seconds = timeElapsed.tv_sec + double(timeElapsed.tv_nsec) * double(1e-9);

        if (seconds > 1.0)
        {
            total_perf = 0;

            for (auto it = pid_perf_v.begin(); it != pid_perf_v.end(); it++)
            {
                total_perf += it->perf;
                printf("pid:%d( taskID: %d,  %f img/sec) ,", it->pid, it->taskID, it->perf);
            }

            printf("  total perf: %f\n", total_perf);
            timeBegin = timeNow;
            pid_perf_v.clear();
        }
    }

    return 0;
}
