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

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <thread>

#include "Task.h"
#include "commandLine.h"
#include "customTask.h"
#include "delayKernel.h"

//!
//! help function for time benchmark
//!

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

//! parse trt engine path from command line,
//! user can specify the engine using the below arg :
//! --GPU=path_to_gpu.engine --DLA_0=path_to_dla0.engine --DLA_1=path_to_dla1.engine

bool parse_model_paths(commandLine &cmdLine, std::vector<std::string> &GPUPath, std::vector<std::string> &DLA0Path,
                       std::vector<std::string> &DLA1Path)
{
    GPUPath  = cmdLine.GetStringList("GPU");
    DLA0Path = cmdLine.GetStringList("DLA_0");
    DLA1Path = cmdLine.GetStringList("DLA_1");
    return true;
}

//! parse device flag, main function will call 'cudaSetDeviceFlags(flag);'
//! there are 4 kinds of flags we can use:
//! no specify(default) : -1, will not call cudaSetDeviceFlags() later;
//! --DeviceFlag=spin : cudaSetDeviceFlags(cudaDeviceScheduleSpin);
//! --DeviceFlag=block : cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
//! --DeviceFlag=yield : cudaSetDeviceFlags(cudaDeviceScheduleYield);
//! --DeviceFlag=auto : cudaSetDeviceFlags(cudaDeviceScheduleAuto);

int parse_device_flag(commandLine &cmdLine)
{
    int         flag        = -1;
    const char *device_flag = cmdLine.GetString("DeviceFlag");
    if (device_flag != NULL)
    {
        do
        {
            if (std::string(device_flag).find("spin") != std::string::npos)
            {
                flag = cudaDeviceScheduleSpin;
                printf("DeviceFlag : cudaDeviceScheduleSpin \n");
                break;
            }
            else if (std::string(device_flag).find("block") != std::string::npos)
            {
                flag = cudaDeviceScheduleBlockingSync;
                printf("DeviceFlag : cudaDeviceScheduleBlockingSync \n");
                break;
            }
            else if (std::string(device_flag).find("yield") != std::string::npos)
            {
                printf("DeviceFlag : cudaDeviceScheduleYield \n");
                flag = cudaDeviceScheduleYield;
                break;
            }
            else if (std::string(device_flag).find("auto") != std::string::npos)
            {
                printf("DeviceFlag : cudaDeviceScheduleAuto \n");
                flag = cudaDeviceScheduleYield;
                break;
            }
            else
            {
                printf("[Error] unrecognized DeviceFlag : '%s' ! enable : cudaDeviceScheduleSpin \n", device_flag);
                flag = -1;
            }
        } while (0);
    }
    return flag;
}

//! parse sync type from user's command line
//! --Synctype=eventDefault : cudaEventCreateWithFlags(&event, cudaEventDefault);
//! --Synctype=eventBlock : cudaEventCreateWithFlags(&event, cudaEventBlockingSync);
//! --Synctype=streamDefault : cudaStreamCreateWithPriority(&stream, cudaStreamDefault);
//! --Synctype=streamNonBlock : cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking);

SyncType parse_synctype(commandLine &cmdLine)
{
    const char *synctypeStr = cmdLine.GetString("Synctype");
    SyncType    synctype    = SyncType::EventDefault;
    if (synctypeStr != NULL)
    {
        do
        {
            if (std::string(synctypeStr).find("eventDefault") != std::string::npos)
            {
                synctype = SyncType::EventDefault;
                printf("SyncType : eventDefault \n");
                break;
            }
            else if (std::string(synctypeStr).find("eventBlock") != std::string::npos)
            {
                synctype = SyncType::EventBlock;
                printf("SyncType : eventBlock \n");
                break;
            }
            else if (std::string(synctypeStr).find("streamDefault") != std::string::npos)
            {
                synctype = SyncType::StreamDefault;
                printf("SyncType : streamDefault \n");
                break;
            }
            else if (std::string(synctypeStr).find("streamNonBlock") != std::string::npos)
            {
                synctype = SyncType::StreamNonBlock;
                printf("SyncType : streamNonBlock \n");
                break;
            }
        } while (0);
    }
    return synctype;
}

//! the thread entry of the gpu/dla task
void *process(void *param)
{
    if (!param)
    {
        printf("NULL thread parameter, exiting thread\n");
        return NULL;
    }
    Task *task = (Task *)param;
    printf(" %s task thread started \n", task->getTaskType());
    while (1)
    {
        if (!task->Run())
            printf("task failed to Run\n");
    }
    printf("exiting  thread\n");
}

bool print_help()
{
    printf(
        "--------------------------------------------------------------------------------------------------------\n");
    printf(
        "----------------------------------------- Device Benchmark ---------------------------------------------\n");
    printf(" '--help': print help information \n");
    printf(" '--GPU=trtGPU0.engine,trtGPU1.engine' Load gpu-trt-engines to GPU, split by `,` \n");
    printf(" '--DLA_0=trtDLA0.engine,trtDLA1.engine' Load dla-trt-engines to DLA0, split by `,` \n");
    printf(" '--DLA_1=trtDLA1.engine,trtDLA2.engine' Load dla-trt-engines to DLA1, split by `,` \n");
    printf(" '--delayKernel=N' launch a kernel that execute for N ms. split by `,` \n");
    printf(" '--custom\n");
    printf("     Specify custom kernel, split by `,`\n");
    printf("     Currently support: CudaKernelTask, D2DCopy, H2DCopy, D2HCopy \n");
    printf("     You can also implement your custom kernel, please refer to src/customtasks/cudaKernelTask \n");
    printf(
        "# belows are the sync options, If you do not care about the evaluate the impact of synchronization behavior, "
        "Just do not set it\n");
    printf(" '--DeviceFlag\n");
    printf("     default(no flag): 'cudaSetDeviceFlags(cudaDeviceScheduleSpin)' \n");
    printf("     'spin': 'cudaSetDeviceFlags(cudaDeviceScheduleSpin)' \n");
    printf("     'block': 'cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)' \n");
    printf("     'yield': 'cudaSetDeviceFlags(cudaDeviceScheduleYield)' \n");
    printf("     'auto': 'cudaSetDeviceFlags(cudaDeviceScheduleAuto)' \n");
    printf(" '--Synctype\n");
    printf("     no flag: sync with cudaEvent('cudaEventCreateWithFlags(&event, cudaEventDefault))' \n");
    printf("     'eventDefault': sync with cudaEvent('cudaEventCreateWithFlags(&event, cudaEventDefault))' \n");
    printf("     'eventBlock': sync with cudaEvent('cudaEventCreateWithFlags(&event, cudaEventBlockingSync))' \n");
    printf("     'streamDefault': sync with cudaStream('cudaStreamCreateWithPriority(&stream, cudaStreamDefault))' \n");
    printf(
        "     'streamNonBlock': sync with cudaStream('cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking))' "
        "\n");
    printf(" '--enableCudaGraphForGpuEngines\n");
    printf("     Specify this if you want for enable CUDA graph for TRT-GPU inference. \n");
    printf("     CUDA graph is not supported on DLA currently. \n");
    printf(" '--duration\n");
    printf("     Specify benchmark duration in seconds, default is 15s. -1 means forever. \n");
    printf("-------------------------------------------------------------------------------------------------------\n");
    return true;
}

int main(int argc, char **argv)
{
    char *CUDA_DEVICE_MAX_CONNECTIONS = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    printf("CUDA_DEVICE_MAX_CONNECTIONS=%s\n", CUDA_DEVICE_MAX_CONNECTIONS);

    commandLine cmdLine(argc, argv);
    //! parse device_flag, see parse_device_flag
    if (cmdLine.GetFlag("help"))
    {
        print_help();
        return 0;
    }
    int device_flag = parse_device_flag(cmdLine);
    if (device_flag != -1)
        cudaSetDeviceFlags(device_flag);
    cudaFree(0);

    //! parse sync type
    SyncType syncType = parse_synctype(cmdLine);

    std::vector<std::string> gpu_model_path;
    std::vector<std::string> dla0_model_path;
    std::vector<std::string> dla1_model_path;
    //! parse trt engine path
    parse_model_paths(cmdLine, gpu_model_path, dla0_model_path, dla1_model_path);

    std::vector<std::thread> threads;
    std::vector<Task *>      tasks;

    //! gpu task
    bool isEnableCudaGraphForGpuEngines = cmdLine.GetFlag("enableCudaGraphForGpuEngines");
    if (!gpu_model_path.empty())
    {
        for (size_t i = 0; i < gpu_model_path.size(); i++)
        {
            printf("add gpu task: %s \n", gpu_model_path[i].c_str());
            TrtBenchTask *ptask_gpu =
                new TrtBenchTask(gpu_model_path[i].c_str(), /*DLACore*/ -1, syncType, isEnableCudaGraphForGpuEngines);
            threads.push_back(std::thread(process, (void *)(ptask_gpu)));
            tasks.push_back(ptask_gpu);
        }
    }
    //! dla0 task
    if (!dla0_model_path.empty())
    {
        for (size_t i = 0; i < dla0_model_path.size(); i++)
        {
            printf("add dla0 task:  %s \n", dla0_model_path[i].c_str());
            TrtBenchTask *ptask_dla0 =
                new TrtBenchTask(dla0_model_path[i].c_str(), /*DLACore*/ 0, syncType, /*enableCudaGraph*/ false);
            threads.push_back(std::thread(process, (void *)(ptask_dla0)));
            tasks.push_back(ptask_dla0);
        }
    }
    //! dla1 task
    if (!dla1_model_path.empty())
    {
        for (size_t i = 0; i < dla1_model_path.size(); i++)
        {
            printf("add dla1 task: %s \n", dla1_model_path[i].c_str());
            TrtBenchTask *ptask_dla1 =
                new TrtBenchTask(dla1_model_path[i].c_str(), /*DLACore*/ 1, syncType, /*enableCudaGraph*/ false);
            threads.push_back(std::thread(process, (void *)(ptask_dla1)));
            tasks.push_back(ptask_dla1);
        }
    }
    //! custom tasks
    //! parse custom tasks
    std::vector<std::string> customTasks;
    customTasks = cmdLine.GetStringList("custom");
    if (!customTasks.empty())
    {
        for (size_t i = 0; i < customTasks.size(); i++)
        {
            printf("add custom task: %s \n", customTasks[i].c_str());
            CustomTask *custom_task = CustomTask::CreateObject(customTasks[i].c_str(), syncType);
            threads.push_back(std::thread(process, (void *)(custom_task)));
            tasks.push_back(custom_task);
        }
    }

    std::vector<std::string> delayKernelTasks;
    delayKernelTasks = cmdLine.GetStringList("delayKernel");
    if (!delayKernelTasks.empty())
    {
        for (size_t i = 0; i < delayKernelTasks.size(); i++)
        {
            printf("add a kernel that delay %d millisecond\n", std::stoi(delayKernelTasks[i]));
            DelayKernelTask *task = new DelayKernelTask(std::stoi(delayKernelTasks[i]));
            threads.push_back(std::thread(process, (void *)(task)));
            tasks.push_back(task);
        }
    }

    timespec timeBegin;
    clock_gettime(CLOCK_REALTIME, &timeBegin);
    std::vector<int> last_processed_img(tasks.size());
    std::vector<int> processed_img_this_time(tasks.size());
    memset(processed_img_this_time.data(), 0, sizeof(int) * tasks.size());
    memset(last_processed_img.data(), 0, sizeof(int) * tasks.size());

    /*
     run benchmark for the duration time and print the benchmark log every 1s
    */
    int  duration = cmdLine.GetInt("duration", /*defaultValue*/ 15, /*allowOtherDelimiters*/ false);
    bool is_running_forever{false};
    if (duration == -1)
    {
        duration           = 1;
        is_running_forever = true;
    }
    int t = 0;
    while (t < duration)
    {
        if (!is_running_forever)
        {
            t = t + 1;
        }
        sleep(1);
        uint64_t totalImages = 0;
        timespec timeNow;
        clock_gettime(CLOCK_REALTIME, &timeNow);
        const timespec timeElapsed = timeDiff(timeBegin, timeNow);

        const double seconds = timeElapsed.tv_sec + double(timeElapsed.tv_nsec) * double(1e-9);

        for (size_t n = 0; n < tasks.size(); n++)
        {
            processed_img_this_time[n] = tasks[n]->imgProcessed() - last_processed_img[n];
            last_processed_img[n]      = tasks[n]->imgProcessed();
        }

        for (size_t n = 0; n < tasks.size(); n++)
            totalImages += processed_img_this_time[n];

        const double imagesPerSec = double(totalImages) / seconds;

        printf("total: %f img/sec  (", imagesPerSec);
        for (size_t n = 0; n < tasks.size(); n++)
        {
            printf("%s %f img/sec", tasks[n]->getTaskType(), double(processed_img_this_time[n]) / seconds);
            if (n < tasks.size() - 1)
                printf(", ");
        }
        printf(")\n");
        clock_gettime(CLOCK_REALTIME, &timeBegin);
    }
}