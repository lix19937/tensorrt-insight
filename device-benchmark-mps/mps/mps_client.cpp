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
#include <commandLine.h>
#include <cuda.h>
#include <customTask.h>
#include <iostream>
#include <mps_utils.h>
#include <sharedCtx.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <zmq_u.hpp>

//!
//! help function for time benchmark
//!
struct pid_and_perf
{
    pid_t  pid;
    int    taskID;
    double perf;
};

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

//! parse custom task from command line,
//! user can specify the engine using the below arg :
//! --custom=task1,task2,task3
bool parse_custom_task(commandLine &cmdLine, std::vector<std::string> &customTasks)
{
    customTasks = cmdLine.GetStringList("custom");
    return;
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

struct threadParam
{
    Task *     task;
};

//! the thread entry
void *process(void *param)
{
    if (!param)
    {
        printf("NULL thread parameter, exiting thread\n");
        return NULL;
    }
    threadParam *param_t = (threadParam *)param;
    Task *       task    = param_t->task;
    // call cuda api in sub thread which Retain the created primary cuda context
    checkCudaErrors(cudaSetDevice(0));

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
        "---------------------------- TensorRT benchmark for jetson ---------------------------------------------\n");
    printf(" '--help': print help information \n");
    printf(" '--GPU=trtGPU0.engine,trtGPU1.engine' Load gpu-trt-engines to GPU, split by `,` \n");
    printf(" '--DLA_0=trtDLA0.engine,trtDLA1.engine' Load dla-trt-engines to DLA0, split by `,` \n");
    printf(" '--DLA_1=trtDLA1.engine,trtDLA2.engine' Load dla-trt-engines to DLA1, split by `,` \n");
    printf(" '--custom=cudaKernelTask,customTask1' Start user defined custom tasks, split by `,`\n");
    printf(" '--mps' : enable mps, otherwise will use multi-cuda-context  \n");
    printf(
        "# belows are the sync options, If you do not care about the evaluate the impact of synchronization behavior, "
        "Just do not set it\n");
    printf(" '--Synctype\n");
    printf("     no flag: sync with cudaEvent('cudaEventCreateWithFlags(&event, cudaEventDefault))' \n");
    printf("     'eventDefault': sync with cudaEvent('cudaEventCreateWithFlags(&event, cudaEventDefault))' \n");
    printf("     'eventBlock': sync with cudaEvent('cudaEventCreateWithFlags(&event, cudaEventBlockingSync))' \n");
    printf("     'streamDefault': sync with cudaStream('cudaStreamCreateWithPriority(&stream, cudaStreamDefault))' \n");
    printf(
        "     'streamNonBlock': sync with cudaStream('cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking))' "
        "\n");
    printf("-------------------------------------------------------------------------------------------------------\n");
    return true;
}

int main(int argc, char **argv)
{
    char *CUDA_DEVICE_MAX_CONNECTIONS = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    printf("CUDA_DEVICE_MAX_CONNECTIONS=%s\n", CUDA_DEVICE_MAX_CONNECTIONS);
    if (CUDA_DEVICE_MAX_CONNECTIONS == NULL || 0 == strlen(CUDA_DEVICE_MAX_CONNECTIONS))
    {
        int ret = setenv("CUDA_DEVICE_MAX_CONNECTIONS", "12", 1);
        printf("set CUDA_DEVICE_MAX_CONNECTIONS to %d ret = %d\n", 12, ret);
    }
    /* we use help class commandLine parse user's cmd :
       sync-type (see: parse_synctype)
       model-path (see: parse_model_paths)
   */
    commandLine cmdLine(argc, argv);
    if (cmdLine.GetFlag("help"))
    {
        print_help();
        return 0;
    }
    SyncType syncType   = parse_synctype(cmdLine);
    bool     use_mps    = cmdLine.GetFlag("mps");
    pid_t    process_id = process_id = getpid();
    //! parse sync type
    SyncType                 synctype = parse_synctype(cmdLine);
    std::vector<std::string> gpu_model_path;
    std::vector<std::string> dla0_model_path;
    std::vector<std::string> dla1_model_path;
    //! parse trt engine path
    parse_model_paths(cmdLine, gpu_model_path, dla0_model_path, dla1_model_path);
    //! parse custom tasks
    std::vector<std::string> customTasks;
    parse_custom_task(cmdLine, customTasks);
    printf("client starting, searching for server...\n");
    printf("The client process id: %d\n", process_id);

    // zmq socket for message send&recv between processes
    zmq::context_t context;
    zmq::socket_t  socket(context, zmq::socket_type::req);
    socket.connect("tcp://0.0.0.0:9999");

    // Create cuda contexts
    if (!use_mps)
    { // no mps
        // call a cudaruntime api, Create cuda context
        cudaSetDevice(0);
        unsigned int flags;
        checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        // verify if SetFlag is success
        checkCudaErrors(cudaGetDeviceFlags(&flags));
        printf("cuda Device flags: %d\n", flags);

    }
    else
    { // mps
        std::cout << "mps was enabled...." << std::endl;
        init_mps_client(socket);
    }
    /*
        start sub-threads run GPU,DLA0,DLA1 tasks,
        first, init TrtBenchTask objects with the flag which provided by the user
        second, init sub-threads run TrtBenchTask::Run() forever
    */
    std::vector<std::thread> threads;
    std::vector<Task *>      tasks;
    //! gpu task
    bool isEnableCudaGraphForGpuEngines = false;
    if (!gpu_model_path.empty())
    {
        for (size_t i = 0; i < gpu_model_path.size(); i++)
        {
            printf("add gpu task: %s \n", gpu_model_path[i].c_str());
            TrtBenchTask *ptask_gpu =
                new TrtBenchTask(gpu_model_path[i].c_str(), /*DLACore*/ -1, syncType, isEnableCudaGraphForGpuEngines);
            threadParam param{ptask_gpu};
            threads.push_back(std::thread(process, (void *)(&param)));
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
            threadParam param{ptask_dla0};
            threads.push_back(std::thread(process, (void *)(&param)));
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
    // custom tasks
    if (!customTasks.empty())
    {
        for (size_t i = 0; i < customTasks.size(); i++)
        {
            printf("add custom task: %s \n", customTasks[i].c_str());
            CustomTask *custom_task = CustomTask::CreateObject(customTasks[i].c_str(), syncType);
            threadParam param{custom_task};
            threads.push_back(std::thread(process, (void *)(&param)));
            tasks.push_back(custom_task);
        }
    }

    timespec timeBegin;
    clock_gettime(CLOCK_REALTIME, &timeBegin);
    std::vector<int> last_processed_img(tasks.size());
    std::vector<int> processed_img_this_time(tasks.size());
    memset(processed_img_this_time.data(), 0, sizeof(int) * tasks.size());
    memset(last_processed_img.data(), 0, sizeof(int) * tasks.size());

    /*
     run benchmark forever and print the benchmark log every 1s
    */
    while (1)
    {
        sleep(1);
        uint64_t totalImages = 0;
        timespec timeNow;
        clock_gettime(CLOCK_REALTIME, &timeNow);
        const timespec timeElapsed = timeDiff(timeBegin, timeNow);

        const double seconds = timeElapsed.tv_sec + double(timeElapsed.tv_nsec) * double(1e-9);

        for (int n = 0; n < tasks.size(); n++)
        {
            processed_img_this_time[n] = tasks[n]->imgProcessed() - last_processed_img[n];
            last_processed_img[n]      = tasks[n]->imgProcessed();
        }

        for (int n = 0; n < tasks.size(); n++)
            totalImages += processed_img_this_time[n];

        const double imagesPerSec = double(totalImages) / seconds;

        printf("total: %f img/sec  (", imagesPerSec);
        pid_and_perf pf;
        pf.pid = process_id;
        for (size_t n = 0; n < tasks.size(); n++)
        {
            printf("%s %f img/sec", tasks[n]->getTaskType(), double(processed_img_this_time[n]) / seconds);
            if (n < tasks.size() - 1)
                printf(", ");
            pf.perf   = double(processed_img_this_time[n]);
            pf.taskID = n;
            zmq::message_t client_message(&pf, sizeof(pid_and_perf));
            socket.send(client_message);
            zmq::message_t reply_message_from_server;
            socket.recv(&reply_message_from_server);
        }
        printf(")\n");
        fflush(stdout);
        clock_gettime(CLOCK_REALTIME, &timeBegin);
    }
}
