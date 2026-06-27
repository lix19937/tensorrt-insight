### 0 kernel 执行过程   
https://developer.nvidia.com/blog/cuda-graphs/



### 1 带有历史帧信息的时序模型如何构建graph ?  

![1](./timing.jpg)   

![3](./captrue_timingv0.png)

![2](./captrue_timing.png)   

### 2 Tensor 的 Base Address（起始地址）保持不变，仅仅是 Shape 不同，会影响 CUDA Graph 的有效性  
会   

### 3 cudaGraph_t, cudaGraphExec_t 生命周期怎么管理      
cudaGraph_t 类型的对象定义了kernel graph的结构和内容；     
cudaGraphExec_t 类型的对象是一个“可执行的graph实例”：它可以以类似于单个内核的方式启动和执行。

### 4 graph 捕获和实例化过程中会执行kernel 吗     
不会执行     

### 5 graph 捕获过程中 有cpu 操作会出现什么情况            
cuda graph支持多个stream间的融合，而且不仅可以包含kernel执行，还可以包括在主机 CPU 上执行的函数和内存拷贝
```cuda
// https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu

    cudaStream_t stream1, stream2, stream3, streamForGraph;
    cudaEvent_t  forkStreamEvent, memsetEvent1, memsetEvent2;
    cudaGraph_t  graph;
    double       result_h = 0.0;

    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));
    checkCudaErrors(cudaStreamCreate(&stream3));
    checkCudaErrors(cudaStreamCreate(&streamForGraph));

    checkCudaErrors(cudaEventCreate(&forkStreamEvent));
    checkCudaErrors(cudaEventCreate(&memsetEvent1));
    checkCudaErrors(cudaEventCreate(&memsetEvent2));

//  cudaStreamCaptureModeThreadLocal  
    checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal)); // ++++++++++++++++++++  capture on s-1

    checkCudaErrors(cudaEventRecord(forkStreamEvent, stream1));
    checkCudaErrors(cudaStreamWaitEvent(stream2, forkStreamEvent, 0));
    checkCudaErrors(cudaStreamWaitEvent(stream3, forkStreamEvent, 0));

    checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1));
    checkCudaErrors(cudaMemsetAsync(outputVec_d, 0,         sizeof(double) * numOfBlocks,                 stream2));

    checkCudaErrors(cudaEventRecord(memsetEvent1, stream2));

    checkCudaErrors(cudaMemsetAsync(result_d, 0, sizeof(double), stream3));
    checkCudaErrors(cudaEventRecord(memsetEvent2, stream3));

    checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent1, 0)); // wait s-2   

    reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

    checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent2, 0));

    reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
    checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1));

    callBackData_t hostFnData = {0};
    hostFnData.data           = &result_h;
    hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
    cudaHostFn_t fn           = myHostNodeCallback;
    checkCudaErrors(cudaLaunchHostFunc(stream1, fn, &hostFnData));
    checkCudaErrors(cudaStreamEndCapture(stream1, &graph));

    cudaGraphNode_t *nodes    = NULL;
    size_t           numNodes = 0;
    checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
    printf("\nNum of nodes in the graph created using stream capture API = %zu\n", numNodes);

    cudaGraphExec_t graphExec;
    checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    cudaGraph_t     clonedGraph;
    cudaGraphExec_t clonedGraphExec;
    checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
    checkCudaErrors(cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
    }

    checkCudaErrors(cudaStreamSynchronize(streamForGraph));

    printf("Cloned Graph Output.. \n");
    for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
        checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
    }

    checkCudaErrors(cudaStreamSynchronize(streamForGraph));

    checkCudaErrors(cudaGraphExecDestroy(graphExec));
    checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaGraphDestroy(clonedGraph));
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));
    checkCudaErrors(cudaStreamDestroy(streamForGraph));
```
### 6 多流捕获与依赖管理   
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cross-stream-dependencies-and-events

### 7 cudagraph 在哪些场景/模型下提升性能显著     
CUDA Graphs 对效率的提升，但更复杂的计算逻辑提供了更多优化提升的空间

### 8 graph 可以跨越多 GPU吗， 通过什么机制      
可以   

### 9 创建graph 时候，一般设置哪个flag    

### 10 cudagraph 是怎么加速的   
CUDA Graphs 将整个计算流程定义为一个图而不是单个操作的列表。 最后通过提供一种由单个 CPU 操作来启动图上的多个 GPU 操作的方式减少kernel提交启动开销，进而解决上述问题。

### 11 vLLM /edge-llm 中 pd 推理中都使用了cudagraph 吗？ 为什么prefill阶段没有使用cudagraph      
https://docs.vllm.ai/en/latest/design/cuda_graphs/   

###
