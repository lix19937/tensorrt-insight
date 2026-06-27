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

在标准decode 阶段     
每次只处理一个新token，序列长度固定是1。输入形状固定，意味着计算图结构固定——每次执行的kernel序列、每个kernel的维度，都一样。    
这刚好是CUDA graph的适用条件：图结构不变，只有数据在换。    
而vLLM的具体做法：提前对几个固定batch size（1、2、4、8、16……）分别capture CUDA graph，运行时根据实际batch size选对应的graph来replay。实际batch size不在列表里，就padding到最近的那个。
这个方案在decode阶段跑得很好。     

prefill阶段为什么难呢      
这里先说一下，prefill是处理输入prompt的过程——你输入”帮我写一首诗”，模型先把这几个字全部过一遍，才开始生成。而prefill和decode最根本的区别是什么呢？ 那就是，输入序列长度是可变的。用户A输入5个token，用户B输入500个，用户C输入5000个——不可预测，不固定。序列长度不固定，计算图就不固定。
注意力计算的输入维度变了，中间激活值的shape变了，kernel配置变了——这不是”数据在变，图不变”，是”图本身在变”。
CUDA graph要求capture时和replay时计算图结构完全一致。图都在变，没办法复用。      

decode阶段用padding解决了batch size不固定的问题——提前capture几个固定shape的graph，用的时候选最近的。那么，prefill阶段，能不能用类似思路，提前capture几个固定序列长度的graph，实际用的时候padding到最近的长度？


问题1：序列长度的分布太分散，padding浪费太大。    

问题2：prefill阶段的计算特性，导致CUDA graph的收益本来就有限。

问题3：连续批处理带来额外动态性    
vLLM用连续批处理，不同请求的prefill被动态打包——这一批三个请求，下一批五个，每次组合不一样。这种动态性让prefill阶段的输入形状变化更不规律，进一步降低了CUDA graph的适用性。

可行的方案：      
方案一：Chunked Prefill + CUDA graph

把长的prefill序列切成固定长度的chunk，每个chunk序列长度固定，就可以capture CUDA graph了。

这是目前最主流的思路。vLLM已经支持了Chunked Prefill，还没有在这个基础上完整接入CUDA graph，但方向是明确的。

工程上还有几个坑要处理：chunk边界处KV Cache怎么管理，跨chunk的注意力计算怎么拼接，padding带来的额外计算怎么控制。技术上可行，工程上在推进。

方案二：提前capture几个固定序列长度的graph

类似decode阶段的做法，针对特定场景可以用。

如果你的应用序列长度分布比较集中——比如某个RAG应用context基本固定在1024左右——可以只capture那几个长度的graph，padding浪费可控，性能收益实在。通用场景下性价比不够高，垂直场景下是可行的工程选择。

方案三：torch.compile和fused kernel

用torch.compile或者自己写triton kernel，把多个操作fuse成一个，某种程度上替代CUDA graph——减少kernel launch次数。

TensorRT-LLM在这个方向走得比vLLM激进，prefill阶段的优化更深，但代价是灵活性大幅下降，支持的模型种类少，不是vLLM的设计哲学。

这里有个根本的权衡：

灵活性和极致性能，很难同时做到。

vLLM选了灵活性——支持几十种模型架构，各种量化方案，动态调度。代价是某些性能没做到极致。TensorRT-LLM选了极致性能——在NVIDIA自家硬件上把每块榨干，代价是灵活性大幅下降。

没有对错，是不同的设计哲学，服务不同的场景。


Chunked Prefill + CUDA graph


### 12 LLM prefill是计算密集型的 or 是访存密集型的？   
如果kernel launch overhead在总时间里占比小，所以CUDA graph带来的收益有限。
依据是 **Roofline**模型——区分一个操作是计算密集型还是访存密集型，看的是算术强度（Arithmetic Intensity），即每字节内存访问对应多少次浮点运算（FLOPs/Byte）。  

算术强度高 → 计算密集型，瓶颈在算力     
算术强度低 → 访存密集型，瓶颈在内存带宽  

decode阶段：     
每次只生成1个token，核心操作是矩阵-向量乘法——把权重矩阵（[d_model, d_model]）和一个向量（[1, d_model]）相乘。    
搬了这么大一个权重矩阵，只做了极少量的计算，算术强度极低。 A100上做FP16推理，decode阶段算术强度大概在1-10 FLOPs/Byte量级，远低于A100的算力/带宽比（约200 FLOPs/Byte）——妥妥的访存密集型。
这也是为什么decode阶段kernel launch overhead是真实瓶颈——GPU的计算单元大量空转，任何额外的overhead占比都会被放大。   

prefill阶段：    
一次处理N个token，核心操作是矩阵-矩阵乘法（GEMM）——输入[N, d_model]，权重[d_model, d_model]。
同样的权重矩阵从HBM搬一次，但这次要做N倍的计算——算术强度随N线性增长。当N足够大（几百到几千），算术强度可以轻松超过GPU的roofline临界点，进入计算密集区间。GPU的算力被充分利用，kernel launch overhead在总时间里占比就很小了。
这也是为什么FlashAttention在prefill阶段的加速效果显著——它的核心优化是减少HBM访问次数，这个优化在计算密集型场景下才特别有价值。                               

师兄的观点是：prefill阶段要把巨大的权重矩阵从显存搬到计算核心，这个搬运本身就是内存带宽受限的。

煮啵的回应是：decode阶段也要搬同样的权重矩阵，但decode每次只产生一个输出向量，算术强度极低；prefill一次处理N个token，同样的搬运成本被分摊到N次计算上，算术强度随N上升，当N足够大，就从访存密集型变成了计算密集型。

这两种说法描述的是同一个现象的不同侧面，不是非此即彼的关系——prefill阶段确实需要搬运权重，这是真的；但当序列很长的时候，计算量的增长速度超过了访存量的增长速度，瓶颈就从访存转移到了计算。

标准的参考是Google 2022年的论文Efficiently Scaling Transformer Inference（Pope et al.），里面有对prefill和decode阶段roofline分析的详细讨论，结论支持煮啵的判断——长序列prefill是compute-bound，decode是memory-bound。

但师兄说他看到的资料里有相反的描述，说prefill是memory-bound。

煮啵承认：如果序列很短，N很小，算术强度不够高，prefill也可能是访存密集型的。这个边界在哪里，取决于具体的序列长度和硬件参数。

所以这里有一个不够严谨的地方——之前煮啵说”GPU计算单元基本跑满了”，这个说法过于绝对。更准确的表述是：

当prefill序列足够长时，算术强度足够高，prefill进入计算密集区间，kernel launch overhead在总耗时里占比相对较小，CUDA graph能带来的收益因此有限。

这个核心逻辑没变，但多了一个”序列足够长”的前提条件。


###
