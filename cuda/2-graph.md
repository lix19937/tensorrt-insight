### 0 kernel 执行过程   
https://developer.nvidia.com/blog/cuda-graphs/



### 1 带有历史帧信息的时序模型如何构建graph ?  

<details>

<summary>时序图</summary>

![1](./timing.jpg)   

![3](./captrue_timingv0.png)

![2](./captrue_timing.png)   

</details>

### 2 Tensor 的 Base Address（起始地址）保持不变，仅仅是 Shape 不同，会影响 CUDA Graph 的有效性  
会   

### 3 cudaGraph_t, cudaGraphExec_t 生命周期怎么管理      
cudaGraph_t 类型的对象定义了kernel graph的结构和内容；     
cudaGraphExec_t 类型的对象是一个“可执行的graph实例”：它可以以类似于单个内核的方式启动和执行。

### 4 graph 捕获和实例化过程中会执行kernel 吗     
不会执行     

### 5 graph 捕获过程中 有cpu 操作会出现什么情况            
cuda graph支持多个stream间的融合，而且不仅可以包含kernel执行，还可以包括在主机 CPU 上执行的函数和内存拷贝     

<details>  
      
<summary>多流捕获</summary>     

```cpp   
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
</details>   

### 6 多流捕获与依赖管理   
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cross-stream-dependencies-and-events

### 7 cudagraph 在哪些场景/模型下提升性能显著     
CUDA Graphs 对效率的提升，但更复杂的计算逻辑提供了更多优化提升的空间

### 8 graph 可以跨越多 GPU吗， 通过什么机制      
可以   

### 9 创建graph 时候，一般设置哪个flag     
cudaStreamCaptureModeGlobal  vs  cudaStreamCaptureModeThreadLocal

<details>

<summary>graph 封装类</summary>   

```cpp
class TrtCudaGraph
{
public:
    explicit TrtCudaGraph() = default;

    TrtCudaGraph(const TrtCudaGraph&) = delete;

    TrtCudaGraph& operator=(const TrtCudaGraph&) = delete;

    TrtCudaGraph(TrtCudaGraph&&) = delete;

    TrtCudaGraph& operator=(TrtCudaGraph&&) = delete;

    ~TrtCudaGraph()
    {
        if (mGraphExec)
        {
            cudaGraphExecDestroy(mGraphExec);
        }
    }

    void beginCapture(TrtCudaStream& stream)
    {
        CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
    }

    bool launch(TrtCudaStream& stream)
    {
        return cudaGraphLaunch(mGraphExec, stream.get()) == cudaSuccess;
    }

    void endCapture(TrtCudaStream& stream)
    {
        CHECK(cudaStreamEndCapture(stream.get(), &mGraph));
        CHECK(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
        CHECK(cudaGraphDestroy(mGraph));
    }

    void endCaptureOnError(TrtCudaStream& stream)
    {
        // There are two possibilities why stream capture would fail:
        // (1) stream is in cudaErrorStreamCaptureInvalidated state.
        // (2) TRT reports a failure.
        // In case (1), the returning mGraph should be nullptr.
        // In case (2), the returning mGraph is not nullptr, but it should not be used.
        const auto ret = cudaStreamEndCapture(stream.get(), &mGraph);
        if (ret == cudaErrorStreamCaptureInvalidated)
        {
            assert(mGraph == nullptr);
        }
        else
        {
            CHECK(ret);
            assert(mGraph != nullptr);
            CHECK(cudaGraphDestroy(mGraph));
            mGraph = nullptr;
        }
        // Clean up any CUDA error.
        cudaGetLastError();
        sample::gLogWarning << "The CUDA graph capture on the stream has failed." << std::endl;
    }

private:
    cudaGraph_t mGraph{};
    cudaGraphExec_t mGraphExec{};
};

```

</details>

### 10 cudagraph 是怎么加速的   
CUDA Graphs 将整个计算流程定义为一个图而不是单个操作的列表。 最后通过提供一种由单个 CPU 操作来启动图上的多个 GPU 操作的方式减少kernel提交启动开销，进而解决上述问题。    
如果kernel launch overhead在总时间里占比小，所以CUDA graph带来的收益有限。    

### 11 vLLM /edge-llm 中 pd 推理中都使用了cudagraph 吗？ 为什么prefill阶段没有使用cudagraph      
https://docs.vllm.ai/en/latest/design/cuda_graphs/   

<details>

<summary>在标准decode 阶段</summary>
     
每次只处理一个新token，序列长度固定是1。输入形状固定，意味着计算图结构固定——每次执行的kernel序列、每个kernel的维度，都一样。    
这刚好是CUDA graph的适用条件：图结构不变，只有数据在换。    
而vLLM的具体做法：提前对几个固定batch size（1、2、4、8、16……）分别capture CUDA graph，运行时根据实际batch size选对应的graph来replay。实际batch size不在列表里，就padding到最近的那个。
这个方案在decode阶段跑得很好。     

</details>

<details>

<summary>prefill 阶段为什么不行呢</summary>
      
这里先说一下，prefill是处理输入prompt的过程——你输入”帮我写一首诗”，模型先把这几个字全部过一遍，才开始生成。而prefill和decode最根本的区别是什么呢？ 那就是，输入序列长度是可变的。用户A输入5个token，用户B输入500个，用户C输入5000个——不可预测，不固定。序列长度不固定，计算图就不固定。
注意力计算的输入维度变了，中间激活值的shape变了，kernel配置变了——这不是”数据在变，图不变”，是”图本身在变”。
CUDA graph要求capture时和replay时计算图结构完全一致。图都在变，没办法复用。      

decode阶段用padding解决了batch size不固定的问题——提前capture几个固定shape的graph，用的时候选最近的。那么，prefill阶段，能不能用类似思路，提前capture几个固定序列长度的graph，实际用的时候padding到最近的长度？    
问题1：序列长度的分布太分散，padding浪费太大        
问题2：prefill阶段的计算特性，导致CUDA graph的收益本来就有限      
问题3：连续批处理带来额外动态性     
vLLM用连续批处理，不同请求的prefill被动态打包——这一批三个请求，下一批五个，每次组合不一样。这种动态性让prefill阶段的输入形状变化更不规律，进一步降低了CUDA graph的适用性。

</details>

<details>

<summary>prefill 阶段可行的方案</summary>   

方案一：Chunked Prefill + CUDA graph     
把长的prefill序列切成固定长度的chunk，每个chunk序列长度固定，就可以capture CUDA graph了。这是目前最主流的思路。vLLM已经支持了Chunked Prefill，还没有在这个基础上完整接入CUDA graph，但方向是明确的。工程上还有几个坑要处理：chunk边界处KV Cache怎么管理，跨chunk的注意力计算怎么拼接，padding带来的额外计算怎么控制。技术上可行，工程上在推进。

方案二：提前capture几个固定序列长度的graph     
类似decode阶段的做法，针对特定场景可以用。如果你的应用序列长度分布比较集中——比如某个RAG应用context基本固定在1024左右——可以只capture那几个长度的graph，padding浪费可控，性能收益实在。通用场景下性价比不够高，垂直场景下是可行的工程选择。

方案三：torch.compile和fused kernel    
用torch.compile或者自己写triton kernel，把多个操作fuse成一个，某种程度上替代CUDA graph——减少kernel launch次数。TensorRT-LLM在这个方向走得比vLLM激进，prefill阶段的优化更深，但代价是灵活性大幅下降，支持的模型种类少，不是vLLM的设计哲学。

这里有个根本的权衡：灵活性和极致性能，很难同时做到。    
vLLM选了灵活性——支持几十种模型架构，各种量化方案，动态调度。代价是某些性能没做到极致。TensorRT-LLM选了极致性能——在NVIDIA自家硬件上把每块榨干，代价是灵活性大幅下降。没有对错，是不同的设计哲学，服务不同的场景。    

</details>

### 12 LLM prefill是计算密集型的 or 是访存密集型的？   
**Roofline**模型      
区分一个操作是计算密集型还是访存密集型，看的是算术强度（Arithmetic Intensity），即**每字节内存访问对应多少次浮点运算（FLOPs/Byte）**      
硬件拐点 = 算力峰值 / 显存带宽峰值  

算术强度高 → 计算密集型，瓶颈在算力     
算术强度低 → 访存密集型，瓶颈在内存带宽     

**decode阶段**       
在给定的一个 Batch size，每次只生成1个token，核心操作是矩阵-向量乘法——把权重矩阵（GEMV）（[d_model, d_model]）和一个向量（[1, d_model]）相乘。搬运权重矩阵，但decode每次只产生一个输出向量，算术强度极低。另外，反复读写KV Cache，每生成一个 token 就要追加 KV 缓存，序列越长，KV 访存开销线性上涨，进一步压低算术强度，加重访存瓶颈，这是线上长文本推理 Decode 吞吐暴跌的核心原因。    
Efficiently Scaling Transformer Inference，里面有对prefill和decode阶段roofline分析的详细讨论，长序列prefill是compute-bound，decode是memory-bound。   
A100上做FP16推理，decode阶段算术强度大概在**1-10 FLOPs/Byte**量级，远低于A100的算力/带宽比（约200 FLOPs/Byte）——访存密集型。
这也是为什么decode阶段kernel launch overhead是真实瓶颈——GPU的计算单元大量空转，任何额外的overhead占比都会被放大。   

**prefill阶段**      
在给定的一个 Batch size，一次处理N (N >>1) 个token，核心操作是矩阵-矩阵乘法（GEMM）——输入[N, d_model]，权重[d_model, d_model]。    
和decode一样要搬运权重矩阵一次，但这次要做N倍的计算——算术强度随N线性增长。当N足够大（几千以上），算术强度可以轻松超过GPU的 roofline临界点，进入计算密集区间。GPU的算力被充分利用，kernel launch overhead在总时间里占比就很小了。这也是为什么FlashAttention在prefill阶段的加速效果显著——它的核心优化是减少HBM访问次数，这个优化在计算密集型场景下才特别有价值。prefill阶段要把巨大的权重矩阵从显存搬到计算核心，这个搬运本身就是内存带宽受限的，当序列N很大的时候，计算量的增长速度超过了访存量的增长速度，算术强度随N上升，瓶颈就从访存转移到了计算。    
prefill在 $N$ 非常小（如几十个 Token）和并行 Batch 较小时，矩阵的形状（Shape）对 Tensor Core 并不友好，导致计算资源无法填满或 高效利用Tensor Core，这时它表现得更像访存密集型。FlashAttention 的视角： FlashAttention 缓解了带宽压力（它是通过减少内存读写实现的）。确实，FlashAttention 的核心是 IO-Awareness，但它通过 Tiling 极大地提升了 Cache 命中率，将原本可能受限的访存压力转化为更高效的片上计算，从而让 GPU 更快地达到计算峰值。  
但在长序列或大 Batch size(短序列)的典型 Prefill 场景下，随着 $N$ 的增加，GEMM 的计算量呈线性增长，算术强度已经越过了 Roofline 的拐点，此时算力成了主要的瓶颈。在这个阶段，通过 CUDA Graph 优化 Launch Overhead 的收益，远不如在 Decode 阶段那么显著。   

长序列（大Batch 或 小Batch）/ 大Batch：优先提升 Tensor Core 利用率、算子融合、张量并行；带宽优化（FlashAttention）为辅；CUDA Graph 收益有限，可后置优化；
短序列小Batch：优先 FlashAttention、KV Cache 压缩、CUDA Graph 降低访存与 launch 开销

流水线重叠（Pipeline Overlap）   
在 Prefill 阶段，GEMM 的计算时间非常长，算子之间存在极大的计算隐藏空间（Compute Hiding）。驱动程序和异步执行引擎有足够的时间在后台发射下一个 Kernel，使得 Launch Overhead 被“隐藏”在了长计算任务之后。而在 Decode 阶段，每个 Kernel 执行时间极短，CPU 还没来得及发完下一个，GPU 就闲置了，所以必须用 CUDA Graph 来压缩这种空窗期。

Prefill 优化优先级
长序列：优先提升 Tensor Core 利用率、算子融合、张量并行；带宽优化（FlashAttention）为辅；CUDA Graph 收益有限，可后置优化；
短序列小 Batch：优先 FlashAttention、KV Cache 压缩、CUDA Graph 降低访存与 launch 开销；

Decode 优化优先级（线上推理核心瓶颈）
第一优先级：CUDA Graph 打包全流程算子、算子融合消除小 launch；
第二优先级：KV Cache 量化、分页缓存 PagedAttention 降低 HBM 读写；
算力优化收益极低，无需投入大量精力提升 FP 计算速度；

硬件选型指导
离线批量文本预处理（纯 Prefill 场景）：优先高算力 GPU（H100/H200）；
线上对话流式生成（大量 Decode）：优先高显存带宽、大容量 HBM 的 GPU。


### 13 FlashAttention 在 Prefill 加速显著，为什么？ 那在decode 阶段呢    
FlashAttention 在 Prefill 加速显著，根源是减少 HBM 读写，优化在计算密集场景价值更高，逻辑闭环：    
标准 Attention 会把巨大的\(QK^T\)、softmax 中间张量写入 / 读出 HBM，全局访存开销爆炸；    
FlashAttention 分块计算，中间结果放在片上 SM 共享内存，规避海量 HBM 往返；   
Prefill 存在大规模并行 N，有大量矩阵运算（计算密集），HBM 访存是隐性冗余开销，削减访存能大幅降低总耗时；    
Decode 单 token 无大规模 Attention 矩阵运算，FlashAttention 收益低，和工业实测完全匹配。      

### 14 CUDA Graph 在prefill 和 decode 阶段各自的影响  
Prefill（大 N、计算受限、单 Kernel 长耗时）：launch overhead 占比极低 → CUDA Graph 提升有限；    
Decode（访存受限、大量短耗时小 Kernel）：launch overhead 占比极高 → CUDA Graph 收益巨大；   

