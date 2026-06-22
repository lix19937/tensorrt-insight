
### 1 CUDA Events    
CUDA Events是用来准确记录事件从而在 streams间进行同步的工具。让Events标记stream执行过程中的一个点，我们就可以检查正在执行的 stream中的操作是否到达该点，我们可以把event当成一个操作插入到 stream中的众多操作中，当执行到该操作时，所做工作就是设置CPU的一个flag来标记表示完成。    

### 2 不同 cuda streams 里的 kernel 会串行吗   
https://forums.developer.nvidia.com/t/kernels-in-cuda-streams-seems-not-running-in-parallel/286549/7    

编译时的优化导致了“空内核”（null kernel）现象（即代码几乎在编译阶段就已经计算完毕），使得内核执行时间极短（小于 1 微秒），远低于内核调用开销（2-8 微秒）。因此，可以通过增加一些内核代码来延长其运行时间，并防止编译器进行过度优化，从而在全局层面上有效地体现出 HyperQ 的性能优势。   

### 4 <<< >>>() 和 cudaLaunchKernel    
<<<...>>> 是 CUDA 的语法糖（Syntax Sugar），而 **cudaLaunchKernel** 是底层的函数实现。     
这是 CUDA Runtime API 提供的标准函数，它是 GPU 任务调度真正的“入口”。

+ 显式参数控制：手动构造参数数组（void args），这在动态生成参数或处理高度灵活的算子库时非常有用。
+ 灵活性：它允许你在不知道函数名的情况下，通过函数指针（const void* func）动态启动 kernel。

### 5 cudaLaunchKernel 本身是线程安全的（Thread-safe），支持多线程并发调用     
多线程并发调用 cudaLaunchKernel 并不等于 GPU 上的 kernel 可以并发执行。你可以在 CPU 端通过多线程“并发”地 Launch 了多个 kernel，但这些 kernel 是否能真的 “并发执行（Concurrent Execution）”，取决于以下三个决定性因素：    

A. CUDA stream 的约束
这是最重要的因素。即便你通过多线程调用 cudaLaunchKernel：        
如果这些 kernel 分配的是同一个非空流（Non-null Stream），它们在 GPU 上绝不会并发，而是严格串行。
如果每个线程分配了不同的流（Different Streams），GPU 调度器才有机会让它们并行执行。      

B. 硬件资源限制 (Occupancy)       
即使你使用了不同的流，如果你的 kernel 1 把 GPU 的所有 SM（流处理器）、寄存器和共享内存都占满了，kernel 2 就算被 Launch 了，也只能在队列里“干瞪眼”等待资源释放。

C. 硬件调度器 (GigaThread Scheduler)       
GPU 硬件调度器会扫描所有活跃流中的任务。如果多个流中有任务准备就绪，且 GPU 剩余资源足够，调度器会自动将这些 Kernel 的 Warp 同时分发到不同的 SM 上运行。   

### 6 多线程 Launch 的工程陷阱      
3个潜在问题：     
+ Launch 顺序不确定性：如前所述，由于 OS 线程调度的不确定性，即便是 thread1{Launch A} 和 thread2{Launch B}，你无法 100% 保证 A 一定比 B 先入队。如果逻辑依赖顺序，必须使用 cudaStreamWaitEvent 进行同步。    
+ 驱动开销（Overhead）：尽管 API 是线程安全的，但在极高并发下（例如上百个线程同时 Launch 极小的 kernel），驱动程序的锁竞争（Lock Contention）会成为性能瓶颈，导致 CPU 侧 Launch 耗时暴增。
+ 上下文切换与同步：在多线程中频繁调用 cudaStreamSynchronize 或 cudaDeviceSynchronize 会导致整个流水线停滞，失去并发意义。

如果你是为了精细控制任务执行顺序：     
请务必通过 **cudaStreamWaitEvent** 构建逻辑依赖，而不是试图依赖 CPU 线程的启动先后顺序。   
硬件同步层级：事件同步是GPU 调度器层面依赖，比 CPU 同步（cudaStreamSynchronize）性能高很多，是多流依赖推荐方案；       
若需要多流链式依赖（A→B→C），需要分别创建 event1 (A→B)、event2 (B→C)，不能共用单个 event。    

### 7 流优先级      
+ 抢占与调度权重：当你创建了具有不同优先级的流，CUDA 驱动程序和硬件调度器会为这些流分配不同的调度权重。     
+ 计算资源偏向：高优先级流（High Priority）并非能够“抢占”正在执行的低优先级内核（kernel），而是在硬件调度器决定下一个执行哪个 kernel 时，高优先级流中的任务具有优先进入计算单元（SM）的权利。      
优先级仅对 **计算核函数 kernel** 的调度有效，对内存拷贝（cudaMemcpyAsync）等传输操作的影响极小。优先级（Priority） 是精细化控制 GPU 资源调度、解决计算与传输竞争（Resource Contention）的利器。
   
```cuda
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

// 通常：数值越小，优先级越高。
// 即 greatestPriority < leastPriority

cudaStream_t highPriorityStream, lowPriorityStream;

// 创建高优先级流 (greatestPriority)
cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority);

// 创建低优先级流 (leastPriority)
cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamNonBlocking, leastPriority);


// 关键计算
keyKernel<<<grid, block, 0, highPriorityStream>>>(...);

// 后台计算
backgroundKernel<<<grid, block, 0, lowPriorityStream>>>(...);
```

### 8 注意    
在实际调优中，不要盲目使用优先级，请遵循以下原则：     
+ 避免滥用：如果你发现系统性能不佳，通常首要任务是优化 kernel 本身（如共享内存使用、访存合并、规避分支分歧），而不是依赖优先级。优先级是最后的调优手段。      
+ 针对场景：最典型的应用场景是 “实时处理 + 背景任务”。例如，自动驾驶系统中，视觉推理任务放入高优先级流，而传感器日志记录或非关键的数据分析放入低优先级流。    
+  资源争用测试：优先级在资源利用率达到 90%+ 时效果最明显。如果你的 GPU 负载很轻，优先级几乎不会带来性能提升。   
+  流同步：请始终注意使用 cudaStreamWaitEvent 来管理不同优先级流之间的依赖关系，防止高优先级任务因等待低优先级任务的数据而发生“优先级反转”。

优先级是 CUDA 提供的一种“软控制”手段，它通过干预调度器在资源竞争时的决策逻辑，来保障核心业务的吞吐量。它不能创造额外的算力，但能显著优化混合负载下的响应延迟（latency）。

### 9 示例   
#### 1 如果 A kernel 使用低优先级流调用， B kernel 采样高优先级流调用， A kernel, B kernel 均在同一个线程中串行调用  
```cuda
B<<<grid, block, 0, highPriorityStream>>>(...);     
A<<<grid, block, 0, lowPriorityStream>>>(...);    
```

结论： B launch 早于  A launch ,  B run 早于 A run     

在同一个 CPU 线程中，launch 是阻塞式串行的  

#### 2  如果 A kernel 使用低优先级流调用， B kernel 采样高优先级流调用， A kernel, B kernel 各在不同线程中调用
```cuda
void fa() { A<<<grid, block, 0, lowPriorityStream>>>(...); }
void fb() { B<<<grid, block, 0, highPriorityStream>>>(...); }

int main()
{
    std::thread ta(fa);  
    std::thread tb(fb); 

    ta.join();
    tb.join();
    return 0;
}
```

结论：不确定  

谁先调用 <<<...>>> launch kernel，取决于：       
OS 调度：操作系统决定哪个线程`ta` or `tb` 先获得 CPU 时间片。     
Driver 响应：cudaLaunchKernel 内部涉及锁机制（mutex），驱动程序需要处理来自不同线程的请求。

#### 3  使用 cudaStreamWaitEvent 实现流之间依赖    
```cuda    
// 假设这是类成员变量   Create/Destroy event 会导致 CPU 侧 Launch 延迟抖动
// cudaEvent_t event; 
// cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

void safeSyncDemo(cudaStream_t streamA, cudaStream_t streamB)
{
    // 1. 提交任务 A
    kernelA<<<grid, block, 0, streamA>>>(...);

    // 2. 记录事件
    cudaEventRecord(event, streamA);

    // 3. 异步等待（不阻塞 CPU）  0：代表无超时，cpu 不等待
    cudaStreamWaitEvent(streamB, event, 0);

    // 4. 提交任务 B
    kernelB<<<grid, block, 0, streamB>>>(...);
    
    // 注意：不要在这里 destroy event
    // 如果必须销毁，请确保 streamB 已经完成了相关任务
}
```

结论： kernelB 提交到 streamB，硬件层面保证 kernelA 执行完才跑 kernelB；
```
cudaEventRecord(event, streamA) 
   把事件插入 streamA 任务队列，仅当 streamA 中前面所有任务（kernelA）跑完，事件才会被标记完成；

     
cudaStreamWaitEvent(streamB, event, 0)
   CPU 侧非阻塞，不会卡住主机，给 streamB 插入一条内部等待指令，GPU 调度器会阻塞 streamB 后续任务，直到 event 完成；
```  
 
