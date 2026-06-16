

### 1 带有历史帧信息的时序模型如何构建graph ?  

![1](./timing.jpg)   

![3](./captrue_timingv0.png)

![2](./captrue_timing.png)   

### 2 Tensor 的 Base Address（起始地址） 保持不变，仅仅是 Shape 不同，会影响 CUDA Graph 的有效性  ？
会   

### 3 cudaGraph_t, cudaGraphExec_t 生命周期怎么管理      
cudaGraph_t 类型的对象定义了kernel graph的结构和内容；
cudaGraphExec_t 类型的对象是一个“可执行的graph实例”：它可以以类似于单个内核的方式启动和执行。

### 4 graph 捕获和实例化过程中会执行kernel 吗     
不会执行     

### 5 graph 捕获过程中 有cpu 操作会出现什么情况     
cuda graph支持多个stream间的融合，而且不仅可以包含kernel执行，还可以包括在主机 CPU 上执行的函数和内存拷贝

### 6 cudagraph 在哪些场景/模型下提升性能显著     
CUDA Graphs 对效率的提升，但更复杂的计算逻辑提供了更多优化提升的空间

### 7 graph 可以跨越多 GPU吗， 通过什么机制      
可以   
