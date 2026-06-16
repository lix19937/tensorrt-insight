

### 1 带有历史帧信息的时序模型如何构建graph ?  

![1](./timing.jpg)   

![3](./captrue_timingv0.png)

![2](./captrue_timing.png)   



### 2 Tensor 的 Base Address（起始地址） 保持不变，仅仅是 Shape 不同，依然会直接影响 CUDA Graph 的有效性。     

### 3 cudaGraph_t, cudaGraphExec_t 生命周期怎么管理   
