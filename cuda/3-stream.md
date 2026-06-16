
### 1 CUDA Events    
CUDA Events是用来准确记录事件从而在Streams间进行同步的工具。让Events标记stream执行过程中的一个点，我们就可以检查正在执行的stream中的操作是否到达该点，我们可以把event当成一个操作插入到stream中的众多操作中，当执行到该操作时，所做工作就是设置CPU的一个flag来标记表示完成。    

### 2 不同cuda stream里的kernels会串行吗   
https://forums.developer.nvidia.com/t/kernels-in-cuda-streams-seems-not-running-in-parallel/286549/7    

编译时的优化导致了“空内核”（null kernel）现象（即代码几乎在编译阶段就已经计算完毕），使得内核执行时间极短（小于 1 微秒），远低于内核调用开销（2-8 微秒）。因此，可以通过增加一些内核代码来延长其运行时间，并防止编译器进行过度优化，从而在全局层面上有效地体现出 HyperQ 的性能优势。   

### 3 
