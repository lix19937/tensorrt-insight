
## CUDA Events    
CUDA Events是用来准确记录事件从而在Streams间进行同步的工具。让Events标记stream执行过程中的一个点，我们就可以检查正在执行的stream中的操作是否到达该点，我们可以把event当成一个操作插入到stream中的众多操作中，当执行到该操作时，所做工作就是设置CPU的一个flag来标记表示完成。    
