可以在量化完之后 进行 稀疏，反之则不行      

## 稀疏原理    

NVIDIA Ampere Architecture GPUs support Structured Sparsity. To make use of this feature to achieve higher inference performance, the `convolution kernel weights` and the `fully connected weights` must meet the following requirements:     
For each output channel and for each spatial pixel in the kernel weights, every four input channels must have at least two zeros. In other words, assuming that the kernel weights have the shape [K, C, R, S] and C % 4 == 0, then the requirement is verified using the following algorithm:
每四个输入通道必须至少有两个零。   

k 输出通道 ， c输入通道   
```py
hasSparseWeights = True
for k in range(0, K):
    for r in range(0, R):
        for s in range(0, S):
            for c_packed in range(0, C // 4):
                // 如果非零数目大于2 了 ，那就不能稀疏  
                if numpy.count_nonzero(weights[k, c_packed*4:(c_packed+1)*4, r, s]) > 2 :
                    hasSparseWeights = False
                    
```
强制内核权重具有结构化稀疏性模式可能导致精度损失。为了通过进一步的微调来恢复丢失的精度，需要进行稀疏训练。  https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity     


实现模型的2：4稀疏剪枝，同时还可以通过开启通道置换算法将绝对值较大的参数进行保留，以求对模型精度的影响最小化。prune_trained_model函数会计算出稀疏mask并将其施加在模型的权重上。

该项目还可以通过开启通道置换算法，来为结构化稀疏后的模型保留最大的精度值。
通道置换算法，顾名思义，就是通过沿着权重矩阵的通道维度进行置换，并对其周围的模型层进行适当调整。


 


## Ref   
apex   
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#structured-sparsity    
