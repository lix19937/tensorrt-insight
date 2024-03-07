apex/apex/contrib/sparsity     
https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity    commit id:b496d85fb88a801d8e680872a12822de310951fd     
### base https://github.com/NVIDIA/apex/commit/b496d85fb88a801d8e680872a12822de310951fd  版本    

-------------------------------

可以在量化完之后 进行 稀疏，反之则不行      
Sparse-QAT  

## 稀疏原理    

NVIDIA Ampere Architecture GPUs support Structured Sparsity. To make use of this feature to achieve higher inference performance, the `convolution kernel weights` and the `fully connected weights` must meet the following requirements:     
For each output channel and for each spatial pixel in the kernel weights, every four input channels must have at least two zeros. In other words, assuming that the kernel weights have the shape [K, C, R, S] and C % 4 == 0, then the requirement is verified using the following algorithm:     

每四个输入通道必须至少有两个零。   

![kcrs](https://github.com/lix19937/tensorrt-insight/assets/38753233/ec85a73c-c704-4f30-ae78-6122a90c7991)

### weight 的布局 kcrs   
k 输出通道 ， c输入(tensor)通道， r 高度， s 宽度        
```py
hasSparseWeights = True
for k in range(0, K):
    for r in range(0, R):  # height  
        for s in range(0, S): # width  
            for c_packed in range(0, C // 4): # 通道整除4  
                // 如果非零数目大于2 了 ，那就不能稀疏  
                if numpy.count_nonzero(weights[k, c_packed*4:(c_packed+1)*4, r, s]) > 2 :
                    hasSparseWeights = False
                    
```
强制内核权重具有结构化稀疏性模式可能导致精度损失。为了通过进一步的微调来恢复丢失的精度，需要进行稀疏训练。  https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity     


实现模型的2：4稀疏剪枝，同时还可以通过开启通道置换算法将绝对值较大的参数进行保留，以求对模型精度的影响最小化。prune_trained_model函数会计算出`稀疏mask`并将其施加在模型的权重上。

在使用ASP对一个新的（未经过稀疏的）推理模型启用结构化稀疏时需要同时调用init_model_for_pruning和compute_sparse_masks方法。  

init_model_for_pruning会为模型层添加新的mask buffer，用于保存compute_sparse_masks生成的mask，因此调用了 compute_sparse_masks后的模型的**state_dict会比之前多出一些数据**，这些数据均以_mma_mask结尾的名字进行命名。       

对于已经使用ASP enable了结构化稀疏的模型，在保存后重新加载时，需要先创建一个新的模型，并调用init_model_for_pruning方法为模型添加mask buffer后再load模型的state_dict，否则因为新模型的state_dict和之前保存的state_dict不同而报错。


该项目还可以通过开启通道置换算法，来为结构化稀疏后的模型保留最大的精度值。
通道置换算法，顾名思义，就是通过沿着权重矩阵的通道维度进行置换，并对其周围的模型层进行适当调整。


 


## Ref   
apex   
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#structured-sparsity    
https://blog.csdn.net/weixin_43669978/article/details/132298127   
https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html   
