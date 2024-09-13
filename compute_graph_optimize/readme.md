## 计算图优化    

例如LayerNorm本身为一个算子，但是ONNX导出会拆分为很多细小的算子，则必须用图优化将其合并回去。又比如ONNX不支持NHWC等数据格式，但是硬件部署往往又需要转换到硬件优化的NHWC或其他数据格式。另外，根据引擎和芯片的特殊性，往往也需要对模型进行一些针对性修改优化，例如一些引擎不支持超过5D的transpose计算。    

## 如何做图优化？  
* 第一个阶段**在线优化**，是直接对代码（网络的forward）进行修改，从而导出的模型直接具有优化后的特性。对于一些算法依赖，通用性并不是很强，或者底层实施图优化困难的优化可以考虑该方法。     
* 第二个阶段**离线优化**，是对导出的模型进行优化，例如对导出的ONNX模型进行优化。一般离线优化前可使用推理引擎如TensorRT进行 layer info profile，看看 layer 的融合情况，再针对性优化onnx 。     
* 第三个阶段是**在深度学习引擎内部运行时进行优化**，可以结合硬件特点和输入shape对应的性能信息进行更加针对性和极致的优化。第三阶段一般也可在第一阶段执行后执行辅助分析，给第二阶段提供参考。    


*  OneHot+MatMul  to  Gather    

* focus

* permute + conv [+ permute]
  

*  多个gather可以替换为split    split adjust     

*  统一归一化函数  
其实由F.normalize(x, p=2.0, dim=-1)导出，可以进行针对性合并。第二个图计算跟第一个图类似，但是为不同代码编写方式，这种最好是修改模型pytorch代码改成同一种norm函数，降低模型图优化代码开发。  

*  SpaceToDepth与DepthToSpace算子   
图中reshape+transpose(perm=[0, 1, 3, 5, 2, 4])计算等价于SpaceToDepth，由于transpose场景特别多，这里替换后可以在计算上进行更加针对性优化，此外，这个优化使得做NCHW到NHWC等格式转换优化也更加容易    

transpose(perm=[0, 1, 4, 2, 5, 3])等价于DepthToSpace   

*  transpose+reshape+transpose to  transpose


* high-mm -> low-mm
    

*  矩阵乘+BN 融合


*  反卷积+BN 也应该能进行融合      


*  合并相邻的Conv2D或MatMul，无非线性层 conv + conv (without non-linear op)    


*  相邻的卷积/matmul+bias，中间没有非线性层的话可以基于卷积和矩阵乘的线性性直接合并 conv + linear (without non-linear op)    


*  特殊的1x1 depthwise卷积替换为elemwise


*  MatMul与Add, Mul向量计算融合


*  LayerNorm算子合并   
 

*  squeeze  unsqueeze 去除      


*  reshape去除
  

*  多路相同计算patten合并为batch计算，显著降低算子数量，提升算子计算密集性       
如图的场景；此外另一个场景是transformer模型中attention三个矩阵乘合可以并成为batch matmul，或者自己实现的特殊能够同时接受多个输入和bias的矩阵乘。  

*  多路并行的slice在特定情况下可以换成gather，并且后面所有的这些elemwise都可以合成一路计算。   
下图中这个稍微复杂点，slice可以替换为gahter，elemwise可以跟着合并，但gatherV2合并需要自定义算子batch gatherV2。   


*  调整顺序从而把div折叠到matmul和bias_add参数里面    
  

通常不鼓励矩阵和大容量内存拷贝的`物理重排`的原因是，数据移动非常昂贵（在能量和性能方面），因此应将其最小化，理想情况下应仅作为`计算的一部分`进行。  
换句话说，每次我们接触一些数据时，我们都想在存储之前对其进行一些变换。


更多算子融合相关见 https://github.com/lix19937/llm-deploy/blob/master/op_fusion.md    


 

 

