## 计算图优化    

例如LayerNorm本身为一个算子，但是ONNX导出会拆分为很多细小的算子，则必须用图优化将其合并回去。又比如ONNX不支持NHWC等数据格式，但是硬件部署往往又需要转换到硬件优化的NHWC或其他数据格式。另外，根据引擎和芯片的特殊性，往往也需要对模型进行一些针对性修改优化，例如一些引擎不支持超过5D的transpose计算。    

## 如何做图优化？  
* 第一个阶段在线优化，是直接对代码（网络的forward）进行修改，从而导出的模型直接具有优化后的特性。对于一些算法依赖，通用性并不是很强，或者底层实施图优化困难的优化可以考虑该方法。     
* 第二个阶段离线优化，是对导出的模型进行优化，例如对导出的ONNX模型进行优化。第三个阶段是在深度学习引擎内部运行时进行优化，可以结合硬件特点和输入shape对应的性能信息进行更加针对性和极致的优化。     


## OneHot+MatMul  to  Gather    


## 多个gather可以替换为split    

## 统一归一化函数  
其实由F.normalize(x, p=2.0, dim=-1)导出，可以进行针对性合并。第二个图计算跟第一个图类似，但是为不同代码编写方式，这种最好是修改模型pytorch代码改成同一种norm函数，降低模型图优化代码开发。  

## SpaceToDepth与DepthToSpace算子   
图中reshape+transpose(perm=[0, 1, 3, 5, 2, 4])计算等价于SpaceToDepth，由于transpose场景特别多，这里替换后可以在计算上进行更加针对性优化，此外，这个优化使得做NCHW到NHWC等格式转换优化也更加容易    

transpose(perm=[0, 1, 4, 2, 5, 3])等价于DepthToSpace   

## transpose+reshape+transpose to  transpose    

## 矩阵乘+BN融合


## 反卷积+BN也应该能进行融合    

## 合并相邻的Conv2D或MatMul   

## 特殊的1x1 depthwise卷积替换为elemwise


## MatMul与Add, Mul向量计算融合


## LayerNorm算子合并
 



 

 

