# overview   
   

深度学习模型运算一般分为训练（training）和 推理（inference）两个部分：   
* 训练：包含前向传播和后向传播两个阶段，针对的是训练集。训练时通过误差反向传播来不断修改网络权重值。  
* 推理：包含前向传播一个阶段，针对的是除了训练集之外的新数据。可以是测试集，但主要针对新数据进行预测，预测的速率是一个很重要的因素。  
* 训练时为了加快速率，会使用多GPU分布式训练。在工程推理时，为了降低成本或产品应用限制，往往使用单个GPU机器甚至嵌入式平台，比如NVIDIA Jetson，Tesla T4等。由于训练的网络模型可能会很大，参数很多，而部署端的机器性能存在差异，就会导致推理速率慢，延迟高。这对于那些高实时性的应用场合是致命的，比如自动驾驶要求实时目标检测，目标追踪等。  
![image](https://github.com/lix19937/tensorrt-cookbook/assets/38753233/e96015e2-0f49-4b05-8c32-574b8ebba9cf)   
使用训练框架执行推理很容易，但是与使用TensorRT之类的优化解决方案相比，相同GPU上的性能往往低得多。训练框架倾向于实施强调通用性的通用代码，而当优化它们时，优化往往集中于有效的训练。TensorRT通过结合抽象出特定硬件细节的高级API和优化推理的实现来解决这些问题，以实现**高吞吐量，低延迟和低设备内存占用**。TensorRT 就是对训练好的模型进行优化，由于TensorRT就只包含推理优化器，优化完的网络不再需要依赖深度学习框架，可以直接通过TensorRT 部署在NVIDIA的各种硬件中。     
![image](https://github.com/lix19937/tensorrt-cookbook/assets/38753233/9d12d383-09c4-4b3a-b9c0-77208f0dc84f)

对于深度学习推理，有五个用于衡量软件的关键因素：   
* 吞吐量：给定时间段内的产出量。每台服务器的吞吐量通常以推断/秒或样本/秒来衡量，对于数据中心的经济高效扩展至关重要。   
* 效率：每单位功率交付的吞吐量，通常表示为性能/瓦特。效率是经济高效地扩展数据中心的另一个关键因素，因为服务器，服务器机架和整个数据中心必须在固定的功率预算内运行。   
* 延迟：执行推理的时间，通常以毫秒为单位。低延迟对于提供快速增长的基于实时推理的服务至关重要。    
* 准确性：训练好的神经网络能够提供正确答案。对于图像分类算法，关键指标为top-5或top-1。 
* 内存使用情况：需要保留以在网络上进行推理的主机和设备内存取大小决于所使用的算法。这限制了哪些网络以及网络的哪些组合可以在给定的推理平台上运行。这对于需要多个网络且内存资源有限的系统尤其重要，例如，在智能视频分析和多摄像机，多网络自动驾驶系统中使用的级联多级检测网络。    

TensorRT的解决方案是：    
* 权重与激活精度校准：通过将模型量化为 FP16或INT8来更大限度地提高吞吐量，同时保持高准确度。   
* 层与张量融合：通过融合内核中的节点，优化GPU 显存和带宽的使用。   
* 内核自动调整：基于目标 GPU 平台选择最佳数据布局和算法。   
* 动态张量显存：更大限度减少显存占用，并高效地为张量重复利用内存。   
* 多流执行：用于并行处理多个输入流的可扩展设计。    

Batch推理    
* 在GPU上使用较大的batch几乎总是更有效，batch的作用在于能尽可能多地并行计算。模型的输入只有单个batch的时候，单个batch的计算量并不能充分的利用CUDA核心的计算资源，有相当一部分的核心在闲置等待中；当输入有多个batch的时候，由于GPU的并行计算的特性，不同的batch会同步到不同的CUDA核心中进行并行计算，提高了单位时间GPU的利用率。   
例如：FullyConnected层有V个输入和K个输出，对于1个batch的实例，可以实现为 1xV 的input矩阵乘以 VxK 的weight矩阵。如果是 N个batch的实例，这就可以实现为 NxV 乘以 VxK 矩阵。将向量-矩阵乘法变为矩阵-矩阵乘法，效率更高。此外，当网络包含MatrixMultiply层或FullyConnected层时，如果硬件支持Tensor Core，对于FP16和INT8的推理模式，将batch大小设置为32的倍数往往具有最佳性能。  

TensorRT对计算图执行优化：  
* 消除输出不被使用的层。
* 消除等同于无操作的操作。
* 卷积，偏置和ReLU操作的融合。
* 汇总具有足够相似的参数和相同的源张量的操作（例如，GoogleNet v5的初始模块中的1x1卷积）。
* 通过将层输出定向到正确的最终目的地来合并串联图层（例如，消除concat）。
* 在构建阶段会在虚拟数据上运行各层，以从其核目录中选择最快的核，并在适当的情况下执行权重预格式化和内存优化。
```
[05/09/2022-12:23:21] [V] [TRT] After Myelin optimization: 1 layers
[05/09/2022-12:23:21] [V] [TRT] Applying ScaleNodes fusions.
[05/09/2022-12:23:21] [V] [TRT] After scale fusion: 1 layers
[05/09/2022-12:23:21] [V] [TRT] After vertical fusions: 1 layers
[05/09/2022-12:23:21] [V] [TRT] After dupe layer removal: 1 layers
[05/09/2022-12:23:21] [V] [TRT] After final dead-layer removal: 1 layers
[05/09/2022-12:23:21] [V] [TRT] After tensor merging: 1 layers
[05/09/2022-12:23:21] [V] [TRT] After concat removal: 1 layers
```   
以一个典型的inception block为例，优化过程如下：    
首先对网络结构进行垂直整合，即将目前主流神经网络的conv、BN、Relu三个层融合为了一个层，称之为CBR。    
然后对网络结构进行水平组合，水平组合是指将输入为相同张量和执行相同操作的层融合一起。inception block中将三个相连的1×1的CBR组合为一个大的1×1的CBR。   
最后处理concat层，将contact层的输入直接送入下面的操作中，不用单独进行concat后在输入计算，相当于减少了一次传输吞吐。  
<div align=center><img src="https://github.com/lix19937/tensorrt-cookbook/assets/38753233/b5f07bf2-9bd3-4543-9b29-9eb5aa54eda7"></div>     

## 量化    
<div align=center><img src="https://github.com/lix19937/tensorrt-cookbook/assets/38753233/ca903fd9-84f6-407f-a42e-d2909833bd61"></div>

<!-- ![range](https://github.com/lix19937/tensorrt-cookbook/assets/38753233/ca903fd9-84f6-407f-a42e-d2909833bd61)   -->   
量化是将数值x映射到y的过程，其中 x 的定义域是一个大集合(通常是连续的)，而 y 的定义域是一个小集合（通常是可数的）。大部分深度学习框架在训练神经网络时网络中的张量都是32位浮点数的精度（Full 32-bit precision，FP32）。一旦网络训练完成，在部署推理的过程中由于不需要反向传播，完全可以适当降低数据精度，比如降为FP16或INT8的精度。量化后模型的体积更小，将带来以下的优势：     
* 减少内存带宽和存储空间  
深度学习模型主要是记录每个layer（比如卷积层/全连接层）的weights和bias。在FP32的模型中，每个 weight数值原本需要32-bit的存储空间，通过INT8量化之后只需要8-bit即可。因此，模型的大小将直接降为将近 1/4。不仅模型大小明显降低，activation采用INT8之后也将明显减少对内存的使用，这也意味着低精度推理过程将明显减少内存的访问带宽需求，提高高速缓存命中率，尤其对于像batch-norm，relu，elmentwise-sum这种内存约束(memory bound)的element-wise算子来说效果更为明显。

* 提高系统吞吐量（throughput），降低系统延时（latency）   
直观的来讲，对于一个专用寄存器宽度为512位的SIMD指令，当数据类型为FP32而言一条指令能一次处理 16个数值，但是当我们采用INT8表示数据时，一条指令一次可以处理64个数值。因此，在这种情况下可以让芯片的理论计算峰值增加4倍。   
Tesla T4 GPU 引入了 Turing Tensor Core 技术，涵盖所有的精度范围，从 FP32 到FP16 到 INT8。在 Tesla T4 GPU 上，Tensor Cores 可以进行30万亿次浮点计算（TOPS）。通过TensorRT我们可以将一个原本为FP32的weight/activation浮点数张量转化成一个fp16/int8/uint8的张量来处理。使用 INT8 和混合精度可以降低内存消耗，这样就跑的模型就可以更大，用于推理的mini-batch size可以更大，模型的单位推理速率就越快。   

TensorRT可以将以单精度（FP32）或者半精度（FP16）训练的模型转化为以INT8量化部署的模型，同时可以最小化准确率损失。由于INT8的表达范围远远小于FP32，生成8位整数精度的网络时不能像FP16一样直接缩减精度，TensorRT对精度为FP32的模型进行校验来确定中间激活的动态范围，从而确定适当的用于量化的缩放因子。INT8量化的本质是一种缩放（scaling）操作，通过缩放因子将模型的分布值从FP32范围缩放到 INT8 范围之内。按照量化阶段的不同，一般将量化分为quantization aware training(QAT)和post-training quantization(PTQ)。QAT需要在训练阶段就对量化误差进行建模，这种方法一般能够获得较低的精度损失。PTQ直接对普通训练后的模型进行量化，过程简单，不需要在训练阶段考虑量化问题，因此，在实际的生产环境中对部署人员的要求也较低，但是在精度上一般要稍微逊色于QAT。  
PTQ的量化方法分为非对称算法和对称算法。**非对称算法**是通过收缩因子和零点将FP32张量的min/max映射分别映射到UINT8数据的min/max， min/max -> 0 ~ 255。**对称算法**是通过一个收缩因子将FP32中的最大绝对值映射到INT8数据的最大值，将最大绝对值的负值映射到INT8数据的最小值，-|max|/|max| -> -128 ~ 127。TensorRT中范围选取为-127 ~ 127。  
对称算法中使用的映射方法又分为**不饱和映射**和**饱和映射**，两种映射的区别是FP32张量的值在映射后是否能够大致均匀分布在0的左右。如果分布不均匀，量化之后将不能够充分利用INT8的数据表示能力。   
简单的将一个tensor 中的 -|max| 和 |max|的FP32值映射到 -127和 127 ，中间值按照线性关系进行映射。这种对称映射关系为**不饱和的（No saturation）**。   
根据tensor的分布计算一个阈值|T|，将范围在 ±|T|的FP32值映映射到±127的范围中，其中|T|<|max|。超出阈值 ±|T|的值直接映射为 ±127。这种不对称的映射关系为**饱和的（Saturate）**。   

|No saturation | Saturate|    
|--------------|---------|    
|Quantize(x, max) = round(s * x) , where s = 127.f / amax, amax = abs(max) | Quantize(x, T) = round(s * clip(x, -T, T)) , where s = 127.f / T|   
|![image](https://github.com/lix19937/tensorrt-cookbook/assets/38753233/11a78549-eac2-41fb-8a75-0db83dee8ab0) |![image](https://github.com/lix19937/tensorrt-cookbook/assets/38753233/575e24e5-7ad1-40a8-a35a-693bf8b4dc6d)|   

对不同网络结构的不同layer的激活值进行直方图统计，如卷积层，有池化层。激活值并不是均匀的分布在[-max, max]之间，过大或者过小的激活值其实只占参数总体的一小部分，因此如果直接使用不饱和的映射关系不能有效利用INT8的表达范围，就会导致比较大的精度损失。如果可以找到一个范围，使网络中tensor 的绝大多数的值都存在于这个范围内，就可以利用这个范围来对 tensor进行量化。只要阈值|T|选取得当，就能将分布散乱的较大的激活值舍弃掉，也就有可能使精度损失不至于降低太多。根据实验，weights在这两种方式上没有显著的差异，而对activation使用饱和的量化方式会有比较显著的性能提升。因此TensorRT在模型上使用了这两种方法进行了混合量化。 **weights采取不饱和量化，activation使用饱和量化**。

由于网络的每个层输出值的分布都是不同的，饱和映射的方法就需要对每个层都计算出一个有效的阈值|T|。为了使INT8的输出结果尽可能的减少精度损失，需要通过校正数据集来衡量不同的INT8分布与原来的FP3F2分布之间的差异程度，选择出一个与原始的FP32分布最为相似的INT8分布。这个衡量分布之间差异的指标就是KL散度（Kullback–Leibler divergence），KL散度通常用来衡量一个分布相比另一个分布的信息损失。   
```
KL_divergence(P,Q):= SUM(P[i] * log(P[i] / Q[i] ), i)
```  
![hist](https://github.com/lix19937/tensorrt-cookbook/assets/38753233/774db75e-a080-4459-846e-9fd24e19be0d)

### INT8量化的过程如下：   
* 准备一个 FP32的模型。   
* 从验证集选取一个子集作为校准集，校准集应该具有代表性，多样性，最好是验证集的一个子集。   
* 在校准数据集上进行FP32推理。   
* 推理过程中遍历网络的每一层：   
  * 收集这一层的激活值，并做直方图，分成几个组别（官方给的一个说明使用的是2048组），分组是为了下面遍历 |T|时，减少遍历次数；  
  * 对于不同的阈值|T|进行遍历，因为这里|T|的取值肯定在第128-2047组之间，所以就选取每组的中间值进行遍历； 
  * 选取使得 KL散度取得最小值的|T|。   
* 返回一系列|T|值，每一层都有一个|T|，创建CalibrationTable。    

```
//首先分成 2048个组，每组包含多个数值（基本都是小数）
Input: FP32 histogram H with 2048 bins: bin[ 0 ], …, bin[ 2047 ] 
  
For i in range( 128 , 2048 ): // |T|的取值肯定在 第128-2047 组之间,取每组的中点
    reference_distribution_P = [ bin[ 0 ] , ..., bin[ i-1 ] ] // 选取前 i 组构成P，i>=128
    outliers_count = sum( bin[ i ] , bin[ i+1 ] , … , bin[ 2047 ] ) //边界外的组
    reference_distribution_P[ i-1 ] += outliers_count //边界外的组加到边界P[i-1]上，没有直接丢掉
    P /= sum(P) // 归一化
      
    // 将前面的P（包含i个组，i>=128），映射到 0-128 上，映射后的称为Q，Q包含128个组，
    // 一个整数是一组
    candidate_distribution_Q = quantize [ bin[ 0 ], …, bin[ i-1 ] ] into 128 levels
    
    //这时的P（包含i个组，i>=128）和Q向量（包含128个组）的大小是不一样的，无法直接计算二者的KL散度
    //因此需要将Q扩展为 i 个组，以保证跟P大小一样
    expand candidate_distribution_Q to ‘ i ’ bins 
    
    Q /= sum(Q) // 归一化
    //计算P和Q的KL散度
    divergence[ i ] = KL_divergence( reference_distribution_P, candidate_distribution_Q)
End For
//找出 divergence[ i ] 最小的数值，假设 divergence[m] 最小，
//那么|T|=( m + 0.5 ) * ( width of a bin )
Find index ‘m’ for which divergence[ m ] is minimal
threshold = ( m + 0.5 ) * ( width of a bin )
```




