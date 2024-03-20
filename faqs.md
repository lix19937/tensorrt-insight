
## onnx模型推理结果正确，但tensorRT量化后的推理结果不正确，大概原因有哪些？
可能原因有：   
a. calibrator的算法选择不对；   
b. calibration过程使用的数据不够；    
c. 对网络敏感层进行了量化；   
d. 对某些算子选择了不适合OP特性的scale计算。  

## 采用tensorRT PTQ量化时，若用不同batchsize校正出来模型精度不一致，这个现象是否正常？    
正常的    
因为calibration（校正）是以tensor为单位计算的。对于每次计算，如果histogram的最大值需要更新，那么PTQ会把histogram的range进行翻倍。不考虑内存不足的问题，推荐使用更大的batch_size，这样每个batch中包含样本更加丰富，校准后的精度会更好。但具体设置多大，需要通过实验确定（从大的batch size开始测试。一点一点往下减）。需要注意的是batch_size越大，校准时间越长。

## 模型量化到INT8后，推理时间反而比FP16慢，这正常吗？   
正常的     
这可能是tensorrt中内核auto tuning机制作怪（会把所有的优化策略都运行一遍，结果发现量化后涉及一堆其他的操作，反而效率不高，索性使用cuda core，而非tensorrt core）。当网络参数和模型架构设计不合理时，trt会添加额外的处理，导致INT8推理时间比FP16长。我们可以通过trt-engine explorer工具可视化engine模型看到。

## 如何创建针对多种不同批次大小进行优化的引擎？   
虽然 TensorRT 允许针对给定批量大小优化的引擎以任何较小的大小运行，但这些较小大小的性能无法得到很好的优化。要针对多个不同的批量大小进行优化，请在分配给OptProfilerSelector::kOPT的维度上创建优化配置文件。

## 引擎和校准表是否可以跨TensorRT版本移植？   
不会。内部实现和格式会不断优化，并且可以在版本之间更改。因此，不保证引擎和校准表与不同版本的TensorRT二进制兼容。使用新版本的TensorRT时，应用程序必须构建新引擎和 INT8 校准表。

## 如何选择最佳的工作空间大小？     
一些 TensorRT 算法需要 GPU 上的额外工作空间。方法IBuilderConfig::setMemoryPoolLimit()控制可以分配的最大工作空间量，并防止构建器考虑需要更多工作空间的算法。在运行时，创建IExecutionContext时会自动分配空间。即使在IBuilderConfig::setMemoryPoolLimit()中设置的数量要高得多，分配的数量也不会超过所需数量。因此，应用程序应该允许 TensorRT 构建器尽可能多的工作空间；在运行时，TensorRT 分配的数量不超过这个，通常更少。

## 如何在多个 GPU 上使用TensorRT ？    
每个ICudaEngine对象在实例化时都绑定到特定的 GPU，无论是由构建器还是在反序列化时。要选择 GPU，请在调用构建器或反序列化引擎之前使用cudaSetDevice() 。每个IExecutionContext都绑定到与创建它的引擎相同的 GPU。调用execute()或enqueue()时，如有必要，请通过调用cudaSetDevice()确保线程与正确的设备相关联。

## 如何从库文件中获取TensorRT的版本？     
符号表中有一个名为tensorrt_version_#_#_#_#的符号，其中包含TensorRT版本号。在 Linux 上读取此符号的一种可能方法是使用nm命令，如下例所示：
```
$ nm -D libnvinfer.so.* | grep tensorrt_version
00000000abcd1234 B tensorrt_version_#_#_#_#   
```

## 如果我的网络产生了错误的答案，我该怎么办？ 
网络生成错误答案的原因有多种。以下是一些有助于诊断问题的故障排除方法：    
打开日志流中的VERBOSE级别消息并检查 TensorRT 报告的内容。
检查您的输入预处理是否正在生成网络所需的输入格式。
如果您使用降低的精度，请在 FP32 中运行网络。如果它产生正确的结果，则较低的精度可能对网络的动态范围不足。
尝试将网络中的中间张量标记为输出，并验证它们是否符合您的预期。   
注意：将张量标记为输出会抑制优化，因此会改变结果。  
您可以使用Polygraphy来帮助您进行调试和诊断。

## 如何在TensorRT中实现批量标准化？   
批量标准化可以使用TensorRT中的IElementWiseLayer序列来实现。进一步来说：
```
adjustedScale = scale / sqrt(variance + epsilon) 
batchNorm = (input + bias - (adjustedScale * mean)) * adjustedScale   
```

## 为什么我的网络在使用 DLA 时比不使用 DLA 时运行得更慢？   
DLA 旨在最大限度地提高能源效率。根据 DLA 支持的功能和 GPU 支持的功能，任何一种实现都可以提高性能。使用哪种实现取决于您的延迟或吞吐量要求以及您的功率预算。由于所有 DLA 引擎都独立于 GPU 并且彼此独立，因此您还可以同时使用这两种实现来进一步提高网络的吞吐量。

## TensorRT支持INT4量化还是INT16量化？   
TensorRT 目前不支持 INT4 和 INT16 量化。  

## TensorRT 何时会在 UFF 解析器中支持我的网络所需的层 XYZ？    
UFF 已弃用。NV建议用户将他们的工作流程切换到 ONNX。 TensorRT ONNX 解析器是一个开源项目。

## 可以使用多个 TensorRT 构建器在不同的目标上进行编译吗？   
TensorRT 假设它所构建的设备的所有资源都可用于优化目的。同时使用多个 TensorRT 构建器（例如，多个trtexec实例）在不同的目标（DLA0、DLA1 和 GPU）上进行编译可能会导致系统资源超额订阅，从而导致未定义的行为（即计划效率低下、构建器失败或系统不稳定）。
建议使用带有 --saveEngine 参数的trtexec分别为不同的目标（DLA 和 GPU）编译并保存它们的计划文件。然后可以重用此类计划文件进行加载（使用带有 --loadEngine 参数的trtexec ）并在各个目标（DLA0、DLA1、GPU）上提交多个推理作业。这个两步过程在构建阶段缓解了系统资源的过度订阅，同时还允许计划文件的执行在不受构建器干扰的情况下继续进行。   

## 张量核心(tensor core)加速了哪些层？     
大多数数学绑定运算将通过张量核(tensor core)加速 – 卷积、反卷积、全连接和矩阵乘法。在某些情况下，特别是对于小通道数或小组大小，另一种实现可能更快并且被选择而不是张量核心实现。

## ref   
https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#faq   
