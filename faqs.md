https://forums.developer.nvidia.com/c/ai-data-science/deep-learning/tensorrt/92    

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
