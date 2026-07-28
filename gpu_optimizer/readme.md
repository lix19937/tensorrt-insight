# gpu 性能优化  

## 软件安装    
NVIDIA GPU从调用方式上可以分为CUDA和OpenACC两种方式：CUDA是由NVIDIA开发的编程框架，需将代码按照CUDA的方式进行编写；OpenACC是由NVIDIA联合多家厂商推出的编程框架，形式上参考了OpenMP，通过引导语对循环代码完成GPU卸载。

CUDA编译运行依赖CUDA driver以及runtime。OpenACC编译运行则额外需要部署NVIDIA HPC SDK  。
  + [cuda](./1.1-install-cuda.md)        
  + [nvidia-hpc-sdk](./1.2-install-NVIDIA-HPC-SDK.md)      

## [优化方法](2.1-optimizer_methods.md)  

## [优化工具](2.2-optimizer_tools.md)  

## [硬件方向](2.3-optimizer_hw.md)  

## [编译方向](2.4-optimizer_compiler.md)  

## [kernel方向](2.5-optimizer_kernel.md)  

## [相关库](2.6-optimizer_librarys.md)  

