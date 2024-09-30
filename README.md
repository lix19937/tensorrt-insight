TensorRT 是Nvidia 推出的跨 nv-gpu架构的半开源高性能AI 推理引擎框架/库，提供了cpp/python接口，以及用户自定义插件方法，涵盖了AI 推理引擎技术的主要方面。          

TensorRT is a semi-open source high-performance AI inference engine framework/library developed by Nvidia, which spans across nv-gpu architectures.    
Provides cpp/python interfaces and user-defined plugin methods, covering the main aspects of AI inference engine technology.   


|**topic**                       | **主题**| **备注**   |      
|    ---                         | --- |     --- |          
|[overview](./overview.md)       |概述  |   |            
|[layout](./layout/)    |内存布局|      |            
|[compute_graph_optimize](./compute_graph_optimize/)    |计算图优化|   |             
|[dynamic_shape](./dynamic_shape/)  |动态shape |     |         
|[plugin](./plugin/)    |插件  |      |           
|[calibration](./calibration/)  |标定 |    |                
|[asp](./asp/)          |稀疏 |        |         
|[qat](./qat/)          |量化感知训练 |      |           
|[trtexec](./trtexec/)     |辅助工具 |     |           
|[runtime](./runtime/)     |运行时|        
|[inferflow](./inferflow/) |模型调度|      |         
|[mps](./device-benchmark-mps/)| MPS|  |    
|[deploy](./deploy/)       |基于onnx部署流程， trt 工具使用 | |    
|[py-tensorrt](./py-tensorrt/) | python tensorrt封装  | 解析 tensorrt `__init__` |        
|[cookbook](https://github.com/lix19937/trt-samples-for-hackathon-cn/blob/master/cookbook/) |食谱|      |       
|[incubator](./incubator/)|孵化器|  |         
|[developer_guide](./developer_guide/)                  |开发者指导|      |           
|[triton-inference-server](./triton-inference-server/)    | triton|      |          
|[cuda](./cuda/)    | cuda编程|      |          
|[onnxruntime op](https://github.com/lix19937/xop)| onnxrt 自定义op|辅助图优化，layer输出对齐   |      

 
    
## Reference     
https://docs.nvidia.com/deeplearning/tensorrt/archives/   
https://developer.nvidia.com/search?page=1&sort=relevance&term=   
https://github.com/HeKun-NVIDIA/TensorRT-Developer_Guide_in_Chinese/tree/main    
https://docs.nvidia.com/deeplearning/tensorrt/migration-guide/index.html      
https://developer.nvidia.com/zh-cn/blog/nvidia-gpu-fp8-training-inference/
