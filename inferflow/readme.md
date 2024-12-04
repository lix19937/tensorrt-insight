> 轻量级TRT推理 见 https://github.com/lix19937/tensorrt-insight/tree/main/dynamic_shape/src    

![image](https://github.com/user-attachments/assets/e2a7ae62-b4db-468c-965d-5fb417f472d0)

 
* [需求设计与功能定义](#需求设计与功能定义)     
* [技术验证与实现](#技术验证与实现)    

## 需求设计与功能定义     
### 环境模型压测与资源分析    
+ 如何利用multi-cudagraph + stream + thread 设计一个多模型调度框架？    @ taskflow + tensorrt

   |  0      |  1              |          2 |     3       |  4               |   5          |   6       |     
   |  -------|  ----           | ---------- | ----------  | ---------------- | ------------ |  ----     |    
   |v-camera | preprocess node | infer node | decode node | postprocess node | display node |  pipeline |     
   | nvmedia |  cuda           | cuda + dla | cuda        |  cpu             |   cpu        |     -     |       
   |  class  | class (Init, Run, DeInit)    | class       |  class           |  class       |  class    | -     |      
   |  debug  | stub/fileio/perf   |  stub/fileio/perf |  stub/fileio/perf      |  stub/fileio/perf             |  stub/fileio/perf | -     |   
  
  zero copy + io-free reformat           

+ 基于 OpenVX 实现一个多模型调度框架
  
+ 组合试验     
  + 隐藏&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;同一时刻，选择合适的模型进行并行，利用overlab掩盖小模型（指低算力）latency，完全消弭小模型推理耗时     
  + 确定性&nbsp;&nbsp;&nbsp;&nbsp;任务编排
  + multi-task模型head拆分并行       

+ 软件工程目录结构参考https://github.com/sogou/workflow
  |目录|说明|备注|       
  |---|---|----|     
  |benchmark|||     
  |docs|||
  |src|||
  |test|||
  |tutorial|||

  -----------    
更多见 [设计](设计.md)



