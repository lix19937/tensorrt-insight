## PTQ     
* 使用 TensorRT 闭源方法进行 PTQ   
* **使用 pytorch-quantization 进行Q-DQ设置，然后进行开源方法 PTQ**
https://github.com/lix19937/pytorch-quantization/tree/main/pytorch_quantization/calib    
  + max   
  + hist
    + 交叉熵
    + mse
    + 统计分位数
       
## 带自定义插件的后量化     
* onnx 上构建带plugin的层，标记plugin的输出tensor      
* plugin 需要支持fp32，然后在calib table中查找输出tensor 的scale     

## 显示量化设置   
插入 Q/DQ 在插件层的前后       
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/99191e22-7c9f-4774-ade8-665575e5f155)       


## fuse  
* PTQ calib 期间可以进行fuse_bn，减少bn layer的标定，降低标定时间和calib 误差      

## sensitivity layer profile  
* 找到所有 quant layer   
* 每次仅使能一层quant layer进行指标eval，记录到dict中 {"layer_name":eval_value}
* 如果是使用 pytorch-quantization calib，则是在pytorch 下寻找敏感层
* 如果是使用 TensorRT calib，则是在 fp32 下，使用 TensorRT 或 onnxruntime 下寻找   

