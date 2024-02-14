## PTQ     
* 使用 TensorRT 闭源方法进行 PTQ     
https://github.com/lix19937/trt-samples-for-hackathon-cn/tree/master/cookbook/03-BuildEngineByTensorRTAPI/MNISTExample-pyTorch/C%2B%2B

  https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_int8_calibrator.html
  
  |类型| 说明|  
  |------------|-------------|   
  |IInt8EntropyCalibrator | Entropy calibrator. This is the Legacy Entropy calibrator. It is less complicated than the legacy calibrator and produces better results.|  
  | IInt8EntropyCalibrator2 | Entropy calibrator 2. This is the preferred calibrator. This is the required calibrator for DLA, as it supports per activation tensor scaling.|
  |IInt8LegacyCalibrator |Legacy calibrator left for backward compatibility with TensorRT 2.0. This calibrator requires user parameterization, and is provided as a fallback option if the other calibrators yield poor results.  |  
  | IInt8MinMaxCalibrator |  It supports per activation tensor scaling. |
  


* **使用 pytorch-quantization 进行Q-DQ设置，然后进行开源方法 PTQ**     
https://github.com/lix19937/pytorch-quantization/tree/main/pytorch_quantization/calib    
  + max   
  + hist
    + 交叉熵
    + mse
    + 统计分位数
https://github.com/lix19937/pytorch-quantization/blob/main/readme_lix.md
       
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

