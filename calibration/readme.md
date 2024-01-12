## ptq   
* 使用 TensorRT 闭源方法进行 ptq  
* 使用 pytorch-quantization 进行Q-DQ设置，然后进行开源方法 ptq

## 带自定义插件的量化     

## fuse  
* ptq calib 期间可以进行fuse_bn，减少bn layer的标定，降低标定时间和calib 误差    

## sensitivity layer profile  
* 找到所有 quant layer   
* 每次仅使能一层quant layer进行指标eval，记录到dict中 {"layer_name":eval_value}
* 如果是使用 pytorch-quantization calib，则是在pytorch 下寻找敏感层
* 如果是使用 TensorRT calib，则是在 fp32 下，使用 TensorRT 或 onnxruntime 下寻找   

