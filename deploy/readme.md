## 无自定义node 的onnx    
```shell  
# onnx optimize   
python -m onnxsim src.onnx  model_sim.onnx

polygraphy surgeon sanitize --fold-constants input_model.onnx  -o folded_model.onnx

# check two onnx output  
python run_onnxrt.py     

# onnx2plan
trtexec --onnx=model_sim.onnx --fp16 --verbose --saveEngine=model_sim.plan --useCudaGraph  --dumpProfile --dumpLayerInfo --separateProfileRun | tee log


# onnx vs plan   result align     
## fp32  
polygraphy run model_sim.onnx --trt --onnxrt   

## fp16
polygraphy run model_sim.onnx --trt --onnxrt --atol 0.001 --rtol 0.001 --fp16

## fp16 dynamic batch    
polygraphy run model_sim.onnx --trt --onnxrt  --fp16 \
        --trt-min-shapes x:[1,3,224,224]  y:[1,1,1,224] \
        --trt-opt-shapes x:[4,3,224,224]  y:[4,1,1,224] \
        --trt-max-shapes x:[8,3,224,224]  y:[8,1,1,224]  

```

## 带自定义node的 onnx    
`带自定义node的onnx 是否可以跳过指定layer ?`  或者手动实现 ort op    


```shell 
trtexec --verbose \
--fp16 \
--dumpProfile \
--separateProfileRun \
--loadEngine=bevf-640-1600-48-48_poly.plan \
--plugins=./libplugin_custom.so \
--dumpLayerInfo --profilingVerbosity=detailed  \
--useCudaGraph --exportLayerInfo=bevf-640-1600-48-48_poly.json \
--loadInputs='img':img-6-3-640-1600.bin,'lidar2img':lidar2img-1-6-4-4.bin,'3':prev_bev-1-2304-256.bin \
--exportOutput=bevf-640-1600-48-48_poly_out.json
```


## 复杂网络导出 onnx 方法      

+ 问题定位  
逐步逼近（提前返回）方法      

## 复杂 onnx 导出 plan 方法      
+ 问题定位  
逐步逼近（提前返回）方法--逐渐导出完整onnx 来进行转换        


## 精度问题   
https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/debug_accuracy.md    

## ref  
https://zhuanlan.zhihu.com/p/535021438   
https://zhuanlan.zhihu.com/p/436017991   

https://github.com/pytorch/pytorch/issues/61277    
