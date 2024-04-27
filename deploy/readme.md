```
# onnx optimize   
python -m onnxsim src.onnx  model_sim.onnx

# check two onnx output  
python run_onnxrt.py     

# onnx2plan
trtexec --onnx=model_sim.onnx --fp16 --verbose --saveEngine=model_sim.plan --useCudaGraph  --dumpProfile --dumpLayerInfo --separateProfileRun | tee log


# onnx vs plan   result align     
## fp32  
polygraphy run model_sim.onnx --trt --onnxrt   

## fp16
polygraphy run model_sim.onnx --trt --onnxrt --atol 1e-4  --fp16

## fp16 dynamic batch    
polygraphy run model_sim.onnx --trt --onnxrt  --fp16 \
        --trt-min-shapes x:[1,3,224,224]  y:[1,1,1,224] \
        --trt-opt-shapes x:[4,3,224,224]  y:[4,1,1,224] \
        --trt-max-shapes x:[8,3,224,224]  y:[8,1,1,224]  

```




## ref  
https://zhuanlan.zhihu.com/p/535021438   
