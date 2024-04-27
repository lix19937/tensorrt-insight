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

polygraphy run model_sim.onnx --trt --onnxrt  --fp16 \
        --trt-min-shapes x:[1,2,28,28]  y:[1,1,28] \
        --trt-opt-shapes x:[4,2,28,28]  y:[4,1,28] \
        --trt-max-shapes x:[8,2,28,28]  y:[8,1,28]  

```




## ref  
https://zhuanlan.zhihu.com/p/535021438   
