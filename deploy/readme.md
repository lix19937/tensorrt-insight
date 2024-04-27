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
        --trt-min-shapes X:[1,2,28,28] \
        --trt-opt-shapes X:[1,2,28,28] \
        --trt-max-shapes X:[1,2,28,28]    

```




## ref  
https://zhuanlan.zhihu.com/p/535021438   
