```
# onnx optimize   
python -m onnxsim src.onnx  model_sim.onnx

# check two onnx output  


# onnx2plan
trtexec --onnx=model_sim.onnx --fp16 --verbose --saveEngine=model_sim.plan --useCudaGraph  --dumpProfile --dumpLayerInfo --separateProfileRun | tee log


# onnx vs plan   result align     
## fp32  
polygraphy run model_sim.onnx --trt --onnxrt   

## fp16
polygraphy run model_sim.onnx --trt --onnxrt --atol 1e-4  --fp16



```




## ref  
https://zhuanlan.zhihu.com/p/535021438   
