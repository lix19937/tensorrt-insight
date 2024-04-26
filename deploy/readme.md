```
python -m onnxsim src.onnx  model_sim.onnx

# onnx2plan
trtexec --onnx=model_sim.onnx --fp16 --verbose --saveEngine=model_sim.plan --useCudaGraph  --dumpProfile --dumpLayerInfo --separateProfileRun | tee log

# onnx vs plan   result align

polygraphy run model_sim.onnx --trt --onnxrt   

```
