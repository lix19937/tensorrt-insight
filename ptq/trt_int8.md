
## trtexec --int8 流程      

### 环境 
drive os 7030  trt10.10.10     
nv ResNet50.onnx MD5:6753285a2ad03ef7100673ea6bb4312a     

### 1 使用autotune

optimized_final.onnx -->  optimized_final_tofp16.onnx     
```bash
trtexec  --verbose  --stronglyTyped --dumpProfile --separateProfileRun --onnx=./optimized_final_tofp16.onnx --profilingVerbosity=detailed --useCudaGraph --dumpLayerInfo --builderOptimizationLevel=5 --tilingOptimizationLevel=3 --warmUp=500 --duration=15 --useSpinWait --noDataTransfers
```
耗时：   
```
[01/09/2022-20:59:38] [I] GPU Compute Time: min = 0.616211 ms, max = 0.62793 ms, mean = 0.618988 ms, median = 0.619141 ms, percentile(90%) = 0.620117 ms, percentile(95%) = 0.620605 ms, percentile(99%) = 0.621094 ms
```

### 1.1 使用autotune产生的 baseline  
```bash
trtexec --verbose --fp16 --dumpProfile --separateProfileRun --onnx=baseline.onnx --profilingVerbosity=detailed --useCudaGraph --dumpLayerInfo --builderOptimizationLevel=5 --tilingOptimizationLevel=3 --warmUp=500 --duration=15 --useSpinWait --noDataTransfers
```
耗时：   
```
[01/09/2022-22:05:17] [I] GPU Compute Time: min = 0.624023 ms, max = 0.636719 ms, mean = 0.627126 ms, median = 0.626953 ms, percentile(90%) = 0.628418 ms, percentile(95%) = 0.628906 ms, percentile(99%) = 0.629883 ms
```

### 2 --fp16   

```bash
trtexec  --verbose  --fp16 --dumpProfile --separateProfileRun --onnx=./ResNet50.onnx --profilingVerbosity=detailed --useCudaGraph --dumpLayerInfo --builderOptimizationLevel=5 --tilingOptimizationLevel=3 --warmUp=500 --duration=15 --useSpinWait --noDataTransfers
```   
耗时：   
```
[01/09/2022-21:29:26] [I] GPU Compute Time: min = 0.611328 ms, max = 0.627197 ms, mean = 0.617594 ms, median = 0.617676 ms, percentile(90%) = 0.619141 ms, percentile(95%) = 0.620117 ms, percentile(99%) = 0.620605 ms
```

### 3 使用pytorch_quantization    
