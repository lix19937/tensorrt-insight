
### 1 [trtexec  --best 过程](./resnet18.log) 

### 2 qat      

```
改造float network graph (insert quant)
              --> 加载 float 收敛的模型权重文件       
                            -->  calib/ ptq  , pth/onnx  fixed scale (act, weight)    
                                         --> qat finetune  , pth/onnx (weight para will update during training)   
```

* 带插件的QAT 
  + 在qat onnx生成后替换相应的op   
  + 插件在网络尾部(插件的输出就是网络的输出) ，这时候插件对应的op可以不参与量化训练       

更多 see https://github.com/lix19937/pytorch-quantization/tree/main/pytorch_quantization/nn   

----------------------

### 3 auto qat    

https://github.com/lix19937/auto_qat    

https://github.com/NVIDIA/TensorRT/issues/3205
https://github.com/NVIDIA/TensorRT/issues/2182       

```
Did you start with a pretrained model w/o QAT? If yes, does the FP32 model (unquantized) also shows instability?
How did you add the QDQ nodes and how did you determine the scales (what SW did you use? did you perform calibration? Was the calibration DS large enough?)?
Did you perform fine-tuning after adding fake quantization? Did you observe the loss vs accuracy curve? Did you check that you did not overfit?
Intuitively I think you should verify that your model is not overfitting because an overfitted model will be unstable when we introduce noise from quantization and limited-precision arithmetic (in float arithmetic different operations ordering can produce small differences in output).
```

### 4 conv + bn 
训练时候不进行融合   

### 5 softmax 如何量化  

### 6 tanh 如何量化  

---------------

### 7 sparse-qat     

-------------

### 8 qad   

------------
 
### 9 function op 怎么量化   
非参数算子（Non-parameterized Operators），它们本身没有权重，其输出依赖于输入数据的缩放因子（Scale）和零点（Zero-point）。   
量化过程主要涉及 量化张量的对齐。  
对齐要求：在进行加法或拼接操作前，所有参与计算的张量必须处于相同的量化空间。这意味着它们的 scale 和 zero_point 必须一致，或者需要通过重量化（Requantization）来强制对齐。

在量化前的观测阶段，框架会强制这些分支使用同一个 Observer 实例，从而确保生成的 scale 和 zero_point 是全局统一的。

