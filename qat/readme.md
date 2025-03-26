
# 量化感知训练   

```
改造float network graph (insert quant)
              --> 加载 float 收敛的模型
                            -->  calib/ ptq  , pth/onnx  fixed scale (act, weight)    
                                         --> qat finetune  , pth/onnx (weight para will update during training)   
```

* 带插件的QAT 
  + 在qat onnx生成后替换相应的op   
  + 插件在网络尾部(插件的输出就是网络的输出) ，这时候插件对应的op可以不参与量化训练       

更多 see https://github.com/lix19937/pytorch-quantization/tree/main/pytorch_quantization/nn   

# auto qat    
https://github.com/lix19937/auto_qat    

https://github.com/NVIDIA/TensorRT/issues/3205


https://github.com/NVIDIA/TensorRT/issues/2182       

```
Did you start with a pretrained model w/o QAT? If yes, does the FP32 model (unquantized) also shows instability?
How did you add the QDQ nodes and how did you determine the scales (what SW did you use? did you perform calibration? Was the calibration DS large enough?)?
Did you perform fine-tuning after adding fake quantization? Did you observe the loss vs accuracy curve? Did you check that you did not overfit?
Intuitively I think you should verify that your model is not overfitting because an overfitted model will be unstable when we introduce noise from quantization and limited-precision arithmetic (in float arithmetic different operations ordering can produce small differences in output).
```


## conv + bn 
训练时候不进行融合   

## softmax 如何量化  

## tanh 如何量化  



## 示例  
* yolox

* yolov7

* centernet(lidar seg)

* lidar od

* resnet

* hrnet

* hourglass

