
# 量化感知训练   

* 带插件的QAT 
  + 在qat onnx生成后替换相应的op   
  + 插件在网络尾部(插件的输出就是网络的输出) ，这时候插件对应的op可以不参与量化训练       

## 示例  
* yolox

* yolov7

* centernet(lidar seg)

* lidar od

* resnet

* hrnet

* hourglass

