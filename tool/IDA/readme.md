
+ 如果是 engine/plan文件 

+ 如果是 protobuf bin文件    
```shell   
protoc --decode_raw < uni_model_fusion_l2_engine_graph.bin 2>&1 | tee uni.json
```

+ 如果是 prototxt 文件  
onnx文件是可以导出为prototxt文件  

+ 如果是 dla文件  



https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md   
