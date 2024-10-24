## 脚本与工具    

|脚本名/目录名| 说明|  
|----  |----|  
|build_from_onnx.py | onnx2plan, export plan_json |   
|np2bin.py| numpy 数据存储为trtexec 可读取的bin文件  |      
|qat2ptq.py| 将qat.onnx 的scale存储为ptq 可读取的calib table |     
|export_onnx_gs.py| by onnx_gs  |       
|export_onnx_torchvision.py|  by torchvision 得到graph|     
|onnx_check.py|onnx有效性检查 |     
|infer_from_engine.py| 使用engine 推理 |   
|dump_network_from_onnx.py| 将onnx 通过trt build 保存为json |   
|list_plugins.py| 列出trt 中支持的plugin|    
|trt_batch_onnx_build.sh| 指定精度和名称进行批量trt build|  
|   |   |  
| - | - |    
|benchmark  | trt 性能测试     |    
|plugins    | trt 自定义插件参考模板   |    
|IDA        | 模型逆向         |    
