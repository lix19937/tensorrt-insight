+ 自定义插件    
index_rebatch_plugin.h    
index_rebatch_plugin.cpp  

+ 导出带int8 输入的 onnx    
sca_with_int8_input.py   

+ trtexec 转换脚本   
build.sh     

--------   

测试发现  --fp16 或 fp32下， onnx 带有int8 或 uint8的不能进行build convert。  
int32的输入是可以进行转换。        
