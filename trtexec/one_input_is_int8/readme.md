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

```
[06/22/2024-21:01:52] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +1225, GPU +338, now: CPU 2397, GPU 1674 (MiB)
[06/22/2024-21:01:52] [V] [TRT] Trying to load shared library libcudnn.so.8
[06/22/2024-21:01:52] [V] [TRT] Loaded shared library libcudnn.so.8
[06/22/2024-21:01:52] [V] [TRT] Using cuDNN as plugin tactic source
[06/22/2024-21:01:53] [V] [TRT] Using cuDNN as core library tactic source
[06/22/2024-21:01:53] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +233, GPU +52, now: CPU 2630, GPU 1726 (MiB)
[06/22/2024-21:01:53] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[06/22/2024-21:01:53] [V] [TRT] Constructing optimization profile number 0 [1/1].
2024-06-22 21:01:53.717815: T index_rebatch_plugin.cpp:66] clone
2024-06-22 21:01:53.717825: T index_rebatch_plugin.cpp:316] initialize
2024-06-22 21:01:53.726751: T index_rebatch_plugin.cpp:153] supportsFormatCombination
2024-06-22 21:01:53.726776: T index_rebatch_plugin.cpp:153] supportsFormatCombination
2024-06-22 21:01:53.726780: T index_rebatch_plugin.cpp:153] supportsFormatCombination
2024-06-22 21:01:53.726794: I index_rebatch_plugin.cpp:170] supportsFormatCombination  +++ desc.type :0
2024-06-22 21:01:53.726797: T index_rebatch_plugin.cpp:153] supportsFormatCombination
2024-06-22 21:01:53.726801: I index_rebatch_plugin.cpp:170] supportsFormatCombination  +++ desc.type :0
2024-06-22 21:01:53.726803: T index_rebatch_plugin.cpp:153] supportsFormatCombination
2024-06-22 21:01:53.726814: T index_rebatch_plugin.cpp:153] supportsFormatCombination
2024-06-22 21:01:54.105838: T index_rebatch_plugin.cpp:323] terminate
2024-06-22 21:01:54.116580: T index_rebatch_plugin.cpp:323] terminate
[06/22/2024-21:01:54] [E] Error[9]: [pluginV2Builder.cpp::reportPluginError::23] Error Code 9: Internal Error (/SCA_IndexRebatch_TRT: could not find any supported formats consistent with input/output data types)
[06/22/2024-21:01:54] [E] Error[2]: [builder.cpp::buildSerializedNetwork::743] Error Code 2: Internal Error (Assertion engine != nullptr failed. )
[06/22/2024-21:01:54] [E] Engine could not be created from network
[06/22/2024-21:01:54] [E] Building engine failed
[06/22/2024-21:01:54] [E] Failed to create engine from model or file.
[06/22/2024-21:01:54] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v8510] # trtexec --onnx=./sca.onnx --plugins=./libplugin_custom.so --verbose --inputIOFormats=fp32:chw,fp32:chw,int8:chw --outputIOFormats=int32:chw,int32:chw,fp32:chw,fp32:chw,fp32:chw
2024-06-22 21:01:54.124162: T index_rebatch_plugin.cpp:323] terminate
```
