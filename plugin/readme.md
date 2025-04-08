# 解读官方OSS 插件细节与优化对比        

对 TensorRT-release-x.y/plugin/ 进行补充    


## 注意  

插件的io format 也很重要     

插件寻找与创建 Successfully created plugin:       
```
[10/09/2024-19:56:45] [V] [TRT] Searching for input: input
[10/09/2024-19:56:45] [V] [TRT] node_of_output [PixelShufflePlugin] inputs: [input -> (1, 256, 60, 36)[FLOAT]],
[10/09/2024-19:56:45] [I] [TRT] No importer registered for op: PixelShufflePlugin. Attempting to import as plugin.
[10/09/2024-19:56:45] [I] [TRT] Searching for plugin: PixelShufflePlugin, plugin_version: 1, plugin_namespace:
[10/09/2024-19:56:45] [I] [TRT] Successfully created plugin: PixelShufflePlugin
[10/09/2024-19:56:45] [V] [TRT] Registering layer: node_of_output for ONNX node: node_of_output
[10/09/2024-19:56:45] [V] [TRT] Registering tensor: output_0 for ONNX tensor: output
[10/09/2024-19:56:45] [V] [TRT] node_of_output [PixelShufflePlugin] outputs: [output -> (1, 64, 120, 72)[FLOAT]],
```

## 示例  
[Detr3d transformer decoder plugin](./svt/svt.md)    

https://github.com/lix19937/trt-samples-for-hackathon-cn/tree/master/cookbook/05-Plugin    


## todo      
开发一个插件单元测试工具    
