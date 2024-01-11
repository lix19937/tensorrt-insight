
动态shape infer 需要在 enqueue 或 execute 之前 进行实时绑定 (动态**输入**绑定即可)      
```cpp
 // Set the input size for the preprocessor
 CHECK_RETURN_W_MSG(mPreprocessorContext->setBindingDimensions(0, inputDims), false, "Invalid binding dimensions.");

 // We can only run inference once all dynamic input shapes have been specified.
 if (!mPreprocessorContext->allInputDimensionsSpecified())
 {
     return false;
 }
```
 https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvinfer1_1_1_i_execution_context.html#a5815b590a2936a3d4066b54f89861a8b  

如 动态batch  

|输入      |输出|  
|-------- | ------|     
|-1 x C x H x W     | -1 x M x N  |       
|-1 x C x H x W <br> 1 x P x Q  | -1 x M x N  |      
|-1 x C x H x W <br> 1 x P x Q  | -1 x M x N <br> 1 x R |   
|-1 x C x H x W <br> 1 x P x Q <br> -1 x K x K| -1 x M x N <br> 1 x R  <br>  -1 x K|   
 
