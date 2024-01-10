
动态shape infer 需要在 enqueue 或 execute 之前 进行实时 绑定    
 https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvinfer1_1_1_i_execution_context.html#a5815b590a2936a3d4066b54f89861a8b  

如 动态batch  

|输入      |输出|  
|-------- | ------|     
|-1x C xH x W  <br> vvv | -1 x M x N  |       
 
