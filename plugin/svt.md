```
1 插件实现  
1.1 插件组成   
1.1.1 svt overview
1.1.2 decoder pipeline (mha --> mha_norm --> svca --> svca_norm --> ffn --> ffn_norm --> reg --> update | --> mha --> mha_norm ... )
1.1.3 svt dummy node in onnx (SvTransformerDecoder: 14 configuration parameter, 160 weight/bias, total 174 attributes)
1.2 kernel融合
1.2.1 纵向: 同一条数据流中操作，elementwise   
1.2.2 横向: 相互操作独立，不同数据流或同一数据流中运算数据无依赖
1.2.3 访存: 减少内存移动    
1.3 高维矩阵乘除法交换与乘法降维        
1.4 cudagraph应用  
1.5 backbone maxpool融合  
1.6 free reformat 
1.6.1 reformatting copynode的产生  
1.6.2 free reformatting的实现 
2 插件封装   
2.1 超参数据储存与加载       
2.1.1 将超参数作为插件的输入，存储到graph.initializer中，每一次迭代都作为只读参数输入给插件    
2.1.2 将超参数作为插件的属性，以const类型存储到value info中，方便大批量权重参数按统一方式设置，减少了插件的输入tensor数目，因此在svt中优先采用
2.2 运行时同时支持fp32、half和int8     
2.3 fake-int8支持    
3 插件联调
3.1 identify layer    
3.2 带插件PTQ  
4 其它
4.1 sigmoid函数加速    
4.1.1 线性逼近
4.2 backbone中slices sampling等价替换 
4.3 permute操作转换辅助函数             
4.4  拓展的torch代码
参考
```

## 1 插件实现    
### 1.1 插件组成    
#### 1.1.1 svt overview     
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/3ca42737-1264-42a4-a9cc-aa5d42c1dadf)   

#### 1.1.2 decoder pipeline (mha --> mha_norm --> svca --> svca_norm --> ffn --> ffn_norm --> reg --> update | --> mha --> mha_norm ... )     
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/69dc3afa-77e2-4a3f-a157-35300612c8c2)     

#### 1.1.3 svt dummy node in onnx (SvTransformerDecoder: 14 configuration parameter, 160 weight/bias, total 174 attributes)     
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/bd35fc7d-6ebd-4105-a36d-92bd21e85f83)     

### 1.2 kernel融合    
#### 1.2.1 纵向: 同一条数据流中操作，elementwise     
Decoder block数据流:
```
| mha --> mha_norm --> svca --> svca_norm --> ffn --> ffn_norm --> reg --> update | [loop] |--> cls    
```    
在各功能单元中可进行拆解然后融合，如使用
svexp::invokeGeneralAddBiasResidualPreLayerNorm与invokeGeneralAddBiasResidualPreLayerNorm进行了跨阶段操作的合并工作，具体如下:   
```
mha_norm_in = mha_out_without_bias + residual + bias + query_pos
svca_norm_in = svca_out_without_bias + residual + bias + svca_pos_feat
ffn_norm_in = ffn_out_without_bias + svca_norm_out_buf + bias   
```

如shape型op与elementwise合并:
```py
import torch 
import numpy as np
from loguru import logger

rand_seed = 123456
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)

# original
num_cam = 6
num_query = 512
reference_points_cam = torch.randn(1, num_cam, 4, num_query)
reference_points_cam_p = reference_points_cam.permute(0, 1, 3, 2) 
mask = reference_points_cam_p[..., 2:3] > 1e-2  # [1, 6, 512, 1]

# equivalent_transformation, cuda version is written in python mode   
h = 4
w = num_query
mask_v1 = (torch.zeros(1, num_cam, num_query, 1) > 1).view(1,-1).squeeze()

reference_points_cam_flatten = reference_points_cam.view(1,-1).squeeze()
for idx in range(num_cam):
  for col_idx in range(w):  
    if reference_points_cam_flatten[idx * (h*w) + 2*w + col_idx] > 1e-2:
      mask_v1[idx * w + col_idx] = True

mask_v1 = mask_v1.view(1, num_cam, num_query, 1)
logger.info('{}'.format(torch.equal(mask, mask_v1)))

# equivalent_transformation, commutative law
mask_v2 = reference_points_cam[..., 2:3, :] > 1e-2
mask_v2 = mask_v2.permute(0, 1, 3, 2)
logger.info('{}'.format(torch.equal(mask, mask_v2)))
```

#### 1.2.2 横向: 相互操作独立，不同数据流或同一数据流中运算数据无依赖   
```
 svca.output_proj(@stream1)      
                                     |--> (@stream1) 
 svca.position_encoder(@stream2)
```

即分两个流:一个流用来计算svca.output_proj操作，另一个流计算svca.position_encoder 操作，最后在主流（@stream1）中同步。   

#### 1.2.3 访存: 减少内存移动      
如采用half2:   
```cpp 
template<typename T>
__global__ void svt_add_bias_slice(
    T* __restrict__ out, const T* __restrict__ in, const T* __restrict__ bias, const int m, const int n, const int s, const bool on_top = true){
    // s = slice_cnt, input_slice_size = s*n,
    // n = element size after concate axis. for example: [2,3,4,5] concate_axis = 1, then n = 4*5=20
    // on_top: concate slice on top or not.
    const int offset = on_top ? 1 : 0;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        int slice_id = id / (s * n);
        out[id + (slice_id + offset) * n] = __ldg(&in[id]) + __ldg(&bias[id % n]);
    }
}

template<>
__global__ void svt_add_bias_slice(
    half* __restrict__ out, const half* __restrict__ in, const half* __restrict__ bias, const int m, const int n, const int s, const bool on_top){
    const int offset = on_top ? 1 : 0;
    const auto in_ptr = (half2*)in;
    const auto bias_ptr = (half2*)bias;
    auto out_ptr = (half2*)out;

    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
        half2 d1 = __ldg(&in_ptr[id]);
        half2 d2 = __ldg(&bias_ptr[id % n]);
        int slice_id = id / (s * n);
        out_ptr[id + (slice_id + offset) * n] = __hadd2(d1, d2);
    }
}

template<typename T>
void SvtAddBiasSlice(T* in, T* out, const T* bias, const int m, const int n, const int s, cudaStream_t stream){
    const int data_type_factor = 4 / sizeof(T);  // 1 for fp32; 2 for fp16
    dim3 block, grid;
    if (n / 4 / data_type_factor <= 1024) {
        block.x = n / 4 / data_type_factor;
        grid.x = m;
    }
    else {
        block.x = 1024;
        grid.x = (m * n + 1023) / 1024;
    }
    svt_add_bias_slice<<<grid, block, 0, stream>>>(in, out, bias, m, n / data_type_factor, s);
}
```

### 1.3 高维矩阵乘除法交换与乘法降维        
```
原始版本:
[1, 6, 512, 4, 4] * [1, 6, 512, 4, 1] / [1, 6, 1, 4]

乘除法交换后版本:
[1, 6, 4, 4] / [4, 1] 
[24, 4] * [4, 512]
```

两个版本对应的代码如下:   
```py
import torch
import numpy as np 
from loguru import logger

def oringinal(lidar2img, img_shape, reference_points):
  B, num_query = reference_points.size()[:2]
  num_cam = lidar2img.size(1)
  reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
  lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
  img_shapes = lidar2img.new_tensor([img_shape[0][1], img_shape[0][0], 1, 1])[None, None, None, :].repeat(B, num_cam, 1, 1)
 
  reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1) / img_shapes # mul div
  return reference_points_cam

# equivalent_transformation, follow python api can be implemented through cuda, and then fused with other op     
def opti(lidar2img, img_shape, reference_points):
  _, num_query = reference_points.size()[:2]
  num_cam = lidar2img.size(1)
  reference_points = reference_points.view(-1, 4).permute(1, 0)
  img_shapes = torch.tensor([[img_shape[0][1]], [img_shape[0][0]], [1.], [1.]])

  lidar2img = (lidar2img/img_shapes).view(-1, 4) # div 
  reference_points_cam = torch.matmul(lidar2img, reference_points) # mul

  reference_points_cam = reference_points_cam.view(1, num_cam, 4, num_query).permute(0, 1, 3, 2) 
  return reference_points_cam

def main():
  num_query = 512; num_cam = 6
  reference_points = torch.rand(1, num_query, 4)
  lidar2img = torch.rand(1, num_cam, 4, 4)
  img_shape = torch.tensor([[288, 736]])

  base = oringinal(lidar2img, img_shape, reference_points).view(1, -1).squeeze().numpy()
  ref = opti(lidar2img, img_shape, reference_points).contiguous().view(1, -1).squeeze().numpy()
  
  # check 
  abs_diff = (np.abs(np.array(ref) - np.array(base))).tolist()
  max_abs_diff = np.max(abs_diff)
  min_abs_diff = np.min(abs_diff)
  mean_abs_diff = np.mean(abs_diff)
  std_abs_diff = np.std(abs_diff)
  tatic_str = "MinAbsDiff = %7.4f, MaxAbsDiff = %7.4f, MeanAbsDiff = %7.4f, StdAbsDiff = %7.4f" %(min_abs_diff, max_abs_diff, mean_abs_diff, std_abs_diff)
  logger.info("{}".format(tatic_str))

if __name__ == '__main__':
  main()
```

### 1.4 cudagraph应用  
主要是解决模型运行的launch bound问题，TRT build infer without cudagraph与with cudagraph对比如下:       
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/61d0f9d5-a639-4f26-922f-32aba6c20b99)

注意:       
1, 在运行时（每一次迭代）没有阻塞式cuda api，如cudaMemcpy/cudaMalloc/cudaMemset等；    
2, 全部使用非默认流；    
3, 流之间同步采用流派生和事件机制。        

### 1.5 backbone maxpool融合  
默认with maxpool（MaxPool由1个Conv操作输出导入）与without maxpool（MaxPool融于到插件中），两者的onnx片段如下:    
 
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/0fb83d6f-65f0-4709-ad86-30c66eb219b8)

MaxPool融合到format转换过程中，完全消除了传统滑窗法“Z”字型全局遍历找到pool目标（MaxPool计算完成后，再进行下一步逻辑，阻塞式串行），优化后（在取值进行MaxPool的同时进行下一步逻辑），具体实现如下代码片段:  
```cpp    
__forceinline__ __device__ void
linear_to_convertchw32_maxpool2d(const size_t idx, const size_t area, const size_t w, const int8_t* __restrict__ input, const float scale, half* __restrict__ value){
    const auto AREA = area << 2;  /// 4 * area, W = 2 * w, before pool
    const auto raw_idx = (idx / area * AREA) + ((idx % area / w) << 1) * (w << 1) + (idx % area % w << 1);

    size_t dst_idx;
    // api of linear_convert_to_chw32plane see section1.6
    linear_convert_to_chw32plane(raw_idx, AREA, &dst_idx);
    *value = __ldg(&input[dst_idx]) * scale;
}
```

## 1.6 free reformat 
#### 1.6.1 reformatting copynode的产生   
```
TensorRT optimizes a network using many different data formats. In order to allow efficient passing of data between TensorRT and a client application, 
these underlying data formats are exposed at network I/O boundaries, that is, for Tensors marked as network input or output, and when passing data to and from plug-ins. 
For other tensors, TensorRT picks formats that result in the fastest overall execution, and may insert reformats to improve performance.
You can assemble an optimal data pipeline by profiling the available I/O formats in combination with the formats most efficient for the operations preceding and following TensorRT.
```
如果采用默认方式，TRT基于全局最优性能自动增加了Reformatting CopyNode，即从FP16 NC/32HW32到FP16 NCHW的layout转换，如下所示:      

![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/e7d140c8-5fda-442b-9737-51ac5a70fdb2)  

free reformat v1与free reformat v2分别如下所示:  
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/cf14d2e4-e1d2-4f98-b907-83f8c67123f1)

![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/be9bfe7d-6e26-4de5-9d9c-db3b46dcdb02)

#### 1.6.2 free reformatting的实现 
kCHW32与kLinear数据分布分别如下:     
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/48a06bc2-cfce-43a4-a654-b2fcc46f1cd8) ![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/7f8f2c22-584d-46df-8e3c-519fe366d830)


kCHW32与kLinear索引转换函数如下:   

```cpp  
/// NCHW --> NC/32HW32,  idx is in linear plane, dst_idx(chw32_idx) is in NC/32HW32 plane
/// Now find *idx(value in idx pos) through chw32_idx 
__forceinline__ __device__ void linear_convert_to_chw32plane(const size_t idx, const int h, const int w, size_t* __restrict__ dst_idx){
    //////////////////////////////// python snippet ////////////////////////////
    // area = H * W
    // voc = 32*area
    // # idx is linear plane, 
    // voc_idx = idx // voc   
    // row_idx = idx % area
    // col_idx = idx // area % 32
    // chw32_idx = voc_idx * voc + row_idx*32 + col_idx  # find idx in NCHW32 plane 
    // dst[idx] = src[chw32_idx]
    ////////////////////////////////////////////////////////////////////////////

    const size_t area = h * w, voc = area << 5;
    *dst_idx = (idx / voc * voc) + (idx % area << 5) + (idx / area & 31);
}
```

对于shape为288*736的模型包含3次reformat，在Orin OS6040 TRT8410环境下，会产生1.9ms左右的耗时，通过linear_convert_to_chw32plane函数实现free reformat，从而消除了这一阶段的耗时。 
## 2 插件封装    
### 2.1 超参数据储存与加载          
插件中如果有超参数（hyperparameter，权重和偏置参数）参与运算，需要将这些只读常量进行存储，有两种方法:作为输入的initializer和作为属性的atttributes。     
#### 2.1.1 将超参数作为插件的输入，存储到graph.initializer中，每一次迭代都作为只读参数输入给插件       
（下图只选取两个超参数block_1__cross_attention__attention_weights__fc__weight, block_1__cross_attention__attention_weights__fc__bias）     

![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/37ce6bc8-0b75-4a74-be89-73e1a8dbfd2b)

#### 2.1.2 将超参数作为插件的属性，以const类型存储到value info中，方便大批量权重参数按统一方式设置，减少了插件的输入tensor数目，因此在svt中优先采用
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/961e1272-610a-4d98-97ff-cda44402a667)    

### 2.2 运行时同时支持fp32、half和int8     
插件的输入输出数据类型是在  
```
int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
  const nvinfer1::PluginTensorDesc* outputDesc,
  const void* const* inputs,
  void* const* outputs,
  void* workspace,
  cudaStream_t stream) noexcept override 
``` 
中inputDesc和outputDesc参数得到的，即运行时才能知道当前tensor数据类型是fp32、half、int8、int32或其它，因此在createPlugin函数中需要创建支持多种数据类型的plugin实现，另外在supportsFormatCombination函数中如下设置:
```cpp 
bool SVTransformerPlugin::supportsFormatCombination(
 int pos,
 const PluginTensorDesc* inOut,
 int nbInputs,
 int nbOutputs) noexcept{
 bool res{false};
 assert(pos >= 0 && pos < SV_PLUGIN_IN_NUM + SV_PLUGIN_OUT_NUM);
 switch (pos) {
   case 0:
   case 1:
   case 2:
     res =
      (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) || 
      (inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == nvinfer1::TensorFormat::kCHW32)   || 
      (inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == nvinfer1::TensorFormat::kCHW32);
     break;

   case 3:
     res = inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
     break;

   case 4:
   case 5:
     res = inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
     break;
   default:
     break;
 }

 return res;
}
```

### 2.3 fake-int8支持    
fake-int8: 在enqueue入口处传入的数据是int8数据类型，而enqueue内部操作流完全使用的是half和fp32，输出也是half和fp32，这样允许上一层选择int8精度运算，同时当前层不掉点，从而提高性能又保证了准确率，代码片段如下:  
```cpp 
int SVTransformerPlugin::enqueue(
 const PluginTensorDesc* inputDesc,
 const PluginTensorDesc* outputDesc,
 const void* const* inputs,
 void* const* outputs,
 void* workspace,
 cudaStream_t stream) noexcept{
 // init some var
 const size_t batch_size{1};
 const size_t ch{inputDesc[0].dims.d[1]};
 check_cuda_error(cublasSetStream(cublas_handle_, stream));

 if (inputDesc[0].type == nvinfer1::DataType::kINT8 && 
     inputDesc[1].type == nvinfer1::DataType::kINT8 && 
     inputDesc[2].type == nvinfer1::DataType::kINT8) {
   typedef int8_t DT;

   std::vector<Tensor> input_tensors{
     Tensor{MEMORY_GPU,
       getTensorType<DT>(),
       {settings_.num_cam, ch, size_t(inputDesc[0].dims.d[2]), size_t(inputDesc[0].dims.d[3])},
       (const DT*)(inputs[0]),
       inputDesc[0].scale},
     Tensor{MEMORY_GPU,
       getTensorType<DT>(),
       {settings_.num_cam, ch, size_t(inputDesc[1].dims.d[2]), size_t(inputDesc[1].dims.d[3])},
       (const DT*)(inputs[1]),
       inputDesc[1].scale},
     Tensor{MEMORY_GPU,
       getTensorType<DT>(),
       {settings_.num_cam, ch, size_t(inputDesc[2].dims.d[2]), size_t(inputDesc[2].dims.d[3])},
       (const DT*)(inputs[2]),
       inputDesc[2].scale},
     Tensor{MEMORY_GPU,
       getTensorType<float>(),
       {settings_.num_cam, settings_.l2i_matr_h, settings_.l2i_matr_w},
       (const float*)(inputs[3])}};

   std::vector<Tensor> output_tensors{
     Tensor{MEMORY_GPU,
       getTensorType<float>(),
       {batch_size, settings_.seq_len, settings_.num_reg_points},
       (float*)(outputs[0])},
     Tensor{MEMORY_GPU,
       getTensorType<float>(),
       {batch_size, settings_.seq_len, settings_.num_classes},
       (float*)(outputs[1])}};
   sv_transformer_->forward(&output_tensors, &input_tensors, stream);
 }
 else {
   // make sure this branch is not selected, we intentionally increase the latency
   std::this_thread::sleep_for(std::chrono::milliseconds(128)); 

   check_cuda_error(cudaMemsetAsync((float*)(outputs[0]), 0x00, batch_size*settings_.seq_len*settings_.num_reg_points, stream)); 
   check_cuda_error(cudaMemsetAsync((float*)(outputs[1]), 0x00, batch_size*settings_.seq_len*settings_.num_classes, stream)); 
 }

 return 0;
}

// dq: int8 --> fp32
mul_tmp = __hmul(__float2half(__ldg(&input[dst_idx]) * dq_scale_1), se);
mul_tmp = __hmul(__float2half(__ldg(&input[dst_idx]) * dq_scale_2), sw);   
```

## 3 插件联调
### 3.1 identify layer    
```
If the output type is explicitly specified via setOutputType, IIdentityLayer can be used to convert from one type to another. 
Other than conversions between the same type (kFLOAT -> kFLOAT for example), the only valid conversions are:

(kFLOAT | kHALF | kINT32 | kBOOL) -> (kFLOAT | kHALF | kINT32)
Conversion also happens implicitly, without calling setOutputType, if the output tensor is a network output.

Two types are compatible if they are identical, or are both in {kFLOAT, kHALF}. Implicit conversion between incompatible types, 
i.e. without using setOutputType, is recognized as incorrect as of TensorRT 8.4, but is retained for API compatibility within TensorRT 8.x releases. 
In a future major release the behavior will change to record an error if the network output tensor type is incompatible with the layer output type. 
E.g., implicit conversion from kFLOAT to kINT32 will not be allowed, and instead such a conversion will require calling setOutputType(DataType::kINT32).
```
without identify layer与with identify layer的onnx节点图如下:     

![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/98308909-0144-4e88-b1ec-d78e9a0dab15) ![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/671c56d5-6151-4382-8813-e092aa370176)


identify layer起到占位符的作用，方便进行格式转换和网络模块构建时数目保持相同。
```
batch_norm = nn.BatchNorm2d
if dont_use_batch_norm:
    batch_norm = Identity


nn.Sequential(
    ...
    batch_norm(N, momentum=0.05),
    ...
)

nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) if _kernel is not None else nn.Identity()
```   
TRT8410 trtexec在build engine如果没有增加Identify layer会报以下错误:     
```
"Error[2]: [optimizer.cpp::getFormatRequirements::2291] Error Code 2: Internal Error (Assertion !n->candidateRequirements.empty() failed. no unquantized formats available)" 
```   
注:     
TRT8411 trtexec已可以直接使用without identify layer版本   

### 3.2 带插件PTQ  
INT8量化的本质是一种缩放（scaling）操作，通过缩放因子将模型的分布值从FP32范围缩放到INT8范围之内，因此必须实现FP32版本的插件跑完整个网络，PTQ会迭代若干次FP32，找到最佳的阈值|T|，使得kl_divergence最小（或余弦相似度最大，或percentile_0.999）。 
以kl_divergence 为例:   
```py   
from scipy.special import rel_entr

# define two probability distributions
P = [.05, .1, .2, .05, .15, .25, .08, .12]
Q = [.23, .1, .2, .12, .14, .02, .09, .11]

# calculate KL(P || Q)
print('kl:', sum(rel_entr(P, Q))) #0.522723562143904
```   
带插件PTQ和不带插件PTQ原则上结果应一致，插件看作单独一层PluginV2，插件的输出会进行scale计算，并存入calib table。如果plugin 是3输入2输出，全int8类型，plugin输入时，先反量化dq，plugin输出时，进行量化q，如下:

```cpp 
const float dq_scale_1 = inputDesc[0].scale;
const float dq_scale_2 = inputDesc[1].scale;
const float dq_scale_3 = inputDesc[2].scale;

const float q_scale_1 = 1.f / outputDesc[0].scale;
const float q_scale_2 = 1.f / outputDesc[1].scale;

inline __device__ void quantize(const float x, const float q_scale, char * __restrict__ tmpq8){
    int tmpq = __float2int_rn(q_scale * x);  // scale and round
    *tmpq8 = min(127, max(-127, tmpq)); // clip and cast
}

inline __device__ void dequantize(const int8_t x, const float dq_scale, half * __restrict__ tmpdq16){
    *tmpdq16 = dq_scale * x;  // scale 
}

// some trick for dequantize process  
inline __device__ void float4_to_char4(uint32_t * __restrict__ dst, 
                                           const float x,
                                           const float y,
                                           const float z,
                                           const float w) {
// nv-orin  87  cuda114
// rtx 3070 86  cuda114
// rtx 3090 80  cuda114 
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
  uint32_t a; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(a) : "f"(x));
  uint32_t b; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(b) : "f"(y));
  uint32_t c; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(c) : "f"(z));
  uint32_t d; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(d) : "f"(w));

  asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2,  0;\n" : "=r"(*dst) : "r"(d), "r"(c));
  asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %0;\n" : "+r"(*dst) : "r"(b), "r"(a));
#else
  char4 tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  tmp.w = w;
  *dst = reinterpret_cast<const uint32_t&>(tmp);
#endif
}

// make sure VPT is multiple of 4  
static_assert(VPT % 4 == 0, "make sure VPT is multiple of 4 !");
uint32_t out_local[VPT/4];
#pragma unroll
for (int it = 0; it < VPT / 4; ++it)    {
     auto step = it << 2;
     const float tmp0 = out[step];
     const float tmp1 = out[step+1];
     const float tmp2 = out[step+2];
     const float tmp3 = out[step+3];
     float4_to_char4(&out_local[it], tmp0 * q_scale_1, tmp1 * q_scale_1, tmp2 * q_scale_1, tmp3 * q_scale_1);
}

//  uint32_t --> int8_t
copy<sizeof(int8_t) * VPT>(out_local, &output[idx]);
```

## 4 其它  
### 4.1 sigmoid函数加速      
#### 4.1.1 线性逼近      
```cpp 
// gt version 
__forceinline__ __device__ void sigmoid(const float x, float* __restrict__ y){
    *y = 1.f / (1.f + __expf(-x));
}

__forceinline__ __device__ void fast_sigmoid(const float x, float* __restrict__ y){
    const auto ax = fabsf(x);
    *y = 0.5 * (__fdividef(x, (__fdividef(0.975, ax) + ax) + 1.f));
}
```

fast sigmoid与sigmoid分布、误差对比图:    
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/b94fca78-c0a6-4cc9-9531-690fe5ed79d3)

![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/8e323a6c-6edd-4941-9cac-0fe223a250de)   

#### 4.1.2 反函数    
sigmoid函数实现，进行恒等处理:![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/1cccab48-35d4-401e-af08-f70c231b6d16) 从而得到inverse_sigmoid。由于inverse_sigmoid函数在定义域内严格单调递增

![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/7c46d021-fe24-4f80-bc6c-9dc64adc0b2f)

可知下面两个if判断等价:  
```cpp
// make sure (1/y - 1) > 0
inline void inverse_sigmoid(const float y){
  return -log(1/y - 1);
}

if (sigmoid(x) > threshold){

}  

// usually threshold is constant var, and inverse_sigmoid(threshold) exec once   
if (x > inverse_sigmoid(threshold)){

}
```

只需要离线将inverse_sigmoid(threshold)计算好即可，从而完全规避了exp和div这类耗时严重的数学运算，只有一个比较运算，因此可显著降低运算时间。       
注:       
仅适用于sigmoid的输出与常量进行比较的情形（对于网络若干输出tensor的产生来自sigmoid的输出，并且在decode中存在与阈值比较）；      
只要反函数定义域内严格单向单调的，即可推广适用，logsigmoid、softmax、logsoftmax、softmin、tanh、softplus、selu可同理优化； 

tanh函数可展开为:tanh(x) = 2sigmoid(2x) − 2，与2sigmoid可等效处理。   

### 4.2 backbone中slices sampling等价替换     

8slices + concat(EE + OE + EO + OO)与reshape + permute的等价替换onnx节点图:    
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/ea0e528d-7074-469f-a935-5f764e015b04)

代码验证如下:      
```py 
from loguru import logger as LOG
import torch

# original  
def img_slice(img_feature):
  B,C,H,W =img_feature.shape
  la = img_feature[:,:,0::2, 0::2 ] # E E   H W
  lb = img_feature[:,:,0::2, 1::2 ] # E O

  lc = img_feature[:,:,1::2, 0::2 ] # O E
  ld = img_feature[:,:,1::2, 1::2 ] # O O
  m = torch.cat((la, lc, lb, ld), dim=1)
  return m

# equivalent_transformation
def img_slice_convert():  
  img_feature = torch.arange(0, 16).view(4,4)
  H, W = img_feature.shape 
  a = img_feature.view(H//2, 2, W//2, 2)

  LOG.info("--0-->>\n{}".format(a))
  LOG.info("--1-->>\n{}".format(a.permute(2, 3, 0, 1)))
  LOG.info("--2-->>\n{}".format(a.permute(2, 3, 0, 1).permute(3, 1, 2, 0)))  
  LOG.info("--3-->>\n{}".format(a.permute(2, 3, 0, 1).permute(3, 1, 2, 0).permute(1, 0, 2, 3)))  
  
  v1 = a.permute(2, 3, 0, 1).permute(3, 1, 2, 0).permute(1, 0, 2, 3)
  
  # permute obey merge rule  
  v2 = a.permute(1, 3, 0, 2).permute(1, 0, 2, 3)
  
  # further merge  
  v3 = a.permute(3, 1, 0, 2)
    
  if not (torch.equal(v1, v2) and torch.equal(v1, v3)):         
    LOG.info("fatal, not reach here !"); exit(1)  
  
  B = 1
  C = 1
  e = v1.reshape(B, C*4, H//2, W//2) 
```

分步处理结果如下:     
![image](https://github.com/lix19937/tensorrt-insight/assets/38753233/28fd3313-0c89-4e58-89ed-98a877b6b655)    

### 4.3 permute操作转换辅助函数 
```cpp 
// get strides
inline void InitStrides(const int* __restrict__ dims, const int num_dims, int* __restrict__ stride) {
  stride[num_dims - 1] = 1;
  for (int i = num_dims - 2; i >= 0; --i) {
    stride[i] = dims[i + 1] * stride[i + 1];
  }
}

// convert 1d offset to ND index
inline void OffsetToNDIndex(int* __restrict__ index, const int offset, const int num_dims, const int* __restrict__ stride) {
  int remaining = offset;
  for (int i = 0; i < num_dims - 1; ++i) {
    const int idx = remaining / stride[i];
    index[i] = idx;
    remaining = remaining - idx * stride[i];
  }
  index[num_dims - 1] = remaining;
}

// convert ND index to 1d offset 
inline void NDIndexToOffset(
    int* __restrict__ offset, const int* __restrict__ index, const int num_dims, const int* __restrict__ stride) {
  *offset = 0;
  for (int i = 0; i < num_dims - 1; ++i) {
    *offset += index[i] * stride[i];
  }
  *offset += index[num_dims - 1];
}

// exec once before PermuteIdx2Idx
void GetStrides(int src_stride[], int dst_stride[], const int src_dims[], const int dst_dims[], const int num_dims) {
InitStrides(src_dims, num_dims, src_stride);
  InitStrides(dst_dims, num_dims, dst_stride);
}

// follow api can be used in kernel and fused with other op
void PermuteIdx2Idx(
    int* src_offset,
    int src_index[], // workspace buff, will be rewritten
    int dst_index[], // workspace buff, will be rewritten
    const int permutation[],
    const int src_strides[],
    const int dst_strides[],
    const int num_dims,
  const int i // i is in ordered i=0, 1, 2, 3， ...
) {
  OffsetToNDIndex(dst_index, i, num_dims, dst_strides);

  for (int k = 0; k < num_dims; ++k) {
    src_index[permutation[k]] = dst_index[k];
  }

  NDIndexToOffset(src_offset, src_index, num_dims, src_strides);
}
```

### 4.4  拓展的torch代码   
```py
import torch 

num_gt = 10
a = torch.randn(6, 512)
cost = torch.randn(10, 1024)
dynamic_ks = torch.randint(0, 512, (1024,))
expanded_strides = torch.randn(1024)

# original  repeat之后数量就增多了，repeat会分配内存和拷贝数据，sigmoid_放后面增加了运算量
a.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
a.float().sigmoid_().unsqueeze(0).repeat(num_gt, 1, 1)

# original item()函数会将gpu的数据转换为python的数据，但不要每个数据都去调用一次，如果每个数据都要转，调用tolist()函数对整个tensor做转换
for gt_idx in range(num_gt):
    print('k:', dynamic_ks[gt_idx].item())
    _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)

ks = dynamic_ks.tolist()
for gt_idx in range(num_gt):
    _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)

grid_shape = 64
stride_this_level = 4
expanded_strides = []
# original Tensor的创建:先对创建的Tensor清0，再填充为stride_this_level，然后做类型转换，其实可以一步做完
expanded_strides.append(torch.zeros(1, grid_shape).fill_(stride_this_level).type_as(cost[0]))
expanded_strides.append(torch.full((1, grid_shape), stride_this_level, dtype=cost[0].dtype, device=cost[0].device))

# slice 一般会产生foreign node或内存移动
cost = torch.randn(2, 10, 1024)
cc0 = cost[:, :2, :] 
cc1 = cost[:, 0::2, :]  

idx = torch.tensor([0, 1], dtype=torch.long)
cc2 = cost[:, idx, :] 
cc3 = cost[:, [0, 1], :]  
print(cc0.is_contiguous(), cc1.is_contiguous(), cc2.is_contiguous(), cc3.is_contiguous())    
```

------------------------------------------------------------------------------    

## 参考    

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html     
https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html 
