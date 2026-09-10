## 计算图优化    

例如LayerNorm本身为一个算子，但是ONNX导出会拆分为很多细小的算子，则必须用图优化将其合并回去。又比如ONNX不支持NHWC等数据格式，但是硬件部署往往又需要转换到硬件优化的NHWC或其他数据格式。另外，根据引擎和芯片的特殊性，往往也需要对模型进行一些针对性修改优化，例如一些引擎不支持超过5D的transpose计算。    

## 如何做图优化？  
* 第一个阶段**在线优化**，是直接对代码（网络的forward）进行修改，从而导出的模型直接具有优化后的特性。对于一些算法依赖，通用性并不是很强，或者底层实施图优化困难的优化可以考虑该方法。     
* 第二个阶段**离线优化**，是对导出的模型进行优化，例如对导出的ONNX模型进行优化。一般离线优化前可使用推理引擎如TensorRT进行 layer info profile，看看 layer 的融合情况，再针对性优化onnx 。     
* 第三个阶段是**在深度学习引擎内部运行时进行优化**，可以结合硬件特点和输入shape对应的性能信息进行更加针对性和极致的优化。第三阶段一般也可在第一阶段执行后执行辅助分析，给第二阶段提供参考。    


## 常见场景   

*  OneHot+MatMul  to  Gather    

* focus

* permute + conv [+ permute]
  

* 多个gather可以替换为split    split adjust     

* 统一归一化函数  
其实由F.normalize(x, p=2.0, dim=-1)导出，可以进行针对性合并。第二个图计算跟第一个图类似，但是为不同代码编写方式，这种最好是修改模型pytorch代码改成同一种norm函数，降低模型图优化代码开发。  

*  SpaceToDepth与DepthToSpace算子   
图中reshape+transpose(perm=[0, 1, 3, 5, 2, 4])计算等价于SpaceToDepth，由于transpose场景特别多，这里替换后可以在计算上进行更加针对性优化，此外，这个优化使得做NCHW到NHWC等格式转换优化也更加容易，transpose(perm=[0, 1, 4, 2, 5, 3])等价于DepthToSpace   

*  transpose+reshape+transpose to  transpose


*  high-mm -> low-mm
    
*  Linear + LayerNorm   --> Matmul + Add +  LayerNormalization   
  
*  矩阵乘+BN 融合


*  反卷积+BN 也应该能进行融合      


*  合并相邻的Conv2D或MatMul，无非线性层 conv + conv (without non-linear op)    


*  相邻的卷积/matmul+bias，中间没有非线性层的话可以基于卷积和矩阵乘的线性性直接合并 conv + linear (without non-linear op)    


*  特殊的1x1 depthwise卷积替换为elemwise


*  MatMul与Add, Mul向量计算融合


*  LayerNorm算子合并   
 

*  squeeze  unsqueeze 去除      


*  reshape去除
  

*  多路相同计算patten合并为batch计算，显著降低算子数量，提升算子计算密集性       
如图的场景；此外另一个场景是transformer模型中attention三个矩阵乘合可以并成为batch matmul，或者自己实现的特殊能够同时接受多个输入和bias的矩阵乘。  

*  多路并行的slice在特定情况下可以换成gather，并且后面所有的这些elemwise都可以合成一路计算。   
下图中这个稍微复杂点，slice可以替换为gahter，elemwise可以跟着合并，但gatherV2合并需要自定义算子batch gatherV2。   


*  调整顺序从而把div折叠到matmul和bias_add参数里面       
通常不鼓励矩阵和大容量内存拷贝的`物理重排`的原因是，数据移动非常昂贵（在能量和性能方面），因此应将其最小化，理想情况下应仅作为`计算的一部分`进行。  
换句话说，每次我们接触一些数据时，我们都想在存储之前对其进行一些变换。

 
```
"融合让节点变少"这个直觉只对消除类的 pass 成立(去冗余、常量折叠),而这个模型的主力 pass 是 MatMul+Add → Gemm,它是用节点数换 kernel 质量的。

  根因:ONNX Gemm 只接受 rank-2

  MatMul 接受任意 rank(会广播 batch 维),Gemm 规范上死死限定 A/B 都是二维矩阵。所以要把一个 rank-3 的 MatMul+Add 融成 Gemm,必须在前后各插一个 Reshape 做降维/升维:

  原:  [1,16,9] --MatMul(W[9,32])--> [1,16,32] --Add(b)--> [1,16,32]      2 个节点
  新:  [1,16,9] --Reshape[-1,9]--> [16,9] --Gemm(W[32,9],b)--> [16,32]
                                            --Reshape[1,16,32]--> [1,16,32]  3 个节点

  上面 4 个真实例子全是这个形状。统计上 515 个 Gemm 里 **497 个(96%)**都被夹在 Reshape -> Gemm -> Reshape 之间:

     497x  Reshape  -> Gemm -> Reshape
       9x  Relu     -> Gemm -> Relu          # 本来就是 rank-2,不用垫
       6x  INPUT    -> Gemm -> Relu
       3x  Concat   -> Gemm -> Relu

  账目对得上

  删掉                              加上
    MatMul           -463             Gemm      +463
    Add              -463             Reshape   +753
    QuantizeLinear    -83
    DequantizeLinear  -83
    Slice             -12
    Mul                -6
    ------------------------          ---------------
    合计            -1110             合计     +1216   →  净 +106

  463 × 2 = 926 个理论上要插的 Reshape,实际只加了 753 个 —— 说明 onnxslim 还复用/折叠掉了约 170 个(相邻 Reshape 合并、rank 本来就对的不插)。这个数字合理,不是失控。

  为什么这笔交易是赚的   

  - 926 个真正做算术的节点(MatMul+Add,各自要读写一遍中间 tensor)变成 463 次单一 Gemm 调用,直接命中 cuBLAS / TensorRT 调优过的 GEMM kernel,还顺带把 bias 吃进 epilogue,省掉 463 次中间 tensor 的写回+读取。
  - 新增的 753 个 Reshape 是纯元信息操作,不碰数据。TensorRT 在 build 期基本会把它们折进 tensor descriptor,大部分在最终 engine 里根本不存在层。
  - 另外 83 对 Q/DQ 是真删除:之前已验证是 72 组共享 (input, scale, zero_point) 的重复 QuantizeLinear,恰好 83 个,DQ 消费者一个没少。

  所以 node 数不是推理开销的代理指标。真要判断快没快,只能看 engine:

  trtexec --onnx=/workspaces/dmt/ptq.onnx --fp16 --plugins=<your_trt_plugin.so>
  trtexec --onnx=./t.onnx               --fp16 --plugins=<your_trt_plugin.so>

  而且这个模型 --check 跑不了(custom:ms_deform_attn_plugin 在 onnxruntime 里没注册),所以数值对齐也只能靠 trtexec 在真实数据上比 —— 之前那 4 个 SKIP 模型是同一个处境。
```


-------------------------

```
 差异明细

  删掉                              加上
    MatMul              -463           Gemm      +463
    Add                 -463           Reshape   +753
    BatchNormalization   -12
    Slice                -12
    Mul                   -6
    --------------------------         ---------------
    合计               -956            合计     +1216   →  净 +260

  和上个量化模型对比:MatMul/Add/Gemm/Reshape/Slice/Mul 六项的数字一模一样(-463/-463/+463/+753/-12/-6)。区别只有两处:上个模型多删了 83 对 Q/DQ(净 -166),这个模型多删了 12 个 BatchNormalization。所以 +106 vs
  +260 的差,正好是 166 - 12 = 154。两个模型是同一份网络的量化/非量化版本,onnxslim 做的事完全一致。

  三笔改动都是对的

  1. MatMul+Add → Gemm(+463 Gemm, +753 Reshape, -926 算术节点)

  515 个 Gemm 里 497 个(96%)夹在 Reshape -> Gemm -> Reshape 之间 —— 因为 ONNX Gemm 规范只接受 rank-2,而原来的 MatMul 是 rank-3,必须垫两个 Reshape 降/升维。Reshape 是纯元信息操作,不碰数据,TensorRT build
  期基本折进 tensor descriptor,最终 engine 里大多不存在这一层。

  515/515 个 Gemm 都带 3 个输入,即 bias 全部吃进了 Gemm 的 C 参数(cuBLAS epilogue),省掉 463 次中间 tensor 的写回+读取。

  2. ConvTranspose+BN 折叠(-12 BN)

  12 个 BN 全部紧跟在 ConvTranspose 后面,是训练遗留的、推理期恒定的仿射变换,被折进了前一层的权重。这是纯收益,零风险。

  3. Conv 一个没动

  Conv=270, with_bias=269,优化前后完全一致 —— 说明 onnxslim 没有去碰 Conv 分支,不是"到处乱改"。

  4. 32 个图输出的名字和 shape 完全不变,ir_version 保持 8(比上个模型的 11/12 友好得多)。

  仍然要注意的

  align = SKIP 的原因还是那个:custom:ms_deform_attn_plugin 在 onnxruntime 里没注册,所以数值对齐没有被验证过,不能当成已对齐。

  不过这个模型有个上个模型没有的优势:ir_version 8 + opset 17 都在 onnxruntime 的支持范围内,唯一的障碍就是那一个 plugin。如果上游 plugin/libcustom_ort_plugin.so 里实现了 ms_deform_attn_plugin,可以直接补上验证:

  python3 ./onnx_sim.py --onnx /workspaces/dmt/gwm_mt4_trtshape_p17_v221_20260827.onnx \
      --out ./t2 --check --custom_lib <path>/libcustom_ort_plugin.so

  但注意 --custom_lib 只对 --pattern onnxsim 生效(onnxslim 代码里完全没有 SessionOptions,传了会被静默忽略)。所以要么先用 onnxsim 跑一遍对齐来交叉验证 onnxslim 的结果,要么直接上 trtexec:

  trtexec --onnx=/workspaces/dmt/gwm_mt4_trtshape_p17_v221_20260827.onnx --fp16 --plugins=<libcustom_trt_plugin.so>
  trtexec --onnx=./t2/sim.onnx --fp16 --plugins=<libcustom_trt_plugin.so>

  后者顺带能回答"这 260 个节点换来的性能提升到底有多少"。

```

```
  - Skill = 按需加载的"流程说明书 + 脚本",解决的是知识注入。确定性的、要被 review 的、要进 git 的东西。
  - Agent(subagent) = 独立 context window,跑完只回一段结论,解决的是上下文隔离。

  你这条流水线的头号杀手不是"不够聪明",是上下文爆炸:onnx_act_stats.py 单模型中间张量 20+ GB,trtexec --verbose 几万行,16 模型批量表格,4 套 kout 差分。这些全走主上下文,流水线跑到第 3 阶段就没窗口了。

  我的建议:三层,skill 当骨架

  Layer 1  scripts/*.py|sh     确定性执行,零 LLM 判断   ← 你已经做对了,继续
  Layer 2  SKILL.md            流水线规格:步骤编号、日志约定、产物布局
  Layer 3  subagent            只包"要搜索/要试错/输出巨大"的步骤

  哪些步骤该丢给 subagent(判断 + 噪声大):

  ┌─────────────────────────┬───────────────────────────────────────────────────────┐
  │          步骤           │                        为什么                         │
  ├─────────────────────────┼───────────────────────────────────────────────────────┤
  │ fast_quant 量化配置搜索 │ 是个搜索循环,要试很多组合                             │
  ├─────────────────────────┼───────────────────────────────────────────────────────┤
  │ 敏感层分析 sla          │ 同上                                                  │
  ├─────────────────────────┼───────────────────────────────────────────────────────┤
  │ 对齐/编译失败的定位     │ 探索性,像我刚才查 attention 那样,过程几百行、结论三行 │
  ├─────────────────────────┼───────────────────────────────────────────────────────┤
  │ eval_data 大规模评测    │ 输出量大,主上下文只需要"通过/不通过 + 差异摘要"       │
  └─────────────────────────┴───────────────────────────────────────────────────────┘

```

 

 

