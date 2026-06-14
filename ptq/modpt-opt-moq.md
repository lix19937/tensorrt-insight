```
[quantize]IN_PARA:
  onnx_path: './gwm_mt3_20260602_plugin.onnx'
  quantize_mode: 'int8'
  calibration_method: 'max'
  calibrate_per_node: False
  calibration_eps: ['trt']
  high_precision_dtype: 'fp32'
  log_level: 'DEBUG'
  op_types_to_exclude: ['Mul', 'MatMul', 'Gemm']
  nodes_to_quantize: ['/img_backbone/*']
  trt_plugins: ['/mnt/vepfs/share/GW00348951/ptq_test/libcustom_cc_trt_plugin-x86.so']
  dq_only: False
  direct_io_types: True
  keep_intermediate_files: False
  output_path: './gwm_mt3_20260602_plugin/int8_max_fp32.onnx'
  log_file: './gwm_mt3_20260602_plugin/int8_max_fp32.log'

Starting quantization process for model: ./gwm_mt3_20260602_plugin.onnx
Quantization mode: int8
Preprocessing the model ./gwm_mt3_20260602_plugin.onnx
 Checking for custom TensorRT ops
 Loading 1 TensorRT plugin(s) Loading static plugin: /mnt/vepfs/share/GW00348951/ptq_test/libcustom_cc_trt_plugin-x86.so
 Total registered plugin creators: 86
 Created TensorRT builder and network
 Found custom layer: /neck_lss/view_transformer/encoder/layers.0/sa/ms_deform_attn_plugin
 Found custom layer: PPScatter_0
 Found custom layer: /neck_lss/view_transformer/encoder/layers.0/ca/deformable_attention/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/decoder/layers.0/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/discrete_decoder/layers.0/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/decoder/layers.1/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/discrete_decoder/layers.1/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/decoder/layers.2/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/discrete_decoder/layers.2/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/decoder/layers.3/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/discrete_decoder/layers.3/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/decoder/layers.4/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/discrete_decoder/layers.4/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/decoder/layers.5/ca/ms_deform_attn_plugin
 Found custom layer: /map_head/bev_decoders.0/discrete_decoder/layers.5/ca/ms_deform_attn_plugin
Found 15 custom layers and 20573 tensors
 Found custom layers: ['/neck_lss/view_transformer/encoder/layers.0/sa/ms_deform_attn_plugin', 'PPScatter_0', '/neck_lss/view_transformer/encoder/layers.0/ca/deformable_attention/ms_deform_attn_plugin', '/map_head/bev_decoders.0/decoder/layers.0/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/discrete_decoder/layers.0/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/decoder/layers.1/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/discrete_decoder/layers.1/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/decoder/layers.2/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/discrete_decoder/layers.2/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/decoder/layers.3/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/discrete_decoder/layers.3/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/decoder/layers.4/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/discrete_decoder/layers.4/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/decoder/layers.5/ca/ms_deform_attn_plugin', '/map_head/bev_decoders.0/discrete_decoder/layers.5/ca/ms_deform_attn_plugin']
Found custom operators: {'ms_deform_attn_plugin', 'PPScatterPlugin'}
Added TRT plugin domain trt.plugins version 1

Updated tensors with type and shape information
 Model size: 897532734 bytes, using external data: False
Model with custom ops is saved to ./gwm_mt3_20260602_plugin/gwm_mt3_20260602_plugin_ort_support.onnx. Model contains custom ops: {'ms_deform_attn_plugin', 'PPScatterPlugin'}
 Model size: 897539356 bytes, using external data: False
Model is cloned to ./gwm_mt3_20260602_plugin/gwm_mt3_20260602_plugin_opset19.onnx with opset_version 19

Duplicating shared constants
 Model size: 1002622156 bytes, using external data: False
Model is cloned to ./gwm_mt3_20260602_plugin/gwm_mt3_20260602_plugin_named.onnx after naming the nodes

 Updating TRT EP support - DDS ops: False, Custom ops: True
Custom op detected, enabling TensorRT EP, Making TensorRT EP first choice in execution providers
Setting up CalibrationDataProvider for calibration
 Getting input shapes from model
 Multi-tensor calibration data with 34 inputs

 Creating 1 calibration iterations
Analyzing MHA nodes for int8 quantization
Creating ORT InferenceSession
 Preparing execution providers list from: ['trt']
Successfully imported the `tensorrt` python package with version 10.10.0.31
Checking for cuDNN library
libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
 Added TensorRT EP
Successfully enabled 1 EPs for ORT: ['TensorrtExecutionProvider']
 Creating session with providers: ['TensorrtExecutionProvider']
 Matmul nodes From MHA to exclude: []
Starting INT8 quantization with method: max
 Loaded model with 9251 nodes
Detecting GEMV patterns for TRT optimization
 Found 591 MatMul nodes to analyze

Creating ORT InferenceSession
 Preparing execution providers list from: ['trt']
Successfully imported the `tensorrt` python package with version 10.10.0.31
Checking for cuDNN library
libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
 Added TensorRT EP
Successfully enabled 1 EPs for ORT: ['TensorrtExecutionProvider']
 Creating session with providers: ['TensorrtExecutionProvider']
 Excluding small-dim MatMul from quantization: /lane_mask_head/navi_encoder/left_lane_action_embed/layers.0/MatMul (N=32, K=9, threshold=16)
 Excluding small-dim MatMul from quantization: /lane_mask_head/navi_encoder/right_lane_action_embed/layers.0/MatMul (N=32, K=9, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_1/MatMul (N=128, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_1/MatMul (N=32, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_1/MatMul (N=32, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_1/MatMul (N=64, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0/MatMul (N=128, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0/MatMul (N=32, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0/MatMul (N=32, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0/MatMul (N=64, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.3/camera_encoder/camera_encoder.0/MatMul (N=256, K=12, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.6/layers/layers.10/MatMul (N=11, K=256, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_2/MatMul (N=128, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_2/MatMul (N=32, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_2/MatMul (N=32, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_2/MatMul (N=64, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.10/camera_encoder/camera_encoder.0/MatMul (N=256, K=12, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.13/layers/layers.10/MatMul (N=11, K=256, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_3/MatMul (N=128, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_3/MatMul (N=32, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_3/MatMul (N=32, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_3/MatMul (N=64, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_reference_points/MatMul (N=8, K=512, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.17/camera_encoder/camera_encoder.0/MatMul (N=256, K=12, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6/MatMul (N=8, K=1024, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.20/layers/layers.10/MatMul (N=11, K=256, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_4/MatMul (N=128, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_4/MatMul (N=32, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_4/MatMul (N=32, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_4/MatMul (N=64, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_1/MatMul (N=8, K=1024, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.24/camera_encoder/camera_encoder.0/MatMul (N=256, K=12, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_2/MatMul (N=8, K=1024, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.27/layers/layers.10/MatMul (N=11, K=256, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_3/MatMul (N=8, K=1024, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_5/MatMul (N=128, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_5/MatMul (N=32, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_5/MatMul (N=32, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_5/MatMul (N=64, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_4/MatMul (N=8, K=1024, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.31/camera_encoder/camera_encoder.0/MatMul (N=256, K=12, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/cls_branches.0/cls_branches.0.0/MatMul (N=4, K=512, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_cls_branches.0/discrete_cls_branches.0.0/MatMul (N=3, K=512, threshold=16)
 Excluding small-dim MatMul from quantization: /map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_5/MatMul (N=8, K=1024, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.34/layers/layers.10/MatMul (N=11, K=256, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_6/MatMul (N=128, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_6/MatMul (N=32, K=3, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_6/MatMul (N=32, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_6/MatMul (N=64, K=2, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.38/camera_encoder/camera_encoder.0/MatMul (N=256, K=12, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.41/cls_layers/cls_layers.6/MatMul (N=7, K=256, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.41/quality_layers/quality_layers.6/MatMul (N=2, K=256, threshold=16)
 Excluding small-dim MatMul from quantization: /fusion_object_head/bev_decoder/layers.41/layers/layers.10/MatMul (N=11, K=256, threshold=16)

 Matmul nodes to exclude: ['/lane_mask_head/navi_encoder/main_action_embed/layers.0/Gemm', '/lane_mask_head/navi_encoder/assistant_action_embed/layers.0/Gemm', '/lane_mask_head/navi_encoder/distance_action_embed/layers.0/Gemm', '/lane_mask_head/navi_encoder/left_passable_embed/layers.0/Gemm', '/lane_mask_head/navi_encoder/right_passable_embed/layers.0/Gemm', '/lane_mask_head/navi_encoder/distance_lane_action_embed/layers.0/Gemm', '/lane_mask_head/navi_encoder/left_lane_action_embed/layers.0/MatMul', '/lane_mask_head/navi_encoder/right_lane_action_embed/layers.0/MatMul', '/lane_mask_head/navi_encoder/main_action_embed/layers.1/Gemm', '/lane_mask_head/navi_encoder/assistant_action_embed/layers.1/Gemm', '/lane_mask_head/navi_encoder/distance_action_embed/layers.1/Gemm', '/lane_mask_head/navi_encoder/left_passable_embed/layers.1/Gemm', '/lane_mask_head/navi_encoder/right_passable_embed/layers.1/Gemm', '/lane_mask_head/navi_encoder/distance_lane_action_embed/layers.1/Gemm', '/fusion_object_head/bev_decoder/MatMul_2', '/lane_mask_head/navi_encoder/action_fuse/layers.0/Gemm', '/lane_mask_head/navi_encoder/passable_fuse/layers.0/Gemm', '/fusion_object_head/bev_decoder/MatMul', '/lane_mask_head/navi_encoder/action_fuse/layers.1/Gemm', '/lane_mask_head/navi_encoder/passable_fuse/layers.1/Gemm', '/lane_mask_head/navi_encoder/lane_action_fuse/layers.0/Gemm', '/lane_mask_head/navi_encoder/lane_action_fuse/layers.1/Gemm', '/fusion_object_head/bev_decoder/MatMul_1', '/fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_1/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_1/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_1/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_1/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0/MatMul', '/fusion_object_head/bev_decoder/layers.3/camera_encoder/camera_encoder.0/MatMul', '/map_head/bev_fusion/layers.0/ca/multihead_attn/MatMul', '/fusion_object_head/bev_decoder/layers.6/layers/layers.10/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_2/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_2/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_2/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_2/MatMul', '/map_head/bev_fusion/layers.0/ca/multihead_attn/MatMul_1', '/map_head/bev_fusion/layers.1/ca/multihead_attn/MatMul', '/map_head/bev_fusion/layers.1/ca/multihead_attn/MatMul_1', '/fusion_object_head/bev_decoder/layers.10/camera_encoder/camera_encoder.0/MatMul', '/map_head/bev_fusion/layers.2/ca/multihead_attn/MatMul', '/map_head/bev_fusion/layers.2/ca/multihead_attn/MatMul_1', '/fusion_object_head/bev_decoder/layers.13/layers/layers.10/MatMul', '/map_head/bev_fusion/fp_layers.0/ca/multihead_attn/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_3/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_3/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_3/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_3/MatMul', '/map_head/bev_fusion/fp_layers.0/ca/multihead_attn/MatMul_1', '/map_head/bev_decoders.0/discrete_reference_points/MatMul', '/fusion_object_head/bev_decoder/layers.17/camera_encoder/camera_encoder.0/MatMul', '/map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6/MatMul', '/fusion_object_head/bev_decoder/layers.20/layers/layers.10/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_4/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_4/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_4/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_4/MatMul', '/map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_1/MatMul', '/fusion_object_head/bev_decoder/layers.24/camera_encoder/camera_encoder.0/MatMul', '/map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_2/MatMul', '/fusion_object_head/bev_decoder/layers.27/layers/layers.10/MatMul', '/map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_3/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_5/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_5/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_5/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_5/MatMul', '/map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_4/MatMul', '/fusion_object_head/bev_decoder/layers.31/camera_encoder/camera_encoder.0/MatMul', '/map_head/bev_decoders.0/cls_branches.0/cls_branches.0.0/MatMul', '/map_head/bev_decoders.0/discrete_cls_branches.0/discrete_cls_branches.0.0/MatMul', '/map_head/bev_decoders.0/discrete_decoder/discrete_reg_branches.0/discrete_reg_branches.0.6_5/MatMul', '/fusion_object_head/bev_decoder/layers.34/layers/layers.10/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/pos_fc/pos_fc.0_6/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/size_fc/size_fc.0_6/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/yaw_fc/yaw_fc.0_6/MatMul', '/fusion_object_head/bev_decoder/anchor_encoder/vel_fc/vel_fc.0_6/MatMul', '/fusion_object_head/bev_decoder/layers.38/camera_encoder/camera_encoder.0/MatMul', '/fusion_object_head/bev_decoder/layers.41/cls_layers/cls_layers.6/MatMul', '/fusion_object_head/bev_decoder/layers.41/quality_layers/quality_layers.6/MatMul', '/fusion_object_head/bev_decoder/layers.41/layers/layers.10/MatMul']
 Excluding 82 MatMul nodes due to GEMV pattern

Scanning for unsupported Conv nodes for quantization
 Found Conv with I/O channel size less than 16: /neck_lss/view_transformer/encoder/layers.0/ca/query_reduce_sum/Conv
Found 1 unsupported Conv nodes for quantization

Configuring ORT for ModelOpt ONNX quantization
 Registering custom QDQ operators
 Patching ORT modules
 Removing non-quantizable ops from QDQ registry
 Preparing execution providers list from: ['trt']
Successfully imported the `tensorrt` python package with version 10.10.0.31
Checking for cuDNN library
libcudnn_adv*.so* is accessible in /usr/lib/x86_64-linux-gnu/libcudnn_adv.so! Please check that this is the correct version needed for your ORT version at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
 Added TensorRT EP
Successfully enabled 1 EPs for ORT: ['TensorrtExecutionProvider']
 Getting quantizable operator types
 Using default quantizable ops: ['Mul', 'Resize', 'AveragePool', 'BatchNormalization', 'Add', 'Clip', 'MaxPool', 'ConvTranspose', 'MatMul', 'HardSwish', 'LRN', 'LayerNormalization', 'GlobalAveragePool', 'Conv', 'Gemm']
Quantizable op types: ['Mul', 'Resize', 'BatchNormalization', 'Add', 'Clip', 'MaxPool', 'ConvTranspose', 'MatMul', 'Conv', 'Gemm']

Final number of nodes to quantize: 86
 Selected 86 nodes to quantize: ['/img_backbone/conv1/Conv', '/img_backbone/relu/Relu', '/img_backbone/maxpool/MaxPool', '/img_backbone/layer1/layer1.0/conv1/Conv', '/img_backbone/layer1/layer1.0/relu/Relu', '/img_backbone/layer1/layer1.0/conv2/Conv', '/img_backbone/layer1/layer1.0/Add', '/img_backbone/layer1/layer1.0/relu_1/Relu', '/img_backbone/layer1/layer1.1/conv1/Conv', '/img_backbone/layer1/layer1.1/relu/Relu', '/img_backbone/layer1/layer1.1/conv2/Conv', '/img_backbone/layer1/layer1.1/Add', '/img_backbone/layer1/layer1.1/relu_1/Relu', '/img_backbone/layer1/layer1.2/conv1/Conv', '/img_backbone/layer1/layer1.2/relu/Relu', '/img_backbone/layer1/layer1.2/conv2/Conv', '/img_backbone/layer1/layer1.2/Add', '/img_backbone/layer1/layer1.2/relu_1/Relu', '/img_backbone/layer2/layer2.0/conv1/Conv', '/img_backbone/layer2/layer2.0/downsample/downsample.0/Conv', '/img_backbone/layer2/layer2.0/relu/Relu', '/img_backbone/layer2/layer2.0/conv2/Conv', '/img_backbone/layer2/layer2.0/Add', '/img_backbone/layer2/layer2.0/relu_1/Relu', '/img_backbone/layer2/layer2.1/conv1/Conv', '/img_backbone/layer2/layer2.1/relu/Relu', '/img_backbone/layer2/layer2.1/conv2/Conv', '/img_backbone/layer2/layer2.1/Add', '/img_backbone/layer2/layer2.1/relu_1/Relu', '/img_backbone/layer2/layer2.2/conv1/Conv', '/img_backbone/layer2/layer2.2/relu/Relu', '/img_backbone/layer2/layer2.2/conv2/Conv', '/img_backbone/layer2/layer2.2/Add', '/img_backbone/layer2/layer2.2/relu_1/Relu', '/img_backbone/layer2/layer2.3/conv1/Conv', '/img_backbone/layer2/layer2.3/relu/Relu', '/img_backbone/layer2/layer2.3/conv2/Conv', '/img_backbone/layer2/layer2.3/Add', '/img_backbone/layer2/layer2.3/relu_1/Relu', '/img_backbone/layer3/layer3.0/conv1/Conv', '/img_backbone/layer3/layer3.0/downsample/downsample.0/Conv', '/img_backbone/layer3/layer3.0/relu/Relu', '/img_backbone/layer3/layer3.0/conv2/Conv', '/img_backbone/layer3/layer3.0/Add', '/img_backbone/layer3/layer3.0/relu_1/Relu', '/img_backbone/layer3/layer3.1/conv1/Conv', '/img_backbone/layer3/layer3.1/relu/Relu', '/img_backbone/layer3/layer3.1/conv2/Conv', '/img_backbone/layer3/layer3.1/Add', '/img_backbone/layer3/layer3.1/relu_1/Relu', '/img_backbone/layer3/layer3.2/conv1/Conv', '/img_backbone/layer3/layer3.2/relu/Relu', '/img_backbone/layer3/layer3.2/conv2/Conv', '/img_backbone/layer3/layer3.2/Add', '/img_backbone/layer3/layer3.2/relu_1/Relu', '/img_backbone/layer3/layer3.3/conv1/Conv', '/img_backbone/layer3/layer3.3/relu/Relu', '/img_backbone/layer3/layer3.3/conv2/Conv', '/img_backbone/layer3/layer3.3/Add', '/img_backbone/layer3/layer3.3/relu_1/Relu', '/img_backbone/layer3/layer3.4/conv1/Conv', '/img_backbone/layer3/layer3.4/relu/Relu', '/img_backbone/layer3/layer3.4/conv2/Conv', '/img_backbone/layer3/layer3.4/Add', '/img_backbone/layer3/layer3.4/relu_1/Relu', '/img_backbone/layer3/layer3.5/conv1/Conv', '/img_backbone/layer3/layer3.5/relu/Relu', '/img_backbone/layer3/layer3.5/conv2/Conv', '/img_backbone/layer3/layer3.5/Add', '/img_backbone/layer3/layer3.5/relu_1/Relu', '/img_backbone/layer4/layer4.0/conv1/Conv', '/img_backbone/layer4/layer4.0/downsample/downsample.0/Conv', '/img_backbone/layer4/layer4.0/relu/Relu', '/img_backbone/layer4/layer4.0/conv2/Conv', '/img_backbone/layer4/layer4.0/Add', '/img_backbone/layer4/layer4.0/relu_1/Relu', '/img_backbone/layer4/layer4.1/conv1/Conv', '/img_backbone/layer4/layer4.1/relu/Relu', '/img_backbone/layer4/layer4.1/conv2/Conv', '/img_backbone/layer4/layer4.1/Add', '/img_backbone/layer4/layer4.1/relu_1/Relu', '/img_backbone/layer4/layer4.2/conv1/Conv', '/img_backbone/layer4/layer4.2/relu/Relu', '/img_backbone/layer4/layer4.2/conv2/Conv', '/img_backbone/layer4/layer4.2/Add', '/img_backbone/layer4/layer4.2/relu_1/Relu']
Finding concat eliminated tensors
 Disabled weight adjustment for INT32 bias in QDQ quantization
Starting static quantization

 Quantization format: QDQ
 Activation type: QInt8
 Weight type: QInt8
 Calibration method: CalibrationMethod.MinMax

 Calibration extra options: {'trt_extra_plugin_lib_paths': '/mnt/vepfs/share/GW00348951/ptq_test/libcustom_cc_trt_plugin-x86.so', 'execution_providers': ['TensorrtExecutionProvider'], 'disable_int32_weight_adjustment': True}
 Creating calibrator
 Creating inference session with Execution Provider configuration
 Execution providers: ['TensorrtExecutionProvider']
 TRT extra plugin paths: /mnt/vepfs/share/GW00348951/ptq_test/libcustom_cc_trt_plugin-x86.so
 TRT EP options: {'trt_extra_plugin_lib_paths': '/mnt/vepfs/share/GW00348951/ptq_test/libcustom_cc_trt_plugin-x86.so', 'trt_max_workspace_size': 85899345920}

 Collecting calibration data
 Computing tensor ranges
Starting post-processing of quantized model

Deleting QDQ nodes from marked inputs to make certain operations fusible

Quantization completed successfully in 504.83616614341736 seconds
Deleting Q nodes in the input of a quantized ONNX model.
 Processing QDQ node for output img_QuantizeLinear_Output
Removed 1 Q node

Total number of nodes: 9428
Total number of quantized nodes: 54
 Quantized type counts: {'Conv': 37, 'MaxPool': 1, 'Add': 16}
 Quantized nodes: ['/img_backbone/conv1/Conv', '/img_backbone/maxpool/MaxPool', '/img_backbone/layer1/layer1.0/conv1/Conv', '/img_backbone/layer1/layer1.0/conv2/Conv', '/img_backbone/layer1/layer1.0/Add', '/img_backbone/layer1/layer1.1/conv1/Conv', '/img_backbone/layer1/layer1.1/conv2/Conv', '/img_backbone/layer1/layer1.1/Add', '/img_backbone/layer1/layer1.2/conv1/Conv', '/img_backbone/layer1/layer1.2/conv2/Conv', '/img_backbone/layer1/layer1.2/Add', '/img_backbone/layer2/layer2.0/conv1/Conv', '/img_backbone/layer2/layer2.0/downsample/downsample.0/Conv', '/img_backbone/layer2/layer2.0/conv2/Conv', '/img_backbone/layer2/layer2.0/Add', '/img_backbone/layer2/layer2.1/conv1/Conv', '/img_backbone/layer2/layer2.1/conv2/Conv', '/img_backbone/layer2/layer2.1/Add', '/img_backbone/layer2/layer2.2/conv1/Conv', '/img_backbone/layer2/layer2.2/conv2/Conv', '/img_backbone/layer2/layer2.2/Add', '/img_backbone/layer2/layer2.3/conv1/Conv', '/img_backbone/layer2/layer2.3/conv2/Conv', '/img_backbone/layer2/layer2.3/Add', '/img_backbone/layer3/layer3.0/conv1/Conv', '/img_backbone/layer3/layer3.0/downsample/downsample.0/Conv', '/img_backbone/layer3/layer3.0/conv2/Conv', '/img_backbone/layer3/layer3.0/Add', '/img_backbone/layer3/layer3.1/conv1/Conv', '/img_backbone/layer3/layer3.1/conv2/Conv', '/img_backbone/layer3/layer3.1/Add', '/img_backbone/layer3/layer3.2/conv1/Conv', '/img_backbone/layer3/layer3.2/conv2/Conv', '/img_backbone/layer3/layer3.2/Add', '/img_backbone/layer3/layer3.3/conv1/Conv', '/img_backbone/layer3/layer3.3/conv2/Conv', '/img_backbone/layer3/layer3.3/Add', '/img_backbone/layer3/layer3.4/conv1/Conv', '/img_backbone/layer3/layer3.4/conv2/Conv', '/img_backbone/layer3/layer3.4/Add', '/img_backbone/layer3/layer3.5/conv1/Conv', '/img_backbone/layer3/layer3.5/conv2/Conv', '/img_backbone/layer3/layer3.5/Add', '/img_backbone/layer4/layer4.0/conv1/Conv', '/img_backbone/layer4/layer4.0/downsample/downsample.0/Conv', '/img_neck/lateral_convs.2/conv/Conv', '/img_backbone/layer4/layer4.0/conv2/Conv', '/img_backbone/layer4/layer4.0/Add', '/img_backbone/layer4/layer4.1/conv1/Conv', '/img_backbone/layer4/layer4.1/conv2/Conv', '/img_backbone/layer4/layer4.1/Add', '/img_backbone/layer4/layer4.2/conv1/Conv', '/img_backbone/layer4/layer4.2/conv2/Conv', '/img_backbone/layer4/layer4.2/Add']
 Model size: 1002740361 bytes, using external data: False
Quantized onnx model is saved as ./gwm_mt3_20260602_plugin/int8_max_fp32.onnx
Cleaning up intermediate files

Validating quantized model

Quantization process completed

```
