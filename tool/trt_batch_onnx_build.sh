
  
# ssh -p 2025 nvidia@10.254.79.202     nvidia  

# /opt/deeproute/driver/config/starter/all_config.jsonnet


#!/bin/bash

data_pairs=(
     "mnist.onnx,best"
   # "BV_PSD_P01_V02_20240619_TEST.onnx,fp16" 
    #"BV_SEG_P01_V00_20240425.onnx,best"
    #"bevf-160-200-20-20-1_simp.onnx,best"
    #"bevf-320-400-20-20-1_simp.onnx,fp16"
    #"bevf-320-400-20-20-1_simp_ops11_1_input.onnx,best"
    #"bevf-64-160-20-20-1_poly.onnx,best"
    #"bevformer_tiny_resnet18_poly.onnx,best"
    #"mobilenetv2-sim.onnx,best"
    #"yolov5s_poly.onnx,best"
)

ONNX_DIR=/ota
echo $ONNX_DIR

for pair in "${data_pairs[@]}"; do
    IFS=',' read -r filename type <<< "$pair"
    echo $filename, $type

    sonfile_name=$(basename $filename)
    prefix=$(echo "$sonfile_name" | awk -F'.' '{print $1}')
    echo $prefix
    base_name=$prefix

    /usr/src/tensorrt/bin/trtexec --onnx=$ONNX_DIR/$filename \
    --saveEngine=$ONNX_DIR/${base_name}.plan \
    --verbose \
    --$type  \
    --dumpProfile \
    --noDataTransfers --useCudaGraph --useSpinWait  --separateProfileRun \
    2>&1 | tee $ONNX_DIR/${base_name}.log
done

