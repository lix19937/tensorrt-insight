#!/bin/bash
# trtexec --onnx=floatv20250227.onnx   --verbose --saveEngine=marker_fp32.plan
# trtexec --onnx=floatv20250227.onnx   --verbose --saveEngine=marker.plan  --fp16   


input_files=("input.1:input_tensor/0/img.bin,onnx::Slice_2:input_tensor/0/point_feat.bin,onnx::Slice_3:input_tensor/0/point_depth.bin"     
             "input.1:input_tensor/1/img.bin,onnx::Slice_2:input_tensor/1/point_feat.bin,onnx::Slice_3:input_tensor/1/point_depth.bin"    
             "input.1:input_tensor/2/img.bin,onnx::Slice_2:input_tensor/2/point_feat.bin,onnx::Slice_3:input_tensor/2/point_depth.bin"    
             "input.1:input_tensor/3/img.bin,onnx::Slice_2:input_tensor/3/point_feat.bin,onnx::Slice_3:input_tensor/3/point_depth.bin"    
             "input.1:input_tensor/4/img.bin,onnx::Slice_2:input_tensor/4/point_feat.bin,onnx::Slice_3:input_tensor/4/point_depth.bin")   

engine_file="onnx/marker.plan"

output_dir="./output_tensor/"  

if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

idx=0
for input_file in "${input_files[@]}"
do
    output_file=$output_dir$idx"_output.json"

    echo "Processing $input_file ..."
    trtexec --loadEngine=$engine_file --loadInputs=$input_file --exportOutput=$output_file 
    if [ $? -ne 0 ]; then
        echo "Error processing $input_file" >> "$output_file"
    else
        echo "Output saved to $output_file"
    fi

    idx=$(($idx+1))
done

echo "All files processed! Outputs are saved in $output_dir."
