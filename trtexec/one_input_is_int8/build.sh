 
trtexec --onnx=./sca.onnx --plugins=./libplugin_custom.so --verbose \
--inputIOFormats=fp32:chw,fp32:chw,int8:chw   \
--outputIOFormats=int32:chw,int32:chw,fp32:chw,fp32:chw,fp32:chw

  trtexec --onnx=./sca.onnx --plugins=./libplugin_custom.so --verbose --fp16 \
--inputIOFormats=fp16:chw,fp16:chw,int8:chw   \
--outputIOFormats=int32:chw,int32:chw,fp16:chw,fp16:chw,fp16:chw
 
