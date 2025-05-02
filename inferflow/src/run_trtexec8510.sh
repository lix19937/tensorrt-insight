export LD_LIBRARY_PATH=/usr/local/tensorrt/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cudnn/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH

export PATH=/usr/local/tensorrt/src/tensorrt/bin:$PATH

trtexec
