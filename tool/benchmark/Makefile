lib_nvinfer=/usr/local/tensorrt/lib/x86_64-linux-gnu
inc_nvinfer=/usr/local/tensorrt/include/x86_64-linux-gnu
target_file=test

all:
	nvcc main.cpp -lnvinfer  \
	-L$(lib_nvinfer) \
	-I$(inc_nvinfer) \
	-Xcompiler -std=c++14 -O2 -o  $(target_file)

	
