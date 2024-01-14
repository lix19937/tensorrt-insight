# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2021-12-17 09:06:42 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2021-12-17 09:06:42 
#  **************************************************************/

cmake_minimum_required(VERSION 3.5.1)
project(trt_bev_infer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "-O2 -ffast-math -Wall -Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")

message(STATUS "------------------------------------------------------------------------")

set(srcdir ${CMAKE_CURRENT_SOURCE_DIR}/../) 
set(debout ${CMAKE_CURRENT_SOURCE_DIR}/debout-8510)

include_directories(${debout}/usr/include/aarch64-linux-gnu)
include_directories(${debout}/usr/src/tensorrt/samples/common)
include_directories(/usr/local/cuda-11.4/include)

######################################################################
link_directories(${debout}/usr/lib/aarch64-linux-gnu)
link_directories(/usr/local/cuda-11.4/lib64)  # or /usr/local/cuda-11.4/targets/aarch64-linux/lib

set(EXE_NAME mytrtexec)

set(SOURCE_FILES ${srcdir}/src/mytrtexec-use-the-same-binds_no_thirdparty.cpp  )

### Build executable
add_executable(${EXE_NAME}     ${SOURCE_FILES})

target_link_libraries(${EXE_NAME}
    /usr/lib/aarch64-linux-gnu/libcudnn.so.8 
    /usr/local/cuda-11.4/lib64/libcublas.so.11 
    /usr/local/cuda-11.4/lib64/libcudart.so.11.0
    nvinfer
    nvparsers
    nvonnxparser
    nvinfer_plugin
    pthread
    dl
    m
)


