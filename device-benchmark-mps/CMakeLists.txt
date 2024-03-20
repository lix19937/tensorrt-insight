# only support ubuntu >= 18.04. which has cmake 3.10
# the reason why we need it is because FindCUDA/find_package(CUDA)
# is depricated since cmake 3.10.
# refer to https://cmake.org/cmake/help/v3.25/module/FindCUDA.html
cmake_minimum_required( VERSION 3.10)

# in drive platform, the cuda bin path hasn't being set by default.
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(TRT_BENCH)

enable_language(CUDA)
# this is enable command like cuda_add_library etc.
find_package(CUDA)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fpic -fpie -fpermissive  -std=c++11 -pthread -W -Wall")

# Set the CUDA architecture to compile for
set(CUDA_ARCH "-arch=sm_70; -arch=sm_72; -arch=sm_75; -arch=sm_80; -arch=sm_86" CACHE STRING "CUDA architecture")
# Add the CUDA compiler flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH})

# global include_directories
include_directories(/usr/local/cuda/include)
include_directories(/usr/include/aarch64-linux-gnu/) # for jetson
include_directories("${CMAKE_SOURCE_DIR}/src/")
include_directories("${CMAKE_SOURCE_DIR}/src/customtasks/")
include_directories("${CMAKE_SOURCE_DIR}/mps/")

# global library path
link_directories("/usr/lib/aarch64-linux-gnu/")
link_directories("/usr/lib/")
link_directories("/usr/local/lib/")
link_directories("/usr/local/cuda/lib64/")

FILE(GLOB_RECURSE TRT_BENCH_CU_SRC src/*.cu)
cuda_add_library(cutask SHARED ${TRT_BENCH_CU_SRC})
target_link_libraries(cutask cuda)

FILE(GLOB_RECURSE TRT_BENCH_SRC src/*.cpp mps/zmq_u.cpp)
add_library(TrtTask SHARED ${TRT_BENCH_SRC})
target_link_libraries(TrtTask PRIVATE nvinfer)
target_link_libraries(TrtTask PRIVATE nvinfer_plugin)
target_link_libraries(TrtTask PRIVATE nvparsers)
target_link_libraries(TrtTask PRIVATE nvonnxparser cudart)

add_executable(trt_bench trt_bench.cpp )
target_link_libraries(trt_bench TrtTask cudart cutask)

add_executable(mps_server mps/mps_server.cpp )
target_link_libraries(mps_server TrtTask cuda cutask cudart)

add_executable(mps_client mps/mps_client.cpp )
target_link_libraries(mps_client TrtTask cuda cutask cudart)