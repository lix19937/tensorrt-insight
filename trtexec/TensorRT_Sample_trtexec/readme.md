This folder maintains CUDA and TRT sample compiling with CMake:
	- CUDA_Sample_nv12BLtoRGB is CUDA Sample compiling with CMake
	- TensorRT_Sample_trtexec is TRT Sample compiling with CMake
	- Toolchain-V5L.cmake is the CMAKE_TOOLCHAIN_FILE we used for compiling

Prerequisites:
Install DOS SDK and setup host cross-compile env following
        For linux: DRIVE_OS_6.0.3.0_Debian_Package_Installation_Guide_NVONLINE::Chapter 4.1
Install cmake 3.14:
    On Host:
        wget https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.sh
        sudo mkdir /opt/cmake
        sudo sh cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake
        sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
    On Target:
	sudo apt install -y cmake

How to compile:
	$ mkdir build
	$ cd build
	$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../../Toolchain-V5L.cmake -DCUDA_DIR=/usr/local/cuda-11.4 -DVIBRANTE_PDK=/drive/drive-linux
	$ make

Those two samples also can be compiled on Target Board which has installed 
    	cuda-repo-ubuntu2004-11-*-local_11.*_arm64.deb
	cudnn-prune-87-repo-ubuntu2004-*_arm64.deb
	nv-tensorrt-repo-ubuntu2004-cuda*_arm64.deb

Compile steps:
        $ mkdir build
	$ cd build
	$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../../Toolchain-V5L.cmake -DCUDA_DIR=/usr/local/cuda-11.4
        $ make
