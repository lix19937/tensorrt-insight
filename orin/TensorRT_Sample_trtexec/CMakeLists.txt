################################################################################
#
# Notice
# ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
# NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
# OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
# WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
# PURPOSE.
#
# NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences
# of use of such information or for any infringement of patents or other rights
# of third parties that may result from its use. No license is granted by
# implication or otherwise under any patent or patent rights of NVIDIA
# CORPORATION & AFFILIATES. No third party distribution is allowed unless
# expressly authorized by NVIDIA. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied. NVIDIA
# CORPORATION & AFFILIATES products are not authorized for use as critical
# components in life support devices or systems without express written approval
# of NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
#
################################################################################

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(SDK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SDK_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_C_STANDARD 99)

if (DEFINED CUDA_DIR)
    if((DEFINED CUDA_TOOLKIT_ROOT_DIR) AND (NOT CUDA_TOOLKIT_ROOT_DIR STREQUAL CUDA_DIR))
        message(FATAL_ERROR "Cannot set both CUDA_DIR and (legacy) CUDA_TOOLKIT_ROOT_DIR")
    endif()
elseif (DEFINED ENV{CUDA_INSTALL_DIR})
    set(CUDA_DIR $ENV{CUDA_INSTALL_DIR})
else()
    set(CUDA_DIR  "/usr/local/cuda/" CACHE PATH "CUDA Toolkit location.")
endif()
if(NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDA_DIR}/bin/nvcc")
endif()

set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CUDA_DIR}/bin/nvcc")
set(CUDA_LIBRARY_DIRS ${CUDA_DIR}/targets/aarch64-linux/lib)
set(CMAKE_EXE_LINKER_FLAGS      "${QNX_LINKER_FLAGS} -L${CUDA_LIBRARY_DIRS} -Wl,-rpath-link,${CUDA_LIBRARY_DIRS}  ${CMAKE_EXE_LINKER_FLAGS}")

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/aarch64-linux-gnu-g++)
set(CUDA_INCLUDE_DIRS ${CUDA_DIR}/targets/aarch64-linux/include)
enable_language(CUDA)

project(trtexec_cmake LANGUAGES CXX CUDA)
#include_directories(BEFORE SYSTEM common/inc)
#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    common/sampleEngines.cpp
    common/sampleInference.cpp
    common/sampleOptions.cpp
    common/sampleReporting.cpp
    common/logger.cpp
    trtexec.cpp
)

set(LIBRARIES -lcudart -lnvinfer -lnvinfer_plugin -lnvcaffe_parser -lnvonnxparser -lnvparsers -pthread -ldl)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME}
    PUBLIC /usr/include/aarch64-linux-gnu
    PUBLIC ${CUDA_INCLUDE_DIRS}
    PRIVATE common
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
