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

set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_LIBRARY_ARCHITECTURE aarch64-linux-gnu)

# check that Vibrante PDK must be set
if(NOT DEFINED VIBRANTE_PDK)
    if(DEFINED ENV{VIBRANTE_PDK})
        message(STATUS "VIBRANTE_PDK = ENV : $ENV{VIBRANTE_PDK}")
        set(VIBRANTE_PDK $ENV{VIBRANTE_PDK} CACHE STRING "Path to the vibrante-XXX-linux path for cross-compilation" FORCE)
    endif()
else()
     message(STATUS "VIBRANTE_PDK = ${VIBRANTE_PDK}")
endif()

if(DEFINED VIBRANTE_PDK)
  if(NOT IS_ABSOLUTE ${VIBRANTE_PDK})
      get_filename_component(VIBRANTE_PDK "${SDK_BINARY_DIR}/${VIBRANTE_PDK}" ABSOLUTE)
  endif()
endif()

set(ARCH "aarch64")

# parse vibrante branch / GCID information from PDK version files
if(((NOT DEFINED VIBRANTE_PDK_BRANCH) OR (NOT DEFINED VIBRANTE_PDK_GCID)) AND VIBRANTE_PDK)
    if(EXISTS "${VIBRANTE_PDK}/lib-target/version-nv-pdk.txt")
        set(VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-nv-pdk.txt")
    elseif(EXISTS "${VIBRANTE_PDK}/lib-target/version-nv-sdk.txt")
        set(VIBRANTE_PDK_FILE "${VIBRANTE_PDK}/lib-target/version-nv-sdk.txt")
    endif()

    if(VIBRANTE_PDK_FILE)
       file(READ ${VIBRANTE_PDK_FILE} version-nv-pdk)
    else()
       message(FATAL_ERROR "Can't open ${VIBRANTE_PDK}/lib-target/version-nv-(pdk/sdk).txt for PDK branch detection")
    endif()

    # Get branch from version-nv-pdk
    if(NOT DEFINED VIBRANTE_PDK_BRANCH)
      if(${version-nv-pdk} MATCHES "^(.+)-[0123456789]+")
          set(VIBRANTE_PDK_BRANCH ${CMAKE_MATCH_1} CACHE STRING "Cross-compilation PDK branch name")
          message(STATUS "VIBRANTE_PDK_BRANCH = ${VIBRANTE_PDK_BRANCH}")
      else()
          message(FATAL_ERROR "Can't determine PDK branch for PDK ${VIBRANTE_PDK}")
      endif()
    endif()

    # Get GCID from version-nv-pdk. Expecting version format to be: "version-gcid"
    if(NOT DEFINED VIBRANTE_PDK_GCID)
        string(REPLACE "-" ";" PDK_VERSION_LIST ${version-nv-pdk})
        list(LENGTH PDK_VERSION_LIST _PDK_VERSION_LIST_LENGTH)
        if (NOT _PDK_VERSION_LIST_LENGTH EQUAL 2)
            message(FATAL_ERROR "Unsupported pdk version string detected. Unable to set VIBRANTE_PDK_GCID. version-nv-pdk is ${version-nv-pdk}")
        endif()
        list(GET PDK_VERSION_LIST 1 VIBRANTE_PDK_GCID)
        string(STRIP ${VIBRANTE_PDK_GCID} VIBRANTE_PDK_GCID)
    endif()
endif()

if(DEFINED VIBRANTE_PDK_BRANCH)
  string(REPLACE "." ";" PDK_VERSION_LIST ${VIBRANTE_PDK_BRANCH})

  # Some PDK's have less than three version numbers. Pad the list so we always
  # have at least three numbers, allowing pre-existing logic depending on major,
  # minor, patch versioning to work without modifications
  list(LENGTH PDK_VERSION_LIST _PDK_VERSION_LIST_LENGTH)
  while(_PDK_VERSION_LIST_LENGTH LESS 3)
    list(APPEND PDK_VERSION_LIST 0)
    math(EXPR _PDK_VERSION_LIST_LENGTH "${_PDK_VERSION_LIST_LENGTH} + 1")
  endwhile()

  set(VIBRANTE_PDK_PATCH 0)
  set(VIBRANTE_PDK_BUILD 0)

  list(LENGTH PDK_VERSION_LIST PDK_VERSION_LIST_LENGTH)

  list(GET PDK_VERSION_LIST 0 VIBRANTE_PDK_MAJOR)
  list(GET PDK_VERSION_LIST 1 VIBRANTE_PDK_MINOR)
  if(PDK_VERSION_LIST_LENGTH GREATER 2)
    list(GET PDK_VERSION_LIST 2 VIBRANTE_PDK_PATCH)
  endif()

  if(PDK_VERSION_LIST_LENGTH GREATER 3)
    list(GET PDK_VERSION_LIST 3 VIBRANTE_PDK_BUILD)
  endif()

  set(VIBRANTE_PDK_VERSION ${VIBRANTE_PDK_MAJOR}.${VIBRANTE_PDK_MINOR}.${VIBRANTE_PDK_PATCH}.${VIBRANTE_PDK_BUILD})

  add_definitions(-DVIBRANTE_PDK_VERSION=\"${VIBRANTE_PDK_VERSION}\") # requires escaping so it is treated as a string
                                                                      # and not an invalid floating point with too many decimal points
  add_definitions(-DVIBRANTE_PDK_MAJOR=${VIBRANTE_PDK_MAJOR})
  add_definitions(-DVIBRANTE_PDK_MINOR=${VIBRANTE_PDK_MINOR})
  add_definitions(-DVIBRANTE_PDK_PATCH=${VIBRANTE_PDK_PATCH})
  add_definitions(-DVIBRANTE_PDK_BUILD=${VIBRANTE_PDK_BUILD})

  math(EXPR VIBRANTE_PDK_DECIMAL "${VIBRANTE_PDK_MAJOR} * 1000000 + \
                                  ${VIBRANTE_PDK_MINOR} * 10000 + \
                                  ${VIBRANTE_PDK_PATCH} * 100 + \
                                  ${VIBRANTE_PDK_BUILD}")
  add_definitions(-DVIBRANTE_PDK_DECIMAL=${VIBRANTE_PDK_DECIMAL})

  message(STATUS "Vibrante version ${VIBRANTE_PDK_VERSION} (${VIBRANTE_PDK_DECIMAL})")
endif()

set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++")
set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gcc")
set(GCC_COMPILER_VERSION "9.3.0" CACHE STRING "GCC Compiler version")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Toolchain-V5_private.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Toolchain-V5_private.cmake)
endif()

set(LD_PATH ${VIBRANTE_PDK}/lib-target)

# Please, be careful looks like "-Wl,-unresolved-symbols=ignore-in-shared-libs" can lead to silent "ld" problems
set(CMAKE_SHARED_LINKER_FLAGS   "-L${LD_PATH} -Wl,--stats -Wl,-rpath,${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_SHARED_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS   "-L${LD_PATH} -Wl,-rpath,${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_MODULE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS      "-L${LD_PATH} -Wl,-rpath,${LD_PATH} -Wl,-rpath-link,${LD_PATH} ${CMAKE_EXE_LINKER_FLAGS}")

# Set default library search path
set(CMAKE_LIBRARY_PATH ${LD_PATH})

# Set cmake root path. If there is no "/usr/local" in CMAKE_FIND_ROOT_PATH then FinCUDA.cmake doesn't work
set(CMAKE_FIND_ROOT_PATH ${VIBRANTE_PDK} ${VIBRANTE_PDK}/filesystem/targetfs/usr/local/ ${VIBRANTE_PDK}/filesystem/targetfs/ ${CUDA_DIR})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)


# set system default include dir
#
# NOTE: Set CUDA_DIR BEFORE PDK include, so that PDK include will actually come first.
# This is important for ensuring that PDK and CUDA directories are searched in a predictable
# deterministic manner. In the case of include files that exist in both places, this ensures
# That we always check PDK first, and THEN CUDA.
include_directories(BEFORE SYSTEM ${CUDA_DIR}/targets/aarch64-linux/include)
include_directories(BEFORE SYSTEM ${VIBRANTE_PDK}/include)
