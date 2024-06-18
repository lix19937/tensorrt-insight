
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

template <typename T>
void rebatch(
    T* queries_rebatch,
    T* reference_points_rebatch,
    const T* query,
    const T* reference_points_cam,
    const int* index, // [num_cams, max_len]
    const int index_len[], // each index valid len
    const int query_stride, // embed_dims
    const int reference_points_stride, // D*2
    const int rebatch_len, // max_len
    const int reference_points_cam_len, // num_query
    const int num_cams,
    cudaStream_t stream[]);
