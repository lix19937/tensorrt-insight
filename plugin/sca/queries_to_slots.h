
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

typedef unsigned char uchar;

template <typename T>
void count_norm(
    T* count_norm, // [bs, num_query]
    const uchar* count, // [bs, num_query, num_cams]
    const int num_cams,
    const int num_query,
    cudaStream_t stream);

template <typename T>
void queries_to_slots(
    T* slots, // [num_cams, num_query, embed_dims]
    const T* queries, // [1, num_cams, max_len, embed_dims]
    const int* index, // [num_cams, max_len]
    const int index_len[], // valid len of each index
    const int stride, // embed_dims
    const int slots_len, //  num_query
    const int queries_len, // max_len
    const int num_cams,
    cudaStream_t stream[]);

template <typename T>
void slots_prod_count_norm(
    T* last_slots, // [bs, num_query, embed_dims]
    const T* slots, // [num_cams, num_query, embed_dims]
    const T* count_norm, // [bs, num_query]
    const int stride,
    const int num_query,
    cudaStream_t stream = 0);
