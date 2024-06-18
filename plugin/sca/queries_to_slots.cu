

#include <stdio.h>
#include <algorithm>
#include "queries_to_slots.h"

// count = count.sum(-1) #  [1, 40000, 6]    NOTE !!!   6 expand to 8
// logger.info(pt("count", count)) # [1, 40000]  torch.int64
// count = torch.clamp(count, min=1.0)
// logger.info(pt("slots", slots)) # [1, 40000, 256])  torch.float32
// logger.info(pt("count[..., None]", count[..., None])) # [1, 40000, 1]
// count = 1 / count[..., None] # [1, 40000, 1]
template <typename T>
__global__ void count_reducesum_clamp_ker(T* count_norm, const uint2* count, const int n) {
  // uchar4
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    uint2 v = count[tid];
    uchar4* a = (uchar4*)&(v.x);
    uchar4* b = (uchar4*)&(v.y);
    count_norm[tid] = 1.f / max(a->x + a->y + a->z + a->w + b->x + b->y, 1);
    // printf("%d %d %d %d %d %d \n", a->x, a->y, a->z, a->w, b->x, b->y);
  }
}

// logger.info(pt("queries", queries)) # [1, 6, max_len, 256]
// logger.info(pt("slots", slots)) # [1, 40000, 256]  torch.float32
// for j in range(bs):
//     for i, index_query_per_img in enumerate(indexes):
//         slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
template <typename T>
__global__ void query_to_slots_ker(
    T* slot, // note, here slots shape is [6, 40000, 256], not [1, 40000, 256], I use follow kernel to torch.sum(0)
    const T* queries,
    const int* index, // attach with each bs, include -1
    const int n,
    const int stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    int data_id = index[tid];

    // if (data_id >= 0)  if numzero can be return right, need not if judge
    memcpy(slot + data_id * stride, queries + tid * stride, stride * sizeof(T));
  }
}

/// has bug
template <typename T>
__global__ void query_to_slots_ker_batch(
    T* slot, // note, here slots shape is [6, 40000, 256], not [1, 40000, 256], I use follow kernel to torch.sum(0)
    const T* queries,
    const int* index[], // attach with each bs, include -1
    const int n[],
    const int slots_len,
    const int queries_len,
    const int stride,
    const int num_cams) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < num_cams; ++i) {
    if (tid < n[i]) {
      int data_id = index[i][tid];

      // clang-format off
      // if (data_id >= 0)  if numzero can be return right, need not if judge
      memcpy(
          slot    + i * slots_len * stride   + data_id * stride,
          queries + i * queries_len * stride + tid     * stride,
          stride * sizeof(T));
      // clang-format on
    }
  }
}

// slots = slots * count[..., None]   /// make sure w % 4 == 0
template <typename T>
__global__ void slots_prod_count_norm_ker(
    T* out, const T* slots, const T* count_norm, const int w, const int plane, const int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = (slots[tid] + slots[tid + plane] + slots[tid + plane * 2] + slots[tid + plane * 3] +
                slots[tid + plane * 4] + slots[tid + plane * 5]) *
        count_norm[tid / w];
  }
}

/// need the last shape of out/slots is even
__global__ void slots_prod_count_norm_ker_half2(
    half2* out, const half2* slots, const half* count_norm, const int w, const int plane, const int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    auto t = count_norm[tid / w];
    half2 scale{t, t};
    out[tid] = (slots[tid] + slots[tid + plane] + slots[tid + plane * 2] + slots[tid + plane * 3] +
                slots[tid + plane * 4] + slots[tid + plane * 5]) *
        scale;
  }
}

template <typename T>
void count_norm(T* count_norm, const uchar* d_count, const int num_cams, const int num_query, cudaStream_t stream) {
  int nthreads = 512;
  int nblocks = (num_query + nthreads - 1) / nthreads;
  if (num_cams != 6) {
    abort();
  }
  count_reducesum_clamp_ker<<<nblocks, nthreads, 0, stream>>>(count_norm, (uint2*)d_count, num_query);
}

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
    cudaStream_t stream[]) {
  const int thread_in_block = 512;

  for (int i = 0; i < num_cams; ++i) {
    query_to_slots_ker<<<(index_len[i] + thread_in_block - 1) / thread_in_block, thread_in_block, 0, stream[i]>>>(
        slots + i * slots_len * stride,
        queries + i * queries_len * stride,
        index + i * queries_len,
        index_len[i],
        stride);
    cudaStreamSynchronize(stream[i]);
  }
  // int max_index_len = *std::max_element(index_len, index_len + num_cams);

  // query_to_slots_ker_batch<<<(max_index_len + thread_in_block - 1) / thread_in_block, thread_in_block, 0,
  // stream[0]>>>(
  //     slots, queries, index, index_len, slots_len, queries_len, stride, num_cams);
  // cudaStreamSynchronize(stream[0]);
}

template <typename T>
void slots_prod_count_norm(
    T* last_slots,
    const T* slots, // [num_cams, num_query, embed_dims]
    const T* count_norm, // [num_query]
    const int stride,
    const int num_query,
    cudaStream_t stream) {
  const int thread_in_block = 512;
  if (std::is_same<T, half>::value) {
    slots_prod_count_norm_ker_half2<<<
        (num_query * stride / 2 + thread_in_block - 1) / thread_in_block,
        thread_in_block,
        0,
        stream>>>(
        (half2*)last_slots,
        (const half2*)slots,
        (const half*)count_norm,
        stride / 2,
        num_query * stride / 2,
        num_query * stride / 2);
  } else {
    slots_prod_count_norm_ker<<<
        (num_query * stride + thread_in_block - 1) / thread_in_block,
        thread_in_block,
        0,
        stream>>>(last_slots, slots, count_norm, stride, num_query * stride, num_query * stride);
  }
}

template void count_norm<float>(
    float* count_norm, const uchar* d_count, const int num_cams, const int num_query, cudaStream_t stream);
template void count_norm<half>(
    half* count_norm, const uchar* d_count, const int num_cams, const int num_query, cudaStream_t stream);

template void queries_to_slots<float>(
    float* slots,
    const float* queries,
    const int* index,
    const int index_len[],
    const int stride,
    const int slots_len,
    const int queries_len,
    const int num_cams,
    cudaStream_t stream[]);
template void queries_to_slots<half>(
    half* slots,
    const half* queries,
    const int* index,
    const int index_len[],
    const int stride,
    const int slots_len,
    const int queries_len,
    const int num_cams,
    cudaStream_t stream[]);

template void slots_prod_count_norm<float>(
    float* last_slots,
    const float* slots,
    const float* count_norm,
    const int stride,
    const int num_query,
    cudaStream_t stream);
template void slots_prod_count_norm<half>(
    half* last_slots,
    const half* slots,
    const half* count_norm,
    const int stride,
    const int num_query,
    cudaStream_t stream);
