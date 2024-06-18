
#include "rebatch.h"

// queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
// reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])

//// bs = 1
// for j in range(bs):
//     for i, reference_points_per_img in enumerate(reference_points_cam):
//         index_query_per_img = indexes[i]
//         queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
//         reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

// index 1 2 5 7 11 89 623 0 0 ... 0
template <typename T>
__global__ void rebatch_ker(
    T* queries_rebatch,
    T* reference_points_rebatch,
    const T* query,
    const T* reference_points_per_img,
    const int* index, // attach with each bs, include -1
    const int index_len,
    const int query_stride,
    const int reference_points_stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < index_len) {
    int data_id = index[tid];

    // if (data_id >= 0) // for `n` has limit the valid range
    {
      memcpy(queries_rebatch + tid * query_stride, query + data_id * query_stride, query_stride * sizeof(T));
      memcpy(
          reference_points_rebatch + tid * reference_points_stride,
          reference_points_per_img + data_id * reference_points_stride,
          reference_points_stride * sizeof(T));
    }
  }
}

template <typename T>
void rebatch(
    T* queries_rebatch,
    T* reference_points_rebatch,
    const T* query,
    const T* reference_points_cam,
    const int* index,
    const int index_len[], // valid len
    const int query_stride, // embed_dims
    const int reference_points_stride, // D*2
    const int rebatch_len, // max_len
    const int reference_points_cam_len, // num_query
    const int num_cams,
    cudaStream_t stream[]) {
  const int thread_in_block = 512;

  for (int i = 0; i < num_cams; ++i) {
    rebatch_ker<<<(index_len[i] + thread_in_block - 1) / thread_in_block, thread_in_block, 0, stream[i]>>>(
        queries_rebatch + i * rebatch_len * query_stride,
        reference_points_rebatch + i * rebatch_len * reference_points_stride,
        query,
        reference_points_cam + i * reference_points_cam_len * reference_points_stride,
        index + i * rebatch_len,
        index_len[i],
        query_stride,
        reference_points_stride);
    cudaStreamSynchronize(stream[i]);
  }
}

template void rebatch<float>(
    float* queries_rebatch,
    float* reference_points_rebatch,
    const float* query,
    const float* reference_points_cam,
    const int* index,
    const int index_len[],
    const int query_stride,
    const int reference_points_stride,
    const int rebatch_len,
    const int reference_points_cam_len,
    const int num_cams,
    cudaStream_t stream[]);

template void rebatch<half>(
    half* queries_rebatch,
    half* reference_points_rebatch,
    const half* query,
    const half* reference_points_cam,
    const int* index,
    const int index_len[],
    const int query_stride,
    const int reference_points_stride,
    const int rebatch_len,
    const int reference_points_cam_len,
    const int num_cams,
    cudaStream_t stream[]);
