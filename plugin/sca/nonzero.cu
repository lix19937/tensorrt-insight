// Get inspiration from
// https://github.com/pytorch/pytorch/blob/65aa16f968af2cd18ff8c25cc657e7abda594bfc/aten/src/ATen/native/cuda/Nonzero.cu
// https://github.com/chenwanqq/candle-nonzero
// https://github.com/dongqixu/nonzero

#include <stdint.h>
#include <cub/cub.cuh>
#include "nonzero.h"

// for i, mask_per_img in enumerate(bev_mask):
//     index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
//     indexes.append(index_query_per_img)
// max_len = max([len(each) for each in indexes])

// count = bev_mask.sum(-1) > 0
// logger.info(pt("count", count)) # [6, 1, 40000]  torch.bool
// count = count.permute(1, 2, 0)  # [6, 1, 40000] -> [1, 40000, 6]

namespace {
struct NonZeroOp {
  __host__ __device__ __forceinline__ bool operator()(const char4& a) const {
    return (a.x != 0 || a.y != 0 || a.z != 0 || a.w != 0);
  }
};

__global__ void write_indices_ker(
    const int* temp_indices,
    const int num_nonzero,
    const int offset,
    const int h,
    const int w,
    int* d_indexs, /* nonzero output */
    uchar* d_count) { /* d_count just transpose, keep dims */
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_nonzero) {
    int temp_index = temp_indices[idx];
    d_indexs[idx] = temp_index;

    /// follow is do transpose
    int cur_idx = temp_index + offset;
    auto w_idx = cur_idx % w;
    auto h_idx = cur_idx / w;
    d_count[w_idx * h + h_idx] = 1;
  }
}
} // namespace

void NonZero::__nonzero(
    int* h_indexs_len,
    int* d_indexs,
    uchar* d_count,
    const char4* d_in,
    const int N,
    const int NC,
    const int threshold,
    const int idx,
    cudaStream_t stream) {
  using T = char4;
  // get d_num_nonzero_
  cub::TransformInputIterator<bool, NonZeroOp, const T*> itr(d_in, NonZeroOp());
  temp_storage_bytes_[idx] = 0;

  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_[idx], itr, d_num_nonzero_[idx], N, stream);
  cub::DeviceReduce::Sum(d_temp_storage_[idx], temp_storage_bytes_[idx], itr, d_num_nonzero_[idx], N, stream);

  // get num_nonzeros_output (d_out_temp_)
  cub::CountingInputIterator<int> counting_itr(0);

  temp_storage_bytes_[idx] = 0;
  cub::DeviceSelect::Flagged(
      nullptr /* */, temp_storage_bytes_[idx], counting_itr, itr, d_out_temp_[idx], d_num_nonzero_[idx], N, stream);
  cub::DeviceSelect::Flagged(
      d_temp_storage_[idx],
      temp_storage_bytes_[idx],
      counting_itr,
      itr,
      d_out_temp_[idx],
      d_num_nonzero_[idx],
      N,
      stream);

  int num_nonzero;
  cudaMemcpyAsync(&num_nonzero, d_num_nonzero_[idx], sizeof(int), cudaMemcpyDeviceToHost, stream);

  *h_indexs_len = std::min(threshold, num_nonzero);

  int nthreads = 512;
  int nblocks = (num_nonzero + nthreads - 1) / nthreads;
  write_indices_ker<<<nblocks, nthreads, 0, stream>>>(
      d_out_temp_[idx], *h_indexs_len, idx * N, NC, N, d_indexs, d_count);
}

void NonZero::nonzero(
    int* h_indexs_len, // [num_cams]
    int* d_indexs, // [num_cams, max_len]
    uchar* d_count, // [num_query, (num_cams + pad)]
    const char4* d_in, // [num_query, num_cams]
    const int N, //  num_query
    const int stride, // num_cams + pad
    const int threshold, // max_len
    const int num_cams,
    cudaStream_t stream[]) {
  for (int i = 0; i < num_cams; ++i) {
    __nonzero(h_indexs_len + i, d_indexs + i * threshold, d_count, d_in + i * N, N, stride, threshold, i, stream[i]);

    cudaStreamSynchronize(stream[i]);
  }
}
