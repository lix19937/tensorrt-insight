// Get inspiration from
// https://github.com/pytorch/pytorch/blob/65aa16f968af2cd18ff8c25cc657e7abda594bfc/aten/src/ATen/native/cuda/Nonzero.cu
// https://github.com/chenwanqq/candle-nonzero
// https://github.com/dongqixu/nonzero

#include <stdint.h>
#include <vector>
#include <cub/cub.cuh>

// for i, mask_per_img in enumerate(bev_mask):
//     index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
//     indexes.append(index_query_per_img)
// max_len = max([len(each) for each in indexes])

namespace {
struct NonZeroOp {
  __host__ __device__ __forceinline__ bool operator()(const char4& a) const {
    return (a.x != 0 || a.y != 0 || a.z != 0 || a.w != 0);
  }
};

__global__ void transform_indices(const uint32_t* temp_indices, const uint32_t num_nonzero, uint32_t* d_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_nonzero) {
    int temp_index = temp_indices[idx];
    d_out[idx] = temp_index;
  }
}
} // namespace

class NonZero {
 public:
  NonZero(const int32_t N) {
    cudaMalloc((void**)&d_num_nonzero_, sizeof(uint32_t));
    cudaMalloc(&d_temp_storage_, 6000); // N
    cudaMalloc((void**)&d_out_temp_, N * sizeof(uint32_t));
  }

  ~NonZero() {
    cudaFree(d_out_temp_);
    cudaFree(d_num_nonzero_);
    cudaFree(d_temp_storage_);
  }

  void nonzero(
      const char4* d_in, const int32_t N, const uint32_t requ_num_nonzero, uint32_t* d_out, cudaStream_t stream);

 private:
  size_t temp_storage_bytes_ = 0;
  size_t* d_num_nonzero_;
  void** d_temp_storage_;
  uint32_t* d_out_temp_;
};

void NonZero::nonzero(
    const char4* d_in, const int32_t N, const uint32_t requ_num_nonzero, uint32_t* d_out, cudaStream_t stream) {
  using T = char4;
///  get d_num_nonzero_
  cub::TransformInputIterator<bool, NonZeroOp, const T*> itr(d_in, NonZeroOp());
  temp_storage_bytes_ = 0;
  printf("#0 temp_storage_bytes_ = %lu\n", temp_storage_bytes_);

  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_, itr, d_num_nonzero_, N, stream);
  cudaStreamSynchronize(stream); // !!!
  printf("#1 temp_storage_bytes_ = %lu\n", temp_storage_bytes_);

  cub::DeviceReduce::Sum(d_temp_storage_, temp_storage_bytes_, itr, d_num_nonzero_, N, stream);
  cudaStreamSynchronize(stream); // !!!

  uint32_t num_nonzero;
  cudaMemcpyAsync(&num_nonzero, d_num_nonzero_, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream); // !!!
  printf("#2 temp_storage_bytes_ = %lu\n", temp_storage_bytes_);

/// get num_nonzeros_output (d_out_temp_)   
  // -------------------------------------------------------------------------
  cub::CountingInputIterator<uint32_t> counting_itr(0);

  temp_storage_bytes_ = 0;
  cub::DeviceSelect::Flagged(
      nullptr /* */, temp_storage_bytes_, counting_itr, itr, d_out_temp_, d_num_nonzero_, N, stream);
  printf("#3 temp_storage_bytes_ = %lu\n", temp_storage_bytes_);

  cub::DeviceSelect::Flagged(
      d_temp_storage_, temp_storage_bytes_, counting_itr, itr, d_out_temp_, d_num_nonzero_, N, stream);
  printf("#4 temp_storage_bytes_ = %lu\n", temp_storage_bytes_);

/// assign to sparse d_out 
  num_nonzero = std::min(requ_num_nonzero, num_nonzero);

  int nthreads = 512;
  int nblocks = (num_nonzero + nthreads - 1) / nthreads;
  transform_indices<<<nblocks, nthreads, 0, stream>>>(d_out_temp_, num_nonzero, d_out);
}

#define __DO__(idx)          \
  cudaStream_t s##idx;       \
  cudaStreamCreate(&s##idx); \
  NonZero nz##idx(N);        \
  nz##idx.nonzero(d_in, N, required_nonzero, d_out, s##idx);

#define __UNDO__(idx) cudaStreamDestroy(s##idx);

int main() {
  const int N = 40000;
  const int T = 3600;
  uint32_t required_nonzero = T;

  std::vector<char4> h_in(N);
  h_in[0] = {0, 0, 0, 0};
  h_in[1] = {0, 0, 0, 0};
  h_in[2] = {1, 0, 0, 0};
  h_in[3] = {0, 0, 0, 0};
  h_in[4] = {0, 0, 0, 0};
  h_in[5] = {0, 0, 0, 0};
  h_in[6] = {0, 0, 1, 0};
  h_in[7] = {1, 0, 0, 0};
  h_in[8] = {1, 1, 0, 0};
  h_in[9] = {1, 1, 0, 0};
  h_in[N - 10] = {1, 1, 0, 0};
  h_in[N - 1] = {1, 1, 0, 0};

  char4* d_in;
  cudaMalloc(&d_in, N * sizeof(char4));
  cudaMemcpy(d_in, h_in.data(), N * sizeof(char4), cudaMemcpyHostToDevice);

  std::vector<uint32_t> h_last_out(T);
  uint32_t* d_out;
  cudaMalloc(&d_out, required_nonzero * sizeof(uint32_t));

  __DO__(1)
  __DO__(2)
  __DO__(3)
  __DO__(4)
  __DO__(5)
  __DO__(6)

  // cudaStreamSynchronize(s1);
  cudaDeviceSynchronize();

  cudaMemcpy(h_last_out.data(), d_out, required_nonzero * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  for (int i = 0; i < required_nonzero; ++i)
    if (h_last_out[i] > 0)
      std::cout << "#>> " << h_last_out[i] << "\n";

  cudaFree(d_in);
  cudaFree(d_out);
  __UNDO__(1)
  __UNDO__(2)
  __UNDO__(3)
  __UNDO__(4)
  __UNDO__(5)
  __UNDO__(6)

  return 0;
}
// nvcc -std=c++14 -arch=sm_86 -O2 ./test_nonzero.cu
