// Get inspiration from
// https://github.com/pytorch/pytorch/blob/65aa16f968af2cd18ff8c25cc657e7abda594bfc/aten/src/ATen/native/cuda/Nonzero.cu
// https://github.com/chenwanqq/candle-nonzero
// https://github.com/dongqixu/nonzero

#include <stdint.h>
#include <vector>
#include <cub/cub.cuh>

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

// get the indices of non-zero elements in an array
void nonzero(const char4* d_in, const int32_t N, const uint32_t requ_num_nonzero, uint32_t* d_out) {
  using T = char4;
  cub::TransformInputIterator<bool, NonZeroOp, const T*> itr(d_in, NonZeroOp());
  size_t temp_storage_bytes = 0;
  size_t* d_num_nonzero;
  cudaMalloc((void**)&d_num_nonzero, sizeof(uint32_t));
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr, d_num_nonzero, N);

  void** d_temp_storage;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr, d_num_nonzero, N);
  uint32_t num_nonzero;
  cudaMemcpy(&num_nonzero, d_num_nonzero, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_temp_storage);
  printf("#   temp_storage_bytes = %lu\n", temp_storage_bytes);

  // -------------------------------------------------------------------------

  cub::CountingInputIterator<uint32_t> counting_itr(0);
  uint32_t* d_out_temp;
  cudaMalloc((void**)&d_out_temp, num_nonzero * sizeof(uint32_t));

  temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr, d_out_temp, d_num_nonzero, N);
  printf("##  temp_storage_bytes = %lu\n", temp_storage_bytes);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_itr, itr, d_out_temp, d_num_nonzero, N);
  printf("### temp_storage_bytes = %lu\n", temp_storage_bytes);

  num_nonzero = std::min(requ_num_nonzero, num_nonzero);

  const int nthreads = 512;
  const int nblocks = (num_nonzero + nthreads - 1) / nthreads;
  transform_indices<<<nblocks, nthreads>>>(d_out_temp, num_nonzero, d_out);

  cudaDeviceSynchronize();
  cudaFree(d_temp_storage);
  cudaFree(d_num_nonzero);
  cudaFree(d_out_temp);
}

int main() {
  const int N = 40000;
  const int T = 25;
  uint32_t required_nonzero = T;

  std::vector<char4> h_in(N);
  h_in[0] = {0, 0, 0, 0};
  h_in[1] = {0, 0, 0, 0};
  h_in[2] = {1, 0, 0, 0};
  h_in[3] = {0, 0, 0, 0};
  h_in[4] = {0, 0, 0, 0};
  h_in[5] = {0, 0, 0, 0};
  h_in[6] = {0, 0, 0, 0};
  h_in[7] = {1, 0, 0, 0};
  h_in[8] = {1, 1, 0, 0};
  h_in[9] = {1, 1, 0, 0};

  h_in[N - 1] = {1, 1, 0, 0};

  char4* d_in;
  cudaMalloc(&d_in, N * sizeof(char4));
  cudaMemcpy(d_in, h_in.data(), N * sizeof(char4), cudaMemcpyHostToDevice);

  std::vector<uint32_t> h_last_out(T);
  uint32_t* d_out;
  cudaMalloc(&d_out, required_nonzero * sizeof(uint32_t));

  nonzero(d_in, N, required_nonzero, d_out);

  cudaMemcpy(h_last_out.data(), d_out, required_nonzero * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  for (int i = 0; i < required_nonzero; ++i)
    std::cout << "#>> " << h_last_out[i] << "\n";

  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}
// nvcc -std=c++14 -arch=sm_86 -O2 ./nonzero.cu
