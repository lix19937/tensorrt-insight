
#include <stdio.h>
#include <vector>
#include "rebatch.h"

template <typename T>
int test() {
  int bs = 1, num_cams = 6, max_len = 8, embed_dims = 6, D = 4, num_query = 12;

  // clang-format off
  /// in
  T *d_query, *d_reference_points_cam;
  cudaMalloc((void**)&d_query,                bs * num_query * embed_dims * sizeof(T));
  cudaMalloc((void**)&d_reference_points_cam, bs * num_cams * num_query * D * 2 * sizeof(T));

  std::vector<T> h_query(bs * num_query * embed_dims);
  for (int i = 0; i < h_query.size(); ++i)
    h_query[i] = i*1.f;
  cudaMemcpy(d_query, h_query.data(), h_query.size() * sizeof(T), cudaMemcpyHostToDevice);

  std::vector<T> h_reference_points_cam(bs * num_cams * num_query * D * 2);
  for (int i = 0; i < h_reference_points_cam.size(); ++i)
    h_reference_points_cam[i] = i*1.f;
  cudaMemcpy(
      d_reference_points_cam,
      h_reference_points_cam.data(),
      h_reference_points_cam.size() * sizeof(T),
      cudaMemcpyHostToDevice);

  /// out
  T *d_queries_rebatch, *d_reference_points_rebatch;
  cudaMalloc((void**)&d_queries_rebatch,          bs * num_cams * max_len * embed_dims * sizeof(T));
  cudaMalloc((void**)&d_reference_points_rebatch, bs * num_cams * max_len * D * 2 * sizeof(T));
  cudaMemset(d_queries_rebatch, 0,                bs * num_cams * max_len * embed_dims * sizeof(T));
  cudaMemset(d_reference_points_rebatch, 0,       bs * num_cams * max_len * D * 2 * sizeof(T));
  // clang-format on

  std::vector<std::vector<int>> h_indexs{{1, 2, 8, 9}, {0, 2, 4, 6}, {7}, {8}, {9}, {10}};

  int indexs_len[num_cams]{0};
  int* d_indexs;
  cudaMalloc((void**)&d_indexs, num_cams * max_len * sizeof(int));

  for (int i = 0; i < num_cams; ++i) {
    int idx_len = h_indexs[i].size();
    indexs_len[i] = idx_len;

    cudaMemcpy(d_indexs + i * max_len, h_indexs[i].data(), idx_len * sizeof(int), cudaMemcpyHostToDevice);
  }

  cudaStream_t stream[num_cams];
  for (int i = 0; i < num_cams; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  rebatch(
      d_queries_rebatch,
      d_reference_points_rebatch,
      d_query,
      d_reference_points_cam,
      (const int*)d_indexs,
      indexs_len,
      embed_dims,
      D * 2,
      max_len,
      num_query,
      num_cams,
      stream);

  /////// debug
  std::vector<T> h_last_out_1(bs * num_cams * max_len * embed_dims);
  cudaMemcpy(h_last_out_1.data(), d_queries_rebatch, h_last_out_1.size() * sizeof(T), cudaMemcpyDeviceToHost);

  std::vector<T> h_last_out_2(bs * num_cams * max_len * D * 2);
  cudaMemcpy(h_last_out_2.data(), d_reference_points_rebatch, h_last_out_2.size() * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h_query.size(); ++i) {
    if (i % embed_dims == 0) {
      printf("\n %2d| ", i / embed_dims);
    }
    printf("%-6.1f", float(h_query[i]));
  }
  printf("\n\n");

  for (int i = 0; i < h_last_out_1.size(); ++i) {
    if (i % embed_dims == 0) {
      printf("\n");
    }
    if (i % (max_len * embed_dims) == 0) {
      printf("\n");
    }

    printf("%-6.1f ", float(h_last_out_1[i]));
  }
  printf("\n\n - end -\n");

  for (int i = 0; i < h_reference_points_cam.size(); ++i) {
    if (i % (num_query * D * 2) == 0) {
      printf("\n");
    }
    if (i % (D * 2) == 0) {
      printf("\n %2d| ", i / (D * 2) % num_query);
    }
    printf("%-6.1f", float(h_reference_points_cam[i]));
  }
  printf("\n");

  for (int i = 0; i < h_last_out_2.size(); ++i) {
    if (i % (D * 2) == 0) {
      printf("\n");
    }
    if (i % (max_len * D * 2) == 0) {
      printf("\n");
    }
    printf("%-6.1f", float(h_last_out_2[i]));
  }
  printf("\n");

  cudaFree(d_queries_rebatch);
  cudaFree(d_reference_points_rebatch);
  cudaFree(d_query);
  cudaFree(d_reference_points_cam);
  cudaFree(d_indexs);

  for (int i = 0; i < num_cams; ++i) {
    cudaStreamDestroy(stream[i]);
  }
  return 0;
}

int main() {
  return test<half>();
}

// nvcc -std=c++14 -arch=sm_86 -O2 ./test_rebatch.cpp rebatch.cu
