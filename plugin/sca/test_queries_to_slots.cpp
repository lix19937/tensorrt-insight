

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "queries_to_slots.h"

template <typename T>
int main1() {
  const int num_cams = 6, bs = 1, num_query = 12, pad = 2;

  // clang-format off
  /// in
  uchar* d_count;
  cudaMalloc((void **)&d_count,       bs * (num_cams + pad) * num_query * sizeof(uchar));
  std::vector<uchar> h_count(bs * (num_cams + pad) * num_query);
  for (int i = 0; i < h_count.size(); ++i) {
    h_count[i] = rand()%2;
    if (i % (num_cams + pad) >= num_cams){
      h_count[i] = 0;
    }
  }
  cudaMemcpy(d_count, h_count.data(), h_count.size() * sizeof(uchar), cudaMemcpyHostToDevice);

  /// out
  T* d_out;
  cudaMalloc((void **)&d_out, bs * num_query * sizeof(T));
  // clang-format on

  for (int i = 0; i < h_count.size(); ++i) {
    if (i % (num_cams + pad) == 0) {
      printf("\n");
    }
    printf("%-4d", int(h_count[i]));
  }
  printf("\n\n");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  count_norm(d_out, d_count, num_cams, num_query, stream); /////-------------------

  cudaStreamSynchronize(stream);

  std::vector<T> h_out(bs * num_query);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h_out.size(); ++i) {
    printf("\n%-4.5f", float(h_out[i]));
  }
  printf("\n");

  cudaFree(d_out);
  cudaFree(d_count);
  cudaStreamDestroy(stream);

  return 0;
}

template <typename T>
int main2() {
  const int num_cams = 6, bs = 1, max_len = 8, embed_dims = 6, num_query = 12;

  std::vector<std::vector<int>> h_indexs{{1, 2, 8, 9}, {0, 2, 4, 6}, {7}, {8}, {9}, {10}};

  int indexs_len[num_cams]{0};
  int* d_indexs;
  cudaMalloc((void**)&d_indexs, num_cams * max_len * sizeof(int));

  for (int i = 0; i < num_cams; ++i) {
    int idx_len = h_indexs[i].size();
    indexs_len[i] = idx_len;

    cudaMemcpy(d_indexs + i * max_len, h_indexs[i].data(), idx_len * sizeof(int), cudaMemcpyHostToDevice);
  }
  // clang-format off

  /// in
  T* d_queries;
  cudaMalloc((void **)&d_queries,       bs * num_cams * max_len * embed_dims * sizeof(T));
  std::vector<T> h_queries(         bs * num_cams * max_len * embed_dims);
  for (int i = 0; i < h_queries.size(); ++i) h_queries[i] = i*1.f;
  cudaMemcpy(d_queries, h_queries.data(), h_queries.size() * sizeof(T), cudaMemcpyHostToDevice);

  T* d_count;
  cudaMalloc((void **)&d_count,       bs * num_query * sizeof(T));
  std::vector<T> h_count(             bs * num_query);
  for (int i = 0; i < h_count.size(); ++i) h_count[i] = 1.f;
  cudaMemcpy(d_count, h_count.data(), h_count.size() * sizeof(T), cudaMemcpyHostToDevice);

  /// out
  T* d_slots;
  cudaMalloc((void **)&d_slots,   bs * num_cams * num_query * embed_dims * sizeof(T));
  cudaMemset(d_slots, 0,          bs * num_cams * num_query * embed_dims * sizeof(T)); // !!! must set 
  
  /// last out
  T* d_last_slots;
  cudaMalloc((void **)&d_last_slots, bs * num_query * embed_dims * sizeof(T));
  // clang-format on

  cudaStream_t stream[num_cams];
  for (int i = 0; i < num_cams; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  queries_to_slots(
      d_slots, (const T*)d_queries, (const int*)d_indexs, indexs_len, embed_dims, num_query, max_len, num_cams, stream);

  slots_prod_count_norm(d_last_slots, d_slots, d_count, embed_dims, num_query, stream[0]); /////-------------------

  std::vector<T> h_slots(bs * num_cams * num_query * embed_dims);
  cudaMemcpy(h_slots.data(), d_slots, h_slots.size() * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h_queries.size(); ++i) {
    if (i % embed_dims == 0) {
      printf("\n");
    }
    if (i % (max_len * embed_dims) == 0) {
      printf("\n");
    }
    printf("%-6.0f", float(h_queries[i]));
  }
  printf("\n\n ---scatter--- ");

  for (int i = 0; i < h_slots.size(); ++i) {
    if (i % (num_query * embed_dims) == 0) {
      printf("\n");
    }
    if (i % embed_dims == 0) {
      printf("\n %2d| ", i / embed_dims % num_query);
    }
    printf("%-6.1f", float(h_slots[i]));
  }
  printf("\n\n --- sum(0)--- ");

  std::vector<T> h_last_slots(bs * num_query * embed_dims);
  cudaMemcpy(h_last_slots.data(), d_last_slots, h_last_slots.size() * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h_last_slots.size(); ++i) {
    if (i % (num_query * embed_dims) == 0) {
      printf("\n");
    }
    if (i % embed_dims == 0) {
      printf("\n %2d| ", i / embed_dims % num_query);
    }
    printf("%-6.1f", float(h_last_slots[i]));
  }
  printf("\n");

  cudaFree(d_slots);
  cudaFree(d_queries);
  cudaFree(d_count);
  cudaFree(d_indexs);

  for (int i = 0; i < num_cams; ++i) {
    cudaStreamDestroy(stream[i]);
  }

  return 0;
}

template int main1<float>();
template int main1<half>();
template int main2<float>();
template int main2<half>();

int main() {
  main1<half>();
  printf("\n ----- \n");
  main2<half>();
  return 0;
}

// nvcc -std=c++14 -arch=sm_86 -O2 ./test_queries_to_slots.cpp ./queries_to_slots.cu
