

#include <stdio.h>
#include <vector>
#include "nonzero.h"
#include "queries_to_slots.h"
#include "rebatch.h"

template <typename T>
int test() {
  int bs = 1, num_cams = 6, max_len = 8, embed_dims = 6, D = 4, num_query = 12;
  int threshold = max_len, pad = 2;

  // clang-format off
  /// in 
  // #in from user  
  T *d_query, *d_reference_points_cam;
  cudaMalloc((void **)&d_query,                      bs * num_query * embed_dims * sizeof(T));
  cudaMalloc((void **)&d_reference_points_cam,       bs * num_cams * num_query * D * 2 * sizeof(T));

  std::vector<T> h_query(bs * num_query * embed_dims);
  for (int i = 0; i < h_query.size(); ++i)  h_query[i] = i*1.f; /// assign
  cudaMemcpy(d_query, h_query.data(), h_query.size() * sizeof(T), cudaMemcpyHostToDevice);

  std::vector<T> h_reference_points_cam(bs * num_cams * num_query * D * 2);
  for (int i = 0; i < h_reference_points_cam.size(); ++i)  h_reference_points_cam[i] = i*1.f;  /// assign
  cudaMemcpy(
      d_reference_points_cam,
      h_reference_points_cam.data(),
      h_reference_points_cam.size() * sizeof(T),
      cudaMemcpyHostToDevice);
  
  /// #in from user 
  char4* d_bev_mask;
  cudaMalloc((void**)&d_bev_mask, num_cams * num_query * sizeof(char4));
  std::vector<char4> h_bev_mask(num_cams * num_query);
  h_bev_mask[                1]  = {1, 0, 0, 0};
  h_bev_mask[                2]  = {0, 0, 1, 0};
  h_bev_mask[                8]  = {1, 1, 0, 0};
  h_bev_mask[                9]  = {1, 1, 0, 0};

  h_bev_mask[num_query     + 0]  = {0, 0, 0, 1};
  h_bev_mask[num_query     + 2]  = {1, 0, 0, 0};
  h_bev_mask[num_query     + 4]  = {1, 0, 0, 0};
  h_bev_mask[num_query     + 6]  = {1, 0, 0, 0};
  
  h_bev_mask[num_query * 2 + 7]  = {0, 0, 0, 1};
  h_bev_mask[num_query * 3 + 8]  = {0, 0, 0, 1};
  h_bev_mask[num_query * 4 + 9]  = {0, 0, 0, 1};
  h_bev_mask[num_query * 5 + 10] = {0, 0, 0, 1};
  cudaMemcpy(d_bev_mask, h_bev_mask.data(), num_cams * num_query * sizeof(char4), cudaMemcpyHostToDevice);

  /// #in 
  T* d_slots;
  cudaMalloc((void **)&d_slots,   bs * num_cams * num_query * embed_dims * sizeof(T));
  cudaMemset(d_slots, 0,          bs * num_cams * num_query * embed_dims * sizeof(T)); // !!! must set
  
  /// #in from msda 
  T* d_queries;  
  cudaMalloc((void **)&d_queries,       bs * num_cams * max_len * embed_dims * sizeof(T));
  std::vector<T> h_queries(             bs * num_cams * max_len * embed_dims);
  for (int i = 0; i < h_queries.size(); ++i) h_queries[i] = i*0.1f; /// 
  cudaMemcpy(d_queries, h_queries.data(), h_queries.size() * sizeof(T), cudaMemcpyHostToDevice);

  /// out of step2
  T *d_queries_rebatch, *d_reference_points_rebatch;
  cudaMalloc((void **)&d_queries_rebatch,            bs * num_cams * max_len * embed_dims * sizeof(T));
  cudaMalloc((void **)&d_reference_points_rebatch,   bs * num_cams * max_len * D * 2 * sizeof(T));
  cudaMemset(d_queries_rebatch,                   0, bs * num_cams * max_len * embed_dims * sizeof(T)); /// must set 
  cudaMemset(d_reference_points_rebatch,          0, bs * num_cams * max_len * D * 2 * sizeof(T)); /// must set

  /// out of step1
  int* d_indexs;
  cudaMalloc((void**)&d_indexs, threshold * num_cams * sizeof(int));
  // cudaMemset(d_indexs, -1,      threshold * num_cams * sizeof(int)); /// need not 

  /// out of step1
  uchar* d_count;
  cudaMalloc((void**)&d_count, num_query * (num_cams + pad) * sizeof(uchar));
  // cudaMemset(d_count, 0,       num_query * (num_cams + pad) * sizeof(uchar)); /// just for debug, need not memset 

  /// out of step3
  T* d_count_norm;
  cudaMalloc((void**)&d_count_norm, num_query * sizeof(T));

  /// last out
  T* d_last_slots;
  cudaMalloc((void **)&d_last_slots, bs * num_query * embed_dims * sizeof(T));
  // clang-format on

  cudaStream_t stream[num_cams];
  for (int i = 0; i < num_cams; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  /////// step1
  NonZero nz(num_query, num_cams);
  int h_indexs_len[num_cams];
  nz.nonzero(h_indexs_len, d_indexs, d_count, d_bev_mask, num_query, num_cams + pad, threshold, num_cams, stream);

  /////// step2
  count_norm(d_count_norm, d_count, num_cams, num_query, stream[0]);
  cudaStreamSynchronize(stream[0]);
  // check d_count_norm

  /////// step3
  rebatch(
      d_queries_rebatch,
      d_reference_points_rebatch,
      d_query,
      d_reference_points_cam,
      (const int*)d_indexs,
      h_indexs_len,
      embed_dims,
      D * 2,
      max_len,
      num_query,
      num_cams,
      stream);
  // check d_queries_rebatch  d_reference_points_rebatch

  ////// step4
  queries_to_slots(
      d_slots,
      (const T*)d_queries,
      (const int*)d_indexs,
      h_indexs_len,
      embed_dims,
      num_query,
      max_len,
      num_cams,
      stream);
  // check d_slots

  ////// step5
  slots_prod_count_norm(d_last_slots, d_slots, d_count_norm, embed_dims, num_query, stream[0]); /////-------------------
  cudaStreamSynchronize(stream[0]);
  // check d_last_slots

  /////////// debug of step1
  if (1) {
    printf("\n step1 out \n");
    std::vector<int> h_indexs(threshold * num_cams);
    cudaMemcpy(h_indexs.data(), d_indexs, num_cams * threshold * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<uchar> h_count(num_query * (num_cams + pad));
    cudaMemcpy(h_count.data(), d_count, (num_cams + pad) * num_query * sizeof(uchar), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_cams * threshold; ++i) {
      if (i % threshold == 0)
        printf("\n");
      // if (h_indexs[i] >= 0)
      printf(">> %d\n", h_indexs[i]);
    }

    for (int i = 0; i < (num_cams + pad) * num_query; ++i) {
      if (i % (num_cams + pad) == 0)
        printf("\n");
      printf("%d ", int(h_count[i]));
    }
    printf("\n\n");

    for (int i = 0; i < num_cams; ++i) {
      printf("%d \n", h_indexs_len[i]);
    }
    printf("\n");
  }

  /////////// debug of step2
  if (1) {
    printf("\n step2 out \n");

    std::vector<T> h_out(bs * num_query);
    cudaMemcpy(h_out.data(), d_count_norm, h_out.size() * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_out.size(); ++i) {
      printf("\n%-6.1f", float(h_out[i]));
    }
    printf("\n");
  }

  /////////// debug of step3
  if (1) {
    printf("\n step3 out \n");

    std::vector<T> h_last_out_1(bs * num_cams * max_len * embed_dims);
    cudaMemcpy(h_last_out_1.data(), d_queries_rebatch, h_last_out_1.size() * sizeof(T), cudaMemcpyDeviceToHost);

    std::vector<T> h_last_out_2(bs * num_cams * max_len * D * 2);
    cudaMemcpy(
        h_last_out_2.data(), d_reference_points_rebatch, h_last_out_2.size() * sizeof(T), cudaMemcpyDeviceToHost);

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
  }

  /////////// debug of step4
  if (1) {
    printf("\n step4-5 out \n");

    std::vector<T> h_slots(bs * num_cams * num_query * embed_dims);
    cudaMemcpy(h_slots.data(), d_slots, h_slots.size() * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_queries.size(); ++i) {
      if (i % embed_dims == 0) {
        printf("\n");
      }
      if (i % (max_len * embed_dims) == 0) {
        printf("\n");
      }
      printf("%-6.1f", float(h_queries[i]));
    }
    printf("\n\n --- scatter --- ");

    for (int i = 0; i < h_slots.size(); ++i) {
      if (i % (num_query * embed_dims) == 0) {
        printf("\n");
      }
      if (i % embed_dims == 0) {
        printf("\n %2d| ", i / embed_dims % num_query);
      }
      printf("%-6.1f", float(h_slots[i]));
    }
    printf("\n\n --- sum(0) --- ");

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
  }
  cudaFree(d_queries_rebatch);
  cudaFree(d_reference_points_rebatch);
  cudaFree(d_query);
  cudaFree(d_reference_points_cam);

  cudaFree(d_bev_mask);
  cudaFree(d_indexs);
  cudaFree(d_count);
  cudaFree(d_count_norm);

  cudaFree(d_slots);
  cudaFree(d_queries);
  cudaFree(d_last_slots);

  for (int i = 0; i < num_cams; ++i) {
    cudaStreamDestroy(stream[i]);
  }
  return 0;
}

template int test<float>();
template int test<half>();

int main() {
  return test<half>();
}

// nvcc -std=c++14 -arch=sm_86 -O2 ./test_sca.cpp rebatch.cu queries_to_slots.cu  nonzero.cu

/*

tensor([[
[ 4.8000,  4.9000,  5.0000,  5.1000,  5.2000,  5.3000],
[ 0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000],
[ 3.0000,  3.1000,  3.2000,  3.3000,  3.4000,  3.5000],
[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
[ 6.0000,  6.1000,  6.2000,  6.3000,  6.4000,  6.5000],
[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
[ 6.6000,  6.7000,  6.8000,  6.9000,  7.0000,  7.1000],
[ 9.6000,  9.7000,  9.8000,  9.9000, 10.0000, 10.1000],
[ 7.8000,  7.9000,  8.0000,  8.1000,  8.2000,  8.3000],
[10.5000, 10.6000, 10.7000, 10.8000, 10.9000, 11.0000],
[24.0000, 24.1000, 24.2000, 24.3000, 24.4000, 24.5000],
[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]])

*/
