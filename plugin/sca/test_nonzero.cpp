
#include <stdio.h>
#include <vector>
#include "nonzero.h"

int main() {
  const int num_cams = 6, num_query = 12, threshold = 8;

  // logger.info(pt("query", query)) # [1, 40000, 256]
  // logger.info(pt("reference_points_cam", reference_points_cam)) # [6, 1, 40000, 4, 2]
  // logger.info(pt("bev_mask", bev_mask)) # [6, 1, 40000, 4]

  std::vector<char4> h_bev_mask(num_cams * num_query);
  h_bev_mask[1] = {1, 0, 0, 0};
  h_bev_mask[2] = {0, 0, 1, 0};
  h_bev_mask[8] = {1, 1, 0, 0};
  h_bev_mask[9] = {1, 1, 0, 0};
  h_bev_mask[num_query + 0] = {0, 0, 0, 1};
  h_bev_mask[num_query + 2] = {1, 0, 0, 0};
  h_bev_mask[num_query + 4] = {1, 0, 0, 0};
  h_bev_mask[num_query + 6] = {1, 0, 0, 0};
  h_bev_mask[num_query * 2 + 7] = {0, 0, 0, 1};
  h_bev_mask[num_query * 3 + 8] = {0, 0, 0, 1};
  h_bev_mask[num_query * 4 + 9] = {0, 0, 0, 1};
  h_bev_mask[num_query * 5 + 10] = {0, 0, 0, 1};

  /// in of nonzero
  char4* d_bev_mask;
  cudaMalloc((void**)&d_bev_mask, num_cams * num_query * sizeof(char4));
  cudaMemcpy(d_bev_mask, h_bev_mask.data(), num_cams * num_query * sizeof(char4), cudaMemcpyHostToDevice);

  //// out of nonzero
  int* d_indexs;
  cudaMalloc((void**)&d_indexs, threshold * num_cams * sizeof(int));
  cudaMemset(d_indexs, -1, threshold * num_cams * sizeof(int));

  //// python
  /// count = bev_mask.sum(-1) > 0
  /// logger.info(pt("count", count)) # [6, 1, 40000]  torch.bool
  /// count = count.permute(1, 2, 0) # [6, 1, 40000] -> [1, 40000, 6]

  //// out of count
  const int pad = 2;
  uchar* d_count;
  cudaMalloc((void**)&d_count, num_query * (num_cams + pad) * sizeof(uchar));
  cudaMemset(d_count, 0, num_query * (num_cams + pad) * sizeof(uchar));

  cudaStream_t stream[num_cams];

  for (int i = 0; i < num_cams; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  NonZero nz(num_query, num_cams);
  int h_indexs_len[num_cams];

  nz.nonzero(h_indexs_len, d_indexs, d_count, d_bev_mask, num_query, num_cams + pad, threshold, num_cams, stream);

  std::vector<int> h_indexs(threshold * num_cams);
  cudaMemcpy(h_indexs.data(), d_indexs, num_cams * threshold * sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<uchar> h_count(num_query * (num_cams + pad));
  cudaMemcpy(h_count.data(), d_count, (num_cams + pad) * num_query * sizeof(uchar), cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_cams * threshold; ++i) {
    if (i % threshold == 0)
      printf("\n");
    if (h_indexs[i] >= 0)
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

  cudaFree(d_bev_mask);
  cudaFree(d_indexs);
  cudaFree(d_count);

  for (int i = 0; i < num_cams; ++i) {
    cudaStreamDestroy(stream[i]);
  }

  return 0;
}

// nvcc -std=c++14 -arch=sm_86 -O2 ./test_nonzero.cpp  ./nonzero.cu
