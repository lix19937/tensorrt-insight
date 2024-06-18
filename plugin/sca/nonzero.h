
#pragma once

#include <cuda_runtime_api.h>

// for i, mask_per_img in enumerate(bev_mask):
//     index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
//     indexes.append(index_query_per_img)
// max_len = max([len(each) for each in indexes])

// count = bev_mask.sum(-1) > 0
// logger.info(pt("count", count)) # [6, 1, 40000]  torch.bool
// count = count.permute(1, 2, 0)  # [6, 1, 40000] -> [1, 40000, 6]

typedef unsigned char uchar;

class NonZero {
 public:
  NonZero(const int N = 40000, const int batch = 6) : rt_batch_(batch) {
    for (int i = 0; i < rt_batch_; ++i) {
      cudaMalloc((void**)&d_num_nonzero_[i], sizeof(int));
      cudaMalloc((void**)&d_temp_storage_[i], 6000); // N
      cudaMalloc((void**)&d_out_temp_[i], N * sizeof(int));
    }
  }

  ~NonZero() {
    for (int i = 0; i < rt_batch_; ++i) {
      cudaFree(d_out_temp_[i]);
      cudaFree(d_num_nonzero_[i]);
      cudaFree(d_temp_storage_[i]);
    }
  }

  void nonzero(
      int* h_indexs_len, // [num_cams]
      int* d_indexs, // [num_cams, max_len]
      uchar* d_count, // [num_query, (num_cams + pad)]
      const char4* d_in, // [num_query, num_cams]
      const int N, //  num_query
      const int NC, // num_cams + pad
      const int threshold, // max_len
      const int num_cams,
      cudaStream_t stream[]);

 private:
  void __nonzero(
      int* h_indexs_len,
      int* d_indexs,
      uchar* d_count,
      const char4* d_in,
      const int N,
      const int NC,
      const int threshold,
      const int idx,
      cudaStream_t stream);

  static constexpr int max_batch_{8};
  int rt_batch_{1};
  size_t temp_storage_bytes_[max_batch_]{0};
  int* d_num_nonzero_[max_batch_];
  int* d_out_temp_[max_batch_];
  void** d_temp_storage_[max_batch_];
};
