

#include <stdio.h>
#include <iostream>
#include <vector>

// queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
// reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])

//// bs = 1
// for j in range(bs):
//     for i, reference_points_per_img in enumerate(reference_points_cam):
//         index_query_per_img = indexes[i]
//         queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
//         reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

// index 1 2 5 7 11 89 623 0 0 ... 0
__global__ void nd_idx(
    float* queries_rebatch,
    float* reference_points_rebatch,
    const float* query,
    const float* reference_points_per_img,
    const int* index, // attach with each bs
    const int cpy_query_num,
    const int cpy_reference_points_num,
    const int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;
  int data_id = index[tid];

  if (data_id < 0)
    return;
  memcpy(queries_rebatch + tid * cpy_query_num, query + data_id * cpy_query_num, cpy_query_num * sizeof(float));
  memcpy(
      reference_points_rebatch + tid * cpy_reference_points_num,
      reference_points_per_img + data_id * cpy_reference_points_num,
      cpy_reference_points_num * sizeof(float));
}

void scatter(
    float* queries_rebatch,
    float* reference_points_rebatch,
    const float* query,
    const float* reference_points_per_img,
    const int* index,
    const int cpy_query_num,
    const int cpy_reference_points_num,
    const int query_num,
    cudaStream_t stream = 0) {
  const int thread_in_block = 512;
  nd_idx<<<(query_num + thread_in_block - 1) / thread_in_block, thread_in_block, 0, stream>>>(
      queries_rebatch,
      reference_points_rebatch,
      query,
      reference_points_per_img,
      index,
      cpy_query_num,
      cpy_reference_points_num,
      query_num);
  cudaDeviceSynchronize();
}

int test_b1() {
  int bs = 1, num_cams = 1, max_len = 8, embed_dims = 6, D = 4, query_num = 12;

  float *queries_rebatch, *reference_points_rebatch;
  cudaMalloc(&queries_rebatch, bs * num_cams * max_len * embed_dims * sizeof(float));
  cudaMalloc(&reference_points_rebatch, bs * num_cams * max_len * D * 2 * sizeof(float));
  cudaMemset(queries_rebatch, 0, bs * num_cams * max_len * embed_dims * sizeof(float));
  cudaMemset(reference_points_rebatch, 0, bs * num_cams * max_len * D * 2 * sizeof(float));

  float *query, *reference_points_cam;
  cudaMalloc(&query, bs * query_num * embed_dims * sizeof(float));
  cudaMalloc(&reference_points_cam, bs * num_cams * query_num * D * 2 * sizeof(float));
  std::vector<float> h_query(bs * query_num * embed_dims);
  for (int i = 0; i < h_query.size(); ++i)
    h_query[i] = i;
  cudaMemcpy(query, h_query.data(), h_query.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::vector<float> h_reference_points_cam(bs * num_cams * max_len * D * 2);
  for (int i = 0; i < h_reference_points_cam.size(); ++i)
    h_reference_points_cam[i] = i;
  cudaMemcpy(
      reference_points_cam,
      h_reference_points_cam.data(),
      h_reference_points_cam.size() * sizeof(float),
      cudaMemcpyHostToDevice);

  std::vector<int> h_index{0, 2, 3, 6, 11};
  int idx_len = 5;
  int* index;
  cudaMalloc(&index, idx_len * sizeof(int));
  cudaMemcpy(index, h_index.data(), h_index.size() * sizeof(int), cudaMemcpyHostToDevice);

  for (int i = 0; i < h_query.size(); ++i) {
    if (i % embed_dims == 0)
      printf("\n %2d| ", i / embed_dims);
    printf("%-3.0f", h_query[i]);
  }
  printf("\n\n");

  ///////
  scatter(queries_rebatch, reference_points_rebatch, query, reference_points_cam, index, embed_dims, D * 2, idx_len);

  ///////
  std::vector<float> h_last_out_1(bs * num_cams * max_len * embed_dims);
  cudaMemcpy(h_last_out_1.data(), queries_rebatch, h_last_out_1.size() * sizeof(float), cudaMemcpyDeviceToHost);

  std::vector<float> h_last_out_2(bs * num_cams * max_len * D * 2);
  cudaMemcpy(
      h_last_out_2.data(), reference_points_rebatch, h_last_out_2.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < h_last_out_1.size(); ++i) {
    if (i % embed_dims == 0)
      printf("\n");
    printf("%-3.0f ", h_last_out_1[i]);
  }
  printf("\n");

  cudaFree(queries_rebatch);
  cudaFree(reference_points_rebatch);
  cudaFree(query);
  cudaFree(reference_points_cam);
  cudaFree(index);
  return 0;
}

int main() {
  return test_b1();
}
// nvcc -std=c++14 -arch=sm_86 -O2 ./test_scatter.cu
