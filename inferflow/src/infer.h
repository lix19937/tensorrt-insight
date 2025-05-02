
/**************************************************************
 * @Author: ljw
 * @Date: 2024-04-12 16:57:59
 * @Last Modified by: ljw
 * @Last Modified time: 2024-04-13 10:14:45
 **************************************************************/

#pragma once

#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nni {

template <typename T>
struct TrtDeleter {
  void operator()(T* p) noexcept {
    if (p != nullptr)
      delete p;
  }
};

template <typename T, template <typename> typename DeleterType = TrtDeleter>
using TrtUniquePtr = std::unique_ptr<T, DeleterType<T>>;

template <typename T>
TrtUniquePtr<T> makeUnique(T* t) {
  return TrtUniquePtr<T>{t};
}

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    // suppress info-level messages
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
};

class Infer {
 public:
  Infer() {}
  virtual ~Infer() {}

  bool Init(const std::string& plan);

  bool Run(const int runtime_batch = 1);

  void SetBindings(void* in[], void* out[]);

 private:
  nvinfer1::ICudaEngine* LoadEngine(const std::string& engine);
  bool InitCudaGraph(const std::vector<std::vector<nvinfer1::Dims>>& bindings_dims);
  bool BuildEngine(const std::string& plan_file);

  TrtUniquePtr<nvinfer1::IRuntime> runtime_{};
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_{};

  std::vector<void*> iobindings_;
  std::vector<nvinfer1::Dims> iobinding_dims_;
  std::vector<nvinfer1::DataType> iobinding_types_;

  bool is_dynamic_{false};
  bool use_cudagraph_{false};
  int num_in_;
  int max_batch_{1};
  int min_batch_{1};
  int dla_core_{-1};
  cudaStream_t stream_{nullptr};

  std::unordered_map<int, cudaGraphExec_t> cudagraph_map_;
};
} // namespace nni