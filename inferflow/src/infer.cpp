
/**************************************************************
 * @Author: ljw
 * @Date: 2024-04-12 16:57:59
 * @Last Modified by: ljw
 * @Last Modified time: 2024-04-13 10:14:45
 **************************************************************/

#include "infer.h"
#include <sys/stat.h>
#include <fstream>

namespace nni {

nvinfer1::ICudaEngine* Infer::LoadEngine(const std::string& engine) {
  printf("open engine ...\n");
  std::ifstream enginefs(engine, std::ios::binary);
  if (!enginefs) {
    printf("open engine file:%s failed\n", engine.c_str());
    return nullptr;
  }

  enginefs.seekg(0, std::ifstream::end);
  long int fsize = enginefs.tellg();
  enginefs.seekg(0, std::ifstream::beg);

  printf("read ... %ld Bytes\n", fsize);
  std::vector<char> engine_data(fsize);
  enginefs.read(engine_data.data(), fsize);
  if (!enginefs) {
    printf("read engine file:%s failed\n", engine.c_str());
    return nullptr;
  }

  printf("createInferRuntime ...\n");
  Logger logger;
  runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

  if (dla_core_ >= 0) {
    printf("setDLACore ...\n");
    runtime_->setDLACore(dla_core_);
  }

  printf("deserializeCudaEngine ...\n");
  return runtime_->deserializeCudaEngine(engine_data.data(), fsize, nullptr);
}

bool Infer::InitCudaGraph(const std::vector<std::vector<nvinfer1::Dims>>& bindings_dims) {
  auto total_kinds = max_batch_ - min_batch_ + 1;
  if (int(bindings_dims.size()) != total_kinds) {
    printf("bad para error %d %d %d\n", int(bindings_dims.size()), max_batch_, min_batch_);
    return false;
  }

  TrtUniquePtr<nvinfer1::IExecutionContext> ctx_ = makeUnique(engine_->createExecutionContext());
  if (!ctx_) {
    printf("createExecutionContext failed.\n");
    return false;
  }
  printf("%p\n", &ctx_);

  std::vector<cudaGraphExec_t> gexec(total_kinds);
  for (int i = min_batch_, k = 0; i <= max_batch_; ++i, ++k) {
    cudaGraph_t graph;

    auto bd = bindings_dims[k];
    if (!bd.empty()) {
      for (int j = 0; j < int(bd.size()); ++j) {
        if (!ctx_->setBindingDimensions(j, bd[j])) {
          printf("setBindingDimensions error\n");
          return false;
        }
      }
    }

    bool status = ctx_->enqueueV2(iobindings_.data(), stream_, nullptr);
    if (!status) {
      printf("enqueueV2 failed !\n");
      return false;
    }

    CheckCudaErrors(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    status = ctx_->enqueueV2(iobindings_.data(), stream_, nullptr);
    if (!status) {
      printf("in capture, enqueueV2 failed !\n");
      return false;
    }

    CheckCudaErrors(cudaStreamEndCapture(stream_, &graph));
    CheckCudaErrors(cudaGraphInstantiate(&gexec[k], graph, NULL, NULL, 0));
    CheckCudaErrors(cudaStreamSynchronize(stream_));
    CheckCudaErrors(cudaGraphDestroy(graph));
    cudagraph_map_.insert(std::make_pair(i, gexec[k]));
    printf("cudagraph_map_ (%d  %p)\n", i, gexec[k]);
  }

  printf("passed\n");
  return true;
}

bool Infer::Init(const std::string& plan) {
  printf("...\n");

  bool ret{false};
  if (!plan.empty()) {
    ret = BuildEngine(plan);
    if (!ret) {
      return false;
    }
  } else {
    printf("plan_file str is empty !\n");
    return false;
  }

  const auto nb = engine_->getNbBindings();
  for (int i = 0; i < nb; ++i) {
    if (engine_->bindingIsInput(i)) {
      ++num_in_;
    }
    iobinding_dims_.push_back(engine_->getBindingDimensions(i));
    iobinding_types_.push_back(engine_->getBindingDataType(i));
  }

  for (int i = 0; i < num_in_; ++i) {
    const auto& it = iobinding_dims_[i];
    if (it.d[0] < 0) {
      is_dynamic_ = true;
      break;
    }
  }

  printf("from plan, num_io:%d, num_in:%d, is_dynamic:%s\n", nb, num_in_, is_dynamic_ ? "TRUE" : "FALSE");

#if 0
  if (in == nullptr || out == nullptr) {
    printf("passed, scenario where the io address cannot be known in advance!\n");
    return true;
  }

  for (int i = 0; i < num_in_; ++i) {
    iobindings_.push_back((*in)[i].data);
  }

  for (int i = 0; i < nb - num_in_; ++i) {
    iobindings_.push_back((*out)[i].data);
  }

  // CUDA_GRAPH
  if (graph_conf_->use_cudagraph) {
    std::vector<std::vector<nvinfer1::Dims>> bindings_dims;
    for (int b = min_batch_; b <= max_batch_; ++b) {
      std::vector<nvinfer1::Dims> t;

      for (int i = 0; i < num_in_; ++i) {
        auto it = iobinding_dims_[i];
        for (int idx = 0; idx < it.nbDims; ++idx) {
          if (it.d[idx] < 0) {
            if (idx == 0) {
              it.d[idx] = b;
            } else {
              printf("current only support dynamic batch.\n");
              exit(-1);
            }
          }
        }
        t.push_back(it);
      }
      bindings_dims.push_back(t);
    }

    // --------------------------------------------------------
    for (auto it : bindings_dims) {
      for (auto jt : it) {
        PRINT_DIMS(jt);
      }
    }
    // --------------------------------------------------------

    if (!InitCudaGraph(bindings_dims)) {
      return false;
    }
  }
#endif

  printf("passed\n");
  return true;
}

bool Infer::BuildEngine(const std::string& plan_file) {
  printf("...\n");

  struct stat buffer;
  if (stat(plan_file.c_str(), &buffer) != 0) {
    printf("engine plan not foud.\n", plan_file.c_str());
    return false;
  } else {
    printf("LoadEngine ...\n");
    engine_ = makeUnique(LoadEngine(plan_file));
  }

  // printf("createExecutionContext ...");
  // ctx_ = makeUnique(engine_->createExecutionContext());
  // if (!ctx_) {
  //   printf("createExecutionContext failed.");
  //   return false;
  // }

  printf("passed\n");
  return true;
}

void Infer::SetBindings(void* in[], void* out[]) {
  iobindings_.clear();
  for (int i = 0; i < num_in_; ++i) {
    iobindings_.push_back(in[i]);
  }

  for (int i = 0; i < int(iobindings_.size()) - num_in_; ++i) {
    iobindings_.push_back(out[i]);
  }
}

bool Infer::Run(const int runtime_batch) {
  if (runtime_batch <= 0) {
    printf("expect runtime_batch > 0, but runtime_batch is %d\n", runtime_batch);
    return false;
  }

  /////////////////////////////////////////////
  TrtUniquePtr<nvinfer1::IExecutionContext> ctx_ = makeUnique(engine_->createExecutionContext());
  if (!ctx_) {
    printf("createExecutionContext failed.\n");
    return false;
  }
  ////////////////////////////////////////////////
  printf("%p\n", &ctx_);

  if (is_dynamic_) {
    for (int i = 0; i < num_in_; ++i) {
      auto dims = engine_->getBindingDimensions(i);
      if (dims.d[0] < 0) {
        dims.d[0] = runtime_batch;
        if (!ctx_->setBindingDimensions(i, dims)) {
          printf("setBindingDimensions error @idx %d\n", i);
          return false;
        }
      }
    }
  }

  if (use_cudagraph_) {
    if (cudagraph_map_.empty()) {
      printf("cudagraph_map_ is empty\n");
      return false;
    }
    auto it = cudagraph_map_.find(runtime_batch);
    if (it != cudagraph_map_.end()) {
      // printf("runtime_batch %d, gexec %p", runtime_batch, it->second);
      CheckCudaErrors(cudaGraphLaunch(it->second, stream_));
      CheckCudaErrors(cudaStreamSynchronize(stream_));
    } else {
      printf("runtime_batch %d not in cudagraph_map_, error.\n", runtime_batch);
      return false;
    }
  } else {
    auto status = ctx_->enqueueV2(iobindings_.data(), stream_, nullptr);
    if (!status) {
      printf("enqueueV2 failed.\n");
      return false;
    }
    CheckCudaErrors(cudaStreamSynchronize(stream_));
  }

  return true;
}

} // namespace nni
