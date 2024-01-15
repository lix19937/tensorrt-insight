
/**************************************************************
 * @Author: ljw
 * @Date: 2024-01-12 16:07:51
 * @Last Modified by: ljw
 * @Last Modified time: 2024-01-13 10:14:41
 **************************************************************/

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <sys/stat.h>

#include <fstream>
#include <unordered_map>
#include <vector>
#include <memory>

#include <NvInfer.h>
#include <NvOnnxParser.h>

// from /usr/src/tensorrt/samples/common/logging.h    
#include "logging.h"

// parse command line from https://github.com/tanakh/cmdline
#include "cmdline.h"

static const std::string red{"\033[31m"};
static const std::string yellow{"\033[33m"};
static const std::string white{"\033[37m"};

namespace sample{
  Logger gLogger{Logger::Severity::kVERBOSE};
  LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
  LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
  LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
  LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
  LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

  void setReportableSeverity(Logger::Severity severity){
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
  }
}  

#define LOG_INFO(format, ...)                                                     \
  do {                                                                            \
    fprintf(stderr, "[I %s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
  } while (0)
#define LOG_DEBUG(format, ...)                                                                                       \
  do {                                                                                                               \
    fprintf(stderr, "%s[D %s:%d] " format "%s\n", yellow.c_str(), __FILE__, __LINE__, ##__VA_ARGS__, white.c_str()); \
  } while (0)
#define LOG_ERROR(format, ...)                                                                                    \
  do {                                                                                                            \
    fprintf(stderr, "%s[E %s:%d] " format "%s\n", red.c_str(), __FILE__, __LINE__, ##__VA_ARGS__, white.c_str()); \
  } while (0)

#define print_dims(dim)                          \
  do {                                           \
    for (int idx = 0; idx < dim.nbDims; ++idx) { \
      LOG_INFO("dim %d=%d", idx, dim.d[idx]);    \
    }                                            \
  } while (0)

inline void cudaCheck(cudaError_t ret, std::ostream& err = std::cerr) {
  if (ret != cudaSuccess) {
    err << "Cuda failure: " << cudaGetErrorString(ret) << std::endl;
    abort();
  }
}

template <typename T>
struct TrtDestroyer {
  void operator()(T* t) { t->destroy(); }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

static auto StreamDeleter = [](cudaStream_t* pStream){
  if (pStream){
    cudaStreamDestroy(*pStream);
    delete pStream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream(){
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess){
    pStream.reset(nullptr);
  }

  return pStream;
}

template <typename T>
inline std::shared_ptr<T> infer_object(T* obj){
  if (!obj){
    throw std::runtime_error("Failed to create object");
  }
  return std::shared_ptr<T>(obj);
}

class SampleDynamicReshape {
 public:
  SampleDynamicReshape() {}
  ~SampleDynamicReshape() {
    if (mstream != nullptr)
      cudaStreamDestroy(mstream);
  }

  bool init(const std::string& plan_file, bool use_graph = false);
  bool infer(int batch);

  struct OnnxParams{
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string onnxFileName; //!< Filename of ONNX file of a network
    int max_batch{1};
    int min_batch{1};
  };

  OnnxParams mParams;

 private:
  bool buildPredictionEngine(
      const TrtUniquePtr<nvinfer1::IBuilder>& builder,
      const TrtUniquePtr<nvinfer1::IRuntime>& runtime,
      cudaStream_t profileStream);
  bool buildPredictionEngine(const std::string& plan_file);
  bool preprocess();
  bool init_graph(
      const std::vector<void*>& bindings,
      const std::vector<std::vector<nvinfer1::Dims>>& bindings_dims = std::vector<std::vector<nvinfer1::Dims>>());

  bool saveEngine(const nvinfer1::ICudaEngine& engine, const std::string& fileName);
  nvinfer1::ICudaEngine* loadEngine(const std::string& engine, int dla_core);

  std::vector<nvinfer1::Dims> mInputDims;
  std::vector<nvinfer1::Dims> mOutputDims;
  std::vector<nvinfer1::DataType> mInputDT;
  std::vector<nvinfer1::DataType> mOutputDT;

  TrtUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
  TrtUniquePtr<nvinfer1::IExecutionContext> ctx{nullptr};
  struct Tensor {
    void *data;
    int nbytes;
  };

  std::map<nvinfer1::DataType, int > dt_map{
    {nvinfer1::DataType::kFLOAT, sizeof(float)}, 
    {nvinfer1::DataType::kHALF, sizeof(float)/2}, 
    {nvinfer1::DataType::kINT8, sizeof(char)},
    {nvinfer1::DataType::kINT32, sizeof(int)},
    {nvinfer1::DataType::kBOOL, sizeof(bool)} };

  inline void cumalloc(Tensor &t, nvinfer1::Dims dims, nvinfer1::DataType dt){
    int n_ele = 1;
    for (int i=0; i<dims.nbDims; ++i){
      n_ele *= dims.d[i];
    }
    t.nbytes = n_ele*dt_map[dt];
    cudaMalloc(&t.data, t.nbytes);
  }

  std::vector<Tensor> mInput;
  std::vector<Tensor> mOutput;

  template <typename T>
  TrtUniquePtr<T> makeUnique(T* t) {
    return TrtUniquePtr<T>{t};
  }

  const float pixel_mean_[3]{0.485f, 0.456f, 0.406f}; // {123.675, 116.28, 103.53}; // in RGB order
  const float pixel_std_[3]{0.229f, 0.224f, 0.225f}; // {58.395, 57.12, 57.375};
  const float pixel_scale_[3]{1 / 255.f, 1 / 255.f, 1 / 255.f}; // 0.003921568

  std::vector<void*> predicitonBindings;
  cudaStream_t mstream{nullptr};
  std::unordered_map<int, cudaGraphExec_t> mgraph_map; // batch, graph
  bool is_dynamic{false};
  bool is_use_graph{false};
  std::map<int, int> dynamic_io_tensor_dimidx_table; // the idx of io_tensors; the idx with dynamic dim of the io_tensor, here default is 0

};

bool SampleDynamicReshape::saveEngine(const nvinfer1::ICudaEngine& engine, const std::string& fileName) {
  std::ofstream engineFile(fileName, std::ios::binary);
  if (!engineFile) {
    LOG_ERROR("Cannot open engine file:%s", fileName.c_str());
    return false;
  }

  TrtUniquePtr<nvinfer1::IHostMemory> serializedEngine{engine.serialize()};
  if (serializedEngine == nullptr) {
    LOG_ERROR("Engine serialization failed");
    return false;
  }

  engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
  return !engineFile.fail();
}

nvinfer1::ICudaEngine* SampleDynamicReshape::loadEngine(const std::string& engine, int dla_core) {
  std::ifstream engineFile(engine, std::ios::binary);
  if (!engineFile) {
    LOG_ERROR("Error opening engine file:%s", engine.c_str());
    return nullptr;
  }

  engineFile.seekg(0, std::ifstream::end);
  long int fsize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);

  std::vector<char> engineData(fsize);
  engineFile.read(engineData.data(), fsize);
  if (!engineFile) {
    LOG_ERROR("Error loading engine file:%s", engine.c_str());
    return nullptr;
  }

  TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
  if (dla_core != -1) {
    runtime->setDLACore(dla_core);
  }

  return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
}

bool SampleDynamicReshape::init_graph(
    const std::vector<void*>& bindings, const std::vector<std::vector<nvinfer1::Dims>>& bindings_dims) {
  if (int(bindings_dims.size()) != mParams.max_batch - mParams.min_batch +1) {
    LOG_ERROR("bad para error %d %d %d", int(bindings_dims.size()), mParams.max_batch, mParams.min_batch);
    return false;
  }
  std::vector<cudaGraphExec_t> gexec(mParams.max_batch - mParams.min_batch +1);
  for (int i = mParams.min_batch, k=0; i <= mParams.max_batch; ++i, ++k) {
    cudaGraph_t graph;

    // if model is dynamic, the binds need set
    auto bd = bindings_dims[k];
    if (!bd.empty()) {
      for (int j = 0; j < int(bd.size()); ++j) {
        if (!ctx->setBindingDimensions(j, bd[j])) {
          LOG_ERROR("setBindingDimensions error");
          return false;
        }
      }
    }

    bool status = ctx->enqueueV2(bindings.data(), mstream, nullptr);
    if (!status) {
      LOG_ERROR("enqueueV2 failed !");
      return false;
    }

    cudaCheck(cudaStreamBeginCapture(mstream, cudaStreamCaptureModeGlobal));
    status = ctx->enqueueV2(bindings.data(), mstream, nullptr);
    if (!status) {
      LOG_ERROR("enqueueV2 failed !");
      return false;
    }

    cudaCheck(cudaStreamEndCapture(mstream, &graph));
    cudaCheck(cudaGraphInstantiate(&gexec[k], graph, NULL, NULL, 0));
    cudaCheck(cudaStreamSynchronize(mstream));
    cudaCheck(cudaGraphDestroy(graph));
    mgraph_map.insert(std::make_pair(i, gexec[k]));
    LOG_DEBUG("mgraph_map (%d  %p)", i, gexec[k]);
  }

  cudaMemset(*bindings.rbegin(), 0, 4*3*4);

  return true;
}

bool SampleDynamicReshape::init(const std::string& plan, bool use_graph) {
  bool result{false};

  auto csc = cudaStreamCreate(&mstream);
  if (int(csc) != 0) {
    LOG_ERROR("Error cudaStreamCreate:%d", csc);
    return false;
  }

  if (!plan.empty()) {
    result = buildPredictionEngine(plan);
  } else {
    auto builder = makeUnique(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
      LOG_ERROR("Create inference builder failed.");
      return false;
    }

    auto runtime = makeUnique(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime) {
      LOG_ERROR("Runtime object creation failed.");
      return false;
    }

    try {
      auto profileStream = makeCudaStream();
      if (!profileStream) {
        LOG_ERROR("makeCudaStream failed.");
        return false;
      }

      result = buildPredictionEngine(builder, runtime, *profileStream);
    } catch (std::runtime_error& e) {
      LOG_ERROR("%s", e.what());
      return false;
    }
  }

  if (!result) {
    return false;
  }

  const auto nb = engine->getNbBindings();
  LOG_INFO("NbBinding:%d", nb);
  for (int i = 0; i < nb; ++i) {
    if (engine->bindingIsInput(i)) {
      mInputDims.push_back(engine->getBindingDimensions(i));
      mInputDT.push_back(engine->getBindingDataType(i));
    } else {
      mOutputDims.push_back(engine->getBindingDimensions(i));
      mOutputDT.push_back(engine->getBindingDataType(i));
    }
  }

  // --------------------------------------------------------
  for (auto it : mInputDims)
    print_dims(it);
  for (auto it : mOutputDims)
    print_dims(it);

  // if dynamic batch, here i only consider all dynamic-inputs and outputs use the same batch  
  std::vector<nvinfer1::Dims> mInputDimsMax, mOutputDimsMax;

  int idx_of_iotensor = 0;
  for (auto it : mInputDims) {
    for (int idx = 0; idx < it.nbDims; ++idx) {
      if (it.d[idx] <= 0) {
        dynamic_io_tensor_dimidx_table.insert(std::make_pair(idx_of_iotensor, 0));
        it.d[idx] = mParams.max_batch; /// !!!
        if (!is_dynamic) {
          is_dynamic = true;
        }
      }
    }
    mInputDimsMax.push_back(it);
  }

  for (auto it : mOutputDims) {
    for (int idx = 0; idx < it.nbDims; ++idx) {
      if (it.d[idx] <= 0) {
        dynamic_io_tensor_dimidx_table.insert(std::make_pair(idx_of_iotensor, 0));
        it.d[idx] = mParams.max_batch; /// !!!
      }
    }
    mOutputDimsMax.push_back(it);
  }

  for (auto it: dynamic_io_tensor_dimidx_table){
    mParams.min_batch = engine->getProfileDimensions(it.first, 0, nvinfer1::OptProfileSelector::kMIN).d[0];
    mParams.max_batch = engine->getProfileDimensions(it.first, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
    break;
  }

  LOG_DEBUG("is_dynamic:%s, min_batch:%d, max_batch:%d", is_dynamic ? "true" : "false", mParams.min_batch, mParams.max_batch);

  for (auto it : mInputDimsMax)
    print_dims(it);
  for (auto it : mOutputDimsMax)
    print_dims(it);

  mInput.resize(mInputDims.size());
  mOutput.resize(mOutputDims.size());

  for (size_t i = 0; i < mInputDimsMax.size(); ++i) {
    cumalloc(mInput[i], mInputDimsMax[i], mInputDT[i]);
    predicitonBindings.push_back(mInput[i].data);
  }

  for (size_t i = 0; i < mOutputDimsMax.size(); ++i) {
    cumalloc(mOutput[i], mOutputDimsMax[i], mOutputDT[i]);
    predicitonBindings.push_back(mOutput[i].data);
  }

  is_use_graph = use_graph;
  if (is_use_graph) {
    std::vector<std::vector<nvinfer1::Dims>> bindings_dims;
    for (int b = mParams.min_batch; b <= mParams.max_batch; ++b) {
      std::vector<nvinfer1::Dims> tt;
      for (auto it : mInputDims) {
        for (int idx = 0; idx < it.nbDims; ++idx) {
          if (it.d[idx] <= 0) {
            if (idx == 0) {
              it.d[idx] = b;  
            } else {
              LOG_ERROR("Error dynamic, current only support dynamic batch, %d", idx);
              exit(-1);
            }            
          }else{
            // donothing
          }
        }
        tt.push_back(it);
      }
      bindings_dims.push_back(tt);
    }

    for (auto it : bindings_dims) {
      for (auto jt :it){
        print_dims(jt);
      }
      printf("\n");  
    }
    init_graph(predicitonBindings, bindings_dims);
  }
  return result;
}

bool SampleDynamicReshape::buildPredictionEngine(
    const TrtUniquePtr<nvinfer1::IBuilder>& builder,
    const TrtUniquePtr<nvinfer1::IRuntime>& runtime,
    cudaStream_t profileStream) {
  LOG_INFO("createNetworkV2 ...");
  auto network = makeUnique(
      builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!network) {
    LOG_ERROR("Create network failed.");
    return false;
  }

  LOG_INFO("parseFromFile ...");
  auto parser = infer_object(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
  auto parsingSuccess =
      parser->parseFromFile(mParams.onnxFileName.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
  if (!parsingSuccess) {
    LOG_ERROR("Failed to parse model.");
    return false;
  }

  LOG_INFO("createBuilderConfig ...");
  auto config = makeUnique(builder->createBuilderConfig());
  if (!config) {
    LOG_ERROR("Create builder config failed.");
    return false;
  }

  // --------------------------------------------------------
  if (builder->platformHasFastFp16()) {
    LOG_INFO("device support FP16.");
  }

  if (builder->platformHasFastInt8()) {
    LOG_INFO("device support Int8.");
  }

  config->setFlag(nvinfer1::BuilderFlag::kTF32);

  builder->setMaxBatchSize(1);
  config->setMaxWorkspaceSize(4096U * (1U << 20));
  if (mParams.fp16) {
    LOG_INFO("use fp16");
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  // --------------------------------------------------------
  const auto plan_file = mParams.onnxFileName + ".plan";
  struct stat buffer;
  if (stat(plan_file.c_str(), &buffer) != 0) {
    LOG_INFO("engine plan not foud.");
    TrtUniquePtr<nvinfer1::IHostMemory> plan = makeUnique(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
      LOG_ERROR("serialized engine init failed.");
      return false;
    }

    engine = makeUnique(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine) {
      LOG_ERROR("engine deserialization failed.");
      return false;
    }
    saveEngine(*engine, plan_file);
  }

  // --------------------------------------------------------
  ctx = makeUnique(engine->createExecutionContext());
  if (!ctx) {
    LOG_ERROR("Prediction context init failed.");
    return false;
  }
  LOG_INFO("done ...");

  return true;
}

bool SampleDynamicReshape::buildPredictionEngine(const std::string& plan_file) {
  struct stat buffer;
  if (stat(plan_file.c_str(), &buffer) != 0) {
    LOG_ERROR("engine plan not foud.");
    exit(0);
  } else {
    LOG_INFO("loadEngine ...");
    engine = makeUnique(loadEngine(plan_file, mParams.dlaCore));
  }

  // --------------------------------------------------------
  ctx = makeUnique(engine->createExecutionContext());
  if (!ctx) {
    LOG_ERROR("Prediction context init failed.");
    return false;
  }

  LOG_INFO("createExecutionContext done");
  return true;
}

bool SampleDynamicReshape::infer(int batch) {
  LOG_INFO("preprocess...");

  preprocess();

  if (is_dynamic) {
    nvinfer1::Dims dims = engine->getBindingDimensions(0);
    dims.d[0] = batch;
    if (!ctx->setBindingDimensions(0, dims)) {
      LOG_ERROR("setBindingDimensions error.");
      return false;
    }
  }

  //////////  executeV2
  // auto status = ctx->executeV2(predicitonBindings.data()); //  executeV2
  // if (!status) {
  //   LOG_ERROR("executeV2 failed.");
  //   return false;
  // }

  if (is_use_graph) {
    LOG_INFO("cudaGraphLaunch...");
    auto it = mgraph_map.find(batch);
    if (it != mgraph_map.end()) {
      LOG_DEBUG("batch %d, gexec %p", batch, it->second);

      cudaCheck(cudaGraphLaunch(it->second, mstream));
      cudaStreamSynchronize(mstream);
    } else {
      LOG_ERROR("batch para error, %d", batch);
      return false;
    }
  } else {
    auto status = ctx->enqueueV2(predicitonBindings.data(), mstream, nullptr); //  enqueueV2
    if (!status) {
      LOG_ERROR("enqueueV2 failed.");
      return false;
    }
    cudaStreamSynchronize(mstream);
  }

  LOG_INFO("cudaMemcpy...");

  // D2H
  std::vector<std::vector<float>> h_out(mOutput.size());

  int i =0;
  for (auto& it : mOutput) {
    h_out[i].resize(it.nbytes/sizeof(float));
    cudaCheck(cudaMemcpy(h_out[i].data(), it.data, it.nbytes, cudaMemcpyDeviceToHost));
    i++;
  }
  i=0;
  LOG_INFO("decode...");
  auto h_plugin_out1 = static_cast<float*>(h_out[i].data());
  for (size_t n = 0; n < h_out[i].size(); ++n) {
    printf(" -->> %f\n", float(h_plugin_out1[n]));
  }

  return true;
}

bool SampleDynamicReshape::preprocess() {
  srand(1); /// time(0)

  auto num_input = mInputDims.size();
  for (size_t i = 0; i < num_input; ++i) {
    auto d_in = static_cast<void *>(mInput[i].data);
    std::vector<float> h_in(mInput[i].nbytes/sizeof(float));
    int l = 0, r = 255;
    for (size_t n = 0; n < h_in.size(); ++n) {
      h_in[n] = ((rand() % (r - l + 1) + l) * pixel_scale_[0] - pixel_mean_[0]) / pixel_std_[0];
    }
    cudaMemcpy(d_in, (void *)h_in.data(), mInput[i].nbytes, cudaMemcpyHostToDevice);
  }

  return true;
}

void MyExit(int status, void* arg) {
  LOG_INFO("before exit() status:%d !", status);
  dlclose(arg);
}

// ./mytrtexec --plan=resnet50.plan -i 1 -b 1
// ./build/mytrtexec --plan=fvModule_vlr_int8_GPU_20230103.plan -i 1 -b 1 -g 1
int main_loop(int argc, char** argv) {
  LOG_INFO("main ...");

  cmdline::parser a;
  a.add<std::string>("onnx", '\0', "onnx file", false, "");
  a.add<std::string>("plan", '\0', "plan file", false, "");
  a.add<std::string>("plugin", 'p', "plugin so", false, "");
  a.add<int>("iterator", 'i', "iterator number", false, 1);
  a.add<int>("batch", 'b', "batch number", false, 1);
  a.add<int>("use_graph", 'g', "use_graph", false, 0);

  a.parse_check(argc, argv);

  const bool use_graph = a.get<int>("use_graph") != 0;

  const int batch = a.get<int>("batch");
  const int iterator = a.get<int>("iterator");
  std::string onnx = a.get<std::string>("onnx");
  std::string plan = a.get<std::string>("plan");
  std::string plugin = a.get<std::string>("plugin");

  LOG_INFO("onnx:%s", onnx.c_str());
  LOG_INFO("plan:%s", plan.c_str());
  LOG_INFO("plugin:%s", plugin.c_str());
  LOG_INFO("batch:%d", batch);
  LOG_INFO("iterator:%d", iterator);

  void* handle = nullptr;
  if (!plugin.empty()) {
    handle = dlopen(plugin.c_str(), RTLD_LAZY);
    LOG_INFO("dlerror ... %s", dlerror());
    if (handle == nullptr){
      exit(1);
    }
  }

  sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);

  SampleDynamicReshape trt;
  trt.mParams.onnxFileName = onnx;
  trt.mParams.max_batch = 4;
  ///===============================================================
  LOG_INFO("init ...");
  if (!trt.init(plan, use_graph)) {
    LOG_ERROR("init failed. plan:%s", plan.c_str());
    return 1;
  }
  LOG_INFO("init done");

  ///===============================================================
  LOG_INFO("loop infer ...");
  for (int i = 0; i < iterator; ++i) {
    if (!trt.infer(batch)) {
      LOG_ERROR("infer failed. iter:%d|%d", i, iterator);
      return 1;
    }
  }

  ///===============================================================
  LOG_INFO("Done, finished %d iterators", iterator);

  if (handle != nullptr) {
    ::on_exit(MyExit, handle);
  }
  return 0;
}

int main(int argc, char** argv) {
  return main_loop(argc, argv);
}

/// ./build/mytrtexec --plan=fvModule_vlr_int8_GPU_20230103.plan -i 1 -b 4 -g 1
//  -->> 0.001941
//  -->> 0.012328
//  -->> 0.010756
//  -->> 0.001925
//  -->> 0.010110
//  -->> 0.011364
//  -->> 0.002940
//  -->> 0.007267
//  -->> 0.007026
//  -->> 0.002988
//  -->> 0.009281
//  -->> 0.007050
