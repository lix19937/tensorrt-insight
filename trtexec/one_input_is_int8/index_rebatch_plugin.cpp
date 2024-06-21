/**************************************************************
 * @Author: ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#include "index_rebatch_plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin_custom;

// # --IN---
// # query                     [bs, num_query, embed_dims]        float32
// # reference_points_cam      [num_cams, bs, num_query, D, 2]    float32
// # bev_mask                  [num_cams, bs, num_query, 4]       int8-p4
// #
// #  --OUT---
// # indexes                   [num_cams, max_len]                int32
// # indexes_len               [num_cams]                         int32
// # queries_rebatch           [bs*num_cams, max_len, embed_dims] float32
// # reference_points_rebatch  [bs*num_cams, max_len, D, 2]       float32
// # count_norm                [bs, num_query, 1]                 float32

// ## include follow steps
// # nonzero
// # count_norm
// # rebatch

namespace {
constexpr int num_in = 3;
constexpr int num_out = 5;
const char* PLUGIN_VERSION{"1"};
const char* PLUGIN_NAME{"SCA_IndexRebatch_TRT"};
} // namespace

// Static class fields initialization
PluginFieldCollection IndexRebatchPluginDynamicCreator::mFC{};
std::vector<PluginField> IndexRebatchPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(IndexRebatchPluginDynamicCreator);

IndexRebatchPluginDynamic::IndexRebatchPluginDynamic(
    const int num_query, const int max_len, const int num_cams, const int embed_dims)
    : num_query_(num_query), max_len_(max_len), num_cams_(num_cams), embed_dims_(embed_dims) {}

IndexRebatchPluginDynamic::IndexRebatchPluginDynamic(const void* data, size_t length) {
  __LOG_PLACEHOLD__;
  // Deserialize in the same order as serialization
  const char *d = reinterpret_cast<const char*>(data), *a = d;

  num_query_ = read<int32_t>(d);
  max_len_ = read<int32_t>(d);
  num_cams_ = read<int32_t>(d);
  embed_dims_ = read<int32_t>(d);

  ASSERT(d == a + length);
}

IndexRebatchPluginDynamic::~IndexRebatchPluginDynamic() {
  terminate();
}

IPluginV2DynamicExt* IndexRebatchPluginDynamic::clone() const noexcept {
  try {
    __LOG_PLACEHOLD__;
    auto* p = new IndexRebatchPluginDynamic(num_query_, max_len_, num_cams_, embed_dims_);
    p->setPluginNamespace(mNamespace.c_str());
    if (0 != p->initialize()) {
      __LOG_ERROR("IndexRebatchPluginDynamic init failed\n");
      return nullptr;
    }
    return p;
  } catch (const std::exception& e) {
    caughtError(e);
    __LOG_ERROR("%s\n", e.what());
  }
  return nullptr;
}

void IndexRebatchPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept {
  return;
}

void IndexRebatchPluginDynamic::detachFromContext() noexcept {}

DimsExprs IndexRebatchPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
  try {
    __LOG_PLACEHOLD__;
    ASSERT(nbInputs == num_in);
    ASSERT(outputIndex < num_out);

    // clang-format off
    int bs         = inputs[0].d[0]->getConstantValue();
    int num_query  = inputs[0].d[1]->getConstantValue();
    int embed_dims = inputs[0].d[2]->getConstantValue();
    int num_cams   = inputs[1].d[0]->getConstantValue();

    ASSERT(num_query == num_query_);
    ASSERT(num_cams == num_cams_);
    ASSERT(embed_dims == embed_dims_);
    ASSERT(bs == 1);

// #  --OUT---
// # indexes                   [num_cams, max_len]                int32
// # indexes_len               [num_cams]                         int32
// # queries_rebatch           [bs*num_cams, max_len, embed_dims] float32
// # reference_points_rebatch  [bs*num_cams, max_len, D, 2]       float32
// # count_norm                [bs, num_query, 1]                 float32

    nvinfer1::DimsExprs ret[num_out];
    ret[0].nbDims = 2;
    ret[0].d[0] = exprBuilder.constant(num_cams_);
    ret[0].d[1] = exprBuilder.constant(max_len_);

    ret[1].nbDims = 1;
    ret[1].d[0] = exprBuilder.constant(num_cams_);

    ret[2].nbDims = 3;
    ret[2].d[0] = exprBuilder.constant(num_cams_ * bs);
    ret[2].d[1] = exprBuilder.constant(max_len_);
    ret[2].d[2] = exprBuilder.constant(embed_dims_);

    ret[3].nbDims = 4;
    ret[3].d[0] = exprBuilder.constant(num_cams_ * bs);
    ret[3].d[1] = exprBuilder.constant(max_len_);
    ret[3].d[2] = exprBuilder.constant(4);
    ret[3].d[3] = exprBuilder.constant(2);

    ret[4].nbDims = 3;
    ret[4].d[0] = exprBuilder.constant(bs);
    ret[4].d[1] = exprBuilder.constant(num_query_);
    ret[4].d[2] = exprBuilder.constant(1);
    // clang-format on
    __LOG_INFO("outputIndex:%d  num_out:%d", outputIndex, num_out);
    __LOG_INFO("bs:%d ", bs);
    __LOG_INFO("num_query:%d ", num_query);
    __LOG_INFO("embed_dims:%d ", embed_dims);
    __LOG_INFO("num_cams:%d ", num_cams);
    __LOG_INFO("max_len_:%d ", max_len_);

    return ret[outputIndex];
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return DimsExprs{};
}

bool IndexRebatchPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
  __LOG_PLACEHOLD__;
  // clang-format off
  ASSERT(nbInputs == num_in);
  ASSERT(nbOutputs == num_out);
  ASSERT(0 <= pos && pos < nbInputs + nbOutputs);
  bool ret1 = false;

  const PluginTensorDesc& desc = inOut[pos];
  switch (pos) {
    case 0: // query
      ret1 = ((desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) && desc.format == TensorFormat::kLINEAR);
      break;
    case 1: // reference_points_cam
      ret1 = (desc.type == inOut[0].type && desc.format == inOut[0].format);
      break;
    case 2: // bev_mask
        __LOG_INFO(" +++ desc.type :%d ", int(desc.type ));

      ret1 = (desc.type == DataType::kINT8 && desc.format == TensorFormat::kLINEAR); //  kINT8
      break;

    case 3: // indexes
    case 4: // indexes_len
      ret1 = (desc.type == DataType::kINT32 && desc.format == TensorFormat::kLINEAR);
      break;

    case 5: // queries_rebatch
      ret1 = (desc.type == inOut[0].type && desc.format == inOut[0].format);
      break;
    case 6: // reference_points_rebatch
      ret1 = (desc.type == inOut[0].type && desc.format == inOut[0].format);
      break;
    case 7: // count_norm
      ret1 = (desc.type == inOut[0].type && desc.format == inOut[0].format);
      break;

    default:
      break;
  }
  // clang-format on
  return ret1;
}

void IndexRebatchPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const DynamicPluginTensorDesc* outputs,
    int nbOutputs) noexcept {
  try {
    ASSERT(nbInputs == num_in);
    ASSERT(nbOutputs == num_out);
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

size_t IndexRebatchPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  return 0;
}

int IndexRebatchPluginDynamic::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  try {
    __LOG_PLACEHOLD__;
    return 0;
    auto& data_type = inputDesc[0].type;
    ASSERT(data_type == DataType::kFLOAT || data_type == DataType::kHALF)

    auto& query = inputs[0];
    auto& reference_points_cam = inputs[1];
    auto& bev_mask = inputs[2];
    auto& indexes = outputs[0];
    auto& indexes_len = outputs[1];
    auto& queries_rebatch = outputs[2];
    auto& reference_points_rebatch = outputs[3];
    auto& count_norm = outputs[4];

    switch (data_type) {
      case DataType::kFLOAT:
        pl_->Forward(
            (int*)indexes,
            (int*)indexes_len,
            (float*)queries_rebatch,
            (float*)reference_points_rebatch,
            (float*)count_norm,

            (const float*)query,
            (const float*)reference_points_cam,
            (const char4*)bev_mask,
            stream);
        break;

      case DataType::kHALF:
        pl_->Forward(
            (int*)indexes,
            (int*)indexes_len,
            (half*)queries_rebatch,
            (half*)reference_points_rebatch,
            (half*)count_norm,

            (const half*)query,
            (const half*)reference_points_cam,
            (const char4*)bev_mask,
            stream);
        break;

      default:
        return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return -1;
}

// # --IN---
// # query                     [bs, num_query, embed_dims]        float32
// # reference_points_cam      [num_cams, bs, num_query, D, 2]    float32
// # bev_mask                  [num_cams, bs, num_query, 4]       int8-p4
// #
// #  --OUT---
// # indexes                   [num_cams, max_len]                int32
// # indexes_len               [num_cams]                         int32
// # queries_rebatch           [bs*num_cams, max_len, embed_dims] float32
// # reference_points_rebatch  [bs*num_cams, max_len, D, 2]       float32
// # count_norm                [bs, num_query, 1]                 float32

DataType IndexRebatchPluginDynamic::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept {
  __LOG_PLACEHOLD__;
  ASSERT(index < num_out);
  ASSERT(nbInputs == num_in);
  ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);

  if (index > 1)
    return inputTypes[0];
  else
    return DataType::kINT32;
}

const char* IndexRebatchPluginDynamic::getPluginType() const noexcept {
  return PLUGIN_NAME;
}

const char* IndexRebatchPluginDynamic::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

int IndexRebatchPluginDynamic::getNbOutputs() const noexcept {
  return num_out;
}

int IndexRebatchPluginDynamic::initialize() noexcept {
  __LOG_PLACEHOLD__;
  pl_ = new index_rebatch::Pipeline(num_query_, max_len_, num_cams_, embed_dims_);
  pl_->Init();
  return 0;
}

void IndexRebatchPluginDynamic::terminate() noexcept {
  __LOG_PLACEHOLD__;

  if (pl_ != nullptr) {
    pl_->DeInit();

    delete pl_;
    pl_ = nullptr;
  }
}

size_t IndexRebatchPluginDynamic::getSerializationSize() const noexcept {
  return sizeof(num_query_) + sizeof(max_len_) + sizeof(num_cams_) + sizeof(embed_dims_);
}

void IndexRebatchPluginDynamic::serialize(void* buffer) const noexcept {
  __LOG_PLACEHOLD__;

  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write<int32_t>(d, num_query_);
  write<int32_t>(d, max_len_);
  write<int32_t>(d, num_cams_);
  write<int32_t>(d, embed_dims_);

  ASSERT(d == a + getSerializationSize());
}

void IndexRebatchPluginDynamic::destroy() noexcept {
  delete this;
}

void IndexRebatchPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* IndexRebatchPluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

IndexRebatchPluginDynamicCreator::IndexRebatchPluginDynamicCreator() {
  mPluginAttributes.clear();

  mPluginAttributes.emplace_back("num_query", nullptr, PluginFieldType::kINT32, 1);
  mPluginAttributes.emplace_back("max_len", nullptr, PluginFieldType::kINT32, 1);
  mPluginAttributes.emplace_back("num_cams", nullptr, PluginFieldType::kINT32, 1);
  mPluginAttributes.emplace_back("embed_dims", nullptr, PluginFieldType::kINT32, 1);

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* IndexRebatchPluginDynamicCreator::getPluginName() const noexcept {
  return PLUGIN_NAME;
}

const char* IndexRebatchPluginDynamicCreator::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

const PluginFieldCollection* IndexRebatchPluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2* IndexRebatchPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  try {
    __LOG_PLACEHOLD__;
    int num_query = 0, max_len = 0, num_cams = 0, embed_dims = 0;
    for (int i = 0; i < fc->nbFields; ++i) {
      const auto& fd = fc->fields[i];

      std::string field_name(fd.name);
      if (field_name.compare("num_query") == 0) {
        num_query = static_cast<const int*>(fd.data)[0];
        ASSERT(fd.type == PluginFieldType::kINT32);
        __LOG_INFO("Building num_query:%d", num_query);
      }

      if (field_name.compare("max_len") == 0) {
        max_len = static_cast<const int*>(fd.data)[0];
        ASSERT(fd.type == PluginFieldType::kINT32);
        __LOG_INFO("Building max_len:%d", max_len);
      }

      if (field_name.compare("num_cams") == 0) {
        num_cams = static_cast<const int*>(fd.data)[0];
        ASSERT(fd.type == PluginFieldType::kINT32);
        __LOG_INFO("Building num_cams:%d", num_cams);
      }

      if (field_name.compare("embed_dims") == 0) {
        embed_dims = static_cast<const int*>(fd.data)[0];
        ASSERT(fd.type == PluginFieldType::kINT32);
        __LOG_INFO("Building embed_dims:%d", embed_dims);
      }
    }

    auto* plugin = new IndexRebatchPluginDynamic(num_query, max_len, num_cams, embed_dims);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (const std::exception& e) {
    caughtError(e);
    __LOG_ERROR("%s\n", e.what());
  }
  return nullptr;
}

IPluginV2* IndexRebatchPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  try {
    auto* plugin = new IndexRebatchPluginDynamic(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return nullptr;
}

void IndexRebatchPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* IndexRebatchPluginDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}
