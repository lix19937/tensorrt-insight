/**************************************************************
 * @Author: lix19937
 * @Date: 2022-03-29 11:42:57
 * @Last Modified by: lix19937
 * @Last Modified time: 2022-04-15 17:40:58
 **************************************************************/

#include "stub_plugin.h"
#include <NvInfer.h>
#include <cuda_fp16.h>

#include "utils/file.h"
#include "utils/logger.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin_custom;

using nvinfer1::plugin_custom::StubPluginDynamic;
using nvinfer1::plugin_custom::StubPluginDynamicCreator;

namespace {
/// m in, 1 out
constexpr int num_in = 8;
constexpr int num_out = 1;
const char* STUB_VERSION{"1"};
const char* STUB_NAME{"SCATT"};
} // namespace

PluginFieldCollection StubPluginDynamicCreator::mFC{};
std::vector<PluginField> StubPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(StubPluginDynamicCreator);

constexpr size_t maxWorkspaceBytes{0};

StubPluginDynamic::StubPluginDynamic(const std::string& name) : mLayerName(name) {
}

StubPluginDynamic::StubPluginDynamic(const std::string& name, const void* data, size_t length) : mLayerName(name) {

  const char *d = reinterpret_cast<const char*>(data), *a = d;
  mType = read<DataType>(d);
  ASSERT(d == a + length);
}

StubPluginDynamic::~StubPluginDynamic() {
  terminate();
}

IPluginV2DynamicExt* StubPluginDynamic::clone() const noexcept {
  try {
    auto p = new StubPluginDynamic(mLayerName);
    p->setPluginNamespace(mNamespace.c_str());
    p->mType = mType;

    if (0 != p->initialize()) {
      __LOG_INFO("StubPluginDynamic init failed");
      return nullptr;
    }

    return p;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return nullptr;
}

void StubPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept {
}

void StubPluginDynamic::detachFromContext() noexcept {
}

DimsExprs StubPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {

  try {
    ASSERT(nbInputs == num_in);

    DimsExprs ret;
    ret.nbDims = inputs[0].nbDims; /// the same as the first input
    for (int i = 0; i < ret.nbDims; ++i) {
      ret.d[i] = inputs[0].d[i];
    }

    return ret;
  } catch (const std::exception& e) {
    caughtError(e);
  }

  return DimsExprs{};
}

bool StubPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {

  ASSERT(0 <= pos && pos < nbInputs + nbOutputs);

  const PluginTensorDesc& desc = inOut[pos];
  bool ret = false;
  switch (pos) {
    case 0:
      // HWC is not support fp16
      ret = (desc.type == DataType::kFLOAT && desc.format == nvinfer1::TensorFormat::kLINEAR) ||
          (desc.type == DataType::kHALF && desc.format == nvinfer1::TensorFormat::kLINEAR) ||
          (desc.type == DataType::kINT8 && desc.format == nvinfer1::TensorFormat::kCHW32);
      break;

    case 1:
    case 2:
    case 3:
    case 5:
      ret = (desc.type == inOut[0].type && desc.format == inOut[0].format);
      break;

    case 4:
    case 6:
    case 7:
      ret = (desc.type == DataType::kINT32 && desc.format == TensorFormat::kLINEAR);
      break;

    case 8:
      ret = (desc.type == inOut[0].type && desc.format == inOut[0].format);
      break;
  }
  return ret;
}

void StubPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const DynamicPluginTensorDesc* outputs,
    int nbOutputs) noexcept {
  mType = inputs[0].desc.type;
}

size_t StubPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  return maxWorkspaceBytes;
}

int StubPluginDynamic::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workSpace,
    cudaStream_t stream) noexcept {

  try {
    auto out = outputs[0];
    size_t len = 1;
    for (int i = 0; i < outputDesc[0].dims.nbDims; ++i) {
      len *= outputDesc[0].dims.d[i];
    }

    for (int i = 0; i < num_in; ++i) {
      __LOG_INFO("enter branch ...I#%d %s", i, get_tensor_desc_fmt_str(inputDesc[i]).c_str());
    }
    for (int i = 0; i < num_out; ++i) {
      __LOG_INFO("enter branch ...O#%d %s", i, get_tensor_desc_fmt_str(outputDesc[i]).c_str());
    }

    if (inputDesc[0].type == DataType::kINT8) {
    } else if (inputDesc[0].type == DataType::kHALF && inputDesc[0].format == nvinfer1::TensorFormat::kLINEAR) {
      // auto in = (const half*)(inputs[0]);

      // std::string base_path = "./";
      // utils::write_from_dbuff(
      //     base_path + "in_fp16_chw",
      //     {1, 5, 1, 2}, // nchw
      //     in);

      // cudaMemcpyAsync(out, in, sizeof(half) * len, cudaMemcpyDeviceToDevice, stream);
    } else if (inputDesc[0].type == DataType::kHALF && inputDesc[0].format == nvinfer1::TensorFormat::kHWC8) {
      // auto in = (const half*)(inputs[0]);

      // std::string base_path = "./";
      // cudaacc::WriteFromDBuff(
      //     base_path + "in_fp16_hwc8",
      //     {1, 1, 2, 8}, // nhwc8
      //     in);

      // cudaMemcpyAsync(out, in, 5 * sizeof(half), cudaMemcpyDeviceToDevice, stream);
      // cudaMemcpyAsync(
      //     (char*)out + 5 * sizeof(half),
      //     (char*)in + 8 * sizeof(half),
      //     sizeof(half) * 5,
      //     cudaMemcpyDeviceToDevice,
      //     stream);
    } else if (inputDesc[0].type == DataType::kFLOAT) {
      auto in = (const float*)(inputs[0]);

      // std::string base_path = "./";
      // utils::write_from_dbuff(
      //     base_path + "in_fp32",
      //     {1, 5, 1, 2}, // nchw
      //     //{1, 1, 2, 8}, // nhwc
      //     //{1, 1, 2, 8}, // nhwc8
      //     in);

      cudaMemcpyAsync(out, in, sizeof(float) * len, cudaMemcpyDeviceToDevice, stream);
    }

    return 0;
  } catch (const std::exception& e) {
    __LOG_INFO("enqueue get exception, failed");
    caughtError(e);
  }

  __LOG_INFO("enqueue failed");
  return -1;
}

DataType StubPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept {
  return inputTypes[0];
}

const char* StubPluginDynamic::getPluginType() const noexcept {
  return STUB_NAME;
}

const char* StubPluginDynamic::getPluginVersion() const noexcept {
  return STUB_VERSION;
}

int StubPluginDynamic::getNbOutputs() const noexcept {
  return num_out;
}

int StubPluginDynamic::initialize() noexcept {
  return 0;
}

void StubPluginDynamic::terminate() noexcept {
}

size_t StubPluginDynamic::getSerializationSize() const noexcept {
  return sizeof(DataType);
}

void StubPluginDynamic::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  write<DataType>(d, mType);
  ASSERT(d == a + getSerializationSize());
}

void StubPluginDynamic::destroy() noexcept {
  delete this;
}

void StubPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* StubPluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
StubPluginDynamicCreator::StubPluginDynamicCreator() {
  mPluginAttributes.clear();

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* StubPluginDynamicCreator::getPluginName() const noexcept {
  return STUB_NAME;
}

const char* StubPluginDynamicCreator::getPluginVersion() const noexcept {
  return STUB_VERSION;
}

const PluginFieldCollection* StubPluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2* StubPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
  try {
    auto p = new StubPluginDynamic(name);
    p->setPluginNamespace(mNamespace.c_str());

    if (0 != p->initialize()) {
      __LOG_INFO("deserializePlugin initialize failed");
      ASSERT(false);
    }
    return p;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV2* StubPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  try {
    auto p = new StubPluginDynamic(name, serialData, serialLength);
    p->setPluginNamespace(mNamespace.c_str());

    if (0 != p->initialize()) {
      __LOG_INFO("deserializePlugin initialize failed");
      ASSERT(false);
    }

    return p;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return nullptr;
}

void StubPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* StubPluginDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

// trtexec --onnx=sca_simp.onnx --verbose --plugins=libplugin_custom.so   --dumpProfile

// trtexec --onnx=sca_simp.onnx --verbose --plugins=libplugin_custom.so --dumpProfile
//  --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,fp32:chw,int8:chw,int32:chw

/*
trtexec --onnx=/mnt/d/workspace/BEVFormer-master-infer/bevf2_simp.onnx \
--verbose --plugins=libplugin_custom.so \
--dumpProfile  --duration=0  --iterations=1 --avgRuns=1   --fp16 2>&1 | tee log

*/
