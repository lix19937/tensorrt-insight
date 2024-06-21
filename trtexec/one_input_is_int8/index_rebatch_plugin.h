/**************************************************************
 * @Author: ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#pragma once

#include <vector>
#include "NvInferPluginCustom.h"
#include "index_rebatch_warpper.h"
#include "pluginCustom.h"

namespace nvinfer1 {
namespace plugin_custom {

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class IndexRebatchPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  IndexRebatchPluginDynamic(const int num_query, const int max_len, const int num_cams, const int embed_dims);

  IndexRebatchPluginDynamic(const void* data, size_t length);

  // It doesn't make sense to make IndexRebatchPluginDynamic without arguments,
  // so we delete default constructor.
  // IndexRebatchPluginDynamic() = delete;

  ~IndexRebatchPluginDynamic() override;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(
      int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nbOutputs) noexcept override;
  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const noexcept override;
  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  void attachToContext(
      cudnnContext* cudnnContext,
      cublasContext* cublasContext,
      nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
  void detachFromContext() noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  std::string mNamespace;
  int num_query_; // num of bev_feature
  int max_len_; // valid num of bev_feature from config file, threshold
  int num_cams_;
  int embed_dims_;
  index_rebatch::Pipeline* pl_{nullptr};
};

class IndexRebatchPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  IndexRebatchPluginDynamicCreator();

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serialData, size_t serialLength) noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
} // namespace plugin_custom
} // namespace nvinfer1
