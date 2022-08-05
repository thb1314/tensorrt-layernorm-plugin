/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <numeric>
#include <stdexcept>
#include "layerNormalizationPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::LayerNormalizationPlugin;
using nvinfer1::plugin::LayerNormalizationPluginCreator;

namespace
{
constexpr const char* LAYER_NORM_VERSION{"1"};
constexpr const char* LAYER_NORM_NAME{"LayerNormalizationPlugin"};
} // namespace

// // Static class fields initialization
PluginFieldCollection LayerNormalizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LayerNormalizationPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(LayerNormalizationPluginCreator);

LayerNormalizationPlugin::LayerNormalizationPlugin(float epsilon, int nbGroups)
    : mEpsilon(epsilon),
      mNbGroups(nbGroups)
{
    // Number of groups should be positive
    assert(nbGroups > 0);
}

int LayerNormalizationPlugin::initialize() noexcept
{
    return 0;
}

LayerNormalizationPlugin::LayerNormalizationPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mEpsilon);
    deserialize_value(&data, &length, &mNbGroups);
}

const char* LayerNormalizationPlugin::getPluginType() const noexcept
{
    return LAYER_NORM_NAME;
}

const char* LayerNormalizationPlugin::getPluginVersion() const noexcept
{
    return LAYER_NORM_VERSION;
}

int LayerNormalizationPlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs LayerNormalizationPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Input (from previous layer), scale and bias are the three inputs to the plugin.
    assert(nbInputs == 3);
    assert(index == 0);
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

// Detach the plugin object from its execution context.
void LayerNormalizationPlugin::detachFromContext() noexcept
{
    
}

int LayerNormalizationPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int batchSize = input_dims.d[0];
    int nbChannels = input_dims.d[1];
    bool use_fp16 = inputDesc[0].type == DataType::kHALF;
    bool use_fp32 = inputDesc[0].type == DataType::kFLOAT;

    mChannelVolume = std::accumulate(input_dims.d + 2, input_dims.d + inputDesc[0].dims.nbDims, 1, std::multiplies<int>());
    
    int need_size = batchSize * nbChannels;
    if(mean_len < need_size) {
        if(mean != nullptr) {
            cudaFree(mean);
            mean = nullptr;
        }
        mean_len = need_size;
        cudaMalloc(&mean, mean_len * sizeof(float));
    }
    if(inv_variance_len < need_size) {
        if(inv_variance != nullptr) {
            cudaFree(inv_variance);
            inv_variance = nullptr;
        }
        inv_variance_len = need_size;
        cudaMalloc(&inv_variance, inv_variance_len * sizeof(float));
    }

    if(use_fp16) {
        LayerNormForwardGpu(stream, batchSize * nbChannels, mChannelVolume,
                        mEpsilon, reinterpret_cast<const __half*>(inputs[0]), reinterpret_cast<const __half*>(inputs[1]),
                        reinterpret_cast<const __half*>(inputs[2]), reinterpret_cast<__half*>(outputs[0]), mean,
                        inv_variance);

    } else if(use_fp32) {
        LayerNormForwardGpu(stream, batchSize * nbChannels, mChannelVolume,
                        mEpsilon, reinterpret_cast<const float*>(inputs[0]), reinterpret_cast<const float*>(inputs[1]),
                        reinterpret_cast<const float*>(inputs[2]), reinterpret_cast<float*>(outputs[0]),mean,
                        inv_variance);        
    } else {
        printf("unsupported type!");
    }

    return 0;
}

size_t LayerNormalizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNbGroups) + sizeof(mEpsilon);
}

void LayerNormalizationPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mEpsilon);
    serialize_value(&buffer, mNbGroups);
}

bool LayerNormalizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(inOut && pos < (nbInputs + nbOutputs));
    // for(int i = 0; i < nbInputs; ++i) {
    //     printf("inOut[%d].type = %d\n", i, inOut[i].type);
    // }
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type);
}

void LayerNormalizationPlugin::terminate() noexcept
{
    cudaFree(mean);
    mean = nullptr;
    cudaFree(inv_variance);
    inv_variance = nullptr;
}

void LayerNormalizationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* LayerNormalizationPlugin::clone() const noexcept
{
    auto* plugin = new LayerNormalizationPlugin(mEpsilon, mNbGroups);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void LayerNormalizationPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    int batchSize = in[0].desc.dims.d[0] < 0 ? 256 : in[0].desc.dims.d[0];
    int nbChannels = in[0].desc.dims.d[1] < 0 ? 256 : in[0].desc.dims.d[1];

    // Allocate device memory and initialize scale and bias values
    int need_size = batchSize * nbChannels;
    if(mean_len < need_size) {
        if(mean != nullptr) {
            cudaFree(mean);
            mean = nullptr;
        }
        mean_len = need_size;
        cudaMalloc(&mean, mean_len * sizeof(float));
    }
    if(inv_variance_len < need_size) {
        if(inv_variance != nullptr) {
            cudaFree(inv_variance);
            inv_variance = nullptr;
        }
        inv_variance_len = need_size;
        cudaMalloc(&inv_variance, inv_variance_len * sizeof(float));
    }
}

nvinfer1::DataType LayerNormalizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

size_t LayerNormalizationPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void LayerNormalizationPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* LayerNormalizationPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

LayerNormalizationPluginCreator::LayerNormalizationPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LayerNormalizationPluginCreator::getPluginName() const noexcept
{
    return LAYER_NORM_NAME;
}

const char* LayerNormalizationPluginCreator::getPluginVersion() const noexcept
{
    return LAYER_NORM_VERSION;
}

const PluginFieldCollection* LayerNormalizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* LayerNormalizationPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void LayerNormalizationPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* LayerNormalizationPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // Set default values
    int nbGroups{1};
    float epsilon{0.00001F};
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
        if (field_name.compare("eps") == 0)
        {
            epsilon = *static_cast<const float*>(fc->fields[i].data);
        }
        if (field_name.compare("num_groups") == 0)
        {
            nbGroups = *static_cast<const int*>(fc->fields[i].data);
        }
    }

    LayerNormalizationPlugin* plugin = new LayerNormalizationPlugin(epsilon, nbGroups);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

IPluginV2DynamicExt* LayerNormalizationPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    LayerNormalizationPlugin* plugin = new LayerNormalizationPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}
