import os
import ctypes
import numpy as np
# cuda: https://nvidia.github.io/cuda-python/
from cuda import cudart
import tensorrt as trt
import torch

soFile = "./layernorm_plugin.so"
epsilon = 1.0e-2
np.random.seed(97)


def printArrayInfo(x, description=""):
    print('%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e' % (
        description, str(x.shape), np.mean(x), np.sum(abs(x)), np.var(x), np.max(x), np.min(x), np.sum(np.abs(np.diff(x.reshape(-1))))))
    print("\t", x.reshape(-1)[:10])


def check(a, b, weak=False):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check:", res, "maxAbsDiff:", diff0, "maxRelDiff:", diff1)


def getLayerNormalizationPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'LayerNormalizationPlugin':
            parameterList = []
            parameterList.append(trt.PluginField(
                "eps", np.float32(1e-5), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None


use_fp16 = True


def run():

    trtFile = "./layernorm-plugin.plan"
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 6 << 30

        if builder.platform_has_fast_fp16 and use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if use_fp16:
            inputT0 = network.add_input(
                'x', trt.DataType.HALF, [-1 for i in range(3)])
            weight = network.add_input('weight', trt.DataType.HALF, [-1])
            bias = network.add_input('bias', trt.DataType.HALF, [-1])
        else:
            inputT0 = network.add_input(
                'x', trt.DataType.FLOAT, [-1 for i in range(3)])
            weight = network.add_input('weight', trt.DataType.FLOAT, [-1])
            bias = network.add_input('bias', trt.DataType.FLOAT, [-1])
        profile.set_shape(inputT0.name, [1, 1, 1], [8, 63, 256], [64, 63, 256])
        profile.set_shape(weight.name, [1, ], [8, ], [64, ])
        profile.set_shape(bias.name, [1, ], [8, ], [64, ])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2(
            [inputT0, weight, bias], getLayerNormalizationPlugin())
        # pluginLayer.
        network.mark_output(pluginLayer.get_output(0))
        if use_fp16:
            pluginLayer.precision = trt.float16
            pluginLayer.set_output_type(0, trt.float16)
            network.get_output(0).dtype = trt.float16
        print('type', network.get_output(0).dtype)
        engineString = builder.build_serialized_network(network, config)
        if engineString is None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    shape = (2, 32, 10)

    context.set_binding_shape(0, shape)
    context.set_binding_shape(1, [shape[-1]])
    context.set_binding_shape(2, [shape[-1]])
    _, stream = cudart.cudaStreamCreate()

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput

    bufferH = []
    data_type = np.float16 if use_fp16 else np.float32
    data = np.random.rand(np.prod(shape)).astype(data_type).reshape(shape) * 200 - 100
    print("min, max:", data.min(), data.max())

    weight_data = np.ones((shape[-1], ), dtype=data_type)
    bias_data = np.zeros((shape[-1], ), dtype=data_type)

    bufferH.append(data)
    bufferH.append(weight_data)
    bufferH.append(bias_data)
    print('nOutput:', nOutput)
    for i in range(nOutput):
        print('context.get_binding_shape(nInput + i)',
              context.get_binding_shape(nInput + i))
        bufferH.append(np.empty(context.get_binding_shape(
            nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMallocAsync(bufferH[i].nbytes, stream)[1])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], np.ascontiguousarray(
            bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v2(bufferD, stream)

    for i in range(nOutput):
        cudart.cudaMemcpyAsync(bufferH[nInput + i].ctypes.data, bufferD[nInput + i],
                               bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)

    mean = np.mean(bufferH[0], axis=-1, keepdims=True)
    std = np.sqrt((np.mean((bufferH[0] - mean) ** 2, axis=-1, keepdims=True) + 1e-5))
    a = (bufferH[0] - mean) / std
    weight = bufferH[1]
    bias = bufferH[2]
    a = weight.reshape(1, 1, -1) * a + bias.reshape(1, 1, -1)
    print("bufferH[-1].dtype: ", bufferH[-1].dtype)
    print('diff abs max', np.abs(a - bufferH[-1].astype(data_type)).max())
    t = torch.as_tensor(bufferH[0])

    cudart.cudaStreamDestroy(stream)
    for buffer in bufferD:
        cudart.cudaFree(buffer)

if __name__ == '__main__':
    os.system('rm ./layernorm-plugin.plan')
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run()
    print("test finish!")
