# tensorrt-layernorm-plugin

A Project for Layernorm TensorRT Plugin.  

Layernorm implementation is modified from oneflow.  

build and test step:
```bash
# change the CUDA_PATH and TRT_PATH in Makefile then
make
python testPlugin.py
```