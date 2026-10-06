#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES=''
export CUDA_HOME=/usr/local/cuda
export CUTLASS_DIR=/home/hwlee/mgo-tools/headline-r4/cutlass
export CPATH=/home/hwlee/mgo-tools/headline-r4/NVTX/c/include:/home/hwlee/anaconda3/envs/qwen3-gemm/include
export LIBRARY_PATH=/home/hwlee/anaconda3/envs/qwen3-gemm/lib
export MAX_JOBS=8
export PATH=/home/hwlee/mgo-tools/headline-r4/infinity-env/bin:/usr/local/cuda/bin:$PATH
cd /home/hwlee/mgo-tools/headline-r4/MoE-Infinity
python setup.py build_ext --build-lib /home/hwlee/mgo-tools/headline-r4/eam-next-build --build-temp /home/hwlee/mgo-tools/headline-r4/eam-next-objects
