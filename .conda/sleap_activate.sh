#!/bin/sh

# Remember the old variables for when we deactivate
export SLEAP_OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export SLEAP_OLD_XLA_FLAGS=$XLA_FLAGS
# Help CUDA find GPUs!
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Help XLA find CUDA
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
# Disable annoying albumentations message
export NO_ALBUMENTATIONS_UPDATE=1