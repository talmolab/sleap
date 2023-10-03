#!/bin/sh

# Remember the old library path for when we deactivate
export SLEAP_OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
# Help CUDA find GPUs!
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH