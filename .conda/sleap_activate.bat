@REM Remember the old library path for when we deactivate
set SLEAP_OLD_XLA_FLAGS=%XLA_FLAGS%
@REM Help XLA find CUDA
set XLA_FLAGS=--xla_gpu_cuda_data_dir=%CONDA_PREFIX%