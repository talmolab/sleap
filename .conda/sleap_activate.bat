@REM Remember the old library path for when we deactivate
set SLEAP_OLD_XLA_FLAGS=%XLA_FLAGS%
set SLEAP_OLD_NO_ALBUMENTATIONS_UPDATE=%NO_ALBUMENTATIONS_UPDATE%
@REM Help XLA find CUDA
set XLA_FLAGS=--xla_gpu_cuda_data_dir=%CONDA_PREFIX%
set NO_ALBUMENTATIONS_UPDATE=1