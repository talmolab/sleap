#!/bin/bash

module load anaconda
module load cudnn/cuda-10.1/7.5.0

conda activate sleap_test

python -m pytest --ignore="tests/gui" tests/