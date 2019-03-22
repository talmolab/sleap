#!/bin/bash

module load anaconda
module load cudnn/cuda-10.0/7.3.1

conda activate sleap

python -m pytest --ignore="tests/gui" tests/