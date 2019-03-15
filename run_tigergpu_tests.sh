#!/bin/bash

module load anaconda

conda activate sleap

python -m pytest --ignore="tests/gui" tests/