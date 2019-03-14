#!/bin/bash

module load anaconda

conda activate sleap

python -m pytest tests/ 