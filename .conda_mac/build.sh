#!/usr/bin/env bash

# Install anything that didn't get conda installed via pip.
# We need to turn pip index back on because Anaconda turns it off for some reason. 

export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

pip install -r requirements.txt

# HACK(LM): (untested) Uninstall all opencv packages and install opencv-contrib-python
conda list | grep opencv | awk '{system("pip uninstall " $1 " -y")}'
pip install "opencv-contrib-python<4.7.0"

python setup.py install --single-version-externally-managed --record=record.txt