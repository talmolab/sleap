#!/usr/bin/env bash

# Install anything that didn't get conda installed via pip.
# We need to turn pip index back on because Anaconda turns
# it off for some reason. Just pip install -r requirements.txt
# doesn't seem to work, tensorflow-gpu, jsonpickle, networkx,
# all get installed twice if we do this. pip doesn't see the
# conda install of the packages.

export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

pip install cattrs==1.0.0rc opencv-python-headless==3.4.1.15 PySide2==5.12.0 imgaug qimage2ndarray==1.8 imgstore jsmin

pip install setuptools-scm

python setup.py install --single-version-externally-managed --record=record.txt
