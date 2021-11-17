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

# pip install numpy==1.19.5
pip install attrs==21.2.0
pip install cattrs==1.1.1
pip install jsonpickle==1.2
pip install networkx
# pip install tensorflow==2.7.0
# pip install h5py==3.10.0
pip install python-rapidjson
# pip install opencv-python-headless==4.2.0.34
pip install pandas
pip install psutil
# pip install PySide2==5.15.2
pip install pyzmq
pip install pyyaml
pip install imgaug==0.4.0
pip install "scipy<=1.4.1"
pip install scikit-image
pip install scikit-learn
pip install scikit-video
pip install imgstore==0.2.9
pip install qimage2ndarray==1.8
pip install jsmin
pip install seaborn
pip install pykalman==0.9.5
pip install segmentation-models==1.0.1
pip install rich==9.10.0

pip install setuptools-scm

python setup.py install --single-version-externally-managed --record=record.txt
