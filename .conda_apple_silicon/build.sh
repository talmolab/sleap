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

# pip install numpy==1.22.3
pip install attrs==21.4.0
pip install cattrs==1.1.1
pip install jsonpickle==1.2
pip install networkx
# pip install tensorflow>=2.6.3,<2.9.0; platform_machine != 'arm64'
pip install tensorflow-macos==2.9.2
pip install tensorflow-metal==0.5.0
# pip install h5py==3.6.0
pip install python-rapidjson
# pip install opencv-python==4.6.0
pip install pandas
pip install psutil
# pip install PySide2==5.15.5
pip install pyzmq
pip install pyyaml
# pip install pillow==8.4.0
pip install imageio<=2.15.0
pip install imgaug==0.4.0
# pip install scipy==1.7.3
pip install scikit-image
pip install scikit-learn==1.0.*
pip install scikit-video
pip install imgstore==0.2.9
pip install qimage2ndarray==1.9.0
pip install jsmin
pip install seaborn
pip install pykalman==0.9.5
pip install segmentation-models==1.0.1
pip install rich==10.16.1
pip install certifi==2021.10.8
pip install pynwb
pip install ndx-pose


pip install setuptools-scm

python setup.py install --single-version-externally-managed --record=record.txt