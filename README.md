[![Build status](https://ci.appveyor.com/api/projects/status/tf5qlylwqse8ack5/branch/develop?svg=true)](https://ci.appveyor.com/project/talmo/sleap/branch/develop)

# Social LEAP Estimates Animal Pose (sLEAP)
A deep learning framework for estimating animal pose.

# Installation

This has proven successful at least once:
```
conda create -n sleap --file win10_conda_spec.txt
conda activate sleap
pip install -r dev_requirements.txt
pip install PySide2==5.12.0
pip install --no-deps -e .
pip install opencv-python==3.4.5.20
conda install ipython
pip install imgaug
pip install sklearn
```
