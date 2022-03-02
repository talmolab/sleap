@echo off

rem # Install anything that didn't get conda installed via pip.
rem # We need to turn pip index back on because Anaconda turns
rem # it off for some reason. Just pip install -r requirements.txt
rem # doesn't seem to work, tensorflow-gpu, jsonpickle, networkx,
rem # all get installed twice if we do this. pip doesn't see the
rem # conda install of the packages.

rem # Install the pip dependencies and their dependencies. Conda turns of
rem # pip index and dependencies by default so re-enable them. Had to figure
rem # this out myself, ughhh.
set PIP_NO_INDEX=False
set PIP_NO_DEPENDENCIES=False
set PIP_IGNORE_INSTALLED=False

pip install numpy==1.19.5
pip install six==1.15.0
pip install imageio==2.15.0
pip install attrs==21.2.0
pip install cattrs==1.1.1
pip install jsonpickle==1.2
pip install networkx
@REM pip install tensorflow>=2.6.3,<=2.7.1
@REM pip install h5py>=3.1.0,<=3.6.0
pip install python-rapidjson
@REM pip install opencv-python-headless>=4.2.0.34,<=4.5.5.62
@REM pip install opencv-python @ git+https://github.com/talmolab/wrap_opencv-python-headless.git@ede49f6a23a73033216339f29515e59d594ba921
@REM pip install pandas
pip install psutil
@REM pip install PySide2>=5.13.2,<=5.14.1
pip install pyzmq
pip install pyyaml
pip install imgaug==0.4.0
@REM pip install scipy>=1.4.1,<=1.7.3
pip install scikit-image
pip install scikit-learn==1.0.*
pip install scikit-video
pip install imgstore==0.2.9
pip install qimage2ndarray==1.8.3
pip install jsmin
pip install seaborn
pip install pykalman==0.9.5
pip install segmentation-models==1.0.1
pip install rich==10.16.1
pip install certifi==2021.10.8

rem # Use and update environment.yml call to install pip dependencies. This is slick.
rem # While environment.yml contains the non pip dependencies, the only thing left
rem # uninstalled should be the pip stuff because that is left out of meta.yml
rem conda env update -f=environment.yml

rem # Install requires setuptools-scm
pip install setuptools-scm

rem # Install sleap itself.
rem # NOTE: This is the recommended way to install packages
python setup.py install --single-version-externally-managed --record=record.txt
