@REM @echo off

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

@REM pip install numpy==1.19.5
@REM pip install six==1.15.0
@REM pip install imageio==2.15.0
@REM pip install attrs==21.2.0
@REM pip install cattrs==1.1.1
@REM pip install jsonpickle==1.2
@REM pip install networkx
@REM pip install nixio>=1.5.3
@REM @REM pip install tensorflow>=2.6.3,<=2.7.1
@REM @REM pip install h5py>=3.1.0,<=3.6.0
@REM pip install python-rapidjson
@REM @REM pip install opencv-python-headless>=4.2.0.34,<=4.5.5.62
@REM @REM pip install opencv-python @ git+https://github.com/talmolab/wrap_opencv-python-headless.git@ede49f6a23a73033216339f29515e59d594ba921
@REM @REM pip install pandas
@REM pip install psutil
@REM @REM pip install PySide2>=5.13.2,<=5.14.1
@REM pip install pyzmq
@REM pip install pyyaml
@REM pip install imgaug==0.4.0
@REM @REM pip install scipy>=1.4.1,<=1.7.3
@REM pip install scikit-image
@REM pip install scikit-learn==1.0.*
@REM pip install scikit-video
@REM pip install tensorflow-hub
@REM pip install imgstore==0.2.9
@REM pip install qimage2ndarray==1.9.0
@REM pip install jsmin
@REM pip install seaborn
@REM pip install pykalman==0.9.5
@REM pip install segmentation-models==1.0.1
@REM pip install rich==10.16.1
@REM pip install certifi==2021.10.8
@REM pip install pynwb
@REM pip install ndx-pose

rem # Use and update environment.yml call to install pip dependencies. This is slick.
rem # While environment.yml contains the non pip dependencies, the only thing left
rem # uninstalled should be the pip stuff because that is left out of meta.yml
rem conda env update -f=environment.yml

rem # Install requires setuptools-scm
@REM pip install setuptools-scm

rem # Install sleap itself.
rem # NOTE: This is the recommended way to install packages
@REM python setup.py install --single-version-externally-managed --record=record.txt

echo "Running bld.bat"
@REM pip install imgstore<0.3.0  # 0.3.3 results in https://github.com/O365/python-o365/issues/591
@REM pip install ndx-pose
@REM pip install nixio>=1.5.3  # Constrain put on by @jgrewe from G-Node
@REM pip install qimage2ndarray  #==1.9.0
@REM pip install segmentation-models

@REM @REM # Conda installing results in https://github.com/h5py/h5py/issues/2037
@REM pip install h5py<3.2  # Newer versions result in error above
@REM pip install pynwb>2.0.0  # Required by ndx-pose
pip install -r requirements.txt

python setup.py install --single-version-externally-managed --record=record.txt
