@REM Install anything that didn't get conda installed via pip.

@REM We need to turn pip index back on because Anaconda turns it off for some reason.
set PIP_NO_INDEX=False
set PIP_NO_DEPENDENCIES=False
set PIP_IGNORE_INSTALLED=False

@REM Install the pip dependencies. Note: Using urls to wheels might be better: 
@REM https://docs.conda.io/projects/conda-build/en/stable/user-guide/wheel-files.html)
pip install -r .\requirements.txt

@REM HACK(LM): (untested) Uninstall all opencv packages and install opencv-contrib-python
for /f "tokens=1" %%a in ('conda list ^| findstr opencv') do pip uninstall %%a -y
pip install "opencv-contrib-python<4.7.0"

@REM Install sleap itself. This does not install the requirements, but will list which 
@REM requirements are missing (see "install_requires") when user attempts to install.
python setup.py install --single-version-externally-managed --record=record.txt
