@REM @echo off

set PIP_NO_INDEX=False
set PIP_NO_DEPENDENCIES=False
set PIP_IGNORE_INSTALLED=False

echo Running bld.bat

@REM Install the pip dependencies. Note: Using urls to wheels might be better: 
@REM https://docs.conda.io/projects/conda-build/en/stable/user-guide/wheel-files.html)
pip install -r .\requirements.txt

@REM Install sleap itself. This does not install the requirements, but will list which 
@REM requirements are missing (see "install_requires") when user attempts to install.
python setup.py install --single-version-externally-managed --record=record.txt
