@REM Install anything that didn't get conda installed via pip.

@REM We need to turn pip index back on because Anaconda turns it off for some reason.
set PIP_NO_INDEX=False
set PIP_NO_DEPENDENCIES=False
set PIP_IGNORE_INSTALLED=False

@REM Install the pip dependencies. Note: Using urls to wheels might be better: 
@REM https://docs.conda.io/projects/conda-build/en/stable/user-guide/wheel-files.html)
pip install --no-cache-dir -r .\requirements.txt

@REM Install sleap itself. This does not install the requirements, but will list which 
@REM requirements are missing (see "install_requires") when user attempts to install.
python setup.py install --single-version-externally-managed --record=record.txt

@REM Copied from https://docs.conda.io/projects/conda-build/en/latest/resources/activate-scripts.html
setlocal EnableDelayedExpansion
:: Copy the [de]activate scripts to %PREFIX%\etc\conda\[de]activate.d.
:: This will allow them to be run on environment activation.
for %%F in (activate deactivate) DO (
    if not exist %PREFIX%\etc\conda\%%F.d mkdir %PREFIX%\etc\conda\%%F.d
    copy %RECIPE_DIR%\%%F.bat %PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F.bat
    :: Copy unix shell activation scripts, needed by Windows Bash users
    copy %RECIPE_DIR%\%%F.sh %PREFIX%\etc\conda\%%F.d\%PKG_NAME%_%%F.sh
)