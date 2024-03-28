# Install anything that didn't get conda installed via pip.

# We need to turn pip index back on because Anaconda turns it off for some reason.
export PIP_NO_INDEX=False
export PIP_NO_DEPENDENCIES=False
export PIP_IGNORE_INSTALLED=False

# Install the pip dependencies. Note: Using urls to wheels might be better: 
# https://docs.conda.io/projects/conda-build/en/stable/user-guide/wheel-files.html)
pip install --no-cache-dir -r ./requirements.txt


# Install sleap itself. This does not install the requirements, but will list which 
# requirements are missing (see "install_requires") when user attempts to install.
python setup.py install --single-version-externally-managed --record=record.txt

# Copy the activate scripts to $PREFIX/etc/conda/activate.d.
# This will allow them to be run on environment activation.
for CHANGE in "activate" "deactivate"
do
    mkdir -p "${PREFIX}/etc/conda/${CHANGE}.d"
    cp "${RECIPE_DIR}/${PKG_NAME}_${CHANGE}.sh" "${PREFIX}/etc/conda/${CHANGE}.d/${PKG_NAME}_${CHANGE}.sh"
done