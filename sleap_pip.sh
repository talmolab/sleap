#!/bin/bash 

# To run this script, first open a terminal and run these two commands:
# chmod +x sleap_pip.sh
# ./sleap_pip.sh

# Create and activate the environment for building the wheel.
micromamba env create -y -n sleap_ci -f environment_build.yml
source micromamba activate sleap_ci
micromamba env list

# # Build the wheel.
# micromamba run -n sleap_ci "python -m build"

# # Create and activate the environment for testing the wheel.
# micromamba env create -y -n sleap_pip python=3.7 pip -c conda-forge -c anaconda
# source activate sleap_pip

# # Purge the cache and install the wheel with extras
# pip cache purge
# pip install 'sleap[pypi] @ file:///home/talmolab/sleap-estimates-animal-poses/pull-requests/sleap/dist/sleap-1.3.2-py3-none-any.whl'

# # Perform simple test.
# python -c "import sleap"