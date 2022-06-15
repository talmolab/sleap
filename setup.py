#!/usr/bin/env python

from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

# Get the sleap version
with open(path.join(here, "sleap/version.py")) as f:
    version_file = f.read()
    sleap_version = re.search('__version__ = "([0-9\\.a]+)"', version_file).group(1)


def get_requirements(require_name=None):
    prefix = require_name + "_" if require_name is not None else ""
    with open(path.join(here, prefix + "requirements.txt"), encoding="utf-8") as f:
        return f.read().strip().replace("-gpu", "").split("\n")


setup(
    name="sleap",
    version=sleap_version,
    setup_requires=["setuptools_scm"],
    install_requires=get_requirements(),
    extras_require={
        "dev": get_requirements("dev"),
    },
    description="SLEAP (Social LEAP Estimates Animal Poses) is a deep learning framework for animal pose tracking.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Talmo Pereira",
    author_email="talmo@salk.edu",
    project_urls={
        "Documentation": "https://sleap.ai/",
        "Bug Tracker": "https://github.com/talmolab/sleap/issues",
        "Source Code": "https://github.com/talmolab/sleap",
    },
    url="https://sleap.ai",
    keywords="deep learning, pose estimation, tracking, neuroscience",
    license="BSD 3-Clause License",
    packages=find_packages(exclude=["tensorflow"]),
    include_package_data=True,
    entry_points={ 
        "console_scripts": [
            "sleap-convert=sleap.io.convert:main",
            "sleap-render=sleap.io.visuals:main",
            "sleap-label=sleap.gui.app:main",
            "sleap-train=sleap.nn.training:main",
            "sleap-track=sleap.nn.inference:main",
            "sleap-inspect=sleap.info.labels:main",
            "sleap-diagnostic=sleap.diagnostic:main",
        ],
    },
    python_requires=">=3.6",
)
