#!/usr/bin/env python

from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

def get_requirements(require_name=None):
    prefix = require_name + '_' if require_name is not None else ''
    with open(path.join(here, prefix + 'requirements.txt'), encoding='utf-8') as f:
        return f.read().strip().replace('-gpu','').split('\n')


setup(
    name='sleap',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=get_requirements(),
    extras_require={
        'dev': get_requirements('dev'),
    },
    description='SLEAP (Social LEAP Estimates Animal Pose) is a deep learning framework for estimating animal pose.',
    long_description=long_description,
    author='Talmo Pereira, David Turner, Nat Tabris',
    author_email='talmo@princeton.edu',
    url='https://github.com/murthylab/sleap',
    keywords='deep learning, pose estimation, tracking, neuroscience',
    license='BSD3',
    packages=find_packages(exclude=['tensorflow']),
    include_package_data=True,
    entry_points = {
        'console_scripts': [
            'sleap-label=sleap.gui.app:main',
            'sleap-train=sleap.nn.training:main',
            'sleap-track=sleap.nn.inference:main'
        ],
    },
    python_requires='>=3.6'

)

