# Bulletin

This is a brown-bag release following insufficient restrictions on allowable `tensorflow` versions for the "pypi" extra `sleap[pypi]` in 1.3.2. While the conda packages for 1.3.2 were not affected (since `tensorflow` is pulled in from anaconda), the PyPI only package installed via `pip install sleap[pypi]` had conflicts between the version of `tensorflow` and the version of `keras`. See [1.3.0](https://github.com/talmolab/sleap/releases/tag/v1.3.0), [1.3.1](https://github.com/talmolab/sleap/releases/tag/v1.3.1), and  [1.3.2](https://github.com/talmolab/sleap/releases/tag/v1.3.2) for previous notable changes. As a reminder:

> The 1.3.1 dependency update requires [Mamba](https://mamba.readthedocs.io/en/latest/index.html) for faster dependency resolution. If you already have anaconda installed, then you _can_ set the solver to libmamba in the base environment:
>```
>conda update -n base conda
>conda install -n base conda-libmamba-solver
>conda config --set solver libmamba
>```
>Any subsequent `mamba` commands in the docs will need to be replaced with `conda` if you choose to use your existing Anaconda installation. 
>
>Otherwise, follow the [recommended installation instruction for Mamba](https://mamba.readthedocs.io/en/latest/installation.html).

# Quick install
**`mamba` (Windows/Linux/GPU)**:
```
mamba create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.3
```

**`mamba` (Mac)**:
```
mamba create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.3.3
```

**`pip` (any OS except Apple Silicon)**:
```
pip install sleap[pypi]==1.3.3
```

# Full Changelog

## Dependencies
* Add version restrictions to tensorflow for pypi by @roomrys in https://github.com/talmolab/sleap/pull/1485

**Full Changelog**: https://github.com/talmolab/sleap/compare/v1.3.2...v1.3.3