# Bulletin

SLEAP 1.3.2 adds some nice usability features thanks to both the community ideas and new contributors! See [1.3.0](https://github.com/talmolab/sleap/releases/tag/v1.3.0) and [1.3.1](https://github.com/talmolab/sleap/releases/tag/v1.3.1) for previous notable changes. As a reminder:

> The 1.3.1 dependency update requires [Mamba](https://mamba.readthedocs.io/en/latest/index.html) for faster dependency resolution. If you already have anaconda installed, then you _can_ set the solver to libmamba in the base environment:
>```
>conda update -n base conda
>conda install -n base conda-libmamba-solver
>conda config --set solver libmamba
>```
>Any subsequent `mamba` commands in the docs will need to be replaced with `conda` if you choose to use your existing Anaconda installation. 
>
>Otherwise, follow the [recommended installation instruction for Mamba](https://mamba.readthedocs.io/en/latest/installation.html).

### Quick install
**`mamba` (Windows/Linux/GPU)**:
```
mamba create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.2
```

**`mamba` (Mac)**:
```
mamba create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.3.2
```

**`pip` (any OS except Apple Silicon)**:
```
pip install sleap[pypi]==1.3.2
```

### Highlights
* Limit max tracks via track-local queues by @shrivaths16 and @talmo in https://github.com/talmolab/sleap/pull/1447
* Add option to remove videos in batch by @gitttt-1234 in https://github.com/talmolab/sleap/pull/1382 and https://github.com/talmolab/sleap/pull/1406
* Add shortcut to export analysis for current video by @KevinZ0217 in https://github.com/talmolab/sleap/pull/1414 and https://github.com/talmolab/sleap/pull/1444
* Add video path and frame indices to metrics by @roomrys in https://github.com/talmolab/sleap/pull/1396
* Add a button for copying model config to clipboard by @KevinZ0217 in https://github.com/talmolab/sleap/pull/1433
* Add Option to Export CSV by @gitttt-1234 in https://github.com/talmolab/sleap/pull/1438

### Full Changelog

#### Enhancements
* Add option to remove videos in batch by @gitttt-1234 in https://github.com/talmolab/sleap/pull/1382 and https://github.com/talmolab/sleap/pull/1406
* Add `Track` when add `Instance` by @roomrys in https://github.com/talmolab/sleap/pull/1408
* Add `Video` to cache when adding `Track` by @roomrys in https://github.com/talmolab/sleap/pull/1407
* Add shortcut to export analysis for current video by @KevinZ0217 in https://github.com/talmolab/sleap/pull/1414 and https://github.com/talmolab/sleap/pull/1444
* Add video path and frame indices to metrics by @roomrys in https://github.com/talmolab/sleap/pull/1396
* Improve error message for detecting video backend by @roomrys in https://github.com/talmolab/sleap/pull/1441
* Add a button for copying model config to clipboard by @KevinZ0217 in https://github.com/talmolab/sleap/pull/1433
* Add Option to Export CSV by @gitttt-1234 in https://github.com/talmolab/sleap/pull/1438
* Limit max tracks via track-local queues by @shrivaths16 and @talmo in https://github.com/talmolab/sleap/pull/1447

#### Fixes
* Minor fix in computation of OKS by @shrivaths16 in https://github.com/talmolab/sleap/pull/1383 and https://github.com/talmolab/sleap/pull/1399
* Fix `Filedialog` to work across (mac)OS by @roomrys in https://github.com/talmolab/sleap/pull/1393
* Fix panning bounding box by @gitttt-1234 in https://github.com/talmolab/sleap/pull/1398
* Fix skeleton templates by @roomrys in https://github.com/talmolab/sleap/pull/1404
* Fix labels export for json by @roomrys in https://github.com/talmolab/sleap/pull/1410
* Correct GUI state emulation by @roomrys in https://github.com/talmolab/sleap/pull/1422
* Update status message on status bar by @shrivaths16 in https://github.com/talmolab/sleap/pull/1411
* Fix error thrown when last video is deleted  by @shrivaths16 in https://github.com/talmolab/sleap/pull/1421
* Add model folder to the unzip path by @roomrys in https://github.com/talmolab/sleap/pull/1445
* Fix drag and drop by @talmo in https://github.com/talmolab/sleap/pull/1449

#### Dependencies
* Pin micromamba version by @roomrys in https://github.com/talmolab/sleap/pull/1376
* Add pip extras by @roomrys in https://github.com/talmolab/sleap/pull/1481

###3 New Contributors
* @shrivaths16 made their first contribution in https://github.com/talmolab/sleap/pull/1383
* @gitttt-1234 made their first contribution in https://github.com/talmolab/sleap/pull/1382
* @KevinZ0217 made their first contribution in https://github.com/talmolab/sleap/pull/1414

**Full Changelog**: https://github.com/talmolab/sleap/compare/v1.3.1...v1.3.2