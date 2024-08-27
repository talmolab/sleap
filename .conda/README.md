This folder defines the conda package build for Linux and Windows. There are runners for both Linux and Windows on GitHub Actions, but it is faster to experiment with builds locally first.
Test workflow: this should not trigger build_manual.

To build, first go to the base repo directory and install the build environment:

```
mamba env create -f environment_build.yml -n sleap_build && conda activate sleap_build
```

And finally, run the build command pointing to this directory:

```
conda build .conda --output-folder build -c conda-forge -c nvidia -c https://conda.anaconda.org/sleap/ -c anaconda
```

To install the local package:

```
mamba create -n sleap_0 -c conda-forge -c nvidia -c ./build -c https://conda.anaconda.org/sleap/ -c anaconda sleap=x.x.x
```

replacing x.x.x with the version of SLEAP that you just built.
