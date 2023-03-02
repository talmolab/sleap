This folder defines the conda package build for Apple silicon Macs. Until there are aarm64 runners, we have to run this manually on Apple M1 silicon.

To build, first go to the base repo directory and install the Apple silicon compatible environment:

```
conda env create -f environment_apple_silicon.yml -n sleap_build && conda activate sleap_build
```

Next, install build dependencies:

```
conda install conda-build=3.21.7 && conda install anaconda-client && conda install conda-verify
```

And finally, run the build command pointing to this directory:

```
conda build .conda_apple_silicon --output-folder build -c https://conda.anaconda.org/sleap/ -c nvidia -c conda-forge -c apple
```
