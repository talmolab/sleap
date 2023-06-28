This folder defines the conda package build for Apple silicon Macs. Until there are aarm64 runners, we have to run this manually on Apple M1 silicon.

To build, first go to the base repo directory and install the build environment:

```
conda env create -f environment_build.yml -n sleap_build && conda activate sleap_build
```

And finally, run the build command pointing to this directory:

```
conda build .conda_mac --output-folder build -c conda-forge -c anaconda
```

To install the local package:

```
conda create -n sleap_0 -c conda-forge -c anaconda -c ./build sleap=x.x.x
```

replacing x.x.x with the version of SLEAP that you just built.
