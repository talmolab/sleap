# Installation

SLEAP can be installed as a Python package on Windows, Linux, and Mac OS.

The newest version of SLEAP can always be found in the [Releases page](https://github.com/talmolab/sleap/releases).

```{contents} Contents
---
local:
---
```


## Package Manager

SLEAP requires many complex dependencies, so we **strongly** recommend using a package manager such as [Miniconda](https://docs.anaconda.com/free/miniconda/) to install SLEAP in its own isolated environment. See the [Miniconda website](https://docs.anaconda.com/free/miniconda/) for installation instructions. The Anaconda or Mamba package managers will also work well; however, take care not to install multiple different conda-based package managers - choose one and stick with it.

````{note}
If you already have Anaconda on your computer (and it is an [older installation](https://conda.org/blog/2023-11-06-conda-23-10-0-release/)), then make sure to [set the solver to `libmamba`](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) in the `base` environment.

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

```{warning}
Any subsequent `mamba` commands in the docs will need to be replaced with `conda` if you choose to use your existing Anaconda installation.
```

````

## Installation methods

SLEAP can be installed three different ways: via {ref}`conda package<condapackage>`, {ref}`conda from source<condasource>`, or {ref}`pip package<pippackage>`. Select one of the methods below to install SLEAP. We recommend {ref}`conda package<condapackage>`.



`````{hint}
   Installation requires entering commands in a terminal. To open one:

   ````{tabs}

      ```{tab} Windows

         Open the *Start menu* and search for the *Anaconda Prompt* (if using Miniconda) or the *Command Prompt* if not.
         
         ```{note}
         On Windows, our personal preference is to use alternative terminal apps like [Cmder](https://cmder.net) or [Windows Terminal](https://aka.ms/terminal).
         ```

      ```

      ```{tab} Linux

         Launch a new terminal by pressing <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>.

      ```

      ```{group-tab} Mac OS

         Launch a new terminal by pressing <kbd>Cmd</kbd> + <kbd>Space</kbd> and searching for _Terminal_.

      ```

   ````

`````

````{tabs}

   ```{tab} conda package

      **This is the recommended installation method**.

      ````{tabs}

         ```{group-tab} Windows and Linux


            ```bash
            mamba create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.4.1a1
            ```

            
            ```{note}
            - This comes with CUDA to enable GPU support. All you need is to have an NVIDIA GPU and [updated drivers](https://nvidia.com/drivers).
            - If you already have CUDA installed on your system, this will not conflict with it.
            - This will also work in CPU mode if you don't have a GPU on your machine.
            ```

         ```

         ```{group-tab} Mac OS

            ```bash
            mamba create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.4.1a1
            ```

            ```{note}
            This will also work in CPU mode if you don't have a GPU on your machine.
            ```
         
         ```

      ````

   ```

   ```{tab} conda from source

      This is the **recommended method for development**.

      1. First, ensure git is installed:

         ```bash
         git --version
         ```

         If `git` is not recognized, then [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

      2. Then, clone the repository:

         ```bash
         git clone https://github.com/talmolab/sleap && cd sleap
         ```

      3. Finally, install SLEAP from the environment file:


         ````{tabs}

            ```{group-tab} Windows and Linux

            
               ````{tabs}

                  ```{group-tab} NVIDIA GPU

                     ```bash
                     mamba env create -f environment.yml -n sleap
                     ```

                  ```

                  ```{group-tab} CPU or other GPU

                     ```bash
                     mamba env create -f environment_no_cuda.yml -n sleap
                     ```
                  
                  ```

               ````

            ```

            ```{group-tab} Mac OS

               ```bash
               mamba env create -f environment_mac.yml -n sleap
               ```
            ```

         ````

      ```{note}
      - This installs SLEAP in development mode, which means that edits to the source code will be applied the next time you run SLEAP.
      - Change the `-n sleap` in the command to create an environment with a different name (e.g., `-n sleap_develop`).
      ```
   
   ```

   ```{tab} pip package

      This is the **recommended method for Google Colab only**.

      ```{warning}
      This will uninstall existing libraries and potentially install conflicting ones.

      We strongly recommend that you **only use this method if you know what you're doing**!
      ```

      ````{tabs}

         ```{group-tab} Windows and Linux

            ```{note}
            - Requires Python 3.7
            - To enable GPU support, make sure that you have **CUDA Toolkit v11.3** and **cuDNN v8.2** installed.
            ```

            Although you do not need Miniconda installed to perform a `pip install`, we recommend [installing Miniconda](https://docs.anaconda.com/free/miniconda/) to create a new environment where we can isolate the `pip install`. Alternatively, you can use a venv if you have an existing Python 3.7 installation. If you are working on **Google Colab**, skip to step 3 to perform the `pip install` without using a conda environment.

            1. Otherwise, create a new conda environment where we will `pip install sleap`:

               ````{tabs}

                  ```{group-tab} NVIDIA GPU


                     ```bash
                     mamba create --name sleap pip python=3.7.12 cudatoolkit=11.3 cudnn=8.2
                     ```

                  ```

                  ```{group-tab} CPU or other GPU

                     ```bash
                     mamba create --name sleap pip python=3.7.12
                     ```

                     
                  ```

               ````

            2. Then activate the environment to isolate the `pip install` from other environments on your computer:

               ```bash
               mamba activate sleap
               ```

               ```{warning}
               Refrain from installing anything into the `base` environment. Always create a new environment to install new packages.
               ```

            3. Finally, we can perform the `pip install`:

               ```bash
               pip install sleap[pypi]==1.4.1a1
               ```

               ```{note}
               The pypi distributed package of SLEAP ships with the following extras:
               - **pypi**: For installation without an mamba environment file. All dependencies come from PyPI.
               - **jupyter**: This installs all *pypi* and jupyter lab dependencies.
               - **dev**: This installs all *jupyter* dependencies and developement tools for testing and building docs.
               - **conda_jupyter**: For installation using a mamba environment file included in the source code. Most dependencies are listed as conda packages in the environment file and only a few come from PyPI to allow jupyter lab support.
               - **conda_dev**: For installation using [a mamba environment](https://github.com/search?q=repo%3Atalmolab%2Fsleap+path%3Aenvironment*.yml&type=code) with a few PyPI dependencies for development tools.
               ```
            
         ```

         ```{group-tab} Mac OS

            Not supported.

         ```

         ````
   
   ```

````

## Testing that things are working

If you installed using `mamba`, first activate the `sleap` environment by opening a terminal and typing:

```bash
mamba activate sleap
```

````{hint}
Not sure what `mamba` environments you already installed? You can get a list of the environments on your system with:
```
mamba env list
```
````

### GUI support

To check that the GUI is working, simply type:

```bash
sleap-label
```

You should see the SLEAP labeling interface pop up within a few moments.

### Importing

To check if SLEAP is installed correctly in non-interactive environments, such as remote servers, confirm that you can import it with:

```bash
python -c "import sleap; sleap.versions()"
```

<small>**Output:**</small>

    (sleap_develop) λ python -c "import sleap; sleap.versions()"
    SLEAP: 1.2.0
    TensorFlow: 2.7.1
    Numpy: 1.21.5
    Python: 3.7.11
    OS: Windows-10-10.0.19041-SP0

### GPU support

Assuming you installed using either of the `mamba`-based methods on Windows or Linux, SLEAP should automatically have GPU support enabled.

To check, verify that SLEAP can detect the GPUs on your system:

```bash
python -c "import sleap; sleap.system_summary()"
```

<small>**Output:**</small>

    (sleap_develop) λ python -c "import sleap; sleap.system_summary()"
    GPUs: 2/2 available
      Device: /physical_device:GPU:0
             Available: True
            Initalized: False
         Memory growth: None
      Device: /physical_device:GPU:1
             Available: True
            Initalized: False
         Memory growth: None

SLEAP uses TensorFlow for GPU acceleration. To directly check if TensorFlow is detecting your GPUs:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

<small>**Output:**</small>

    (sleap_develop) λ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]

```{note}
- GPU support requires an NVIDIA GPU.
- If you haven't yet (or in a while), update to the [latest NVIDIA drivers for your GPU](https://nvidia.com/drivers).
- We use the official conda packages for [cudatoolkit](https://anaconda.org/anaconda/cudatoolkit) and [cudnn](https://anaconda.org/anaconda/cudnn), so no external installations are required. If you already have those installed on your system, they should not interfere with the ones in the SLEAP environment.
- TensorFlow 2.6-2.8 are compatible with **CUDA Toolkit v11.3** and **cuDNN v8.2**.
```

````{warning}
TensorFlow 2.7+ is currently failing to detect CUDA Toolkit and CuDNN on some systems (see [Issue thread](https://github.com/tensorflow/tensorflow/issues/52988)).

If you run into issues, either try downgrading the TensorFlow 2.6:
```bash
pip install tensorflow==2.6.3
```
or follow the note below.
````

````{note}
If you are on Linux, have a NVIDIA GPU, but cannot detect your GPU:

```bash
W tensorflow/stream_executor/platform/default/dso_loader.cc:64 Could not load dynamic
library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object
file: No such file or directory
```

then activate the environment:

```bash
mamba activate sleap
```

and run the commands:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo '#!/bin/sh' >> $CONDA_PREFIX/etc/conda/activate.d/sleap_activate.sh
echo 'export SLEAP_OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/sleap_activate.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/sleap_activate.sh
source $CONDA_PREFIX/etc/conda/activate.d/sleap_activate.sh
```

This will set the environment variable `LD_LIBRARY_PATH` each time the environment is activated. The environment variable will remain set in the current terminal even if we deactivate the environment. Although not strictly necessary, if you would also like the environment variable to be reset to the original value when deactivating the environment, we can run the following commands:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo '#!/bin/sh' >> $CONDA_PREFIX/etc/conda/deactivate.d/sleap_deactivate.sh
echo 'export LD_LIBRARY_PATH=$SLEAP_OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/sleap_deactivate.sh
```

These commands only need to be run once and will subsequently run automatically upon [de]activating your `sleap` environment.
````

## Upgrading and uninstalling

We **strongly recommend** installing SLEAP in a fresh environment when updating. This is because dependency versions might change, and depending on the state of your previous environment, directly updating might break compatibility with some of them.

To uninstall an existing environment named `sleap`:

```bash
mamba env remove -n sleap
```

````{hint}
Not sure what `mamba` environments you already installed? You can get a list of the environments on your system with:
```bash
mamba env list
```
````

Once the environment has been removed, you are free to install SLEAP using any of the installation methods above into an environment of the same name.

## Getting help

If you run into any problems, check out the [Github Discussions](https://github.com/talmolab/sleap/discussions) and [GitHub Issues](https://github.com/talmolab/sleap/issues) to see if others have had the same problem.

If you get any errors or the GUI fails to launch, try running the diagnostics to see what SLEAP is able to detect on your system:

```bash
sleap-diagnostic
```

If you were not able to get SLEAP installed, activate the mamba environment it is in and generate a list of the package versions installed:

```bash
mamba list
```

Then, [open a new Issue](https://github.com/talmolab/sleap/issues) providing the versions from either command above, as well as any errors you saw in the console during the installation. Or [start a discussion](https://github.com/talmolab/sleap/discussions) to get help from the community.
