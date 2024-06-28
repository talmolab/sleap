# Installation

SLEAP can be installed as a Python package on Windows, Linux, Mac OS X, and Mac OS Apple Silicon.

SLEAP requires many complex dependencies, so we **strongly** recommend using [Mambaforge](https://mamba.readthedocs.io/en/latest/installation.html) to install it in its own isolated environment. See {ref}`Installing Mambaforge<mambaforge>` below for more instructions.

The newest version of SLEAP can always be found in the [Releases page](https://github.com/talmolab/sleap/releases).

```{contents} Contents
---
local:
---
```

````{hint}
Installation requires entering commands in a terminal. To open one:

**Windows:** Open the *Start menu* and search for the *Miniforge Prompt* (if using Mambaforge) or the *Command Prompt* if not.
```{note}
On Windows, our personal preference is to use alternative terminal apps like [Cmder](https://cmder.net) or [Windows Terminal](https://aka.ms/terminal).
```

**Linux:** Launch a new terminal by pressing <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>.

**Mac:** Launch a new terminal by pressing <kbd>Cmd</kbd> + <kbd>Space</kbd> and searching for _Terminal_.

````

(apple-silicon)=

### Macs Pre-M1 (Pre-Installation)

SLEAP can be installed on Macs by following these instructions:

1. Make sure you're on **macOS Monterey** or later, i.e., version 12+.

2. If you don't have it yet, [install **homebrew**](https://brew.sh/), a convenient package manager for Macs (skip this if you can run `brew` from the terminal):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   This might take a little while since it'll also install Xcode (which we'll need later). Once it's finished, your terminal should give you two extra commands to run listed under **Next Steps**.

   ````{note}
   We recommend running the commands given in your terminal which will be similar to (but may differ slightly) from the commands below:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   ```

   ```bash
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

   ````

   Then, close and re-open the terminal for it to take effect.

3. Install wget, a CLI downloading utility (also makes sure your homebrew setup worked):

   ```bash
   brew install wget
   ```

(mambaforge)=

## Installing Mambaforge

**Anaconda** is a Python environment manager that makes it easy to install SLEAP and its necessary dependencies without affecting other Python software on your computer.

[**Mambaforge**](https://mamba.readthedocs.io/en/latest/installation.html) is a lightweight installer of Anaconda with speedy package resolution that we recommend.

````{note}
If you already have Anaconda on your computer, then you can [set the solver to `libmamba`](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) in the `base` environment (and skip the Mambaforge installation):

```bash
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

```{warning}
Any subsequent `mamba` commands in the docs will need to be replaced with `conda` if you choose to use your existing Anaconda installation.
```

````

Otherwise, to install Mamba:

**On Windows**, just click through the installation steps.

1.  Go to: https://github.com/conda-forge/miniforge#mambaforge
2.  Download the latest version for your OS.
3.  Follow the installer instructions.

We recommend using the following settings:

- Install for: All Users (requires admin privileges)
- Destination folder: `C:\mambaforge`
- Advanced Options: Add MambaForge to the system PATH environment variable
- Advanced Options: Register MambaForge as the system Python 3.X
  These will make sure that MambaForge is easily accessible from most places on your computer.

**On Linux**, it might be easier to do this straight from the terminal (<kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>) with this one-liner:

```bash
wget -nc https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && bash Mambaforge-Linux-x86_64.sh -b && ~/mambaforge/bin/conda init bash
```

Restart the terminal after running this command.

```{note}
For other Linux architectures (arm64 and POWER8/9), replace the `.sh` filenames above with the correct installer name for your architecture. See the Download column in [this table](https://github.com/conda-forge/miniforge#mambaforge) for the correct filename.

```

**On Macs (pre-M1)**, you can run the installer using this terminal command:

```bash
wget -nc https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-x86_64.sh && bash Mambaforge-MacOSX-x86_64.sh -b && ~/mambaforge/bin/conda init zsh
```

**On Macs (Apple Silicon)**, use this terminal command:

```bash
curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-arm64.sh -o Mambaforge3-MacOSX-arm64.sh && chmod +x Mambaforge3-MacOSX-arm64.sh && ./Mambaforge3-MacOSX-arm64.sh -b -p ~/mambaforge3 && rm Mambaforge3-MacOSX-arm64.sh && ~/mambaforge3/bin/conda init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
```

## Installation methods

SLEAP can be installed three different ways: via {ref}`conda package<condapackage>`, {ref}`conda from source<condasource>`, or {ref}`pip package<pippackage>`. Select one of the methods below to install SLEAP. We recommend {ref}`conda package<condapackage>`.

(condapackage)=

### `conda` package

**Windows** and **Linux**

```bash
mamba create -y -n sleap -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.4.1a2
```

**Mac OS X** and **Apple Silicon**

```bash
mamba create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.4.1a2
```

**This is the recommended installation method**.

```{note}
- This comes with CUDA to enable GPU support. All you need is to have an NVIDIA GPU and [updated drivers](https://nvidia.com/drivers).
- If you already have CUDA installed on your system, this will not conflict with it.
- This will also work in CPU mode if you don't have a GPU on your machine.
```

(condasource)=

### `conda` from source

1. First, ensure git is installed:

   ```bash
   git --version
   ```

   If 'git' is not recognized, then [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

2. Then, clone the repository:

   ```bash
   git clone https://github.com/talmolab/sleap && cd sleap
   ```

3. Finally, install from the environment file (differs based on OS and GPU):

   **Windows** and **Linux**

   ```bash
   mamba env create -f environment.yml -n sleap
   ```

   If you do not have a NVIDIA GPU, then you should use the no CUDA environment file:

   ```bash
   mamba env create -f environment_no_cuda.yml -n sleap
   ```

   **Mac OS X** and **Apple Silicon**

   ```bash
   mamba env create -f environment_mac.yml -n sleap
   ```

   This is the **recommended method for development**.

```{note}
- This installs SLEAP in development mode, which means that edits to the source code will be applied the next time you run SLEAP.
- Change the `-n sleap` in the command to create an environment with a different name (e.g., `-n sleap_develop`).
```

(pippackage)=

### `pip` package

Although you do not need Mambaforge installed to perform a `pip install`, we recommend {ref}`installing Mambaforge<mambaforge>` to create a new environment where we can isolate the `pip install`. Alternatively, you can use a venv if you have an existing python installation. If you are working on **Google Colab**, skip to step 3 to perform the `pip install` without using a conda environment.

1. Otherwise, create a new conda environment where we will `pip install sleap`:

   either without GPU support:

   ```bash
   mamba create --name sleap pip python=3.7.12
   ```

   or with GPU support:

   ```bash
   mamba create --name sleap pip python=3.7.12 cudatoolkit=11.3 cudnn=8.2
   ```

2. Then activate the environment to isolate the `pip install` from other environments on your computer:

   ```bash
   mamba activate sleap
   ```

   ```{warning}
   Refrain from installing anything into the `base` environment. Always create a new environment to install new packages.
   ```

3. Finally, we can perform the `pip install`:

   ```bash
   pip install sleap[pypi]==1.4.1a2
   ```

   This works on **any OS except Apple silicon** and on **Google Colab**.

   ```{note}
   The pypi distributed package of SLEAP ships with the following extras:
   - **pypi**: For installation without an mamba environment file. All dependencies come from PyPI.
   - **jupyter**: This installs all *pypi* and jupyter lab dependencies.
   - **dev**: This installs all *jupyter* dependencies and developement tools for testing and building docs.
   - **conda_jupyter**: For installation using a mamba environment file included in the source code. Most dependencies are listed as conda packages in the environment file and only a few come from PyPI to allow jupyter lab support.
   - **conda_dev**: For installation using [a mamba environment](https://github.com/search?q=repo%3Atalmolab%2Fsleap+path%3Aenvironment*.yml&type=code) with a few PyPI dependencies for development tools.
   ```

   ```{note}
   - Requires Python 3.7
   - To enable GPU support, make sure that you have **CUDA Toolkit v11.3** and **cuDNN v8.2** installed.
   ```

   ```{warning}
   This will uninstall existing libraries and potentially install conflicting ones.

   We strongly recommend that you **only use this method if you know what you're doing**!
   ```

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
