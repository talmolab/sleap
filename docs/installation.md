# Installation

SLEAP can be installed as a Python package on Windows, Linux and Mac OS X. We currently provide {ref}`experimental support for M1 Macs <m1mac>`.

SLEAP requires many complex dependencies, so we **strongly** recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install it in its own isolated environment. See {ref}`Installing Miniconda<miniconda>` below for more instructions.

The newest version of SLEAP can always be found in the [Releases page](https://github.com/murthylab/sleap/releases).

```{contents} Contents
---
local:
---
```

(miniconda)=
## Installing Miniconda
**Anaconda** is a Python environment manager that makes it easy to install SLEAP and its necessary dependencies without affecting other Python software on your computer.

**Miniconda** is a lightweight version of Anaconda that we recommend. To install it:

1. Go to: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links
2. Download the latest version for your OS.
3. Follow the installer instructions.

**On Windows**, just click through the installation steps. We recommend using the following settings:
- Install for: All Users (requires admin privileges)
- Destination folder: `C:\Miniconda3`
- Advanced Options: Add Miniconda3 to the system PATH environment variable
- Advanced Options: Register Miniconda3 as the system Python 3.X
These will make sure that Anaconda is easily accessible from most places on your computer.

**On Linux**, it might be easier to do this straight from the terminal (<kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>) with this one-liner:
```bash
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b && ~/miniconda3/bin/conda init bash
```
Restart the terminal after running this command.

**On Macs**, you can run the graphical installer using the pkg file, or this terminal command:
```bash
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh && bash Miniconda3-latest-MacOSX-x86_64.sh -b && ~/miniconda3/bin/conda init zsh
```

## Installation methods
````{hint}
Installation requires entering commands in a terminal. To open one:

**Windows:** Open the *Start menu* and search for the *Anaconda Command Prompt* (if using Miniconda) or the *Command Prompt* if not.
```{note}
On Windows, our personal preference is to use alternative terminal apps like [Cmder](https://cmder.net) or [Windows Terminal](https://aka.ms/terminal).
```

**Linux:** Launch a new terminal by pressing <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>.

**Mac:** Launch a new terminal by pressing <kbd>Cmd</kbd> + <kbd>Space</kbd> and searching for *Terminal*.
````
### `conda` package
```bash
conda create -y -n sleap -c sleap -c nvidia -c conda-forge sleap=1.2.2
```
**This is the recommended installation method**. Works on **Windows** and **Linux**.

```{note}
- This comes with CUDA to enable GPU support. All you need is to have an NVIDIA GPU and [updated drivers](https://nvidia.com/drivers).
- If you already have CUDA installed on your system, this will not conflict with it.
- This will also work in CPU mode if you don't have a GPU on your machine.
```

### `conda` from source
1. First, clone the repository:
   ```bash
   git clone https://github.com/murthylab/sleap && cd sleap
   ```

2. Then, install from the environment file:
   ```bash
   conda env create -f environment.yml -n sleap
   ```
This works on **Windows**, **Linux** and **Mac OS X** (pre-M1). This is the **recommended method for development**.

```{note}
- This install SLEAP in development mode, which means that edits to the source code will be applied the next time you run SLEAP.
- Change the `-n sleap` in the command to create an environment with a different name (e.g., `-n sleap_develop`).
```

### `pip` package
```bash
pip install sleap==1.2.2
```
This works on **any OS** and on **Google Colab**.
```{note}
- Requires Python 3.7 or 3.8.
- To enable GPU support, make sure that you have **CUDA Toolkit v11.3** and **cuDNN v8.2** installed.
```
```{warning}
This will uninstall existing libraries and potentially install conflicting ones.

We strongly recommend that you **only use this method if you know what you're doing**!
```

(m1mac)=
### M1 Macs
SLEAP can be installed on newer M1 Macs by following these instructions:

1. In addition to being on an M1 Mac, make sure you're on **macOS Monterey**, i.e., version 12+. I tested this on a MacBook Pro (14-inch, 2021) running macOS version 12.0.1.

2. If you don't have it yet, install **homebrew**, a convenient package manager for Macs (skip this if you can run `brew` from the terminal):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   This might take a little while since it'll also install Xcode (which we'll need later). Once it's finished, run this to enable the `brew` command in your shell, then close and re-open the terminal for it to take effect:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile && eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

3. Install wget, a CLI downloading utility (also makes sure your homebrew setup worked):
   ```bash
   brew install wget
   ```

4. Install the **M1 Mac version of Miniconda** -- this is important, so make sure you don't have the regular Mac version! If you're not sure, type `which conda` and delete the containing directory to uninstall your existing conda. To install the correct Miniconda, just run:
   ```bash
   wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh && bash Miniconda3-latest-MacOSX-arm64.sh -b && rm Miniconda3-latest-MacOSX-arm64.sh && ~/miniconda3/bin/conda init zsh
   ```
   Then close and re-open the terminal again.

5. **Download the SLEAP M1 branch**:
   ```bash
   cd ~ && git clone -b talmo/m1 --single-branch https://github.com/murthylab/sleap.git sleap_m1 && cd sleap_m1
   ```
   *Note:* This installs SLEAP in development mode, so changes to the source code are immediately applied in case you wanted to mess around with it. You can also just do a `git pull` to update it (no need to re-do any of the previous steps).

6. **Install SLEAP in a conda environment**:
   ```bash
   conda env create -f environment_m1.yml
   ```
   Your Mac will then automatically sign a devil's pact with Apple to install the correct versions of everything on your system. Once the blood sacrifice/installation process completes, SLEAP will be available in an environment called `sleap_m1`.

7. **Test it out** by activating the environment and opening the GUI!
   ```bash
   conda activate sleap_m1 && sleap-label
   ```

See [this Issue](https://github.com/murthylab/sleap/issues/579#issuecomment-1028602327) for more information on M1 support.


## Testing that things are working
If you installed using `conda`, first activate the `sleap` environment by opening a terminal and typing: 
```bash
conda activate sleap
```
````{hint}
Not sure what `conda` environments you already installed? You can get a list of the environments on your system with:
```
conda env list
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
Assuming you installed using either of the `conda`-based methods on Windows or Linux, SLEAP should automatically have GPU support enabled.

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

If you run into issues, try downgrading the TensorFlow 2.6:
```bash
pip install tensorflow==2.6.3
```
````

## Upgrading and uninstalling
We **strongly recommend** installing SLEAP in a fresh environment when updating. This is because dependency versions might change, and depending on the state of your previous environment, directly updating might break compatibility with some of them.

To uninstall an existing environment named `sleap`:
```bash
conda env remove -n sleap
```

````{hint}
Not sure what `conda` environments you already installed? You can get a list of the environments on your system with:
```bash
conda env list
```
````

Once the environment has been removed, you are free to install SLEAP using any of the installation methods above into an environment of the same name.


## Getting help
If you run into any problems, check out the [GitHub Issues](https://github.com/murthylab/sleap/issues) to see if others have had the same problem.

If you get any errors or the GUI fails to launch, try running the diagnostics to see what SLEAP is able to detect on your system:
```bash
sleap-diagnostic
```

If you were not able to get SLEAP installed, activate the conda environment it is in and generate a list of the package versions installed:
```bash
conda list
```

Then, [open a new Issue](https://github.com/murthylab/sleap/issues) providing the versions from either command above, as well as any errors you saw in the console during the installation.
