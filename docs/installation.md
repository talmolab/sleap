# Installation

SLEAP can be installed as a Python package on Windows, Linux and Mac OS X. We currently provide {ref}`experimental support for M1 Macs <m1mac>`.

SLEAP requires many complex dependencies, so we **strongly** recommend using [Miniconda](https://https://docs.conda.io/en/latest/miniconda.html) to install it in its own isolated environment. See {ref}`Installing Miniconda<miniconda>` below for more instructions.

## Installation methods
Installing SLEAP requires entering commands in a terminal.

**On Windows**, open the **Start menu** and search for the **Anaconda Command Prompt** (if using Miniconda) or the **Command Prompt** if not.

**On Linux**, launch a new terminal by pressing <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>.

**On Macs**, launch a new terminal by pressing <kbd>Cmd</kbd> + <kbd>Space</kbd> and searching for **Terminal**.

### `conda` package
```bash
conda create -y -n sleap -c sleap -c nvidia -c conda-forge sleap=1.2.0
```
- **This is the recommended installation method**.
- Works on **Windows** and **Linux**.
- This comes with CUDA to enable GPU support. All you need is to have an NVIDIA GPU and [updated drivers](https://nvidia.com/drivers).
- If you already have CUDA installed on your system, this will not conflict with it.
- This will also work in CPU mode if you don't have a GPU on your machine.
- Activate the `sleap` environment with `conda activate sleap`. Change the `-n sleap` in the command to create an environment with a different name (e.g., `-n sleap_develop`).

### `conda` from source
**1.** clone the repository:
```bash
git clone https://github.com/talmolab/sleap && cd sleap
```
**2.** Install from the environment file:
```bash
conda env create -f environment.yml -n sleap
```
- This works on **Windows**, **Linux** and **Mac OS X** (pre-M1).
- Activate the `sleap` environment with `conda activate sleap`. Change the `-n sleap` in the command to create an environment with a different name (e.g., `-n sleap_develop`).
- This install SLEAP in development mode, which means that edits to the source code will be applied the next time you run SLEAP.

### `pip` package
```bash
pip install sleap==1.2.0
```
- This works on any OS and on Google Colab.
- We strongly recommend that you **only use this method if you know what you're doing**!
- To enable GPU support, make sure that you have **CUDA Toolkit v11.3** and **cuDNN v8.2** installed.

## Testing and running SLEAP
To make sure SLEAP was installed correctly, first activate the environment (if using conda):
```bash
conda activate sleap
```

Then launch the labeling GUI with:
```bash
sleap-label
```

If you get any errors or the GUI fails to launch, try running the diagnostics to see what was detected on your system:
```bash
sleap-diagnostic
```

If you run into any problems, check out the [GitHub Issues](https://github.com/talmolab/sleap/issues) to see if others have had the same problem or open a new issue.

(miniconda)=
## Installing Miniconda
**Anaconda** is a Python environment manager that makes it easy to install SLEAP and its necessary dependencies without affecting other Python software on your computer.

**Miniconda** is a lightweight version of Anaconda that we recommend. To install it:

1. Go to: https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links
2. Download the latest version for your OS.
3. Follow the installer instructions.

**On Windows**, just click through the installation steps. We recommend using the following settings:
- Install for: All Users (requires admin privileges)
- Destination folder: `C:/Miniconda3`
- Advanced Options: Add Miniconda3 to the system PATH environment variable
- Advanced Options: Register Miniconda3 as the system Python 3.9
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

(m1mac)=
## M1 Macs
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
cd ~ && git clone -b talmo/m1 --single-branch https://github.com/talmolab/sleap.git sleap_m1 && cd sleap_m1
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

See [this Issue](https://github.com/talmolab/sleap/issues/579#issuecomment-1028602327) for more information.
