# Installation

!!! warning "Documentation for New SLEAP Versions"
    This documentation is for the **latest version of SLEAP**.  
    If you are using **SLEAP version 1.4.1 or earlier**, please visit the [legacy documentation](http://legacy.sleap.ai).


SLEAP can be installed as a Python package on Windows, Linux, and Mac OS. For a quick sample using [`uv`](https://docs.astral.sh/uv/)- an ultra-fast Python package and project manager, see below (no installation required!):

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uvx --from "sleap[nn]" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 sleap-label
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--extra-index-url` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "macOS/CPU Only"
    ```bash
    uvx --from "sleap[nn]" sleap-label
    ```

=== "SLEAP GUI Only"
    ```bash
    uvx --from "sleap" sleap-label
    ```
    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.

!!! tip "Sample with `uvx`"
    Note that opening SLEAP w/ `uvx` will **not** install SLEAP onto your system, it will only **'invoke'** SLEAP.

For more in-depth installation instructions, see the [installation methods](#installation-methods). The newest version of SLEAP can always be found in the [Releases page](https://github.com/talmolab/sleap/releases).

!!! note
    SLEAP currently supports GPU-accelerated training on **NVIDIA GPUs** and **Apple Silicon Macs**. Training on other GPU architectures (e.g., AMD GPUs and older Macs) is not supported and may lead to failures or unexpected behavior. If no supported GPU is available, SLEAP can still run in CPU mode, but training speed may be reduced. 
    
    For detailed GPU setup instructions, see [gpu-support](#gpu-support).

!!! note
    For labeling purposes, the SLEAP GUI can be installed on its own. However, **sleap-nn** is a neural network backend available for training and inference w/ the SLEAP GUI. 
    
    For more details on this backend, see [sleap-nn](https://github.com/talmolab/sleap-nn).

---

## Installation methods

**How to open a terminal:**
!!! hint ""
    Installation requires entering commands in a terminal. To open one:

    === "Windows"
        Open the *Start menu* and search for the *Command Prompt*.

        !!! note
            On Windows, our personal preference is to use alternative terminal apps like [Cmder](https://cmder.app) or [Windows Terminal](https://aka.ms/terminal).

    === "Linux"
        Launch a new terminal by pressing <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd>.

    === "Mac OS"
        Launch a new terminal by pressing <kbd>Cmd</kbd> + <kbd>Space</kbd> and searching for _Terminal_.

**Prerequisites:** Python 3.11+ (required for all installation methods)

!!! tip "Choose Your Installation Method"
    - **[Installation with as a system-wide tool with uv](#installation-with-uv-tool-install)**: Use `uv tool install` to install SLEAP globally as a tool (Installation needed, **strongly recommended**)
    - **[Installation with uv pip](#installation-with-uv-pip)**: Use `uv pip` to install from pypi in a uv virtual env.
    - **[Installation with pip](#installation-with-pip)**: Use `pip` to install from pypi in a conda env. (Recommended to use with a conda env)
    - **[Installation from source](#development-setup-with-uv)**: Use `uv sync` to install from source. (For developmental purposes)

--- 

## Installation with uv tool install

`uv tool install` automatically installs and creates the `sleap-label` tool inside a virtual environment located inside the [uv tool directory](https://docs.astral.sh/uv/concepts/tools/#tools-directory). This means that running this global tool will use this installed version.

!!! warning "First Time uv Setup"
    Install [`uv`](https://github.com/astral-sh/uv) first - an ultra-fast Python package manager:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Platform-Specific Commands

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv tool install "sleap[nn]" --index-rul https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118 

    # CUDA 11.8
    uv tool install "sleap[nn]" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--extra-index-url` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "macOS/CPU Only"
    ```bash
    uv tool install "sleap[nn]"
    ```

=== "SLEAP GUI Only"
    ```bash
    uv tool install "sleap"
    ```
    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.


!!! tip "How uv tool install Works"
    - **Automatic Installation**: Downloads and installs SLEAP with dependencies
    - **Isolated Environment**: Each run after installation runs in a clean, separated virtual environment
    - **No Conflicts**: Won't interfere with your existing Python packages/dependencies
    - **Uses Recent Packages**: Uses the latest version from PyPI

### Verify Installation
```bash
# Test the installation
sleap-label --help
```

---

## Installation with uv pip
This method creates a dedicated project environment using uv's modern Python project management. It initializes a new project with `uv init`, creates an isolated virtual environment with `uv venv`, and installs SLEAP using `uv pip`.

!!! note "Install and set-up uv"
    Step-1: Install [`uv`](https://github.com/astral-sh/uv) - an ultra-fast Python package manager:
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    Step-2: Move to your project directory and initialize the virtual env.
    ```bash
    uv init 
    uv venv
    ```

### Platform-Specific Commands

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv pip install "sleap[nn]" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 

    # CUDA 11.8
    uv pip install "sleap[nn]" --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118  
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--extra-index-url` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "macOS/CPU Only"
    ```bash
    uv pip install "sleap[nn]"
    ```

=== "SLEAP GUI Only"
    ```bash
    uv pip install "sleap"
    ```
    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.

!!! info "Running With `uv run`"
    `uv sync` & `uv pip install` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, <u>**you must run commands with `uv run`**</u> (e.g., `uv run sleap-label ...` or `uv run pytest ...`) with these installation methods.

### Verify Installation
```bash
# Test the installation
uv run sleap-label --help
```

!!! warning "SLEAP not recognized after installation?"
    If running the verification step above gives an error like `sleap-nn: command not found` or `'sleap-nn' is not recognized as an internal or external command`, try the following workarounds:

    - Activate your virtual environment (the venv name should be the same as your current working dir name). If you used `uv`, activate it and then run:
    ```bash
    uv run --active sleap-label --help
    ```
    This ensures the command runs in the correct environment.
    - Another workaround (not recommended):
      Check if you have any *empty* `pyproject.toml` or `uv.lock` files in `Users/<your-user-name>`. If you find empty files with these names, delete them, and try again (Empty files here can sometimes interfere with uv's environment resolution.)


---

## Installation with pip

SLEAP requires many complex dependencies, so we **strongly** recommend using a package manager such as [Miniforge](https://github.com/conda-forge/miniforge) or [Miniconda](https://docs.anaconda.com/free/miniconda/) to install SLEAP in its own isolated environment, if not using `uv`.

!!! note
    If you already have Anaconda on your computer (and it is an [older installation](https://conda.org/blog/2023-11-06-conda-23-10-0-release/)), then make sure to [set the solver to `libmamba`](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) in the `base` environment.

    ```bash
    conda update -n base conda
    conda install -n base conda-libmamba-solver
    conda config --set solver libmamba
    ```

    !!! warning
        Any subsequent `conda` commands in the docs will need to be replaced with `mamba` if you have [Mamba](https://mamba.readthedocs.io/en/latest/) installed instead of Anaconda or Miniconda.

If you don't have a `conda` package manager installation, here are some quick install options:

### Miniforge

Miniforge is a minimal installer for conda that includes the `conda` package manager and is maintained by the [conda-forge](https://conda-forge.org) community. The only difference between Miniforge and Miniconda is that Miniforge uses the `conda-forge` channel by default, which provides a much wider selection of community-maintained packages.

=== "Windows"
    Open a new PowerShell terminal (does not need to be admin) and enter:

    ```bash
    Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe"; Start-Process -FilePath "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe" -ArgumentList "/InstallationType=JustMe /RegisterPython=1 /S" -Wait; Remove-Item -Path "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe"
    ```

=== "Linux"
    Open a new terminal and enter:

    ```bash
    curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$HOME/Downloads/Miniforge3-Linux-x86_64.sh" && chmod +x "$HOME/Downloads/Miniforge3-Linux-x86_64.sh" && "$HOME/Downloads/Miniforge3-Linux-x86_64.sh" -b -p "$HOME/miniforge3" && rm "$HOME/Downloads/Miniforge3-Linux-x86_64.sh" && "$HOME/miniforge3/bin/conda" init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
    ```

=== "Mac (Apple Silicon)"
    Open a new terminal and enter:

    ```bash
    curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o "$HOME/Downloads/Miniforge3-MacOSX-arm64.sh" && chmod +x "$HOME/Downloads/Miniforge3-MacOSX-arm64.sh" && "$HOME/Downloads/Miniforge3-MacOSX-arm64.sh" -b -p "$HOME/miniforge3" && rm "$HOME/Downloads/Miniforge3-MacOSX-arm64.sh" && "$HOME/miniforge3/bin/conda" init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
    ```

=== "Mac (Intel)"
    Open a new terminal and enter:

    ```bash
    curl -fsSL --compressed https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -o "$HOME/Downloads/Miniforge3-MacOSX-x86_64.sh" && chmod +x "$HOME/Downloads/Miniforge3-MacOSX-x86_64.sh" && "$HOME/Downloads/Miniforge3-MacOSX-x86_64.sh" -b -p "$HOME/miniforge3" && rm "$HOME/Downloads/Miniforge3-MacOSX-x86_64.sh" && "$HOME/miniforge3/bin/conda" init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
    ```

### Miniconda

This is a minimal installer for conda that includes the `conda` package manager and is maintained by the [Anaconda](https://www.anaconda.com) company.

=== "Windows"
    Open a new PowerShell terminal (does not need to be admin) and enter:

    ```bash
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe; Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait; del miniconda.exe
    ```

=== "Linux"
    Open a new terminal and enter:

    ```bash
    mkdir -p "$HOME/miniconda3" && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda3/miniconda.sh" && bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3" && rm "$HOME/miniconda3/miniconda.sh" && "$HOME/miniconda3/bin/conda" init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
    ```

=== "Mac (Apple Silicon)"
    Open a new terminal and enter:

    ```bash
    curl -fsSL --compressed https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o "$HOME/Downloads/Miniconda3-latest-MacOSX-arm64.sh" && chmod +x "$HOME/Downloads/Miniconda3-latest-MacOSX-arm64.sh" && "$HOME/Downloads/Miniconda3-latest-MacOSX-arm64.sh" -b -u -p "$HOME/miniconda3" && rm "$HOME/Downloads/Miniconda3-latest-MacOSX-arm64.sh" && "$HOME/miniconda3/bin/conda" init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
    ```

=== "Mac (Intel)"
    Open a new terminal and enter:

    ```bash
    curl -fsSL --compressed https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o "$HOME/Downloads/Miniconda3-latest-MacOSX-x86_64.sh" && chmod +x "$HOME/Downloads/Miniconda3-latest-MacOSX-x86_64.sh" && "$HOME/Downloads/Miniconda3-latest-MacOSX-x86_64.sh" -b -u -p "$HOME/miniconda3" && rm "$HOME/Downloads/Miniconda3-latest-MacOSX-x86_64.sh" && "$HOME/miniconda3/bin/conda" init "$(basename "${SHELL}")" && source "$HOME/.$(basename "${SHELL}")rc"
    ```

See the [Miniconda website](https://docs.anaconda.com/free/miniconda/) for up-to-date installation instructions if the above instructions don't work for your system.

SLEAP can be installed with pip via `pip package`. See below.

This is the **recommended method for Google Colab**.

!!! note
    - Requires Python 3.11+

Although you do not need Miniconda installed to perform a `pip install`, we recommend [installing Miniconda](https://docs.anaconda.com/free/miniconda/) to create a new environment where we can isolate the `pip install`. Alternatively, you can use a virtual environment (venv) if you have an existing Python 3.11+ installation. If you are working on **Google Colab**, skip to step 3 to perform the `pip install` without using a conda environment.

!!! note

    1. Otherwise, create a new conda environment where we will `pip install sleap`:

        ```bash
        conda create --name sleap python=3.11
        ```

    2. Then activate the environment to isolate the `pip install` from other environments on your computer:
        ```bash
        conda activate sleap
        ```

        !!! warning
            Refrain from installing anything into the `base` environment. Always create a new environment to install new packages.

### Platform-Specific Installation

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    pip install sleap[nn-gpu] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128

    # CUDA 11.8
    pip install sleap[nn-gpu] --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118
    ```

=== "macOS/CPU Only"
    ```bash
    pip install sleap[nn-cpu]
    ```

=== "SLEAP GUI Only"
    ```bash
    pip install sleap
    ```

!!! note
    The pypi distributed package of SLEAP ships with the following extras:

    - **dev**: This installs all *jupyter* dependencies and developement tools for testing and building docs.
    - **docs**: This installs all *docs*-related dependencies (ex. mkdocs).
    - **nn-cpu**: This installs sleap-nn with torch-cpu.
    - **nn-gpu**: This installs sleap-nn with torch-cuda128.
    - **jupyter**: This installs all *pypi* and jupyter lab dependencies.

### Verify Installation
```bash
sleap-label --help
```

--- 

## Development Setup with uv

For contributing to SLEAP or development workflows.

!!! info "Running With `uv run`"
    `uv sync` & `uv pip install` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, <u>**you must run commands with `uv run`**</u> (e.g., `uv run sleap-label ...` or `uv run pytest ...`) with these installation methods.

**1. Clone the Repository**

```bash
git clone https://github.com/talmolab/sleap.git
cd sleap
```

**2. Install uv (skip if already installed)**

=== "macOS/Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```bash
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

**3. Install Dependencies**

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uv sync --extra dev --extra nn-gpu
    ```

=== "macOS/CPU Only"
    ```bash
    uv sync --extra dev --extra nn-cpu
    ```

=== "GUI Only"
    ```bash
    uv sync --extra dev
    ```

<!-- Additionally, if using CUDA 12.8 w/ `sleap-nn` backend on Windows/Linux (CUDA 12.8):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --reinstall

# Skip if using PyTorch CPU wheel
``` -->

**4. Verify Development Setup**

```bash
# Run tests
uv run pytest tests

# Check code formatting
uv run ruff check sleap tests

# Run CLI command
uv run sleap-label ...
```

---


## Testing that things are working

**If you installed using `conda`**, first activate the `sleap` environment by opening a terminal and typing:

```
conda activate sleap
```
!!! hint "Not sure what `conda` environments you already installed?"
    You can get a list of the environments on your system with:
    ```
    conda env list
    ```

### GUI support

To check that the GUI is working, simply type:

```
sleap-label
```

You should see the SLEAP labeling interface pop up within a few moments.


### Importing

To check if SLEAP is installed correctly in non-interactive environments, such as remote servers, confirm that you can import it with:

```
python -c "import sleap; sleap.versions()"
```
<small>Output:</small>

```
(sleap) λ python -c "import sleap; sleap.versions()"
SLEAP: 1.5.0
Numpy: 2.3.2
Python: 3.12.1
OS: macOS-14.4.1-arm64-arm-64bit
```


### GPU support

Assuming you installed using either `uv` or the `pip`-based methods, SLEAP should automatically have GPU support enabled.

<!-- To check, verify that SLEAP can detect the GPUs on your system:

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
         Memory growth: None -->

SLEAP uses PyTorch for GPU acceleration. To directly check if PyTorch is detecting your GPUs:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

<small>**Output:**</small>

    (sleap) λ python -c "import torch; print(torch.cuda.is_available())"
    False

    # Test run on Apple Silicon architecture

!!! info "macOS MPS Support"
    Even with `[nn-cpu]` when installing SLEAP, macOS automatically enables MPS (Metal Performance Shaders) for Apple Silicon acceleration via [sleap-nn backend](https://nn.sleap.ai/dev/installation/#platform-specific-installation).


<!-- !!! note
    - GPU support requires an NVIDIA GPU.
    - If you haven't yet (or in a while), update to the [latest NVIDIA drivers for your GPU](https://nvidia.com/drivers).
    - We use the official conda packages for [cudatoolkit](https://anaconda.org/anaconda/cudatoolkit) and [cudnn](https://anaconda.org/anaconda/cudnn), so no external installations are required. If you already have those installed on your system, they should not interfere with the ones in the SLEAP environment.
    - TensorFlow 2.6-2.8 are compatible with **CUDA Toolkit v11.3** and **cuDNN v8.2**.

!!! warning
    TensorFlow 2.7+ is currently failing to detect CUDA Toolkit and CuDNN on some systems (see [Issue thread](https://github.com/tensorflow/tensorflow/issues/52988)).

    If you run into issues, either try downgrading to TensorFlow 2.6:
    ```bash
    pip install tensorflow==2.6.3
    ```
    or follow the note below. -->

<!-- !!! note
    If you are on Linux, have a NVIDIA GPU, but cannot detect your GPU:

    ```bash
    W tensorflow/stream_executor/platform/default/dso_loader.cc:64 Could not load dynamic
    library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object
    file: No such file or directory
    ```

    then activate the environment:

    ```bash
    conda activate sleap
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

    These commands only need to be run once and will subsequently run automatically upon [de]activating your `sleap` environment. -->

## Upgrading and uninstalling

We **strongly recommend** installing SLEAP in a fresh environment when updating. This is because dependency versions might change, and depending on the state of your previous environment, directly updating might break compatibility with some of them.

To uninstall an existing `uv` venv or `conda` environment named `sleap`:

=== "uv venv"
    ```bash
    # Installed with uv
    rm -rf path/to/venv
    rm -rf path/to/uv.lock
    ```

=== "conda environment"
    ```bash
    # Installed with pip
    conda env remove -n sleap
    ```
    !!! hint "Not sure what `conda` environments you already installed?"
        You can get a list of the environments on your system with:
        ```
        conda env list
        ```

Once the environment has been removed, you are free to install SLEAP using any of the installation methods above into a venv or conda environment of the same name.

## Getting help

If you run into any problems, check out the [Github Discussions](https://github.com/talmolab/sleap/discussions) and [GitHub Issues](https://github.com/talmolab/sleap/issues) to see if others have had the same problem.

If you get any errors or the GUI fails to launch, try running the diagnostics to see what SLEAP is able to detect on your system:

```bash
sleap-diagnostic
```

!!! hint "If you were not able to get SLEAP installed:"

    === "Installed w/ uv:"
        Run the following command on the venv created by `uv tool install`:

        ```bash
        uv pip list 
        ```

    === "Installed w/ pip:"
        Activate the conda environment it is in and generate a list of the package versions installed:

        ```bash
        conda list
        ```

Then, [open a new Issue](https://github.com/talmolab/sleap/issues) providing the versions from either command above, as well as any errors you saw in the console during the installation. Or [start a discussion](https://github.com/talmolab/sleap/discussions) to get help from the community.