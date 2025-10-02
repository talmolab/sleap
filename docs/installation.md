# Installation

!!! warning "Documentation for New SLEAP Versions"
    This documentation is for the **latest version of SLEAP**.  
    If you are using **SLEAP version 1.4.1 or earlier**, please visit the [legacy documentation](http://legacy.sleap.ai).

SLEAP can be installed as a Python package on Windows, Linux, and Mac OS. The newest version of SLEAP can always be found in the [Releases page](https://github.com/talmolab/sleap/releases).

!!! note "GPU Support"
    SLEAP offers GPU-accelerated training on **NVIDIA GPUs** (Windows/Linux) and **Apple Silicon Macs** (macOS). Other GPU types (such as AMD GPUs or Intel graphics) are **not supported** for training and may result in errors or unexpected behavior. If you do not have a supported GPU, SLEAP will automatically use CPU mode—this works for all features, but training will be significantly slower.

    For more details on GPU, see the [GPU Support section](#gpu-support).

!!! info "SLEAP GUI and Neural Network Backend"
    The SLEAP GUI for labeling can be installed and used on its own. For training and inference with neural networks, SLEAP uses the **sleap-nn** backend, which integrates seamlessly with the GUI (use `nn` extra dependency to train/ run inference).

    To learn more about sleap-nn and its capabilities, visit the [sleap-nn repository](https://github.com/talmolab/sleap-nn).

---

## Installation methods

**Prerequisites:** Python 3.11+ (required for all installation methods)

!!! tip "Choose Your Installation Method"
    - **[Installation as a system-wide tool with uv](#installation-with-uv-tool-install)**: Use `uv tool install` to install SLEAP globally as a tool (Installation needed, **strongly recommended**)
    - **[Installation with uvx](#installation-with-uvx)**: Use `uvx` for one-off commands. (no installation needed!)
    - **[Installation with uv pip](#installation-with-uv-pip)**: Use `uv pip` to install from pypi in a uv virtual env.
    - **[Installation with pip](#installation-with-pip)**: Use `pip` to install from pypi in a conda env.
    - **[Installation from source](#installation-from-source)**: Use `uv sync` to install from source. (For developmental purposes)

**How to open a terminal**

To install SLEAP, you'll need to enter commands in a terminal. Here's how to open one on your system:

=== "Windows"
    - Open the **Start menu** and search for **Command Prompt**.
    - *Tip:* You may prefer alternative terminal apps like [Cmder](https://cmder.app) or [Windows Terminal](https://aka.ms/terminal).

=== "Linux"
    - Press <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd> to launch a new terminal window.

=== "macOS"
    - Press <kbd>Cmd</kbd> + <kbd>Space</kbd>, type **Terminal**, and press <kbd>Enter</kbd> to open it.

--- 

## Installation with uv tool install

`uv tool install` installs SLEAP globally as a system-wide CLI tool, making all SLEAP CLI commands available from anywhere in your terminal.

!!! note "Install uv"
    Install [`uv`](https://github.com/astral-sh/uv) - a fast Python package manager:
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
    uv tool install "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--index` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).

=== "Windows/Linux (CPU)"
    ```bash
    uv tool install "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv tool install "sleap[nn]"
    ```
    !!! info "Mac MPS support"
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "SLEAP GUI Only"
    ```bash
    uv tool install "sleap"
    ```
    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.


!!! tip "How uv tool install Works"
    - **Automatic Installation**: Downloads and installs SLEAP with dependencies
    - **No Conflicts**: Won't interfere with your existing Python packages/dependencies
    - **Uses Recent Packages**: Uses the latest version from PyPI

### Verify Installation
```bash
# Test the installation
sleap-label --help
```

---
## Installation with uvx
`uvx` automatically installs sleap and runs your command inside a temporary virtual environment (venv). This means each run is fully isolated and leaves no trace on your system— perfect for trying out SLEAP without any **permanent** installation.

!!! note "Install uv"
    Install [`uv`](https://docs.astral.sh/uv/)- an ultra-fast Python package manager:
    ```bash
    # macOS/Linux
    curl -LsSf http://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Platform-Specific Commands

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uvx --from "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap-label
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--index` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).

=== "Windows/Linux (CPU)"
    ```bash
    uvx --from "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple sleap-label
    ```

=== "macOS"
    ```bash
    uvx --from "sleap[nn]" sleap-label
    ```
    !!! info "Mac MPS support"
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "SLEAP GUI Only"
    ```bash
    uvx --from "sleap" sleap-label
    ```
    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.

!!! note "uvx Installation"
    Because `uvx` installs packages fresh on every run, it's ideal for quick one-off tests. For regular use, you could install with [`uv tool install`](#installation-with-uv-tool-install) or setting up a development environment with [`uv sync`](#installation-from-source) to avoid repeated downloads.

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
    uv pip install "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--index` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).

=== "Windows/Linux (CPU)"
    ```bash
    # CUDA 12.8
    uv pip install "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv pip install "sleap[nn]"
    ```
    !!! info "Mac MPS support"
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "SLEAP GUI Only"
    ```bash
    uv pip install "sleap"
    ```
    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.

!!! info "Running With `uv run`"
    `uv pip install` creates a `.venv` (virtual environment) inside your current working directory. To use all installed packages, <u>**you must run commands with `uv run`**</u> (e.g., `uv run sleap-label ...` or `uv run pytest ...`) with these installation methods.

### Verify Installation
```bash
# Test the installation
uv run sleap-label --help
```

!!! warning "SLEAP not recognized after installation?"
    If running the verification step above gives an error like `command not found` or `not recognized as an internal or external command`, try the following workarounds:

    - Activate your virtual environment (the venv name should be the same as your current working dir name). If you used `uv`, activate it and then run:
    ```bash
    uv run --active sleap-label --help
    ```
    This ensures the command runs in the correct environment.
    - Another workaround **(not recommended)**:
      Check if you have any *empty* `pyproject.toml` or `uv.lock` files in `Users/<your-user-name>`. If you find empty files with these names, delete them, and try again (Empty files here can sometimes interfere with uv's environment resolution.)


---
## Installation with pip

We recommend creating a dedicated environment with [conda](https://docs.conda.io/en/latest/miniconda.html) or [mamba/miniforge](https://github.com/conda-forge/miniforge) before installing `sleap` with pip. This helps avoid dependency conflicts and keeps your Python setup clean. After installing Miniconda or Miniforge, create and activate an environment, then run the pip install commands below inside the activated environment.

To create a conda environment, run:
```bash
conda create -n sleap python=3.12
conda activate sleap
```

### Platform-Specific Commands

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    pip install "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cu128 --index-url https://pypi.org/simple
    ```
    !!! info "Other CUDA versions"
        - For more information on which CUDA version to use for your system, see the [PyTorch installation](https://pytorch.org/get-started/locally/) guide. The `--extra-index-url` in the install command should match the CUDA version you need (e.g., `https://download.pytorch.org/whl/cuda118` for CUDA 11.8, `https://download.pytorch.org/whl/cuda128` for CUDA 12.8, etc.).

=== "Windows/Linux (CPU)"
    ```bash
    pip install "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cpu --index-url https://pypi.org/simple
    ```

=== "macOS Only"
    ```bash
    pip install "sleap[nn]"
    ```
    !!! info "Mac MPS support"
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "SLEAP GUI Only"
    ```bash
    pip install sleap
    ```

    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.

### Verify Installation
```bash
sleap-label --help
```

--- 

## Installation from source

For contributing to SLEAP or development workflows.

!!! info "Running With `uv sync`"
    `uv sync` creates a `.venv` (virtual environment) inside your current working directory. This environment is only active within that directory and can't be directly accessed from outside. To use all installed packages, <u>**you must run commands with `uv run`**</u> (e.g., `uv run sleap-label ...` or `uv run pytest ...`) with these installation methods.

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

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv sync --extra dev --extra nn-cuda128 --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple

    # CUDA 11.8
    uv sync --extra dev --extra nn-cuda118 --index https://download.pytorch.org/whl/cu118 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv sync --extra dev --extra nn-cpu --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv sync --extra dev --extra nn-cpu --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

    !!! info "Mac MPS support"
        - On macOS, MPS (Metal Performance Shaders) is automatically enabled for Apple Silicon acceleration.

=== "GUI Only"
    ```bash
    uv sync --extra dev 
    ```

    !!! warning "GUI <u>ONLY</u>"
        Installing this version of SLEAP will **NOT** include any training/inference capabilities, as it will not include the sleap-nn backend. This should primarily be used for **labeling**.

!!! note "SLEAP `uv sync` Extras"
    The `uv sync` comes with the following **extras** (for local builds):

    - **nn-cpu**: Installs `sleap-nn` with the default torch-cpu backend.
    - **nn-cuda118**: Installs `sleap-nn` with the torch CUDA 11.8.
    - **nn-cuda128**: Installs `sleap-nn` with the torch CUDA 12.8.
    - **dev**: Installs all development tools for testing.
    - **docs**: Installs all documentation-related dependencies (e.g., mkdocs).
    - **jupyter**: Installs all Jupyter and JupyterLab dependencies.


### Verify Installation

```bash
# Run tests
uv run pytest tests

# Check code formatting
uv run ruff check sleap tests

# Run CLI command
uv run sleap-label
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

!!! note "Using `uv run`"
    If you installed SLEAP using `uv pip install` or `uv sync`, you must prefix commands with `uv run` to ensure they run inside the correct virtual environment. For example, use:
    ```
    uv run sleap-label
    ```
    instead of just `sleap-label`.

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
# for windows/ linux
python -c "import torch; print(torch.cuda.is_available())"

# for mac
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Upgrading and uninstalling

We **strongly recommend** installing SLEAP in a fresh environment when updating. This is because dependency versions might change, and depending on the state of your previous environment, directly updating might break compatibility with some of them.

To uninstall SLEAP (if installed with `uv tool`):

```bash
uv tool uninstall sleap
```

To uninstall an existing `uv` venv or `conda` environment named `sleap`:

=== "uv venv"
    ```bash
    # Installed with uv
    # Delete `.venv/` and `uv.lock`

    #For eg:
    rm -rf path/to/venv
    rm uv.lock
    ```

=== "conda environment"
    ```bash
    # Installed with pip in a conda env
    conda env remove -n sleap
    ```
    !!! hint "Not sure what `conda` environments you already installed?"
        You can get a list of the environments on your system with:
        ```
        conda env list
        ```

Once the environment has been removed, you are free to install SLEAP using any of the [installation methods](#installation-methods) above into a venv or conda environment of the same name.

## Getting help

If you run into any problems, check out the [Github Discussions](https://github.com/talmolab/sleap/discussions) and [GitHub Issues](https://github.com/talmolab/sleap/issues) to see if others have had the same problem.

If you get any errors or the GUI fails to launch, try running the diagnostics to see what SLEAP is able to detect on your system:

!!! note "Using `uv run`"
    If you installed SLEAP using `uv pip install` or `uv sync`, you must prefix commands with `uv run` to ensure they run inside the correct virtual environment. For example, use:
    ```
    uv run sleap-diagnostic
    ```
    instead of just `sleap-diagnostic`.

```bash
sleap-diagnostic
```

!!! hint "If you were not able to get SLEAP installed:"

    === "Installed w/ uv:"
        Run the following command on the venv (not using `uv tool install`):

        ```bash
        uv pip list 
        ```

    === "Installed w/ pip:"
        Activate the conda environment it is in and generate a list of the package versions installed:

        ```bash
        conda list
        ```

Then, [open a new Issue](https://github.com/talmolab/sleap/issues) providing the versions from either command above, as well as any errors you saw in the console during the installation. Or [start a discussion](https://github.com/talmolab/sleap/discussions) to get help from the community.