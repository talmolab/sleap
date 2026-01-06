# Installation

!!! warning "Documentation for New SLEAP Versions"
    This documentation is for the **latest version of SLEAP**. If you are using **SLEAP version 1.4.1 or earlier**, please visit the [legacy documentation](https://legacy.sleap.ai).

SLEAP can be installed as a Python package on Windows, Linux, and Mac OS. The newest version of SLEAP can always be found in the [Releases page](https://github.com/talmolab/sleap/releases).

**Installation methods:**

- **[Installation as a system-wide tool with uv](#installation-with-uv-tool-install)**: Use `uv tool install` to install SLEAP globally as a tool (**strongly recommended**)
- **[Installation with conda/pip](#installation-with-conda/pip)**: Use `pip` to install from pypi in a conda env.
- **[Installation from source](#installation-from-source)**: Use `uv sync` to install from source. (For developmental purposes)

If you need to import SLEAP as a library in your own scripts, add custom packages for analysis, or share reproducible environments with collaborators, see **[Installation with uv add](installation-uv-add.md)**.

Each installation method above supports two modes:

- **GUI-only**: Install the base package (`sleap`) for labeling and data management
- **With training/inference**: Add the `nn` extra (`sleap[nn]`) to enable training and inference using the [**sleap-nn**](https://github.com/talmolab/sleap-nn) backend


!!! tip "How to open a terminal"
    To install SLEAP, you'll need to enter commands in a terminal. Here's how to open one on your system:

    === "Windows"
        - Open the **Start menu** and search for **Command Prompt**.
        - *Tip:* You may prefer alternative terminal apps like [Cmder](https://cmder.app) or [Windows Terminal](https://aka.ms/terminal).

    === "Linux"
        - Press <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>T</kbd> to launch a new terminal window.

    === "macOS"
        - Press <kbd>Cmd</kbd> + <kbd>Space</kbd>, type **Terminal**, and press <kbd>Enter</kbd> to open it.

---

## Prerequisites

**Python Version Requirements:** Python 3.11, 3.12, or 3.13 is required for all installation methods.

!!! warning "Python 3.14 is not yet supported"
    SLEAP currently supports **Python 3.11, 3.12, and 3.13**. **Python 3.14 is not yet tested or supported.** If you have Python 3.14 installed, you must specify `--python 3.13` in your install commands.

**Install uv (for uv-based methods):** If you plan to use `uv tool install`, `uv add`, or install from source, you'll need to install [`uv`](https://github.com/astral-sh/uv) first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

--- 

## Installation with uv tool install

`uv tool install` installs SLEAP globally as a system-wide CLI tool, making all SLEAP CLI commands available from anywhere in your terminal.

### Platform-Specific Commands

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv tool install --python 3.13 "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv tool install --python 3.13 "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv tool install --python 3.13 "sleap[nn]"
    ```

=== "SLEAP GUI Only"
    ```bash
    uv tool install --python 3.13 "sleap"
    ```


!!! tip "About uv tool install"
    This method automatically downloads SLEAP with all dependencies, won't interfere with your existing Python packages, and always uses the latest version from PyPI.

    **Limitation:** Unlike conda or virtual environments, this does not create an activatable environment. You cannot `import sleap` in Python scripts or access dependencies directly—only CLI commands (e.g., `sleap-label`, `sleap-track`) are available. If you need SLEAP as a library, use [uv add](installation-uv-add.md) or [conda/pip](#installation-with-conda/pip) instead.

### Verify Installation
```bash
# Test the installation
sleap-label --help
```

### Updating Dependencies

To update SLEAP and its dependencies (including `sleap-nn` and `sleap-io`):

```bash
# Update SLEAP to the latest version
uv tool upgrade sleap
```

!!! note "Version Constraints"
    `uv` respects any version constraints specified during installation. If you need to upgrade to a specific version or change platform-specific extras (e.g., switching CUDA versions), you may need to uninstall (`uv tool uninstall sleap`) and reinstall with the desired configuration.

---

## Installation with conda/pip

!!! note "No conda package available"
    Starting with SLEAP 1.5, we only distribute SLEAP via pip (PyPI). There is no `conda install sleap` package. However, we recommend using conda/mamba to manage your Python environment before installing SLEAP with pip.

We recommend creating a dedicated environment with [conda](https://docs.conda.io/en/latest/miniconda.html) or [mamba/miniforge](https://github.com/conda-forge/miniforge) before installing `sleap` with pip. This helps avoid dependency conflicts and keeps your Python setup clean. After installing Miniconda or Miniforge, create and activate an environment, then run the pip install commands below inside the activated environment.

To create a conda environment, run:
```bash
conda create -n sleap python=3.13
conda activate sleap
```

### Platform-Specific Commands

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    pip install "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cu128 --index-url https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    pip install "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cpu --index-url https://pypi.org/simple
    ```

=== "macOS Only"
    ```bash
    pip install "sleap[nn]"
    ```

=== "SLEAP GUI Only"
    ```bash
    pip install sleap
    ```

### Verify Installation
```bash
sleap-label --help
```

### Updating Dependencies

To update SLEAP and its dependencies in your conda environment:

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    pip install --upgrade "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cu128 --index-url https://pypi.org/simple
    ```

=== "Windows/Linux (CUDA 11.8)"
    ```bash
    pip install --upgrade "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cu118 --index-url https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    pip install --upgrade "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cpu --index-url https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    pip install --upgrade "sleap[nn]"
    ```

!!! tip "Updating Specific Dependencies"
    To update only a specific dependency like `sleap-nn` or `sleap-io`:
    ```bash
    pip install --upgrade sleap-nn
    pip install --upgrade sleap-io
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

**2. Install Dependencies**

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv sync --extra nn-cuda128 --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple

    # CUDA 11.8
    uv sync --extra nn-cuda118 --index https://download.pytorch.org/whl/cu118 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv sync --extra nn-cpu --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv sync --extra nn-cpu --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "GUI Only"
    ```bash
    uv sync
    ```

!!! note "SLEAP `uv sync` Extras and Dependency Groups"
    `uv sync` automatically installs the **dev** dependency group (pytest, ruff, etc.) for development.

    The following **extras** are also available:

    - **nn-cpu**: Installs `sleap-nn` with the default torch-cpu backend.
    - **nn-cuda118**: Installs `sleap-nn` with the torch CUDA 11.8.
    - **nn-cuda128**: Installs `sleap-nn` with the torch CUDA 12.8.
    - **docs**: Installs all documentation-related dependencies (e.g., mkdocs).
    - **jupyter**: Installs all Jupyter and JupyterLab dependencies.

!!! tip "Upgrading All Dependencies"
    To ensure you have the latest versions of all dependencies, use the `--upgrade` flag with `uv sync`:
    ```bash
    uv sync --upgrade
    ```
    This will upgrade all installed packages in your environment to the latest available versions compatible with your `pyproject.toml`.


### Verify Installation

```bash
# Run tests
uv run pytest tests

# Check code formatting
uv run ruff check sleap tests

# Run CLI command
uv run sleap-label
```

### Updating Dependencies

To update all dependencies (including `sleap-nn` and `sleap-io`) when working from source, use the `--upgrade` flag with `uv sync`:

=== "Windows/Linux (CUDA 12.8)"
    ```bash
    uv sync --extra nn-cuda128 --upgrade --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CUDA 11.8)"
    ```bash
    uv sync --extra nn-cuda118 --upgrade --index https://download.pytorch.org/whl/cu118 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv sync --extra nn-cpu --upgrade --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv sync --extra nn-cpu --upgrade --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "GUI Only"
    ```bash
    uv sync --upgrade
    ```

!!! info "Updating Local Editable Dependencies"
    If you're developing on multiple related packages (e.g., you have local clones of `sleap`, `sleap-nn`, and `sleap-io`), you can install them all in editable mode:

    ```bash
    # Install sleap-io in editable mode
    uv add --editable "../sleap-io"

    # Install sleap-nn in editable mode
    uv add --editable "../sleap-nn[torch]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple

    # Then sync with upgrade
    uv sync --upgrade
    ```

    This allows you to make changes to any of these packages and see them reflected immediately in your development environment.

---

## Verifying Your Installation

### Running the GUI

```bash
sleap-label
```

You should see the SLEAP labeling interface pop up within a few moments.

!!! note "Environment-specific commands"
    - **conda**: Activate first with `conda activate sleap`
    - **uv add / uv sync**: Prefix with `uv run`, e.g., `uv run sleap-label`
    - **uv tool install**: Commands work directly from any terminal

### GPU Support

SLEAP supports GPU-accelerated training on **NVIDIA GPUs** (Windows/Linux via CUDA) and **Apple Silicon Macs** (via MPS). Other GPU types are not supported. Without a supported GPU, SLEAP uses CPU mode automatically.

To verify GPU detection (for `uv add`/`pip`/`conda` installs):

=== "Windows/Linux (CUDA)"
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

=== "macOS (MPS)"
    ```bash
    python -c "import torch; print(torch.backends.mps.is_available())"
    ```

!!! note "uv tool install"
    GPU verification commands are not available with `uv tool install` since there's no activatable environment. GPU support will be detected automatically when you run training.

!!! tip "CUDA version selection"
    The examples above use CUDA 12.8 (`cu128`). For CUDA 11.8, replace `cu128` with `cu118` in the index URL. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.

### GUI-Only Mode

Installing `sleap` without the `[nn]` extra provides labeling functionality only—no training or inference. Install with `sleap[nn]` for full functionality.

## Uninstalling

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

```bash
sleap-diagnostic
```

(Remember to use `uv run sleap-diagnostic` if you installed with `uv add` or `uv sync`.)

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