# Installation

SLEAP is a tool for tracking animal poses in video. This guide will help you install it.

**What do you want to do?**

- **Just use SLEAP** → [Install SLEAP](#install-sleap) (most users)
- **Try it first without installing** → [Quick Run](#quick-run)
- **Test new features before release** → [Pre-release Versions](#pre-release-versions)
- **Test a bug fix from a branch** → [Install from Git](#install-from-git)
- **Develop or contribute to SLEAP** → [Development Setup](#development-setup)
- **Use SLEAP programmatically** → [Programmatic Usage](#programmatic-usage)
- **Previously used conda?** → [Migrating from Conda](#migrating-from-conda)

---

## Before You Start

### System Requirements

- **Operating System:** Windows 10+, macOS 12+, or Linux
- **RAM:** 8GB minimum, 16GB+ recommended for training
- **Disk Space:** 5GB for installation
- **GPU (optional):** NVIDIA GPU with CUDA support speeds up training significantly

### Do You Have an NVIDIA GPU?

Training neural networks is much faster with a GPU. Check if you have one:

=== "Windows"
    1. Press the **Windows key**, type `Device Manager`, press **Enter**
    2. Click the arrow next to **Display adapters**
    3. If you see **NVIDIA** in the name, you have a compatible GPU → use **CUDA** commands
    4. If you see Intel, AMD, or something else → use **CPU** commands

=== "Linux"
    Run in terminal: `lspci | grep -i nvidia`

    - If output shows NVIDIA → use **CUDA** commands
    - If empty → use **CPU** commands

=== "macOS"
    Apple Silicon Macs (M1/M2/M3/M4) have GPU acceleration built-in. No special setup needed.

**Not sure?** Use CPU commands. You can always reinstall with CUDA later.

### Install uv

`uv` is the package manager used to install SLEAP. Install it first:

=== "Windows"
    1. Press the **Windows key**, type `cmd`, press **Enter** to open Command Prompt
    2. Copy and paste this command, then press **Enter**:
    ```
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    3. **Close Command Prompt and reopen it** (required for the installation to take effect)
    4. Verify it worked by typing: `uv --version`

    You should see something like `uv 0.5.14`. If you see "command not found", restart your computer and try again.

=== "macOS"
    1. Press **Cmd+Space**, type `Terminal`, press **Enter**
    2. Copy and paste this command, then press **Enter**:
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    3. Close Terminal and reopen it
    4. Verify: `uv --version`

=== "Linux"
    1. Open a terminal (Ctrl+Alt+T on most systems)
    2. Run:
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    3. Close terminal and reopen it
    4. Verify: `uv --version`

---

## Install SLEAP

This installs SLEAP as a system-wide tool. After installation, you can run `sleap` from any terminal.

Choose your platform (Windows and Linux commands are the same):

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --python 3.12 "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --python 3.12 "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --python 3.12 "sleap[nn]"
    ```

**What does `[nn]` mean?** The `[nn]` installs neural network dependencies (PyTorch) for training models. If you only need to view and annotate data (no training), you can omit it: `uv tool install --python 3.12 sleap`

**Expected output:**
```
Resolved X packages in Xs
Installed sleap
 + sleap
 ...
```

**Verify it worked:**
```
sleap
```
A window should open within a few seconds. Close it with the X button.

**Update to latest stable version:**
```
uv tool upgrade sleap
```

**Check installed version:**
```
sleap doctor
```

**Uninstall:**
```
uv tool uninstall sleap
```

---

## Pre-release Versions

To test new features before official release, you can install pre-release versions (alpha, beta, release candidates). Pre-releases require the `--prerelease allow` flag.

!!! warning "Pre-release software"
    Pre-release versions may contain bugs or incomplete features. Use stable releases for production annotation work.

### Install Latest Pre-release

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]" --prerelease allow --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]" --prerelease allow --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --force --python 3.12 "sleap[nn]" --prerelease allow
    ```

### Install a Specific Version

Pin to an exact version for reproducibility or to test a specific release:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.6.0a0" --prerelease allow --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.6.0a0" --prerelease allow --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.6.0a0" --prerelease allow
    ```

### Pin Dependency Versions

For full reproducibility, you can explicitly pin the versions of `sleap-io` and `sleap-nn` using the `--with` flag:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.6.0a0" --with "sleap-io==0.6.0" --with "sleap-nn==0.1.0a0" --prerelease allow --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.6.0a0" --with "sleap-io==0.6.0" --with "sleap-nn==0.1.0a0" --prerelease allow --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.6.0a0" --with "sleap-io==0.6.0" --with "sleap-nn==0.1.0a0" --prerelease allow
    ```

!!! info "Version compatibility"
    The SLEAP ecosystem has three packages with coordinated releases:

    | SLEAP | sleap-io | sleap-nn |
    |-------|----------|----------|
    | 1.6.0aN | 0.6.x | 0.1.0aN |
    | 1.6.x | 0.6.x | 0.1.x |
    | 1.5.x | 0.5.x | 0.0.x |

    When pinning versions, ensure you use compatible combinations. Mismatched versions may cause errors.

### Rollback to Stable

If you encounter issues with a pre-release, rollback to the latest stable version:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.5.2" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.5.2" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --force --python 3.12 "sleap[nn]==1.5.2"
    ```

### CUDA 13.0 Support

Starting with v1.6.0, SLEAP supports CUDA 13.0 for the latest NVIDIA GPUs:

```
uv tool install --force --python 3.12 "sleap[nn]" --prerelease allow --index https://download.pytorch.org/whl/cu130 --index https://pypi.org/simple
```

---

## Quick Run

Run SLEAP without installing it permanently. Useful for quickly viewing or annotating data on any computer with `uv`.

**View and annotate (no training):**
```
uvx sleap
```

Open a file directly:
```
uvx sleap labels.slp
```

This works on any platform—no GPU or extra setup needed.

**With training support:**

If you need to train models, include the `[nn]` extra:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uvx --python 3.12 --from "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap
    ```

=== "Windows/Linux without GPU"
    ```
    uvx --python 3.12 --from "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple sleap
    ```

=== "macOS"
    ```
    uvx --python 3.12 --from "sleap[nn]" sleap
    ```

---

## Install from Git

Install SLEAP from source to test bug fixes before they're released. The SLEAP ecosystem has three packages, and you may need to install one or more from git depending on where the fix is.

### Quick Reference

| What needs testing | Command pattern |
|-------------------|-----------------|
| Only `sleap` | `uv tool install "sleap[nn] @ git+https://github.com/talmolab/sleap@BRANCH" ...` |
| Only `sleap-io` | `uv tool install "sleap[nn]" --with "sleap-io @ git+https://github.com/talmolab/sleap-io@BRANCH" ...` |
| Only `sleap-nn` | `uv tool install "sleap[nn]" --with "sleap-nn @ git+https://github.com/talmolab/sleap-nn@BRANCH" ...` |
| All from `develop` | See [Testing everything from develop](#testing-everything-from-develop) |

Replace `BRANCH` with the branch name (e.g., `fix/video-loading`) or `develop` for the latest development version.

### Testing a branch in `sleap`

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap@BRANCH" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap@BRANCH" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap@BRANCH"
    ```

### Testing a branch in `sleap-io`

Install SLEAP from PyPI but override `sleap-io` with a git version:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --python 3.12 "sleap[nn]" --with "sleap-io @ git+https://github.com/talmolab/sleap-io@BRANCH" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --python 3.12 "sleap[nn]" --with "sleap-io @ git+https://github.com/talmolab/sleap-io@BRANCH" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --python 3.12 "sleap[nn]" --with "sleap-io @ git+https://github.com/talmolab/sleap-io@BRANCH"
    ```

### Testing a branch in `sleap-nn`

Install SLEAP from PyPI but override `sleap-nn` with a git version:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --python 3.12 "sleap[nn]" --with "sleap-nn @ git+https://github.com/talmolab/sleap-nn@BRANCH" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --python 3.12 "sleap[nn]" --with "sleap-nn @ git+https://github.com/talmolab/sleap-nn@BRANCH" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --python 3.12 "sleap[nn]" --with "sleap-nn @ git+https://github.com/talmolab/sleap-nn@BRANCH"
    ```

### Testing everything from develop

Install the latest development version of all three packages:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap@develop" --with "sleap-io @ git+https://github.com/talmolab/sleap-io@develop" --with "sleap-nn @ git+https://github.com/talmolab/sleap-nn@develop" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap@develop" --with "sleap-io @ git+https://github.com/talmolab/sleap-io@develop" --with "sleap-nn @ git+https://github.com/talmolab/sleap-nn@develop" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap@develop" --with "sleap-io @ git+https://github.com/talmolab/sleap-io@develop" --with "sleap-nn @ git+https://github.com/talmolab/sleap-nn@develop"
    ```

### Pinning to a specific commit

For reproducibility, you can install from a specific commit hash instead of a branch:

```
uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap@abc123def"
```

---

## Development Setup

For contributing to SLEAP or modifying the source code.

### Prerequisites

**Git must be installed.** Check by running `git --version`. If not installed:

- Windows: Download from [git-scm.com/downloads](https://git-scm.com/downloads)
- macOS: Run `xcode-select --install`
- Linux: Run `sudo apt install git` (Ubuntu/Debian) or `sudo dnf install git` (Fedora)

### Setup

These commands download SLEAP's source code to a folder called `sleap` in your current directory:

```bash
git clone https://github.com/talmolab/sleap.git
cd sleap
```

Then install dependencies:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv sync --python 3.12 --extra dev --extra nn-cuda128 --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv sync --python 3.12 --extra dev --extra nn-cpu --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv sync --python 3.12 --extra dev --extra nn-cpu
    ```

### Running Commands

Use `uv run` to run commands in the development environment:

```bash
uv run sleap              # Launch GUI
uv run sleap doctor       # Check system diagnostics
uv run pytest tests/      # Run tests
uv run ruff check sleap   # Check code style
```

!!! tip "Activate the virtual environment"

    To avoid typing `uv run` before every command, activate the virtual environment:

    === "Windows (Command Prompt)"
        ```
        .venv\Scripts\activate.bat
        ```

    === "Windows (PowerShell)"
        ```powershell
        .venv\Scripts\Activate.ps1
        ```

    === "macOS/Linux"
        ```bash
        source .venv/bin/activate
        ```

    Once activated, you can run commands directly (e.g., `sleap`, `pytest tests/`).

### Developing Multiple Packages

If you're working on `sleap-io` or `sleap-nn` alongside `sleap`, you can install them in editable mode after running `uv sync`:

```bash
# After uv sync, install local packages in editable mode
uv pip install -e ../sleap-io
uv pip install -e ../sleap-nn
```

Note: You'll need to re-run these commands after each `uv sync`, as syncing resets the environment to match the lockfile.

---

## Programmatic Usage

The main [`sleap`](https://github.com/talmolab/sleap) package is primarily the GUI frontend and is not designed to be used as a library. If you want to use SLEAP programmatically, consider these options:

1. **[Command-line interface](reference/command-line-interfaces.md):** Use SLEAP's CLI (`sleap track`, `sleap convert`, etc.) for batch processing and automation.

2. **[`sleap-io`](https://io.sleap.ai):** For working with `.slp` files, labels, skeletons, and videos programmatically. This is the **recommended** library for most scripting needs.
    ```bash
    uv add sleap-io
    ```
    ```python
    import sleap_io as sio
    labels = sio.load_slp("predictions.slp")
    ```

3. **[`sleap-nn`](https://nn.sleap.ai):** For working with the deep learning backend, training models programmatically, or running inference.
    ```bash
    uv add sleap-nn
    ```

---

## Migrating from Conda

Prior to SLEAP v1.5, installation used conda and conda environments. Starting with v1.5, we switched to `uv` for faster, more reliable installations.

The [Install SLEAP](#install-sleap) and [Development Setup](#development-setup) sections above cover all use cases. However, if you prefer working within conda environments, you can still do so:

### Using uv inside a conda environment

```bash
# Create and activate a conda environment
conda create -n sleap python=3.12
conda activate sleap

# Install uv inside the conda environment
pip install uv

# Install SLEAP using uv pip (works inside conda)
uv pip install "sleap[nn]"
```

### Disabling conda auto-activation

If you're switching fully to uv and want to prevent conda from activating automatically:

```bash
conda config --set auto_activate_base false
conda deactivate
```

Then install Python via uv:

```bash
uv python install 3.12
uv python pin 3.12
```

Now you can use any of the installation methods above without conda interference.

---

## Troubleshooting

### "command not found" after install

1. Close your terminal completely and reopen it
2. Run `uv tool list` to verify SLEAP is installed
3. If still not working, restart your computer

### "running scripts is disabled" (Windows)

Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy RemoteSigned
```

### Python version errors

Always include `--python 3.12` in your commands. Python 3.14 is not yet supported.

### Installation seems stuck

Large packages like PyTorch take time to download. Installation can take 5-15 minutes on slower connections. If it takes more than 30 minutes, cancel with Ctrl+C and try again.

### Check your installation

```bash
sleap doctor
```

This shows your system info and whether GPU is detected.

### Still stuck?

1. Run `sleap doctor` and copy the output
2. Ask for help at [github.com/talmolab/sleap/discussions](https://github.com/talmolab/sleap/discussions)
