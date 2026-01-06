# Installation

SLEAP is a tool for tracking animal poses in video. This guide will help you install it.

**What do you want to do?**

- **Just use SLEAP** → [Install SLEAP](#install-sleap) (most users)
- **Try it first without installing** → [Quick Run](#quick-run)
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

**Update to latest version:**
```
uv tool upgrade sleap
```

**Uninstall:**
```
uv tool uninstall sleap
```

---

## Quick Run

Run SLEAP without installing it permanently. Useful for trying SLEAP or testing a quick fix.

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

Replace `sleap` with subcommands like `sleap label`, `sleap doctor`, or legacy commands like `sleap-track`.

---

## Install from Git

Install SLEAP from a specific branch. Use this when a developer asks you to test a fix before it's released.

**First, replace `BRANCH_NAME` in the command below with the actual branch name** (e.g., `fix/video-loading`), then run:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap.git@BRANCH_NAME" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap.git@BRANCH_NAME" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap.git@BRANCH_NAME"
    ```

**Example:** To install from a branch called `fix/video-loading`:
```
uv tool install --python 3.12 "sleap[nn] @ git+https://github.com/talmolab/sleap.git@fix/video-loading"
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

Or activate the environment to run commands directly:

=== "Windows"
    ```
    .venv\Scripts\activate.bat
    sleap
    ```

=== "macOS/Linux"
    ```bash
    source .venv/bin/activate
    sleap
    ```

### Developing Multiple Packages

To develop SLEAP and sleap-nn together with live edits:

=== "Windows"
    ```
    uvx --python 3.12 --from "sleap[nn]" --with-editable "D:\sleap" --with-editable "D:\sleap-nn" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap
    ```

=== "macOS/Linux"
    ```bash
    uvx --python 3.12 --from "sleap[nn]" --with-editable ~/sleap --with-editable ~/sleap-nn sleap
    ```

---

## Programmatic Usage

The main [`sleap`](https://github.com/talmolab/sleap) package is primarily the GUI frontend and is not designed to be used as a library. If you want to use SLEAP programmatically, consider these options:

1. **[Command-line interface](cli.md):** Use SLEAP's CLI (`sleap track`, `sleap convert`, etc.) for batch processing and automation.

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
