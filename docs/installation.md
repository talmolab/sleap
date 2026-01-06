# Installation

SLEAP is a tool for tracking animal poses in video. This guide will help you install it.

**What do you want to do?**

- **Just use SLEAP** → [Install SLEAP](#install-sleap) (most users)
- **Try it first without installing** → [Quick Run](#quick-run)
- **Test a bug fix from a branch** → [Install from Git](#install-from-git)
- **Develop or contribute to SLEAP** → [Development Setup](#development-setup)
- **Use SLEAP in your own Python scripts** → [Use as Library](#use-as-library)

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

This installs SLEAP as a system-wide tool. After installation, you can run `sleap-label` from any terminal.

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
 + sleap-label
 + sleap-track
 + sleap-convert
 + sleap-diagnostic
 ...
```

**Verify it worked:**
```
sleap-label
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
    uvx --python 3.12 --from "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap-label
    ```

=== "Windows/Linux without GPU"
    ```
    uvx --python 3.12 --from "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple sleap-label
    ```

=== "macOS"
    ```
    uvx --python 3.12 --from "sleap[nn]" sleap-label
    ```

Replace `sleap-label` with any other command like `sleap-track` or `sleap-convert`.

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
uv run sleap-label           # Launch GUI
uv run pytest tests/         # Run tests
uv run ruff check sleap      # Check code style
```

Or activate the environment to run commands directly:

=== "Windows"
    ```
    .venv\Scripts\activate.bat
    sleap-label
    ```

=== "macOS/Linux"
    ```bash
    source .venv/bin/activate
    sleap-label
    ```

### Developing Multiple Packages

To develop SLEAP and sleap-nn together with live edits:

=== "Windows"
    ```
    uvx --python 3.12 --from "sleap[nn]" --with-editable "D:\sleap" --with-editable "D:\sleap-nn" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple sleap-label
    ```

=== "macOS/Linux"
    ```bash
    uvx --python 3.12 --from "sleap[nn]" --with-editable ~/sleap --with-editable ~/sleap-nn sleap-label
    ```

### Switching from Conda

If you previously used conda:

```bash
conda config --set auto_activate_base false
conda deactivate
uv python install 3.12
uv python pin 3.12
```

Then follow the setup steps above.

---

## Use as Library

To use SLEAP in your own Python scripts (e.g., `import sleap`):

```bash
mkdir my-analysis
cd my-analysis
uv init
```

Then add SLEAP:

=== "Windows/Linux with NVIDIA GPU"
    ```
    uv add --python 3.12 "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux without GPU"
    ```
    uv add --python 3.12 "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```
    uv add --python 3.12 "sleap[nn]"
    ```

Run your scripts:
```bash
uv run python my_script.py
```

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
sleap-diagnostic
```

This shows your system info and whether GPU is detected.

### Still stuck?

1. Run `sleap-diagnostic` and copy the output
2. Open an issue at [github.com/talmolab/sleap/issues](https://github.com/talmolab/sleap/issues)
