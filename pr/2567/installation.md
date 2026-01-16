# Installation

SLEAP is a tool for tracking animal poses in video. This guide will get you up and running.

**What do you want to do?**

- [**Use SLEAP**](#install-sleap) (most users)
- [**Update SLEAP**](#updating)
- [**Try pre-release features**](#pre-release-versions)
- [**Develop or contribute**](#development-setup)
- [**Use SLEAP as a library**](#programmatic-usage)

---

## Before You Start

??? question "Why do I need `uv`?"
    **What is Python package management?**

    Python packages often depend on other packages, which in turn have their own dependencies—each requiring specific versions. Without careful management, you can end up in "dependency hell" where different projects need conflicting versions. Package managers solve this by creating isolated environments where each project gets exactly the versions it needs.

    **Why did SLEAP use `conda` before?**

    SLEAP's neural networks required GPU libraries (CUDA) that were notoriously difficult to install correctly. Conda handled this by bundling CUDA inside isolated environments, making GPU-accelerated training "just work." For many years, this was the only reliable way to install SLEAP.

    **What changed?**

    Starting in SLEAP 1.5, we transitioned from TensorFlow to PyTorch. Unlike TensorFlow, PyTorch bundles all GPU dependencies directly in its `pip` package—no separate CUDA installation needed. This eliminated the main reason we needed conda.

    Conda also had drawbacks: it was slow (environment creation could take 10+ minutes), and you had to remember to "activate" your environment every time you wanted to use SLEAP. If you forgot, you'd get confusing errors.

    **What is `uv` and why use it?**

    `uv` is a modern Python package manager that's blazingly fast (10-100x faster than pip or conda). Beyond speed, `uv` has a killer feature: it can install packages as **tools** that are available system-wide without needing to activate anything.

    When you run `uv tool install sleap`, it creates an isolated environment behind the scenes, but exposes the `sleap` command globally. You just type `sleap` and it works—no activation, no environment management, no mental overhead.

    Because `uv` is so fast, it's even practical to have multiple versions installed or switch between them. But for most users, the best part is that you can just install SLEAP once and forget about environments entirely.

### Install uv

SLEAP uses `uv` to manage installation. It's a fast, modern package manager that handles everything automatically—including GPU detection.

=== "Windows"
    1. Press the **Windows key**, type `PowerShell`, press **Enter**
    2. Paste this command and press **Enter**:
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```
    3. **Close and reopen** PowerShell
    4. Verify: `uv --version`

=== "macOS"
    1. Press **Cmd+Space**, type `Terminal`, press **Enter**
    2. Paste this command and press **Enter**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    3. **Close and reopen** Terminal
    4. Verify: `uv --version`

=== "Linux"
    1. Open a terminal (usually **Ctrl+Alt+T**)
    2. Paste this command and press **Enter**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    3. **Close and reopen** the terminal
    4. Verify: `uv --version`

---

## Install SLEAP

One command works on all platforms. It automatically detects your GPU and installs the right version of PyTorch.

```bash
uv tool install --python 3.13 "sleap[nn]==1.6.0a1" --with "sleap-io==0.6.1" --with "sleap-nn==0.1.0a1" --prerelease allow --torch-backend auto
```

That's it! SLEAP is now available system-wide. Run it from any terminal:

```bash
sleap
```

A window should open within a few seconds.

??? info "What does this command do?"
    - `--python 3.13` — Uses Python 3.13
    - `sleap[nn]` — Installs SLEAP with neural network support for training
    - `--with "sleap-io==..."` — Pins dependency versions for compatibility
    - `--prerelease allow` — Required for alpha/beta versions
    - `--torch-backend auto` — Automatically detects your GPU (NVIDIA, AMD, Intel, or CPU)

??? tip "Getting an error about `--torch-backend`?"
    Update uv to the latest version:
    ```bash
    uv self update
    ```

### Verify installation

```bash
sleap doctor
```

This shows your system info, package versions, and confirms GPU detection.

### Just viewing or annotating (no training)

If you only need to view and annotate data without training models, you don't even need to install anything:

```bash
uvx sleap labels.slp
```

This runs SLEAP directly without a permanent installation. Replace `labels.slp` with your file, or omit it to open SLEAP with an empty project.

---

## Updating

### Check your current version

```bash
sleap doctor
```

### Upgrade everything to latest

```bash
uv tool upgrade sleap
```

This upgrades SLEAP and all its dependencies to the latest compatible versions. It remembers your original settings (like `--prerelease allow` and `--torch-backend auto`).

### Upgrade just sleap-io or sleap-nn

If there's a new release of a dependency but not SLEAP itself:

```bash
uv tool upgrade sleap --upgrade-package sleap-io
```

```bash
uv tool upgrade sleap --upgrade-package sleap-nn
```

```bash
uv tool upgrade sleap --upgrade-package sleap-io --upgrade-package sleap-nn
```

### Upgrade to a specific version

If you need specific versions (for reproducibility or to match a collaborator), reinstall:

```bash
uv tool install --python 3.13 "sleap[nn]==1.6.0a1" --with "sleap-io==0.6.1" --with "sleap-nn==0.1.0a1" --prerelease allow --torch-backend auto
```

This replaces the existing installation with the exact versions specified.

### Downgrade

Just reinstall with the older version:

```bash
uv tool install --python 3.13 "sleap[nn]==1.5.2" --torch-backend auto
```

### Uninstall

```bash
uv tool uninstall sleap
```

??? note "When to use `--reinstall`"
    Most of the time, you don't need it. Use `--reinstall` when:

    - Something is broken and you want a completely fresh environment
    - Installing from local source code (to pick up changes)

    ```bash
    uv tool install --reinstall --python 3.13 "sleap[nn]==1.6.0a1" --with "sleap-io==0.6.1" --with "sleap-nn==0.1.0a1" --prerelease allow --torch-backend auto
    ```

---

## Pre-release Versions

Pre-releases let you try new features before official release. They may have bugs, so use stable versions for important annotation work.

### Latest pre-release

```bash
uv tool install --python 3.13 "sleap[nn]" --prerelease allow --torch-backend auto
```

### Version compatibility

The SLEAP ecosystem has three packages that work together:

| SLEAP | sleap-io | sleap-nn |
|-------|----------|----------|
| 1.6.0a1 | 0.6.1 | 0.1.0a1 |
| 1.6.0a0 | 0.6.0 | 0.1.0a0 |
| 1.5.x | <0.6.0 | <0.1.0 |

Always use compatible versions when pinning.

??? note "Force a specific PyTorch backend"
    If `--torch-backend auto` doesn't detect your GPU correctly, you can specify it manually:

    | Backend | For |
    |---------|-----|
    | `cu128` | NVIDIA GPUs (CUDA 12.8) |
    | `cu130` | Newest NVIDIA GPUs (CUDA 13.0) |
    | `cpu` | No GPU / CPU only |
    | `rocm` | AMD GPUs |
    | `xpu` | Intel GPUs |

    ```bash
    uv tool install --python 3.13 "sleap[nn]==1.6.0a1" --with "sleap-io==0.6.1" --with "sleap-nn==0.1.0a1" --prerelease allow --torch-backend cu128
    ```

---

## Development Setup

For contributors and developers who want to modify SLEAP's source code.

### Full ecosystem setup (all three repos)

**1. Clone the repositories:**

```bash
git clone https://github.com/talmolab/sleap
git clone https://github.com/talmolab/sleap-nn
git clone https://github.com/talmolab/sleap-io
cd sleap
```

**2. Install with editable local packages:**

```bash
uv sync --extra nn --reinstall
uv pip install -e "../sleap-io[all]"
uv pip install -e "../sleap-nn[torch]" --torch-backend=auto
```

??? warning "Note about `uv sync`"
    Running `uv sync` again will overwrite your local editable installs with PyPI versions. After any `uv sync`, re-run the `uv pip install -e` commands.

**3. Activate the environment:**

=== "Linux/macOS"
    ```bash
    source .venv/bin/activate
    ```

=== "Windows (PowerShell)"
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

=== "Windows (Command Prompt)"
    ```cmd
    .venv\Scripts\activate.bat
    ```

**4. Run commands:**

```bash
sleap
pytest tests/
```

Or without activating the environment:

```bash
uv run sleap
uv run pytest tests/
```

### Use local dev as system tool

Want to run your modified SLEAP from anywhere without activating a venv? Install from local source:

```bash
uv tool install --reinstall --python 3.13 ".[nn]" --with "../sleap-io[all]" --with "../sleap-nn" --prerelease allow --torch-backend auto
```

Now you can run `sleap` from anywhere and it uses your local code!

Re-run with `--reinstall` after making changes to pick them up.

---

## Programmatic Usage

The `sleap` package is primarily the GUI application. For scripting and automation, use these libraries:

### sleap-io

For working with `.slp` files, labels, skeletons, and videos programmatically. Use this for loading predictions, extracting coordinates, filtering data, merging projects, and custom analysis pipelines.

Documentation and installation: [io.sleap.ai](https://io.sleap.ai)

### sleap-nn

For training models, running inference, and evaluating predictions programmatically. Use this for custom training workflows, batch inference, and integration with other ML pipelines.

Documentation and installation: [nn.sleap.ai](https://nn.sleap.ai)

---

## Troubleshooting

### Verify your installation

```bash
sleap doctor
```

This is always the first step. It shows:

- SLEAP version and all package versions
- Python version
- GPU detection status
- System information

### `--torch-backend` not recognized

Update uv:

```bash
uv self update
```

### Force a clean reinstall

If something is broken:

```bash
uv tool install --reinstall --python 3.13 "sleap[nn]==1.6.0a1" --with "sleap-io==0.6.1" --with "sleap-nn==0.1.0a1" --prerelease allow --torch-backend auto
```

### Installation seems stuck

Large packages like PyTorch take time. Installation can take 5-15 minutes on slower connections. Wait up to 30 minutes before cancelling.

### Still stuck?

1. Run `sleap doctor` and copy the output
2. Ask for help at [github.com/talmolab/sleap/discussions](https://github.com/talmolab/sleap/discussions)
