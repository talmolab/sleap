# Install SLEAP

SLEAP tracks animal poses in video. Three commands get you from zero to a running GUI.

!!! abstract "TL;DR — already have `uv`?"
    **Install** (auto-detects your GPU):
    ```bash
    uv tool install --python 3.13 "sleap[nn]" --torch-backend auto
    ```
    **Upgrade:**
    ```bash
    uv tool upgrade sleap
    ```
    **Develop** (editable install of all three repos):
    ```bash
    git clone https://github.com/talmolab/sleap && git clone https://github.com/talmolab/sleap-io && git clone https://github.com/talmolab/sleap-nn && cd sleap && uv sync --extra nn --reinstall && uv pip install -e "../sleap-io[all]" && uv pip install -e "../sleap-nn[torch]" --torch-backend=auto
    ```
    New here? Follow the [step-by-step quick start](#quick-start) below — it installs `uv` first.

!!! note "Using SLEAP 1.4 or earlier?"
    This guide is for SLEAP 1.5+ (`uv`-based). For older conda installs, see the [legacy documentation](https://legacy.sleap.ai) or the [migration guide](guides/migrating-to-sleap-1-5.md).

---

## Quick start

SLEAP installs with [`uv`](https://docs.astral.sh/uv/), a fast Python package manager that automatically detects your GPU. Install `uv` once, then install SLEAP.

### 1. Install uv

=== "Windows"
    Open **PowerShell** (press the **Windows key**, type `PowerShell`, press **Enter**), then run:
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    **Close and reopen** PowerShell, then check it worked:
    ```powershell
    uv --version
    ```
    You should see `uv 0.x.x`. If you instead see `uv is not recognized`, fully close all PowerShell windows and reopen (or restart your computer).

=== "macOS"
    Open **Terminal** (press **Cmd+Space**, type `Terminal`, press **Enter**), then run:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    **Close and reopen** Terminal, then check it worked:
    ```bash
    uv --version
    ```
    You should see `uv 0.x.x`.

=== "Linux"
    Open a terminal (often **Ctrl+Alt+T**), then run:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    **Close and reopen** the terminal, then check it worked:
    ```bash
    uv --version
    ```
    You should see `uv 0.x.x`.

### 2. Install SLEAP

One command works on all platforms — it auto-detects your GPU (NVIDIA, AMD, Intel, or CPU) and installs the matching PyTorch build:

```bash
uv tool install --python 3.13 "sleap[nn]" --torch-backend auto
```

SLEAP is now available system-wide — no environment to activate.

??? info "What does this command do?"
    - `--python 3.13` — pins Python 3.13. **Always include this.** Without it, `uv` may download Python 3.14, which SLEAP does not support yet. (Python 3.12 also works: use `--python 3.12`.)
    - `sleap[nn]` — SLEAP plus neural-network support for training and inference.
    - `--torch-backend auto` — detects your GPU and installs the right PyTorch build.

    Need exact, reproducible versions instead? See [Version compatibility](#version-compatibility).

### 3. Launch SLEAP

```bash
sleap
```

A window opens within a few seconds. To check your install — package versions and GPU detection:

```bash
sleap doctor
```

!!! tip "Just viewing or annotating? No install needed."
    To view and label data without training models, run SLEAP straight from `uv`:
    ```bash
    uvx sleap labels.slp
    ```
    Replace `labels.slp` with your file, or omit it to open an empty project. Training and inference need the full install (`sleap[nn]`) above.

---

## Common commands

**Upgrade to the latest version:**

```bash
uv tool upgrade sleap
```

This upgrades SLEAP and its dependencies, keeping your original settings (like `--torch-backend auto`).

**Set up a development install** (editable checkout of all three repos):

```bash
git clone https://github.com/talmolab/sleap && git clone https://github.com/talmolab/sleap-io && git clone https://github.com/talmolab/sleap-nn && cd sleap && uv sync --extra nn --reinstall && uv pip install -e "../sleap-io[all]" && uv pip install -e "../sleap-nn[torch]" --torch-backend=auto
```

See [Developer setup](#developer-setup) for the step-by-step version.

**Try without installing:**

```bash
uvx sleap labels.slp
```

??? note "Manage your install — upgrade one package, pin, downgrade, uninstall"
    **Upgrade just a dependency** (e.g. a new `sleap-io` release but not SLEAP itself):
    ```bash
    uv tool upgrade sleap --upgrade-package sleap-io
    ```
    Repeat `--upgrade-package` for each one, e.g. `--upgrade-package sleap-io --upgrade-package sleap-nn`.

    **Pin or downgrade to exact versions** — just reinstall, pinning all three packages (see the [compatibility table](#version-compatibility)):
    ```bash
    uv tool install --python 3.13 "sleap[nn]==1.6.1" --with "sleap-io==0.6.4" --with "sleap-nn==0.1.0" --torch-backend auto
    ```

    **Uninstall:**
    ```bash
    uv tool uninstall sleap
    ```

    Add `--reinstall` to any install command for a completely fresh environment — use it when something is broken, or when installing from local source.

??? note "Install development versions — latest fixes from GitHub"
    To pull in unreleased fixes, install SLEAP directly from the `develop` branch:
    ```bash
    uv tool install --reinstall --python 3.13 "sleap[nn] @ git+https://github.com/talmolab/sleap@develop" --prerelease allow --torch-backend auto
    ```
    Re-run the same command to update to the latest `develop` commit (`--reinstall` re-fetches it).

    To pull a fix from **sleap-io** or **sleap-nn** into your existing install *without* changing SLEAP, reinstall with a git override (their development branch is `main`):
    ```bash
    # latest sleap-io
    uv tool install --reinstall --python 3.13 "sleap[nn]" --with "sleap-io[all] @ git+https://github.com/talmolab/sleap-io@main" --prerelease allow --torch-backend auto

    # latest sleap-nn
    uv tool install --reinstall --python 3.13 "sleap[nn]" --with "sleap-nn[torch] @ git+https://github.com/talmolab/sleap-nn@main" --prerelease allow --torch-backend auto

    # both at once
    uv tool install --reinstall --python 3.13 "sleap[nn]" --with "sleap-io[all] @ git+https://github.com/talmolab/sleap-io@main" --with "sleap-nn[torch] @ git+https://github.com/talmolab/sleap-nn@main" --prerelease allow --torch-backend auto
    ```
    Development versions may be unstable. If a dependency's dev version isn't compatible with the released SLEAP, install SLEAP from `develop` (first command) as well.

---

## Version compatibility

The SLEAP ecosystem is three packages that release together. Use compatible versions when pinning.

| SLEAP | sleap-io | sleap-nn |
|-------|----------|----------|
| {{ sleap_version }} | {{ sleap_io_version }} | {{ sleap_nn_version }} |
| 1.6.1 | 0.6.4 | 0.1.0 |

??? note "Older versions"
    | SLEAP | sleap-io | sleap-nn |
    |-------|----------|----------|
    | 1.6.0 | 0.6.4 | 0.1.0 |
    | 1.6.0a3 | 0.6.3 | 0.1.0a4 |
    | 1.6.0a2 | 0.6.2 | 0.1.0a2 |
    | 1.6.0a1 | 0.6.1 | 0.1.0a1 |
    | 1.6.0a0 | 0.6.0 | 0.1.0a0 |
    | 1.5.x | <0.6.0 | <0.1.0 |

**Reproducible install** (exact versions — e.g. to match a collaborator):

```bash
uv tool install --python 3.13 "sleap[nn]=={{ sleap_version }}" --with "sleap-io=={{ sleap_io_version }}" --with "sleap-nn=={{ sleap_nn_version }}" --torch-backend auto
```

??? note "Try a pre-release"
    Pre-releases let you try new features early. They may have bugs, so use stable versions for important annotation work.
    ```bash
    uv tool install --python 3.13 "sleap[nn]" --prerelease allow --torch-backend auto
    ```

??? note "Force a specific GPU backend"
    If `--torch-backend auto` doesn't detect your hardware correctly, set it explicitly:

    | Backend | For |
    |---------|-----|
    | `cu128` | NVIDIA GPUs (CUDA 12.8) |
    | `cu130` | Newest NVIDIA GPUs (CUDA 13.0) |
    | `cu118` | NVIDIA GPUs with older drivers (CUDA 11.8) |
    | `cpu` | No GPU / CPU only |
    | `rocm6.4` | AMD GPUs (use the version matching your ROCm install) |
    | `xpu` | Intel GPUs |

    ```bash
    uv tool install --python 3.13 "sleap[nn]" --torch-backend cu128
    ```
    Run `uv tool install --help` for the current list of backend values.

---

## Developer setup

For contributors who want to modify SLEAP's source. (A copy-paste one-liner is in [Common commands](#common-commands) above.)

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

??? warning "`uv sync` overwrites editable installs"
    Running `uv sync` again replaces your local `-e` installs with PyPI versions. Re-run the two `uv pip install -e` commands after any `uv sync`.

**3. Run SLEAP** — without activating anything:

```bash
uv run sleap
uv run pytest tests/
```

Or activate the environment first, then run `sleap` / `pytest tests/` directly:

=== "Windows"
    PowerShell:
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
    Command Prompt:
    ```bat
    .venv\Scripts\activate.bat
    ```

=== "macOS"
    ```bash
    source .venv/bin/activate
    ```

=== "Linux"
    ```bash
    source .venv/bin/activate
    ```

??? note "Run your local dev build from anywhere"
    Install your working copy as a global tool, so `sleap` runs your local code from any terminal without activating a venv:
    ```bash
    uv tool install --reinstall --python 3.13 ".[nn]" --with "../sleap-io[all]" --with "../sleap-nn[torch]" --prerelease allow --torch-backend auto
    ```
    Re-run with `--reinstall` after making changes to pick them up.

---

## Troubleshooting

**First step:** run `sleap doctor` and read the output for errors.

??? note "`--torch-backend` not recognized"
    Update `uv` to the latest version:
    ```bash
    uv self update
    ```

??? note "GPU not detected"
    If `sleap doctor` shows no GPU:

    1. **Check the driver:** run `nvidia-smi`. If it fails, [install drivers](https://www.nvidia.com/drivers). CUDA 12.8 requires driver 525+.
    2. **Set the backend explicitly:** reinstall with `--torch-backend cu128` instead of `auto` (see the GPU-backend table under [Version compatibility](#version-compatibility)).

??? note "Installation seems stuck"
    Large packages like PyTorch take time — 5–15 minutes is normal on slower connections. Wait up to 30 minutes before cancelling.

??? note "Start over with a clean install"
    ```bash
    uv tool install --reinstall --python 3.13 "sleap[nn]" --torch-backend auto
    ```

**Still stuck?** Run `sleap doctor`, copy the output, and ask on [GitHub Discussions](https://github.com/talmolab/sleap/discussions).

---

## Advanced & alternatives

??? note "Model export (ONNX / TensorRT)"
    To export trained models for deployment, add the export extras. [Learn more about exporting models](https://nn.sleap.ai/latest/guides/export/).

    **Tool install** — add the extra to your install command:
    ```bash
    # ONNX, CPU runtime
    uv tool install --python 3.13 "sleap[nn,nn-export]" --torch-backend auto

    # ONNX, GPU runtime (faster inference)
    uv tool install --python 3.13 "sleap[nn,nn-export-gpu]" --torch-backend auto

    # TensorRT (Linux/Windows only) — needs a CUDA backend
    uv tool install --python 3.13 "sleap[nn,nn-tensorrt]" --torch-backend cu128
    ```

    **Developer setup** — add the extra to `uv sync`:
    ```bash
    uv sync --extra nn --extra nn-export            # ONNX CPU runtime
    uv sync --extra nn --extra nn-export-gpu         # ONNX GPU runtime
    uv sync --extra nn-cuda128 --extra nn-tensorrt   # TensorRT
    ```

    TensorRT is not supported on macOS.

??? note "Install with pip (alternative)"
    Prefer `pip`, or integrating SLEAP into an existing environment? Create a virtual environment, then install with the PyTorch index for your hardware:
    ```bash
    python3.13 -m venv sleap_env
    # Windows:      sleap_env\Scripts\activate
    # macOS/Linux:  source sleap_env/bin/activate

    # CPU only
    pip install "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cpu

    # NVIDIA GPU (CUDA 12.8)
    pip install "sleap[nn]" --extra-index-url https://download.pytorch.org/whl/cu128
    ```
    Unlike `uv --torch-backend`, pip can't guarantee which PyTorch build it picks — if you need a specific CPU/GPU build, prefer the `uv` install above. A conda environment works too, but `uv` (or a plain venv) is recommended.

??? note "Use SLEAP as a library"
    The `sleap` package is primarily the GUI application. For scripting and automation, use the libraries directly:

    | Library | Use for | Docs |
    |---------|---------|------|
    | **sleap-io** | `.slp` files, labels, skeletons, videos, merging projects, custom analysis | [io.sleap.ai](https://io.sleap.ai) |
    | **sleap-nn** | Training models, running inference, evaluating predictions, batch processing | [nn.sleap.ai](https://nn.sleap.ai) |

??? question "Why `uv` instead of conda?"
    SLEAP 1.5+ switched from TensorFlow to PyTorch, which bundles its own GPU libraries — so the conda/CUDA juggling that older versions needed is gone. `uv` installs SLEAP as a global tool (`uv tool install`) that works from any terminal with no environment to activate, and it's far faster than conda. Full background is in the [migration guide](guides/migrating-to-sleap-1-5.md).
