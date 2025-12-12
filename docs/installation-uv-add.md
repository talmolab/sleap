# Installation with uv add

This method creates a dedicated project environment using uv's modern Python project management. It's ideal for **project-based workflows** where you need fine-grained control over dependencies and have a reproducible environment.

## Why use this method?

- **Reproducible environments**: Dependencies are tracked in `pyproject.toml` and locked in `uv.lock`, making it easy to share and recreate environments
- **Project isolation**: Each project has its own environment, avoiding conflicts between projects
- **Use SLEAP as a library**: Import SLEAP in your own Python scripts (e.g., `import sleap`)
- **Add custom dependencies**: Easily add other packages alongside SLEAP for your analysis workflows
- **Development-friendly**: Supports editable installs for working on SLEAP or related packages

## Setup

Initialize your project environment:

```bash
uv init
uv venv
```

This creates a `pyproject.toml` file and a `.venv` virtual environment in your working directory.

## Platform-Specific Installation

=== "Windows/Linux (CUDA)"
    ```bash
    # CUDA 12.8
    uv add "sleap[nn]" --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
    ```

=== "Windows/Linux (CPU)"
    ```bash
    uv add "sleap[nn]" --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
    ```

=== "macOS"
    ```bash
    uv add "sleap[nn]"
    ```

=== "SLEAP GUI Only"
    ```bash
    uv add "sleap"
    ```

!!! warning "Windows: MarkupSafe Installation Issue"
    On **Windows**, you may encounter errors due to an incompatibility with the MarkupSafe wheel.
    Similar issues: [#11532](https://github.com/astral-sh/uv/issues/11532) and [#12620](https://github.com/astral-sh/uv/issues/12620).

    **Workaround:** Before running `uv add "sleap[nn]" ...`, manually install a compatible version:

    ```bash
    uv add git+https://github.com/pallets/markupsafe@3.0.2
    ```

## Running Commands

To use SLEAP, you **must prefix commands with `uv run`**:

```bash
uv run sleap-label
uv run sleap-track --help
uv run python -c "import sleap; print(sleap.__version__)"
```

!!! tip "How `uv add` works"
    - `uv add "sleap[nn]"` adds SLEAP as a dependency in your `pyproject.toml` and installs it
    - To add other packages: `uv add <package>`
    - After adding packages, run `uv sync` to update your environment
    - Use `uv sync --upgrade` to update all dependencies to latest compatible versions

## Verify Installation

```bash
uv run sleap-label --help
```

!!! warning "SLEAP not recognized after installation?"
    If you get `command not found`:

    - Try activating your virtual environment first, then run:
      ```bash
      uv run --active sleap-label --help
      ```
    - Check for empty `pyproject.toml` or `uv.lock` files in `Users/<your-user-name>` that may interfere with uv's environment resolution

## Updating Dependencies

```bash
# Update only SLEAP
uv add "sleap[nn]" --upgrade-package sleap

# Update a specific dependency (e.g., sleap-nn)
uv add "sleap[nn]" --upgrade-package sleap-nn

# Update all dependencies to latest compatible versions
uv sync --upgrade
```

!!! tip "Platform-Specific Upgrades"
    When updating with platform-specific PyTorch requirements, include the appropriate index URLs:

    === "Windows/Linux (CUDA 12.8)"
        ```bash
        uv add "sleap[nn]" --upgrade-package sleap --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple
        ```

    === "Windows/Linux (CPU)"
        ```bash
        uv add "sleap[nn]" --upgrade-package sleap --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
        ```

    === "macOS"
        ```bash
        uv add "sleap[nn]" --upgrade-package sleap
        ```

## Sharing Your Environment

To share your environment with collaborators or reproduce it on another machine:

1. **Share these files** from your working directory:
    - `pyproject.toml` - lists your dependencies
    - `uv.lock` - locks exact versions for reproducibility

2. **Collaborators can recreate the environment** by placing these files in their working directory and running:
    ```bash
    uv sync
    ```

This installs the exact same package versions, ensuring consistent results across machines.
