# SLEAP Development Guidelines for Claude

## Package Management

- **Always use `uv` for python commands and environment management.**
- Never use `conda` or `pip`.
- Do not call `python` directly, use `uv run python`.

### Basic usage
We manage our dependencies in `pyproject.toml` and will only edit the versions or add dependencies very sparingly and with a lot of careful planning.

```bash
# Sync environment with dependencies (this removes packages installed with `uv pip install`)
uv sync --extra nn

# Run Python in the uv environment
uv run python script.py

# Install local packages (e.g., for sleap-io or sleap-nn development)
uv pip install -e "../sleap-io[all]"
uv pip install -e "../sleap-nn[torch]" --torch-backend=auto

# Only when trying out new packages or pulling something in for an experiment
uv pip install XYZ
```

### Reference for `uv`
- When looking up `uv` command syntax, use `uv --help`, `uv sync --help`, etc.
- For advanced usage, like resolution mechanics, refer to the docs in `scratch/repos/uv/docs`. If it doesn't exist, use `gh` to clone `astral-sh/uv` into that folder.

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_specific.py

# Run with coverage
uv run pytest --cov=sleap
```

## Linting

```bash
# Run ruff for linting
uv run ruff check .

# Run ruff with auto-fix
uv run ruff check --fix .
```

## Project Structure

- `sleap/` - Main SLEAP package
- `tests/` - Test suite
- `docs/` - Documentation
- `scratch/` - Investigation and prototype files (not part of main package)

## Related Projects

- `../sleap-io/` - sleap-io library
  - data backend, data model, I/O backends, merging, labels project (.slp) manipulation, advanced workflows like filtering, rendering or embedding
  - If not available locally in `../sleap-io` (typical local dev setup), use `gh` to clone `talmolab/sleap-io` in `scratch/repos/sleap-io` to have a local copy to read for reference.
- `../sleap-nn/` - sleap-nn neural network package
  - neural network backend, training, inference, evaluation metrics, tracking, model configuration, hyperparameters
  - If not available locally in `../sleap-nn` (typical local dev setup), use `gh` to clone `talmolab/sleap-nn` to `scratch/repos/sleap-nn` to have a local copy to read for reference.
