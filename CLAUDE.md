# SLEAP Development Guidelines for Claude

## Package Management

**Always use `uv` for package management.** Never use conda or pip directly.

```bash
# Sync environment with dependencies
uv sync --extra nn

# Install local packages (e.g., sleap-io in development)
uv pip install "../sleap-io[all]"

# Run Python in the uv environment
uv run python script.py

# Or activate the virtual environment
source .venv/bin/activate
python script.py
```

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

- `../sleap-io/` - sleap-io library (install with `uv pip install "../sleap-io[all]"`)
- `../sleap-nn/` - sleap-nn neural network package
