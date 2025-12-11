# SLEAP Project

## Overview

SLEAP (Social LEAP Estimates Animal Poses) is a deep learning framework for multi-animal pose tracking. This repository is the main GUI application and CLI toolset that integrates with the sleap-nn neural network backend and sleap-io data utilities.

SLEAP is a multi-animal pose tracking system consisting of three repositories that work together.

### Repositories

#### sleap (this repo)
- **URL**: https://github.com/talmolab/sleap
- **Purpose**: Main GUI application for labeling, training, and running inference
- **Key components**:
  - Qt-based GUI (`sleap/gui/`)
  - CLI tools (`sleap-label`, `sleap-train`, `sleap-track`, etc.)
  - Data structures for labels, skeletons, videos
- **Docs**: https://sleap.ai

#### sleap-nn
- **URL**: https://github.com/talmolab/sleap-nn
- **Purpose**: PyTorch neural network backend for training and inference
- **Key components**:
  - Model architectures (Single Instance, Top-Down, Bottom-Up, Multi-Class)
  - Training pipelines
  - Inference engines
- **Docs**: https://nn.sleap.ai
- **Python**: 3.11, 3.12, 3.13

#### sleap-io
- **URL**: https://github.com/talmolab/sleap-io
- **Purpose**: Lightweight I/O utilities with minimal dependencies
- **Key components**:
  - File format readers/writers (SLEAP, NWB, DeepLabCut, etc.)
  - Video backends (OpenCV, FFMPEG, PyAV)
  - Data conversion utilities
- **Docs**: https://io.sleap.ai
- **Design philosophy**: Minimal dependencies, focused scope

### Repository Interactions

```
┌─────────────────────────────────────────────────────┐
│                      sleap                          │
│                   (GUI + CLI)                       │
│                                                     │
│    Uses sleap-nn for training/inference             │
│    Uses sleap-io for data loading/saving            │
└──────────────┬─────────────────────┬────────────────┘
               │                     │
               ▼                     ▼
┌──────────────────────┐   ┌──────────────────────────┐
│      sleap-nn        │   │       sleap-io           │
│  (neural networks)   │◄──│   (data I/O utilities)   │
│                      │   │                          │
│  Uses sleap-io for   │   │                          │
│  data loading        │   │                          │
└──────────────────────┘   └──────────────────────────┘
```

- **sleap** depends on both sleap-nn and sleap-io
- **sleap-nn** depends on sleap-io for data loading
- **sleap-io** is the foundational layer with minimal dependencies

## Technology Stack

- **Language**: Python 3.11, 3.12, 3.13
- **Package Manager**: uv (with PyTorch index configurations)
- **GUI**: PySide6 / QtPy
- **Deep Learning**: sleap-nn (PyTorch backend)
- **I/O**: sleap-io (minimal dependency I/O utilities)
- **Data Processing**: NumPy, pandas, scipy, scikit-image, scikit-learn
- **Visualization**: matplotlib, seaborn, opencv-python
- **Video**: imageio, imageio-ffmpeg
- **Configuration**: OmegaConf, PyYAML, attrs
- **Testing**: pytest, pytest-cov, pytest-qt, pytest-xvfb
- **Linting**: ruff (with black-compatible line length 88)
- **Documentation**: mkdocs with mkdocs-material theme

## Project Structure

```
sleap/
├── sleap/
│   ├── gui/                # PySide6-based GUI application
│   ├── io/                 # Data I/O and conversion utilities
│   ├── cli/                # Command-line interface implementations
│   ├── nn/                 # Neural network training/inference interfaces
│   ├── info/               # Inspection and diagnostic tools
│   └── legacy_cli_adaptors # Backward compatibility layer
├── tests/                  # Test suite with fixtures and data
├── docs/                   # MkDocs documentation source
├── openspec/               # OpenSpec specifications
└── .claude/                # Claude Code configuration
```

## Conventions

### Code Style
- **Formatting**: Black-compatible (88 char line length), enforced via ruff
- **Linting**: ruff with PEP8 compliance (select E, F, W rules)
- **Type hints**: Encouraged for public APIs
- **Docstrings**: Google-style format

### Development Practices
- **TDD**: Write tests first, then implementation
- **Test location**: Tests mirror source structure in `tests/` directory
- **Code coverage**: Aim for high coverage, tracked via codecov
- **Dependencies**: Minimal where possible (follow sleap-io's philosophy)

### Git Workflow
- **Base branch**: `develop` (features merge here)
- **Release branch**: `main` (stable releases)
- **Feature branches**: Use format `name/descriptive-keyword` (e.g., `elizabeth/openspec-support-agent`)
- **Commits**: Conventional commit messages preferred
- **PRs**: Draft → Ready for review when tests pass and coverage is good
- **Merge strategy**: Squash + merge into `develop`

## CLI Commands

SLEAP provides several command-line tools:

| Command | Purpose |
|---------|---------|
| `sleap-label` | Launch GUI application |
| `sleap-train` | Legacy training interface |
| `sleap-track` | Legacy tracking interface |
| `sleap-nn-train` | Training via sleap-nn backend |
| `sleap-nn-track` | Tracking via sleap-nn backend |
| `sleap-convert` | Convert between data formats |
| `sleap-render` | Visualization and rendering |
| `sleap-inspect` | Label file inspection |
| `sleap-diagnostic` | System diagnostics |

## Testing Strategy

- **Framework**: pytest with extensions (pytest-cov, pytest-qt, pytest-xvfb)
- **Test data**: Fixtures in `tests/data/` with various formats (HDF5, JSON, video)
- **Coverage**: Tracked via codecov, aim to cover edge cases
- **GUI testing**: Uses pytest-qt for Qt widget testing
- **Headless testing**: pytest-xvfb for running GUI tests without display

## Package Management with uv

SLEAP uses uv with custom index configurations for PyTorch variants:

- **CPU-only**: `extra = "nn-cpu"` → PyTorch CPU builds
- **CUDA 11.8**: `extra = "nn-cuda118"` → PyTorch CUDA 11.8
- **CUDA 12.8**: `extra = "nn-cuda128"` → PyTorch CUDA 12.8 (default)

These extras are mutually exclusive (enforced via `tool.uv.conflicts`).

## Error Handling

- Actionable error messages for non-technical users
- Validation at system boundaries (user input, file I/O)
- Propagate errors to appropriate handling layer (GUI vs CLI)

## Logging

- Use of `rich` library for terminal output
- GUI logging through Qt mechanisms
- Diagnostic output via `sleap-diagnostic` command

## Documentation

- **Primary site**: https://sleap.ai
- **Build system**: MkDocs with Material theme
- **Jupyter integration**: mkdocs-jupyter for notebook embedding
- **Versioning**: mike for multi-version docs
- **API docs**: mkdocstrings for Python API reference

## Related Repositories

- **sleap-nn**: https://github.com/talmolab/sleap-nn (PyTorch backend)
- **sleap-io**: https://github.com/talmolab/sleap-io (I/O utilities)

## Common Support Issues

### Installation
- GPU/CUDA setup issues
- Platform-specific problems (especially Windows)
- Conda vs pip vs uv confusion

### Training
- Model config questions
- GPU memory issues
- Training not converging

### Inference
- Video format compatibility
- Tracking quality issues
- Export format questions

### Cross-Repo Issues
- sleap GUI + sleap-nn backend integration
- sleap-io format compatibility with sleap and sleap-nn
- Data pipeline issues spanning multiple repos

## Community

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions (Help, Ideas, General)
- **Code of Conduct**: See [code-of-conduct.md](../docs/code-of-conduct.md)