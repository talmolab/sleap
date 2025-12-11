# SLEAP Project Constitution

## Mission

SLEAP (Social LEAP Estimates Animal Poses) enables researchers to track animal behavior through deep learning-based pose estimation. We prioritize accessibility for scientists who may not have ML expertise.

## The SLEAP Ecosystem

| Repository | Purpose | Primary Users |
|------------|---------|---------------|
| [sleap](https://github.com/talmolab/sleap) | GUI application for labeling, training, and inference | End users, researchers |
| [sleap-nn](https://github.com/talmolab/sleap-nn) | PyTorch neural network backend | Developers, advanced users |
| [sleap-io](https://github.com/talmolab/sleap-io) | Lightweight I/O utilities for pose data | Developers, analysts |

## Core Principles

### 1. User-First Design
- Our users are scientists, not software engineers
- Error messages should be actionable, not cryptic
- Documentation is as important as code

### 2. Reproducibility
- Research must be reproducible
- Model configs and training parameters should be explicit
- Data formats should be well-documented and stable

### 3. Accessibility
- Support multiple platforms (Linux, macOS, Windows)
- Support multiple hardware configs (GPU, Apple Silicon, CPU-only)
- Minimize dependencies where possible (see: sleap-io's design)

### 4. Community Support
- Respond to issues and discussions promptly
- Distinguish between bugs, feature requests, and support questions
- Point users to existing documentation before writing custom responses

## Code Standards

### Python
- Type hints for public APIs
- Google-style docstrings
- TDD: write tests first, then implementation
- Format with `black` and lint with `ruff`

### Package Management
- Use `uv` for dependency management
- Follow uv best practices for lockfiles and environments

### Git
- Conventional commit messages preferred
- PRs should reference issues when applicable
- Keep PRs focused; split large changes

## Support Philosophy

- Many "bugs" are actually usage questions - redirect kindly to docs
- Installation issues are the most common - we have guides for this
- Cross-repo issues (sleap + sleap-nn) require careful triage