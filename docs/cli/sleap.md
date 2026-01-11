# `sleap` CLI

SLEAP provides a unified command-line interface (CLI) for running operations without the graphical interface.

**What do you want to do?**

- **Launch the GUI** → [`sleap label`](#sleap-label)
- **Check your system setup** → [`sleap doctor`](#sleap-doctor)
- **Inspect a labels file** → [`sleap show`](#sleap-show)
- **Convert between formats** → [`sleap convert`](#sleap-convert)
- **Split data for training** → [`sleap split`](#sleap-split)
- **Fix video paths** → [`sleap filenames`](#sleap-filenames)
- **Render videos with poses** → [`sleap render`](#sleap-render)
- **Train or run inference** → [Training & Inference](#training-inference)
- **Use legacy commands** → [Legacy Commands](legacy.md)
- **New to the terminal?** → [Getting Started](#getting-started)

---

## Getting Started

If you've never used a terminal before, here's a quick primer.

### Opening the Terminal

=== "macOS"
    Press ++cmd+space++, type `Terminal`, press ++enter++

=== "Windows"
    Press ++win++, type `cmd`, press ++enter++

=== "Linux"
    Press ++ctrl+alt+t++

### Running Commands

Type a command and press ++enter++. For example:

```bash
sleap --version
```

### Getting Help

Add `--help` or `-h` to any command:

```bash
sleap --help           # List all commands
sleap show --help      # Help for a specific command
```

### Tips

- Press ++tab++ to auto-complete file names
- Press ++up++ to recall previous commands
- Press ++ctrl+c++ to cancel a running command

!!! note "Using `uv`?"
    If you installed SLEAP with `uv pip`, prefix commands with `uv run`:
    ```bash
    uv run sleap doctor
    ```

---

## Application Commands

### `sleap label`

Launch the SLEAP labeling GUI.

```bash
sleap label [OPTIONS] [FILE]
```

**Examples:**
```bash
sleap label                    # Open empty GUI
sleap label project.slp        # Open a project file
sleap project.slp              # Shorthand (same as above)
sleap label --reset            # Reset GUI if display issues
```

**Options:**

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Show detailed startup info |
| `--reset` | Reset GUI preferences to defaults |
| `--no-usage-data` | Disable anonymous usage data |
| `-h`, `--help` | Show help |

---

### `sleap doctor`

Show system diagnostics for troubleshooting.

```bash
sleap doctor [OPTIONS]
```

**Examples:**
```bash
sleap doctor           # Show system info
sleap doctor --json    # Output as JSON
```

Displays your Python environment, GPU status, and package versions. Copy this output when reporting issues.

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |
| `-h`, `--help` | Show help |

---

## Data Commands

These commands are powered by [sleap-io](https://io.sleap.ai). For detailed documentation and advanced options, see the [sleap-io CLI reference](https://io.sleap.ai/cli/).

### `sleap show`

Inspect labels files and videos.

```bash
sleap show <PATH> [OPTIONS]
```

**Examples:**
```bash
sleap show labels.slp              # Basic summary
sleap show labels.slp --all        # Full details
sleap show labels.slp --skeleton   # Skeleton structure
sleap show labels.slp --lf 0       # Labeled frame details
sleap show video.mp4               # Video info
```

**Options:**

| Option | Description |
|--------|-------------|
| `-s`, `--skeleton` | Show skeleton details |
| `-v`, `--video` | Show video details |
| `-t`, `--tracks` | Show track info |
| `-a`, `--all` | Show everything |
| `--lf N` | Show labeled frame N |
| `-h`, `--help` | Show help |

[Full documentation →](https://io.sleap.ai/cli/#sleap-show-inspect-labels-and-video-files)

---

### `sleap convert`

Convert between data formats.

```bash
sleap convert -i <INPUT> -o <OUTPUT> [OPTIONS]
```

**Supported formats:**

- **Input:** SLP, NWB, COCO, Label Studio, DLC, JABS, Ultralytics, LEAP
- **Output:** SLP, NWB, COCO, Label Studio, JABS, Ultralytics

**Examples:**
```bash
sleap convert -i labels.slp -o labels.nwb
sleap convert -i labels.slp -o labels.pkg.slp --embed user
sleap convert -i data.json -o labels.slp --from coco
sleap convert -i labels.slp -o dataset/ --to ultralytics
```

**Options:**

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Input file path (required) |
| `-o`, `--output` | Output file path (required) |
| `--from` | Input format (if ambiguous) |
| `--to` | Output format (if ambiguous) |
| `--embed` | Embed frames: `user`, `all`, `suggestions` |
| `-h`, `--help` | Show help |

[Full documentation →](https://io.sleap.ai/cli/#sleap-convert-convert-between-formats)

---

### `sleap split`

Split labels into train/validation/test sets.

```bash
sleap split -i <INPUT> -o <OUTPUT_DIR> [OPTIONS]
```

**Examples:**
```bash
sleap split -i labels.slp -o splits/                    # 80/20 train/val
sleap split -i labels.slp -o splits/ --train 0.7 --val 0.15 --test 0.15
sleap split -i labels.slp -o splits/ --seed 42          # Reproducible
sleap split -i labels.slp -o splits/ --remove-predictions
```

**Options:**

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Input labels file (required) |
| `-o`, `--output` | Output directory (required) |
| `--train` | Training fraction (default: 0.8) |
| `--val` | Validation fraction |
| `--test` | Test fraction |
| `--seed` | Random seed for reproducibility |
| `--remove-predictions` | Keep only user labels |
| `-h`, `--help` | Show help |

[Full documentation →](https://io.sleap.ai/cli/#sleap-split-create-trainvaltest-splits)

---

### `sleap filenames`

List or update video paths in a labels file.

```bash
sleap filenames -i <INPUT> [OPTIONS]
```

**Examples:**
```bash
sleap filenames -i labels.slp                           # List video paths
sleap filenames -i labels.slp -o out.slp --prefix /old /new
sleap filenames -i labels.slp -o out.slp --map old.mp4 new.mp4
```

**Options:**

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Input labels file (required) |
| `-o`, `--output` | Output file (for updates) |
| `--prefix OLD NEW` | Replace path prefix |
| `--map OLD NEW` | Replace specific filename |
| `-h`, `--help` | Show help |

[Full documentation →](https://io.sleap.ai/cli/#sleap-filenames-inspect-and-update-video-paths)

---

## Visualization

### `sleap render`

Render videos or images with pose overlays.

```bash
sleap render -i <INPUT> [OPTIONS]
```

**Examples:**
```bash
sleap render -i predictions.slp                    # Full video
sleap render -i predictions.slp --preset preview   # Quick preview (0.25x)
sleap render -i predictions.slp --lf 0             # Single frame as PNG
sleap render -i predictions.slp --start 100 --end 200
sleap render -i predictions.slp --color-by track
```

**Options:**

| Option | Description |
|--------|-------------|
| `-i`, `--input` | Input labels file (required) |
| `-o`, `--output` | Output path |
| `--preset` | Quality: `preview`, `draft`, `final` |
| `--lf N` | Render single frame N as PNG |
| `--start`, `--end` | Frame range |
| `--color-by` | Color by: `track`, `instance`, `node` |
| `-h`, `--help` | Show help |

[Full documentation →](https://io.sleap.ai/cli/#sleap-render-render-pose-videos-and-images)

---

## Training & Inference

!!! note "Coming to the unified CLI"
    Training and inference commands are being integrated into the `sleap` CLI.
    For now, use the options below.

### Using sleap-nn (Recommended)

The [sleap-nn](https://nn.sleap.ai) package provides PyTorch-based training and inference:

```bash
# Training
sleap-nn train --config-name baseline --config-dir /path/to/config/

# Inference
sleap-nn track --data_path video.mp4 --model_paths models/
```

- [sleap-nn training docs →](https://nn.sleap.ai/latest/training/)
- [sleap-nn inference docs →](https://nn.sleap.ai/latest/inference/)

### Using the GUI

You can also train from the GUI: **Predict → Run Training...**

See the [Training Tutorial](../tutorial/training-a-model.md) for details.

### Legacy Commands

For backwards compatibility, these commands still work:

| Command | Description |
|---------|-------------|
| `sleap-train` | Legacy training wrapper |
| `sleap-track` | Legacy inference wrapper |

See [Legacy Commands](legacy.md) for documentation.

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `sleap` | Launch GUI |
| `sleap label FILE` | Open file in GUI |
| `sleap doctor` | System diagnostics |
| `sleap show FILE` | Inspect labels file |
| `sleap convert -i IN -o OUT` | Convert formats |
| `sleap split -i IN -o DIR` | Split dataset |
| `sleap filenames -i FILE` | List video paths |
| `sleap render -i FILE` | Render with poses |
