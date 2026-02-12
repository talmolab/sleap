# Command Line Reference

SLEAP provides a unified command-line interface through the `sleap` command, with subcommands for different tasks.

```bash
sleap [COMMAND] [OPTIONS]
```

Run `sleap --help` to see all available commands, or `sleap <command> --help` for details on a specific command.

---

## Quick Reference

**GUI & Diagnostics**

| Command | Description |
|---------|-------------|
| [`sleap`](#sleap-label) | Launch the GUI (same as `sleap label`) |
| [`sleap label`](#sleap-label) | Launch the SLEAP labeling GUI |
| [`sleap doctor`](#sleap-doctor) | Show system diagnostics for troubleshooting |

**Neural Network** (requires `sleap[nn]`)

| Command | Description |
|---------|-------------|
| [`sleap train`](#sleap-train) | Train pose estimation models |
| [`sleap track`](#sleap-track) | Run inference and tracking |
| [`sleap eval`](#other-neural-network-commands) | Evaluate predictions against ground truth |
| [`sleap export-model`](#other-neural-network-commands) | Export model for deployment |
| [`sleap predict`](#other-neural-network-commands) | Run inference with exported model |
| [`sleap system`](#other-neural-network-commands) | Show sleap-nn system information |

**Data Commands** (from sleap-io)

| Command | Description |
|---------|-------------|
| [`sleap convert`](#sleap-convert) | Convert between pose data formats |
| [`sleap export`](#sleap-export) | Export pose data for analysis (CSV, HDF5) |
| [`sleap show`](#sleap-show) | Print labels file summary |
| [`sleap render`](#sleap-render) | Render predictions as video or image |
| [`sleap merge`](#sleap-merge) | Merge multiple labels files |
| [`sleap split`](#sleap-split) | Split labels into train/val/test sets |
| [`sleap embed`](#other-data-commands) | Embed video frames into labels file |
| [`sleap unembed`](#other-data-commands) | Remove embedded frames |
| [`sleap filenames`](#other-data-commands) | List or update video filenames |
| [`sleap fix`](#other-data-commands) | Fix common issues in labels files |
| [`sleap trim`](#other-data-commands) | Trim video and labels to frame range |
| [`sleap reencode`](#other-data-commands) | Reencode video for seekability |
| [`sleap transform`](#other-data-commands) | Transform video and adjust coordinates |
| [`sleap unsplit`](#other-data-commands) | Merge split files back into one |

---

## GUI

### `sleap label`

Launch the SLEAP labeling GUI. You can also just run `sleap` or `sleap <file.slp>`.

```bash
sleap label [OPTIONS] [LABELS.slp]

# Examples
sleap                        # Launch empty GUI
sleap label                  # Same as above
sleap my_project.slp         # Open existing project
sleap label --reset          # Reset GUI preferences
sleap --video-backend ffmpeg # Use imageio-ffmpeg for video
```

**Options:**

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed startup info including GPU status |
| `--reset` | Reset GUI preferences to defaults |
| `--no-usage-data` | Disable anonymous usage data collection |
| `--nonnative` | Use non-native file dialogs |
| `--video-backend` | Video backend plugin: `opencv`, `FFMPEG`, or `pyav`. Persists to preferences on use. |

---

## Diagnostics

### `sleap doctor`

Show system diagnostics for troubleshooting. Output is designed to be copy-pasted when reporting issues.

```bash
sleap doctor [OPTIONS]

# Examples
sleap doctor              # Show diagnostics
sleap doctor --json       # Output as JSON
sleap doctor -o report.txt  # Save to file
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON for programmatic use |
| `-o, --output` | Save output to file |

---

## Training & Inference

Training and inference are powered by **sleap-nn**, the PyTorch-based neural network package.

!!! note
    These commands require `sleap[nn]` to be installed. They wrap the sleap-nn CLI.

### `sleap train`

Train pose estimation models.

```bash
sleap train [OPTIONS] <config.yaml>

# Examples
sleap train config.yaml                    # Train with config file
sleap train --config configs/baseline.yaml
```

[:octicons-arrow-right-24: Full training documentation](https://nn.sleap.ai/latest/training/)

### `sleap track`

Run inference and tracking on videos.

```bash
sleap track [OPTIONS] -i <video> -m <model>

# Examples
sleap track -i video.mp4 -m models/centroid/
sleap track -i video.mp4 -m models/centroid/ -m models/instance/ --tracking
```

[:octicons-arrow-right-24: Full inference documentation](https://nn.sleap.ai/latest/inference/)

### Other Neural Network Commands

| Command | Description |
|---------|-------------|
| `sleap eval` | Evaluate model predictions against ground truth |
| `sleap export-model` | Export model to ONNX for deployment |
| `sleap predict` | Run inference with exported ONNX model |
| `sleap system` | Show sleap-nn system information |

[:octicons-arrow-right-24: Exporting models to ONNX](https://nn.sleap.ai/latest/guides/export/)

[:octicons-arrow-right-24: Full sleap-nn CLI documentation](https://nn.sleap.ai/latest/reference/cli/)

---

## Data Commands

These commands are provided by **sleap-io** for working with labels files.

[:octicons-arrow-right-24: Full sleap-io CLI documentation](https://io.sleap.ai/latest/cli/)

### `sleap convert`

Convert between pose data formats.

```bash
sleap convert [OPTIONS] <input> -o <output>

# Examples
sleap convert labels.slp -o labels.nwb           # SLP to NWB
sleap convert labels.slp -o labels.pkg.slp --embed user  # Embed frames
sleap convert annotations.json -o labels.slp --from coco  # COCO to SLP
```

**Supported formats:**

- **Input:** slp, nwb, coco, labelstudio, alphatracker, jabs, dlc, ultralytics, leap
- **Output:** slp, nwb, coco, labelstudio, jabs, ultralytics, csv

### `sleap export`

Export pose data to analysis-ready formats (CSV, HDF5).

Unlike `convert` which transforms between label formats, `export` creates dense outputs optimized for analysis with full control over frame padding, video selection, and output structure.

```bash
sleap export [OPTIONS] <input> -o <output>

# Examples
sleap export predictions.slp -o analysis.csv      # Export as CSV
sleap export predictions.slp -o analysis.h5       # Export as HDF5
sleap export labels.slp -o sparse.csv --no-empty-frames  # Only frames with instances
sleap export multi.slp -o export.csv -v 0         # Export specific video
sleap export multi.slp -o export.csv -v all       # Export all videos
sleap export large.slp -o data.csv --chunk-size 10000  # Memory-efficient
```

**Key options:**

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path (required) |
| `--format csv\|h5` | Output format (inferred from extension if not specified) |
| `--csv-format` | CSV layout: `sleap`, `dlc`, `points`, `instances`, `frames` (default) |
| `--h5-dim-order` | HDF5 axis ordering: `matlab` (SLEAP-compatible) or `standard` |
| `-v, --video` | Video index (0, 1, ...) or `all` for multi-video files |
| `--start, --end` | Frame range (inclusive start, exclusive end) |
| `--no-empty-frames` | Only include frames with instances (sparse output) |
| `--chunk-size` | Write CSV in chunks for large files |

[:octicons-arrow-right-24: Full export documentation](https://io.sleap.ai/latest/cli/#sio-export)

### `sleap show`

Print a summary of a labels file.

```bash
sleap show labels.slp
```

### `sleap render`

Render pose predictions as video or image.

```bash
sleap render [OPTIONS] <labels.slp> -o <output.mp4>

# Examples
sleap render predictions.slp -o video.mp4
sleap render predictions.slp -o frame.png --frames 100
```

### `sleap merge`

Merge multiple labels files into one.

```bash
sleap merge file1.slp file2.slp -o merged.slp
```

### `sleap split`

Split labels into train/validation/test sets.

```bash
sleap split labels.slp -o output_dir/ --val 0.1 --test 0.1
```

### Other Data Commands

| Command | Description |
|---------|-------------|
| `sleap embed` | Embed video frames into a labels file |
| `sleap unembed` | Remove embedded frames, restore video references |
| `sleap filenames` | List or update video filenames |
| `sleap fix` | Fix common issues in labels files |
| `sleap trim` | Trim video and labels to a frame range |
| `sleap transform` | Transform video and adjust coordinates |
| `sleap reencode` | Reencode video for improved seekability |
| `sleap unsplit` | Merge split files back into one |

---

## Legacy CLI

The following commands are provided for backwards compatibility. **For new projects, use the unified `sleap` commands above.**

<details class="plain" markdown>
<summary>sleap-label</summary>

Legacy entry point for the GUI. **Use `sleap label` instead.**

```none
usage: sleap-label [-h] [--nonnative] [--profiling] [--reset] [labels_path]

positional arguments:
  labels_path  Path to labels file

optional arguments:
  -h, --help   show this help message and exit
  --nonnative  Don't use native file dialogs
  --profiling  Enable performance profiling
  --reset      Reset GUI state and preferences
```

</details>

<details class="plain" markdown>
<summary>sleap-train</summary>

Legacy training command. **Use `sleap train` for new projects.**

```none
usage: sleap-train [-h] [--video-paths VIDEO_PATHS] [--val_labels VAL_LABELS]
                   [--test_labels TEST_LABELS] [--save_viz] [--keep_viz] [--zmq]
                   [--run_name RUN_NAME] [--prefix PREFIX] [--suffix SUFFIX]
                   [--cpu] [--first-gpu] [--last-gpu] [--gpu GPU]
                   training_job_path [labels_path]

positional arguments:
  training_job_path     Path to training job profile JSON/YAML file
  labels_path           Path to labels file for training

optional arguments:
  --video-paths         List of paths for finding videos
  --val_labels, --val   Path to validation labels file
  --test_labels, --test Path to test labels file
  --base_checkpoint     Path to checkpoint to resume from
  --save_viz            Save prediction visualizations
  --keep_viz            Keep visualization images after training
  --zmq                 Enable ZMQ logging (for GUI)
  --run_name            Run name for saving files
  --prefix              Prefix to prepend to run name
  --suffix              Suffix to append to run name
  --cpu                 Run on CPU only
  --first-gpu           Run on first GPU
  --last-gpu            Run on last GPU
  --gpu GPU             Run on specific GPU (or 'auto')
```

</details>

<details class="plain" markdown>
<summary>sleap-track</summary>

Legacy inference command. **Use `sleap track` for new projects.**

```none
usage: sleap-track [-h] [-m MODELS] [--frames FRAMES] [-o OUTPUT] [--batch_size BATCH_SIZE]
                   [--peak_threshold PEAK_THRESHOLD] [-n MAX_INSTANCES]
                   [--tracking.tracker TRACKER] [--tracking.max_tracks MAX_TRACKS]
                   [--cpu | --first-gpu | --last-gpu | --gpu GPU]
                   [data_path]

positional arguments:
  data_path             Path to video, .slp file, or folder of videos

optional arguments:
  -m, --model           Path to trained model directory (can specify multiple)
  --frames              Frames to predict (e.g., "1,2,3" or "1-100")
  --only-labeled-frames Only predict on labeled frames
  --only-suggested-frames Only predict on suggested frames
  -o, --output          Output filename (default: [data_path].predictions.slp)
  --no-empty-frames     Clear empty frames before saving
  --batch_size          Frames per batch (default: 4)
  --peak_threshold      Minimum confidence threshold
  -n, --max_instances   Limit instances per frame
  --open-in-gui         Open results in GUI when done

Tracking options:
  --tracking.tracker    Tracker type: simple, flow, simplemaxtracks, flowmaxtracks
  --tracking.max_tracks Maximum number of tracks
  --tracking.track_window Frames to look back for matches (default: 5)
  --tracking.similarity Similarity metric: instance, centroid, iou
  --tracking.match      Matching algorithm: hungarian, greedy

GPU options:
  --cpu                 Run on CPU only
  --first-gpu           Run on first GPU
  --last-gpu            Run on last GPU
  --gpu GPU             Run on specific GPU (or 'auto')
```

**Examples:**

```bash
# Simple inference
sleap-track -m "models/my_model" -o "predictions.slp" "video.mp4"

# Multi-model pipeline (top-down)
sleap-track -m "models/centroid" -m "models/instance" -o "predictions.slp" "video.mp4"

# With tracking
sleap-track -m "models/my_model" --tracking.tracker simple -o "predictions.slp" "video.mp4"

# Re-track existing predictions
sleap-track --tracking.tracker simple -o "retracked.slp" "predictions.slp"
```

</details>

<details class="plain" markdown>
<summary>sleap-nn-train / sleap-nn-track</summary>

Direct sleap-nn entry points. **Use `sleap train` and `sleap track` for new projects.**

These commands provide the same functionality as `sleap train` and `sleap track` but bypass the unified CLI wrapper.

**sleap-nn-train:**
```bash
sleap-nn-train --config <config.yaml> [overrides]

# Examples
sleap-nn-train --config /path/to/config/baseline.yaml
sleap-nn-train --config baseline.yaml trainer_config.max_epochs=100
```

**sleap-nn-track:**
```bash
sleap-nn-track --data_path <video> --model_paths <model> [options]

# Examples
sleap-nn-track -i video.mp4 -m models/ckpt_folder/
sleap-nn-track --data_path video.mp4 --model_paths models/ --tracking
```

See [sleap-nn documentation](https://nn.sleap.ai) for full details.

</details>

<details class="plain" markdown>
<summary>sleap-convert (legacy)</summary>

Legacy conversion command. **Use `sleap convert` instead.**

```none
usage: sleap-convert [-h] [-o OUTPUT] [--format FORMAT] [--video VIDEO] input_path

positional arguments:
  input_path    Path to input file

optional arguments:
  -o, --output  Path to output file
  --format      Output format: slp, analysis, analysis.nix, analysis.csv, h5, json
  --video       Path to video (if needed)
```

**Example:**
```bash
sleap-convert --format analysis -o "predictions.analysis.h5" "predictions.slp"
```

</details>

<details class="plain" markdown>
<summary>sleap-inspect</summary>

Legacy inspection command. **Use `sleap show` instead.**

```none
usage: sleap-inspect [-h] [--verbose] data_path

positional arguments:
  data_path   Path to labels file (.slp) or model folder

optional arguments:
  --verbose   Show detailed information
```

</details>

<details class="plain" markdown>
<summary>sleap-render (legacy)</summary>

Legacy render command. **Use `sleap render` instead.**

```none
usage: sleap-render [-h] [-o OUTPUT] [-f FPS] [--scale SCALE] [--frames FRAMES]
                    [--video-index VIDEO_INDEX] [--palette PALETTE] data_path

positional arguments:
  data_path             Path to labels file

optional arguments:
  -o, --output          Output path
  --video-index         Video index in labels (default: 0)
  --frames              Frames to render (e.g., "1,2,3" or "1-100")
  -f, --fps             Output FPS (default: 25)
  --scale               Image scale (default: 1.0)
  --show_edges          Draw lines between nodes (default: 1)
  --marker_size         Marker size in pixels (default: 4)
  --palette             Color palette: alphabet, five+, solarized, standard
  --distinctly_color    Color by: instances, edges, nodes
```

</details>

<details class="plain" markdown>
<summary>sleap-diagnostic</summary>

Legacy diagnostics command. **Use `sleap doctor` instead.**

```none
usage: sleap-diagnostic [-h] [-o OUTPUT] [--gui-check]

optional arguments:
  -o, --output   Path for saving output
  --gui-check    Check if Qt GUI widgets can be used
```

</details>

---

!!! note
    For help with any command, run with `--help` (e.g., `sleap convert --help`).
