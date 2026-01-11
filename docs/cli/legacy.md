# Legacy Commands

These commands are maintained for **backwards compatibility** with existing scripts and pipelines. For new projects, we recommend using the unified `sleap` CLI or `sleap-nn` commands.

## Migration Quick Reference

| Legacy Command | Modern Equivalent | Notes |
|----------------|-------------------|-------|
| `sleap-label` | `sleap label` | Identical functionality |
| `sleap-label file.slp` | `sleap file.slp` | Direct file opening |
| `sleap-inspect` | `sleap show` | Enhanced with rich formatting |
| `sleap-diagnostic` | `sleap doctor` | Enhanced with rich formatting |
| `sleap-convert` | `sleap convert` | Different syntax (see below) |
| `sleap-render` | `sleap render` | Different syntax (see below) |
| `sleap-train` | `sleap-nn train` | PyTorch backend |
| `sleap-track` | `sleap-nn track` | PyTorch backend |

---

## sleap-label

!!! info "Modern equivalent: `sleap label`"

Launches the SLEAP labeling GUI.

```bash
sleap-label [labels_path] [--nonnative] [--profiling] [--reset]
```

**Migration:** Simply replace `sleap-label` with `sleap label` or just `sleap`.

---

## sleap-inspect

!!! info "Modern equivalent: `sleap show`"

Displays information about a labels file or model folder.

```bash
sleap-inspect [--verbose] data_path
```

**Migration:**
```bash
# Old
sleap-inspect labels.slp --verbose

# New (with enhanced output)
sleap show labels.slp --all
```

---

## sleap-diagnostic

!!! info "Modern equivalent: `sleap doctor`"

Shows system diagnostic information.

```bash
sleap-diagnostic [-o OUTPUT] [--gui-check]
```

**Migration:**
```bash
# Old
sleap-diagnostic

# New (with enhanced output)
sleap doctor
sleap doctor --json  # For programmatic use
```

---

## sleap-convert

!!! info "Modern equivalent: `sleap convert`"

Converts between dataset file formats.

```bash
sleap-convert input_path [-o OUTPUT] [--format FORMAT] [--video VIDEO]
```

**Key differences:**

| sleap-convert | sleap convert |
|---------------|---------------|
| Positional input | `-i/--input` flag |
| `--format analysis` | `-o file.analysis.h5` |
| Limited formats | More formats supported |

**Migration:**
```bash
# Old: Convert to analysis HDF5
sleap-convert labels.slp --format analysis -o output.h5

# New: Format inferred from extension
sleap convert -i labels.slp -o output.analysis.h5
```

---

## sleap-render

!!! info "Modern equivalent: `sleap render`"

Renders videos with pose overlays.

```bash
sleap-render data_path [-o OUTPUT] [-f FPS] [--scale SCALE] [--frames FRAMES]
```

**Key differences:**

| sleap-render | sleap render |
|--------------|--------------|
| Limited options | Many rendering options |
| Basic output | Presets (preview, draft, final) |
| Video only | Video or single frame PNG |

**Migration:**
```bash
# Old
sleap-render labels.slp -o output.mp4 --frames 0-100

# New (with more options)
sleap render -i labels.slp -o output.mp4 --start 0 --end 100
sleap render -i labels.slp --lf 0  # Single frame as PNG
sleap render -i labels.slp --preset preview  # Quick preview
```

---

## sleap-train

!!! info "Modern equivalent: `sleap-nn train`"

Trains pose estimation models.

```bash
sleap-train training_job_path [labels_path] [OPTIONS]
```

See [sleap-nn training documentation](https://nn.sleap.ai/latest/training/) for the modern approach using Hydra configuration.

**Migration:**
```bash
# Old (JSON config)
sleap-train config.json labels.slp

# New (Hydra config)
sleap-nn train --config-name baseline --config-dir /path/to/config/
```

!!! warning
    Training configurations are different between the legacy and modern CLIs. You may need to recreate your training profile. See [Creating a Custom Training Profile](../guides/creating-a-custom-training-profile.md).

---

## sleap-track

!!! info "Modern equivalent: `sleap-nn track`"

Runs inference on videos using trained models.

```bash
sleap-track data_path [-m MODELS] [--frames FRAMES] [-o OUTPUT] [OPTIONS]
```

**Migration:**
```bash
# Old
sleap-track video.mp4 -m models/model1 -m models/model2 -o predictions.slp

# New
sleap-nn track --data_path video.mp4 --model_paths models/model1 --model_paths models/model2 -o predictions.slp
```

See [sleap-nn inference documentation](https://nn.sleap.ai/latest/inference/) for full options.

---

## Full Legacy Documentation

For complete documentation of legacy command options, see the archived documentation:

- [sleap-train options](https://legacy.sleap.ai/guides/cli.html#sleap-train)
- [sleap-track options](https://legacy.sleap.ai/guides/cli.html#sleap-track)
- [sleap-convert options](https://legacy.sleap.ai/guides/cli.html#sleap-convert)

---

## Deprecation Timeline

These legacy commands will continue to work for the foreseeable future. We have no plans to remove them, as we understand many users have existing scripts and workflows that depend on them.

However, new features will only be added to the modern equivalents (`sleap` unified CLI and `sleap-nn`).
