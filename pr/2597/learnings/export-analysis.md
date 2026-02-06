# Export for Analysis

!!! note "New to exporting?"
    If you're just getting started, see the [Exporting the Results](../tutorial/exporting-the-results.md) tutorial for a guided introduction.

SLEAP provides multiple ways to export your pose tracking data for downstream analysis.

!!! tip "Comprehensive format documentation"
    For detailed format specifications, all supported formats, and programmatic access, see the [sleap-io formats documentation](https://io.sleap.ai/latest/formats/).

---

## Quick Export from GUI

The easiest way to export data is through the **File** menu:

| Menu Option | Output Format | Best For |
|-------------|---------------|----------|
| **Export Analysis HDF5...** | `.h5` | MATLAB, NumPy arrays |
| **Export Analysis CSV...** | `.csv` | Spreadsheets, pandas |
| **Export NWB...** | `.nwb` | Neuroscience data sharing |

---

## Command-Line Export

For batch processing or scripting, use the `sleap export` command for analysis-ready outputs:

```bash
# Export to CSV (dense, with empty frames padded)
sleap export predictions.slp -o analysis.csv

# Export to Analysis HDF5
sleap export predictions.slp -o analysis.h5

# Export only frames with instances (sparse)
sleap export predictions.slp -o sparse.csv --no-empty-frames

# Export specific video from multi-video file
sleap export multi.slp -o video0.csv -v 0

# Export all videos from multi-video file
sleap export multi.slp -o batch.csv -v all
# Creates: batch.video0.csv, batch.video1.csv, ...

# Memory-efficient chunked export for large files
sleap export large.slp -o analysis.csv --chunk-size 10000
```

See `sleap export --help` or the [sleap-io CLI documentation](https://io.sleap.ai/latest/cli/#sio-export) for all options.

## Command-Line Conversion

For converting between different label formats, use the `sleap convert` command:

```bash
# Convert to NWB (Neurodata Without Borders)
sleap convert predictions.slp -o data.nwb

# Convert to COCO format
sleap convert predictions.slp -o annotations.json --to coco

# Create a portable package with embedded frames
sleap convert labels.slp -o labels.pkg.slp --embed user

# Export to Ultralytics YOLO format
sleap convert labels.slp -o yolo_dataset/ --to ultralytics
```

See `sleap convert --help` or the [sleap-io CLI documentation](https://io.sleap.ai/latest/cli/#sio-convert) for all options.

---

## Analysis HDF5 Format

The Analysis HDF5 format exports pose data as dense NumPy arrays, optimized for MATLAB and Python.

### Reading in MATLAB

```matlab
tracks = h5read('analysis.h5', '/tracks');
occupancy = h5read('analysis.h5', '/track_occupancy');
node_names = h5read('analysis.h5', '/node_names');

% Get coordinates for track 1, node 1, frame 100
x = tracks(1, 1, 1, 100);
y = tracks(1, 2, 1, 100);
```

### Reading in Python

```python
import h5py

with h5py.File('analysis.h5', 'r') as f:
    tracks = f['tracks'][:]             # (n_tracks, 2, n_nodes, n_frames)
    occupancy = f['track_occupancy'][:]  # (n_frames, n_tracks)
    node_names = [n.decode() for n in f['node_names'][:]]

print(f"Tracks shape: {tracks.shape}")
print(f"Nodes: {node_names}")
```

For dataset schemas, axis ordering presets, and advanced options, see the [sleap-io Analysis HDF5 documentation](https://io.sleap.ai/latest/formats/#sleap-analysis-hdf5-format-h5).

---

## Supported Formats

| Format | Extension | CLI | Python API | Documentation |
|--------|-----------|:---:|:----------:|---------------|
| **Analysis HDF5** | `.h5` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#sleap-analysis-hdf5-format-h5) |
| **CSV** | `.csv` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#csv-format-csv) |
| **NWB** | `.nwb` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#nwb-format-nwb) |
| **COCO** | `.json` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#coco-format-json) |
| **Label Studio** | `.json` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#label-studio-format-json) |
| **Ultralytics** | directory | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#ultralytics-yolo-format) |
| **JABS** | `.h5` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#jabs-format-h5) |
