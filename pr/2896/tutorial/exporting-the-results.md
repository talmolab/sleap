# 8. Exporting the results

SLEAP stores labeled data, predictions, and all metadata in `.slp` files. These files use a custom format that is optimized for SLEAP workflows and contain metadata, but ultimately are just HDF5 files that implement our [data model](../notebooks/Data_structures.ipynb).

These `.slp` files are not intended for final use in analysis since they require SLEAP to be parsed appropriately.

Once you're ready to analyze your results, you have several options for different file formats for export. While these do not contain all the metadata used during the labeling and training stage, they are more convenient for analysis and portable for use in different downstream software and libraries.

We recommend exporting your proofread predictions to each of these formats so you can explore your data in the next step of the tutorial.

!!! tip "Comprehensive format documentation"
    For detailed format specifications and programmatic access, see the [sleap-io formats documentation](https://io.sleap.ai/latest/formats/).

---

## Quick Export from GUI

The easiest way to export is through the **File** menu:

| Menu Option | Output Format | Best For |
|-------------|---------------|----------|
| **Export Analysis HDF5...** | `.h5` | MATLAB, NumPy arrays |
| **Export Analysis CSV...** | `.csv` | Spreadsheets, pandas |
| **Export NWB...** | `.nwb` | Neuroscience data sharing |

---

## Export Formats

### NWB

The <a href="https://www.nwb.org/" target="_blank">Neurodata Without Borders (NWB)</a> format provides a data standard for neural and behavioral data.

We strongly recommend NWB for its portability, standardization, and compatibility with the scientific ecosystem. NWB data can be hosted for free in the <a href="https://www.dandiarchive.org/" target="_blank">DANDI Archive</a>.

SLEAP <a href="https://nwb-overview.readthedocs.io/en/latest/tools/sleap/sleap.html" target="_blank">natively supports NWB</a> through the <a href="https://github.com/rly/ndx-pose" target="_blank">ndx-pose extension</a>.

**GUI:** **File** → **Export NWB...**

**CLI:**
```bash
sleap convert predictions.slp -o data.nwb
```

For reading NWB files, see the <a href="https://nwb-overview.readthedocs.io/en/latest/file_read/file_read.html" target="_blank">NWB reading guide</a>.

### Analysis HDF5

The <a href="https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html" target="_blank">HDF5</a> format stores pose data as dense NumPy arrays, ideal for MATLAB and Python.

**GUI:** **File** → **Export Analysis HDF5...** → **Current Video...**

**CLI:**
```bash
sleap export predictions.slp -o analysis.h5
```

#### Reading in MATLAB

```matlab
tracks = h5read('analysis.h5', '/tracks');
occupancy = h5read('analysis.h5', '/track_occupancy');
node_names = h5read('analysis.h5', '/node_names');

% Get coordinates for track 1, node 1, frame 100
x = tracks(1, 1, 1, 100);
y = tracks(1, 2, 1, 100);
```

#### Reading in Python

```python
import h5py

with h5py.File('analysis.h5', 'r') as f:
    tracks = f['tracks'][:]             # (n_tracks, 2, n_nodes, n_frames)
    occupancy = f['track_occupancy'][:]  # (n_frames, n_tracks)
    node_names = [n.decode() for n in f['node_names'][:]]

print(f"Tracks shape: {tracks.shape}")
print(f"Nodes: {node_names}")
```

For dataset schemas and axis ordering, see the [sleap-io Analysis HDF5 documentation](https://io.sleap.ai/latest/formats/#sleap-analysis-hdf5-format-h5).

### CSV

Comma-separated value (CSV) files are the simplest format for tabular data, compatible with Excel, Google Sheets, and pandas.

**GUI:** **File** → **Export Analysis CSV...** → **Current Video...**

**CLI:**
```bash
sleap export predictions.slp -o analysis.csv
```

---

## Command-Line Export

For batch processing or scripting, use the `sleap export` command:

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

---

## Command-Line Conversion

For converting between label formats, use `sleap convert`:

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

## All Supported Formats

| Format | Extension | CLI | Python API | Documentation |
|--------|-----------|:---:|:----------:|---------------|
| **Analysis HDF5** | `.h5` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#sleap-analysis-hdf5-format-h5) |
| **CSV** | `.csv` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#csv-format-csv) |
| **NWB** | `.nwb` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#nwb-format-nwb) |
| **COCO** | `.json` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#coco-format-json) |
| **Label Studio** | `.json` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#label-studio-format-json) |
| **Ultralytics** | directory | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#ultralytics-yolo-format) |
| **JABS** | `.h5` | ✅ | ✅ | [sleap-io docs](https://io.sleap.ai/latest/formats/#jabs-format-h5) |

---

## sleap-io Python API

For programmatic export and conversion, use <a href="https://io.sleap.ai" target="_blank">sleap-io</a>:

```python
import sleap_io as sio

# Load labels
labels = sio.load_file("predictions.slp")

# Export to different formats
sio.save_file(labels, "output.nwb")
sio.save_file(labels, "output.csv")
```

See the <a href="https://io.sleap.ai/latest/" target="_blank">sleap-io documentation</a> for full API details.

---

You did it!

[*Next up:* I'm done SLEAPing, now what?](i-m-done-sleaping-now-what.md)
