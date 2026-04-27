# Using the GUI

The SLEAP labeling interface is launched via the `sleap label` command (or legacy `sleap-label`). See [Command Line Interfaces](../reference/command-line-interfaces.md) for details.

!!! tip "Keyboard shortcuts"
    Most menu commands have keyboard shortcuts. View and customize them via **Help → Keyboard Shortcuts**.

---

## Menus

### File

| Command | Description |
|---------|-------------|
| **New...** | Create a new project |
| **Open...** | Open an existing project |
| **Save** / **Save As...** | Save the current project |
| **Import...** | Import from external formats: [COCO] (`.json`), [DeepLabCut] (`.csv`), [DeepPoseKit] (`.h5`), [LEAP] (`.mat`) |
| **Merge Data From...** | Copy labels/predictions from another SLEAP project into the current one |
| **Add Videos...** | Add videos to the current project |
| **Replace Videos...** | Swap videos with copies at different paths (useful for moving between drives) |

### Go

| Command | Description |
|---------|-------------|
| **Next Labeled Frame** | Jump to next frame with any labels (user or predicted) |
| **Next User Labeled Frame** | Jump to next frame with user labels only |
| **Next Suggestion** | Jump to the next suggested frame |
| **Next Track Spawn Frame** | Jump to where a new track starts (useful for proofreading) |
| **Next Video** | Switch to the next video in the project |
| **Go to Frame...** | Jump to a specific frame number |
| **Select to Frame...** | Select a clip from current frame to a specified frame |

### View

| Command | Description |
|---------|-------------|
| **Color Predicted Instances** | Toggle coloring predictions by track (enable for proofreading) |
| **Color Palette** | Choose colors for instances (see note below) |
| **Apply Distinct Colors To** | Color by tracks, nodes, or edges |
| **Show Instances** | Toggle visibility of all instances |
| **Show Non-Visible Nodes** | Toggle visibility of occluded/missing nodes |
| **Show Node Names** | Toggle node name labels |
| **Show Edges** | Toggle edge visibility |
| **Edge Style** | Lines or wedges (wedges show orientation) |
| **Trail Length** | Show instance movement trails (useful for detecting swaps) |
| **Fit Instances to View** | Auto-zoom to instances in each frame |
| **Seekbar Header** | Plot information above the seekbar |
| **Crop Size Overlay** | Show the crop region for top-down training pipelines |

!!! note "Color palettes"
    - **"alphabet"** has 26 visually distinct colors
    - Palettes ending with **"+"** don't cycle (useful for proofreading—"five+" shows track 4+ as orange)
    - Customize palettes in `~/.sleap/colors.yaml`

### Labels

| Command | Description |
|---------|-------------|
| **Add Instance** | Add an instance to the current frame |
| **Instance Placement Method** | Choose how new instances are positioned ("Best" uses predictions first) |
| **Delete Instance** | Delete the selected instance |
| **Set Instance Track** | Assign selected instance to a different track |
| **Propagate Track Labels** | Apply track changes to all subsequent frames |
| **Transpose Instance Tracks** | Swap tracks between two instances |
| **Delete Instance and Track** | Delete all instances in a track across all frames |
| **Custom Instance Delete...** | Delete instances matching specific criteria |
| **Select Next Instance** | Cycle through instances in the frame |
| **Clear Selection** | Deselect the current instance |

### Predict

| Command | Description |
|---------|-------------|
| **Run Training...** | Train models from your labeled data |
| **Run Inference...** | Generate predictions using trained models |
| **Evaluate Metrics for Trained Models...** | View recall, precision, and other metrics |
| **Add Instances from All Predictions on Current Frame** | Convert all predictions to editable instances |
| **Delete All Predictions...** | Remove all predictions from current video |
| **Delete All Predictions from Clip...** | Remove predictions from selected frame range |
| **Delete All Predictions from Area...** | Remove predictions within a rectangular region |
| **Delete All Predictions with Low Score...** | Remove predictions below a score threshold |
| **Delete All Predictions beyond Frame Limit...** | Keep only top N predictions per frame |
| **Delete Predictions on User-Labeled Frames** | Remove predictions from frames with user labels |
| **Export Video with Visual Annotations...** | Export video with poses overlaid |

!!! tip "Random sample (current video)"
    When running inference, you can now choose "Random sample (current video)" to predict on a random subset of frames from just the current video.

### Analyze

| Command | Description |
|---------|-------------|
| **Instance Size Distribution...** | View bounding box size distribution (helps choose crop sizes) |
| **Label QC...** | Open the [Label Quality Control](../guides/label-quality-control.md) panel to find annotation errors |

### Help

| Command | Description |
|---------|-------------|
| **Keyboard Shortcuts** | View and customize shortcuts |
| **Check for Updates** | Check GitHub for newer SLEAP versions and shows a dialog with release notes if an update is available |

---

## Analysis Widgets

### Instance Size Distribution

Access via **Analyze → Instance Size Distribution...** or the toolbar.

This widget shows the distribution of instance bounding box sizes in your labeled data. Use it to:

- **Choose crop sizes** for top-down models—crop size should encompass most instances
- **Identify outliers** that may indicate labeling errors
- **Verify consistency** across your labeled frames

For a detailed walkthrough including rotation augmentation, statistics interpretation, and programmatic access, see [Instance Size Distribution Guide](../guides/instance-size-distribution.md).

### Crop Size Overlay

Enable via **View → Crop Size Overlay**.

Shows the crop region that will be used for top-down training pipelines, similar to the receptive field overlay. Helps verify your crop size setting captures the full animal.

### Label QC

Access via **Analyze → Label QC...**

This panel uses statistical anomaly detection to automatically find potential labeling errors in your dataset. It flags instances with unusual edge lengths, joint angles, node spacing, or other features that deviate from the norm.

Use it to:

- **Catch errors before training**—find labeling mistakes early
- **Verify proofreading**—check tracking corrections after proofreading
- **Identify systematic issues**—see patterns in labeling errors

For a complete guide including sensitivity tuning, issue types, and programmatic access, see [Label Quality Control Guide](../guides/label-quality-control.md).

---

## Mouse Controls

| Action | How |
|--------|-----|
| Zoom in/out | Mouse wheel on image |
| Pan image | Left-click + drag |
| Toggle node visibility | Right-click on node |
| Add instance | Right-click elsewhere on image |
| Zoom to region | Alt + left-click + drag |
| Zoom out | Alt + double-click |
| Move entire instance | Alt + drag on node |
| Rotate entire instance | Alt + mouse wheel on node |
| Create instance from prediction | Double-click on predicted instance |
| Add missing nodes to instance | Double-click on editable instance |
| Select instance | Click on instance |
| Clear selection | Click elsewhere |
| Duplicate instance | Ctrl + drag on instance |

!!! note "macOS"
    Substitute **Option** for **Alt** and **Command** for **Control**.

---

## Keyboard Navigation

| Key | Action |
|-----|--------|
| **→** / **←** | Move one frame forward/back |
| **Ctrl + →** / **←** | Move medium step (4 frames) |
| **Ctrl + Alt + →** / **←** | Move large step (100 frames) |
| **Home** / **End** | First / last frame |
| **Shift + navigation** | Select frames while moving |
| **1-9** | Select instance by number |
| **Ctrl (hold)** | Show tracks legend |
| **Escape** | Deselect all |

---

## Seekbar Controls

| Action | How |
|--------|-----|
| Select frame range | Shift + drag |
| Clear selection | Shift + click |
| Zoom to range | Alt + drag |
| Zoom out (show all) | Alt + click |

---

## Labeling Suggestions

Generate suggested frames for labeling via **Labeling Suggestions** panel:

| Method | Description |
|--------|-------------|
| **Sample** | Evenly spaced ("stride") or random frames from each video |
| **Image Features** | Visually distinctive frames for training diversity (slower) |
| **Prediction Score** | Frames with low-confidence predictions (for proofreading) |
| **Velocity** | Frames where instances move unusually fast (may indicate tracking errors) |

---

[coco]: http://cocodataset.org/#format-data
[deeplabcut]: http://deeplabcut.org
[deepposekit]: http://deepposekit.org
[leap]: https://github.com/talmo/leap
