# Migrating to SLEAP 1.5

SLEAP 1.5 represents a major milestone with significant architectural improvements, performance enhancements, and new installation methods. Here are the key changes:

## Major Changes

### UV-Based Installation
SLEAP 1.5+ now uses [**uv**](https://docs.astral.sh/uv/) for installation, making it much faster than previous methods. Get up and running in seconds with our streamlined installation process.

### PyTorch Backend
Neural network backend switched from TensorFlow to PyTorch, providing:

- **Much faster training and inference speeds**
- **Modern deep learning capabilities**
- **Improved developer experience**
- **Multi-GPU training**

### Standalone Libraries
SLEAP GUI is now supported by two new packages for modular workflows:

#### [SLEAP-IO](https://io.sleap.ai)
I/O backend for handling labels, processing `.slp` files, and data manipulation. Essential for any SLEAP workflow and can be used independently for data processing tasks.

#### [SLEAP-NN](https://nn.sleap.ai)
PyTorch-based neural network backend for training and inference. Perfect for custom training pipelines, remote processing, and headless server deployments.

## Torch Backend Changes

### New Backbones
SLEAP 1.5 introduces three powerful new backbone architectures (check [here](https://nn.sleap.ai/latest/models/#backbone-architectures) for more details):

- **UNet** - Classic encoder-decoder architecture for precise pose estimation
- **SwinT** - Swin Transformer for state-of-the-art performance
- **ConvNeXt** - Modern convolutional architecture with improved efficiency

### Legacy Support
We've maintained full backward compatibility:

- **GUI Support**: SLEAP now uses a new <u>YAML-based</u> config file structure, but you can still upload and work with old SLEAP JSON files in the GUI. For details on converting legacy SLEAP 1.4 config/JSON files to the new YAML format, see our [conversion guide](https://nn.sleap.ai/latest/config/#converting-legacy-sleap-14-configjson-to-sleap-nn-yaml).
- **Using TensorFlow Model Weights**: Continue to support running inference on SLEAP <1.4 TensorFlow model weights (UNet backbone only). Check [using legacy models](https://nn.sleap.ai/latest/inference/#legacy-sleap-model-support) for more details.


## What's New in v1.6

SLEAP 1.6 builds on the v1.5 foundation with additional features and improvements:

### Unified CLI
The new `sleap` command provides a single entry point for all SLEAP functionality:

```bash
sleap                    # Launch the GUI
sleap doctor             # System diagnostics
sleap train ...          # Training (via sleap-nn)
sleap track ...          # Inference (via sleap-nn)
sleap show labels.slp    # View labels summary (via sleap-io)
```

See [Command Line Interfaces](../reference/command-line-interfaces.md) for the full list of commands.

### Label Quality Control
New **Analyze → Label QC...** menu for automated detection of labeling errors including:
- Temporal jitter (unstable predictions across frames)
- Visibility inconsistencies
- Scale anomalies
- Potential identity swaps

See [Label Quality Control](label-quality-control.md) for details.

### Instance Size Distribution
New **Analyze → Instance Size Distribution...** widget helps determine optimal crop sizes for top-down models by analyzing the distribution of instance bounding box sizes in your labeled data.

See [Instance Size Distribution](instance-size-distribution.md) for details.

### ONNX/TensorRT Export
Export trained models to ONNX or TensorRT format for 3-6x faster inference:

```bash
sleap export-model models/my_model --format onnx
```

### Additional Improvements
- **Real-time inference progress**: Live FPS and ETA display during inference
- **Filter overlapping instances**: New controls in training/inference dialogs
- **Crop size visualization**: Overlay showing crop region for top-down models

*For a complete list of changes, see our [Changelog](../changelog.md).*
