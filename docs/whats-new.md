# What's New in SLEAP 1.5

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
- **TensorFlow Model Inference**: Continue to support running inference on old TensorFlow models (UNet backbone only). Check [using legacy models](https://nn.sleap.ai/latest/inference/#legacy-sleap-model-support) for more details.


*For a complete list of changes, see our [Changelog](changelog.md).*
