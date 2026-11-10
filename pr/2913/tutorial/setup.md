---
hide:
  - toc
---

# 1. Setup

SLEAP uses deep neural networks to learn how to predict poses from data. Training these models is **much** faster when using a GPU[^+] for acceleration.

[^+]: Graphics processing unit. This is a hardware component that parallelizes computations across thousands of cores, making them particularly effective for the algorithms used to train deep neural networks.

If you know you have a GPU on your machine or have a [Mac with Apple Silicon](https://support.apple.com/en-us/116943), you can install SLEAP locally and follow along this tutorial.

!!! tip "sleap-nn neural-network backend"
    The SLEAP GUI can be installed and used independently of the sleap-nn backend for **labeling**. However, for this tutorial it is important that you have sleap-nn installed with the correct **PyTorch and CUDA versions** according to your machine (ex. CPU or GPU).

    To check which PyTorch and CUDA versions you should have installed, see [here](https://nn.sleap.ai/dev/installation/).

## Install SLEAP locally

**See the [main SLEAP installation instructions](../installation.md) for detailed installation instructions.**

If you have either a Linux or Windows laptop with a GPU, or a [Mac with Apple Silicon](https://support.apple.com/en-us/116943), SLEAP will work natively with hardware acceleration.

[*Next up:* Importing data](importing-data.md)