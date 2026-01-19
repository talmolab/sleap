---
hide:
  - toc
---

# SLEAP System Overview

## Understanding the SLEAP Workflow

The diagram below provides a high-level overview of the SLEAP System, highlighting its three main stages: Data, Training, and Inference.

1. Data: Import videos and labels, annotate keypoints using the sleap-label GUI, and prepare training datasets.
2. Training: Configure models, preprocess data, train using deep learning architectures, and evaluate performance.
3. Inference: Use trained models to predict poses on new videos, track subjects, and optimize inference performance.

Each module in the diagram represents a key component of the SLEAP pipeline, illustrating the complete workflow from data input to inference results.

![system-overview](../assets/images/systemOverview.png)