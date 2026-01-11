---
hide:
  - toc
---

# Notebooks

Interactive Jupyter notebooks for hands-on learning with SLEAP. These notebooks can be run on [Google Colaboratory](https://colab.research.google.com) (Colab), which is great for running training and inference if you don't have access to a local machine with a supported GPU.

## Basic Usage

### [Training and inference on an example dataset](Training_and_inference_on_an_example_dataset.ipynb)

Learn how to install SLEAP on Colab, download a sample dataset, run training and inference, and download the predictions. A great starting point since you can see how everything works without your own data.

### [Training and inference using Google Drive](Training_and_inference_using_Google_Drive.ipynb)

Run training and inference on your own SLEAP dataset using Google Drive to copy data to and from Colab.

### [Analysis examples](Analysis_examples.ipynb)

Learn how to read and interpret the data in SLEAP's analysis HDF5 files for downstream analysis.

## Advanced Topics

### [Data structures](Data_structures.ipynb)

Explore SLEAP's core data structures for labels, predictions, and metadata. Useful for developing custom analysis scripts and applications.

### [Post-inference tracking](Post_inference_tracking.ipynb)

Re-track existing predictions to experiment with different tracking parameters without re-running pose estimation.

### [Interactive and resumable training](Interactive_and_resumable_training.ipynb)

Use SLEAP's Python API for customizable training workflows, including resumable training from existing models.

### [Interactive and realtime inference](Interactive_and_realtime_inference.ipynb)

Load trained models in Python, predict on new frames, and implement realtime SLEAP tracking for closed-loop applications.

### [Model evaluation](Model_evaluation.ipynb)

Compute benchmarking metrics for comparing trained models.

## Workshops & Tutorials

### [Cosyne 2024 Tutorial](SLEAP_Tutorial_at_Cosyne_2024_Using_exported_data.ipynb)

Tutorial notebook from the Cosyne 2024 workshop on using exported SLEAP data.

### [idTracker.ai Integration](sleap_io_idtracker_IDs.ipynb)

Combine SLEAP pose estimation with idTracker.ai identity tracking.
