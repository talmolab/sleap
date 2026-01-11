---
hide:
  - toc
---

# Guides

These guides help you accomplish specific tasks with SLEAP. Unlike the step-by-step [Tutorial](../tutorial/index.md), each guide is self-contained and focused on a particular goal.

## Migration & Setup

!!! info "Major Changes in SLEAP 1.5+"
    Want to learn about the major changes and updates in the latest release?
    See [Migrating to SLEAP 1.5+](migrating-to-sleap-1-5.md) for a summary of what's new and how to update your workflows.

## Data & Labeling

- [Importing predictions for labeling](importing-predictions-for-labeling.md) - When you have predictions that aren't in the same project as your original training data and you want to correct some of the predictions and use these corrections to train a better model.

- [Tracking and proofreading](tracking-and-proofreading.md) - Tips and tools you can use to speed up proofreading when you're happy enough with the frame-by-frame predictions but you need to correct the identities tracked across frames.

## Training & Inference

- [Training on Colab](run-training-and-inference-on-colab.md) - When you have a project with labeled training data and you'd like to run training or inference in a Colab notebook.

- [Custom training profiles](creating-a-custom-training-profile.md) - Creating custom training profiles (i.e., non-default model hyperparameters) from the GUI.

- [Remote training](running-sleap-remotely.md) - When you have a project with training data and you want to train on a different machine using a command-line interface.

## Notebooks

Interactive Jupyter notebooks for hands-on learning and experimentation. See the [Notebooks section](../notebooks/index.md) for the full collection.

!!! note "Bonsai Integration"
    **Bonsai is not natively supported with the new Torch backend in SLEAP.**
    If you want to use Bonsai with legacy SLEAP models, please refer to the [legacy Bonsai guide](https://legacy.sleap.ai/guides/bonsai.html).
