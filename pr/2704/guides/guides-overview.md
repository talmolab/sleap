# Overview
**Here's an overview of the guides:**

!!! warning "Documentation for New SLEAP Versions"
    This documentation is for the **latest version of SLEAP**.  
    If you are using **SLEAP version 1.4.1 or earlier**, please visit the [legacy documentation](https://legacy.sleap.ai).

!!! info "Major Changes in SLEAP 1.5+"
    Want to learn about the major changes and updates in the latest release?
    See [Migrating to SLEAP 1.5+](migrating-to-sleap-1-5.md) for a summary of what's new and how to update your workflows.

    **New in v1.6:** [Label Quality Control](label-quality-control.md) for automated detection of labeling errors.

[Importing predictions for labeling](importing-predictions-for-labeling.md) when you have predictions that aren’t in the same project as your original training data and you want to correct some of the predictions and use these corrections to train a better model.

[Tracking and proofreading](tracking-and-proofreading.md) provides tips and tools you can use to speed up proofreading when you're happy enough with the frame-by-frame predictions but you need to correct the identities tracked across frames.

[Label Quality Control](label-quality-control.md) for detecting and fixing labeling errors using automated quality checks.

[Instance Size Distribution](instance-size-distribution.md) for determining the optimal crop size for top-down models.

[Run training and inference on Colab](run-training-and-inference-on-colab.md) when you have a project with labeled training data and you’d like to run training or inference in a Colab notebook.

[Creating a custom training profile](creating-a-custom-training-profile.md) for creating custom training profiles (i.e., non-default model hyperparameters) from the GUI.

[Running SLEAP remotely](running-sleap-remotely.md) when you have a project with training data and you want to train on a different machine using a command-line interface.

!!! note "Bonsai Integration"
    **Bonsai is not natively supported with the new Torch backend in SLEAP.**  
    If you want to use Bonsai with legacy SLEAP models, please refer to the [legacy Bonsai guide](https://legacy.sleap.ai/guides/bonsai.html).

