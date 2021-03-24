.. _guides:

Detailed Guides
===============

Getting better results
----------------------

:ref:`skeleton_design` answers some questions about designing the skeleton for your animals.

:ref:`choosing_models` provides information about the types of models you should.

:ref:`merging` when you have predictions that aren't in the same project as your original training data and you want to correct some of the predictions and use these corrections to train a better model.

:ref:`proofreading` provides tips and tools you can use to speed up proofreading when you're happy enough with the frame-by-frame predictions but you need to correct the identities tracked across frames.


Running remotely
-----------------

:ref:`colab` when you have a project with labeled training data and you'd like to run training or inference in a **Colab** notebook.

:ref:`remote_train` when you have a project with training data and you want to train on a different machine using a **command-line interface**.

:ref:`custom_training` for creating **custom training profiles** (i.e., non-default model hyperparameters) from the GUI.

:ref:`remote_inference` when you trained models and you want to run inference on a different machine using a **command-line interface**.

:ref:`pretrained_weights_remote` explains how to download pretrained weights if you're training a network with pretrained weights on a machine without internet access.

All Guides
----------

.. toctree::
    :maxdepth: 2
    :glob:

    *