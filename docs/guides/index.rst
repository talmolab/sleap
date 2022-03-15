.. _guides:

Guides
=======

General
-------

:ref:`gui` is a comprehensive list of the functionality in the SLEAP GUI for labeling, proofreading and inspection.

:ref:`cli` describes the command line interfaces provided by SLEAP.


Getting better results
----------------------

:ref:`skeletons` answers some questions about designing the skeleton for your animals.

:ref:`choosing_models` provides information about the types of models you should.

:ref:`merging` when you have predictions that aren't in the same project as your original training data and you want to correct some of the predictions and use these corrections to train a better model.

:ref:`proofreading` provides tips and tools you can use to speed up proofreading when you're happy enough with the frame-by-frame predictions but you need to correct the identities tracked across frames.


Running remotely
-----------------

:ref:`colab` when you have a project with labeled training data and you'd like to run training or inference in a **Colab** notebook.

:ref:`remote-train` when you have a project with training data and you want to train on a different machine using a **command-line interface**.

:ref:`custom-training` for creating **custom training profiles** (i.e., non-default model hyperparameters) from the GUI.

:ref:`remote-inference` when you trained models and you want to run inference on a different machine using a **command-line interface**.



.. toctree::
    :hidden:
    :maxdepth: 2
    
    gui
    cli
    skeletons
    choosing-models
    merging
    proofreading
    colab
    custom-training
    remote
    