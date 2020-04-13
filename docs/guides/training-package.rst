.. _training_package:

Export a training package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Case: You've created a project with training data on one computer, and you want to use a different computer for training models. This could be another desktop with a GPU, an HPC cluster, or a Colab notebook.*

The easiest way to move your training data to another machine is to export a **training package**. This is a single HDF5 file which contains both labeled data as well as the images which will be used for training. This makes it easy to transport your training data since you won't have to worry about paths to your video files.

To export a training package, use the "**Export Training Package...**" command in the "Predict" menu of the GUI app.

Pretty much anything you can do with a regular SLEAP file (i.e., a labels file or a predictions file), you can do with a training package file. In particular, you can:

- open a training package in the GUI (you can only see frames with labeled data, since only these are included in the training package)
- use a training package as the :code:`labels_path` parameter to the :code:`sleap-track` command-line interface