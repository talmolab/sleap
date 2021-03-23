.. _training_job_package:

Export a training job package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Case: You've created a project with training data on one computer, and you want to use a different computer for training models. This could be another desktop with a GPU, an HPC cluster, or a Colab notebook.*

The easiest way to move your training data to another machine is to export a **training job package**. This is a single ZIP file containing the training configuration files, scripts and the HDF5 file with both labeled data as well as the images which will be used for training. This makes it easy to transport your training data since you won't have to worry about paths to your video files.

To export a training job package, use the "**Export Training Job Package...**"  in the "Predict" menu of the GUI app.

Pretty much anything you can do with a regular SLEAP file (i.e., a labels file or a predictions file), you can do with the .slp file in your training job package. In particular, you can:

- open it in the GUI (you can only see frames with labeled data, since only these are included in the training job package)
- use it as the :code:`labels_path` parameter to the :code:`sleap-track` command-line interface