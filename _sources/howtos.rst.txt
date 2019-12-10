.. _howtos:

How-Tos
=======

Training and Inference
-----------------------------

.. _training_package:

Export a training package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Case: You've created a project with training data on one computer, and you want to use a different computer for training models. This could be another desktop with a GPU, an HPC cluster, or a Colab notebook.*

The easiest way to move your training data to another machine is to export a **training package**. This is a single HDF5 file which contains both labeled data as well as the images which will be used for training. This makes it easy to transport your training data since you won't have to worry about paths to your video files.

To export a training package, use the "**Export Training Package...**" command in the "Predict" menu of the GUI app.

Pretty much anything you can do with a regular SLEAP file (i.e., a labels file or a predictions file), you can do with a training package file. In particular, you can:

- open a training package in the GUI (you can only see frames with labeled data, since only these are included in the training package)
- use a training package as the `labels_path` parameter to the `sleap-track` command-line interface

Run training and/or inference on Colab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Case: You already have a project with labeled training data and you'd like to run training or inference in a Colab notebook.*

`This notebook <https://colab.research.google.com/drive/1jLS4UQ8p-DCQE8WET8w8i8Jf2Apxsq47>`_ will walk you through the process.

You'll need a `Google Drive <https://www.google.com/drive/>`_ where you can upload your training data (as a tracking package file), store models and predictions.

Remote training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Case: You already have a project with training data and you want to train on a different machine using a command-line interface.*

You need three things to run training:

1. You need to install SLEAP on the remote machine where you'll run training.
2. Labels and images to use for training.
3. A training profile which defines the training parameters (e.g., learning rate, image augmentation methods).

**Installing SLEAP**:

See the :ref:`installation` instructions.

**Training labels and images**:

Usually the easiest and best way to make the training labels and images available is to export a training package and copy that to the remote machine. See the instructions above to :ref:`training_package`.

Although it's easiest if you bundle the labels and images into training package, there are alternatives. If the files are already on a shared network drive, it may be possible to use the original labels project and videos for training. But this can be tricky, because often the full paths to the files will be different when accessed from different machines (i.e., different paths on Windows and Linux machines or different paths from how the network drive is mounted). To use the original labels and video files, you'll either need to ensure that the file paths to videos used in the project are the same on the remote machine as on the local machine where you last saved the project, **or** if all the video files have distinct filenames, you can place the videos inside the same directory which contains the labels project file.

But in most cases it's best to create a training package and just use that for remote training.

**Training profile**:

SLEAP comes with "default" training profiles for training confidence maps, part affinity fields, centroids, or top-down confidence maps (which allow multi-instance inference without using part affinity fields). Any file in the `training_profiles <https://github.com/murthylab/sleap/tree/master/sleap/training_profiles>`_ directory of the SLEAP package can be used by specifying it's filename (e.g., :code:`default_confmaps.json`) as the training profile—the full path isn't required.

You can also use a custom training profile. There's a GUI **training editor** which gives you access to many of the profile parameters (:code:`python -m sleap.gui.training_editor`, as described in the :ref:`reference`), or you can directly edit a profile :code:`.json` file in a text editor. To use a custom training profile, you'll need to specify the full path to the file when you run training.

**Command-line training**:

Once you have your training package (or labels project file) and training profile, you can run training like so:

::

  sleap-train path/to/your/training_profile.json another/path/to/training_package.h5

The model will be saved in the :code:`models/` directory within the same directory as the **training package** (in this case, :code:`another/path/to/models/run_name/`). You can specify the :code:`run_name` to use when saving the model with the :code:`-o` argument, otherwise the run name will include a timestamp, the output type and model architecture.

Remote inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Case: You already have models and you want to run inference on a different machine using a command-line interface.*

Here's what you need to run inference:

1. You need to install SLEAP on the remote machine where you'll run training.
2. You need a compatible set of trained model files.
3. You need a video for which you want predictions.

**Installing SLEAP**:

See the :ref:`installation` instructions.

**Trained models**

When you train a model, you'll get a directory with the `run_name` of the model. This will typically be something like :code:`191205_162402.UNet.confmaps` (i.e., :code:`<timestamp>.<architecture>.<output type>`), although you can also specify the run name in the training command-line interface.

The model directory will contain two or three files:

- :code:`training_job.json` is the training profile used to train the model, together with some additional information about the trained model. Amongst other things, this specifies the network architecture of the model.
- :code:`best_model.h5` and/or :code:`final_model.h5` are the weights for the trained model.

You'll need this entire directory for each model you're going to use for inference.

Inference will run in different modes depending on the output types of the models you supply. See the instructions for :ref:`choosing_models`.

For this example, let's suppose you have three models: confidence maps (confmaps), part affinity fields (pafs), and centroids. This is the typical case for multi-instance predictions.

**Video**

SLEAP uses OpenCV to read a variety of video formats including `mp3` and `avi` files. You'll just need the file path to run inference on such a video file.

SLEAP can also read videos stored as a datasets inside an HDF5 file. To run inference on an HDF5 video, you'll need the file path, the dataset path, and whether the video data is formatted is formatted as `(channels, images, height, width)` or `(images, height, width, channels)`.

For this example, let's suppose you're working with an HDF5 video at :code:`path/to/video.h5`, and the video data is stored in the :code:`video/` dataset with channels as the index.

**Command-line inference**:

To run inference, you'll call :code:`sleap-track` with the paths to each trained model and your video file, like so:

::

  sleap-track path/to/video.h5 \
  --video.dataset video --video.input_format channels_last \
  -m path/to/models/191205_162402.UNet.confmaps \
  -m path/to/models/191205_163413.LeapCNN.pafs \
  -m path/to/models/191205_170118.UNet.centroids \

(The order of the models doesn't matter.)

This will run inference on the entire video. If you only want to run inference on some range of frames, you can specify this with the :code:`--frames 123-456` command-line argument.

This will give you predictions frame-by-frame, but will not connect those predictions across frames into `tracks`. If you want cross-frame identity tracking, you'll need to choose a tracker and specify this from the command-line with the :code:`--tracking.tracker` argument. For optical flow, use :code:`--tracking.tracker flow`. For matching identities without optical flow and using each instance centroid (rather than all the predicted nodes), use :code:`--tracking.tracker simple --tracking.similarity centroid`.

It's also possible to run tracking separately after you've generated a predictions file (see :ref:`reference`). This makes it easy to try different tracking methods and parameters without needing to re-run the full inference process.

When inference is finished, it will save the predictions in a new HDF5 file. This file has the same format as a standard SLEAP project file, and you can use the GUI to proofread this file or merge the predictions into an existing SLEAP project. The file will be in the same directory as the video and the filename will be :code:`{video filename}.predictions.h5`.

.. _choosing_models:

Choosing a set of models
~~~~~~~~~~~~~~~~~~~~~~~~~

Inference will run in different modes depending on the output types of the models you supply. SLEAP currently support four different output types:

1. **Confidence maps** (confmaps) are used to predict point locations.

2. **Part affinity fields** (pafs) are used to connect points which belong to the same animal instance.

3. **Centroids** are used to crop the video frame around each animal instance.

4. **Top-down confidence maps** (topdown) are used to predict point locations for a *single* instance at the center of a cropped image.

When there's only a **single** instance in the video, run with confidence maps. Centroids are optional.

When there are **multiple** instances in the video, you have two options:

1. Confidence maps (*required*) and part affinity fields (*required*), with centroids *optional*.
2. Top-down confidence maps and centroids (*required*).

Note that top-down confidence maps rely on centroid cropping, since they're trained to give predictions for the single instance centered in the (cropped) image.

Improving predictions
----------------------------

Add more training data to a project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Case: You have predictions that aren't in the same project as your original training data and you want to correct some of the predictions and use these corrections to train a better model.*

**TODO**

- correct predictions
- make copy
- delete predictions
- import into project with original training data
- train new models

Experimenting with training parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**TODO**

Proofreading
----------------------

*Case: You're happy enough with the frame-by-frame predictions but you need to correct the identities tracked across frames.*

**TODO**

General:

- use colors, pick palette with enough distinct colors, modify palette if background makes some colors hard to see
- try alternate tracking methods
- add more training data (tracking could be poor because there are too many frames with one or more instances without predictions)

For breaks:

- step between spawn frames
- try track cleaner

For swaps:

- longer trails and step 50 frames at a time (to find swaps)
- short trails and velocity-based suggestions (to find swaps)
