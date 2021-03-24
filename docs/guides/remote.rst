.. _remote_train:

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

Usually the easiest and best way to make the training labels and images available is to export a training job package ("Predict" -> "Run Training.." -> "Export Training Job Package..") and copy that to the remote machine.

Although it's easiest if you bundle the labels and images into training job package, there are alternatives. If the files are already on a shared network drive, it may be possible to use the original labels project and videos for training. But this can be tricky, because often the full paths to the files will be different when accessed from different machines (i.e., different paths on Windows and Linux machines or different paths from how the network drive is mounted). To use the original labels and video files, you'll either need to ensure that the file paths to videos used in the project are the same on the remote machine as on the local machine where you last saved the project, **or** if all the video files have distinct filenames, you can place the videos inside the same directory which contains the labels project file.

But in most cases it's best to create a training job package and just use that for remote training.

**Training profile**:

SLEAP comes with "default" training profiles for training confidence maps, part affinity fields, centroids, or top-down confidence maps (which allow multi-instance inference without using part affinity fields). Any file in the `training_profiles <https://github.com/murthylab/sleap/tree/main/sleap/training_profiles>`_ directory of the SLEAP package can be used by specifying its filename (e.g., :code:`baseline.bottomup.json`) as the training profile—the full path isn't required.

Our guide to :ref:`custom_training` explains how to use the GUI to export custom training profiles. You can also use the :code:`initial_config.json` file saved from previous training run as a template for a new training config. You can copy the :code:`json` file and edit it in any text editor.

When you run training, specify the full path to the :code:`json` file.

**Command-line training**:

Once you have your training job package (or labels package and training profile), you can run training like so:

::

  sleap-train path/to/your/training_profile.json another/path/to/labels.pkg.slp

The model will be saved in the :code:`models/` directory within the same directory as the **training job package** (in this case, :code:`another/path/to/models/run_name/`). You can specify the :code:`run_name` to use when saving the model with the :code:`-o` argument, otherwise the run name will be the date and time of the run (or whatever is specified as the run path inside the config file).

.. _pretrained_weights_remote:

Downloading pretrained weights for remote training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**How can I download pretrained weights to use for my backbones?**
Normally, pretrained weights will be automatically downloaded when the network is trained. This can be an issue when training in an environment that does not have internet access (e.g., HPC clusters). A simple solution is to manually trigger downloading of the pretrained weights in the same environment (e.g., the head node of a cluster). These one-liners will trigger weight downloading for the appropriate backbones:

.. code-block:: bash

  python -c "from tensorflow.keras import applications; applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"

  python -c "from tensorflow.keras import applications; applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"
  python -c "from tensorflow.keras import applications; applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"
  python -c "from tensorflow.keras import applications; applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"

  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.25)"
  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.5)"
  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.75)"
  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.0)"

  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.35)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.5)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.75)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.0)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.3)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.4)"

  python -c "from tensorflow.keras import applications; applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))"
  python -c "from tensorflow.keras import applications; applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))"


Be sure to run these with the same ``python`` binary as will be used for training so the weights can be loaded from the disk appropriately.


.. _remote_inference:

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

When you train a model, you'll get a directory with the `run_name` of the model.

The model directory will contain at least these two files:

- :code:`training_config.json` is the training profile used to train the model, together with some additional information about the trained model. Amongst other things, this specifies the network architecture of the model.
- :code:`best_model.h5` (and/or :code:`final_model.h5`) contains the weights for the trained model.

You'll need both of these files for each model you're going to use for inference.

The directory may also contains other files with optional outputs from the training run (e.g., :code:`training_log.csv` or a :code:`viz/` subdirectory).

Inference will run in different modes depending on the output types of the models you supply. See the instructions for :ref:`choosing_models`.

For this example, let's suppose you have two models: centroids and instance-centered confidence maps. This is the typical "top-down" case for multi-instance predictions.

**Video**

SLEAP uses OpenCV to read a variety of video formats including `mp4` and `avi` files. You'll just need the file path to run inference on such a video file.

SLEAP can also read videos stored as a datasets inside an HDF5 file. To run inference on an HDF5 video, you'll need the file path, the dataset path, and whether the video data is formatted is formatted as `(channels, images, height, width)` or `(images, height, width, channels)`.

For this example, let's suppose you're working with an HDF5 video at :code:`path/to/video.h5`, and the video data is stored in the :code:`video/` dataset with channels as the index.

**Command-line inference**:

To run inference, you'll call :code:`sleap-track` with the paths to each trained model and your video file, like so:

::

  sleap-track path/to/video.h5 \
  --video.dataset video --video.input_format channels_last \
  -m path/to/models/191205_162402 \
  -m path/to/models/191205_163413

(The order of the models doesn't matter.)

This will run inference on the entire video. If you only want to run inference on some range of frames, you can specify this with the :code:`--frames 123-456` command-line argument.

This will give you predictions frame-by-frame, but will not connect those predictions across frames into `tracks`. If you want cross-frame identity tracking, you'll need to choose a tracker and specify this from the command-line with the :code:`--tracking.tracker` argument. For optical flow, use :code:`--tracking.tracker flow`. For matching identities without optical flow and using each instance centroid (rather than all the predicted nodes), use :code:`--tracking.tracker simple --tracking.similarity centroid`.

**In future versions** it will also be possible to run tracking separately after you've generated a predictions file (see :ref:`reference`). This makes it easy to try different tracking methods and parameters without needing to re-run the full inference process.

When inference is finished, it will save the predictions in a new HDF5 file. This file has the same format as a standard SLEAP project file, and you can use the GUI to proofread this file or merge the predictions into an existing SLEAP project. The file will be in the same directory as the video and the filename will be :code:`{video filename}.predictions.h5`.
