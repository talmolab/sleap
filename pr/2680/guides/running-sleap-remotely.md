*Case: You already have a project with training data and you want to train on a different machine using a command-line interface.*

You need three things to run training:

!!! note
    If you only need to run training and inference from the command line, you do **not** need to install the full SLEAP package—just installing `sleap-nn` is sufficient.

1. You need to install `sleap-nn` on the remote machine where you'll run training.
2. Labels and images to use for training.
3. A training profile which defines the training parameters (e.g., learning rate, image augmentation methods).

**Installing sleap-nn**:

See the [installation instructions](https://nn.sleap.ai/latest/installation).

**Training labels and images**:

Usually the easiest and best way to make the training labels and images available is to export a training job package ("Predict" -> "Run Training.." -> "Export Training Job Package..") and copy that to the remote machine.

Although it's easiest if you bundle the labels and images into training job package, there are alternatives. If the files are already on a shared network drive, it may be possible to use the original labels project and videos for training. But this can be tricky, because often the full paths to the files will be different when accessed from different machines (i.e., different paths on Windows and Linux machines or different paths from how the network drive is mounted). To use the original labels and video files, you'll either need to ensure that the file paths to videos used in the project are the same on the remote machine as on the local machine where you last saved the project, **or** if all the video files have distinct filenames, you can place the videos inside the same directory which contains the labels project file.

But in most cases it's best to create a training job package and just use that for remote training.

**Training profile**:

sleap-nn comes with "default" training profiles for training confidence maps, part affinity fields, centroids, or top-down confidence maps (which allow multi-instance inference without using part affinity fields). Any file in the [sample configs](https://github.com/talmolab/sleap-nn/tree/main/docs/sample_configs) can be downloaded/ used, and path to the config file is passed to `sleap-nn train` CLI.

Our guide to [creating a custom training profile](creating-a-custom-training-profile.md) explains how to use the GUI to export custom training profiles. You can also use the `initial_config.yaml` file saved from previous training run as a template for a new training config. You can copy the `yaml` file and edit it in any text editor.

**Command-line training**:

Once you have your training job package (or labels package and training profile), you can run training using the [`sleap-nn train`](https://nn.sleap.ai/latest/training/#using-cli) command like so:

```sh
sleap-nn train --config <path/to/config.yaml> "data_config.train_labels_path=[<path/to/slp/file>]" trainer_config.ckpt_dir="models" trainer_config.run_name=<run_name>
```

The model will be saved in the `models/` directory within the same directory as the **training job package** (in this case, `models/run_name/`).

!!! note
    If you exported the training package as a ZIP file, it contains both the `.pkg.slp` and `.yaml` files necessary to train with the configuration you selected in the GUI. Before running the [`sleap-nn train`](https://nn.sleap.ai/latest/training/#using-cli) command, make sure to unzip this file:

    === "macOS/Linux"

        ```bash
        unzip training_job.zip -d training_job
        ```

    === "Windows (PowerShell)"

        ```powershell
        Expand-Archive -Path training_job.zip -DestinationPath training_job
        ```

    === "Windows (Command Prompt)"

        ```cmd
        tar -xf training_job.zip -C training_job
        ```

## Remote inference

*Case: You already have models and you want to run inference on a different machine using a command-line interface.*

Here's what you need to run inference:

1. You need to install sleap-nn on the remote machine where you'll run training.
2. You need a compatible set of trained model files.
3. You need a video for which you want predictions.

**Installing sleap-nn**:

See the [installation instructions](https://nn.sleap.ai/latest/installation).

**Trained models**

When you train a model, you'll get a directory with the `run_name` of the model.

The model directory will contain at least these two files:

- `training_config.yaml` (or `training_config.json` from SLEAP<=1.4.1) is the training profile used to train the model, together with some additional information about the trained model. Amongst other things, this specifies the network architecture of the model.
- `best.ckpt` (or `best_model.h5` from SLEAP<=1.4.1) contains the weights for the trained model.

You'll need both of these files for each model you're going to use for inference. 

!!! note "Legacy SLEAP Model Support"
    SLEAP-NN supports running inference on models trained with legacy SLEAP (version 1.4.1 or earlier).  
    You can use the `sleap-nn track` command with legacy model files (`.h5` and `.json`) as described in the [Legacy SLEAP Model Support documentation](https://nn.sleap.ai/latest/inference/#legacy-sleap-model-support).  
    This allows you to run inference on older models without needing to retrain them with the new backend.


Inference will run in different modes depending on the output types of the models you supply. See the instructions for [Model Configuration](https://nn.sleap.ai/latest/reference/models/).

For this example, let's suppose you have two models: centroids and instance-centered confidence maps. This is the typical "top-down" case for multi-instance predictions.

**Video**

sleap-nn (which uses `sleap-io`) uses OpenCV to read a variety of video formats including `mp4` and `avi` files. You'll just need the file path to run inference on such a video file.

For this example, let's suppose you're working with an HDF5 video at `path/to/video.mp4`.

**Command-line inference**:

To run inference, you'll call [`sleap-nn track`](https://nn.sleap.ai/latest/inference/#run-inference-with-cli) with the paths to each trained model and your video file, like so:

```sh
sleap-nn track -i path/to/video.mp4 --video_dataset video --video_input_format channels_last -m path/to/models/centroid -m path/to/models/centered-instance
```

This will run inference on the entire video. If you only want to run inference on some range of frames, you can specify this with the `--frames 123-456` command-line argument.

This will give you predictions frame-by-frame, but will not connect those predictions across frames into `tracks`. If you want cross-frame identity tracking, set `--tracking` argument. For optical flow, use `--use_flow`. For matching identities without optical flow and using each instance centroid (rather than all the predicted nodes), use `--features centroids --scoring_method euclidean_dist`.

It's also be possible to run tracking separately after you've generated a predictions file (see [`Track-only workflow`](https://nn.sleap.ai/latest/inference/#track-only-workflow)). This makes it easy to try different tracking methods and parameters without needing to re-run the full inference process.

When inference is finished, it will save the predictions in a new slp file. This file has the same format as a standard SLEAP project file, and you can use the GUI to proofread this file or merge the predictions into an existing SLEAP project. The file will be in the same directory as the video and the filename will be `{video filename}.predictions.slp`.