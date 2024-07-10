(cli)=

# Command line interfaces

SLEAP provides several types of functionality accessible through a command prompt.

## GUI

(sleap-label)=

### `sleap-label`

{code}`sleap-label` runs the GUI application for labeling and viewing {code}`.slp` files.

```none
usage: sleap-label [-h] [--nonnative] [--profiling] [--reset] [labels_path]

positional arguments:
  labels_path  Path to labels file

optional arguments:
  -h, --help   show this help message and exit
  --nonnative  Don't use native file dialogs
  --profiling  Enable performance profiling
  --reset      Reset GUI state and preferences. Use this flag if the GUI
               appears incorrectly or fails to open.
```

## Training

(sleap-train)=

### `sleap-train`

{code}`sleap-train` is the command-line interface for training. Use this for training on a remote machine/cluster/colab notebook instead of through the GUI.

```none
usage: sleap-train [-h] [--video-paths VIDEO_PATHS] [--val_labels VAL_LABELS]
                   [--test_labels TEST_LABELS] [--tensorboard] [--save_viz] 
                   [--keep_viz] [--zmq] [--run_name RUN_NAME] [--prefix PREFIX]
                   [--suffix SUFFIX]
                   training_job_path [labels_path]

positional arguments:
  training_job_path     Path to training job profile JSON file.
  labels_path           Path to labels file to use for training. If specified,
                        overrides the path specified in the training job
                        config.

optional arguments:
  -h, --help            show this help message and exit
  --video-paths VIDEO_PATHS
                        List of paths for finding videos in case paths inside
                        labels file are not accessible.
  --val_labels VAL_LABELS, --val VAL_LABELS
                        Path to labels file to use for validation. If
                        specified, overrides the path specified in the
                        training job config.
  --test_labels TEST_LABELS, --test TEST_LABELS
                        Path to labels file to use for test. If specified,
                        overrides the path specified in the training job
                        config.
  --base_checkpoint BASE_CHECKPOINT
                        Path to base checkpoint (directory containing best_model.h5)
                        to resume training from.
  --tensorboard         Enable TensorBoard logging to the run path if not
                        already specified in the training job config.
  --save_viz            Enable saving of prediction visualizations to the run
                        folder if not already specified in the training job
                        config.
  --keep_viz            Keep prediction visualization images in the run
                        folder after training if --save_viz is enabled.
  --zmq                 Enable ZMQ logging (for GUI) if not already specified
                        in the training job config.
  --run_name RUN_NAME   Run name to use when saving file, overrides other run
                        name settings.
  --prefix PREFIX       Prefix to prepend to run name.
  --suffix SUFFIX       Suffix to append to run name.
  --cpu                 Run training only on CPU. If not specified, will use
                        available GPU.
  --first-gpu           Run training on the first GPU, if available.
  --last-gpu            Run training on the last GPU, if available.
  --gpu GPU             Run training on the i-th GPU on the system. If 'auto', run on
                        the GPU with the highest percentage of available memory.
```

(sleap-export)=

### `sleap-export`

{code}`sleap-export` is a command-line interface for exporting trained models as a TensorFlow graph for use in other applications. See [this guide](https://www.tensorflow.org/guide/saved_model) for details on how TensorFlow saves models and the [`sleap.nn.inference.InferenceModel.export_model`](sleap.nn.inference.InferenceModel.export_model) documentation.

```none
usage: sleap-export [-h] [-m MODELS] [-e [EXPORT_PATH]]

optional arguments:
  -h, --help            show this help message and exit
  -m MODELS, --model MODELS
                        Path to trained model directory (with training_config.json). Multiple
                        models can be specified, each preceded by --model.
  -e [EXPORT_PATH], --export_path [EXPORT_PATH]
                        Path to output directory where the frozen model will be exported to.
                        Defaults to a folder named 'exported_model'.
  -r, --ragged RAGGED
                        Keep tensors ragged if present. If ommited, convert
                        ragged tensors into regular tensors with NaN padding.
  -n, --max_instances MAX_INSTANCES
                        Limit maximum number of instances in multi-instance models.
                        Not available for ID models. Defaults to None.
```

## Inference and Tracking

(sleap-track)=

### `sleap-track`

{code}`sleap-track` is the command-line interface for running inference using models which have already been trained. Use this for running inference on a remote machine such as an HPC cluster or Colab notebook.

If you specify how many identities there should be in a frame (i.e., the number of animals) with the {code}`--tracking.clean_instance_count` argument, then we will use a heuristic method to connect "breaks" in the track identities where we lose one identity and spawn another. This can be used as part of the inference pipeline (if models are specified), as part of the tracking-only pipeline (if the predictions file is specified and no models are specified), or by itself on predictions with pre-tracked identities (if you specify {code}`--tracking.tracker none`). See {ref}`proofreading` for more details on tracking.

```none
usage: sleap-track [-h] [-m MODELS] [--frames FRAMES] [--only-labeled-frames] [--only-suggested-frames] [-o OUTPUT] [--no-empty-frames]
                   [--verbosity {none,rich,json}] [--video.dataset VIDEO.DATASET] [--video.input_format VIDEO.INPUT_FORMAT]
                   [--video.index VIDEO.INDEX] [--cpu | --first-gpu | --last-gpu | --gpu GPU] [--max_edge_length_ratio MAX_EDGE_LENGTH_RATIO]
                   [--dist_penalty_weight DIST_PENALTY_WEIGHT] [--batch_size BATCH_SIZE] [--open-in-gui] [--peak_threshold PEAK_THRESHOLD]
                   [-n MAX_INSTANCES] [--tracking.tracker TRACKING.TRACKER] [--tracking.max_tracking TRACKING.MAX_TRACKING]
                   [--tracking.max_tracks TRACKING.MAX_TRACKS] [--tracking.target_instance_count TRACKING.TARGET_INSTANCE_COUNT]
                   [--tracking.pre_cull_to_target TRACKING.PRE_CULL_TO_TARGET] [--tracking.pre_cull_iou_threshold TRACKING.PRE_CULL_IOU_THRESHOLD]
                   [--tracking.post_connect_single_breaks TRACKING.POST_CONNECT_SINGLE_BREAKS]
                   [--tracking.clean_instance_count TRACKING.CLEAN_INSTANCE_COUNT] [--tracking.clean_iou_threshold TRACKING.CLEAN_IOU_THRESHOLD]
                   [--tracking.similarity TRACKING.SIMILARITY] [--tracking.match TRACKING.MATCH] [--tracking.robust TRACKING.ROBUST]
                   [--tracking.track_window TRACKING.TRACK_WINDOW] [--tracking.min_new_track_points TRACKING.MIN_NEW_TRACK_POINTS]
                   [--tracking.min_match_points TRACKING.MIN_MATCH_POINTS] [--tracking.img_scale TRACKING.IMG_SCALE]
                   [--tracking.of_window_size TRACKING.OF_WINDOW_SIZE] [--tracking.of_max_levels TRACKING.OF_MAX_LEVELS]
                   [--tracking.save_shifted_instances TRACKING.SAVE_SHIFTED_INSTANCES] [--tracking.kf_node_indices TRACKING.KF_NODE_INDICES]
                   [--tracking.kf_init_frame_count TRACKING.KF_INIT_FRAME_COUNT]
                   [data_path]

positional arguments:
  data_path             Path to data to predict on. This can be a labels (.slp) file or any supported video format.

optional arguments:
  -h, --help            show this help message and exit
  -m MODELS, --model MODELS
                        Path to trained model directory (with training_config.json). Multiple models can be specified, each preceded by --model.
  --frames FRAMES       List of frames to predict when running on a video. Can be specified as a comma separated list (e.g. 1,2,3) or a range
                        separated by hyphen (e.g., 1-3, for 1,2,3). If not provided, defaults to predicting on the entire video.
  --only-labeled-frames
                        Only run inference on user labeled frames when running on labels dataset. This is useful for generating predictions to compare
                        against ground truth.
  --only-suggested-frames
                        Only run inference on unlabeled suggested frames when running on labels dataset. This is useful for generating predictions for
                        initialization during labeling.
  -o OUTPUT, --output OUTPUT
                        The output filename to use for the predicted data. If not provided, defaults to '[data_path].predictions.slp'.
  --no-empty-frames     Clear any empty frames that did not have any detected instances before saving to output.
  --verbosity {none,rich,json}
                        Verbosity of inference progress reporting. 'none' does not output anything during inference, 'rich' displays an updating
                        progress bar, and 'json' outputs the progress as a JSON encoded response to the console.
  --video.dataset VIDEO.DATASET
                        The dataset for HDF5 videos.
  --video.input_format VIDEO.INPUT_FORMAT
                        The input_format for HDF5 videos.
  --video.index VIDEO.INDEX
                        Integer index of video in .slp file to predict on. To be used with an .slp path as an alternative to specifying the video
                        path.
  --cpu                 Run inference only on CPU. If not specified, will use available GPU.
  --first-gpu           Run inference on the first GPU, if available.
  --last-gpu            Run inference on the last GPU, if available.
  --gpu GPU             Run training on the i-th GPU on the system. If 'auto', run on the GPU with the highest percentage of available memory.
  --max_edge_length_ratio MAX_EDGE_LENGTH_RATIO
                        The maximum expected length of a connected pair of points as a fraction of the image size. Candidate connections longer than
                        this length will be penalized during matching. Only applies to bottom-up (PAF) models.
  --dist_penalty_weight DIST_PENALTY_WEIGHT
                        A coefficient to scale weight of the distance penalty. Set to values greater than 1.0 to enforce the distance penalty more
                        strictly. Only applies to bottom-up (PAF) models.
  --batch_size BATCH_SIZE
                        Number of frames to predict at a time. Larger values result in faster inference speeds, but require more memory.
  --open-in-gui         Open the resulting predictions in the GUI when finished.
  --peak_threshold PEAK_THRESHOLD
                        Minimum confidence map value to consider a peak as valid.
  -n MAX_INSTANCES, --max_instances MAX_INSTANCES
                        Limit maximum number of instances in multi-instance models. Not available for ID models. Defaults to None.
  --tracking.tracker TRACKING.TRACKER
                        Options: simple, flow, simplemaxtracks, flowmaxtracks, None (default: None)
  --tracking.max_tracking TRACKING.MAX_TRACKING
                        If true then the tracker will cap the max number of tracks. (default: False)
  --tracking.max_tracks TRACKING.MAX_TRACKS
                        Maximum number of tracks to be tracked by the tracker. (default: None)
  --tracking.target_instance_count TRACKING.TARGET_INSTANCE_COUNT
                        Target number of instances to track per frame. (default: 0)
  --tracking.pre_cull_to_target TRACKING.PRE_CULL_TO_TARGET
                        If non-zero and target_instance_count is also non-zero, then cull instances over target count per frame *before* tracking.
                        (default: 0)
  --tracking.pre_cull_iou_threshold TRACKING.PRE_CULL_IOU_THRESHOLD
                        If non-zero and pre_cull_to_target also set, then use IOU threshold to remove overlapping instances over count *before*
                        tracking. (default: 0)
  --tracking.post_connect_single_breaks TRACKING.POST_CONNECT_SINGLE_BREAKS
                        If non-zero and target_instance_count is also non-zero, then connect track breaks when exactly one track is lost and exactly
                        one track is spawned in frame. (default: 0)
  --tracking.clean_instance_count TRACKING.CLEAN_INSTANCE_COUNT
                        Target number of instances to clean *after* tracking. (default: 0)
  --tracking.clean_iou_threshold TRACKING.CLEAN_IOU_THRESHOLD
                        IOU to use when culling instances *after* tracking. (default: 0)
  --tracking.similarity TRACKING.SIMILARITY
                        Options: instance, centroid, iou (default: instance)
  --tracking.match TRACKING.MATCH
                        Options: hungarian, greedy (default: greedy)
  --tracking.robust TRACKING.ROBUST
                        Robust quantile of similarity score for instance matching. If equal to 1, keep the max similarity score (non-robust).
                        (default: 1)
  --tracking.track_window TRACKING.TRACK_WINDOW
                        How many frames back to look for matches (default: 5)
  --tracking.min_new_track_points TRACKING.MIN_NEW_TRACK_POINTS
                        Minimum number of instance points for spawning new track (default: 0)
  --tracking.min_match_points TRACKING.MIN_MATCH_POINTS
                        Minimum points for match candidates (default: 0)
  --tracking.img_scale TRACKING.IMG_SCALE
                        For optical-flow: Image scale (default: 1.0)
  --tracking.of_window_size TRACKING.OF_WINDOW_SIZE
                        For optical-flow: Optical flow window size to consider at each pyramid (default: 21)
  --tracking.of_max_levels TRACKING.OF_MAX_LEVELS
                        For optical-flow: Number of pyramid scale levels to consider (default: 3)
  --tracking.save_shifted_instances TRACKING.SAVE_SHIFTED_INSTANCES
                        If non-zero and tracking.tracker is set to flow, save the shifted instances between elapsed frames (default: 0)
  --tracking.kf_node_indices TRACKING.KF_NODE_INDICES
                        For Kalman filter: Indices of nodes to track. (default: )
  --tracking.kf_init_frame_count TRACKING.KF_INIT_FRAME_COUNT
                        For Kalman filter: Number of frames to track with other tracker. 0 means no Kalman filters will be used. (default: 0)
```

#### Examples:

**1. Simple inference without tracking:**

```none
sleap-track -m "models/my_model" -o "output_predictions.slp" "input_video.mp4"
```

**2. Inference with multi-model pipelines (e.g., top-down):**

```none
sleap-track -m "models/centroid" -m "models/centered_instance" -o "output_predictions.slp" "input_video.mp4"
```

**3. Inference on suggested frames of a labeling project:**

```none
sleap-track -m "models/my_model" --only-suggested-frames -o "labels_with_predictions.slp" "labels.v005.slp"
```

The resulting `labels_with_predictions.slp` can then merged into the base labels project from the SLEAP GUI via **File** --> **Merge into project...**.

**4. Inference with simple tracking:**

```none
sleap-track -m "models/my_model" --tracking.tracker simple -o "output_predictions.slp" "input_video.mp4"
```

**5. Inference with max tracks limit:**

```none
sleap-track -m "models/my_model" --tracking.tracker simplemaxtracks --tracking.max_tracking 1 --tracking.max_tracks 4 -o "output_predictions.slp" "input_video.mp4"
```

**6. Re-tracking without pose inference:**

```none
sleap-track --tracking.tracker simplemaxtracks --tracking.max_tracking 1 --tracking.max_tracks 4 -o "retracked.slp" "input_predictions.slp"
```

**7. Select GPU for pose inference:**

```none
sleap-track --gpu 1 ...
```

**8. Select subset of frames to predict on:**

```none
sleap-track -m "models/my_model" --frames 1000-2000 "input_video.mp4"
```

## Dataset files

(sleap-convert)=

### `sleap-convert`

{code}`sleap-convert` allows you to convert between various dataset file formats. Amongst other things, it can be used to export data from a SLEAP dataset into an HDF5 file that can be easily used for analysis (e.g., read from MATLAB). See {py:mod}`sleap.io.convert` for more information.

```none
usage: sleap-convert [-h] [-o OUTPUT] [--format FORMAT] [--video VIDEO]
                     input_path

positional arguments:
  input_path            Path to input file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to output file (optional). The analysis format expects an
                        output path per video in the project. Otherwise, the default
                        naming convention
                        <slp path>.<video index>_<video filename>.analysis.h5 will be
                        used for every video without a specified output path. Multiple
                        outputs can be specified, each preceeded by --output.

                        Example (analysis format):
                          Input:
                            predictions.slp: Path to .slp file to convert which has two
                            videos:
                            - first-video.mp4 at video index 0 and
                            - second-video.mp4 at video index 1.
                          Command:
                            sleap-convert predictions.slp --format analysis --output analysis_video_0.h5
                          Output analysis files:
                            analysis_video_0.h5: Analysis file for first-video.mp4
                              (at index 0) in predictions.slp.
                            predictions.001_second-video.analysis.h5: Analysis file for
                              second-video.mp4 (at index 1) in predictions.slp. Since
                              only a single --output argument was specified, the
                              analysis file for the latter video is given a default name.
  --format FORMAT       Output format. Default ('slp') is SLEAP dataset;
                        'analysis' results in analysis.h5 file; 'analysis.nix' results
                        in an analysis nix file; 'analysis.csv' results
                        in an analysis csv file; 'h5' or 'json' results in SLEAP dataset
                        with specified file format.
  --video VIDEO         Path to video (if needed for conversion).
```

For example, to convert a predictions SLP file to an analysis HDF5 file:

```
sleap-convert --format analysis -o "session1.predictions.analysis.h5" "session1.predictions.slp"
```

See [Analysis examples](../notebooks/Analysis_examples.html) for how to work with these outputs.

(sleap-inspect)=

### `sleap-inspect`

{code}`sleap-inspect` gives you various information about a SLEAP dataset file such as a list of videos and a count of the frames with labels. If you're inspecting a predictions dataset (i.e., the output from running {code}`sleap-track` or inference in the GUI) it will also include details about how those predictions were created (i.e., the models, the version of SLEAP, and any inference parameters).

You can also specify a model folder to get a quick summary of the configuration and metrics (if available).

```none
usage: sleap-inspect [-h] [--verbose] data_path

positional arguments:
  data_path   Path to labels file (.slp) or model folder

optional arguments:
  -h, --help  show this help message and exit
  --verbose
```

## Rendering

(sleap-render)=

### `sleap-render`

{code}`sleap-render` allows you to render videos directly from the CLI. It is used to render video clips with Instances.

```none
usage: sleap-render [-h] [-o OUTPUT] [-f FPS] [--scale SCALE] [--crop CROP] [--frames FRAMES] [--video-index VIDEO_INDEX] data_path
positional arguments:
  data_path             Path to labels json file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path for saving output (default: None)
  --video-index VIDEO_INDEX
                        Index of video in labels dataset (default: 0)
  --frames FRAMES       List of frames to predict. Either comma separated list (e.g. 1,2,3)
                        or a range separated by hyphen (e.g. 1-3). (default is entire video)
  -f FPS, --fps FPS     Frames per second for output video (default: 25)
  --scale SCALE         Output image scale (default: 1.0)
  --crop CROP           Crop size as <width>,<height> (default: None)
  --show_edges SHOW_EDGES
                        Whether to draw lines between nodes (default: 1)
  --edge_is_wedge EDGE_IS_WEDGE
                        Whether to draw edges as wedges (default: 0)
  --marker_size MARKER_SIZE
                        Size of marker in pixels before scaling by SCALE (default: 4)
  --palette PALETTE     SLEAP color palette to use. Options include: "alphabet", "five+",
                        "solarized", or "standard" (default: "standard")
  --distinctly_color DISTINCTLY_COLOR
                        Specify how to color instances. Options include: "instances",
                        "edges", and "nodes" (default: "instances")
  --background BACKGROUND
                        Specify the type of background to be used to save the videos.
                        Options: original, black, white and grey. (default: "original")
```

## Debugging

(sleap-diagnostic)=

### `sleap-diagnostic`

There's also a script to output diagnostic information which may help us if you need to contact us about problems installing or running SLEAP. If you were able to install the SLEAP Python package, you can run this script with {code}`sleap-diagnostic`. Otherwise, you can download [diagnostic.py](https://raw.githubusercontent.com/talmolab/sleap/main/sleap/diagnostic.py) and run {code}`python diagnostic.py`.

```none
usage: sleap-diagnostic [-h] [-o OUTPUT] [--gui-check]

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path for saving output
  --gui-check           Check if Qt GUI widgets can be used
```

:::{note}
For more details about any command, run with the {code}`--help` argument (e.g., {code}`sleap-track --help`).
:::
