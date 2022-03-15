.. _cli:

Command line interfaces
========================

SLEAP provides several types of functionality accessible through a command prompt.


GUI
---

.. _`sleap-label`:

``sleap-label``
+++++++++++++++++

:code:`sleap-label` runs the GUI application for labeling and viewing :code:`.slp` files.

.. code-block:: none

    usage: sleap-label [-h] [--nonnative] [--profiling] [--reset] [labels_path]

    positional arguments:
      labels_path  Path to labels file

    optional arguments:
      -h, --help   show this help message and exit
      --nonnative  Don't use native file dialogs
      --profiling  Enable performance profiling
      --reset      Reset GUI state and preferences. Use this flag if the GUI
                   appears incorrectly or fails to open.

Training
--------

.. _`sleap-train`:

``sleap-train``
+++++++++++++++++

:code:`sleap-train` is the command-line interface for training. Use this for training on a remote machine/cluster/colab notebook instead of through the GUI.

.. code-block:: none

    usage: sleap-train [-h] [--video-paths VIDEO_PATHS] [--val_labels VAL_LABELS]
                       [--test_labels TEST_LABELS] [--tensorboard] [--save_viz]
                       [--zmq] [--run_name RUN_NAME] [--prefix PREFIX]
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
      --tensorboard         Enable TensorBoard logging to the run path if not
                            already specified in the training job config.
      --save_viz            Enable saving of prediction visualizations to the run
                            folder if not already specified in the training job
                            config.
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
      --gpu GPU             Run training on the i-th GPU on the system.


Inference and Tracking
----------------------

.. _`sleap-track`:

``sleap-track``
+++++++++++++++++

:code:`sleap-track` is the command-line interface for running inference using models which have already been trained. Use this for running inference on a remote machine such as an HPC cluster or Colab notebook.

If you specify how many identities there should be in a frame (i.e., the number of animals) with the :code:`--tracking.clean_instance_count` argument, then we will use a heuristic method to connect "breaks" in the track identities where we lose one identity and spawn another. This can be used as part of the inference pipeline (if models are specified), as part of the tracking-only pipeline (if the predictions file is specified and no models are specified), or by itself on predictions with pre-tracked identities (if you specify :code:`--tracking.tracker none`). See :ref:`proofreading` for more details on tracking.

.. code-block:: none

    usage: sleap-track [-h] [-m MODELS] [--frames FRAMES] [--only-labeled-frames]
                       [--only-suggested-frames] [-o OUTPUT] [--no-empty-frames]
                       [--verbosity {none,rich,json}]
                       [--video.dataset VIDEO.DATASET]
                       [--video.input_format VIDEO.INPUT_FORMAT]
                       [--cpu | --first-gpu | --last-gpu | --gpu GPU]
                       [--peak_threshold PEAK_THRESHOLD] [--batch_size BATCH_SIZE]
                       [--open-in-gui] [--tracking.tracker TRACKING.TRACKER]
                       [--tracking.target_instance_count TRACKING.TARGET_INSTANCE_COUNT]
                       [--tracking.pre_cull_to_target TRACKING.PRE_CULL_TO_TARGET]
                       [--tracking.pre_cull_iou_threshold TRACKING.PRE_CULL_IOU_THRESHOLD]
                       [--tracking.post_connect_single_breaks TRACKING.POST_CONNECT_SINGLE_BREAKS]
                       [--tracking.clean_instance_count TRACKING.CLEAN_INSTANCE_COUNT]
                       [--tracking.clean_iou_threshold TRACKING.CLEAN_IOU_THRESHOLD]
                       [--tracking.similarity TRACKING.SIMILARITY]
                       [--tracking.match TRACKING.MATCH]
                       [--tracking.track_window TRACKING.TRACK_WINDOW]
                       [--tracking.min_new_track_points TRACKING.MIN_NEW_TRACK_POINTS]
                       [--tracking.min_match_points TRACKING.MIN_MATCH_POINTS]
                       [--tracking.img_scale TRACKING.IMG_SCALE]
                       [--tracking.of_window_size TRACKING.OF_WINDOW_SIZE]
                       [--tracking.of_max_levels TRACKING.OF_MAX_LEVELS]
                       [--tracking.kf_node_indices TRACKING.KF_NODE_INDICES]
                       [--tracking.kf_init_frame_count TRACKING.KF_INIT_FRAME_COUNT]
                       [data_path]

    positional arguments:
      data_path             Path to data to predict on. This can be a labels
                            (.slp) file or any supported video format.

    optional arguments:
      -h, --help            show this help message and exit
      -m MODELS, --model MODELS
                            Path to trained model directory (with
                            training_config.json). Multiple models can be
                            specified, each preceded by --model.
      --frames FRAMES       List of frames to predict when running on a video. Can
                            be specified as a comma separated list (e.g. 1,2,3) or
                            a range separated by hyphen (e.g., 1-3, for 1,2,3). If
                            not provided, defaults to predicting on the entire
                            video.
      --only-labeled-frames
                            Only run inference on user labeled frames when running
                            on labels dataset. This is useful for generating
                            predictions to compare against ground truth.
      --only-suggested-frames
                            Only run inference on unlabeled suggested frames when
                            running on labels dataset. This is useful for
                            generating predictions for initialization during
                            labeling.
      -o OUTPUT, --output OUTPUT
                            The output filename to use for the predicted data. If
                            not provided, defaults to
                            '[data_path].predictions.slp'.
      --no-empty-frames     Clear any empty frames that did not have any detected
                            instances before saving to output.
      --verbosity {none,rich,json}
                            Verbosity of inference progress reporting. 'none' does
                            not output anything during inference, 'rich' displays
                            an updating progress bar, and 'json' outputs the
                            progress as a JSON encoded response to the console.
      --video.dataset VIDEO.DATASET
                            The dataset for HDF5 videos.
      --video.input_format VIDEO.INPUT_FORMAT
                            The input_format for HDF5 videos.
      --cpu                 Run inference only on CPU. If not specified, will use
                            available GPU.
      --first-gpu           Run inference on the first GPU, if available.
      --last-gpu            Run inference on the last GPU, if available.
      --gpu GPU             Run inference on the i-th GPU specified.
      --peak_threshold PEAK_THRESHOLD
                            Minimum confidence map value to consider a peak as
                            valid.
      --batch_size BATCH_SIZE
                            Number of frames to predict at a time. Larger values
                            result in faster inference speeds, but require more
                            memory.
      --open-in-gui         Open the resulting predictions in the GUI when
                            finished.
      --tracking.tracker TRACKING.TRACKER
                            Options: simple, flow, None (default: None)
      --tracking.target_instance_count TRACKING.TARGET_INSTANCE_COUNT
                            Target number of instances to track per frame.
                            (default: 0)
      --tracking.pre_cull_to_target TRACKING.PRE_CULL_TO_TARGET
                            If non-zero and target_instance_count is also non-
                            zero, then cull instances over target count per frame
                            *before* tracking. (default: 0)
      --tracking.pre_cull_iou_threshold TRACKING.PRE_CULL_IOU_THRESHOLD
                            If non-zero and pre_cull_to_target also set, then use
                            IOU threshold to remove overlapping instances over
                            count *before* tracking. (default: 0)
      --tracking.post_connect_single_breaks TRACKING.POST_CONNECT_SINGLE_BREAKS
                            If non-zero and target_instance_count is also non-
                            zero, then connect track breaks when exactly one track
                            is lost and exactly one track is spawned in frame.
                            (default: 0)
      --tracking.clean_instance_count TRACKING.CLEAN_INSTANCE_COUNT
                            Target number of instances to clean *after* tracking.
                            (default: 0)
      --tracking.clean_iou_threshold TRACKING.CLEAN_IOU_THRESHOLD
                            IOU to use when culling instances *after* tracking.
                            (default: 0)
      --tracking.similarity TRACKING.SIMILARITY
                            Options: instance, centroid, iou (default: instance)
      --tracking.match TRACKING.MATCH
                            Options: hungarian, greedy (default: greedy)
      --tracking.track_window TRACKING.TRACK_WINDOW
                            How many frames back to look for matches (default: 5)
      --tracking.min_new_track_points TRACKING.MIN_NEW_TRACK_POINTS
                            Minimum number of instance points for spawning new
                            track (default: 0)
      --tracking.min_match_points TRACKING.MIN_MATCH_POINTS
                            Minimum points for match candidates (default: 0)
      --tracking.img_scale TRACKING.IMG_SCALE
                            For optical-flow: Image scale (default: 1.0)
      --tracking.of_window_size TRACKING.OF_WINDOW_SIZE
                            For optical-flow: Optical flow window size to consider
                            at each pyramid (default: 21)
      --tracking.of_max_levels TRACKING.OF_MAX_LEVELS
                            For optical-flow: Number of pyramid scale levels to
                            consider (default: 3)
      --tracking.kf_node_indices TRACKING.KF_NODE_INDICES
                            For Kalman filter: Indices of nodes to track.
                            (default: )
      --tracking.kf_init_frame_count TRACKING.KF_INIT_FRAME_COUNT
                            For Kalman filter: Number of frames to track with
                            other tracker. 0 means no Kalman filters will be used.
                            (default: 0)


Dataset files
---------------

.. _`sleap-convert`:

``sleap-convert``
+++++++++++++++++

:code:`sleap-convert` allows you to convert between various dataset file formats. Amongst other things, it can be used to export data from a SLEAP dataset into an HDF5 file that can be easily used for analysis (e.g., read from MATLAB). See :py:mod:`sleap.io.convert` for more information.

.. code-block:: none

    usage: sleap-convert [-h] [-o OUTPUT] [--format FORMAT] [--video VIDEO]
                         input_path

    positional arguments:
      input_path            Path to input file.

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            Path to output file (optional).
      --format FORMAT       Output format. Default ('slp') is SLEAP dataset;
                            'analysis' results in analysis.h5 file; 'h5' or 'json'
                            results in SLEAP dataset with specified file format.
      --video VIDEO         Path to video (if needed for conversion).


For example, to convert a predictions SLP file to an analysis HDF5 file:

::

  sleap-convert --format analysis -o "session1.predictions.analysis.h5" "session1.predictions.slp"

See `Analysis examples <../notebooks/Analysis_examples.html>`_ for how to work with these outputs.


.. _`sleap-inspect`:

``sleap-inspect``
+++++++++++++++++

:code:`sleap-inspect` gives you various information about a SLEAP dataset file such as a list of videos and a count of the frames with labels. If you're inspecting a predictions dataset (i.e., the output from running :code:`sleap-track` or inference in the GUI) it will also include details about how those predictions were created (i.e., the models, the version of SLEAP, and any inference parameters).

You can also specify a model folder to get a quick summary of the configuration and metrics (if available).

.. code-block:: none

    usage: sleap-inspect [-h] [--verbose] data_path

    positional arguments:
      data_path   Path to labels file (.slp) or model folder

    optional arguments:
      -h, --help  show this help message and exit
      --verbose


Debugging
---------

.. _`sleap-diagnostic`:

``sleap-diagnostic``
++++++++++++++++++++

There's also a script to output diagnostic information which may help us if you need to contact us about problems installing or running SLEAP. If you were able to install the SLEAP Python package, you can run this script with :code:`sleap-diagnostic`. Otherwise, you can download `diagnostic.py <https://raw.githubusercontent.com/murthylab/sleap/main/sleap/diagnostic.py>`_ and run :code:`python diagnostic.py`.


.. code-block:: none

    usage: sleap-diagnostic [-h] [-o OUTPUT] [--gui-check]

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            Path for saving output
      --gui-check           Check if Qt GUI widgets can be used

.. note::

    For more details about any command, run with the :code:`--help` argument (e.g., :code:`sleap-track --help`).


