.. _reference:

Feature Reference
=================

Command Line Interfaces
-----------------------

:code:`sleap-label` (or :code:`python -m sleap.gui.app`) runs the GUI application.

:code:`sleap-train` (or :code:`python -m sleap.nn.train`) is the command-line interface for *training*. Use this for training on a remote machine/cluster/colab notebook.

:code:`sleap-track` (or :code:`python -m sleap.nn.inference`) is the command-line interface for running *inference* using models which have already been trained. Use this for running inference on a remote machine such as an HPC cluster or Colab notebook. All training parameters are exposed.

:code:`python -m sleap.nn.tracking` allows you to run the cross-frame identity tracker (or re-run with different parameters) without needed to re-run inference. You give it a prediction file.

:code:`python -m sleap.info.trackcleaner` is an experimental script which tries to clean the resuls of cross-frame identity tracking by connecting "breaks" where we lose one identity and spawn another. You specify how many identities there should be in a frame (i.e., the number of animals).

:code:`python -m sleap.gui.training_editor` allows you to view and create new training profiles. These are the files which specify what the model will be used for (confidence maps, part affinity fields, centroids, or top-down confidence maps), the network architecture (e.g., UNet), and the other training parameters (e.g., learning rate, image rescaling, image augmentation methods). If you want to view an existing profile—including a `training_job.json` file associated with a trained model—you can specify it's path as the first command-line parameter. You can also do this if you want to use an exiting profile as a template for creating a new training profile.

:code:`python -m sleap.info.write_tracking_h5` allows you to export the tracking data from a SLEAP dataset into an HDF5 file that can be easily used for analysis (e.g., read from MATLAB).

**Note**: For more details about any command, run with the :code:`--help` argument (e.g., :code:`sleap-track --help`).

Menus
-----

Note that many of the menu command have keyboard shortcuts which can be configured from "**Keyboard Reference**" in the "**Help**" menu.

File
~~~~

"**New...**", "**Open...**", "Save, and "**Save As...**" have their usual behavior.

"**Import...**" allows you to import the data external formats into a new SLEAP dataset. This includes COCO_ keypoint detection and DeepPoseKit_.

.. _COCO: http://cocodataset.org/#format-data
.. _DeepPoseKit: http://deepposekit.org

"**Merge Data From...**" allows you to copy labels and/or predictions from another SLEAP dataset into the currently open project. This is useful because you can only train on data from a single SLEAP dataset, so you may need to import data before training. For instance, suppose you've trained a model and use it to get predictions on another video. You then open the predictions file in the GUI and realize that you'd like to correct some of the predictions and add those corrections to your training data. You can do this by importing data from the predictions file into the SLEAP project which has your training data.

"**Add Videos...**" allows you to add videos to your currently open project.

Go
~~

"**Next Labeled Frame**" will take you to the next frame of the video which has any labeled data—either labels added by the user or predictions.

"**Next User Labeled Frame**" will take you to the next frame which has labels added by the user, skipping any frames which are unlabeled or have only predictions.

"**Next Suggestion**" will take you to the next frame in the list of suggested frames. If you are currently on a suggested frame, it will take you to the next frame in the list, which may not be a later frame in the video (or even a frame in the same video). If you are not currently on a suggested frame, it will take you to the nearest subsequent suggested frame.

"**Next Track Spawn Frame**" will take you to the next frame on which a new track starts—i.e., a predicted instance which were not able to identify as the same instance from some prior frame. This is useful when proofreading predictions, since you can skip between track spawn frames and then manually join the new track with one of the pre-existing tracks from prior frames.

"**Next Video**" will show the next video in the project (if your project has multiple videos). All the videos are listed in the "Videos" window.

"**Go to Frame...**" allows you to go to a specific frame (in the current video) by frame number. It's also handy if you want to copy the current frame number to the clipboard.

.. _view:

View
~~~~

"**Color Predicted Instances**" allows you to toggle whether *predicted* instances are all shown in yellow, or whether to apply distinct colors to (e.g.) track identities. You should turn this on when proofreading predictions for a video.

"**Color Palette**" allows you to choose the palette which will be used for coloring instances. Most of the palettes cycle colors, so that if there are five colors in the palette, the sixth item to color will be the same as the first. A few things to know:

- The "alphabet" palette has 26 visually distinct colors, which can be useful when there are many items to color.

- If the palette name ends with "+", the colors won't cycle and everything beyond the number of colors in the palette will use the last color in the palette. This is especially useful for proofreading when you're trying to merge all the track identities in the first few tracks (assuming you have a small number of instances in each frame). In particular, the "five+" palette will show any instance in the fourth or subsequent track as orange.

- Color palettes can be customized by modifying the :code:`colors.yaml` file that SLEAP creates inside the :code:`.sleap` directory in your home directory. You can add your own palette or modify those already present in the file. Each color in a palette is listed on it's own line as r,g,b values (between 0 and 255).

"**Apply Distinct Colors To**" allows you to determine whether distinct colors are used for distinct tracks (instance identities), nodes, or edges. Try it!

"**Show Node Names**" allows you to toggle the visibility of the node names. This is useful if you have lots of nearby instances or very dense skeletons and the node names make it hard to see where the nodes are located.

"**Show Edges**" allows you to toggle the visibility of the edges which connect the nodes. This can be useful when you have lots of edges which make it hard to see the features of animals in your video.

"**Edge Style**" controls whether edges are drawn as thin lines or as wedges which indicate the :ref:`orientation` of the instance (as well as the direction of the part affinity field which would be used to predict the connection between nodes).

"**Show Trails**" allows you to toggle the visibility of trail lines, which show where each node was located in recent prior frames. This can be useful when proofreading predictions since it can help you detect swaps in the identities of animals across frames.

"**Trail Length**" allows you to control how many prior frames to include in the trails.

"**Fit Instances to View**" allows you to toggle whether the view is auto-zoomed to the instances in each frame. This can be useful when proofreading predictions.

"**Seekbar Header**" allows you to plot relevant information above the seekbar. Note that this doesn't currently work well for very long videos since you can't zoom the seekbar. Also note that the seekbar is not updated when you modify instances; it only updates when you select a new seekbar header. (These issues will be fixed in a future version.)

"**Videos**", "**Skeleton**", "**Instances**", and "**Labeling Suggestions**" allow you to toggle which information windows are shown (by default these are docked to the right side of the main GUI window).

Labels
~~~~~~

"**Add Instance**" will add an instance to the current frame. You can also add an instance by right-clicking within the frame. For predicted instances, you can also "convert" the predicted instance to a regular, editable instance by double-clicking on the predicted instance (the predictions are still there, but they won't be shown unless you delete the editable instance you just created).

"**Instance Placement Method**" allows you to pick how we determine where to place the instance and its nodes on the video frame. "Best" (the default) will first check for predicted instances in the current frame and create a new editable instance from one of the predicted instances (if you add multiple instances, it will do this for each predicted instance in turn). Otherwise, it will copy the location from another instance in a prior or the current frame, or will locate the nodes randomly (somewhere within the currently visible portion of the current video frame, so you can zoom in to where you want the instance to be place).

"**Delete Instance**" will delete the currently selected instance (the selected instance will have an outline drawn around it and will be highlighted in the "Instances" window).

"**Set Instance Track**" sets the track for the currently selected instance. If the new track already has an instance assigned to it, then the tracks are swapped (the other instance is assigned to the track currently assigned to the selected instance). These changes are applied to instances in the same tracks in every subsequent frame, not just the current frame.

"**Transpose Instance Tracks**" swaps the tracks assigned to two instances. If there are only two instances in the current frame, then this command will be applied to those. If there are more then two instances, then you'll be prompted to select the two instances in sequence. (This has the same functionality as selecting an instance and using "**Set Instance Track**" with the track of the other instance).

"**Delete Instance and Track**" deletes all instances in the track of the currently selected instance. This applied to *all* frames in the current video.

"**Select Next Instance**" allows you to cycle through the instances.

"**Clear Selection**" allows you to deselect the selected instance.

Predict
~~~~~~~
"**Run Training...**" allows you to train a set of models from the data in your open project, and then optionally predict on some frames in the project.

"**Run Inference...**" allows you to generate predictions using a pre-trained set of models. Any trained models in the `models` directory next to your current project will be listed.

"**Expert Controls...**" is a less-friendly interface for running training and/or inference, and allows you to select and create custom training profiles.

"**Visualize Model Outputs...**" allows you to load a single trained model and see visualize its output—i.e., confidence maps, centroids, or part affinity fields—on frames of your current video. This is useful for understanding where your individuals models are doing well and where they're doing poorly. This feature is slower but usable without a GPU. Also note that you can only visualize one model at a time, and you have to quit when you want to stop visualizing model outputs.

"**Delete All Predictions...**" deletes *all* predicted instances across *all* frames in the current video. (You'll be asked to confirm before the instances are deleted.)

"**Delete All Predictions from Clip...**" deletes all instances from within a selected range of frames. You can select a clip by shift-dragging in the seekbar (or shift + other movement key).

"**Delete All Predictions from Area...**" deletes all instances which are entirely within some rectangular area on any frame in the current video. You'll be asked to select the rectangle, and then asked to confirm before instances are deleted. This is useful when there's something in the video which is visually similar to the animal instances and which creates many false-positives.

"**Delete All Predictions with Low Score...**" deletes all instances with a prediction score below the specified value. You'll be asked for the value, and then asked to confirm before instances are deleted. Instance scores are shown in the "Instances" table and below the selected instance in the current frame.

"**Delete All Predictions beyond Frame Limit...**" deletes the lowest scoring instances beyond some set number of instances in each frame. For example, if you know that there are only two animals in the video, this would let you keep just the two best predicted instances. You'll be asked for the number of instances to keep, and then asked to confirm before instances are deleted.

"**Export Training Package...**" allows you to export a training package. This is a single HDF5 file which contains both labeled data as well as the images which will be used for training. This makes it easy to transport your training data, especially if you need to run training on another machine (e.g., an HPC cluster). Training packages can be opened just like regular SLEAP dataset files, although you'll only be able to view the frames which have labeled data (since only these are included in the file).

"**Export Labeled Clip...**" allows you to export a video clip with any instances drawn on the frame (much as you can see in the GUI). To use this command, first select a clip in the seekbar.

Help
~~~~
"**Keyboard Reference**" allows you to view and change keyboard shortcuts for common menu and GUI actions.

Main GUI Window
---------------

Mouse
~~~~~

**Right-click** (or control + click) on node: Toggle visibility

**Right-click** (or control + click) elsewhere on image: Add instance (with pop-up menu)

**Alt + drag**: Zoom into region

**Alt + double-click**: Zoom out

**Alt + drag** on node (or node label): Move entire instance

**Alt + click and hold** on node (or node label) **+ mouse wheel**: Rotate entire instance

(On a Mac, substitute **Option** for **Alt**.)

**Double-click** on predicted instance: Create new editable instance from prediction

**Double-click** on editable instance: Any missing nodes (nodes added to the skeleton after this instance was created) will be added and marked as "non-visible"

**Click** on instance: Select that instance

**Click** elsewhere on image: Clear selection

Navigation Keys
~~~~~~~~~~~~~~~

**Right arrow** key: Move one frame forward

**Left arrow** key: Move one frame back

**Down arrow** key: Move 50 frames forward

**Up arrow** key: Move 50 frames back

**Space** key: Move 500 frames forward

**Home** key: Move to the first frame of the video

**End** key: Move to the last frame of the video

**Shift** + *any navigation key*: Select the frames over which you've moved

Selection Keys
~~~~~~~~~~~~~~

*Number* (e.g., **2**) key: Select the instance corresponding to that number

**Escape** key: Deselect all instances

Seekbar
~~~~~~~

**Shift + drag**: Select a range of frames

**Shift + click**: Clear frame selection