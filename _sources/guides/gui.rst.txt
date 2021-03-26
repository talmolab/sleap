.. _gui:

GUI
====

The SLEAP labeling interface is accessible via the :code:`sleap_label` command (see :ref:`cli`).



Menus
-----

Note that many of the menu command have keyboard shortcuts which can be configured from "**Keyboard Reference**" in the "**Help**" menu.

File
----

"**New...**", "**Open...**", "Save, and "**Save As...**" have their usual behavior.

"**Import...**" allows you to import the data external formats into a new SLEAP dataset. This includes COCO_ keypoint detection (:code:`.json` files), DeepLabCut_ (:code:`.csv`), DeepPoseKit_ (:code:`.h5`), and LEAP_ (:code:`.mat`).

.. _COCO: http://cocodataset.org/#format-data
.. _DeepPoseKit: http://deepposekit.org
.. _DeepLabCut: http://deeplabcut.org
.. _LEAP: https://github.com/talmo/leap

"**Merge Data From...**" allows you to copy labels and/or predictions from another SLEAP dataset into the currently open project. This is useful because you can only train on data from a single SLEAP dataset, so you may need to import data before training. For instance, suppose you've trained a model and use it to get predictions on another video. You then open the predictions file in the GUI and realize that you'd like to correct some of the predictions and add those corrections to your training data. You can do this by importing data from the predictions file into the SLEAP project which has your training data.

"**Add Videos...**" allows you to add videos to your currently open project.

"**Replace Videos...**" allows you to *swap* the videos currently in your project with other videos. This is useful if you want to have your project access copies of the videos at a different path, e.g., if you copy the videos between a network drive and a local drive and want to change which is used by your project. This can also be used if you want to replace you videos with copies that have been re-encoded, cropped, or edited in some other way that doesn't affect the frame numbers (since your annotations will be placed directly on the corresponding frames of the new video).

Go
--

"**Next Labeled Frame**" will take you to the next frame of the video which has any labeled data—either labels added by the user or predictions.

"**Next User Labeled Frame**" will take you to the next frame which has labels added by the user, skipping any frames which are unlabeled or have only predictions.

"**Next Suggestion**" will take you to the next frame in the list of suggested frames. If you are currently on a suggested frame, it will take you to the next frame in the list, which may not be a later frame in the video (or even a frame in the same video). If you are not currently on a suggested frame, it will take you to the nearest subsequent suggested frame.

"**Next Track Spawn Frame**" will take you to the next frame on which a new track starts—i.e., a predicted instance which were not able to identify as the same instance from some prior frame. This is useful when proofreading predictions, since you can skip between track spawn frames and then manually join the new track with one of the pre-existing tracks from prior frames.

"**Next Video**" will show the next video in the project (if your project has multiple videos). All the videos are listed in the "Videos" window.

"**Go to Frame...**" allows you to go to a specific frame (in the current video) by frame number. It's also handy if you want to copy the current frame number to the clipboard.

"**Select to Frame...**" allows you to select the clip from the current frame to a specified frame. If you want to select from frames X to Y, you can use **Go to** to go to X then **Select to** to select from X to Y.

.. _view:

View
----

"**Color Predicted Instances**" allows you to toggle whether *predicted* instances are all shown in yellow, or whether to apply distinct colors to (e.g.) track identities. You should turn this on when proofreading predictions for a video.

"**Color Palette**" allows you to choose the palette which will be used for coloring instances. Most of the palettes cycle colors, so that if there are five colors in the palette, the sixth item to color will be the same as the first. A few things to know:

- The "alphabet" palette has 26 visually distinct colors, which can be useful when there are many items to color.

- If the palette name ends with "+", the colors won't cycle and everything beyond the number of colors in the palette will use the last color in the palette. This is especially useful for proofreading when you're trying to merge all the track identities in the first few tracks (assuming you have a small number of instances in each frame). In particular, the "five+" palette will show any instance in the fourth or subsequent track as orange.

- Color palettes can be customized by modifying the :code:`colors.yaml` file that SLEAP creates inside the :code:`.sleap` directory in your home directory. You can add your own palette or modify those already present in the file. Each color in a palette is listed on it's own line as r,g,b values (between 0 and 255).

"**Apply Distinct Colors To**" allows you to determine whether distinct colors are used for distinct tracks (instance identities), nodes, or edges. Try it!

"**Show Node Names**" allows you to toggle the visibility of the node names. This is useful if you have lots of nearby instances or very dense skeletons and the node names make it hard to see where the nodes are located.

"**Show Edges**" allows you to toggle the visibility of the edges which connect the nodes. This can be useful when you have lots of edges which make it hard to see the features of animals in your video.

"**Edge Style**" controls whether edges are drawn as thin lines or as wedges which indicate the :ref:`orientation` of the instance (as well as the direction of the part affinity field which would be used to predict the connection between nodes when using a "bottom-up" approach).

"**Trail Length**" allows you to show a trail of where each instance was located in prior frames (the length of the trail is the number of prior frames). This can be useful when proofreading predictions since it can help you detect swaps in the identities of animals across frames.

"**Fit Instances to View**" allows you to toggle whether the view is auto-zoomed to the instances in each frame. This can be useful when proofreading predictions.

"**Seekbar Header**" allows you to plot relevant information above the seekbar. Note that the seekbar is not updated when you modify instances; it only updates when you select a new seekbar header.

"**Videos**", "**Skeleton**", "**Instances**", and "**Labeling Suggestions**" allow you to toggle which information windows are shown (by default these are docked to the right side of the main GUI window).

Labels
------

"**Add Instance**" will add an instance to the current frame. You can also add an instance by right-clicking within the frame. For predicted instances, you can also "convert" the predicted instance to a regular, editable instance by double-clicking on the predicted instance (the predictions are still there, but they won't be shown unless you delete the editable instance you just created).

"**Instance Placement Method**" allows you to pick how we determine where to place the instance and its nodes on the video frame. "Best" (the default) will first check for predicted instances in the current frame and create a new editable instance from one of the predicted instances (if you add multiple instances, it will do this for each predicted instance in turn). Otherwise, it will copy the location from another instance in a prior or the current frame, or will use a force-directed graph algorithm to place the nodes. You can also choose the "average" method which creates an "average" instance from the instances you've already labeled.

"**Delete Instance**" will delete the currently selected instance (the selected instance will have an outline drawn around it and will be highlighted in the "Instances" window).

"**Set Instance Track**" sets the track for the currently selected instance. If the new track already has an instance assigned to it, then the tracks are swapped (the other instance is assigned to the track currently assigned to the selected instance). These changes are applied to instances in the same tracks in every subsequent frame, not just the current frame.

"**Transpose Instance Tracks**" swaps the tracks assigned to two instances. If there are only two instances in the current frame, then this command will be applied to those. If there are more then two instances, then you'll be prompted to select the two instances in sequence. (This has the same functionality as selecting an instance and using "**Set Instance Track**" with the track of the other instance).

"**Delete Instance and Track**" deletes all instances in the track of the currently selected instance. This applied to *all* frames in the current video.

"**Custom Instance Delete...**" allows you to delete all the instances which meet criteria you specify: whether they are user instances or predicted instances, which frames they are on, and which track identities they have.

"**Select Next Instance**" allows you to cycle through the instances in the current frame.

"**Clear Selection**" allows you to deselect the selected instance.

Predict
-------
"**Run Training...**" allows you to train a set of models from the data in your open project, and then optionally predict on some frames in the project.

"**Run Inference...**" allows you to generate predictions using a pre-trained set of models. Any trained models in the `models` directory next to your current project will be listed, and you also have the option to select models saved elsewhere.

"**Evaluate Metrics for Trained Models...**" provides you with information to evaluate all of your trained models. For example, you can see the recall and precision of predictions on frames with ground truth validation data (i.e., data withheld when training).

"**Visualize Model Outputs...**" allows you to select a trained model and see the intermediate inference data plotted on the frame image. In particular, looking at the confidence maps used to predict each node can be helpful for understanding why and where your model isn't preforming well.

"**Add Instances from All Predictions on Current Frame**" converts ever predicted instance on the current frame into a user editable instance (which allows you to make corrections and/or use it for training).

"**Delete All Predictions...**" deletes *all* predicted instances across *all* frames in the current video. (You'll be asked to confirm before the instances are deleted.)

"**Delete All Predictions from Clip...**" deletes all instances from within a selected range of frames. You can select a clip by shift-dragging in the seekbar (or shift + other movement key).

"**Delete All Predictions from Area...**" deletes all instances which are entirely within some rectangular area on any frame in the current video. You'll be asked to select the rectangle, and then asked to confirm before instances are deleted. This is useful when there's something in the video which is visually similar to the animal instances and which creates many false-positives.

"**Delete All Predictions with Low Score...**" deletes all instances with a prediction score below the specified value. You'll be asked for the value, and then asked to confirm before instances are deleted. Instance scores are shown in the "Instances" table and below the selected instance in the current frame.

"**Delete All Predictions beyond Frame Limit...**" deletes the lowest scoring instances beyond some set number of instances in each frame. For example, if you know that there are only two animals in the video, this would let you keep just the two best predicted instances. You'll be asked for the number of instances to keep, and then asked to confirm before instances are deleted.

"**Export Video with Visual Annotations...**" allows you to export a video clip with any instances drawn on the frame (much as you can see in the GUI). If you select a clip in the seekbar, just those frames will be included in the new video; otherwise the whole (current) video will be used.

Help
----
"**Keyboard Shortcuts**" allows you to view and change keyboard shortcuts for common menu and GUI actions.

Application GUI
---------------

Mouse
-----

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
---------------

**Right arrow** key: Move one frame forward

**Left arrow** key: Move one frame back

**Down arrow** key: Move a *medium* step forward (4 frames by default)

**Up arrow** key: Move a *medium* step backward (4 frames by default)

**Space** key: Move a *large* step forward (100 frames by default)

**/** key: Move a *large* step backward (100 frames by default)

**Home** key: Move to the first frame of the video

**End** key: Move to the last frame of the video

**Shift** + *any navigation key*: Select the frames over which you've moved

.. note::

    These keys are the defaults; you can configure them with **Keyboard Shortcuts** in the **Help** menu.

Selection Keys
--------------

*Number* (e.g., **2**) key: Select the instance corresponding to that number

**Escape** key: Deselect all instances

Seekbar
-------

**Shift + drag**: Select a range of frames

**Shift + click**: Clear frame selection

**Alt + drag**: Zoom into a range of frames

**Alt + click**: Zoom out so that all frames are visible in seekbar

.. _suggestion-methods:

Labeling Suggestions
---------------------

There are various methods to generate a list "suggested" frames for labeling or proofreading.

The **sample** method is a quick way to get some number of frames for every video in your project. You can tell it how many samples (frames) to take from each video, and whether they should be evenly spaced throughout a video (the "stride" sampling method) or randomly distributed.

The **image feature** method uses various algorithms to give you visually distinctive frames, since you will be able to train more robust models if the frames you've labeled are more representative of the visual variations in your videos. Generating suggestions based on image features can be slow.

The **prediction score** method will identify frames which have more than some number of instances predicted and where the instance prediction score is below some threshold. This method can be useful when proofreading frame-by-frame prediction results. The instance score depends on your specific skeleton so you'll need to look at the instance scores you're getting to decide an appropriate threshold.

The **velocity** method will identify frames where a predicted instance appears to move more than is typical in the video. This is based on the tracking results, so it can be useful for finding frames where the tracker incorrectly matched up two identities (since this will make the identity "jump").