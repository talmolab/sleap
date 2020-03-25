.. _new-project:

Stage 1: Creating a project
---------------------------

When you first start SLEAP you’ll see an open dialog. Since you don’t
yet have a project to open, click “**Cancel**” and you’ll be left with a
new, empty project.

Opening a video
~~~~~~~~~~~~~~~

Add a **video** by clicking the “**Add Video**” button in the “Videos” panel
on the right side of the main window.

|image0|

You’ll then be able to select one or more video files and click “**Open**”.
SLEAP currently supports mp4, avi, and h5 files. For mp4 and avi files,
you’ll be asked whether to import the video as grayscale. For h5 files,
you’ll be asked the dataset and whether the video is stored with
channels first or last.

|image1|

Creating the Skeleton
~~~~~~~~~~~~~~~~~~~~~

Create a new **skeleton** using the “Skeleton” panel on the right side
of the main window.

Use the “**New Node**” button to add a node (i.e., joint or body part).
Double-click the node name to rename it (hit enter after you type the
new name). Repeat until you have created all your nodes. You then need
to connect the nodes with edges. Directly to the left of the “Add edge”
button you’ll see two drop-down menus. Use these to select a pair of
nodes, and then click “**Add Edge**”. Repeat until you’ve entered all the
edges.

|image2|

Stage 2a: Initial Labeling
--------------------------

We start by assembling a candidate group of images to label. You can
either pick your own frames or let the system suggest a set of frames
using the “Labeling Suggestions” panel. SLEAP can give you random or
evenly-spaced samples, or it can try to give you distinctive groups of
frames by taking the image features into account.

For now, let's just get 20 random frames. Choose "sample" as the method and "random" as the sampling method, then click "**Generate Suggestions**".

|image3|

Labeling the first frame
~~~~~~~~~~~~~~~~~~~~~~~~

Start by adding an **instance** of the skeleton to the current image by
clicking the “**Add Instance**” button in the Instances panel. The
first instance will have its points located at random. Move the points
to their appropriate positions by dragging with the mouse. Use the mouse
scroll-wheel to **zoom**.

|image4|

You can **move the entire instance** by holding down the Alt key while
you click and drag the instance. You can **rotate the instance** by
using the scroll-wheel while holding down the Alt key (or Option on a Mac).

For body parts that are not visible in the frame, right-click the node
(or its name) to **toggle visibility**. The node will appear smaller to show
that it’s marked as “not visible”. If you can determine where an
occluded node would be in the image, you may label it as “visible” in
order to train the model to predict the node even when it’s occluded.

|image5|

Saving
~~~~~~

Since this is a new project, you’ll need to select a location and name
the first time you save. SLEAP will ask you to save before closing any
project that has been changed to avoid losing any work. Note: There is
not yet an “undo” feature built into SLEAP. If you want to make
temporary changes to a project, use the “**Save As…**” command first to save
a copy of your project.

Labeling more frames
~~~~~~~~~~~~~~~~~~~~

After labeling the first frame saving the project, it’s time to label
more frames. Node positions will be copied from the instances in the
prior labeled frame to increase labeling speed. Since you generated a list
of suggested frames, you can go to the next frame in the labeling set by clicking “**Next**” under the list of suggested frames.

You can also always pick a frame to label by using the seekbar under
the video.

Try adding an instance by **right-clicking** on the location of the animal in the video. You'll see a pop-up menu with options for how we determine the initial node placement. Feel free to try the different options.

There’s no need to be consistent about which animal you label with which
instance for the case of multiple animals. For instance, suppose you
have a male and a female. It’s fine to label the male with the blue
instance in some frames and the orange instance in others. Tracking (and
track proofreading) is the final stage in the workflow and occurs after
predicting body part locations.

When you label a frame, it’s best if you can label all the instances of
your animal in the frame. Otherwise, the models may learn to not
identify things that look like the instances you didn’t label.

.. |image0| image:: ../_static/add-video.gif
.. |image1| image:: ../_static/video-options.gif
.. |image2| image:: ../_static/add-skeleton.gif
.. |image3| image:: ../_static/suggestions.jpg
.. |image4| image:: ../_static/labeling.gif
.. |image5| image:: ../_static/toggle-visibility.gif
.. |image6| image:: ../_static/training-dialog.jpg
.. |model| image:: ../_static/training-model-dialog.jpg
.. |receptive-field| image:: ../_static/receptive-field.jpg
.. |imagefix| image:: ../_static/fixing-predictions.gif
.. |tracker| image:: ../_static/tracker.jpg
.. |model-selection| image:: ../_static/model-selection.jpg
.. |image9| image:: ../_static/fixing-track.gif