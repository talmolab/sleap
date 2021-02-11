.. _new-project:

Creating a project
---------------------------

When you first start SLEAP you’ll see a new, empty project.

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

.. _new-skeleton:

Creating a Skeleton
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

Continue to :ref:`initial-labeling`.

.. |image0| image:: ../_static/add-video.gif
.. |image1| image:: ../_static/video-options.gif
.. |image2| image:: ../_static/add-skeleton.gif