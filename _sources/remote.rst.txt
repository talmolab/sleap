.. _remote:

Remote Machines
===============

It's easy to install **SLEAP** on a remote machine for running training or inference without the GUI. This could be on a cluster or in a colab notebook.


old
~~~

It’s also possible to run inference using the command line interface, which is
useful if you’re going to run on a cluster. The command to run inference on
an entire video is:

::

   python -m sleap.nn.inference \
      path/to/video.mp4 --with-tracking \
      -m path/to/models/your_confmap_model.json \
      -m path/to/models/your_paf_model.json \
      -m path/to/models/your_centroid_model.json

The predictions will be saved in path/to/video.mp4.predictions.h5,
which you can open from the GUI app. You can also import these predictions into
your project by opening your project and then using the "Import Predictions..."
command in the "Predict" menu.

Note that if you want to copy your trained models so that they can be used on
another machine (e.g., from a cluster), you'll need to copy both the json files
and the corresponding directories within your models directory.
For example, you might copy:

::

   190711_185300.centroids.UNet.n=113.json
   190711_183049.pafs.LeapCNN.n=226.json
   190711_181028.confmaps.UNet.n=226.json
   190711_181028.confmaps.UNet.n=226/
   190711_183049.pafs.LeapCNN.n=226/
   190711_185300.centroids.UNet.n=113/