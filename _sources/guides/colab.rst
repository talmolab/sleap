.. _colab:

Run training and inference on Colab
--------------------------------------

*Case: You already have a project with labeled training data and you'd like to run training or inference in a Colab notebook.*

Take a look at our `example notebooks <https://sleap.ai/notebooks>`_ which you can run on Colab! These will walk you through the entire process of running training and inference on Colab.

About Colab
~~~~~~~~~~~

`Google Colaboratory <https://colab.research.google.com/>`_ is a hosted service from Google that lets you run Python code (like SLEAP) on Google's servers with access to GPUs for training and inference. It's free!

When you start running code in a notebook, Google connects your notebook to a virtual server instance. Think of this as a new computer with Python, TensorFlow and a few other things pre-installed. You can run things on this server instance for at most 12 hours (it will timeout sooner if idle but this depends on how much demand there is for Google's servers).

Once the server instance times out, you lose all the data that's stored on the server instance. This means that you'll need to install SLEAP from the notebook at the beginning of each new session, you'll need to move your data to the server instance, and you'll need to copy any results off the server instance when you're done.

You can also pay for `Colab Pro <https://colab.research.google.com/signup>`_ (currently available only in the US) which gives you priority access to better hardware and longer runtimes (they say 24 hours and more lenient idle timeouts).

Moving Data to/from Colab
~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to work with your own data on Colab is to use `Google Drive <https://www.google.com/drive/>`_. This is another service provided by Google. It's also free! If your university has `G Suite for Education <https://edu.google.com/products/gsuite-for-education/>`_ you'll have unlimited storage space.

Google Drive is similar to Dropbox, Microsoft OneDrive, or Apple iCloud. You can install software and you'll you get a special folder which is automatically synced: anything you copy into it will be available from other computers (using your account), and anything copied into it from elsewhere will automatically be downloaded into the folder on your computer.

For training on your own data, it's probably easiest to export a training job package from the `sleap-label` GUI. The training job package will contain model files as well as all of your labeled data and all of the images for labeled frames within a single file, so that you don't have to copy the full training videos and you don't have to worry about paths to the videos.

If you'd rather copy all your videos over to Colab—perhaps because you plan to run inference on the full videos—you can place the videos in the same directory as your SLEAP project file and as long as each video has a distinctive filename, SLEAP will be able to find them (it tries to locate the videos in the same directory as the project if it can't find the videos at the path they had when added to the project).
