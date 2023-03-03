.. _notebooks:

Notebooks
=========

Here are Jupyter notebooks you can run to try SLEAP on `Google Colaboratory <https://colab.research.google.com>`_ (Colab). Colab is great for running training and inference on your data if you don't have access to a local machine with a supported GPU.

Basic usage
------------

`Training and inference on an example dataset <./Training_and_inference_on_an_example_dataset.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this notebook we'll show you how to install SLEAP on Colab, download a dataset from the `repository of sample datasets <https://github.com/talmolab/sleap-datasets>`_, run training and inference on that dataset using the SLEAP command-line interface, and then download the predictions.
This notebook can be a good place to start since you'll be able to see how training and inference work without any of your own data and without having to edit anything in the notebook to get it to run correctly.


`Training and inference on your own data using Google Drive <./Training_and_inference_using_Google_Drive.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you're ready to run training and inference on your own SLEAP dataset, this notebook walks you through the process of using `Google Drive <https://www.google.com/drive>`_ to copy data to and from Colab (as well as running training and inference on your dataset).


`Analysis examples <./Analysis_examples.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you've used SLEAP to successfully estimate animal pose and track animals in your videos, you'll want to use the resulting data. This notebook walks you through some analysis examples which illustrate how to read and interpret the data in the analysis HDF5 files which you can export from SLEAP.

Advanced SLEAPing
------------------

`Data structures <./Data_structures.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SLEAP uses a set of core data structures that contain labels, predictions and other metadata. In this notebook we show you how to use them to develop custom analysis scripts and applications.

We demonstrate how to work with these data structures by interactively generating predictions from a trained model.


`Post-inference tracking <./Post_inference_tracking.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this notebook, we show how to use the tracking functionality within SLEAP to re-track existing predictions. This is useful when experimenting with new ID tracking parameters without having to re-run pose estimation.


`Interactive and resumable training <./Interactive_and_resumable_training.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training in SLEAP can be done via the GUI, CLI or interactively in Python. Here we show how to use SLEAP's Python API to enable customizable training workflows, including resumable training for initialization from existing models.


`Interactive and realtime inference <./Interactive_and_realtime_inference.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have trained models, you can run inference via the GUI, CLI or interactively in Python. Here we show how to load trained models, use them to predict on new frames, and implement a basic version of a realtime SLEAP tracker for closed-loop applications.


`Model evaluation <./Model_evaluation.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After you've trained several models, you may want to compute some metrics for benchmarking and comparisons. This notebook walks through some of the types of metrics that SLEAP can compute for you, as well as how to recompute them.


.. toctree::
    :hidden:
       
    Training_and_inference_on_an_example_dataset
    Training_and_inference_using_Google_Drive
    Model_evaluation
    Analysis_examples
    Data_structures
    Post_inference_tracking
    Interactive_and_resumable_training
    Interactive_and_realtime_inference
