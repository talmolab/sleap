.. _notebooks:

Notebooks
---------

Here are Jupyter notebooks you can run to try SLEAP on `Google Colaboratory <https://colab.research.google.com>`_ (Colab). Colab is great for running training and inference on your data if you don't have access to a local machine with a supported GPU.

`Training and inference on an example dataset <./Training_and_inference_on_an_example_dataset.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this notebook we'll show you how to install SLEAP on Colab, download a dataset from the `repository of sample datasets <https://github.com/murthylab/sleap-datasets>`_, run training and inference on that dataset using the SLEAP command-line interface, and then download the predictions.
This notebook can be a good place to start since you'll be able to see how training and inference work without any of your own data and without having to edit anything in the notebook to get it to run correctly.

.. toctree::
    :maxdepth: 2

    Training_and_inference_on_an_example_dataset

`Using your own data on Google Drive <./Training_and_inference_using_Google_Drive.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you're ready to run training and inference on your own SLEAP dataset, this notebook walks you through the process of using `Google Drive <https://www.google.com/drive>`_ to copy data to and from Colab (as well as running training and inference on your dataset).

.. toctree::
    :maxdepth: 2

    Training_and_inference_using_Google_Drive

`Training with custom hyper-parameters <./Training_with_custom_hyperparameters.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After you're able to run training and inference on your own SLEAP dataset using Colab, you're ready to learn how to train with custom hyperparameters. This notebook will walk you through that process.

.. toctree::
    :maxdepth: 2

    Training_with_custom_hyperparameters

`Analysis examples <./Analysis_examples.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you've used SLEAP to successfully estimate animal pose and track animals in your videos, you'll want to use the resulting data. This notebook walks you through some analysis examples which illustrate how to read and interpret the data in the analysis HDF5 files which you can export from SLEAP.

.. toctree::
    :maxdepth: 2

    Analysis_examples
