.. _tutorial:

Tutorial
========

Before you can use SLEAP, you’ll need to install it. Follow the
instructions at :ref:`Installation` to install SLEAP and
start the GUI app.

There are three main stages of using SLEAP:

1. Creating a project, opening a movie and defining the skeleton;

2. Labeling and learning, labeling of video frames assisted by network
   predictions;

3. Prediction and proofreading, final network predictions of body-part
   positions and proofreading of track identities in full videos.

The tutorial will walk you through this entire process. You can read through the tutorial and then try running SLEAP on our `sample datasets <https://github.com/murthylab/sleap-datasets>`_—this will allow you to try out training and inference before you’ve labeled your own data for training.

.. toctree::
    :numbered:
    
    new-project
    initial-labeling
    initial-training
    assisted-labeling
    proofreading
    analysis