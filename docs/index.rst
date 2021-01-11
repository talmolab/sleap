.. include:: ../README.rst

Getting started
--------------------------

Once you install SLEAP (see our :ref:`Installation` instructions) you should start by following the step-by-step :ref:`Tutorial`. This will guide you through the process of creating a project, adding videos, creating a skeleton for each animal in the video, labeling a few frames, and then training a model.

After installation you can also try running SLEAP on our `sample datasets <https://github.com/murthylab/sleap-datasets>`_—this allows you to try out training and inference before you’ve labeled your own data for training.

When you're ready to train you will have three choices for models: single animal, multi-animal top-down, or multi-animal bottom-up. To deal with more than one animal (multiple instances), we implement both “top-down” and “bottom-up” approaches.

In **top-down** mode, a network first finds each animal and then a separate network estimates the pose of each animal:

.. image:: _static/topdown_approach.jpg

In **bottom-up** mode, a network first finds all of the body parts in an image, and then another network groups them into instances using part affinity fields (`Cao et al., 2017 <https://arxiv.org/abs/1611.08050>`_):

.. image:: _static/bottomup_approach.jpg

We find that top-down mode works better for some multi-animal datasets while bottom-up works better for others. SLEAP uses UNET as its backbone, but you can choose other backbones (LEAP, resnet, etc.) from a drop down menu.

Once you have made the choice between a top-down or bottom-up approach and trained models, then you enter into a human-in-the-loop training cycle, in which you receive predictions and use them to continue to label more examples. You will continue to iterate this process until you have achieved the desired pose labeling accuracy for your dataset.

The goal at this stage is to get accurate pose predictions on individual frames. This means that the points for each body part are predicted in the correct locations and that the points are correctly grouped into distinct animal instances.

After you have accurate frame-by-frame prediction, you’re ready to predict for entire video clips and to track animal identities. We use a variety of heuristic algorithms for tracking identities across time (see :ref:`tracking-method-details` for more details). SLEAP also includes a graphical proof-reading tool for quickly assessing the accuracy of tracking and correcting problems.

.. _Learn more:

Learn more
----------------------

Want to learn more about SLEAP and multi-animal pose tracking?

Check out our `preprint on bioRxiv <https://doi.org/10.1101/2020.08.31.276246>`_ or watch our `tutorial on SLEAP <https://cbmm.mit.edu/video/decoding-animal-behavior-through-pose-tracking>`_.


.. _Contact:

Contact
-------

Follow `@MurthyLab <https://twitter.com/MurthyLab>`_ on Twitter for news and updates!

**Technical issue with the software?** `Open an issue on GitHub. <https://github.com/murthylab/sleap/issues>`_

**Press inquiries? Interested in using SLEAP in a commercial application?** Reach out at `sleap@princeton.edu`_.

.. _sleap@princeton.edu: sleap@princeton.edu


.. _Contributors:

Contributors
------------

* **Talmo Pereira**, Princeton Neuroscience Institute, Princeton University
* **Nat Tabris**, Princeton Neuroscience Institute, Princeton University
* **David Turner**, Research Computing and Princeton Neuroscience Institute, Princeton University
* **Joshua Shaevitz**, Physics and Lewis-Sigler Institute, Princeton University
* **Mala Murthy**, Princeton Neuroscience Institute, Princeton University

SLEAP was developed in the Murthy and Shaevitz labs at Princeton University. Funding: NIH BRAIN Initative R01 NS104899 and Princeton Innovation Accelerator Fund.

Documentation
-------------

.. _sleap:
.. toctree::
    :maxdepth: 2

    installation
    tutorials/tutorial
    guides/index
    guides/reference
    API <api>
    GitHub <https://github.com/murthylab/sleap>
    Releases <https://github.com/murthylab/sleap/releases>

.. _Indices_and_Tables:

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`