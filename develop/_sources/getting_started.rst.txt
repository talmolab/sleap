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
