.. _`high-level overview`:

High-level Overview
===================

.. image:: _static/workflow.png

In a nutshell, SLEAP can be used to analyze animal behaviour from recorded video clips.
A typical end-to-end SLEAP workflow consists of the following steps:

1. Creating a new project and importing video file(s)
    You may import all or part of the video clips from your experiment footage. These video files will be used to build a ground-truth dataset and train the machine learning model that estimates animal pose. See details :here:
2. Defining animal skeleton(s)
    In this step you will define the animal skeleton by listing its body parts and their connections. See details :here:
3. Selecting frames from imported video(s) for initial labeling
    Next you will select the initial set of frames for the labeling activity. SLEAP provides several options for selecting frames based on sampling or image features. See details :here:
4. Manually labeling animal pose(s) in selected frames
    Labeling is the activity of mapping skeleton body parts to animal instances in the selected frames. This is the most laborious part, however SLEAP helps accelerate the process with a purpose-built user interface and iterative human-in-the-loop process that involves training machine learning models on partially labeled data and manually correcting its predictions. See details :here:
5. Training a machine learning model using the labeled frames
    After we labeled a few frames we can train a machine learning model. SLEAP supports several approaches for training, each with it's own set of parameters that can be configured. See :training:.
6. Applying the trained model to predict animal poses in unlabeled frames (inference)
    Once the training is complete, the trained model is applied to all unlabeled frames to predict animal pose (this step is also called *inference*). The accuracy of these predictions depends on many parameters, among them the quality of the labeling work, amount of labeled frames, and the configuration of the training job. See details :here:
7. Refining the predicted labels manually, and repeating the training step until desired model accuracy is achieved
    In this step you can inspect the predictions and correct them - this is similar to the labeling step, but should be easier since the predictions place the skeletons approximately right. See details :here:
8. [Optional] Importing additional videos from your experiment, and applying the trained model to predict animal poses
    Once the machine learning model performance is satisfactory (w.r.t. pose estimation quality), the next step is to predict the animal poses across all video frames from your experiment. This step is only needed if not all video(s) were imported in the first step.
9. Applying the tracking algorithm to track animal instances across frames
    Tracking associates animal instances in consequent frames. Here too SLEAP provides several algorithms for tracking with their own configuration parameters. See :tracking:.
10. Proofreading and potentially correcting instance tracks
     This is a manual step where you can use SLEAP GUI to verify that the tracking is accurate, and make corrections as needed. See :ref:`track_proofreading` for types of erros that may occur and how to correct them.
11. Exporting data for analysis
     Finally you can export the generated data (including animal instance occupancy matrices and tracks) for further analysis (e.g. in Matlab or Python). See :ref:`Export Data For Analysis` and example :ref:`Notebooks` for details.
