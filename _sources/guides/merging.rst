.. _merging:

Importing predictions for labeling
==================================

*Case: You have predictions that aren't in the same project as your original training data and you want to correct some of the predictions and use these corrections to train a better model.*

All of your training data must be in a single SLEAP project file (or labels package), so if you have data in multiple files, you'll need to merge them before you can train on the entire set of data.

When you run inference from the GUI, the predictions will be added to the same project as your training data (they'll also be saved in a separate file). When you run inference from the command-line, they'll only be in a separate file.

If you open a separate predictions file, make corrections there and train a new model from that file, then new models will be trained from scratch using only those corrections. The new models will not be trained on any of the original data that was used to train the previous models—i.e., the models used to generate these predictions. Usually you'll want to include both the original data and the new corrections.

**Note** that uncorrected predictions will never be used for training. Only predictions which you've "converted" into an editable instance will be used for training. To convert a predicted instance into an editable instance, you can **double-click** on the predicted instance or use the "**Add Instance**" command in the "Labels" menu (there's also a keyboard shortcut). As you might guess, once you have an editable instance you can move nodes and toggle their "visibility" (see the :ref:`tutorial` if you're not familiar with how to do this). When you've created an editable instance from a predicted instance, the predicted instance will no longer be shown, although it will re-appear if you delete the editable instance.

Let's suppose we have a project file and a predictions file with corrections, and we'd like to merge the corrections into the original project file.

If you want to merge *only* the corrections, then you should first make a copy of the predictions file. You can either just copy the file itself, or make a copy from the GUI using "**Save As..**" in the "File" menu. Open the copy of the file in the GUI and use the "**Delete All Predictions...**" command in the "Predictions" menu to remove all of the predicted instances from the file. Save and you'll be left with a file which just contains your corrections.

Open the original project file (or whatever file you want to merge **into**). Then, use the "**Merge Data From...**" command in the "File" menu. You'll need to select the file **from which** you are getting the data to merge—this would be the file with your corrected predictions.

You'll then see a window with information about the merge:

|clean-merge|

If there are no merge conflicts, then you can click "**Finish Merge**. If the two files contain conflicts—frames from the same video which both have editable instances or both have predicted instances—then you'll need to decide how to resolve the conflicts. You can choose to use the "base" version (i.e., the original project file **into which** you are merging), the "new" version (i.e., from the predictions file with the data which you're adding to the original project), or neither. Whichever you choose, you'll also get all of the frames which can be merged without conflicts.

After merging you should save (or save a copy of the project with the "**Save As...**" command). Once you have a single project file which contains both your old and new training data, you can train new models.

.. |clean-merge| image:: ../_static/clean-merge.jpg
