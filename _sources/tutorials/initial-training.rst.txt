.. _initial-training:

Quick Training and Inference
----------------------------

Let's see how quickly we can train models and start getting some predictions.

Start by labeling about 10 frames—this should be enough to at least get some initial results. Once you have the frames labeled, save your project.

Now you're ready to train some models! To run training, select “**Run Training…**” from the “Predict”
menu. For this tutorial, let's use the default settings for training and predict on 20 random frames.

|image6|

The default settings will train two models: one for locating each instance in the frame confidence maps, and one for locating the parts for each of those instances. The models will be trained in that order.

For this "topdown" prediction approach, it's a good idea to choose an **anchor** part which has a relatively stable position near the center of your animal. You may also want to turn on the option to "**Visualize Predictions During Training**" (although this will make training run a bit more slowly).

Once you hit the **Run** button, you should see a window which shows you a graph of training and validation loss for each model as it trains.

Just for this tutorial, let's stop each training session after about 10 epochs. This should take a minute or two for each model (assuming you have a usable GPU!), and should be good enough to get some initial predictions.

After each model is trained, inference will run and if everything is successful, you should get a dialog telling you how many frames got predictions. Frames with labels will be marked in the seekbar, so try clicking on the newly marked frames or use the "**Next Labeled Frame**" command in the "Go" menu to step through frames with labels.

Continue to :ref:`assisted-labeling`.

.. |image6| image:: ../_static/training-dialog.jpg