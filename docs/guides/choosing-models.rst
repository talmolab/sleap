.. _choosing_models:

Choosing a set of models
~~~~~~~~~~~~~~~~~~~~~~~~~

Inference will run in different modes depending on the output types of the models you supply. SLEAP currently support two distinct modes for multi-animal inference.

1. The "**bottom-up**" approach uses a single model which outputs **confidence maps** and **part affinity fields** for all instances in a given frame. The confidence maps are used to predict node locations and the part affinity fields are used to group nodes into distinct animal instances.

2. The "**top-down**" approach starts by using a **centroid** model to predict the location of each animal in a given frame, and then a **instance centered confidence map** model is used to predict the locations of all the nodes for each animal separately.

Each approach has its advantages, and you may wish to try out both to see which gives you better results.
