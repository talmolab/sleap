.. _faq:

Frequently Asked Questions
==========================

Skeleton
--------

**Can I add or remove nodes in the skeleton after I've already created instances?**

Yes.

Removing nodes is straightforward: just remove them. If you're using part affinity fields for inference, you should make sure that the skeleton graph is still connected.

Adding nodes is a little more complicated. First add the nodes to your skeleton. Then, to add these nodes to any instance which already exists, you'll need to **double-click** on the instance (on the video frame image). The new nodes will be added and marked as "non-visible"; you'll need to **right-click** on each node you want to make visible, and move it to the correct location.

**Can I add or remove edges in the skeleton after I've already created instances?**

Yes, adding or removed edges is straightforward and the change will be applied to all instances.

Pretraining
-----------
**How can I download pretrained weights to use for my backbones?**
Normally, pretrained weights will be automatically downloaded when the network is trained. This can be an issue when training in an environment that does not have internet access (e.g., HPC clusters). A simple solution is to manually trigger downloading of the pretrained weights in the same environment (e.g., the head node of a cluster). These one-liners will trigger weight downloading for the appropriate backbones:

.. code-block:: bash

  python -c "from tensorflow.keras import applications; applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"

  python -c "from tensorflow.keras import applications; applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"
  python -c "from tensorflow.keras import applications; applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"
  python -c "from tensorflow.keras import applications; applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(256, 256, 3))"

  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.25)"
  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.5)"
  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.75)"
  python -c "from tensorflow.keras import applications; applications.MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.0)"

  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.35)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.5)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=0.75)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.0)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.3)"
  python -c "from tensorflow.keras import applications; applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3), alpha=1.4)"

  python -c "from tensorflow.keras import applications; applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))"
  python -c "from tensorflow.keras import applications; applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))"


Be sure to run these with the same ``python`` binary as will be used for training so the weights can be loaded from the disk appropriately.
