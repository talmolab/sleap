.. _skeletons:

Skeleton design
================

**Can I add or remove nodes in the skeleton after I've already created instances?**

Yes.

Removing nodes is straightforward: just remove them. If you're using part affinity fields for inference, you should make sure that the skeleton graph is still connected.

Adding nodes is a little more complicated. First add the nodes to your skeleton. Then, to add these nodes to any instance which already exists, you'll need to **double-click** on the instance (on the video frame image). The new nodes will be added and marked as "non-visible"; you'll need to **right-click** on each node you want to make visible, and move it to the correct location.

**Can I add or remove edges in the skeleton after I've already created instances?**

Yes, adding or removed edges is straightforward and the change will be applied to all instances.