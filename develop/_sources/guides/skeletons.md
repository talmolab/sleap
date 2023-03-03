(skeletons)=

# Skeleton design

In SLEAP, skeletons are defined as a set of *nodes* (landmarks types) and *edges*
(connections between landmarks).

**Nodes** are essentially just a list of body part names. The actual naming doesn't
matter; SLEAP just uses their relative order to keep track of which body part is which
in the labels and in trained models.

**Edges** describe the way that the nodes are connected. Each edge is represented by a
*source* and *destination* node. With the exception of bottom-up models, these edges
serve primarily for visualization.

In bottom-up models, the edges are important and must connect all nodes to one another.


## Tips

**How do I choose the best skeleton for my data?**

In general, here are the rules of thumb you want to follow for designing an ideal
skeleton:

1. Choose *nodes* that will be easy to locate in new images. It's important to be as
   consistent as possible about the relative placement of body parts.
2. When choosing *edges*, try to form a shallow tree. This is because it's preferable to
   have fewer parent nodes, since if they are not detected, their children nodes cannot
   be grouped appropriately.

**Can I add or remove nodes in the skeleton after I've already created instances?**

Yes.

Removing nodes is straightforward: just remove them. If you're using part affinity fields for inference, you should make sure that the skeleton graph is still connected.

Adding nodes is a little more complicated. First add the nodes to your skeleton. Then, to add these nodes to any instance which already exists, you'll need to **double-click** on the instance (on the video frame image). The new nodes will be added and marked as "non-visible"; you'll need to **right-click** on each node you want to make visible, and move it to the correct location.

**Can I add or remove edges in the skeleton after I've already created instances?**

Yes, adding or removed edges is straightforward and the change will be applied to all instances.
