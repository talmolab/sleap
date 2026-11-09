# Skeleton Design

A well-designed skeleton is crucial for accurate pose estimation. This guide covers best practices for creating effective skeletons in SLEAP.

---

## Skeleton Components

| Component | Description | Role |
|-----------|-------------|------|
| **Nodes** | Body part landmarks (e.g., "nose", "tail_base") | Define what points to track |
| **Edges** | Connections between nodes | Visualization + bottom-up grouping |

!!! note "Node naming"
    Node names are for your reference only—SLEAP uses their order internally. Choose descriptive names that make labeling intuitive.

---

## Design Guidelines

### Choosing Nodes

Choose nodes that will be easy to locate in new images. It's important to be as consistent as possible about the relative placement of body parts.

### Designing Edges

Edges connect nodes into a graph structure. For most models, edges are primarily for **visualization**. However, for **bottom-up models** (using part affinity fields), edges are critical and must connect all nodes to one another.

!!! tip "Shallow trees work best"
    Create a shallow hierarchy with few parent nodes. If a parent node isn't detected, its children can't be grouped correctly in bottom-up inference.

**Good structure:**
```
        head
       / | \
    nose ear  ear
      |
    spine
   /  |  \
 leg leg tail
```

**Avoid deep chains:**
```
head → neck → spine → hip → leg → foot → toe  ❌
```

---

## Modifying Skeletons

### Adding Nodes

1. Add the new node(s) to your skeleton definition
2. **Double-click** an existing instance (on the video frame) to edit it
3. New nodes will be added and marked as "non-visible"
4. **Right-click** each new node to make it visible, then drag to the correct location

### Removing Nodes

Simply delete the node from the skeleton. Existing instances will update automatically.

!!! warning "Part affinity fields"
    If using part affinity fields (bottom-up inference), ensure your skeleton graph remains fully connected after removing nodes.

### Modifying Edges

Add or remove edges freely—changes apply to all instances automatically. Edges don't affect existing labels, only visualization and bottom-up inference.

---

## Examples

### Mouse (simple)
```
Nodes: nose, left_ear, right_ear, spine_mid, tail_base
Edges: nose→left_ear, nose→right_ear, nose→spine_mid, spine_mid→tail_base
```

### Fly (detailed)
```
Nodes: head, thorax, abdomen, wing_L, wing_R, leg_L1, leg_L2, leg_L3, leg_R1, leg_R2, leg_R3
Edges: head→thorax, thorax→abdomen, thorax→wing_L, thorax→wing_R,
       thorax→leg_L1, thorax→leg_L2, thorax→leg_L3,
       thorax→leg_R1, thorax→leg_R2, thorax→leg_R3
```

---

## Common Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| Too many nodes | Slower labeling, harder training | Start with 5-10 essential nodes |
| Deep edge chains | Poor grouping in bottom-up | Use shallow tree structure |
| Ambiguous landmarks | Inconsistent labels | Choose anatomically distinct points |
| Symmetric naming confusion | Mixing up left/right | Use clear prefixes: `L_`, `R_` or `left_`, `right_` |
