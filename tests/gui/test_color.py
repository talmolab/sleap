from sleap.gui.color import ColorManager


def test_color_manager():
    color_manager = ColorManager()

    color_manager.palette = "standard"

    colors = [color_manager.get_color_by_idx(i) for i in range(3)]

    # make sure we can set palette by passing list of color tuples
    color_manager.palette = colors

    for i in range(3):
        assert color_manager.get_color_by_idx(i) == colors[i]

    # make sure standard palette is used if name isn't valid
    color_manager.palette = "something that doesn't exist"
    for i in range(3):
        assert color_manager.get_color_by_idx(i) == colors[i]


def test_track_color(centered_pair_predictions):

    labels = centered_pair_predictions

    instances = labels.labeled_frames[-1].instances
    tracks = [inst.track for inst in instances]
    inst_0 = instances[0]

    # Test track colors
    color_manager = ColorManager(labels=labels)

    assert list(color_manager.get_track_color(tracks[3])) == [119, 172, 48]

    assert color_manager.get_item_color(inst_0) == color_manager.get_color_by_idx(0)

    # Make sure that predicted node is not colored when it shouldn't be
    color_manager.color_predicted = False
    assert (
        color_manager.get_item_color(inst_0.nodes[0], inst_0)
        == color_manager.uncolored_prediction_color
    )

    # Make sure that predicted node is now colored
    color_manager.color_predicted = True
    assert color_manager.get_item_color(
        inst_0.nodes[1], inst_0
    ) == color_manager.get_color_by_idx(0)

    # Check line width for node
    assert (
        color_manager.get_item_pen_width(inst_0.nodes[1], inst_0)
        == color_manager.medium_pen_width
    )

    # Make sure that nodes can be distinctly colored
    color_manager.distinctly_color = "nodes"
    assert color_manager.get_item_color(
        inst_0.nodes[1], inst_0
    ) == color_manager.get_color_by_idx(3)

    # Check line width for node
    assert (
        color_manager.get_item_pen_width(inst_0.nodes[1], inst_0)
        == color_manager.thick_pen_width
    )

    # Make sure that edges can be distinctly colored
    color_manager.distinctly_color = "edges"

    for edge_idx in range(4):
        assert color_manager.get_item_color(
            inst_0.skeleton.edges[edge_idx], inst_0
        ) == color_manager.get_color_by_idx(edge_idx)
