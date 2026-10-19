import itertools

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

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

    # This is the 26th track in the project (the fixture has 27), so it gets the
    # 26th color of the palette. It used to wrap around to the 5th color because
    # the palette only had 7 colors.
    assert labels.tracks.index(tracks[3]) == 25
    assert list(color_manager.get_track_color(tracks[3])) == [60, 180, 120]

    assert color_manager.get_item_color(inst_0) == color_manager.get_color_by_idx(0)

    # Make sure that predicted node is not colored when it shouldn't be
    color_manager.color_predicted = False
    assert (
        color_manager.get_item_color(inst_0.skeleton[0], inst_0)
        == color_manager.uncolored_prediction_color
    )

    # Make sure that predicted node is now colored
    color_manager.color_predicted = True
    assert color_manager.get_item_color(
        inst_0.skeleton[1], inst_0
    ) == color_manager.get_color_by_idx(0)

    # Check line width for node
    assert (
        color_manager.get_item_pen_width(inst_0.skeleton[1], inst_0)
        == color_manager.medium_pen_width
    )

    # Make sure that nodes can be distinctly colored
    color_manager.distinctly_color = "nodes"
    assert color_manager.get_item_color(
        inst_0.skeleton.nodes[1], inst_0
    ) == color_manager.get_color_by_idx(1)

    # Check line width for node
    assert (
        color_manager.get_item_pen_width(inst_0.skeleton.nodes[1], inst_0)
        == color_manager.thick_pen_width
    )

    # Make sure that edges can be distinctly colored
    color_manager.distinctly_color = "edges"

    for edge_idx in range(4):
        assert color_manager.get_item_color(
            inst_0.skeleton.edges[edge_idx], inst_0
        ) == color_manager.get_color_by_idx(edge_idx)


def test_palette_colors_are_distinct(centered_pair_predictions):
    """Tracks should not be drawn in colors that repeat or are hard to tell apart.

    The standard palette used to hold only 7 colors, so in a project with 10+
    tracks the 8th track was drawn in exactly the same color as the 1st.
    """
    color_manager = ColorManager(labels=centered_pair_predictions)
    n = len(color_manager._color_map)
    assert n >= 24, "palette too small to color a typical multi-animal project"

    colors = [color_manager.get_color_by_idx(i) for i in range(n)]
    assert len(set(colors)) == n, "palette contains duplicate colors"

    # CIEDE2000 measures perceptual difference; ~2.3 is the just-noticeable
    # difference and anything under ~10 is easily confused at a glance.
    lab = rgb2lab(np.array(colors, dtype=float).reshape(-1, 1, 3) / 255).reshape(-1, 3)
    worst = min(
        float(deltaE_ciede2000(lab[i : i + 1], lab[j : j + 1])[0])
        for i, j in itertools.combinations(range(n), 2)
    )
    assert worst > 10, f"palette has two near-identical colors (dE={worst:.1f})"


def min_perceptual_distance(colors):
    """Smallest CIEDE2000 distance between any two of `colors`."""
    lab = rgb2lab(np.array(colors, dtype=float).reshape(-1, 1, 3) / 255).reshape(-1, 3)
    return min(
        float(deltaE_ciede2000(lab[i : i + 1], lab[j : j + 1])[0])
        for i, j in itertools.combinations(range(len(colors)), 2)
    )


def test_palette_extends_past_its_length():
    """A project with more tracks than palette colors should not reuse colors."""
    color_manager = ColorManager()
    color_manager.palette = "standard"
    n = len(color_manager._color_map)

    # The color right past the end of the palette used to wrap back to the first.
    assert color_manager.get_color_by_idx(n) != color_manager.get_color_by_idx(0)

    colors = [color_manager.get_color_by_idx(i) for i in range(60)]
    assert len(set(colors)) == 60, "colors repeat past the end of the palette"
    assert min_perceptual_distance(colors) > 8

    # Growing the palette must not recolor tracks that are already on screen.
    assert colors[:n] == [
        color_manager.color_to_tuple(c) for c in color_manager._color_map
    ]


def test_palette_extension_preserves_user_colors():
    """A custom palette keeps its own colors first, then gets extended."""
    color_manager = ColorManager()
    color_manager.palette = ["255,0,0", "0,255,0"]

    assert color_manager.get_color_by_idx(0) == (255, 0, 0)
    assert color_manager.get_color_by_idx(1) == (0, 255, 0)
    assert color_manager.get_color_by_idx(2) not in {(255, 0, 0), (0, 255, 0)}


def test_clip_palettes_still_clamp():
    """Palettes ending in "+" intentionally reuse the last color; keep that."""
    color_manager = ColorManager()
    color_manager.palette = "five+"

    assert color_manager.index_mode == "clip"
    last = color_manager.get_color_by_idx(4)
    assert color_manager.get_color_by_idx(5) == last
    assert color_manager.get_color_by_idx(50) == last
