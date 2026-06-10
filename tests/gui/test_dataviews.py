import numpy as np
import sleap_io as sio

from sleap.gui.dataviews import *


def test_skeleton_nodes(qtbot, centered_pair_predictions):
    table = GenericTableView(
        model=SkeletonNodesTableModel(items=centered_pair_predictions.skeletons[0])
    )

    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "neck"

    table = GenericTableView(
        model=SkeletonEdgesTableModel(items=centered_pair_predictions.skeletons[0])
    )
    table.selectRow(2)
    assert table.model().data(table.currentIndex()) == "thorax"

    table = GenericTableView(
        row_name="video",
        model=VideosTableModel(items=centered_pair_predictions.videos),
        multiple_selection=True,
    )
    table.selectRow(0)
    assert (
        table.model().data(table.currentIndex()).find("centered_pair_low_quality.mp4")
        > -1
    )
    assert table.state["selected_video"] == centered_pair_predictions.videos[0]

    table = GenericTableView(
        row_name="instance",
        name_prefix="",
        model=LabeledFrameTableModel(
            items=centered_pair_predictions.labeled_frames[13]
        ),
    )
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "21/24"


def test_table_sort(qtbot, centered_pair_predictions):
    table = GenericTableView(
        row_name="instance",
        is_sortable=True,
        name_prefix="",
        model=LabeledFrameTableModel(
            items=centered_pair_predictions.labeled_frames[13]
        ),
    )
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "21/24"

    inst = centered_pair_predictions.labeled_frames[13].instances[0]
    table.selectRow(0)
    assert table.getSelectedRowItem().score == inst.score

    inst = centered_pair_predictions.labeled_frames[13].instances[1]
    table.selectRow(1)
    assert table.getSelectedRowItem().score == inst.score

    # Now sort the instances and make sure things are different
    table.model().sort(2)  # "score" column, should reverse initial order
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "24/24"

    # Instance 0 should be in row 1
    inst = centered_pair_predictions.labeled_frames[13].instances[0]
    table.selectRow(1)
    assert table.getSelectedRowItem().score == inst.score

    # Instance 1 should be in row 0
    inst = centered_pair_predictions.labeled_frames[13].instances[1]
    table.selectRow(0)
    assert table.getSelectedRowItem().score == inst.score


def test_table_sort_string(qtbot):
    table_model = GenericTableModel(
        items=[dict(a=1, b=2), dict(a=2, b="")], properties=["a", "b"]
    )

    table = GenericTableView(is_sortable=True, model=table_model)

    # Make sure we can sort with both numbers and strings (i.e., "")
    table.model().sort(0)
    table.model().sort(1)


def test_labeled_frame_mean_node_score(qtbot, centered_pair_predictions):
    """The 'mean node score' column should reflect mean of point scores over
    visible (non-NaN) nodes, matching sleap-nn's filter definition."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    assert "mean node score" in model.properties

    for row, instance in enumerate(lf.instances):
        cell = model._data[row]["mean node score"]

        pts = instance.points
        visible = ~np.isnan(pts["xy"]).any(axis=1)
        scores = pts["score"][visible]
        scores = scores[~np.isnan(scores)]
        expected = f"{float(np.mean(scores)):.2f}"
        assert cell == expected


def test_labeled_frame_mean_node_score_user_instance(qtbot, centered_pair_predictions):
    """User instances have no per-point scores; the column should be empty."""
    skeleton = centered_pair_predictions.skeletons[0]
    video = centered_pair_predictions.videos[0]
    user_inst = sio.Instance.from_numpy(
        np.zeros((len(skeleton.nodes), 2)), skeleton=skeleton
    )
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[user_inst])

    model = LabeledFrameTableModel(items=lf)
    assert model._data[0]["mean node score"] == ""


def test_labeled_frame_mean_node_score_all_nan(qtbot, centered_pair_predictions):
    """If all keypoints are NaN, the column should be empty rather than NaN."""
    skeleton = centered_pair_predictions.skeletons[0]
    video = centered_pair_predictions.videos[0]
    n_nodes = len(skeleton.nodes)
    pred_inst = sio.PredictedInstance.from_numpy(
        points_data=np.full((n_nodes, 2), np.nan),
        skeleton=skeleton,
        point_scores=np.full(n_nodes, 0.9),
        score=0.0,
    )
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred_inst])

    model = LabeledFrameTableModel(items=lf)
    assert model._data[0]["mean node score"] == ""


def _checkstate(model, row, key):
    """Return the CheckStateRole value for the given row and column key."""
    col = model.properties.index(key)
    index = model.index(row, col)
    return model.data(index, QtCore.Qt.CheckStateRole)


def _set_checkstate(model, row, key, checked):
    """Set the CheckStateRole for the given row and column key."""
    col = model.properties.index(key)
    index = model.index(row, col)
    value = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
    return model.setData(index, value, QtCore.Qt.CheckStateRole)


def test_labeled_frame_visibility_columns_present(qtbot, centered_pair_predictions):
    """The visibility and view-only checkbox columns should exist."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)
    assert "visibility" in model.properties
    assert "view only" in model.properties
    # Appended last so existing name-indexed lookups stay valid.
    assert model.properties.index("visibility") > model.properties.index("skeleton")


def test_labeled_frame_visibility_defaults(qtbot, centered_pair_predictions):
    """By default every visibility box is checked and view-only is unchecked."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)
    for row in range(model.rowCount()):
        assert _checkstate(model, row, "visibility") == QtCore.Qt.Checked
        assert _checkstate(model, row, "view only") == QtCore.Qt.Unchecked


def test_labeled_frame_uncheck_visibility(qtbot, centered_pair_predictions):
    """Unchecking visibility marks the instance hidden but keeps its row."""
    from sleap.gui.state import instance_visible

    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)
    n_rows = model.rowCount()

    inst0 = model.original_items[0]
    inst1 = model.original_items[1]

    assert _set_checkstate(model, 0, "visibility", False)
    assert _checkstate(model, 0, "visibility") == QtCore.Qt.Unchecked
    # Other rows unaffected.
    assert _checkstate(model, 1, "visibility") == QtCore.Qt.Checked
    # Row count unchanged: hidden instances stay listed.
    assert model.rowCount() == n_rows

    # Effective visibility reflects the hidden set.
    assert not instance_visible(model._vis_state, inst0)
    assert instance_visible(model._vis_state, inst1)

    # Re-checking restores visibility.
    assert _set_checkstate(model, 0, "visibility", True)
    assert _checkstate(model, 0, "visibility") == QtCore.Qt.Checked
    assert instance_visible(model._vis_state, inst0)


def test_labeled_frame_view_only_exclusivity(qtbot, centered_pair_predictions):
    """Checking view-only on one row auto-unchecks the previous (radio-like)."""
    from sleap.gui.state import instance_visible

    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    inst0 = model.original_items[0]
    inst1 = model.original_items[1]

    # Check view-only on row 0.
    assert _set_checkstate(model, 0, "view only", True)
    assert _checkstate(model, 0, "view only") == QtCore.Qt.Checked
    assert _checkstate(model, 1, "view only") == QtCore.Qt.Unchecked
    # Only instance 0 is visible.
    assert instance_visible(model._vis_state, inst0)
    assert not instance_visible(model._vis_state, inst1)

    # During view-only the visibility column is greyed but STAYS enabled and
    # user-checkable, so clicking a visibility box can exit view-only mode (per
    # the spec). The greying is conveyed via a BackgroundRole brush, not by
    # disabling the cell (a disabled cell would make the exit gesture
    # unreachable in the real view).
    vis_col = model.properties.index("visibility")
    flags = model.flags(model.index(0, vis_col))
    assert flags & QtCore.Qt.ItemIsEnabled
    assert flags & QtCore.Qt.ItemIsUserCheckable
    assert flags & QtCore.Qt.ItemIsSelectable
    assert model.data(model.index(0, vis_col), QtCore.Qt.BackgroundRole) is not None

    # Check view-only on row 1: row 0 auto-unchecks.
    assert _set_checkstate(model, 1, "view only", True)
    assert _checkstate(model, 0, "view only") == QtCore.Qt.Unchecked
    assert _checkstate(model, 1, "view only") == QtCore.Qt.Checked
    assert not instance_visible(model._vis_state, inst0)
    assert instance_visible(model._vis_state, inst1)


def test_labeled_frame_visibility_click_exits_view_only(
    qtbot, centered_pair_predictions
):
    """Clicking any visibility box exits view-only mode."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    # Enter view-only on row 0.
    _set_checkstate(model, 0, "view only", True)
    assert _checkstate(model, 0, "view only") == QtCore.Qt.Checked

    # Toggling any visibility box exits view-only.
    _set_checkstate(model, 1, "visibility", False)
    assert _checkstate(model, 0, "view only") == QtCore.Qt.Unchecked

    # Visibility column re-enabled.
    vis_col = model.properties.index("visibility")
    flags = model.flags(model.index(0, vis_col))
    assert flags & QtCore.Qt.ItemIsEnabled
    assert flags & QtCore.Qt.ItemIsUserCheckable


def test_labeled_frame_visibility_uncheck_view_only(qtbot, centered_pair_predictions):
    """Unchecking the active view-only row clears view-only mode."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    _set_checkstate(model, 0, "view only", True)
    assert _checkstate(model, 0, "view only") == QtCore.Qt.Checked

    # Unchecking it returns to the default (all visible) state.
    _set_checkstate(model, 0, "view only", False)
    assert _checkstate(model, 0, "view only") == QtCore.Qt.Unchecked
    for row in range(model.rowCount()):
        assert _checkstate(model, row, "visibility") == QtCore.Qt.Checked


def test_labeled_frame_track_column_unaffected(qtbot, centered_pair_predictions):
    """Adding checkbox columns must not break the existing track column."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    track_col = model.properties.index("track")
    points_col = model.properties.index("points")

    # Non-checkbox columns still return text via DisplayRole.
    assert model.data(model.index(1, points_col)) == "21/24"

    # Track column flags delegate to the base implementation.
    track_flags = model.flags(model.index(0, track_col))
    assert track_flags & QtCore.Qt.ItemIsSelectable
    assert track_flags & QtCore.Qt.ItemIsEnabled

    # Checkbox columns have no DisplayRole text.
    vis_col = model.properties.index("visibility")
    assert model.data(model.index(0, vis_col), QtCore.Qt.DisplayRole) is None


def test_instance_visible_respects_global_show_instances():
    """Global "show instances" off hides everything (regression for #2755).

    The instance overlay re-applies `instance_visible` on every replot, so if it
    ignored the global toggle it would override the global Hide. Per-instance
    state may only further hide instances, never force a globally-hidden one back
    on.
    """
    from sleap.gui.state import GuiState, instance_visible, VIEW_ONLY_INSTANCE_KEY

    state = GuiState()
    inst_a, inst_b = object(), object()

    # Global on: default per-instance state -> both visible.
    state["show instances"] = True
    assert instance_visible(state, inst_a)
    assert instance_visible(state, inst_b)

    # View-only on A (global on): only A visible.
    state[VIEW_ONLY_INSTANCE_KEY] = id(inst_a)
    assert instance_visible(state, inst_a)
    assert not instance_visible(state, inst_b)

    # Global off: nothing visible, even the view-only instance.
    state["show instances"] = False
    assert not instance_visible(state, inst_a)
    assert not instance_visible(state, inst_b)
