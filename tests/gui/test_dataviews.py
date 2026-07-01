import numpy as np
import pytest
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


def test_videos_table_unreadable_video(qtbot, centered_pair_predictions):
    """A video whose frame can't be read shows '?' dims instead of raising.

    Reading `img_shape` hits the disk and can fail intermittently (e.g. a video
    on a flaky network drive). One unreadable video must not blank the whole
    Videos table (discussion #2742).
    """

    class UnreadableBackend:
        filename = "/path/to/unreadable.mp4"
        frames = 100

        @property
        def img_shape(self):
            raise IndexError("Failed to read frame index 0.")

    model = VideosTableModel(items=centered_pair_predictions.videos)

    # The good video still reports real (integer) dimensions.
    good = model.item_to_data(None, centered_pair_predictions.videos[0])
    assert isinstance(good["height"], int)
    assert isinstance(good["width"], int)

    # The unreadable video keeps its row, with placeholder dimensions.
    bad = model.item_to_data(None, UnreadableBackend())
    assert bad["name"] == "unreadable.mp4"
    assert bad["frames"] == 100
    assert bad["height"] == "?"
    assert bad["width"] == "?"
    assert bad["channels"] == "?"


def test_items_setter_ends_reset_on_error(qtbot):
    """If building a row raises, the model still ends the reset.

    Leaving the model in a half-reset state (beginResetModel without a matching
    endResetModel) is what blanks the table, so endResetModel must always run.
    """

    class RaisingModel(GenericTableModel):
        def item_to_data(self, obj, item):
            raise RuntimeError("boom")

    model = RaisingModel(properties=["a"])

    ended = []
    real_end = model.endResetModel
    model.endResetModel = lambda: (ended.append(True), real_end())[1]

    with pytest.raises(RuntimeError):
        model.items = [object()]

    assert ended == [True]


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


# -- "Invisible Nodes" per-instance override column (#2782) ----------------------


def test_labeled_frame_invisible_nodes_column_present(qtbot, centered_pair_predictions):
    """The per-instance "invisible nodes" checkbox column should exist last."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)
    assert "invisible nodes" in model.properties
    # Appended LAST, after the #2755 columns.
    assert model.properties.index("invisible nodes") > model.properties.index(
        "view only"
    )


def test_labeled_frame_invisible_nodes_default_checked(
    qtbot, centered_pair_predictions
):
    """With no override and no global flag set, the effective default is True."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)
    for row in range(model.rowCount()):
        assert _checkstate(model, row, "invisible nodes") == QtCore.Qt.Checked


def test_labeled_frame_invisible_nodes_default_follows_global(
    qtbot, centered_pair_predictions
):
    """The default checkbox follows the global "show non-visible nodes" flag."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)
    model._vis_state["show non-visible nodes"] = False
    for row in range(model.rowCount()):
        assert _checkstate(model, row, "invisible nodes") == QtCore.Qt.Unchecked


def test_labeled_frame_toggle_invisible_nodes_sets_override(
    qtbot, centered_pair_predictions
):
    """Toggling the column writes an explicit per-instance override."""
    from sleap.gui.state import (
        SHOW_NONVISIBLE_OVERRIDE_KEY,
        instance_shows_non_visible,
    )

    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    inst0 = model.original_items[0]
    inst1 = model.original_items[1]

    assert _set_checkstate(model, 0, "invisible nodes", False)
    assert _checkstate(model, 0, "invisible nodes") == QtCore.Qt.Unchecked
    # Other rows unaffected.
    assert _checkstate(model, 1, "invisible nodes") == QtCore.Qt.Checked

    override = model._vis_state.get(SHOW_NONVISIBLE_OVERRIDE_KEY)
    assert override[id(inst0)] is False
    # Effective value reflects the override for inst0, default for inst1.
    assert instance_shows_non_visible(model._vis_state, inst0, True) is False
    assert instance_shows_non_visible(model._vis_state, inst1, True) is True

    # Re-checking stores an explicit True (never popped).
    assert _set_checkstate(model, 0, "invisible nodes", True)
    assert _checkstate(model, 0, "invisible nodes") == QtCore.Qt.Checked
    override = model._vis_state.get(SHOW_NONVISIBLE_OVERRIDE_KEY)
    assert override[id(inst0)] is True


def test_labeled_frame_invisible_nodes_independent_of_visibility(
    qtbot, centered_pair_predictions
):
    """The override map is orthogonal to the visibility / view-only columns."""
    from sleap.gui.state import SHOW_NONVISIBLE_OVERRIDE_KEY

    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)
    inst0 = model.original_items[0]

    _set_checkstate(model, 0, "invisible nodes", False)
    assert model._vis_state.get(SHOW_NONVISIBLE_OVERRIDE_KEY)[id(inst0)] is False

    # Toggling visibility off/on does not touch the override entry.
    _set_checkstate(model, 0, "visibility", False)
    _set_checkstate(model, 0, "visibility", True)
    assert model._vis_state.get(SHOW_NONVISIBLE_OVERRIDE_KEY)[id(inst0)] is False
    assert _checkstate(model, 0, "invisible nodes") == QtCore.Qt.Unchecked

    # Toggling view only does not touch the override map either.
    _set_checkstate(model, 0, "view only", True)
    assert model._vis_state.get(SHOW_NONVISIBLE_OVERRIDE_KEY)[id(inst0)] is False


def test_labeled_frame_invisible_nodes_flags_checkable(
    qtbot, centered_pair_predictions
):
    """The "invisible nodes" cell is user-checkable with no DisplayRole text."""
    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    col = model.properties.index("invisible nodes")
    index = model.index(0, col)
    flags = model.flags(index)
    assert flags & QtCore.Qt.ItemIsEnabled
    assert flags & QtCore.Qt.ItemIsUserCheckable
    assert flags & QtCore.Qt.ItemIsSelectable
    assert model.data(index, QtCore.Qt.DisplayRole) is None


def test_checkbox_toggle_falls_back_to_manual_qc_mode(qtbot, centered_pair_predictions):
    """A manual per-instance checkbox edit drops an active Label QC display mode
    back to "manual" (#2783), so the mode stops clobbering the user's edit."""
    from sleap.gui.state import (
        QC_DISPLAY_MODE_KEY,
        QC_MODE_SELECTED_ONLY,
        QC_MODE_MANUAL,
    )

    lf = centered_pair_predictions.labeled_frames[13]
    model = LabeledFrameTableModel(items=lf)

    # Every per-instance checkbox column resets the mode to manual.
    for column in ("visibility", "view only", "invisible nodes"):
        model._vis_state[QC_DISPLAY_MODE_KEY] = QC_MODE_SELECTED_ONLY
        assert _set_checkstate(model, 0, column, False)
        assert model._vis_state[QC_DISPLAY_MODE_KEY] == QC_MODE_MANUAL


# -- Videos table tolerates a Video whose backend failed to open (#2794) ----------


def _videos_table_row(video):
    """Render a single videos-table row dict via VideosTableModel."""
    model = VideosTableModel(items=[video])
    return model._data[0]


def test_videos_table_unopened_backend_renders_from_metadata(qtbot, tmp_path):
    """A Video whose file is inaccessible has `backend is None`; the row must
    fall back to filename + cached shape instead of crashing (#2794).

    Reproduces the reporter's `AttributeError: 'NoneType' object has no
    attribute 'filename'`: before the fix, `item_to_data` dereferenced the None
    backend's `.filename` and the whole videos table refresh raised.
    """
    missing = tmp_path / "sub" / "1-doi1.mp4"  # never created -> backend stays None
    video = sio.Video(
        filename=str(missing),
        backend_metadata={"shape": [9746, 1080, 1920, 1], "type": "MediaVideo"},
    )
    assert video.backend is None  # precondition: the exact crash trigger

    row = _videos_table_row(video)
    assert row["name"] == "1-doi1.mp4"
    assert row["filepath"] == str(missing.parent)
    assert row["frames"] == 9746
    assert row["height"] == 1080
    assert row["width"] == 1920
    assert row["channels"] == 1


def test_videos_table_unopened_backend_without_metadata(qtbot, tmp_path):
    """No cached shape -> '?' placeholders for shape fields, still no crash."""
    missing = tmp_path / "missing.mp4"
    video = sio.Video(filename=str(missing))
    assert video.backend is None

    row = _videos_table_row(video)
    assert row["name"] == "missing.mp4"
    assert row["filepath"] == str(missing.parent)
    assert row["frames"] == "?"
    assert row["height"] == row["width"] == row["channels"] == "?"


def test_videos_table_none_item_does_not_crash(qtbot):
    """A bare None among the videos renders an all-'?' row rather than crashing."""
    model = VideosTableModel(items=[None])
    row = model._data[0]
    assert all(row[prop] == "?" for prop in model.properties)


def test_videos_table_open_backend_row_unchanged(qtbot, centered_pair_predictions):
    """Regression: a normally-openable video still renders its full row -- the
    None-backend fallback must not alter the happy path (#2794)."""
    video = centered_pair_predictions.videos[0]
    assert video.backend is not None  # fixture video opens normally

    row = _videos_table_row(video)
    assert row["name"].endswith("centered_pair_low_quality.mp4")
    assert row["filepath"]  # non-empty parent directory
    assert isinstance(row["frames"], int) and row["frames"] > 0
    # height/width/channels come from a frame read; present as int or "?".
    for key in ("height", "width", "channels"):
        assert row[key] == "?" or (isinstance(row[key], int) and row[key] > 0)


# -- QC display mode -> shared transient keys (#2783 <-> shared model) ------------


def test_qc_mode_maps_onto_transient_keys(qtbot, centered_pair_predictions):
    """A non-manual QC mode's flags drive the same model the overlay reads.

    Emulates `MainWindow._recompute_qc_flags_into_state` (writing the three
    transient keys from `compute_qc_visibility`) and confirms `instance_visible`
    / `instance_shows_non_visible` agree with the computed flags -- i.e. the
    #2783 mode result is consistent with the shared per-instance model that the
    instance overlay applies on replot. No player / replot is involved.
    """
    from sleap.gui.state import (
        GuiState,
        INSTANCE_HIDDEN_KEY,
        VIEW_ONLY_INSTANCE_KEY,
        SHOW_NONVISIBLE_OVERRIDE_KEY,
        QC_MODE_SELECTED_ONLY,
        compute_qc_visibility,
        instance_visible,
        instance_shows_non_visible,
    )
    from sleap.sleap_io_adaptors.lf_labels_utils import get_instances_to_show

    lf = centered_pair_predictions.labeled_frames[13]
    instances = get_instances_to_show(lf)
    assert len(instances) >= 2

    state = GuiState()
    selected = instances[0]
    flags = compute_qc_visibility(
        QC_MODE_SELECTED_ONLY, selected, instances, global_show_non_visible=True
    )
    # Write the three transient keys exactly as the app helper does.
    state[INSTANCE_HIDDEN_KEY] = {iid for iid, (vis, _) in flags.items() if not vis}
    state[VIEW_ONLY_INSTANCE_KEY] = None
    state[SHOW_NONVISIBLE_OVERRIDE_KEY] = {iid: snv for iid, (_, snv) in flags.items()}

    # selected_only with the global gate on: only the selected instance, with its
    # hidden points.
    assert instance_visible(state, instances[0]) is True
    assert instance_visible(state, instances[1]) is False
    assert instance_shows_non_visible(state, instances[0], False) is True
    assert instance_shows_non_visible(state, instances[1], False) is False


def test_node_symmetry_combo_delegate(qtbot):
    """The Nodes table symmetry column edits via a node-name dropdown."""
    from sleap.gui.commands import CommandContext
    from sleap.sleap_io_adaptors.skeleton_utils import get_symmetry_node

    skeleton = sio.Skeleton(["nose", "eye_L", "eye_R"])
    labels = sio.Labels(skeletons=[skeleton])
    context = CommandContext.from_labels(labels)
    model = SkeletonNodesTableModel(items=skeleton, context=context)
    table = GenericTableView(model=model)
    delegate = NodeSymmetryComboDelegate(table)
    sym_col = model.properties.index("symmetry")

    # The editor for the "nose" row lists the OTHER nodes plus a blank, and
    # excludes the row's own node (no self-symmetry, no free-text phantom nodes).
    index = model.index(0, sym_col)
    editor = delegate.createEditor(table, None, index)
    assert [editor.itemText(i) for i in range(editor.count())] == [
        "",
        "eye_L",
        "eye_R",
    ]

    # Choosing a partner routes through the command and sets the symmetry.
    editor.setCurrentText("eye_L")
    delegate.setModelData(editor, model, index)
    assert get_symmetry_node(skeleton, "nose") == "eye_L"

    # setEditorData reflects the current partner back into the combo box.
    model.items = skeleton  # refresh cached rows after the edit
    index = model.index(0, sym_col)
    editor = delegate.createEditor(table, None, index)
    delegate.setEditorData(editor, index)
    assert editor.currentText() == "eye_L"
