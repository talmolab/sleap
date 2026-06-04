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
