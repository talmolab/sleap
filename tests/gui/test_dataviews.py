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
