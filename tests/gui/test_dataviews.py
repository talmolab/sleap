import pytest
import pytestqt

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
        row_name="video", model=VideosTableModel(items=centered_pair_predictions.videos)
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
        model=LabeledFrameTableModel(items=centered_pair_predictions.labels[13]),
    )
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "21/24"


def test_table_sort(qtbot, centered_pair_predictions):
    table = GenericTableView(
        row_name="instance",
        is_sortable=True,
        name_prefix="",
        model=LabeledFrameTableModel(items=centered_pair_predictions.labels[13]),
    )
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "21/24"

    inst = centered_pair_predictions.labels[13].instances[0]
    table.selectRow(0)
    assert table.getSelectedRowItem().score == inst.score

    inst = centered_pair_predictions.labels[13].instances[1]
    table.selectRow(1)
    assert table.getSelectedRowItem().score == inst.score

    # Now sort the instances and make sure things are different
    table.model().sort(2)  # "score" column, should reverse initial order
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "24/24"

    # Instance 0 should be in row 1
    inst = centered_pair_predictions.labels[13].instances[0]
    table.selectRow(1)
    assert table.getSelectedRowItem().score == inst.score

    # Instance 1 should be in row 0
    inst = centered_pair_predictions.labels[13].instances[1]
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
