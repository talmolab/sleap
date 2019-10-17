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
