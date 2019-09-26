import pytest
import pytestqt

from PySide2.QtWidgets import QApplication
from sleap.skeleton import Skeleton
from sleap.gui.dataviews import (
    VideosTable,
    SkeletonNodesTable,
    SkeletonEdgesTable,
    LabeledFrameTable,
    SkeletonNodeModel,
)


def test_skeleton_nodes(qtbot, centered_pair_predictions):

    table = SkeletonNodesTable(centered_pair_predictions.skeletons[0])
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "neck"

    table = SkeletonEdgesTable(centered_pair_predictions.skeletons[0])
    table.selectRow(2)
    assert table.model().data(table.currentIndex()) == "thorax"

    table = VideosTable(centered_pair_predictions.videos)
    table.selectRow(0)
    assert (
        table.model().data(table.currentIndex()).find("centered_pair_low_quality.mp4")
        > -1
    )

    table = LabeledFrameTable(
        centered_pair_predictions.labels[13], centered_pair_predictions
    )
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "21/24"
