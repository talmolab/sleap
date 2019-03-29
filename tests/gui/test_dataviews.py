import pytest
import pytestqt

from PySide2.QtWidgets import QApplication
from sleap.skeleton import Skeleton
from sleap.gui.dataviews import SkeletonNodesTable


def test_skeleton_nodes(qtbot):
    skeleton = Skeleton("skeleton")
    skeleton.add_nodes(['1','2','3','4','5','6'])
    skeleton.add_edge('1', '2')
    skeleton.add_edge('3', '4')
    skeleton.add_edge('5', '6')
    skeleton.add_symmetry('1', '5')
    skeleton.add_symmetry('3', '6')

    # app = QApplication([])
    table = SkeletonNodesTable(skeleton)
    qtbot.addWidget(table)
    table.show()
    # app.exec_()

    
