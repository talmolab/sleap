from PySide2 import QtCore
from PySide2.QtCore import Qt

from PySide2.QtGui import QKeyEvent

from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QDockWidget
from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout
from PySide2.QtWidgets import QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox
from PySide2.QtWidgets import QTableWidget, QTableView, QTableWidgetItem, QAbstractItemView
from PySide2.QtWidgets import QMenu, QAction
from PySide2.QtWidgets import QFileDialog, QMessageBox

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sleap.gui.video import QtVideoPlayer, QtInstance, QtEdge, QtNode
from sleap.io.video import Video, HDF5Video, MediaVideo
from sleap.io.labels import Labels
from sleap.skeleton import Skeleton

class SkeletonNodesTable(QTableView):
    """Table view widget backed by a custom data model for displaying and
    editing Skeleton nodes. """
    def __init__(self, skeleton: Skeleton):
        super(SkeletonNodesTable, self).__init__()
        self.setModel(SkeletonNodesTableModel(skeleton))
        self.setSelectionMode(QAbstractItemView.SingleSelection)

class SkeletonNodesTableModel(QtCore.QAbstractTableModel):
    _props = ["name", "symmetry"]

    def __init__(self, skeleton: Skeleton):
        super(SkeletonNodesTableModel, self).__init__()
        self._skeleton = skeleton

    @property
    def skeleton(self):
        return self._skeleton

    def data(self, index: QtCore.QModelIndex, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            node_idx = index.row()
            prop = self._props[index.column()]
            node_name = self.skeleton.nodes[node_idx]

            if prop == "name":
                return node_name
            elif prop == "symmetry":
                return self.skeleton.get_symmetry(node_name)

        return None

    def rowCount(self, parent):
        return len(self.skeleton.nodes)

    def columnCount(self, parent):
        return len(SkeletonNodesTableModel._props)

    def headerData(self, section, orientation: QtCore.Qt.Orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section

        return None

    def setData(self, index: QtCore.QModelIndex, value: str, role=Qt.EditRole):
        if role == Qt.EditRole:
            node_idx = index.row()
            prop = self._props[index.column()]
            node_name = self.skeleton.nodes[node_idx]

            try:
                if prop == "name":
                    if len(value) > 0:
                        self._skeleton.relabel_node(node_name, value)
                    # else:
                        # self._skeleton.delete_node(node_name)
                elif prop == "symmetry":
                    if len(value) > 0:
                        self._skeleton.add_symmetry(node_name, value)
                    else:
                        self._skeleton.delete_symmetry(node_name, self._skeleton.get_symmetry(node_name))

                return True
            except:
                # TODO: display feedback on error?
                pass

        return False

    def flags(self, index: QtCore.QModelIndex):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable


class SkeletonEdgesTable(QTableView):
    """Table view widget backed by a custom data model for displaying and
    editing Skeleton edges. """
    def __init__(self, skeleton: Skeleton):
        super(SkeletonEdgesTable, self).__init__()
        self.setModel(SkeletonEdgesTableModel(skeleton))
        self.setSelectionMode(QAbstractItemView.SingleSelection)

class SkeletonEdgesTableModel(QtCore.QAbstractTableModel):
    _props = ["source", "destination"]

    def __init__(self, skeleton: Skeleton):
        super(SkeletonEdgesTableModel, self).__init__()
        self._skeleton = skeleton

    @property
    def skeleton(self):
        return self._skeleton

    def data(self, index: QtCore.QModelIndex, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            idx = index.row()
            prop = self._props[index.column()]
            edge = self.skeleton.edges[idx]

            if prop == "source":
                return edge[0]
            elif prop == "destination":
                return edge[1]

        return None

    def rowCount(self, parent):
        return len(self.skeleton.edges)

    def columnCount(self, parent):
        return len(SkeletonNodesTableModel._props)

    def headerData(self, section, orientation: QtCore.Qt.Orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._props[section]
            elif orientation == QtCore.Qt.Vertical:
                return section

        return None

    def flags(self, index: QtCore.QModelIndex):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable





if __name__ == "__main__":
    s1 = Skeleton("s1")
    s1.add_nodes(['1','2','3','4','5','6'])
    s1.add_edge('1', '2')
    s1.add_edge('3', '4')
    s1.add_edge('5', '6')
    s1.add_symmetry('1', '5')
    s1.add_symmetry('3', '6')

    app = QApplication([])

    # table = SkeletonNodesTable(s1)
    table = SkeletonEdgesTable(s1)
    table.show()

    app.exec_()