import pytest
from sleap.gui.widgets.video import *
import numpy as np
from sleap import Instance, Skeleton

from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QColor

def test_add_node(qtbot, small_robot_mp4_vid, centered_pair_labels):
    pass