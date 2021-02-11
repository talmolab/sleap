from typing import Optional, List, Dict

import sleap
from sleap.gui.dataviews import (
    GenericTableModel,
    GenericCheckableTableModel,
    GenericTableView,
)
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import FieldComboWidget
from sleap.gui.learning.configs import TrainingConfigsGetter, ConfigFileInfo
from sleap.gui.widgets.models_table import ModelsTableWidget
from sleap.gui.widgets.videos_table import VideosTableWidget

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QWidget,
    QLabel,
    QDialog,
    QGridLayout,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QButtonGroup,
    QListWidget,
    QSpinBox,
)


class TrackingWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        #### Videos ####
        videos_box = QGroupBox("Videos")
        self.videos = VideosTableWidget()

        vb = QVBoxLayout()
        vb.addWidget(self.videos)
        videos_box.setLayout(vb)
        layout.addWidget(videos_box)
        ####

        #### Models ####
        models_box = QGroupBox("Select model(s)")
        self.models = ModelsTableWidget()

        vb = QVBoxLayout()
        vb.addWidget(self.models)
        models_box.setLayout(vb)
        layout.addWidget(models_box)
        ####

        #### Tracking ####
        tracking_box = QGroupBox("Tracking")
        tracking_box.setCheckable(True)
        layout.addWidget(tracking_box)

        tracking = QFormLayout()
        tracking_box.setLayout(tracking)

        ql = QLabel(
            "Tracking is used to associate multiple detected instances across frames. "
            "This is not necessary for single animal videos."
        )
        ql.setWordWrap(True)
        tracking.addRow(ql)

        self.tracking_method_widget = FieldComboWidget()
        self.tracking_method_widget.set_options(
            ["Simple", "Flow shift", "Kalman"], "Simple"
        )
        tracking.addRow("Tracker:", self.tracking_method_widget)

        self.tracking_window_widget = QSpinBox()
        self.tracking_window_widget.setRange(0, 1000)
        self.tracking_window_widget.setValue(5)
        self.tracking_window_widget.setToolTip(
            "Number of past frames to consider when associating tracks."
        )
        tracking.addRow("Window size:", self.tracking_window_widget)
        # TODO:
        # - number of instances
        # - centroid/instance similarity/iou

        ####

        # TODO:
        # - batch size
        # - output folder or same as videos (.prediction.slp)
        # - [ ] open results in GUI
        # - [ ] export analysis file

        ####
        hb = QHBoxLayout()
        hbw = QWidget()
        hbw.setLayout(hb)
        layout.addWidget(hbw)

        hb.addWidget(QPushButton("Save as script"))
        hb.addWidget(QPushButton("Export for Colab"))
        hb.addWidget(QPushButton("Run"))
        ####

        self.videos.add_videos(
            [
                "tests/data/videos/centered_pair_small.mp4",
                "tests/data/videos/small_robot.mp4",
            ]
        )
        self.models.add_models(
            # r"D:\sleap-data\datasets\wt_gold.13pt\sample\models")
            "tests/data/models"
        )


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication

    app = QApplication([])
    window = TrackingWidget()
    window.show()
    app.exec_()
