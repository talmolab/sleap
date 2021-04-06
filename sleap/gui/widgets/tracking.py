from typing import Optional, List, Dict, Text

"""
import sleap
from sleap.gui.dataviews import (
    GenericTableModel,
    GenericCheckableTableModel,
    GenericTableView,
)
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.learning.configs import TrainingConfigsGetter, ConfigFileInfo

from sleap.gui.dialogs.formbuilder import FieldComboWidget
from sleap.gui.widgets.models_table import ModelsTableWidget
from sleap.gui.widgets.videos_table import VideosTableWidget
"""
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


class FieldComboWidget(QtWidgets.QComboBox):
    """
    Custom ComboBox-style widget with method to easily add set of options.

    Arguments:
        result_as_idx: If True, then set/get for value will use idx of option
            rather than string.
        add_blank_option: If True, then blank ("") option will be added at
            beginning of list (which will return "" as val instead of idx if
            result_as_idx is True).
    """

    def __init__(
        self,
        result_as_idx: bool = False,
        add_blank_option: bool = False,
        *args,
        **kwargs,
    ):
        super(FieldComboWidget, self).__init__(*args, **kwargs)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.setMinimumContentsLength(3)
        self.result_as_idx = result_as_idx
        self.add_blank_option = add_blank_option
        self.options_list = []

    def set_options(self, options_list: List[Text], select_item: Optional[Text] = None):
        """
        Sets list of menu options.

        Args:
            options_list: List of items (strings) to show in menu.
            select_item: Item to select initially.

        Returns:
            None.
        """
        self.clear()
        self.options_list = options_list

        if self.add_blank_option:
            self.addItem("")
        for item in options_list:
            if item == "---":
                self.insertSeparator(self.count())
            else:
                self.addItem(item)
        if select_item is not None:
            self.setValue(select_item)

    def value(self):
        if self.result_as_idx:
            val = self.currentIndex()
            if self.add_blank_option:
                val -= 1
        else:
            val = self.currentText()

        return val

    def setValue(self, val):
        if type(val) == int and val < len(self.options_list) and self.result_as_idx:
            val = self.options_list[val]
        super(FieldComboWidget, self).setCurrentText(str(val))


class SandboxWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        #### Videos ####
        videos_box = QGroupBox("Videos")
        self.videos = QWidget()  # VideosTableWidget()

        vb = QVBoxLayout()
        vb.addWidget(self.videos)
        videos_box.setLayout(vb)
        layout.addWidget(videos_box)
        ####

        #### Models ####
        models_box = QGroupBox("Select model(s)")
        self.models = QWidget()  # ModelsTableWidget()

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

        """
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
        """


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication

    app = QApplication([])
    window = SandboxWidget()
    window.show()
    app.exec_()
