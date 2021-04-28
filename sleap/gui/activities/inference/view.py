import sys
from typing import Text, Optional, List

from PySide2.QtCore import QObject
from PySide2.QtWidgets import *

import sleap
from sleap.gui.activities.inference.controller import InferenceGuiController
from sleap.gui.activities.inference.model import InferenceGuiModel
from sleap.gui.activities.inference.enums import ModelType
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.learning.configs import ConfigFileInfo
from sleap.gui.widgets.videos_table import VideosTableWidget


class InferenceActivity(QMainWindow):
    def __init__(self, parent: Optional[QObject], ctrl: InferenceGuiController):
        super().__init__(parent)
        self.controller = ctrl
        self.input_widgets = InferenceActivityInputWidgets()

        self.title = "Inference"
        self.setWindowTitle(self.title)

        self.central_widget = InferenceActivityCentralWidget(self)
        self.setCentralWidget(self.central_widget)

        self.connect_widgets()
        self.update_widgets()

        self.setMinimumWidth(800)
        self.show()

    def update_widgets(self):
        self.update_model_type()
        self.update_enable_tracking()

    def connect_widgets(self):
        self.input_widgets.model_type.currentIndexChanged.connect(
            lambda: self.update_model_type())

        self.input_widgets.enable_tracking.stateChanged.connect(
            lambda: self.update_enable_tracking())

    def update_model_type(self) -> None:
        model_type = self.input_widgets.model_type.currentText()
        if model_type == ModelType.SINGLE_INSTANCE.value:
            self.input_widgets.single_instance_model.setEnabled(True)
            self.input_widgets.bottom_up_model.setEnabled(False)
            self.input_widgets.top_down_centroid_model.setEnabled(False)
            self.input_widgets.top_down_centered_instance_model.setEnabled(False)
        elif model_type == ModelType.BOTTOM_UP.value:
            self.input_widgets.single_instance_model.setEnabled(False)
            self.input_widgets.bottom_up_model.setEnabled(True)
            self.input_widgets.top_down_centroid_model.setEnabled(False)
            self.input_widgets.top_down_centered_instance_model.setEnabled(False)
        elif model_type == ModelType.TOP_DOWN.value:
            self.input_widgets.single_instance_model.setEnabled(False)
            self.input_widgets.bottom_up_model.setEnabled(False)
            self.input_widgets.top_down_centroid_model.setEnabled(True)
            self.input_widgets.top_down_centered_instance_model.setEnabled(True)
        else:
            raise ValueError(f"Invalid model type {model_type}")

    def update_enable_tracking(self):
        tracking_enabled = self.input_widgets.enable_tracking.isChecked()
        self.input_widgets.tracking_method.setEnabled(tracking_enabled)
        self.input_widgets.tracking_window_size.setEnabled(tracking_enabled)


class InferenceActivityInputWidgets(object):
    """
    Contains all input widgets, and provides methods for loading a state (populating widgets) and exporting
    a state (collecting widget values).
    Using slots to enforce registration off all input widgets.
    """

    __slots__ = [
        # models tab
        "model_type",
        "single_instance_model",
        "bottom_up_model",
        "top_down_centroid_model",
        "top_down_centered_instance_model",
        # videos tab
        "videos_table",
        # instances tab
        "max_num_instances_in_frame",
        "enable_tracking",
        "tracking_method",
        "tracking_window_size",
        # output tab
        "output_dir_path",
        "output_file_name",
        "include_empty_frames",
        "verbosity",
    ]


class InferenceActivityCentralWidget(QWidget):
    def __init__(self, parent):
        super(InferenceActivityCentralWidget, self).__init__(parent)
        layout = QVBoxLayout()

        # Tabs
        layout.addWidget(self.build_tabs_widget())

        # Separator
        layout.addSpacing(5)

        # Action buttons
        layout.addWidget(self.build_action_buttons_widget())

        self.setLayout(layout)

    @property
    def controller(self) -> InferenceGuiController:
        return self.parent().controller

    @property
    def input_widgets(self) -> InferenceActivityInputWidgets:
        return self.parent().input_widgets

    def build_tabs_widget(self):
        tabs = QTabWidget(self)
        self.add_models_tab(tabs)
        self.add_videos_tab(tabs)
        self.add_instances_tab(tabs)
        self.add_output_tab(tabs)
        return tabs

    def add_models_tab(self, tabs: QTabWidget):
        # group box
        model_group_box = QGroupBox("Select trained model(s) for inference")
        model_form_layout = QFormLayout()
        model_group_box.setLayout(model_form_layout)

        # model type
        model_type_widget = QComboBox()
        model_type_widget.addItems(self.controller.get_model_type_names())
        model_type_widget.setMaximumWidth(250)
        model_form_layout.addRow("Type", model_type_widget)
        self.input_widgets.model_type = model_type_widget

        # model dirs
        self.input_widgets.single_instance_model = self.add_browse_widget(
            model_form_layout,
            directory=True,
            caption="Single Instance Model Directory",
        )
        self.input_widgets.bottom_up_model = self.add_browse_widget(
            model_form_layout,
            directory=True,
            caption="Bottom Up Model Directory",
        )
        self.input_widgets.top_down_centroid_model = self.add_browse_widget(
            model_form_layout,
            directory=True,
            caption="Top Down Centroid Model Directory",
        )
        self.input_widgets.top_down_centered_instance_model = self.add_browse_widget(
            model_form_layout,
            directory=True,
            caption="Top Down Centered Instance Model Directory",
        )

        # set layout and add
        layout = QVBoxLayout()
        layout.addWidget(model_group_box)
        tab = QWidget(self)
        tab.setLayout(layout)
        tabs.addTab(tab, "Models")

    def add_videos_tab(self, tabs: QTabWidget):
        layout = QVBoxLayout()
        videos_table_widget = VideosTableWidget(
            table_model=self.controller.get_video_table_model()
        )
        layout.addWidget(videos_table_widget)
        self.input_widgets.videos_table = videos_table_widget

        tab = QWidget(self)
        tab.setLayout(layout)
        tabs.addTab(tab, "Videos")

    def add_instances_tab(self, tabs: QTabWidget):
        layout = QVBoxLayout()

        # num instances
        num_instances_box = QGroupBox("Animal instances")
        num_instances = QFormLayout(self)
        num_instances_box.setLayout(num_instances)
        num_instances_widget = QSpinBox()
        num_instances_widget.setRange(1, 100)
        num_instances_widget.setValue(2)
        num_instances_widget.setMaximumWidth(50)
        num_instances.addRow("Max number of instances in frame", num_instances_widget)
        layout.addWidget(num_instances_box)
        self.input_widgets.max_num_instances_in_frame = num_instances_widget

        # tracking box
        tracking_box = QGroupBox("Instance tracking")
        tracking_layout = QFormLayout(self)
        tracking_box.setLayout(tracking_layout)

        ql = QLabel(
            "Instance tracking is performed after inference to associate multiple detected animal instances "
            "across frames. Note: Instance tracking is not necessary for single animal videos.\n"
        )
        ql.setWordWrap(True)
        tracking_layout.addRow("What?", ql)

        # enable tracking
        enable_tracking = QCheckBox()
        tracking_layout.addRow("Enable tracking", enable_tracking)
        self.input_widgets.enable_tracking = enable_tracking

        # tracking method
        tracking_method_widget = QComboBox()
        tracking_method_widget.addItems(self.controller.get_tracking_method_names())
        tracking_method_widget.setMaximumWidth(150)
        tracking_layout.addRow("Tracking method", tracking_method_widget)
        self.input_widgets.tracking_method = tracking_method_widget

        # tracking window
        tracking_window_widget = QSpinBox()
        tracking_window_widget.setRange(0, 1000)
        tracking_window_widget.setValue(5)
        tracking_window_widget.setToolTip(
            "Number of past frames to consider when associating tracks."
        )
        tracking_window_widget.setMaximumWidth(50)
        tracking_layout.addRow("Window size", tracking_window_widget)
        self.input_widgets.tracking_window_size = tracking_window_widget

        layout.addWidget(tracking_box)

        # set layout and add tab
        tab = QWidget(self)
        tab.setLayout(layout)
        tabs.addTab(tab, "Instances")

    def add_output_tab(self, tabs: QTabWidget):
        output_box = QGroupBox("Inference Output")
        output_box_layout = QFormLayout(self)
        output_box.setLayout(output_box_layout)

        # output dir and file name
        self.input_widgets.output_dir_path = self.add_browse_widget(
            output_box_layout, directory=True, caption="Output dir"
        )
        output_file_name = QLineEdit()
        output_box_layout.addRow("Output file name", output_file_name)
        self.input_widgets.output_file_name = output_file_name

        # include empty frames
        empty_frames = QCheckBox()
        empty_frames.setToolTip(
            "Include frames with no detected instances in the saved output file."
        )
        output_box_layout.addRow("Include empty frames  ", empty_frames)
        self.input_widgets.include_empty_frames = empty_frames

        # verbosity
        verbosity_widget = QComboBox()
        verbosity_widget.addItems(self.controller.get_verbosity_names())
        verbosity_widget.setMaximumWidth(150)
        output_box_layout.addRow("Log format / verbosity", verbosity_widget)
        self.input_widgets.verbosity = verbosity_widget

        # set layout and add tab
        layout = QVBoxLayout()
        layout.addWidget(output_box)
        tab = QWidget(self)
        tab.setLayout(layout)
        tabs.addTab(tab, "Output")

    def add_browse_widget(
        self,
        layout: QLayout,
        directory: bool,
        caption: Text,
        from_dir: Optional[Text] = None,
        filters: Optional[List[Text]] = None,
    ) -> QLineEdit:
        widget = QHBoxLayout()
        path_text = QLineEdit()
        widget.addWidget(path_text)

        def browse():
            if directory:
                path = FileDialog.openDir(self, dir=from_dir, caption=caption)
            else:
                path = FileDialog.open(
                    self, dir=from_dir, caption=caption, filters=";;".join(filters)
                )
            path_text.setText(path)

        browse_button = QPushButton("Browse..")
        browse_button.clicked.connect(lambda: browse())
        widget.addWidget(browse_button)
        layout.addRow(caption, widget)
        return path_text

    def build_action_buttons_widget(self):
        action_buttons = QWidget(self)
        action_buttons.layout = QHBoxLayout()

        # Run inference button
        action_buttons.run_button = QPushButton(parent=self, text=" Run ")
        action_buttons.run_button.clicked.connect(lambda: self.controller.run())
        action_buttons.layout.addWidget(action_buttons.run_button)

        # Save configuration button
        action_buttons.save_button = QPushButton(
            parent=self, text=" Save configuration.. "
        )
        action_buttons.save_button.clicked.connect(lambda: self.controller.save())
        action_buttons.layout.addWidget(action_buttons.save_button)

        # Export inference job button
        action_buttons.export_button = QPushButton(
            parent=self, text=" Export inference job package.. "
        )
        action_buttons.export_button.clicked.connect(lambda: self.controller.export())
        action_buttons.layout.addWidget(action_buttons.export_button)

        action_buttons.setLayout(action_buttons.layout)
        return action_buttons


if __name__ == "__main__":
    app = QApplication()
    model = InferenceGuiModel()

    # Populate mock data in the GUI model
    videos = [
        sleap.load_video(f"C://Users//ariem//work//sleap_data//videos//{p}")
        for p in ["small_robot.mp4", "centered_pair_small.mp4"]
    ]
    model.videos.videos_table_model.items = videos

    model.models.centroid_model = ConfigFileInfo(path="cmp", config=None)
    model.models.centered_instance_model = ConfigFileInfo(path="cip", config=None)

    controller = InferenceGuiController(model)

    ex = InferenceActivity(parent=None, ctrl=controller)

    sys.exit(app.exec_())
