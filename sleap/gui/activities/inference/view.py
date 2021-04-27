import sys
from typing import Text, Optional

from PySide2.QtCore import QObject
from PySide2.QtWidgets import *

import sleap
from sleap.gui.activities.inference.controller import InferenceGuiController
from sleap.gui.activities.inference.model import InferenceGuiModel
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

        self.setMinimumWidth(800)

        self.show()


class InferenceActivityInputWidgets(object):
    __slots__ = [
        # models tab
        'model_type',
        'single_instance_model',
        'bottom_up_model',
        'top_down_centroid_model',
        'top_down_centered_instance_model',
        # videos tab
        'videos_table',
        # instances tab
        'max_num_instances_in_frame',
        'enable_tracking',
        'tracking_method',
        'tracking_window_size',
        # output tab
        'output_dir_path',
        'output_file_name',
        'include_empty_frames',
        'verbosity',
    ]


class InferenceActivityCentralWidget(QWidget):
    def __init__(self, parent):
        super(InferenceActivityCentralWidget, self).__init__(parent)
        self.layout = QVBoxLayout()

        # Tabs screen
        self.tabs = self.build_tabs_widget()
        self.layout.addWidget(self.tabs)

        # Separator
        self.layout.addSpacing(5)

        # Action buttons
        self.action_buttons = self.build_action_buttons_widget()
        self.layout.addWidget(self.action_buttons)

        self.setLayout(self.layout)

    @property
    def controller(self) -> InferenceGuiController:
        return self.parent().controller

    @property
    def input_widgets(self) -> InferenceActivityInputWidgets:
        return self.parent().input_widgets

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

    def build_tabs_widget(self):
        tabs = QTabWidget(self)
        # Add tabs
        self.add_models_tab(tabs)
        self.add_videos_tab(tabs)
        self.add_instances_tab(tabs)
        self.add_output_tab(tabs)
        return tabs

    def add_videos_tab(self, tabs: QTabWidget):
        tab = QWidget(self)
        tab.layout = QVBoxLayout()
        tab.videos_widget = VideosTableWidget(
            table_model=self.controller.get_video_table_model()
        )
        tab.layout.addWidget(tab.videos_widget)
        self.input_widgets.videos_table = tab.videos_widget
        tab.setLayout(tab.layout)
        tabs.addTab(tab, "Videos")

    def add_models_tab(self, tabs: QTabWidget):
        tab = QWidget(self)
        tab.layout = QVBoxLayout()
        tab.setLayout(tab.layout)
        self.add_model_type_box(tab.layout)
        tabs.addTab(tab, "Models")

    def add_output_tab(self, tabs: QTabWidget):
        tab = QWidget(self)
        tab.layout = QVBoxLayout()
        tab.setLayout(tab.layout)
        self.add_output_box(tab.layout)
        tabs.addTab(tab, "Output")

    def add_output_box(self, layout: QLayout):
        output_box = QGroupBox("Inference Output")
        output_box_layout = QFormLayout(self)
        output_box.setLayout(output_box_layout)

        # output dir and file name
        self.input_widgets.output_dir_path = self.add_file_browser_row(
            output_box_layout, "Output dir", controller.log)
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

        layout.addWidget(output_box)

    def add_instances_tab(self, tabs: QTabWidget):
        tab = QWidget(self)
        tab.layout = QVBoxLayout()
        tab.setLayout(tab.layout)

        self.add_num_instances_box(tab)
        self.add_tracking_box(tab)

        # TODO:
        # - number of instances
        # - centroid/instance similarity/iou

        ####

        # TODO:
        # - batch size
        # - output folder or same as videos (.prediction.slp)
        # - [ ] open results in GUI
        # - [ ] export analysis file

        tabs.addTab(tab, "Instances")

    def add_model_type_box(self, layout: QLayout):
        model_type_box = QGroupBox("Select trained model(s) for inference")
        model_type_layout = QFormLayout()
        model_type_box.setLayout(model_type_layout)

        # model type
        model_type_widget = QComboBox()
        model_type_widget.addItems(self.controller.get_model_type_names())
        model_type_widget.setMaximumWidth(250)
        model_type_layout.addRow("Type", model_type_widget)
        self.input_widgets.model_type = model_type_widget

        # model dirs
        self.input_widgets.single_instance_model = self.add_file_browser_row(
            model_type_layout, "Single Instance Model", self.controller.set_single_instance_model_path)
        self.input_widgets.bottom_up_model = self.add_file_browser_row(
            model_type_layout, "Bottom Up model", self.controller.set_single_instance_model_path)
        self.input_widgets.top_down_centroid_model = self.add_file_browser_row(
            model_type_layout, "Top Down Centroid model", self.controller.set_single_instance_model_path)
        self.input_widgets.top_down_centered_instance_model = self.add_file_browser_row(
            model_type_layout, "Top Down Centered Instance model", self.controller.set_single_instance_model_path)

        layout.addWidget(model_type_box)

    def add_file_browser_row(self, layout: QLayout, caption: Text, path_setter: callable) -> QLineEdit:
        widget = QHBoxLayout()
        path_text = QLineEdit()
        widget.addWidget(path_text)

        path_text.textChanged.connect(path_setter(path_text.text()))

        def browse():
            path = FileDialog.openDir(None, dir=None, caption="Select model folder...")
            path_text.setText(path)

        browse_button = QPushButton("Browse..")

        browse_button.clicked.connect(
            lambda: browse()
        )
        widget.addWidget(browse_button)
        layout.addRow(caption, widget)
        return path_text

    def add_num_instances_box(self, tab):
        num_instances_box = QGroupBox("Animal instances")
        num_instances = QFormLayout(self)
        num_instances_box.setLayout(num_instances)

        num_instances_widget = QSpinBox()
        num_instances_widget.setRange(1, 100)
        num_instances_widget.setValue(2)
        num_instances_widget.setMaximumWidth(50)
        num_instances.addRow("Max number of instances in frame", num_instances_widget)
        tab.layout.addWidget(num_instances_box)
        self.input_widgets.max_num_instances_in_frame = num_instances_widget

    def add_tracking_box(self, tab):
        tracking_box = QGroupBox("Instance tracking")
        tracking = QFormLayout(self)
        tracking_box.setLayout(tracking)

        ql = QLabel(
            "Instance tracking is performed after inference to associate multiple detected animal instances "
            "across frames. Note: Instance tracking is not necessary for single animal videos.\n"
        )
        ql.setWordWrap(True)
        tracking.addRow("What?", ql)

        # enable tracking
        enable_tracking = QCheckBox()
        tracking.addRow("Enable tracking", enable_tracking)
        self.input_widgets.enable_tracking = enable_tracking

        # tracking method
        tracking_method_widget = QComboBox()
        tracking_method_widget.addItems(self.controller.get_tracking_method_names())
        tracking_method_widget.setMaximumWidth(150)
        tracking.addRow("Tracking method", tracking_method_widget)
        self.input_widgets.tracking_method = tracking_method_widget

        # tracking window
        tracking_window_widget = QSpinBox()
        tracking_window_widget.setRange(0, 1000)
        tracking_window_widget.setValue(5)
        tracking_window_widget.setToolTip(
            "Number of past frames to consider when associating tracks."
        )
        tracking_window_widget.setMaximumWidth(50)
        tracking.addRow("Window size", tracking_window_widget)
        self.input_widgets.tracking_window_size = tracking_window_widget

        tab.layout.addWidget(tracking_box)


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
