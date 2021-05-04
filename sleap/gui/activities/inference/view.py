import sys
from typing import Text, Optional, List, Callable

from PySide2 import QtGui
from PySide2.QtCore import QObject
from PySide2.QtWidgets import *

import sleap
from sleap.gui.activities.inference.controller import InferenceGuiController
from sleap.gui.activities.inference.model import InferenceGuiModel
from sleap.gui.activities.inference.enums import ModelType, TrackerType, Verbosity
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.learning.configs import ConfigFileInfo
from sleap.gui.learning.dialog import TrainingEditorWidget
from sleap.gui.widgets.videos_table import VideosTableWidget, VideosTableModel
from sleap.nn.inference import main as run_inference_and_tracking_from_cli


class InferenceActivity(QMainWindow):
    def __init__(self, parent: Optional[QObject], ctrl: InferenceGuiController):
        super().__init__(parent)
        self.controller = ctrl
        self.input_widgets = InferenceActivityInputWidgets()

        self.title = "Inference and Tracking"
        self.setWindowTitle(self.title)

        self.central_widget = InferenceActivityCentralWidget(self)
        self.setCentralWidget(self.central_widget)

        self.load_content()
        self.connect_widgets()
        self.update_widgets()

        self.setMinimumWidth(1000)
        self.show()

    def load_content(self):
        self.input_widgets.set_content(
            model_type=self.controller.get_model_type().display(),
            single_instance_model=self.controller.get_single_instance_model_path(),
            bottom_up_model=self.controller.get_bottom_up_model_path(),
            top_down_centroid_model=self.controller.get_top_down_centroid_model_path(),
            top_down_centered_instance_model=self.controller.get_top_down_centered_instance_model_path(),
            video_paths=self.controller.get_video_paths(),
            video_frames=self.controller.get_video_frames(),
            max_num_instances_in_frame=self.controller.get_max_num_instances_in_frame(),
            enable_tracking=self.controller.get_enable_tracking(),
            tracking_method=self.controller.get_tracking_method().display(),
            tracking_window_size=self.controller.get_tracking_window_size(),
            output_dir_path=self.controller.get_output_dir_path(),
            output_file_suffix=self.controller.get_output_file_suffix(),
            include_empty_frames=self.controller.get_include_empty_frames(),
            verbosity=self.controller.get_verbosity().display(),
        )

    def update_widgets(self):
        self.update_model_type()
        self.update_enable_tracking()

    def connect_widgets(self):
        self.input_widgets.model_type.currentIndexChanged.connect(
            lambda: self.update_model_type()
        )

        self.input_widgets.enable_tracking.stateChanged.connect(
            lambda: self.update_enable_tracking()
        )

    def update_model_type(self) -> None:
        model_type = self.input_widgets.model_type.currentText()
        if model_type == ModelType.SINGLE_INSTANCE.display():
            self.input_widgets.single_instance_model.setEnabled(True)
            self.input_widgets.bottom_up_model.setEnabled(False)
            self.input_widgets.top_down_centroid_model.setEnabled(False)
            self.input_widgets.top_down_centered_instance_model.setEnabled(False)
        elif model_type == ModelType.BOTTOM_UP.display():
            self.input_widgets.single_instance_model.setEnabled(False)
            self.input_widgets.bottom_up_model.setEnabled(True)
            self.input_widgets.top_down_centroid_model.setEnabled(False)
            self.input_widgets.top_down_centered_instance_model.setEnabled(False)
        elif model_type == ModelType.TOP_DOWN.display():
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
        "output_file_suffix",
        "include_empty_frames",
        "verbosity",
    ]

    def extract_content(self) -> dict:
        res = {
            "model_type": self.model_type.currentText(),
            "single_instance_model": self.single_instance_model.text(),
            "bottom_up_model": self.bottom_up_model.text(),
            "top_down_centroid_model": self.top_down_centroid_model.text(),
            "top_down_centered_instance_model": self.top_down_centered_instance_model.text(),
            "video_paths": self.videos_table.checked_video_paths,
            "video_frames": self.videos_table.checked_video_frames,
            "max_num_instances_in_frame": self.max_num_instances_in_frame.text(),
            "enable_tracking": self.enable_tracking.isChecked(),
            "tracking_method": self.tracking_method.currentText(),
            "tracking_window_size": self.tracking_window_size.text(),
            "output_dir_path": self.output_dir_path.text(),
            "output_file_suffix": self.output_file_suffix.text(),
            "include_empty_frames": self.include_empty_frames.isChecked(),
            "verbosity": self.verbosity.currentText(),
        }
        return res

    def set_content(
        self,
        model_type: str,
        single_instance_model: str,
        bottom_up_model: str,
        top_down_centroid_model: str,
        top_down_centered_instance_model: str,
        video_paths: List[str],
        video_frames: List[str],
        max_num_instances_in_frame: int,
        enable_tracking: bool,
        tracking_method: str,
        tracking_window_size: int,
        output_dir_path: str,
        output_file_suffix: str,
        include_empty_frames: bool,
        verbosity: str,
    ) -> None:
        self.model_type.setCurrentText(model_type)
        self.single_instance_model.setText(single_instance_model)
        self.bottom_up_model.setText(bottom_up_model)
        self.top_down_centroid_model.setText(top_down_centroid_model)
        self.top_down_centered_instance_model.setText(top_down_centered_instance_model)

        self.videos_table.set_videos(video_paths, video_frames)

        self.max_num_instances_in_frame.setValue(max_num_instances_in_frame)
        self.enable_tracking.setChecked(enable_tracking)
        self.tracking_method.setCurrentText(tracking_method)
        self.tracking_window_size.setValue(tracking_window_size)

        self.output_dir_path.setText(output_dir_path)
        self.output_file_suffix.setText(output_file_suffix)
        self.include_empty_frames.setChecked(include_empty_frames)
        self.verbosity.setCurrentText(verbosity)


class InferenceActivityCentralWidget(QWidget):
    config_viewer_widgets = {}

    tabs_view = None
    progress_view = None
    run_button = None
    stop_button = None
    video_processing_label = None
    frame_processing_label = None
    eta_processing_label = None
    stop_requested = False

    def __init__(self, parent):
        super(InferenceActivityCentralWidget, self).__init__(parent)
        layout = QVBoxLayout()

        # Tabs
        self.tabs_view = self.build_tabs_widget()
        layout.addWidget(self.tabs_view)

        # Progress viewer
        self.progress_view = self.build_progress_widget()
        layout.addWidget(self.progress_view)

        # Separator
        layout.addSpacing(5)

        # Action buttons
        layout.addWidget(self.build_action_buttons_widget())

        self.set_inference_running(False)

        self.setLayout(layout)

    @property
    def controller(self) -> InferenceGuiController:
        return self.parent().controller

    @property
    def input_widgets(self) -> InferenceActivityInputWidgets:
        return self.parent().input_widgets

    def view_config(self, config_path: str) -> None:
        if not config_path:
            QMessageBox(
                windowTitle="No file", text="Training config file not specified."
            ).exec_()
            return
        widget = self.config_viewer_widgets.get(config_path)
        if widget is None:
            try:
                config_file_info = ConfigFileInfo.from_config_file(config_path)
                widget = TrainingEditorWidget.from_trained_config(config_file_info)
                # Without this the viewer goes out of scope and destroys itself
                self.config_viewer_widgets[config_path] = widget
            except Exception as e:
                QMessageBox(windowTitle="Error", text=str(e)).exec_()
                return

        widget.show()
        widget.raise_()
        widget.activateWindow()

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

        # model pipeline type
        model_type_widget = QComboBox()
        model_type_widget.addItems([mt.display() for mt in ModelType])
        model_type_widget.setMaximumWidth(250)
        model_form_layout.addRow("Type", model_type_widget)
        self.input_widgets.model_type = model_type_widget

        # model training config paths
        file_dialog_filter = "Config (training_config.json)"

        self.input_widgets.top_down_centroid_model = self.add_browse_widget(
            model_form_layout,
            directory=False,
            caption="Top Down Centroid Training Config",
            filter=file_dialog_filter,
            view_callback=lambda: self.view_config(
                self.input_widgets.top_down_centroid_model.text()
            ),
        )
        self.input_widgets.top_down_centered_instance_model = self.add_browse_widget(
            model_form_layout,
            directory=False,
            caption="Top Down Centered Instance Training Config",
            filter=file_dialog_filter,
            view_callback=lambda: self.view_config(
                self.input_widgets.top_down_centered_instance_model.text()
            ),
        )
        self.input_widgets.bottom_up_model = self.add_browse_widget(
            model_form_layout,
            directory=False,
            caption="Bottom Up Training Config",
            filter=file_dialog_filter,
            view_callback=lambda: self.view_config(
                self.input_widgets.bottom_up_model.text()
            ),
        )
        self.input_widgets.single_instance_model = self.add_browse_widget(
            model_form_layout,
            directory=False,
            caption="Single Instance Training Config",
            filter=file_dialog_filter,
            view_callback=lambda: self.view_config(
                self.input_widgets.single_instance_model.text()
            ),
        )

        # set layout and add
        layout = QVBoxLayout()
        layout.addWidget(model_group_box)
        tab = QWidget(self)
        tab.setLayout(layout)
        tabs.addTab(tab, "Models")

    def add_videos_tab(self, tabs: QTabWidget):
        layout = QVBoxLayout()
        videos_table_widget = VideosTableWidget(table_model=VideosTableModel())
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
        tracking_method_widget.addItems([tm.display() for tm in TrackerType])
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
        tabs.addTab(tab, "Instance Tracking")

    def add_output_tab(self, tabs: QTabWidget):
        output_box = QGroupBox("Inference Output")
        output_box_layout = QFormLayout(self)
        output_box.setLayout(output_box_layout)

        # output dir and file name
        self.input_widgets.output_dir_path = self.add_browse_widget(
            output_box_layout, directory=True, caption="Output dir"
        )
        output_file_suffix = QLineEdit()
        output_box_layout.addRow("Output file suffix", output_file_suffix)
        self.input_widgets.output_file_suffix = output_file_suffix

        # include empty frames
        empty_frames = QCheckBox()
        empty_frames.setToolTip(
            "Include frames with no detected instances in the saved output file."
        )
        output_box_layout.addRow("Include empty frames  ", empty_frames)
        self.input_widgets.include_empty_frames = empty_frames

        # verbosity
        verbosity_widget = QComboBox()
        verbosity_widget.addItems([v.display() for v in Verbosity])
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
        filter: Optional[Text] = None,
        view_callback: Optional[Callable[[], None]] = None,
    ) -> QLineEdit:
        widget = QHBoxLayout()
        path_text = QLineEdit()
        widget.addWidget(path_text)

        def browse():
            if directory:
                path = FileDialog.openDir(self, dir=from_dir, caption=caption)
            else:
                open_res = FileDialog.open(
                    self, dir=from_dir, caption=caption, filter=filter
                )
                path = open_res[0] if open_res else None
            if path:
                path_text.setText(path)

        browse_button = QPushButton("Browse..")
        browse_button.clicked.connect(lambda: browse())
        widget.addWidget(browse_button)

        if view_callback is not None:
            view_button = QPushButton("View")
            view_button.clicked.connect(lambda: view_callback())
            widget.addWidget(view_button)

        layout.addRow(caption, widget)
        return path_text

    def build_progress_widget(self):
        widget = QWidget(self)
        widget.layout = QFormLayout(self)

        self.video_processing_label = QLabel(text="")
        widget.layout.addRow("Processing file", self.video_processing_label)

        self.frame_processing_label = QLabel(text="")
        widget.layout.addRow("Frames", self.frame_processing_label)

        self.eta_processing_label = QLabel(text="")
        widget.layout.addRow("ETA for current file", self.eta_processing_label)

        widget.setLayout(widget.layout)
        return widget

    def build_action_buttons_widget(self):
        action_buttons = QWidget(self)
        action_buttons.layout = QHBoxLayout()

        # Run inference button
        self.run_button = QPushButton(parent=self, text=" Run ")
        self.run_button.clicked.connect(
            lambda: [
                self.set_inference_running(True),
                self.controller.run(
                    content=self.input_widgets.extract_content(),
                    callback=lambda output: self.inference_callback(output),
                ),
            ]
        )
        action_buttons.layout.addWidget(self.run_button)

        # Stop inference button
        self.stop_button = QPushButton(parent=self, text=" Stop ")
        self.stop_button.clicked.connect(lambda: self.request_stop())
        action_buttons.layout.addWidget(self.stop_button)

        # TODO: Implement these
        import_export_enabled = False
        if import_export_enabled:
            # Save configuration button
            action_buttons.save_button = QPushButton(
                parent=self, text=" Save configuration.. "
            )
            action_buttons.save_button.clicked.connect(
                lambda: self.controller.save(
                    content=self.input_widgets.extract_content()
                )
            )
            action_buttons.layout.addWidget(action_buttons.save_button)

            # Export inference job button
            action_buttons.export_button = QPushButton(
                parent=self, text=" Export inference job package.. "
            )
            action_buttons.export_button.clicked.connect(
                lambda: self.controller.export()
            )
            action_buttons.layout.addWidget(action_buttons.export_button)

        action_buttons.setLayout(action_buttons.layout)
        return action_buttons

    def inference_callback(self, output: dict) -> bool:
        if output:
            self.controller.log(f"Processing from view: {output}")
            self.video_processing_label.setText(output.get("video", ""))
            if "n_processed" in output:
                self.frame_processing_label.setText(
                    f"Processed {output['n_processed']} frames out of {output['n_total']}"
                )
            else:
                self.frame_processing_label.setText("")
            if "eta" in output:
                self.eta_processing_label.setText(f"{output['eta']} seconds")
            else:
                self.eta_processing_label.setText("loading..")
        if "status" in output:
            self.set_inference_running(False)
        QApplication.processEvents()
        return self.stop_requested

    def set_inference_running(self, running: bool) -> None:
        self.controller.log(f"Updating running state: {running}")
        if running:
            self.run_button.setVisible(False)
            self.stop_button.setVisible(True)
            self.tabs_view.setVisible(False)
            self.progress_view.setVisible(True)
        else:
            self.run_button.setVisible(True)
            self.stop_button.setVisible(False)
            self.tabs_view.setVisible(True)
            self.progress_view.setVisible(False)
            self.stop_requested = False
        QApplication.processEvents()

    def request_stop(self):
        self.stop_requested = True


def launch_inference_activity():
    app = QApplication()
    app.setApplicationName(
        f"Inference and Tracking | SLEAP v{sleap.version.__version__}"
    )
    app.setWindowIcon(QtGui.QIcon(sleap.util.get_package_file("sleap/gui/icon.png")))

    model = InferenceGuiModel()

    # Populate mock data in the GUI model
    videos = [
        f"C://Users//ariem//work//sleap_data//videos//{p}"
        for p in ["small_robot.mp4", "centered_pair_small.mp4"]
    ]
    model.videos.paths = videos

    model.models.centroid_model_path = "C:/Users/ariem/work/sleap_data/models/210225_170029.centroid.n=5/training_config.json"
    model.models.centered_instance_model_path = "C:/Users/ariem/work/sleap_data/models/210225_170213.centered_instance.n=5/training_config.json"
    model.output.output_dir_path = "C:/Users/ariem/work/sleap_data/predictions"

    controller = InferenceGuiController(model)

    ex = InferenceActivity(parent=None, ctrl=controller)

    sys.exit(app.exec_())


def main():
    if len(sys.argv) == 1:
        print(f"No args provided, launching inference and tracking GUI")
        launch_inference_activity()
    else:
        run_inference_and_tracking_from_cli()


if __name__ == "__main__":
    main()
