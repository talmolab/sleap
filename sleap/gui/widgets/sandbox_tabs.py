import sys
from PySide2.QtCore import *
from PySide2.QtWidgets import *

from sleap.gui.widgets.models_table import ModelsTableWidget
from sleap.gui.widgets.videos_table import VideosTableWidget


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 tabs - pythonspot.com'
        self.setWindowTitle(self.title)

        self.table_widget = InferenceConfigWidget(self)
        self.setCentralWidget(self.table_widget)

        self.setMinimumWidth(800)

        self.show()


class InferenceConfigWidget(QWidget):

    def __init__(self, parent):
        super(InferenceConfigWidget, self).__init__(parent)
        self.layout = QVBoxLayout()

        # Tabs screen
        self.tabs = InferenceConfigWidget.build_tabs_widget()
        self.layout.addWidget(self.tabs)

        # Separator
        self.layout.addSpacing(5)

        # Action buttons
        self.action_buttons = InferenceConfigWidget.build_action_buttons_widget()
        self.layout.addWidget(self.action_buttons)

        self.setLayout(self.layout)

    @staticmethod
    def build_action_buttons_widget():
        action_buttons = QWidget()
        action_buttons.layout = QHBoxLayout()

        action_buttons.train_button = QPushButton(" Start ")
        action_buttons.layout.addWidget(action_buttons.train_button)

        action_buttons.save_button = QPushButton(" Save configuration ")
        action_buttons.layout.addWidget(action_buttons.save_button)

        action_buttons.export_button = QPushButton(" Export inference job package ")
        action_buttons.layout.addWidget(action_buttons.export_button)

        action_buttons.setLayout(action_buttons.layout)
        return action_buttons

    @staticmethod
    def build_tabs_widget():
        tabs = QTabWidget()
        # Add tabs
        InferenceConfigWidget.add_settings_tab(tabs)
        InferenceConfigWidget.add_videos_tab(tabs)
        InferenceConfigWidget.add_models_tab(tabs)
        return tabs

    @staticmethod
    def add_videos_tab(tabs):
        tab = QWidget()
        tab.layout = QVBoxLayout()
        tabs.videos_widget = VideosTableWidget()
        tab.layout.addWidget(tabs.videos_widget)
        tab.setLayout(tab.layout)
        tabs.addTab(tab, "Videos")

    @staticmethod
    def add_models_tab(tabs):
        tab = QWidget()
        tab.layout = QVBoxLayout()
        tab.models_widget = ModelsTableWidget()
        tab.layout.addWidget(tab.models_widget)
        tab.setLayout(tab.layout)
        tabs.addTab(tab, "Models")

    @staticmethod
    def add_settings_tab(tabs):
        tab = QWidget()
        tab.layout = QVBoxLayout()
        tab.setLayout(tab.layout)

        InferenceConfigWidget.add_model_type_box(tab)
        InferenceConfigWidget.add_num_instances_box(tab)
        InferenceConfigWidget.add_tracking_box(tab)

        # TODO:
        # - number of instances
        # - centroid/instance similarity/iou

        ####

        # TODO:
        # - batch size
        # - output folder or same as videos (.prediction.slp)
        # - [ ] open results in GUI
        # - [ ] export analysis file

        tabs.addTab(tab, "Settings")

    @staticmethod
    def add_model_type_box(tab):
        model_type_box = QGroupBox("Trained model type")
        model_type = QFormLayout()
        model_type_box.setLayout(model_type)

        model_type_widget = QComboBox()
        model_type_widget.addItems(
            ["Multi Instance / Top Down", "Multi Instance / Bottom Up", "Single Instance"]
        )
        model_type_widget.setMaximumWidth(250)
        model_type.addRow("Model type", model_type_widget)

        tab.layout.addWidget(model_type_box)

    @staticmethod
    def add_num_instances_box(tab):
        num_instances_box = QGroupBox("Number of instances")
        num_instances = QFormLayout()
        num_instances_box.setLayout(num_instances)

        num_instances_widget = QSpinBox()
        num_instances_widget.setRange(1, 100)
        num_instances_widget.setValue(2)
        num_instances_widget.setToolTip(
            "Max number of instances in single frame."
        )
        num_instances_widget.setMaximumWidth(50)
        num_instances.addRow("Max instances in frame", num_instances_widget)

        tab.layout.addWidget(num_instances_box)

    @staticmethod
    def add_tracking_box(tab):
        tracking_box = QGroupBox("Instance tracking")
        tracking = QFormLayout()
        tracking_box.setLayout(tracking)

        ql = QLabel(
            "Instance tracking is performed after inference to associate multiple detected animal instances "
            "across frames. Note: Instance tracking is not necessary for single animal videos.\n"
        )
        ql.setWordWrap(True)
        tracking.addRow("What?", ql)

        enable_tracking = QCheckBox()
        tracking.addRow("Enable tracking", enable_tracking)

        tracking_method_widget = QComboBox()
        tracking_method_widget.addItems(
            ["Simple", "Flow shift", "Kalman filter"]
        )
        tracking_method_widget.setMaximumWidth(150)
        tracking.addRow("Tracking method", tracking_method_widget)

        tracking_window_widget = QSpinBox()
        tracking_window_widget.setRange(0, 1000)
        tracking_window_widget.setValue(5)
        tracking_window_widget.setToolTip(
            "Number of past frames to consider when associating tracks."
        )
        tracking_window_widget.setMaximumWidth(50)
        tracking.addRow("Window size", tracking_window_widget)

        tab.layout.addWidget(tracking_box)

    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())