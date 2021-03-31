import sys
from PySide2.QtCore import *
from PySide2.QtWidgets import *



class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 tabs - pythonspot.com'
        self.setWindowTitle(self.title)

        self.table_widget = InferenceConfigWidget(self)
        self.setCentralWidget(self.table_widget)

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

        action_buttons.train_button = QPushButton("Train")
        action_buttons.layout.addWidget(action_buttons.train_button)

        action_buttons.save_button = QPushButton("Save")
        action_buttons.layout.addWidget(action_buttons.save_button)

        action_buttons.export_button = QPushButton("Export")
        action_buttons.layout.addWidget(action_buttons.export_button)

        action_buttons.setLayout(action_buttons.layout)
        return action_buttons

    @staticmethod
    def build_tabs_widget():
        tabs = QTabWidget()

        # Add tabs
        InferenceConfigWidget.add_videos_tab(tabs)
        InferenceConfigWidget.add_models_tab(tabs)
        InferenceConfigWidget.add_tracking_tab(tabs)

        return tabs

    @staticmethod
    def add_videos_tab(tabs):
        tab = QWidget()
        tab.layout = QVBoxLayout()
        tab.push_button = QPushButton("PyQt5 button videos")
        tab.layout.addWidget(tab.push_button)
        tab.setLayout(tab.layout)
        tabs.addTab(tab, "Videos")

    @staticmethod
    def add_models_tab(tabs):
        tab = QWidget()
        tab.layout = QVBoxLayout()
        tab.push_button = QPushButton("PyQt5 button models")
        tab.layout.addWidget(tab.push_button)
        tab.setLayout(tab.layout)
        tabs.addTab(tab, "Models")

    @staticmethod
    def add_tracking_tab(tabs):
        tab = QWidget()
        tab.layout = QVBoxLayout()

        tab.enable_checkbox = QCheckBox("Enable tracking")
        tab.layout.addWidget(tab.enable_checkbox)

        tab.tracking_type = QComboBox()
        tab.tracking_type.addItems(["1", "2", "3"])
        tab.layout.addWidget(tab.tracking_type)

        tab.push_button = QPushButton("PyQt5 button tracking")
        tab.layout.addWidget(tab.push_button)
        tab.setLayout(tab.layout)
        tabs.addTab(tab, "Tracking")

    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())