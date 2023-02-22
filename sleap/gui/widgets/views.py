"""GUI code for the views (e.g. Videos, Skeleton, Labeling Suggestions, etc.)."""

from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QToolButton,
    QFrame,
    QSizePolicy,
)
from qtpy.QtCore import Qt


class CollapsibleWidget(QWidget):
    """An animated collapsible QWidget.

    Derived from: https://stackoverflow.com/a/37119983/13281260
    """

    def __init__(self, title: str, parent: QWidget = None):
        super().__init__(parent=parent)

        # Create our custom toggle button.
        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)

        self.toggle_button.clicked.connect(self.toggle_button_callback)

        # Create the header line.
        header_line = QFrame()
        header_line.setFrameShape(QFrame.HLine)
        header_line.setFrameShadow(QFrame.Sunken)
        header_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # Created the layout for the header.
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(header_line)

        # Create a widget to apply the header layout to.
        header_widget = QWidget()
        header_widget.setLayout(header_layout)

        # Content area for setting an external layout to.
        self.content_area = QWidget()

        # Tie everything together in a main layout.
        main_layout = QVBoxLayout()
        main_layout.addWidget(header_widget)
        main_layout.addWidget(self.content_area)
        self.setLayout(main_layout)

    def toggle_button_callback(self, checked: bool):
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

        # Hide children if we're collapsing
        for child in self.content_area.findChildren(QWidget):
            child.setVisible(checked)

    def set_content_layout(self, content_layout):
        self.content_area.setLayout(content_layout)
