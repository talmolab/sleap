"""GUI code for the views (e.g. Videos, Skeleton, Labeling Suggestions, etc.)."""

from typing import Tuple
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QToolButton,
    QFrame,
    QSizePolicy,
    QComboBox,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor


class CollapsibleWidget(QWidget):
    """An animated collapsible QWidget.

    Derived from: https://stackoverflow.com/a/37119983/13281260
    """

    def __init__(self, title: str, parent: QWidget = None):
        super().__init__(parent=parent)

        # Create the header widget which contains the toggle button.
        self.header_widget, self.toggle_button = self.create_header_widget(title)

        # Content area for setting an external layout to.
        self.content_area = QWidget()

        # Tie everything together in a main layout.
        main_layout = self.create_main_layout()
        self.setLayout(main_layout)

    def create_toggle_button(self, title="") -> QToolButton:
        """Create our custom toggle button."""

        toggle_button = QToolButton()
        toggle_button.setStyleSheet("QToolButton { border: none; }")
        toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        toggle_button.setText(title)
        toggle_button.setCheckable(True)
        toggle_button.setChecked(False)
        toggle_button.setCursor(QCursor(Qt.PointingHandCursor))

        toggle_button.clicked.connect(self.toggle_button_callback)

        return toggle_button

    def create_header_widget(self, title="") -> Tuple[QWidget, QToolButton]:
        """Create header widget which includes `QToolButton` and `QFrame`."""

        # Create our custom toggle button.
        toggle_button = self.create_toggle_button(title)

        # Create the header line.
        header_line = QFrame()
        header_line.setFrameShape(QFrame.HLine)
        header_line.setFrameShadow(QFrame.Plain)
        header_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        header_line.setStyleSheet("color: #dcdcdc")

        # Created the layout for the header.
        header_layout = QHBoxLayout()
        header_layout.addWidget(toggle_button)
        header_layout.addWidget(header_line)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Create a widget to apply the header layout to.
        header_widget = QWidget()
        header_widget.setLayout(header_layout)

        return header_widget, toggle_button

    def create_main_layout(self) -> QVBoxLayout:
        """Tie everything together in a main layout."""

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.header_widget)
        main_layout.addWidget(self.content_area)
        main_layout.setContentsMargins(0, 0, 0, 0)

        return main_layout

    def toggle_button_callback(self, checked: bool):
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

        # Hide children if we're collapsing
        for child in self.content_area.findChildren(QWidget):
            child.setVisible(checked)

        # Collapse combo box (otherwise, visiblity opens combo)
        if checked:
            combo = self.content_area.findChild(QComboBox)
            combo.hidePopup()

    def set_content_layout(self, content_layout):
        self.content_area.setLayout(content_layout)
        self.toggle_button_callback(self.toggle_button.isChecked())
