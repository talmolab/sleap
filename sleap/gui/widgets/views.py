"""GUI code for the views (e.g. Videos, Skeleton, Labeling Suggestions, etc.)."""

from qtpy.QtWidgets import QWidget, QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox
from qtpy.QtCore import Qt


class CollapsibleGroupBox(QGroupBox):
    """A collapsible group box."""

    def __init__(self, title, parent=None):
        super().__init__(title, parent=parent)

        self.setCheckable(True)
        self.setChecked(True)

        self.toggled.connect(self.on_toggled)

        # Set checkbox to be a down arrow when collapsed
        self.setStyleSheet(
            """            
            QGroupBox::indicator:unchecked {
            image: url(file://down-arrow.png);
            }
            
            QGroupBox::indicator:checked {
            width: 0; 
            height: 0; 
            border-top: 10px solid white;
            border-bottom: 10px solid white;
            border-right: 10px solid white;
            border-left: 10px solid black;
            }
            """
        )

    def on_toggled(self, checked):
        """Called when the group box is toggled."""

        # Hide children if we're collapsing
        for child in self.findChildren(QWidget):
            child.setVisible(checked)
