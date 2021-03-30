"""
Module for Qt Widget to show multiple checkboxes for selecting.

Example: ::

    >>> mc = MultiCheckWidget(count=5, selected=[0,1],title="My Items")
    >>> mc.selectionChanged.connect(window.plot)
    >>> window.layout.addWidget(mc)

"""

from typing import List, Optional

from PySide2.QtCore import QRectF, Signal
from PySide2.QtWidgets import QGridLayout, QGroupBox, QButtonGroup, QCheckBox


class MultiCheckWidget(QGroupBox):
    """Qt Widget to show multiple checkboxes for a sequence of numbers.

    Args:
        count: The number of checkboxes to show.
        title: Display title for group of checkboxes.
        selected: List of checkbox numbers to initially check.
        default: Whether to default boxes as checked.
    """

    def __init__(
        self,
        *args,
        count: int,
        title: Optional[str] = "",
        selected: Optional[List] = None,
        default: Optional[bool] = False,
        **kwargs
    ):
        super(MultiCheckWidget, self).__init__(*args, **kwargs)

        # QButtonGroup is the logical container
        # it allows us to get list of checked boxes more easily
        self.check_group = QButtonGroup()
        self.check_group.setExclusive(False)  # more than one can be checked

        if title != "":
            self.setTitle(title)
            self.setFlat(False)
        else:
            self.setFlat(True)

        if selected is None:
            selected = list(range(count)) if default else []

        check_layout = QGridLayout()
        self.setLayout(check_layout)
        for i in range(count):
            check = QCheckBox("%d" % (i))
            # call signal/slot on self when one of the checkboxes is changed
            check.stateChanged.connect(lambda e: self.selectionChanged.emit())
            self.check_group.addButton(check, i)
            check_layout.addWidget(check, i // 8, i % 8)
        self.setSelected(selected)

    """
    selectionChanged signal sent when a checkbox gets a stateChanged signal
    """
    selectionChanged = Signal()

    def getSelected(self) -> list:
        """Method to get list of the checked checkboxes.

        Returns:
            list of checked checkboxes
        """
        selected = []
        for check_button in self.check_group.buttons():
            if check_button.isChecked():
                selected.append(self.check_group.id(check_button))
        return selected

    def setSelected(self, selected: list):
        """Method to set some checkboxes as checked.

        Args:
            selected: List of checkboxes to check.

        Returns:
            None
        """
        for check_button in self.check_group.buttons():
            if self.check_group.id(check_button) in selected:
                check_button.setChecked(True)
            else:
                check_button.setChecked(False)

    def boundingRect(self) -> QRectF:
        """Method required by Qt."""
        return QRectF()

    def paint(self, painter, option, widget=None):
        """Method required by Qt."""
        pass
