"""
GUI for viewing/modifying keyboard shortcuts.
"""

from typing import List

from PySide2 import QtWidgets

from sleap.gui.shortcuts import Shortcuts


class ShortcutDialog(QtWidgets.QDialog):
    """
    Dialog window for reviewing and modifying the keyboard shortcuts.
    """

    _column_len = 14

    def __init__(self, *args, **kwargs):
        super(ShortcutDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Keyboard Shortcuts")
        self.load_shortcuts()
        self.make_form()

    def accept(self):
        """Triggered when form is accepted; saves the shortcuts."""
        for action, widget in self.key_widgets.items():
            self.shortcuts[action] = widget.keySequence().toString()
        self.shortcuts.save()
        self.info_msg()
        super(ShortcutDialog, self).accept()

    def info_msg(self):
        """Display information about changes."""
        msg = QtWidgets.QMessageBox()
        msg.setText(
            "Application must be restarted before changes to keyboard shortcuts take "
            "effect."
        )
        msg.exec_()

    def reset(self):
        """Reset to defaults."""
        self.shortcuts.reset_to_default()
        self.info_msg()
        super(ShortcutDialog, self).accept()

    def load_shortcuts(self):
        """Load shortcuts object."""
        self.shortcuts = Shortcuts()

    def make_form(self):
        """Creates the form with fields for all shortcuts."""
        self.key_widgets = dict()  # dict to store QKeySequenceEdit widgets

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.make_shortcuts_widget())
        layout.addWidget(self.make_buttons_widget())
        self.setLayout(layout)

    def make_buttons_widget(self) -> QtWidgets.QDialogButtonBox:
        """Make the form buttons."""
        buttons = QtWidgets.QDialogButtonBox()
        save = QtWidgets.QPushButton("Save")
        save.clicked.connect(self.accept)
        buttons.addButton(save, QtWidgets.QDialogButtonBox.AcceptRole)

        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addButton(cancel, QtWidgets.QDialogButtonBox.RejectRole)

        reset = QtWidgets.QPushButton("Reset to defaults")
        reset.clicked.connect(self.reset)
        buttons.addButton(reset, QtWidgets.QDialogButtonBox.ActionRole)

        return buttons

    def make_shortcuts_widget(self) -> QtWidgets.QWidget:
        """Make the widget will fields for all shortcuts."""
        shortcuts = self.shortcuts

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()

        # show shortcuts in columns
        for a in range(0, len(shortcuts), self._column_len):
            b = min(len(shortcuts), a + self._column_len)
            column_widget = self.make_column_widget(shortcuts[a:b])
            layout.addWidget(column_widget)
        widget.setLayout(layout)
        return widget

    def make_column_widget(self, shortcuts: List) -> QtWidgets.QWidget:
        """Make a single column of shortcut fields.

        Args:
            shortcuts: The list of shortcuts to include in this column.

        Returns:
            The widget.
        """
        column_widget = QtWidgets.QWidget()
        column_layout = QtWidgets.QFormLayout()
        for action in shortcuts:
            item = QtWidgets.QKeySequenceEdit(shortcuts[action])
            column_layout.addRow(action.title(), item)
            self.key_widgets[action] = item
        column_widget.setLayout(column_layout)
        return column_widget
