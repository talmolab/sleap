from typing import List

from PySide2 import QtWidgets

from sleap.gui.shortcuts import Shortcuts


class ShortcutDialog(QtWidgets.QDialog):
    """
    Dialog window for reviewing and modifying the keyboard shortcuts.
    """

    _column_len = 13

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

        super(ShortcutDialog, self).accept()

    def load_shortcuts(self):
        """Loads shortcuts object."""
        self.shortcuts = Shortcuts()

    def make_form(self):
        """Creates the form with fields for all shortcuts."""
        self.key_widgets = dict()  # dict to store QKeySequenceEdit widgets

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.make_shortcuts_widget())
        layout.addWidget(
            QtWidgets.QLabel(
                "Any changes to keyboard shortcuts will not take effect "
                "until you quit and re-open the application."
            )
        )
        layout.addWidget(self.make_buttons_widget())
        self.setLayout(layout)

    def make_buttons_widget(self) -> QtWidgets.QDialogButtonBox:
        """Makes the form buttons."""
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        return buttons

    def make_shortcuts_widget(self) -> QtWidgets.QWidget:
        """Makes the widget will fields for all shortcuts."""
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
        """Makes a single column of shortcut fields.

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
