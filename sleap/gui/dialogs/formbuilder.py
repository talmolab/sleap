"""
Widgets and dialogues for YAML-based forms.

Most of the time you can use :py:class:`YamlFormWidget`. If you want to show
the widget as a model dialog, :py:class:`FormBuilderModalDialog` makes this
a little more convenient (it provides methods for adding a message for the
dialog and for getting the results when the dialog is closed).

Example of form widget: ::

   >>> widget = YamlFormWidget(yaml_file="example.yaml")
   >>> widget.mainAction.connect(my_function)

my_function will get called with form data when user clicks the main button
(main button has type "button" and default "main action")

Example of modal dialog: ::

   >>> results = FormBuilderModalDialog(form_name="example").get_results()

The results will be empty dictionary if the user hit "cancel", otherwise it
will contain all data from form (dict keys matching names of fields).

The logic which creates each form field based on the data from the YAML file
is :py:method:`FormBuilderLayout.add_item()`. Look there if you want to know
what's supported in the YAML file, what exactly each field type does, or if you
want to add a new type of supported form field.
"""

import yaml

from typing import Any, Dict, List, Optional, Text

from PySide2 import QtWidgets, QtCore

from sleap.gui.dialogs.filedialog import FileDialog
from sleap.util import get_package_file


class YamlFormWidget(QtWidgets.QGroupBox):
    """
    Widget which shows form created from a YAML file.

    Typically you'll want to save the YAML in `sleap/config/` and use the
    :py:meth:`from_name` method to make the form (e.g., if your form data is in
    `sleap/config/foo.yaml`, then you can create form like so: ::

       >>> widget = YamlFormWidget.from_name("foo")

    Args:
        yaml_file: filename of YAML file to load.
        which_form (optional): key to form in YAML file, default is "main".
            this allows a single YAML file to contain data for multiple forms.
    """

    mainAction = QtCore.Signal(dict)
    valueChanged = QtCore.Signal()

    def __init__(
        self,
        yaml_file: Text,
        which_form: Text = "main",
        field_options_lists: Optional[Dict[Text, list]] = None,
        *args,
        **kwargs,
    ):
        super(YamlFormWidget, self).__init__(*args, **kwargs)

        with open(yaml_file, "r") as form_yaml:
            items_to_create = yaml.load(form_yaml, Loader=yaml.SafeLoader)

        self.which_form = which_form
        self.form_layout = FormBuilderLayout(
            items_to_create[self.which_form], field_options_lists=field_options_lists
        )

        self.setLayout(self.form_layout)

        if items_to_create[self.which_form]:
            for item in items_to_create[self.which_form]:
                if (
                    item["type"] == "button"
                    and item.get("default", "") == "main action"
                ):
                    self.buttons[item["name"]].clicked.connect(self.trigger_main_action)

        self.form_layout.valueChanged.connect(self.valueChanged)

    def __getitem__(self, key):
        """Return value for specified form field."""
        return FormBuilderLayout.get_widget_value(self.fields[key])

    def __setitem__(self, key, val):
        """Set value for specified form field."""
        FormBuilderLayout.set_widget_value(self.fields[key], val)

    @classmethod
    def from_name(cls, form_name: Text, *args, **kwargs) -> "YamlFormWidget":
        """
        Instantiate class from the short name of form (e.g., "suggestions").

        Short name is converted to path to YAML file in `sleap/config/`,
        and then class is instantiated using this path.

        Args:
            form_name: Short name of form, corresponds to name of YAML file.
            args: Positional args passed to class initializer.
            kwargs: Named args passed to class initializer.

        Returns:
            Instance of `YamlFormWidget` class.
        """
        yaml_path = get_package_file(f"sleap/config/{form_name}.yaml")
        return cls(yaml_path, *args, **kwargs)

    @property
    def buttons(self):
        """Returns a list of buttons in form (so we can connect to handlers)."""
        return self.form_layout.buttons

    @property
    def fields(self):
        """Return a dict of {name: widget} fields in form."""
        return self.form_layout.fields

    def set_form_data(self, data):
        """Set data for form from dict."""
        self.form_layout.set_form_data(data)

    def get_form_data(self):
        """Returns dict of form data."""
        return self.form_layout.get_form_data()

    def set_field_options(self, field_name: Text, options_list: List[Text], **kwargs):
        """Sets option list for specified field."""
        self.form_layout.set_field_options(field_name, options_list, **kwargs)

    def set_field_enabled(self, field_name: Text, is_enabled: bool):
        self.form_layout.set_field_enabled(field_name, is_enabled)

    def set_enabled(self, enabled: bool):
        self.form_layout.setEnabled(enabled)

    def trigger_main_action(self):
        """Emit mainAction signal with form data."""
        self.mainAction.emit(self.get_form_data())


class FormBuilderModalDialog(QtWidgets.QDialog):
    """
    Blocking modal dialog to use with :py:class:`YamlFormWidget` widget.

    You can either initialize with a :py:class:`YamlFormWidget` widget, or
    provide the name of the YAML form (i.e., the string you'd pass to
    :py:meth:`YamlFormWidget.from_name()`).
    """

    def __init__(
        self,
        form_widget: Optional[YamlFormWidget] = None,
        form_name: Optional[Text] = None,
        *args,
        **kwargs,
    ):
        super(FormBuilderModalDialog, self).__init__()

        if not form_widget and form_name:
            form_widget = YamlFormWidget.from_name(form_name)

        if not form_widget:
            raise ValueError(
                "FormBuilderModalDialog must have either form widget or name."
            )

        self._results = None
        self.form_widget = form_widget
        self.message_fields = []

        # Layout for buttons
        buttons = QtWidgets.QDialogButtonBox()
        self.cancel_button = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.run_button = buttons.addButton(QtWidgets.QDialogButtonBox.Ok)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(buttons, alignment=QtCore.Qt.AlignTop)

        buttons_layout_widget = QtWidgets.QWidget()
        buttons_layout_widget.setLayout(buttons_layout)

        # Layout for entire dialog
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.form_widget)
        layout.addWidget(buttons_layout_widget)

        self.setLayout(layout)

        # Connect actions for buttons
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)

    def add_message(self, message: Text):
        """Adds text message between form fields and buttons."""
        field = QtWidgets.QLabel(message)
        field.setWordWrap(True)

        self.message_fields.append(field)

        self.layout().insertWidget(1, field)

    def set_message(self, message: Text):
        """Adds/replaces text message between form fields and buttons."""
        if self.message_fields:
            self.message_fields[0].setText(message)
        else:
            self.add_message(message)

    def on_accept(self):
        self._results = self.form_widget.get_form_data()
        self.accept()

    def get_results(self) -> Optional[Dict[Text, Any]]:
        """Shows dialog, blocks till submitted, returns dict of form data."""
        self._results = None
        self.exec_()
        return self._results


class FormBuilderLayout(QtWidgets.QFormLayout):
    """
    Custom `QFormLayout` which populates itself from list of form fields.

    Args:
        items_to_create: list of form items, each item in list is a dictionary
            with information about item. see :py::meth:`build_form` for details
            about the keys/values. typically this list comes straight from a
            YAML file.
        field_options_lists: allows you to set list of options to use for "list"
            fields (i.e., :py:class:`FieldComboWidget` or
            :py:class:`TextOrListWidget` widgets)
            when creating the form programmatically, instead of having options
            as "options" key of dictionary (which typically would be hard-coded
            in YAML).
    """

    valueChanged = QtCore.Signal()

    def __init__(
        self,
        items_to_create: List[Dict[Text, Any]],
        field_options_lists: Optional[Dict[Text, List[Text]]] = None,
        *args,
        **kwargs,
    ):
        super(FormBuilderLayout, self).__init__(*args, **kwargs)

        self.form_text_widget = None

        self.buttons = dict()
        self.fields = dict()
        self.field_options_lists = field_options_lists or dict()
        self.build_form(items_to_create)

    @property
    def stacked(self) -> list:
        """List of all "stack" widgets in form."""
        return [w for w in self.fields.values() if type(w) == StackBuilderWidget]

    def setEnabled(self, enabled: bool):
        for field in self.fields.values():
            if hasattr(field, "set_fields_enabled"):
                field.set_fields_enabled(enabled)
            if hasattr(field, "setEnabled"):
                field.setEnabled(enabled)

    def get_form_data(self) -> dict:
        """Gets all user-editable data from the widgets in the form layout.

        Returns:
            Dict with key:value for each user-editable widget in layout
        """
        widgets = self.fields.values()
        data = {
            w.objectName(): self.get_widget_value(w)
            for w in widgets
            if len(w.objectName())
            and type(w) not in (QtWidgets.QLabel, QtWidgets.QPushButton)
        }

        stacked_data = [w.get_data() for w in widgets if type(w) == StackBuilderWidget]
        for stack in stacked_data:
            data.update(stack)
        return data

    def set_form_data(self, data: Dict[Text, Any]):
        """Set specified user-editable data.

        Args:
            data: dictionary of data, key should match field name
        """
        widgets = self.fields
        for name, val in data.items():
            # print(f"Attempting to set {name} to {val}")
            if name in widgets:
                # print(f"set {name} to {val} ({type(val)})")
                self.set_widget_value(widgets[name], val)
            else:
                # print(f"cannot set {name} for {widgets}")
                pass

        for stacked_widget in self.stacked:
            stacked_widget.set_form_data(data)

    def set_field_options(self, field_name: Text, options_list: List[Text]):
        """Sets custom list of options for specified field."""
        self.field_options_lists[field_name] = options_list

        for subform in self.stacked:
            subform.set_field_options(field_name, options_list)

        self.update_field_options()

    def update_field_options(self):
        """Updates options list for every field with custom list."""
        for field_name, field in self.fields.items():
            if field_name in self.field_options_lists:
                field.set_options(self.field_options_lists[field_name])

    def set_field_enabled(self, field_name: Text, is_enabled: bool):
        """Sets whether the field is enabled/disabled."""
        fields = self.find_field(field_name)

        for field in fields:
            field.setEnabled(is_enabled)

    def find_field(self, field_name: Text):
        """
        Finds form fields by name.

        Args:
            field_name: Name of field to find.

        Returns:
            List of field widgets, including any in active stacked widget.
        """
        widgets = self.fields.values()
        found = [
            w
            for w in widgets
            if w.objectName() == field_name
            and type(w) not in (QtWidgets.QLabel, QtWidgets.QPushButton)
        ]
        stacks = [w for w in widgets if type(w) == StackBuilderWidget]
        for stack in stacks:
            found.extend(stack.find_field(field_name))
        return found

    @staticmethod
    def set_widget_value(widget: QtWidgets.QWidget, val):
        """Set value for specific widget."""

        if hasattr(widget, "isChecked"):
            widget.setChecked(val)
        elif hasattr(widget, "setValue"):
            widget.setValue(val)
        elif hasattr(widget, "currentText"):
            widget.setCurrentText(str(val))
        elif hasattr(widget, "text"):
            widget.setText(str(val))
        else:
            print(f"don't know how to set value for {widget}")
        # for macOS we need to call repaint (bug in Qt?)
        widget.repaint()

    @staticmethod
    def get_widget_value(widget: QtWidgets.QWidget) -> Any:
        """Returns value of form field.

        This determines the method appropriate for the type of widget.

        Args:
            widget: The widget for which to return value.
        Returns:
            value (can be bool, numeric, string, or None)
        """
        if hasattr(widget, "isChecked"):
            val = widget.isChecked()
        elif hasattr(widget, "value"):
            val = widget.value()
        elif hasattr(widget, "currentText"):
            val = widget.currentText()
        elif hasattr(widget, "text"):
            val = widget.text()
        elif hasattr(widget, "currentIndex"):
            val = widget.currentIndex()
        else:
            val = None
        if widget.property("field_data_type") == "sci":
            val = float(val)
        elif widget.property("field_data_type") == "int":
            val = int(val)
        elif widget.property("field_data_type").startswith("file_"):
            val = None if val == "None" else val
        return val

    def build_form(self, items_to_create: List[Dict[Text, Any]]):
        """Adds widgets to form layout for each item in items_to_create.

        Args:
            items_to_create: list of dictionaries with keys

              * name: used as key when we return form data as dict
              * label: string to show in form
              * type: supports double, int, bool, list, button, stack
              * default: default value for form field
              * [options]: comma separated list of options,
                used for list or stack field-types
              * for stack, array of dicts w/ form data for each stack page

        A "stack" has a dropdown menu that determines which stack page to show.

        Returns:
            None.
        """
        if not items_to_create:
            return
        for item in items_to_create:
            self.add_item(item)

    def add_item(self, item: Dict[Text, Any]):
        if item["type"] == "text":
            field = QtWidgets.QLabel(item["text"])
            field.setWordWrap(True)

            # We don't need to keep track of this text-only field so we'll
            # add it to the form and skip the other things we usually do.
            self.addRow(field)
            return

        # double: show spinbox (number w/ up/down controls)
        elif item["type"] == "double":
            field = QtWidgets.QDoubleSpinBox()

            min, max = -1000, 1000
            if "range" in item.keys():
                min, max = list(map(float, item["range"].split(",")))
            field.setRange(min, max)
            field.setSingleStep(0.25)

            field.setValue(item["default"])

            field.valueChanged.connect(lambda: self.valueChanged.emit())

        # int: show spinbox (number w/ up/down controls)
        elif item["type"] == "int":
            field = QtWidgets.QSpinBox()
            if "range" in item.keys():
                min, max = list(map(int, item["range"].split(",")))
                field.setRange(min, max)
            elif item["default"] > 100:
                min, max = 0, item["default"] * 10
                field.setRange(min, max)
            field.setValue(item["default"])
            field.valueChanged.connect(lambda: self.valueChanged.emit())

        elif item["type"] in ("optional_int", "optional_double", "auto_int"):
            spin_type = item["type"].split("_")[-1]
            none_string = "auto" if item["type"].startswith("auto") else "none"
            none_label = item.get("none_label", None)
            field = OptionalSpinWidget(
                type=spin_type, none_string=none_string, none_label=none_label
            )
            if "range" in item.keys():
                min, max = list(map(int, item["range"].split(",")))
                field.setRange(min, max)
            elif item["default"] is not None and item["default"] > 100:
                min, max = 0, item["default"] * 10
                field.setRange(min, max)

            field.setValue(item["default"])
            if item.get("default_disabled", False):
                field.setToNone()

            field.valueChanged.connect(lambda: self.valueChanged.emit())

        # bool: show checkbox
        elif item["type"] == "bool":
            field = QtWidgets.QCheckBox()
            field.setChecked(item["default"])
            field.stateChanged.connect(lambda: self.valueChanged.emit())

        # list: show drop-down menu
        elif item["type"] in ("list", "optional_list"):
            type_options = item.get("type-options", "")

            result_as_optional_idx = False
            add_blank_option = False
            if type_options == "optional_index":
                result_as_optional_idx = True
                add_blank_option = True
            if item["type"] == "optional_list":
                add_blank_option = True

            field = TextOrListWidget(
                result_as_idx=result_as_optional_idx,
                add_blank_option=add_blank_option,
            )

            if item["name"] in self.field_options_lists:
                field.set_options(self.field_options_lists[item["name"]])
            elif "options" in item:
                field.set_options(
                    item["options"].split(","), select_item=item.get("default", "")
                )

            field.valueChanged.connect(lambda: self.valueChanged.emit())

        # button
        elif item["type"] == "button":
            field = QtWidgets.QPushButton(item["label"])
            self.buttons[item["name"]] = field

        # string
        elif item["type"] in ("string", "optional_string"):
            field = QtWidgets.QLineEdit()
            val = item.get("default", "")
            val = "" if val is None else val
            field.setText(str(val))

        elif item["type"] == "string_list":
            field = StringListWidget()
            val = item.get("default", "")
            field.setValue(val)

        # stacked: show menu and form panel corresponding to menu selection
        elif item["type"] == "stacked":
            field = StackBuilderWidget(
                item, field_options_lists=self.field_options_lists
            )
            field.valueChanged.connect(lambda: self.valueChanged.emit())

        # If we don't recognize the type, just show a text box
        else:
            field = QtWidgets.QLineEdit()
            field.setText(str(item.get("default", "")))
            if item["type"].split("_")[0] == "file":
                field.setDisabled(True)

        # Store name and type on widget
        field.setObjectName(item["name"])
        field.setProperty("field_data_type", item.get("dtype", item["type"]))

        # Set tooltip for field
        if "help" in item:
            field.setToolTip(item["help"])

        # Store widget by name
        self.fields[item["name"]] = field

        # Add field (and label if appropriate) to form layout
        if item["type"] in ("stacked"):
            self.addRow(field)
        elif item["type"] in ("button"):
            self.addRow("", field)
        else:
            self.addRow(item["label"] + ":", field)

        # file_[open|dir]: show button to select file/directory
        if item["type"].split("_")[0] == "file":
            self.addRow("", self._make_file_button(item, field))

    def _make_file_button(
        self, item: Dict, field: QtWidgets.QWidget
    ) -> QtWidgets.QPushButton:
        """Creates the button for a file_* field-type."""
        file_button = QtWidgets.QPushButton("Select " + item["label"])

        if item["type"].split("_")[-1] == "open":
            # Define function for button to trigger
            def select_file(*args, x=field):
                filter = item.get("filter", "Any File (*.*)")
                filename, _ = FileDialog.open(
                    None, directory=None, caption="Open File", filter=filter
                )
                if len(filename):
                    x.setText(filename)
                self.valueChanged.emit()

        elif item["type"].split("_")[-1] == "dir":
            # Define function for button to trigger
            def select_file(*args, x=field):
                filename = FileDialog.openDir(None, directory=None, caption="Open File")
                if len(filename):
                    x.setText(filename)
                self.valueChanged.emit()

        else:
            select_file = lambda: print(f"no action set for type {item['type']}")

        file_button.clicked.connect(select_file)
        return file_button


class StackBuilderWidget(QtWidgets.QWidget):
    """
    Widget that shows different subforms depending on menu selection.

    Arguments:
        stack_data: Dictionary for field from `items_to_create`.
            The "options" key will give the list of options to show in
            menu. Each of the "options" will also be the key of a dictionary
            within stack_data that has the same structure as the dictionary
            passed to :py:meth:`FormBuilderLayout.build_form()`.
        field_options_list: passed to stacked forms, see
            :py:class:`FormBuilderLayout` for details.
    """

    valueChanged = QtCore.Signal()

    def __init__(self, stack_data, field_options_lists=None, *args, **kwargs):
        super(StackBuilderWidget, self).__init__(*args, **kwargs)

        self.option_list = stack_data["options"].split(",")

        self.field_options_lists = field_options_lists or None

        multi_layout = QtWidgets.QFormLayout()
        multi_layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.combo_box = QtWidgets.QComboBox()
        self.stacked_widget = ResizingStackedWidget()

        self.combo_box.activated.connect(self.switch_to_idx)

        self.page_layouts = dict()

        for page in self.option_list:
            # add page
            self.page_layouts[page] = FormBuilderLayout(
                stack_data[page], field_options_lists=self.field_options_lists
            )

            self.page_layouts[page].valueChanged.connect(self.valueChanged)

            page_widget = QtWidgets.QGroupBox()
            page_widget.setLayout(self.page_layouts[page])

            self.stacked_widget.addWidget(page_widget)

            # add option to menu
            self.combo_box.addItem(page)

        if len(stack_data.get("label", "")):
            combo_label = f"{stack_data['label']}:"
        else:
            combo_label = ""

        multi_layout.addRow(combo_label, self.combo_box)
        multi_layout.addRow(self.stacked_widget)

        self.setLayout(multi_layout)

        self.setValue(stack_data["default"])

    def switch_to_idx(self, idx):
        """Switch currently shown widget from stack."""
        self.stacked_widget.setCurrentIndex(idx)
        # Only show if the widget contains more than an empty layout
        if len(self.stacked_widget.currentWidget().children()) > 1:
            self.stacked_widget.show()
        else:
            self.stacked_widget.hide()
        self.valueChanged.emit()

    def value(self):
        """Returns value of menu."""
        return self.combo_box.currentText()

    def setValue(self, value):
        """Sets value of menu."""
        if value not in self.option_list:
            return
        idx = self.option_list.index(value)
        self.combo_box.setCurrentIndex(idx)
        self.switch_to_idx(idx)

    def setEnabled(self, val):
        self.combo_box.setEnabled(val)

    def set_fields_enabled(self, val):
        for subform in self.page_layouts.values():
            subform.setEnabled(val)

    def get_data(self):
        """Returns value from currently shown subform."""
        return self.page_layouts[self.value()].get_form_data()

    def find_field(self, *args, **kwargs):
        """Returns result of find_field method on currently shown subform."""
        return self.page_layouts[self.value()].find_field(*args, **kwargs)

    def set_field_options(self, *args, **kwargs):
        """Calls set_field_options for every subform."""
        for subform in self.page_layouts.values():
            subform.set_field_options(*args, **kwargs)

    def set_form_data(self, data):
        for layout in self.page_layouts.values():
            layout.set_form_data(data)


class OptionalSpinWidget(QtWidgets.QWidget):
    """
    Numeric (spin) widget with checkbox to disable.

    Arguments:
        type: can be "int" or "double".
        none_string: value to use when numeric entry is disabled; i.e., widget
            returns this as value if checkbox is set, and setting value to this
            will make the checkbox checked.
        note_label: text to show next to checkbox; defaults to none_string (in
            title case).
    """

    valueChanged = QtCore.Signal()

    def __init__(
        self,
        type="int",
        none_string="none",
        none_label: Optional[Text] = None,
        *args,
        **kwargs,
    ):
        super(OptionalSpinWidget, self).__init__(*args, **kwargs)

        self.none_string = none_string

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.spin_widget = (
            QtWidgets.QDoubleSpinBox() if type is "double" else QtWidgets.QSpinBox()
        )
        none_label = none_label if none_label is not None else self.none_string.title()
        self.check_widget = QtWidgets.QCheckBox(none_label)

        self.spin_widget.valueChanged.connect(self.updateState)
        self.check_widget.stateChanged.connect(self.updateState)

        layout.addWidget(self.spin_widget)
        layout.addWidget(self.check_widget)

        self.setLayout(layout)

    def updateState(self, valueChanged=True):
        self.spin_widget.setDisabled(self.check_widget.isChecked())
        if valueChanged:
            self.valueChanged.emit()

    def isNoneVal(self, val) -> bool:
        if val is None:
            return True
        if hasattr(val, "lower"):
            if self.none_string.lower() in ("", "none"):
                if val.lower() in ("none"):
                    return True
            elif self.none_string.lower() in ("auto"):
                if val.lower() in ("auto"):
                    return True
        return False

    @property
    def noneVal(self):
        if self.none_string.lower() in ("", "none"):
            return None
        else:
            return self.none_string

    def setToNone(self):
        self.setValue(self.noneVal)

    def value(self):
        if self.check_widget.isChecked():
            return self.noneVal
        return self.spin_widget.value()

    def setValue(self, val):
        is_none = self.isNoneVal(val)
        self.check_widget.setChecked(is_none)
        if not is_none:
            self.spin_widget.setValue(val)
        self.updateState(valueChanged=False)

    def setRange(self, min, max):
        self.spin_widget.setRange(min, max)


class FieldComboWidget(QtWidgets.QComboBox):
    """
    Custom ComboBox-style widget with method to easily add set of options.

    Arguments:
        result_as_idx: If True, then set/get for value will use idx of option
            rather than string.
        add_blank_option: If True, then blank ("") option will be added at
            beginning of list (which will return "" as val instead of idx if
            result_as_idx is True).
    """

    def __init__(
        self,
        result_as_idx: bool = False,
        add_blank_option: bool = False,
        *args,
        **kwargs,
    ):
        super(FieldComboWidget, self).__init__(*args, **kwargs)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.setMinimumContentsLength(3)
        self.result_as_idx = result_as_idx
        self.add_blank_option = add_blank_option
        self.options_list = []

    def set_options(self, options_list: List[Text], select_item: Optional[Text] = None):
        """
        Sets list of menu options.

        Args:
            options_list: List of items (strings) to show in menu.
            select_item: Item to select initially.

        Returns:
            None.
        """
        self.clear()
        self.options_list = options_list

        if self.add_blank_option:
            self.addItem("")
        for item in options_list:
            if item == "---":
                self.insertSeparator(self.count())
            else:
                self.addItem(item)
        if select_item is not None:
            self.setValue(select_item)

    def value(self):
        if self.result_as_idx:
            val = self.currentIndex()
            if self.add_blank_option:
                val -= 1
        else:
            val = self.currentText()

        return val

    def setValue(self, val):
        if type(val) == int and val < len(self.options_list) and self.result_as_idx:
            val = self.options_list[val]
        super(FieldComboWidget, self).setCurrentText(str(val))


class StringListWidget(QtWidgets.QLineEdit):
    """
    Free-form text field which converts value to/from list.

    Arguments:
        delim: the list delimiter to use; note that list values won't be
            trimmed, so "," and "item a, item b" will result in
            ["item a", " item b"].
    """

    def __init__(self, delim=" ", *args, **kwargs):
        super(StringListWidget, self).__init__(*args, **kwargs)
        self.delim = delim

    def setValue(self, val):
        # Check if we have a list (not *sequence* since this would include str)
        if isinstance(val, list):
            val = self.delim.join(val)
        self.setText(str(val))

    def getValue(self):
        return self.text().split(self.delim)


class TextOrListWidget(QtWidgets.QWidget):
    """
    Widget with free-form text field or drop-down menu.

    The "text" or "list" mode can be set using `setMode` method.

    This widget is useful when we want a drop-down menu but need to fall back
    to a text field when (e.g.) the current value isn't in list.

    Arguments:
        result_as_idx: If True, then set/get for value will use idx of option
            rather than string.
        add_blank_option: If True, then blank ("") option will be added at
            beginning of list (which will return "" as val instead of idx if
            result_as_idx is True).
    """

    valueChanged = QtCore.Signal()

    def __init__(self, result_as_idx=False, add_blank_option=False, *args, **kwargs):
        super(TextOrListWidget, self).__init__(*args, **kwargs)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.text_widget = QtWidgets.QLineEdit()
        self.list_widget = FieldComboWidget(
            result_as_idx=result_as_idx,
            add_blank_option=add_blank_option,
        )

        layout.addWidget(self.text_widget)
        layout.addWidget(self.list_widget)

        self.setLayout(layout)

        self.setMode("text")

        self.list_widget.currentIndexChanged.connect(self.emitValueChanged)
        self.text_widget.textChanged.connect(self.emitValueChanged)

    def emitValueChanged(self, *args):
        self.valueChanged.emit()

    def setMode(self, mode):
        self.mode = mode
        self.text_widget.setVisible(self.mode == "text")
        self.list_widget.setVisible(self.mode == "list")

    def value(self):
        if self.mode == "text":
            return self.text_widget.text()
        else:
            return self.list_widget.value()

    def setValue(self, val):
        self.text_widget.setText(str(val))
        self.list_widget.setValue(val)

    def set_options(self, *args, **kwargs):
        self.setMode("list")
        self.list_widget.set_options(*args, **kwargs)


class ResizingStackedWidget(QtWidgets.QStackedWidget):
    """
    QStackedWidget that updates its sizeHint and minimumSizeHint as needed.
    """

    def __init__(self, *args, **kwargs):
        super(ResizingStackedWidget, self).__init__(*args, **kwargs)

    def sizeHint(self):
        """Qt method."""
        return self.currentWidget().sizeHint()

    def minimumSizeHint(self):
        """Qt method."""
        return self.currentWidget().minimumSizeHint()
