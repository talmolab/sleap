"""Module for creating a form from a yaml file.

Example:
>>> widget = YamlFormWidget(yaml_file="example.yaml")
>>> widget.mainAction.connect(my_function)

my_function will get called with form data when user clicks the main button
(main button has type "button" and default "main action")

"""

import yaml

from PySide2 import QtWidgets, QtCore

class YamlFormWidget(QtWidgets.QGroupBox):
    """
    Custom QWidget which creates form based on yaml file.

    Args:
        yaml_file: filename
    """

    mainAction = QtCore.Signal(dict)

    def __init__(self, yaml_file, *args, **kwargs):
        super(YamlFormWidget, self).__init__(*args, **kwargs)

        with open(yaml_file, 'r') as form_yaml:
            items_to_create = yaml.load(form_yaml, Loader=yaml.SafeLoader)

        self.form_layout = FormBuilderLayout(items_to_create["main"])
        self.setLayout(self.form_layout)

        for item in items_to_create["main"]:
            if item["type"] == "button" and item.get("default", "") == "main action":
                self.buttons[item["name"]].clicked.connect(self.trigger_main_action)

    @property
    def buttons(self):
        """Returns a list of buttons in form (so we can connect to handlers)."""
        return self.form_layout.buttons

    def get_form_data(self):
        """Returns dict of form data."""
        return self.form_layout.get_form_data()

    def trigger_main_action(self):
        """Emit mainAction signal with form data."""
        self.mainAction.emit(self.get_form_data())

class FormBuilderLayout(QtWidgets.QFormLayout):
    """
    Custom QFormLayout which populates itself from list of form fields.

    Args:
        items_to_create: list which gets passed to get_form_data()
                         (see there for details about format)
    """

    valueChanged = QtCore.Signal()

    def __init__(self, items_to_create, *args, **kwargs):
        super(FormBuilderLayout, self).__init__(*args, **kwargs)

        self.buttons = dict()
        self.build_form(items_to_create)

    def get_form_data(self) -> dict:
        """Gets all user-editable data from the widgets in the form layout.

        Returns:
            Dict with key:value for each user-editable widget in layout
        """
        widgets = [self.itemAt(i).widget() for i in range(self.count())]
        data = {w.objectName(): self.get_widget_value(w)
                for w in widgets
                if len(w.objectName())
                   and type(w) not in (QtWidgets.QLabel, QtWidgets.QPushButton)}
        stacked_data = [w.get_data() for w in widgets if type(w) == StackBuilderWidget]
        for stack in stacked_data:
            data.update(stack)
        return data

    @staticmethod
    def get_widget_value(widget):
        """Get value of form field (using whichever method appropriate for widget).

        Args:
            widget: subclass of QtWidget
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
            print(widget)
            val = None
        return val

    def build_form(self, items_to_create):
        """Add widgets to form layout for each item in items_to_create.

        Args:
            items_to_create: list of dicts with fields
              * name: used as key when we return form data as dict
              * label: string to show in form
              * type: supports double, int, bool, list, button, stack
              * default: default value for form field
              * [options]: comma separated list of options, used for list or stack
              * for stack, array of dicts w/ form data for each stack page

        Note: a "stack" has a dropdown menu that determines which stack page to show

        Returns:
            None.
        """
        for item in items_to_create:
            field = None

            # double: show spinbox (number w/ up/down controls)
            if item["type"] == "double":
                field = QtWidgets.QDoubleSpinBox()
                field.setValue(item["default"])

            # int: show spinbox (number w/ up/down controls)
            elif item["type"] == "int":
                field = QtWidgets.QSpinBox()
                # temp fix to ensure default within range
                if item["default"] > 99:
                    field.setRange(0, item["default"]*10)
                field.setValue(item["default"])

            # bool: show checkbox
            elif item["type"] == "bool":
                field = QtWidgets.QCheckBox()
                field.setChecked(item["default"])

            # list: show drop-down menu
            elif item["type"] == "list":
                field = QtWidgets.QComboBox()
                for opt in item["options"].split(","):
                    field.addItem(opt)
                # set default
                if item["default"] in item["options"].split(","):
                    idx = item["options"].split(",").index(item["default"])
                    field.setCurrentIndex(idx)

            # button
            elif item["type"] == "button":
                field = QtWidgets.QPushButton(item["label"])
                self.buttons[item["name"]] = field

            # stacked: show menu and form panel corresponding to menu selection
            elif item["type"] == "stacked":
                field = StackBuilderWidget(item)

            # If we don't recognize the type, just show a text box
            else:
                field = QtWidgets.QLineEdit()
                field.setText(item.get("default", ""))
                if item["type"].split("_")[0] == "file":
                    field.setDisabled(True)

            # Add field (and label if appropriate) to form layout
            field.setObjectName(item["name"])
            if item["type"] in ("button", "stacked"):
                self.addRow(field)
            else:
                self.addRow(item["label"] + ":", field)

            # file_[open|dir]: show button to select file/directory
            if item["type"].split("_")[0] == "file":
                file_button = QtWidgets.QPushButton("Select "+item["label"])

                if item["type"].split("_")[-1] == "open":
                    # Define function for button to trigger
                    def select_file(*args, x=field):
                        filter = item.get("filter", "Any File (*.*)")
                        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, directory=None, caption="Open File", filter=filter)
                        if len(filename): x.setText(filename)
                        self.valueChanged.emit()

                elif item["type"].split("_")[-1] == "dir":
                    # Define function for button to trigger
                    def select_file(*args, x=field):
                        filename = QtWidgets.QFileDialog.getExistingDirectory(None, directory=None, caption="Open File")
                        if len(filename): x.setText(filename)
                        self.valueChanged.emit()

                else:
                    select_file = lambda: print(f"no action set for type {item['type']}")

                file_button.clicked.connect(select_file)
                self.addRow("", file_button)

class StackBuilderWidget(QtWidgets.QWidget):
    def __init__(self, stack_data, *args, **kwargs):
        super(StackBuilderWidget, self).__init__(*args, **kwargs)

        multi_layout = QtWidgets.QFormLayout()
        self.combo_box = QtWidgets.QComboBox()
        self.stacked_widget = QtWidgets.QStackedWidget()

        self.combo_box.activated.connect(lambda x: self.stacked_widget.setCurrentIndex(x))

        self.page_layouts = dict()

        for page in stack_data["options"].split(","):

            # add page
            self.page_layouts[page] = FormBuilderLayout(stack_data[page])

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

        default_page = stack_data["options"].split(",").index(stack_data["default"])
        self.combo_box.setCurrentIndex(default_page)
        self.stacked_widget.setCurrentIndex(default_page)

        self.setLayout(multi_layout)

    def value(self):
        return self.combo_box.currentText()

    def get_data(self):
        return self.page_layouts[self.value()].get_form_data()
