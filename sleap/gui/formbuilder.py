"""Custom QFormLayout which populates itself from list of form fields."""

class FormBuilderLayout(QtWidgets.QFormLayout):
    """
    Custom QFormLayout which populates itself from list of form fields.

    Args:
        items_to_create: list which gets passed to get_form_data()
                         (see there for details about format)
    """

    def __init__(self, items_to_create, *args, **kwargs):
        super(FormBuilderLayout, self).__init__(*args, **kwargs)

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
        else:
            val = None
        return val

    def build_form(self, items_to_create):
        """Add widgets to form layout for each item in items_to_create.
        
        Args:
            items_to_create: list of dicts with fields
              * name: used as key when we return form data as dict
              * label: string to show in form
              * type: supports double, int, bool, list
              * default: default value for form field
              * [options]: comma separated list of options, used for list type
        Returns:
            None.
        """
        for item in items_to_create:
            field = None
            if item["type"] == "double":
                field = QtWidgets.QDoubleSpinBox()
                field.setValue(item["default"])
            elif item["type"] == "int":
                field = QtWidgets.QSpinBox()
                # temp fix to ensure default within range
                if item["default"] > 99:
                    field.setRange(0, item["default"]*10)
                field.setValue(item["default"])
            elif item["type"] == "bool":
                field = QtWidgets.QCheckBox()
                field.setChecked(item["default"])
            elif item["type"] == "list":
                field = QtWidgets.QComboBox()
                for opt in item["options"].split(","):
                    field.addItem(opt)
                # set default
            else:
                field = QtWidgets.QLineEdit()
                field.setText(item["default"])
    
            field.setObjectName(item["name"])
            self.addRow(item["label"] + ":", field)