import yaml

from sleap.gui import formbuilder


def test_formbuilder(qtbot):
    form_yaml = """
- name: method
  label: Method
  type: stacked
  default: two
  options: one,two,three

  one:
    - name: per_video
      label: Samples Per Video
      type: int
      default: 20
      range: 1,3000
    - name: sampling_method
      label: Sampling method
      type: list
      options: random,stride
      default: stride

  two:
    - name: node
      label: Node
      type: list
    - name: foo
      label: Avogadro
      type: sci
      default: 6.022e23

  three:
    - name: node
      label: Node
      type: list
"""

    items_to_create = yaml.load(form_yaml, Loader=yaml.SafeLoader)

    field_options_lists = dict(node=("first option", "second option"))

    layout = formbuilder.FormBuilderLayout(
        items_to_create, field_options_lists=field_options_lists
    )

    form_data = layout.get_form_data()

    assert "node" in form_data
    assert form_data["node"] == "first option"

    layout.set_field_options("node", ("new option", "another new option"))

    form_data = layout.get_form_data()
    assert form_data["node"] == "new option"


def test_optional_spin_widget(qtbot):
    widget = formbuilder.OptionalSpinWidget()

    widget.setValue(3)
    assert widget.value() == 3

    widget.check_widget.setChecked(True)
    assert widget.value() is None

    widget.check_widget.setChecked(False)
    assert widget.value() == 3

    widget.setValue("none")
    assert widget.value() is None


def test_auto_double_widget(qtbot):
    widget = formbuilder.OptionalSpinWidget(type="double", none_string="auto")

    widget.setValue(3.2)
    assert widget.value() == 3.2

    widget.check_widget.setChecked(True)
    assert widget.value() is None

    widget.check_widget.setChecked(False)
    assert widget.value() == 3.2

    widget.setValue("auto")
    assert widget.value() is None
