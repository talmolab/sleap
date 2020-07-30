import pytest

from sleap.gui.state import GuiState


def test_gui_state():
    state = GuiState()

    # use a global var to count how many times callback is called
    times_x_changed = 0

    def count_change_callback():
        nonlocal times_x_changed
        times_x_changed += 1

    # make sure that value can be passed to callback
    y = None

    def set_y_from_val_param_callback(x):
        nonlocal y
        y = x

    state.connect("x", count_change_callback)
    state.connect("x", set_y_from_val_param_callback)

    assert times_x_changed == 0

    # set initial value (should trigger callback)
    state["x"] = 2
    assert times_x_changed == 1
    assert y == state["x"]

    # setting to same value (no change) should not trigger callback
    state["x"] = 2
    assert times_x_changed == 1
    assert y == state["x"]

    # change value (should trigger callback)
    state["x"] = 3
    assert times_x_changed == 2
    assert y == state["x"]

    # test incrementing value
    state.increment("x")
    assert times_x_changed == 3
    assert state["x"] == 4

    # test incrementing value with modulus
    state.increment("x", mod=3)
    assert times_x_changed == 4
    assert state["x"] == 2

    # test emitting callbacks without changing value
    state.emit("x")
    assert times_x_changed == 5


def test_gui_state_bool():
    state = GuiState()

    assert "x" not in state

    state.toggle("x")
    assert state["x"] == True

    state.toggle("x")
    assert state["x"] == False


def test_gui_state_delete():
    state = GuiState()

    assert "x" not in state
    state.set("x", 5)

    assert "x" in state

    del state["x"]
    assert "x" not in state


def test_gui_state_get_default():
    state = GuiState()

    assert "x" not in state
    assert state.get("x", "default value to check") == "default value to check"


def test_gui_state_inc_default():
    state = GuiState()

    state.increment("x")
    assert state["x"] == 0

    state.increment("y", default=5)
    assert state["y"] == 5


def test_gui_state_list():
    value_list = ["foo", "bar", "zip"]
    state = GuiState()

    state.increment_in_list("x", value_list)
    assert state["x"] == value_list[0]

    state.increment_in_list("x", value_list)
    assert state["x"] == value_list[1]

    state.increment_in_list("x", value_list)
    assert state["x"] == value_list[2]

    state.increment_in_list("x", value_list)
    assert state["x"] == value_list[0]

    # check reverse
    state.increment_in_list("x", value_list, reverse=True)
    assert state["x"] == value_list[-1]

    state.increment_in_list("x", value_list, reverse=True)
    assert state["x"] == value_list[-2]

    # check reverse on new key
    state.increment_in_list("y", value_list, reverse=True)
    assert state["y"] == value_list[-1]


def test_gui_state_callbacks():
    def f():
        raise RuntimeError("this shouldn't stop test...")

    def g(x):
        pass

    state = GuiState()
    state.connect("x", [f, g])

    # make sure we can't add callback
    with pytest.raises(ValueError):
        state.connect("y", [f, 5])

    state["x"] = "value to trigger callbacks"
