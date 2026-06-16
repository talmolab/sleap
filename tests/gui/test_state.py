import pytest

from sleap.gui.state import (
    GuiState,
    SHOW_NONVISIBLE_OVERRIDE_KEY,
    QC_MODE_MANUAL,
    QC_MODE_SELECTED_ONLY,
    QC_MODE_ALL_VISIBLE,
    QC_MODE_ALL_PLUS_SELECTED,
    instance_shows_non_visible,
    compute_qc_visibility,
)


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

    # Test incrementing value with modulus of 1
    state.increment("x", mod=1)
    assert times_x_changed == 5
    assert state["x"] == 0

    # test emitting callbacks without changing value
    state.emit("x")
    assert times_x_changed == 6


def test_gui_state_bool():
    state = GuiState()

    assert "x" not in state

    state.toggle("x")
    assert state["x"]

    state.toggle("x")
    assert not state["x"]


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


# -- Per-instance "show non-visible nodes" override (#2782 shared model) ---------


def test_instance_shows_non_visible_default():
    """With no override, the global default is returned verbatim."""
    state = GuiState()
    inst = object()
    assert instance_shows_non_visible(state, inst, True) is True
    assert instance_shows_non_visible(state, inst, False) is False


def test_instance_shows_non_visible_override_wins():
    """An explicit per-instance override beats the global default."""
    state = GuiState()
    inst = object()
    other = object()

    state[SHOW_NONVISIBLE_OVERRIDE_KEY] = {id(inst): False}
    assert instance_shows_non_visible(state, inst, True) is False
    # A second un-overridden instance still falls back to the global default.
    assert instance_shows_non_visible(state, other, True) is True

    state[SHOW_NONVISIBLE_OVERRIDE_KEY] = {id(inst): True}
    assert instance_shows_non_visible(state, inst, False) is True
    assert instance_shows_non_visible(state, other, False) is False


def test_instance_shows_non_visible_none_state():
    """A ``None`` state returns the global default (no override possible)."""
    assert instance_shows_non_visible(None, object(), True) is True
    assert instance_shows_non_visible(None, object(), False) is False


# -- QC display-mode -> per-instance flags (#2783 pure helper) -------------------


def test_compute_qc_visibility_manual_returns_empty():
    """Manual mode is the no-op sentinel: an empty dict."""
    instances = [object(), object()]
    assert compute_qc_visibility(QC_MODE_MANUAL, instances[0], instances, True) == {}


def test_compute_qc_visibility_selected_only():
    """selected_only: only the selected instance is visible (with hidden pts)."""
    instances = [object(), object(), object()]
    flags = compute_qc_visibility(QC_MODE_SELECTED_ONLY, instances[1], instances, True)
    assert flags[id(instances[1])] == (True, True)
    assert flags[id(instances[0])] == (False, False)
    assert flags[id(instances[2])] == (False, False)


def test_compute_qc_visibility_all_visible_only():
    """all_visible_only: every instance visible, occluded points hidden."""
    instances = [object(), object(), object()]
    flags = compute_qc_visibility(QC_MODE_ALL_VISIBLE, instances[0], instances, True)
    for inst in instances:
        assert flags[id(inst)] == (True, False)


def test_compute_qc_visibility_all_plus_selected_invisible():
    """all_plus_selected_invisible: all visible; only selected shows hidden pts."""
    instances = [object(), object(), object()]
    flags = compute_qc_visibility(
        QC_MODE_ALL_PLUS_SELECTED, instances[2], instances, True
    )
    assert flags[id(instances[2])] == (True, True)
    assert flags[id(instances[0])] == (True, False)
    assert flags[id(instances[1])] == (True, False)


def test_compute_qc_visibility_selected_none_or_foreign():
    """No/foreign selection falls back to the FIRST instance (never blank)."""
    a, b = object(), object()
    instances = [a, b]

    # selected=None -> selected_only keeps only the FIRST instance (with its
    # hidden points), so the mode is visibly doing something and never blanks.
    flags = compute_qc_visibility(QC_MODE_SELECTED_ONLY, None, instances, True)
    assert flags[id(a)] == (True, True)
    assert flags[id(b)] == (False, False)

    # selected=None -> all_plus_selected shows all, with the FIRST instance's
    # hidden points.
    flags = compute_qc_visibility(QC_MODE_ALL_PLUS_SELECTED, None, instances, True)
    assert flags[id(a)] == (True, True)
    assert flags[id(b)] == (True, False)

    # A selected instance NOT in the list behaves identically to None.
    foreign = object()
    flags = compute_qc_visibility(QC_MODE_SELECTED_ONLY, foreign, instances, True)
    assert flags[id(a)] == (True, True)
    assert flags[id(b)] == (False, False)
    flags = compute_qc_visibility(QC_MODE_ALL_PLUS_SELECTED, foreign, instances, True)
    assert flags[id(a)] == (True, True)
    assert flags[id(b)] == (True, False)
