"""
Module with object for storing and accessing gui state variables.

Each project open in the GUI will have its own instance of GuiState, as will any
video player (`QtVideoPlayer` widget) which shows different images than in the
main app GUI (e.g., `QtImageDirectoryWidget` used for visualizing results
during training).

The state object makes it easier to separate code which updates state (e.g.,
sets current frame or current video) and code which updates the GUI in response
to state-change.

The state object is effectively a dictionary which allows you to bind functions
to keys so that the functions each get called when the value for that key
changes (or is initially set).

Note that there's no type checking, e.g., to ensure that state["video"] is
being set to a `Video` object. This is a potential source of bugs since
callbacks connected to some key will often assume that value will always be of
some specific type.
"""

import inspect
from typing import Any, Callable, List, Union, Optional


GSVarType = str
NO_ARG = object()

# Transient (session-only) GuiState keys for per-instance canvas visibility.
# These are intentionally NOT persisted to the `.slp`/`Labels` model; they are a
# view preference that resets when the frame changes (instances differ per
# frame). See `sleap.gui.dataviews.LabeledFrameTableModel` (the checkbox columns
# that drive them) and `sleap.gui.overlays.instance.InstanceOverlay` (which
# re-applies them on every replot).
#
# - INSTANCE_HIDDEN_KEY -> set of ``id(instance)`` whose "Visibility" box is
#   unchecked (hidden on the canvas).
# - VIEW_ONLY_INSTANCE_KEY -> ``id(instance)`` of the single "View Only"
#   instance, or ``None``. When set, only that instance is visible and the whole
#   Visibility column is disabled (radio-like exclusivity).
# - SHOW_NONVISIBLE_OVERRIDE_KEY -> dict ``{id(instance): bool}``, a per-instance
#   override of the global "show non-visible nodes" flag. An absent key means
#   "use the global default"; an explicit ``True``/``False`` overrides it.
INSTANCE_HIDDEN_KEY = "instance_hidden"
VIEW_ONLY_INSTANCE_KEY = "view_only_instance"
SHOW_NONVISIBLE_OVERRIDE_KEY = "show_nonvisible_override"

# Label QC "display mode" (persisted app preference, NOT per-instance/per-frame).
QC_DISPLAY_MODE_KEY = "qc_display_mode"
QC_MODE_MANUAL = "manual"
QC_MODE_SELECTED_ONLY = "selected_only"
QC_MODE_ALL_VISIBLE = "all_visible_only"
QC_MODE_ALL_PLUS_SELECTED = "all_plus_selected_invisible"

# (label, mode) pairs for the Label QC display-mode selectors -- the QC dock's
# "Display:" combo and the View-menu submenu both build from this, so the two
# selectors cannot drift apart.
QC_MODE_CHOICES = (
    ("Manual", QC_MODE_MANUAL),
    ("Only selected (with hidden points)", QC_MODE_SELECTED_ONLY),
    ("All instances, visible points only", QC_MODE_ALL_VISIBLE),
    ("All visible + selected hidden points", QC_MODE_ALL_PLUS_SELECTED),
)


def instance_visible(state: "GuiState", instance: Any) -> bool:
    """Return the effective canvas visibility for an instance.

    This is the single source of truth shared by the table model (which sets
    the state) and the instance overlay (which applies it on every replot), so
    the two cannot drift.

    Args:
        state: The `GuiState` holding the transient visibility keys.
        instance: The `Instance`/`PredictedInstance` object (keyed by identity).

    Returns:
        ``True`` if the instance should be drawn, ``False`` if it should be
        hidden. The global "show instances" toggle takes precedence: when it is
        off, every instance is hidden. Otherwise, if a view-only instance is
        set, only that instance is visible; else an instance is visible unless
        its id is in the hidden set.
    """
    if state is None:
        return True

    # The global "show instances" toggle wins: per-instance state can only
    # further hide instances, never force a globally-hidden one back on. Without
    # this, the per-instance re-apply loop in `InstanceOverlay.add_to_scene`
    # would override the global Hide toggle on every replot.
    if not state.get("show instances", default=True):
        return False

    view_only = state.get(VIEW_ONLY_INSTANCE_KEY, default=None)
    if view_only is not None:
        return id(instance) == view_only

    hidden = state.get(INSTANCE_HIDDEN_KEY, default=None)
    if not hidden:
        return True
    return id(instance) not in hidden


def instance_shows_non_visible(
    state: "GuiState", instance: Any, global_default: bool
) -> bool:
    """Return whether THIS instance's non-visible (occluded/NaN) nodes are drawn.

    Orthogonal to `instance_visible`: that decides whether the instance is drawn
    at all; this decides whether its occluded keypoints draw. Per-instance
    overrides (the "Invisible Nodes" column / a non-manual QC mode) beat the
    global "show non-visible nodes" flag; absent override -> the global default.

    Args:
        state: The `GuiState` holding the transient override key.
        instance: The `Instance`/`PredictedInstance` object (keyed by identity).
        global_default: The current global "show non-visible nodes" flag, used
            when there is no per-instance override.

    Returns:
        ``True`` if this instance's non-visible nodes should be drawn.
    """
    if state is None:
        return global_default
    override = state.get(SHOW_NONVISIBLE_OVERRIDE_KEY, default=None)
    if not override:
        return global_default
    return override.get(id(instance), global_default)


def compute_qc_visibility(
    mode: str,
    selected_instance: Any,
    instances: list,
    global_show_non_visible: bool,
) -> dict:
    """Map a QC display mode + selection -> per-instance visibility flags.

    Returns ``{id(instance): (visible, show_non_visible)}``. An empty dict is the
    "manual" sentinel: the caller leaves the per-instance transient state alone.
    Selection match is by ``id()``; if ``selected_instance`` is ``None`` or not in
    ``instances``, the selection-relative modes fall back to the FIRST instance
    (so the mode stays visible and the canvas is never blank), narrowing to the
    real instance once one is selected.

    Args:
        mode: One of the ``QC_MODE_*`` constants.
        selected_instance: The currently selected instance, or ``None``.
        instances: The instances shown on the current frame.
        global_show_non_visible: The global "show non-visible nodes" flag (unused
            by the locked modes, kept for symmetry / future modes).

    Returns:
        A dict ``{id(instance): (visible, show_non_visible)}`` (empty for manual).
    """
    if mode == QC_MODE_MANUAL:
        return {}

    sel_id = id(selected_instance) if selected_instance is not None else None
    ids_present = {id(i) for i in instances}
    sel_present = sel_id in ids_present

    # The selection-relative modes (`selected_only`, `all_plus_selected_invisible`)
    # need a target instance. When there is no valid on-frame selection -- right
    # after switching modes, navigating frames, or opening QC before clicking a
    # flag -- fall back to the FIRST instance so the mode stays visibly meaningful
    # (it shows/keeps one instance) instead of collapsing to "show all", and so it
    # never blanks the canvas. It narrows to the real instance once the user
    # selects one.
    if (
        not sel_present
        and instances
        and mode in (QC_MODE_SELECTED_ONLY, QC_MODE_ALL_PLUS_SELECTED)
    ):
        sel_id = id(instances[0])
        sel_present = True

    flags = {}
    for inst in instances:
        iid = id(inst)
        is_sel = sel_present and iid == sel_id
        if mode == QC_MODE_SELECTED_ONLY:
            flags[iid] = (is_sel, is_sel)  # show only selected; its hidden pts on
        elif mode == QC_MODE_ALL_VISIBLE:
            flags[iid] = (True, False)  # all visible, occluded pts hidden
        elif mode == QC_MODE_ALL_PLUS_SELECTED:
            flags[iid] = (True, is_sel)  # all visible; selected also shows hidden pts
        else:
            flags[iid] = (True, False)  # unknown mode -> safe "all visible"
    return flags


class GuiState(object):
    """
    Class for passing persistent gui state variables.

    Arbitrary variables can be set, bools can be toggled, and callbacks can be
    automatically triggered on variable changes.

    This allows us to separate controls (which set state variables) and views
    (which can update themselves when the relevant state variables change).
    """

    def __init__(self):
        self._state_vars = dict()
        self._callbacks = dict()

    def __repr__(self) -> str:
        message = "GuiState("
        for key in self._state_vars:
            message += f"'{key}'={self.get(key)}, "
        return f"{message[:-2]})"

    def __getitem__(self, key: GSVarType) -> Any:
        """Gets value for key, or None if no value."""
        return self.get(key, default=None)

    def __setitem__(self, key: GSVarType, value):
        """Sets value for key, triggering any callbacks bound to key."""
        old_val = self.get(key, default=object())
        self._state_vars[key] = value
        if old_val != value:
            self.emit(key)

    def __contains__(self, key) -> bool:
        """Does state contain key?"""
        return key in self._state_vars

    def __delitem__(self, key: GSVarType):
        """Removes key from state. Doesn't trigger callbacks."""
        if key in self:
            del self._state_vars[key]

    def get(self, key: GSVarType, default=NO_ARG) -> Any:
        """Getter with support for default value."""
        if default is not NO_ARG:
            return self._state_vars.get(key, default)
        return self._state_vars.get(key)

    def set(self, key: GSVarType, value: Any):
        """Functional version of setter (for use in lambdas)."""
        self[key] = value

    def toggle(self, key: GSVarType, default: bool = False):
        """Toggle boolean value for specified key."""
        self[key] = not self.get(key, default=default)

    def increment(
        self, key: GSVarType, step: int = 1, mod: Optional[int] = None, default: int = 0
    ):
        """Increment numeric value for specified key.

        Args:
            key: The key.
            step: What to add to current value.
            mod: Wrap value (i.e., apply modulus) if not None.
            default: Set value to this if there's no current value for key.

        Returns:
            None.
        """
        if key not in self._state_vars:
            self[key] = default
            return

        new_value = self.get(key) + step

        # Wrap the value if it's out of bounds.
        if mod is not None:
            new_value %= mod

        self[key] = new_value

    def increment_in_list(
        self, key: GSVarType, value_list: list, reverse: bool = False
    ):
        """Advance to subsequent (or prior) value in list.

        When current value for key is not found in list, the value is set to
        the first (or last, if reverse) item in list.

        Args:
            key: The key.
            value_list: List of values of any type which supports equality check.
            reverse: Whether to use next or previous item in value list.

        Returns:
            None.
        """
        if self[key] not in value_list:
            if reverse:
                self[key] = value_list[-1]
            else:
                self[key] = value_list[0]
        else:
            idx = value_list.index(self[key])
            step = 1 if not reverse else -1
            self[key] = value_list[(idx + step) % len(value_list)]

    def connect(self, key: GSVarType, callbacks: Union[Callable, List[Callable]]):
        """
        Connects one or more callbacks for state variable.

        Callbacks are called (triggered) whenever the state is changed, i.e.,
        when the value for some key is set either (i) initially or (ii) to
        a different value than the current value.

        This is analogous to connecting a function to a Qt slot.

        Callback should take a single arg, which will be the current (new)
        value of whatever state var is triggering the callback.
        """
        if callable(callbacks):
            self._connect_callback(key, callbacks)
        else:
            for callback in callbacks:
                self._connect_callback(key, callback)

    def _connect_callback(self, key: GSVarType, callback: Callable):
        """Connect a callback for state variable."""
        if not callable(callback):
            raise ValueError("callback must be callable")
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def emit(self, key: GSVarType):
        """
        Trigger callbacks for state variable.

        This calls each callback for the specified key, without needing to
        change the value of the key.

        This is analogous to emitting a Qt signal.
        """
        if key in self._state_vars and key in self._callbacks:
            val = self.get(key)
            for i, callback in enumerate(self._callbacks[key]):
                try:
                    # if callback doesn't take positional args, just call it
                    if not inspect.signature(callback).parameters:
                        callback()
                    # otherwise, pass value as first positional arg
                    else:
                        callback(val)
                except Exception as e:
                    print(f"Error occurred during callback {i} for {key}!")
                    print(self._callbacks[key])
                    print(e)
