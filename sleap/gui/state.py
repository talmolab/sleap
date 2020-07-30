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
from typing import Any, Callable, List, Union


GSVarType = str
NO_ARG = object()


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

    def increment(self, key: GSVarType, step: int = 1, mod: int = 1, default: int = 0):
        """Increment numeric value for specified key.

        Args:
            key: The key.
            step: What to add to current value.
            mod: Wrap value (i.e., apply modulus) if not 1.
            default: Set value to this if there's no current value for key.

        Returns:
            None.
        """
        if key not in self._state_vars:
            self[key] = default
        else:
            new_value = self.get(key) + step

            # take modulo of value if mod arg is not 1
            if mod != 1:
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
