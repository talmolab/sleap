from enum import Enum
from typing import Any, Callable, Iterable, List, Union


class GSVar(Enum):
    COLOR_PREDICTED = 1
    FILENAME = 2
    FIT = 3
    FRAME_IDX = 4
    FRAME_RANGE = 5
    HAS_CHANGES = 6
    HAS_FRAME_RANGE = 7
    INSTANCE = 8
    LABELED_FRAME = 9
    LABELS = 10
    PALETTE = 11
    SEEKBAR_HEADER = 12
    SHOW_EDGES = 13
    SHOW_LABELS = 14
    SHOW_TRAILS = 15
    SKELETON = 16
    SUGGESTION_IDX = 17
    TRAIL_LENGTH = 18
    VIDEO = 19


GSVarType = str


class GuiState(object):
    _state_vars = dict()
    _callbacks = dict()

    def __getitem__(self, item):
        return self.get(item, default=None)

    def __setitem__(self, key: GSVarType, value):
        old_val = self.get(key, default=object())
        self._state_vars[key] = value
        if old_val != value:
            self.emit(key)

    def __contains__(self, item):
        return item in self._state_vars

    def __delitem__(self, key: GSVarType):
        if key in self:
            del self._state_vars[key]

    def get(self, key: GSVarType, **kwargs):
        """Getter with support for default value."""
        if "default" in kwargs:
            return self._state_vars.get(key, kwargs["default"])
        return self._state_vars.get(key)

    def set(self, key: GSVarType, value: Any):
        """Functional version of setter (for use in lambdas)."""
        self[key] = value

    def toggle(self, key: GSVarType, default: bool = False):
        self[key] = not self.get(key, default=default)

    def increment(self, key: GSVarType, step: int = 1, mod: int = 1, default: int = 0):
        if key not in self._state_vars:
            self[key] = default
        else:
            self[key] = (self.get(key) + step) % mod

    def increment_in_list(
        self, key: GSVarType, value_list: list, reverse: bool = False
    ):
        if self[key] not in value_list:
            if reverse:
                self[key] = value_list[-1]
            else:
                self[key] = value_list[0]
        else:
            idx = value_list.index(self[key])
            step = 1 if not reverse else -1
            self[key] = value_list[(idx + step) % len(value_list)]

    def next_int_in_list(
        self, key: GSVarType, value_list: Iterable[int], reverse=False
    ):
        """Advances to subsequent (or prior) value in list.

        Goes to value subsequent (or prior) to current value, regardless
        of whether current value is member of list.
        """
        raise NotImplementedError("next_int_in_list not yet implemented!")

    def connect(self, key: GSVarType, callbacks: Union[Callable, List[Callable]]):
        """Connects one or more callbacks for state variable."""
        if callable(callbacks):
            self._connect_callback(key, callbacks)
        else:
            for callback in callbacks:
                self._connect_callback(key, callback)

    def _connect_callback(self, key: GSVarType, callback: Callable):
        """Connects a callback for state variable."""
        if callback is None:
            raise ValueError("callback cannot be None!")
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def emit(self, key: GSVarType):
        """Triggers callbacks for state variable."""
        if key in self._state_vars and key in self._callbacks:
            val = self.get(key)
            for i, callback in enumerate(self._callbacks[key]):
                try:
                    callback(val)
                except Exception as e:
                    print(f"Error occurred during callback {i} for {key}!")
                    print(self._callbacks[key])
                    print(e)
