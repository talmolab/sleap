"""
Handles SLEAP preferences.

Importing this module creates `prefs`, instance of `Preferences` class.
"""

from sleap import util


class Preferences(object):
    """Class for accessing SLEAP preferences."""

    _prefs = None
    _defaults = {
        "medium step size": 10,
        "large step size": 100,
        "color predicted": False,
        "propagate track labels": True,
        "palette": "standard",
        "bold lines": False,
        "trail length": 0,
        "trail width": 4.0,
        "trail node count": 1,
        "marker size": 4,
        "edge style": "Line",
        "window state": b"",
        "node label size": 12,
        "show non-visible nodes": True,
    }
    _filename = "preferences.yaml"

    def __init__(self):
        self.load()

    def load(self):
        """Load preferences from file, if not already loaded."""
        if self._prefs is None:
            self.load_()

    def load_(self):
        """Load preferences from file (regardless of whether loaded already)."""
        try:
            self._prefs = util.get_config_yaml(self._filename)
            if not hasattr(self._prefs, "get"):
                self._prefs = self._defaults
        except FileNotFoundError:
            self._prefs = self._defaults

    def save(self):
        """Save preferences to file."""
        util.save_config_yaml(self._filename, self._prefs)

    def reset_to_default(self):
        """Reset preferences to default."""
        util.save_config_yaml(self._filename, self._defaults)
        self.load()

    def _validate_key(self, key):
        if key not in self._defaults:
            raise KeyError(f"No preference matching '{key}'")

    def __contains__(self, item) -> bool:
        return item in self._defaults

    def __getitem__(self, key):
        self.load()
        self._validate_key(key)
        return self._prefs.get(key, self._defaults[key])

    def __setitem__(self, key, value):
        self.load()
        self._validate_key(key)
        self._prefs[key] = value


prefs = Preferences()

# save preference so that user editable file is created if it doesn't exist
prefs.save()
