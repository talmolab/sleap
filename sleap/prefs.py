from sleap import util


class Preferences(object):
    """Class for accessing SLEAP preferences."""

    _prefs = None
    _defaults = {
        "medium step size": 4,
        "large step size": 100,
        "color predicted": False,
        "palette": "standard",
        "trail length": 0,
        "trail width": 4.0,
        "hide videos dock": False,
        "hide skeleton dock": False,
        "hide instances dock": False,
        "hide labeling suggestions dock": False,
    }
    _filename = "preferences.yaml"

    def __init__(self):
        self.load()

    def load(self):
        if self._prefs is None:
            self.load_()

    def load_(self):
        try:
            self._prefs = util.get_config_yaml(self._filename)
            if not hasattr(self._prefs, "get"):
                self._prefs = self._defaults
        except FileNotFoundError:
            self._prefs = self._defaults

    def save(self):
        util.save_config_yaml(self._filename, self._prefs)

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
prefs.save()
