"""
Handles SLEAP preferences.

Importing this module creates `prefs`, instance of `Preferences` class.
"""

import logging
from pathlib import Path

import yaml

from sleap import util

logger = logging.getLogger(__name__)


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
        "trail shade": "Normal",
        "trail width": 4.0,
        "trail node count": 1,
        "marker size": 4,
        "edge style": "Line",
        "window state": b"",
        "node label size": 12,
        "show non-visible nodes": True,
        "share usage data": True,
        "node marker sizes": (1, 2, 3, 4, 6, 8, 12),
        "node label sizes": (6, 9, 12, 18, 24, 36),
        # Training pipeline settings (system-level, persist across projects)
        "training data pipeline framework": "Cache in Memory",
        "training num workers": 0,
        "training num devices": None,  # None = auto-detect
        "training accelerator": "auto",
    }
    _filename = "preferences.yaml"

    def __init__(self):
        self._load_or_create()

    def _get_file_path(self) -> Path:
        """Get the path to the preferences file."""
        return Path(util.get_config_file(self._filename, ignore_file_not_found=True))

    def _load_or_create(self):
        """Load preferences from file, creating file if it doesn't exist."""
        file_path = self._get_file_path()
        file_existed = file_path.exists()

        if file_existed:
            try:
                self._prefs = util.get_config_yaml(self._filename) or {}
                logger.debug(f"Loaded preferences from {file_path}")
            except yaml.YAMLError as e:
                logger.warning(
                    f"Invalid preferences file at {file_path}: {e}\n"
                    f"Using defaults. Delete the file or fix the YAML syntax."
                )
                self._prefs = {}
            except Exception as e:
                logger.warning(f"Error loading preferences from {file_path}: {e}")
                self._prefs = {}
        else:
            self._prefs = {}

        # Apply defaults for missing keys
        missing_keys = []
        for k, v in self._defaults.items():
            if k not in self._prefs:
                self._prefs[k] = v
                missing_keys.append(k)

        if missing_keys:
            logger.debug(
                f"Applied defaults for {len(missing_keys)} missing preference keys: "
                f"{missing_keys}"
            )

        # Create file if it didn't exist (so user can edit it)
        if not file_existed:
            self.save()
            logger.info(f"Created preferences file: {file_path}")

    def load(self):
        """Load preferences from file, if not already loaded."""
        if self._prefs is None:
            self._load_or_create()

    def load_(self):
        """Reload preferences from file (regardless of whether loaded already)."""
        self._prefs = None
        self._load_or_create()

    def save(self):
        """Save preferences to file."""
        util.save_config_yaml(self._filename, self._prefs)
        logger.debug(f"Saved preferences to {self._get_file_path()}")

    def reset_to_default(self):
        """Reset preferences to default."""
        util.save_config_yaml(self._filename, self._defaults)
        self.reload()

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
