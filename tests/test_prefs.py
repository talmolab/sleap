from sleap.prefs import prefs
import pytest


def test_prefs():
    assert prefs["palette"] != ""

    with pytest.raises(KeyError):
        prefs["not a valid preference key"]
