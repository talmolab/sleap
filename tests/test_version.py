import re

from sleap.version import __version__


def test_version():
    with open("sleap/version.py") as f:
        version_file = f.read()
        sleap_version = re.search("\\d.+(?=['\"])", version_file).group(0)

        assert sleap_version == __version__
