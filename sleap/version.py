"""
Current version of SLEAP package.

This is used to set sleap.__version__, and it is also read by `setup.py` to set
the version for the package (i.e., the version on PyPI).

The version should be manually updated before creating a new release, and then
the release should be created with a tag matching the version number set here.
For example, if you set the version to X.Y.Z, then the tag should be "vX.Y.Z".

Must be a semver string, "aN" should be appended for alpha releases.
"""


__version__ = "1.2.2"


def versions():
    """Print versions of SLEAP and other libraries."""
    import tensorflow as tf
    import numpy as np
    import platform

    vers = {}
    vers["SLEAP"] = __version__
    vers["TensorFlow"] = tf.__version__
    vers["Numpy"] = np.__version__
    vers["Python"] = platform.python_version()
    vers["OS"] = platform.platform()

    msg = "\n".join([f"{k}: {v}" for k, v in vers.items()])
    print(msg)
