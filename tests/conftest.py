
import os
import gc
import pytest
import matplotlib
from matplotlib import pyplot as plt
from PySide6 import QtWidgets


# import logging

# try:
#     pass
# except Exception:
#     logging.warning("Could not import pytestqt, skipping GUI tests.")
#     collect_ignore_glob = ["gui/*"]

from tests.fixtures.skeletons import *
from tests.fixtures.instances import *
from tests.fixtures.datasets import *
from tests.fixtures.videos import *
from tests.fixtures.models import *

"""Guard against Qt/Matplotlib teardown segfaults in CI.

- Force QtAgg at session start (so GUI tests behave as intended).
- Before interpreter shutdown, close all figures and switch Matplotlib to Agg,
  then drain Qt events. This breaks the Qt <-> Matplotlib linkage cleanly.
"""


def pytest_sessionstart(session):
    """Use QtAgg for GUI tests even if MPLBACKEND was set in CI."""
    os.environ.pop("MPLBACKEND", None)
    matplotlib.use("QtAgg", force=True)


@pytest.fixture(autouse=True, scope="session")
def _qt_mpl_teardown_guard():
    """Clean teardown to avoid segfaults on process exit."""
    yield
    # Close any straggler figures first.
    plt.close("all")

    # Break Matplotlib's Qt bindings before Python finalizes Qt.
    # (Switching to Agg here prevents backend_qtagg objects from
    # being torn down after Qt has already started tearing down.)
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass

    # Ask Qt to close top-level widgets and process pending events.
    app = QtWidgets.QApplication.instance()
    if app is not None:
        for w in QtWidgets.QApplication.topLevelWidgets():
            try:
                w.close()
            except Exception:
                pass
        app.processEvents()

    # Encourage deterministic finalization order.
    gc.collect()