import logging
import pytest
import atexit

try:
    pass
except Exception:
    logging.warning("Could not import pytestqt, skipping GUI tests.")
    collect_ignore_glob = ["gui/*"]

from tests.fixtures.skeletons import *
from tests.fixtures.instances import *
from tests.fixtures.datasets import *
from tests.fixtures.videos import *
from tests.fixtures.models import *


@pytest.fixture(autouse=True)
def cleanup_qt_video_players():
    """Ensure video player threads are cleaned up after each test."""
    yield
    # After test completes, clean up any video players
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            # Process any pending events before cleanup
            app.processEvents()

            # Find and cleanup any QtVideoPlayer instances
            from sleap.gui.widgets.video import QtVideoPlayer

            for widget in app.allWidgets():
                if isinstance(widget, QtVideoPlayer):
                    try:
                        widget.cleanup()
                    except Exception:
                        pass
    except Exception:
        pass  # Ignore cleanup errors


def cleanup_all_threads():
    """Final cleanup of any remaining threads at exit."""
    try:
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.processEvents()

            # Stop any remaining QThreads
            for thread in [obj for obj in app.children() if isinstance(obj, QThread)]:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(100)
    except Exception:
        pass


# Register cleanup at exit
atexit.register(cleanup_all_threads)
