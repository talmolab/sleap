"""Module for web-related functionality."""

from typing import Dict, Any

import requests


ANALYTICS_ENDPOINT = "https://analytics.sleap.ai/ping"


def get_analytics_data() -> Dict[str, Any]:
    """Gather data to be transmitted to analytics backend."""
    import os
    import sleap
    from pathlib import Path
    import platform

    return {
        "sleap_version": sleap.__version__,
        "python_version": platform.python_version(),
        "tf_version": "N/A",  # TensorFlow no longer bundled with GUI
        "conda_env": Path(os.environ.get("CONDA_PREFIX", "")).stem,
        "platform": platform.platform(),
    }


def ping_analytics():
    """Ping analytics service with anonymous usage data.

    Notes:
        This only gets called when the GUI starts and obeys user preferences for data
        collection.

        See https://docs.sleap.ai/latest/help/#usage for more information.
    """
    import threading

    analytics_data = get_analytics_data()

    def _ping_analytics():
        try:
            requests.post(
                ANALYTICS_ENDPOINT,
                json=analytics_data,
            )
        except (requests.ConnectionError, requests.Timeout):
            pass

    # Fire and forget.
    threading.Thread(target=_ping_analytics).start()
