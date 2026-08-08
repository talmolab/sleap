"""Temporary feature flags for in-progress functionality.

NOTE: This module is intentionally short-lived. The centroid-only models flag
gates UI for a feature whose *inference* half is still blocked on sleap-nn
(epic talmolab/sleap-nn#508 / PR #562). Once centroid-only inference lands and
the feature is fully supported, delete this module and its call sites
(grep for is_centroid_models_enabled).

In the GUI, this feature is enabled via the "Experimental Features" toggle in
the Help menu. The SLEAP_ENABLE_CENTROID_MODELS environment variable remains
available as a developer override.
"""

import os


def is_centroid_models_enabled(experimental_features: bool = False) -> bool:
    """Return True if the experimental centroid-only models UI is enabled.

    Enabled when EITHER:
    - experimental_features is True (the "Experimental Features" toggle in
      the Help menu, threaded in by the caller), or
    - the SLEAP_ENABLE_CENTROID_MODELS environment variable is truthy
      ("1"/"true"/"yes"/"on", case-insensitive).
    """
    if experimental_features:
        return True
    return os.environ.get("SLEAP_ENABLE_CENTROID_MODELS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
