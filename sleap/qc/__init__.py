"""Label Quality Control module for SLEAP.

This module provides tools to detect annotation errors in pose labeling data.

Example usage:
    import sleap_io as sio
    from sleap.qc import LabelQCDetector, QCConfig

    # Load labels
    labels = sio.load_file("labels.slp")

    # Create detector with default config
    detector = LabelQCDetector()

    # Fit on labels (uses all instances for training)
    detector.fit(labels)

    # Get results
    results = detector.score(labels)

    # Get flagged instances above threshold
    flagged = results.get_flagged(threshold=0.7)
"""

from sleap.qc.config import QCConfig
from sleap.qc.detector import LabelQCDetector
from sleap.qc.results import QCResults, QCFlag

__all__ = [
    "QCConfig",
    "LabelQCDetector",
    "QCResults",
    "QCFlag",
]
