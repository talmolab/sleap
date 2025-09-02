import logging
import sys


# Setup logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Import submodules we want available at top-level
from sleap.version import __version__, versions
from sleap_io import (
    Labels,
    LabeledFrame,
    Skeleton,
    Node,
    Instance,
    PredictedInstance,
    Video,
    SuggestionFrame,
)
