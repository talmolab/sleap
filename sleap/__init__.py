import logging
import sys


# Setup logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Import submodules we want available at top-level
from sleap.version import __version__, versions
from sleap.io.dataset import Labels, load_file
from sleap.io.video import Video, load_video
from sleap.instance import LabeledFrame, Instance, PredictedInstance, Track
from sleap_io.model.skeleton import Skeleton
