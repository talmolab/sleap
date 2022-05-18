import logging
import sys

# Setup logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Import submodules we want available at top-level
from sleap.version import __version__, versions
from sleap.io.dataset import Labels, load_file
from sleap.io.video import Video, load_video
from sleap.instance import LabeledFrame, Instance, PredictedInstance, Track
from sleap.skeleton import Skeleton
import sleap.nn
from sleap.nn.data import pipelines
from sleap.nn import inference
from sleap.nn.inference import load_model
from sleap.nn.system import use_cpu_only, disable_preallocation
from sleap.nn.system import summary as system_summary
from sleap.nn.config import TrainingJobConfig, load_config
from sleap.nn.evals import load_metrics
