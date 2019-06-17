"""
The Model class is a wrapper class that specifies an interface to pose
estimation models supported by sLEAP. It is fairly straighforward class
that allows specification of the underlying architecture, the data generator
for training, and the instantiation of a predictor object for inference.
"""

import attr
import keras

from typing import Callable, Tuple, Dict

@attr.s(auto_attribs=True)
class Model:

    data_generator: keras.utils.Sequence
    backbone: Callable
    backbone_args: Tuple
    backbone_kwargs: Dict

