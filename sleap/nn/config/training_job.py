"""Serializable configuration classes for specifying all training job parameters.

These configuration classes are intended to specify all the parameters required to run
a training job or perform inference from a serialized one.

They are explicitly not intended to implement any of the underlying functionality that
they parametrize. This serves two purposes:

    1. Parameter specification through simple attributes. These can be read/edited by a
       human, as well as easily be serialized/deserialized to/from simple dictionaries
       and JSON.

    2. Decoupling from the implementation. This makes it easier to design functional
       modules with attributes/parameters that contain objects that may not be easily
       serializable or may implement additional logic that relies on runtime information
       or other parameters.

In general, classes that implement the actual functionality related to these
configuration classes should provide a classmethod for instantiation from the
configuration class instances. This makes it easier to implement other logic not related
to the high level parameters at creation time.

Conveniently, this format also provides a single location where all user-facing
parameters are aggregated and documented for end users (as opposed to developers).
"""

import attr
from sleap.nn.config.data import DataConfig
from sleap.nn.config.model import ModelConfig
from sleap.nn.config.optimization import OptimizationConfig
from sleap.nn.config.outputs import OutputsConfig


@attr.s(auto_attribs=True)
class TrainingJobConfig:
    """Configuration of a training job.

    Attributes:
        data: Configuration options related to the training data.
        model: Configuration options related to the model architecture.
        optimization: Configuration options related to the training.
        outputs: Configuration options related to outputs during training.
    """

    data: DataConfig = attr.ib(factory=DataConfig)
    model: ModelConfig = attr.ib(factory=ModelConfig)
    optimization: OptimizationConfig = attr.ib(factory=OptimizationConfig)
    outputs: OutputsConfig = attr.ib(factory=OutputsConfig)
    # TODO: store fixed config format version + SLEAP version?
