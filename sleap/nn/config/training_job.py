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
