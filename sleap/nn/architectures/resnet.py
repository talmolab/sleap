import tensorflow as tf
import keras
from keras import applications

import attr

@attr.s(auto_attribs=True)
class ResNet50:
  """ResNet50 pretrained backbone.

  Args:
    x_in: Input 4-D tf.Tensor or instantiated layer.
    num_output_channels: The number of output channels of the block.
    upsampling_layers: Use upsampling instead of transposed convolutions.
    interp: Method to use for interpolation when upsampling smaller features.
    up_blocks: Number of upsampling steps to perform. The backbone reduces
      the output scale by 1/32. If set to 5, outputs will be upsampled to the
      input resolution.
    refine_conv_up: If true, applies a 1x1 conv after each upsampling step.
    pretrained: Load pretrained ImageNet weights for transfer learning. If
      False, random weights are used for initialization.
  """

  upsampling_layers: bool = True
  interp: str = "bilinear"
  up_blocks: int = 5
  refine_conv_up: bool = False
  pretrained: bool = True

  def output(self, x_in, num_output_channels):
      """
      Generate a tensorflow graph for the backbone and return the output tensor.

      Args:
          x_in: Input 4-D tf.Tensor or instantiated layer. Must have height and width
          that are divisible by `2^down_blocks.
          num_output_channels: The number of output channels of the block. These
          are the final output tensors on which intermediate supervision may be
          applied.

      Returns:
          x_out: tf.Tensor of the output of the block of with `num_output_channels` channels.
      """
      return resnet50(x_in, num_output_channels, **attr.asdict(self))

  @property
  def down_blocks(self):
    """Returns the number of downsampling steps in the model."""

    # This is a fixed constant for ResNet50.
    return 5
  

  @property
  def output_scale(self):
    """Returns relative scaling factor of this backbone."""

    return (1 / (2 ** (self.down_blocks - self.up_blocks)))


def preprocess_input(X):
  """Rescale input to [-1, 1] and tile if not RGB."""
  X = (X * 2) - 1

  if tf.shape(X)[-1] != 3:
    X = tf.tile(X, [1, 1, 1, 3])

  return X


def resnet50(x_in, num_output_channels, up_blocks=5, upsampling_layers=True,
  interp="bilinear", refine_conv_up=False, pretrained=True):
  """Build ResNet50 backbone."""

  # Input should be rescaled from [0, 1] to [-1, 1] and needs to be 3 channels (RGB)
  x = keras.layers.Lambda(preprocess_input)(x_in)

  # Automatically downloads weights
  resnet_model = applications.ResNet50(
    include_top=False,
    input_shape=(int(x_in.shape[-3]), int(x_in.shape[-2]), 3),
    weights="imagenet" if pretrained else None,
    )

  # Output size is reduced by factor of 32 (2 ** 5)
  x = resnet_model(x)

  for i in range(up_blocks):
    if upsampling_layers:
      x = keras.layers.UpSampling2D(size=(2, 2), interpolation=interp)(x)
    else:
      x = keras.layers.Conv2DTranspose(2 ** (8 - i), kernel_size=3, strides=2, padding="same", kernel_initializer="glorot_normal")(x)

    if refine_conv_up:
      x = keras.layers.Conv2D(2 ** (8 - i), kernel_size=1, padding="same")(x)

  x = keras.layers.Conv2D(num_output_channels, (3, 3), padding="same")(x)

  return x
