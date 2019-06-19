from sleap.nn.architectures.leap import LeapCNN
from sleap.nn.architectures.unet import UNet, StackedUNet
from sleap.nn.architectures.hourglass import StackedHourglass

# TODO: We can set this up to find all classes under sleap.nn.architectures
available_archs = [LeapCNN, UNet, StackedUNet, StackedHourglass]

__all__ = ['available_archs'] + [arch.__name__ for arch in available_archs]
