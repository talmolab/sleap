from sleap.nn.architectures.leap import LeapCNN
from sleap.nn.architectures.unet import UNet, StackedUNet
from sleap.nn.architectures.hourglass import StackedHourglass
from sleap.nn.architectures.resnet import ResNet50
from typing import TypeVar

# TODO: We can set this up to find all classes under sleap.nn.architectures
available_archs = [LeapCNN, UNet, StackedUNet, StackedHourglass, ResNet50]
available_arch_names = [arch.__name__ for arch in available_archs]
BackboneType = TypeVar("BackboneType", *available_archs)

__all__ = ["available_archs", "available_arch_names", "BackboneType"] + [
    arch.__name__ for arch in available_archs
]
