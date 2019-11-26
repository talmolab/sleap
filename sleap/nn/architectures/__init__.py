from sleap.nn.architectures.leap import LeapCNN
from sleap.nn.architectures.unet import UNet, StackedUNet
from sleap.nn.architectures.hourglass import StackedHourglass
from sleap.nn.architectures.resnet import ResNet50
from sleap.nn.architectures.densenet import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    GeneralizedDenseNet,
    UDenseNet
)
from sleap.nn.architectures.mobilenets import MobileNetV1, MobileNetV2
from sleap.nn.architectures.nasnet import NASNetMobile, NASNetLarge, GeneralizedNASNet
from sleap.nn.architectures.hrnet import HigherHRNet
from typing import TypeVar


# TODO: We can set this up to find all classes under sleap.nn.architectures
available_archs = [
    LeapCNN,
    UNet,
    StackedUNet,
    StackedHourglass,
    ResNet50,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    GeneralizedDenseNet,
    UDenseNet,
    MobileNetV1,
    MobileNetV2,
    NASNetMobile,
    NASNetLarge,
    GeneralizedNASNet,
    HigherHRNet,
]
available_arch_names = [arch.__name__ for arch in available_archs]
BackboneType = TypeVar("BackboneType", *available_archs)

__all__ = ["available_archs", "available_arch_names", "BackboneType"] + [
    arch.__name__ for arch in available_archs
]
