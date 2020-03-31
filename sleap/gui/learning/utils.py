from typing import Any, Dict, Optional, Text

import attr
import cattr
import numpy as np

from sleap import Skeleton
from sleap.nn.config import TrainingJobConfig, ModelConfig
from sleap.nn.model import Model


@attr.s(auto_attribs=True)
class ScopedKeyDict:

    key_val_dict: Dict[Text, Any]

    @classmethod
    def set_hierarchical_key_val(cls, current_dict, key, val):
        # Ignore "private" keys starting with "_"
        if key[0] == "_":
            return

        if "." not in key:
            current_dict[key] = val
        else:
            top_key, *subkey_list = key.split(".")
            if top_key not in current_dict:
                current_dict[top_key] = dict()
            subkey = ".".join(subkey_list)
            cls.set_hierarchical_key_val(current_dict[top_key], subkey, val)

    def to_hierarchical_dict(self):
        hierarch_dict = dict()
        for key, val in self.key_val_dict.items():
            self.set_hierarchical_key_val(hierarch_dict, key, val)
        return hierarch_dict

    @classmethod
    def from_hierarchical_dict(cls, hierarch_dict):
        return cls(key_val_dict=cls._make_flattened_dict(hierarch_dict))

    @classmethod
    def _make_flattened_dict(cls, hierarch_dicts, scope_string=""):
        flattened_dict = dict()
        for key, val in hierarch_dicts.items():
            if isinstance(val, Dict):
                # Dict so recurse adding node to scope string
                flattened_dict.update(
                    cls._make_flattened_dict(
                        hierarch_dicts=val,
                        scope_string=cls._subscope_key(scope_string, key),
                    )
                )
            else:
                # Leafs (non-dict)
                flattened_dict[cls._subscope_key(scope_string, key)] = val
        return flattened_dict

    @staticmethod
    def _subscope_key(scope_string, key):
        return key if not scope_string else f"{scope_string}.{key}"


def apply_cfg_transforms_to_key_val_dict(key_val_dict):
    if "outputs.tags" in key_val_dict and isinstance(key_val_dict["outputs.tags"], str):
        key_val_dict["outputs.tags"] = [
            tag.strip() for tag in key_val_dict["outputs.tags"].split(",")
        ]

    if "_ensure_channels" in key_val_dict:
        ensure_channels = key_val_dict["_ensure_channels"].lower()
        ensure_rgb = False
        ensure_grayscale = False
        if ensure_channels == "rgb":
            ensure_rgb = True
        elif ensure_channels == "grayscale":
            ensure_grayscale = True

        key_val_dict["data.preprocessing.ensure_rgb"] = ensure_rgb
        key_val_dict["data.preprocessing.ensure_grayscale"] = ensure_grayscale

    if "model.backbone.resnet.upsampling.skip_connections" in key_val_dict:
        if key_val_dict["model.backbone.resnet.upsampling.skip_connections"] == "":
            key_val_dict["model.backbone.resnet.upsampling.skip_connections"] = None


def make_training_config_from_key_val_dict(key_val_dict):
    apply_cfg_transforms_to_key_val_dict(key_val_dict)
    cfg_dict = ScopedKeyDict(key_val_dict).to_hierarchical_dict()

    cfg = cattr.structure(cfg_dict, TrainingJobConfig)

    return cfg


def make_model_config_from_key_val_dict(key_val_dict):
    apply_cfg_transforms_to_key_val_dict(key_val_dict)
    cfg_dict = ScopedKeyDict(key_val_dict).to_hierarchical_dict()

    if "model" in cfg_dict:
        cfg_dict = cfg_dict["model"]

    return cattr.structure(cfg_dict, ModelConfig)


def compute_rf(down_blocks: int, convs_per_block: int = 2, kernel_size: int = 3) -> int:
    """
    Ref: https://distill.pub/2019/computing-receptive-fields/ (Eq. 2)
    """
    # Define the strides and kernel sizes for a single down block.
    # convs have stride 1, pooling has stride 2:
    block_strides = [1] * convs_per_block + [2]

    # convs have `kernel_size` x `kernel_size` kernels, pooling has 2 x 2 kernels:
    block_kernels = [kernel_size] * convs_per_block + [2]

    # Repeat block parameters by the total number of down blocks.
    strides = np.array(block_strides * down_blocks)
    kernels = np.array(block_kernels * down_blocks)

    # L = Total number of layers
    L = len(strides)

    # Compute the product term of the RF equation.
    rf = 1
    for l in range(L):
        rf += (kernels[l] - 1) * np.prod(strides[:l])

    return int(rf)


def receptive_field_info_from_model_cfg(model_cfg: ModelConfig) -> dict:
    rf_info = dict(
        size=None,
        max_stride=None,
        down_blocks=None,
        convs_per_block=None,
        kernel_size=None,
    )

    try:
        model = Model.from_config(model_cfg, Skeleton())
    except ZeroDivisionError:
        # Unable to create model from these config parameters
        return rf_info

    if hasattr(model_cfg.backbone.which_oneof(), "max_stride"):
        rf_info["max_stride"] = model_cfg.backbone.which_oneof().max_stride

    if hasattr(model.backbone, "down_convs_per_block"):
        rf_info["convs_per_block"] = model.backbone.down_convs_per_block
    elif hasattr(model.backbone, "convs_per_block"):
        rf_info["convs_per_block"] = model.backbone.convs_per_block

    if hasattr(model.backbone, "kernel_size"):
        rf_info["kernel_size"] = model.backbone.kernel_size

    rf_info["down_blocks"] = model.backbone.down_blocks

    if rf_info["down_blocks"] and rf_info["convs_per_block"] and rf_info["kernel_size"]:
        rf_info["size"] = compute_rf(
            down_blocks=rf_info["down_blocks"],
            convs_per_block=rf_info["convs_per_block"],
            kernel_size=rf_info["kernel_size"],
        )

    return rf_info
