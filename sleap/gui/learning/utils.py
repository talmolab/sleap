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
