"""
Conversion between flat (form data) and hierarchical (config object) dicts.
"""

from typing import Any, Dict, Optional, Text

import attr
import cattr

from sleap.nn.config import TrainingJobConfig, ModelConfig


@attr.s(auto_attribs=True)
class ScopedKeyDict:
    """
    Class to support conversion between flat and hierarchical dictionaries.

    Flat dictionaries have scoped keys, e.g., "foo.bar". These typically come
    from user-editable forms.

    Hierarchical dictionaries have dictionaries as values of other dictionaries,
    e.g., `{"foo": {"bar": ... } }`. These are typically used when serializing
    and deserializing objects.

    Attributes:
        key_val_dict: Data stores in a *flat* dictionary with scoped keys.
    """

    key_val_dict: Dict[Text, Any]

    @classmethod
    def set_hierarchical_key_val(cls, current_dict: dict, key: Text, val: Any):
        """Sets value in hierarchical dict via scoped key."""
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

    def to_hierarchical_dict(self) -> dict:
        """Converts internal flat dictionary to hierarchical dictionary."""
        hierarch_dict = dict()
        for key, val in self.key_val_dict.items():
            self.set_hierarchical_key_val(hierarch_dict, key, val)
        return hierarch_dict

    @classmethod
    def from_hierarchical_dict(cls, hierarch_dict: dict):
        """Constructs object (with flat dict) from hierarchical dictionary."""
        return cls(key_val_dict=cls._make_flattened_dict(hierarch_dict))

    @classmethod
    def _make_flattened_dict(
        cls, hierarch_dicts: dict, scope_string: Text = ""
    ) -> dict:
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
    def _subscope_key(scope_string: Text, key: Text) -> Text:
        return key if not scope_string else f"{scope_string}.{key}"


def apply_cfg_transforms_to_key_val_dict(key_val_dict: dict):
    """
    Transforms data from form to correct data types before converting to object.

    Arguments:
        key_val_dict: Flat dictionary from :py:class:`TrainingEditorWidget`.
    Returns:
        None, modifies dict in place.
    """
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


def make_training_config_from_key_val_dict(key_val_dict: dict) -> TrainingJobConfig:
    """
    Make :py:class:`TrainingJobConfig` object from flat dictionary.

    Arguments:
        key_val_dict: Flat dictionary from :py:class:`TrainingEditorWidget`.
    Returns:
        The :py:class:`TrainingJobConfig` object.
    """
    apply_cfg_transforms_to_key_val_dict(key_val_dict)
    cfg_dict = ScopedKeyDict(key_val_dict).to_hierarchical_dict()

    cfg = cattr.structure(cfg_dict, TrainingJobConfig)

    return cfg


def make_model_config_from_key_val_dict(key_val_dict: dict) -> ModelConfig:
    """
    Make :py:class:`ModelConfig` object from flat dictionary.

    Arguments:
        key_val_dict: Flat dictionary from :py:class:`TrainingEditorWidget`.
    Returns:
        The :py:class:`ModelConfig` object.
    """

    apply_cfg_transforms_to_key_val_dict(key_val_dict)
    cfg_dict = ScopedKeyDict(key_val_dict).to_hierarchical_dict()

    if "model" in cfg_dict:
        cfg_dict = cfg_dict["model"]

    return cattr.structure(cfg_dict, ModelConfig)
