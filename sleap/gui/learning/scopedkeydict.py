"""
Conversion between flat (form data) and hierarchical (config object) dicts.
"""

from typing import Any, Dict, Optional, Text, Tuple

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

    # Overwrite backbone strides with stride from head.
    backbone_name = find_backbone_name_from_key_val_dict(key_val_dict)
    if backbone_name is not None:
        max_stride, output_stride = resolve_strides_from_key_val_dict(
            key_val_dict, backbone_name
        )
        key_val_dict[f"model.backbone.{backbone_name}.output_stride"] = output_stride
        key_val_dict[f"model.backbone.{backbone_name}.max_stride"] = max_stride

    # Convert random flip dropdown selection to config.
    random_flip = key_val_dict.get(
        "optimization.augmentation_config.random_flip", "none"
    )
    if random_flip == "none":
        key_val_dict["optimization.augmentation_config.random_flip"] = False
    else:
        key_val_dict["optimization.augmentation_config.random_flip"] = True
        key_val_dict["optimization.augmentation_config.flip_horizontal"] = (
            random_flip == "horizontal"
        )


def find_backbone_name_from_key_val_dict(key_val_dict: dict):
    """Find the backbone model name from the config dictionary."""
    backbone_name = None
    for key in key_val_dict:
        if key.startswith("model.backbone."):
            backbone_name = key.split(".")[2]

    return backbone_name


def resolve_strides_from_key_val_dict(
    key_val_dict: dict, backbone_name: str
) -> Tuple[int, int]:
    """Find the valid max and output strides from the config dictionary."""
    max_stride = key_val_dict.get(f"model.backbone.{backbone_name}.max_stride", None)
    output_stride = key_val_dict.get(
        f"model.backbone.{backbone_name}.output_stride", None
    )

    for key in [
        "model.heads.single_instance.output_stride",
        "model.heads.centered_instance.output_stride",
        "model.heads.centroid.output_stride",
        "model.heads.multi_instance.confmaps.output_stride",
        "model.heads.multi_instance.pafs.output_stride",
    ]:
        stride = key_val_dict.get(key, None)
        if stride is not None:
            stride = int(stride)
            max_stride = (
                max(int(max_stride), stride) if max_stride is not None else stride
            )
            output_stride = (
                min(int(output_stride), stride) if output_stride is not None else stride
            )

    if output_stride is None:
        output_stride = max_stride

    return max_stride, output_stride


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
