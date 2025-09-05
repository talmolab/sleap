from typing import Tuple
from omegaconf import OmegaConf, DictConfig
import sleap_io as sio


def filter_cfg(cfg):
    """Filter out keys that start with underscore to get training config."""
    for k, v in cfg.items():
        if not isinstance(v, DictConfig) and k.startswith("_"):
            del cfg[k]
        elif isinstance(v, DictConfig):
            filter_cfg(v)
    return cfg


def get_keyval_dict_from_omegaconf(cfg, parent_key="", sep="."):
    """Get a flat dictionary from an OmegaConf object."""
    items = {}
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.update(get_keyval_dict_from_omegaconf(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_omegaconf_from_gui_form(flat_dict):
    """Get an OmegaConf object from a flat dictionary."""
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return OmegaConf.create(result)


def get_backbone_from_omegaconf(cfg: OmegaConf):
    """Get the backbone model name from the config."""
    for k, v in cfg.model_config.backbone_config.items():
        if v is not None:
            return k
    return None


def get_head_from_omegaconf(cfg: OmegaConf):
    """Get the head model name from the config."""
    for k, v in cfg.model_config.head_configs.items():
        if v is not None:
            return k
    return None


def find_backbone_name_from_key_val_dict(key_val_dict: dict):
    """Find the backbone model name from the config dictionary."""
    backbone_name = None
    for key in key_val_dict:
        if key.startswith("model_config.backbone_config."):
            backbone_name = key.split(".")[2]

    return backbone_name


def resolve_strides_from_key_val_dict(
    key_val_dict: dict, backbone_name: str
) -> Tuple[int, int]:
    """Find the valid max and output strides from the config dictionary."""
    max_stride = key_val_dict.get(
        f"model_config.backbone_config.{backbone_name}.max_stride", None
    )
    output_stride = key_val_dict.get(
        f"model_config.backbone_config.{backbone_name}.output_stride", None
    )

    for key in [
        "model_config.head_configs.single_instance.confmaps.output_stride",
        "model_config.head_configs.centered_instance.confmaps.output_stride",
        "model_config.head_configs.centroid.confmaps.output_stride",
        "model_config.head_configs.bottomup.confmaps.output_stride",
        "model_config.head_configs.bottomup.pafs.output_stride",
        "model_config.head_configs.multi_class_topdown.confmaps.output_stride",
        "model_config.head_configs.multi_class_bottomup.confmaps.output_stride",
        "model_config.head_configs.multi_class_bottomup.class_maps.output_stride",
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


def apply_cfg_transforms_to_key_val_dict(key_val_dict):
    """
    Transforms data from form to correct data types before converting to object.

    Arguments:
        key_val_dict: Flat dictionary from :py:class:`TrainingEditorWidget`.
    Returns:
        None, modifies dict in place.
    """
    if "_ensure_channels" in key_val_dict:
        ensure_channels = key_val_dict["_ensure_channels"].lower()
        ensure_rgb = False
        ensure_grayscale = False
        if ensure_channels == "rgb":
            ensure_rgb = True
        elif ensure_channels == "grayscale":
            ensure_grayscale = True

        key_val_dict["data_config.preprocessing.ensure_rgb"] = ensure_rgb
        key_val_dict["data_config.preprocessing.ensure_grayscale"] = ensure_grayscale

    # Overwrite backbone strides with stride from head.
    backbone_name = find_backbone_name_from_key_val_dict(key_val_dict)
    if backbone_name is not None:
        max_stride, output_stride = resolve_strides_from_key_val_dict(
            key_val_dict, backbone_name
        )
        key_val_dict[f"model_config.backbone_config.{backbone_name}.output_stride"] = (
            output_stride
        )
        key_val_dict[f"model_config.backbone_config.{backbone_name}.max_stride"] = (
            max_stride
        )

    # batch size for val
    key_val_dict["trainer_config.val_data_loader.batch_size"] = key_val_dict[
        "trainer_config.train_data_loader.batch_size"
    ]
    key_val_dict["trainer_config.val_data_loader.num_workers"] = key_val_dict[
        "trainer_config.train_data_loader.num_workers"
    ]


def get_skeleton_from_config(skeleton_config: OmegaConf):
    """Create Sleap-io Skeleton objects from config.

    Args:
        skeleton_config: OmegaConf object containing the skeleton config.

    Returns:
        Returns a list of `sio.Skeleton` objects created from the skeleton config
        stored in the `training_config.yaml`.

    """
    skeletons = []
    for skel_cfg in skeleton_config:
        skel = sio.Skeleton(
            nodes=[n["name"] for n in skel_cfg.nodes], name=skel_cfg.name
        )
        skel.add_edges(
            [(e["source"]["name"], e["destination"]["name"]) for e in skel_cfg.edges]
        )
        if skel_cfg.symmetries:
            for n1, n2 in skel_cfg.symmetries:
                skel.add_symmetry(n1["name"], n2["name"])

        skeletons.append(skel)

    return skeletons
