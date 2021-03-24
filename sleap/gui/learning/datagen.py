"""
Preview of training data (i.e., confidence maps and part affinity fields).
"""

import copy
from collections import defaultdict
from typing import List, Text

import numpy as np

from sleap import Labels, Video
from sleap.gui.learning.configs import ConfigFileInfo
from sleap.gui.overlays.confmaps import demo_confmaps
from sleap.gui.overlays.pafs import demo_pafs
from sleap.nn.config import (
    TrainingJobConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfig,
)
from sleap.nn.data import pipelines
from sleap.nn.data.instance_cropping import find_instance_crop_size
from sleap.nn.data.providers import LabelsReader
from sleap.nn.data.resizing import Resizer


MAX_FRAMES_TO_PREVIEW = 20


def show_datagen_preview(labels: Labels, config_info_list: List[ConfigFileInfo]):
    """
    Shows window(s) with preview images of training data for model configs.
    """

    labels_reader = LabelsReader.from_user_instances(labels)

    win_x = 300

    def show_win(
        results: dict, key: Text, head_name: Text, video: Video, scale_to_height=None
    ):
        nonlocal win_x

        scale = None
        if scale_to_height:
            overlay_height = results[key].shape[1]  # frames, height, width, channels
            scale = scale_to_height // overlay_height

        if key == "confmap":
            win = demo_confmaps(results[key], video, scale=scale)
        elif key == "paf":
            win = demo_pafs(results[key], video, scale=scale, decimation=2)
        else:
            raise ValueError(f"Cannot show preview window for {key}")

        win.activateWindow()
        win.setWindowTitle(f"{head_name} {key}")
        win.resize(400, 400)
        win.move(win_x, 300)
        win_x += 420

    for cfg_info in config_info_list:
        results = make_datagen_results(labels_reader, cfg_info.config)

        if "image" in results:
            vid = Video.from_numpy(results["image"])

            if "confmap" in results:
                show_win(
                    results,
                    "confmap",
                    cfg_info.head_name,
                    vid,
                    scale_to_height=vid.height,
                )

            if "paf" in results:
                show_win(
                    results, "paf", cfg_info.head_name, vid, scale_to_height=vid.height
                )


def make_datagen_results(reader: LabelsReader, cfg: TrainingJobConfig) -> np.ndarray:
    """
    Gets (subset of) raw images used for training.

    TODO: Refactor so we can get this data without digging into details of the
      the specific pipelines (e.g., key for confmaps depends on head type).
    """
    cfg = copy.deepcopy(cfg)
    output_keys = dict()

    if cfg.data.preprocessing.pad_to_stride is None:
        cfg.data.preprocessing.pad_to_stride = (
            cfg.model.backbone.which_oneof().max_stride
        )

    pipeline = pipelines.Pipeline(reader)
    pipeline += Resizer.from_config(cfg.data.preprocessing)

    head_config = cfg.model.heads.which_oneof()
    if isinstance(head_config, CentroidsHeadConfig):
        pipeline += pipelines.InstanceCentroidFinder.from_config(
            cfg.data.instance_cropping, skeletons=reader.labels.skeletons
        )
        pipeline += pipelines.MultiConfidenceMapGenerator(
            sigma=cfg.model.heads.centroid.sigma,
            output_stride=cfg.model.heads.centroid.output_stride,
            centroids=True,
        )

        output_keys["image"] = "image"
        output_keys["confmap"] = "centroid_confidence_maps"

    elif isinstance(head_config, CenteredInstanceConfmapsHeadConfig):
        if cfg.data.instance_cropping.crop_size is None:
            cfg.data.instance_cropping.crop_size = find_instance_crop_size(
                labels=reader.labels,
                padding=cfg.data.instance_cropping.crop_size_detection_padding,
                maximum_stride=cfg.model.backbone.which_oneof().max_stride,
            )

        pipeline += pipelines.InstanceCentroidFinder.from_config(
            cfg.data.instance_cropping, skeletons=reader.labels.skeletons
        )
        pipeline += pipelines.InstanceCropper.from_config(cfg.data.instance_cropping)
        pipeline += pipelines.InstanceConfidenceMapGenerator(
            sigma=cfg.model.heads.centered_instance.sigma,
            output_stride=cfg.model.heads.centered_instance.output_stride,
        )

        output_keys["image"] = "instance_image"
        output_keys["confmap"] = "instance_confidence_maps"

    elif isinstance(head_config, MultiInstanceConfig):
        output_keys["image"] = "image"
        output_keys["confmap"] = "confidence_maps"
        output_keys["paf"] = "part_affinity_fields"

        pipeline += pipelines.MultiConfidenceMapGenerator(
            sigma=cfg.model.heads.multi_instance.confmaps.sigma,
            output_stride=cfg.model.heads.multi_instance.confmaps.output_stride,
        )
        pipeline += pipelines.PartAffinityFieldsGenerator(
            sigma=cfg.model.heads.multi_instance.pafs.sigma,
            output_stride=cfg.model.heads.multi_instance.pafs.output_stride,
            skeletons=reader.labels.skeletons,
            flatten_channels=True,
        )

    ds = pipeline.make_dataset()

    output_lists = defaultdict(list)
    i = 0
    for example in ds:
        for key, from_key in output_keys.items():
            output_lists[key].append(example[from_key])
        i += 1
        if i == MAX_FRAMES_TO_PREVIEW:
            break

    outputs = dict()
    for key in output_lists.keys():
        outputs[key] = np.stack(output_lists[key])

    return outputs
