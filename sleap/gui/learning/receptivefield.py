"""
Widget for previewing receptive field on sample image using model hyperparams.
"""

from typing import Optional, Text, Tuple
import math

import sleap_io as sio
import numpy as np
from qtpy import QtWidgets, QtGui, QtCore
from omegaconf import OmegaConf
from sleap.gui.widgets.video import GraphicsView
from sleap.gui.config_utils import get_head_from_omegaconf
from sleap.gui.learning import unet_utils


def compute_rf(down_blocks: int, convs_per_block: int = 2, kernel_size: int = 3) -> int:
    """
    Computes receptive field for specified model architecture.

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


def receptive_field_info_from_model_cfg(cfg: OmegaConf) -> dict:
    """Gets receptive field and architecture information from model configuration.

    Returns a dict with:
        - size: Receptive field size in pixels
        - max_stride: Maximum stride (bottleneck)
        - down_blocks: Number of encoder downsampling blocks
        - convs_per_block: Convolutions per block (fixed at 2)
        - kernel_size: Convolution kernel size (fixed at 3)
        - output_stride: Minimum head output stride (for RF calculation)
        - params: Total backbone parameter count
        - params_formatted: Human-readable param count (e.g., "1.30M")
        - head_features: List of (head_name, output_stride, channels) for each head
        - backbone_type: Type of backbone (unet, convnext, swint)
        - model_type: For convnext/swint, the model variant (tiny, small, base, large)
    """
    model_cfg = cfg.model_config
    backbone_config = model_cfg.backbone_config

    rf_info = dict(
        size=None,
        max_stride=None,
        down_blocks=None,
        convs_per_block=None,
        kernel_size=None,
        output_stride=None,
        params=None,
        params_formatted=None,
        head_features=[],  # List of (head_name, output_stride, channels)
        backbone_type=None,  # e.g., "unet", "convnext", "swint"
        model_type=None,  # For convnext/swint: "tiny", "small", "base", "large"
    )

    # Detect backbone type
    backbone_type = None
    for bt in ["unet", "convnext", "swint"]:
        if hasattr(backbone_config, bt) and getattr(backbone_config, bt) is not None:
            backbone_type = bt
            break

    rf_info["backbone_type"] = backbone_type

    if backbone_type is None:
        return rf_info

    # Get max_stride based on backbone type
    if backbone_type == "unet":
        backbone_config.unet.max_stride = int(backbone_config.unet.max_stride)
        max_stride = backbone_config.unet.max_stride
    else:
        # ConvNeXt and SwinT have fixed max_stride of 32
        max_stride = 32

    rf_info["max_stride"] = max_stride

    head_type = get_head_from_omegaconf(cfg)

    # Collect output strides for each sub-head
    head_output_strides = []  # List of (sub_head_name, output_stride)
    for k, head_cfg in model_cfg.head_configs[head_type].items():
        if k == "class_vectors":
            head_output_strides.append((k, int(max_stride)))
        else:
            head_output_strides.append((k, int(head_cfg.output_stride)))

    output_strides = [s for _, s in head_output_strides]
    output_stride = min(output_strides)
    rf_info["output_stride"] = output_stride

    # Handle backbone-specific RF calculations
    if backbone_type == "unet":
        try:
            _ = np.log2(max_stride / output_stride)
        except ZeroDivisionError:
            return rf_info

        rf_info["convs_per_block"] = 2
        rf_info["kernel_size"] = 3

        stem_stride = None
        stem_blocks = 0
        if hasattr(backbone_config.unet, "stem_stride"):
            cfg_stem_stride = backbone_config.unet.stem_stride
            if cfg_stem_stride is not None:
                stem_stride = int(cfg_stem_stride)
                stem_blocks = np.log2(cfg_stem_stride).astype(int)

        down_blocks = np.log2(max_stride).astype(int) - stem_blocks
        rf_info["down_blocks"] = down_blocks

        has_rf_params = (
            rf_info["down_blocks"]
            and rf_info["convs_per_block"]
            and rf_info["kernel_size"]
        )
        if has_rf_params:
            rf_info["size"] = compute_rf(
                down_blocks=rf_info["down_blocks"],
                convs_per_block=rf_info["convs_per_block"],
                kernel_size=rf_info["kernel_size"],
            )

        # Extract UNet config for architecture calculations
        unet_cfg = backbone_config.unet
        filters = int(getattr(unet_cfg, "filters", 32))
        filters_rate = float(getattr(unet_cfg, "filters_rate", 1.5))
        middle_block = bool(getattr(unet_cfg, "middle_block", True))
        up_interpolate = bool(getattr(unet_cfg, "up_interpolate", False))

        # Compute channel counts at each stride
        try:
            min_output_stride = min(s for _, s in head_output_strides)
            stride_to_channels = unet_utils.compute_unet_channels(
                filters=filters,
                filters_rate=filters_rate,
                max_stride=max_stride,
                output_stride=min_output_stride,
                stem_stride=stem_stride,
            )
            for head_name, head_stride in head_output_strides:
                channels = stride_to_channels.get(head_stride)
                if channels is not None:
                    rf_info["head_features"].append((head_name, head_stride, channels))
        except Exception:
            pass

        # Compute total parameter count
        try:
            params = unet_utils.compute_unet_params(
                filters=filters,
                filters_rate=filters_rate,
                max_stride=max_stride,
                output_stride=output_stride,
                stem_stride=stem_stride,
                middle_block=middle_block,
                up_interpolate=up_interpolate,
            )
            rf_info["params"] = params
            rf_info["params_formatted"] = unet_utils.format_params(params)
        except Exception:
            pass

    elif backbone_type in ("convnext", "swint"):
        # ConvNeXt and SwinT have fixed max_stride=32
        rf_info["down_blocks"] = 5  # log2(32) = 5

        # Get model type (tiny, small, base, large)
        backbone_cfg = getattr(backbone_config, backbone_type)
        model_type = getattr(backbone_cfg, "model_type", "tiny")
        rf_info["model_type"] = model_type

        # Approximate parameter counts for pretrained models
        # These are rough estimates based on torchvision model sizes
        if backbone_type == "convnext":
            param_counts = {
                "tiny": 28_600_000,
                "small": 50_200_000,
                "base": 88_600_000,
                "large": 197_800_000,
            }
        else:  # swint
            param_counts = {
                "tiny": 28_300_000,
                "small": 49_600_000,
                "base": 87_800_000,
            }

        params = param_counts.get(model_type)
        if params:
            rf_info["params"] = params
            rf_info["params_formatted"] = unet_utils.format_params(params)

        # RF size is architecture-dependent and complex for transformer-based models
        # For now, leave as None since exact RF calculation is non-trivial

    return rf_info


#: Centroid derivation methods, mirroring `sleap_nn.data.instance_centroids`
#: (and `sleap_io`'s `Instance.to_centroid`).
CENTROID_METHODS = ("center_of_mass", "bbox_center", "geometric_median", "anchor")

#: The subset that reduces a whole point set, i.e. the valid anchor fallbacks.
REDUCE_METHODS = ("center_of_mass", "bbox_center", "geometric_median")

#: Fallback used when a configured anchor node is not visible on an instance.
#: sleap-nn always falls back rather than yielding a NaN centroid.
DEFAULT_ANCHOR_FALLBACK = "center_of_mass"

#: `min_crop_size` default from `sleap_nn.config.data_config.PreprocessingConfig`.
#: The GUI form does not expose the field, but training applies it as a floor, so
#: the preview has to as well or it will under-report small-animal projects.
DEFAULT_MIN_CROP_SIZE = 100


def compute_eff_scale(
    img_hw: Tuple[int, int],
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> float:
    """Return the scale the size matcher would apply to a frame of this size.

    Mirrors `sleap_nn.data.resizing.compute_eff_scale`: the size matcher fits
    each frame into ``(max_height, max_width)`` preserving aspect ratio, so the
    scale is the *smaller* of the two ratios.

    Args:
        img_hw: The frame's ``(height, width)`` in its native resolution.
        max_hw: The configured ``(max_height, max_width)``. ``None``, or a
            ``None`` in either slot, leaves the frame at its native size on
            that axis.

    Returns:
        The scale factor, ``1.0`` when the frame already matches the target.
    """
    img_height, img_width = img_hw
    max_height, max_width = (None, None) if max_hw is None else max_hw
    if max_height is None:
        max_height = img_height
    if max_width is None:
        max_width = img_width
    if img_height == max_height and img_width == max_width:
        return 1.0
    return min(max_height / img_height, max_width / img_width)


def _frame_eff_scale(
    lf: sio.LabeledFrame,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]],
) -> float:
    """Return the size-matcher scale that will be applied to a labeled frame.

    Args:
        lf: The labeled frame, whose video supplies the native resolution.
        max_hw: The configured ``(max_height, max_width)``, or ``None`` to skip
            size matching entirely.

    Returns:
        The scale factor, ``1.0`` when size matching does not apply or the
        video's resolution cannot be determined.
    """
    if max_hw is None:
        return 1.0
    try:
        shape = lf.video.shape
    except Exception:
        # Reading `shape` opens the backend, which fails for a missing or
        # unreadable video. A preview should degrade, not crash.
        shape = None
    if shape is None or len(shape) < 3:
        return 1.0
    return compute_eff_scale((int(shape[1]), int(shape[2])), max_hw)


def resolve_max_hw(
    labels: Optional[sio.Labels],
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Resolve the size-matcher target the way the trainer's first pass does.

    `ModelTrainer._setup_preprocessing_config` fills in an unset
    ``max_height``/``max_width`` from the maximum over every training video, so
    a project with mixed resolutions is size-matched whether or not the user
    typed a target. The GUI form does not expose these fields at all, which
    means leaving them unresolved would measure crops in native pixels and
    reproduce the bug sleap-nn #748 fixed.

    Args:
        labels: The labels whose videos supply the fallback target.
        max_height: The configured max height, or ``None`` to derive it.
        max_width: The configured max width, or ``None`` to derive it.

    Returns:
        ``(max_height, max_width)``, or ``None`` if neither was configured and
        no video resolution could be read (in which case no size matching can
        be predicted and native pixels are the best available answer).
    """
    if max_height is not None and max_width is not None:
        return int(max_height), int(max_width)

    auto_h, auto_w = 0, 0
    for video in getattr(labels, "videos", None) or []:
        try:
            shape = video.shape
        except Exception:
            shape = None
        if shape is None or len(shape) < 3:
            continue
        auto_h = max(auto_h, int(shape[1]))
        auto_w = max(auto_w, int(shape[2]))

    height = max_height if max_height is not None else auto_h
    width = max_width if max_width is not None else auto_w
    if not height or not width:
        return None
    return int(height), int(width)


def _iter_scaled_instances(
    labels: sio.Labels,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]],
    user_instances_only: bool,
):
    """Yield every non-empty instance's points, in size-matched pixels.

    The scale is memoized per video: `sio.Video.shape` opens the video backend,
    so reading it once per labeled frame (as sleap-nn's `_iter_frame_instances`
    does) would hit the disk on every frame of the project.

    Args:
        labels: A `sio.Labels` to walk.
        max_hw: The configured ``(max_height, max_width)``, or ``None`` for no
            size matching (native pixels).
        user_instances_only: When ``True``, skip `sio.PredictedInstance`s. This
            must match ``data_config.user_instances_only``, or the crop size is
            derived from instances the model is never trained on.

    Yields:
        Each instance's ``(n_nodes, 2)`` point array, scaled into the space the
        crop is taken in.
    """
    scales = {}
    for lf in labels:
        key = id(lf.video)
        if key not in scales:
            scales[key] = _frame_eff_scale(lf, max_hw)
        eff_scale = scales[key]
        for inst in lf.instances:
            if user_instances_only and isinstance(inst, sio.PredictedInstance):
                continue
            if inst.is_empty:  # every point NaN
                continue
            yield inst.numpy() * eff_scale


def _points_mean(pts: np.ndarray) -> np.ndarray:
    """Mean of the visible points, counting non-NaN values per axis.

    Per-axis counting matches `sleap_nn.data.instance_centroids.find_points_mean`
    (sleap-nn #584); it differs from a per-point count only for a half-NaN
    point, which `sleap_io` does not write.

    Args:
        pts: An ``(n_nodes, 2)`` point array.

    Returns:
        The ``(2,)`` mean, or all-NaN when no point is visible on either axis.
    """
    mask = ~np.isnan(pts)
    counts = np.clip(mask.sum(axis=0), 1, None)
    means = np.where(mask, pts, 0.0).sum(axis=0) / counts
    if not mask.any():
        return np.full(2, np.nan)
    return means


def _points_bbox_midpoint(pts: np.ndarray) -> np.ndarray:
    """Midpoint of the visible points' bounding box.

    Args:
        pts: An ``(n_nodes, 2)`` point array.

    Returns:
        The ``(2,)`` midpoint, or all-NaN when no point is visible.
    """
    if np.isnan(pts).all():
        return np.full(2, np.nan)
    return (np.nanmin(pts, axis=0) + np.nanmax(pts, axis=0)) * 0.5


def _reduce_points(pts: np.ndarray, method: str) -> np.ndarray:
    """Reduce a point set to one centroid by the named method.

    Args:
        pts: An ``(n_nodes, 2)`` point array.
        method: One of `REDUCE_METHODS`.

    Returns:
        The ``(2,)`` centroid.

    Raises:
        ValueError: If ``method`` is not a known reduce method.
    """
    if method == "center_of_mass":
        return _points_mean(pts)
    if method == "bbox_center":
        return _points_bbox_midpoint(pts)
    if method == "geometric_median":
        # Reuse sleap-io's Weiszfeld implementation rather than keeping a second
        # copy here: it is pure numpy (no torch, so no GUI startup cost), and it
        # is the routine sleap-nn's batched `find_points_geometric_median` is
        # asserted equal to.
        from sleap_io.model.centroid import _geometric_median

        visible = pts[~np.isnan(pts).any(axis=1)]
        if not len(visible):
            return np.full(2, np.nan)
        return np.asarray(_geometric_median(visible), dtype=float)
    message = (
        f"Unknown centroid reduce method {method!r}. Expected one of "
        f"{', '.join(repr(m) for m in REDUCE_METHODS)}."
    )
    raise ValueError(message)


def _instance_centroid(
    pts: np.ndarray,
    anchor_ind: Optional[int] = None,
    method: Optional[str] = None,
    fallback: Optional[str] = None,
) -> np.ndarray:
    """Return the centroid a crop would be centered on.

    The single-instance numpy twin of
    `sleap_nn.data.instance_centroids.generate_centroids`, which is the op that
    positions the crop at training time.

    Args:
        pts: An ``(n_nodes, 2)`` point array.
        anchor_ind: Index of the anchor node, or ``None`` for a reduce method.
        method: One of `CENTROID_METHODS`, or ``None`` to infer from
            ``anchor_ind`` (the pre-#586 behavior).
        fallback: The reduce method used when the anchor node is not visible.

    Returns:
        The ``(2,)`` centroid, all-NaN when it is undefined.
    """
    if method is None:
        method = "anchor" if anchor_ind is not None else "center_of_mass"
    if method != "anchor":
        return _reduce_points(pts, method)
    if anchor_ind is None or anchor_ind >= len(pts):
        # An unresolvable anchor degrades to its fallback, as
        # `degrade_anchor_if_unresolved` does in sleap-nn.
        return _reduce_points(pts, fallback or DEFAULT_ANCHOR_FALLBACK)
    anchor = pts[anchor_ind]
    if np.isnan(anchor).any():
        return _reduce_points(pts, fallback or DEFAULT_ANCHOR_FALLBACK)
    return np.asarray(anchor, dtype=float)


def resolve_centroid_method(
    anchor_part: Optional[Text] = None,
    centroid_method: Optional[Text] = None,
    centroid_fallback: Optional[Text] = None,
) -> Tuple[Text, Optional[Text]]:
    """Resolve the head config's centroid knobs into ``(method, fallback)``.

    Mirrors `sleap_nn.data.instance_centroids.resolve_centroid_method`, minus
    its validation: this is a preview, so an unrecognized value degrades to the
    historical default instead of raising over a config the training run will
    report on properly.

    Args:
        anchor_part: The configured anchor node name, or ``None``. Setting it
            implies ``method="anchor"``.
        centroid_method: One of `CENTROID_METHODS`, or ``None`` to infer.
        centroid_fallback: The reduce method for a non-visible anchor node.

    Returns:
        ``(method, fallback)``, where ``fallback`` is only meaningful for the
        anchor method.
    """
    if centroid_method is not None and centroid_method not in CENTROID_METHODS:
        centroid_method = None
    if centroid_fallback is not None and centroid_fallback not in REDUCE_METHODS:
        centroid_fallback = None

    if anchor_part is not None:
        return "anchor", centroid_fallback or DEFAULT_ANCHOR_FALLBACK
    if centroid_method == "anchor":
        # An anchor method without a node to anchor on; sleap-nn raises, the
        # preview degrades.
        return DEFAULT_ANCHOR_FALLBACK, None
    return (centroid_method or "center_of_mass"), None


def resolve_anchor_ind(
    labels: Optional[sio.Labels], anchor_part: Optional[Text]
) -> Optional[int]:
    """Resolve an anchor node name to its index in the labels' skeleton.

    Args:
        labels: The labels whose first skeleton the name is resolved against,
            as `ModelTrainer._resolve_crop_centroid` does.
        anchor_part: The node name, or ``None``.

    Returns:
        The node index, or ``None`` when there is no anchor or it does not name
        a node in the skeleton.
    """
    if not anchor_part or labels is None:
        return None
    skeletons = getattr(labels, "skeletons", None) or []
    if not skeletons:
        return None
    names = skeletons[0].node_names
    return names.index(anchor_part) if anchor_part in names else None


def find_max_instance_bbox_size(
    labels: sio.Labels,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    user_instances_only: bool = True,
) -> float:
    """Find the maximum bounding box dimension across all instances in labels.

    This is a local implementation that avoids importing sleap_nn (which would
    trigger importing torch, adding ~2s to startup time). It mirrors
    `sleap_nn.data.instance_cropping.find_max_instance_bbox_size`, except that
    ``user_instances_only`` defaults to ``True`` here: the GUI form has no such
    field, and training's config default is ``True``.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        max_hw: The configured ``(max_height, max_width)``. When given, each
            instance is measured in the size-matched space it will actually be
            cropped in, rather than in its video's native pixels.
        user_instances_only: When ``True`` (default), ignore predicted
            instances.

    Returns:
        The maximum bounding box dimension (max of width or height) across all
        instances.
    """
    max_length = 0.0
    for pts in _iter_scaled_instances(labels, max_hw, user_instances_only):
        diff_x = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
        diff_x = 0 if np.isnan(diff_x) else diff_x
        max_length = np.maximum(max_length, diff_x)
        diff_y = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
        diff_y = 0 if np.isnan(diff_y) else diff_y
        max_length = np.maximum(max_length, diff_y)
    return float(max_length)


def iter_required_crop_sizes(
    labels: sio.Labels,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    anchor_ind: Optional[int] = None,
    centroid_method: Optional[Text] = None,
    centroid_fallback: Optional[Text] = None,
    user_instances_only: bool = True,
):
    """Yield the crop size each labeled instance needs to avoid being clipped.

    Mirrors `sleap_nn.data.instance_cropping.iter_required_crop_sizes`. Crops
    are **centered on the instance's centroid**, not on its bounding-box
    midpoint, so the size an instance requires is twice its greatest node
    offset from that centroid -- *not* its bounding-box extent. The two agree
    only when the centroid happens to sit mid-bbox; for an anchor near one end
    of the animal the required size approaches twice the extent.

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        max_hw: The configured ``(max_height, max_width)``. When given, points
            are measured in the size-matched space the crop is taken in.
        anchor_ind: Index of the anchor node, or ``None`` for a reduce method.
        centroid_method: The resolved centroid method, or ``None`` to infer
            from ``anchor_ind``.
        centroid_fallback: The reduce method used when the anchor node is not
            visible on an instance.
        user_instances_only: When ``True`` (default), ignore predicted
            instances.

    Yields:
        The required crop side length, in size-matched pixels, per instance.
        Instances whose centroid is undefined are skipped.
    """
    for scaled in _iter_scaled_instances(labels, max_hw, user_instances_only):
        centroid = _instance_centroid(
            scaled,
            anchor_ind=anchor_ind,
            method=centroid_method,
            fallback=centroid_fallback,
        )
        if np.isnan(centroid).any():
            continue
        # Largest per-axis offset from the crop center; the crop spans
        # centroid +/- size/2 on each axis, so it must be twice this to reach it.
        offsets = np.abs(scaled - centroid)
        reach = np.nan_to_num(offsets, nan=0.0).max()
        yield float(2.0 * reach)


def find_instance_crop_size(
    labels: sio.Labels,
    padding: int = 0,
    maximum_stride: int = 2,
    min_crop_size: Optional[int] = None,
    max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    anchor_ind: Optional[int] = None,
    centroid_method: Optional[Text] = None,
    centroid_fallback: Optional[Text] = None,
    user_instances_only: bool = True,
) -> int:
    """Compute a crop size that contains every labeled instance.

    This is a local implementation that avoids importing sleap_nn (which would
    trigger importing torch, adding ~2s to startup time). It mirrors
    `sleap_nn.data.instance_cropping.find_instance_crop_size`: the size is
    measured the way the crop is actually taken -- centered on each instance's
    centroid (`iter_required_crop_sizes`) and, when ``max_hw`` is given, in the
    size-matched pixel space cropping happens in. Both matter; a bounding-box
    measurement in native pixels under-sizes the crop whenever the centroid is
    off-center or the video is rescaled to ``max_hw`` (sleap-nn #748).

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        padding: Integer number of pixels to add to the bounds as margin
            padding.
        maximum_stride: Ensure that the returned crop size is divisible by this
            value. Useful for ensuring that the crop size will not be truncated
            in a given architecture.
        min_crop_size: A floor for the returned crop size, before padding.
        max_hw: The configured ``(max_height, max_width)``, so instances are
            measured in the space the size matcher puts them in. ``None``
            measures native pixels.
        anchor_ind: Index of the anchor node the crop is centered on, or
            ``None`` for a reduce method.
        centroid_method: The resolved centroid method, or ``None`` to infer
            from ``anchor_ind``.
        centroid_fallback: The reduce method used when the anchor node is not
            visible on an instance.
        user_instances_only: When ``True`` (default), ignore predicted
            instances.

    Returns:
        An integer crop size denoting the length of the side of the bounding
        boxes that will contain the instances when cropped. The returned crop
        size will be larger or equal to the input `min_crop_size`.
    """
    min_crop_size = 0 if min_crop_size is None else min_crop_size

    # `min_crop_size` is a floor, applied before padding is added -- and outside
    # the loop, so it still applies to a project with nothing to measure.
    max_length = float(min_crop_size - padding)
    for required in iter_required_crop_sizes(
        labels,
        max_hw=max_hw,
        anchor_ind=anchor_ind,
        centroid_method=centroid_method,
        centroid_fallback=centroid_fallback,
        user_instances_only=user_instances_only,
    ):
        max_length = max(max_length, required)

    max_length = max(max_length, 0.0) + float(padding)
    crop_size = math.ceil(max_length / float(maximum_stride)) * maximum_stride

    return int(crop_size)


def compute_augmentation_padding(
    bbox_size: float,
    rotation_max: float = 0.0,
    scale_max: float = 1.0,
) -> int:
    """Compute padding needed to accommodate augmentation transforms.

    A local copy of `sleap_nn.data.instance_cropping.compute_augmentation_padding`
    -- the sleap_nn module imports torch, which would add ~2s the first time the
    crop preview updated.

    Args:
        bbox_size: The size of the instance bounding box (max of width/height).
        rotation_max: Maximum absolute rotation angle in degrees. For symmetric
            rotation ranges like [-180, 180], pass 180.
        scale_max: Maximum scaling factor. For scale range [0.9, 1.1], pass 1.1.

    Returns:
        Padding in pixels to add around the bounding box (total, not per side).
    """
    if rotation_max == 0.0 and scale_max <= 1.0:
        return 0

    # For a square bbox rotated by angle t, the new bbox has side length
    # L' = L * (|cos(t)| + |sin(t)|), maximized at 45 degrees (L * sqrt(2)).
    rotation_rad = math.radians(min(abs(rotation_max), 90))
    rotation_factor = abs(math.cos(rotation_rad)) + abs(math.sin(rotation_rad))
    if abs(rotation_max) >= 45:
        # Any range that includes 45 degrees hits the worst case.
        rotation_factor = math.sqrt(2)

    expansion_factor = rotation_factor * max(scale_max, 1.0)
    return int(math.ceil(bbox_size * expansion_factor - bbox_size))


def _compute_padding_from_aug_form(aug_form_data: dict, bbox_size: float) -> int:
    """Compute augmentation padding from the augmentation form's virtual fields.

    The augmentation form uses virtual fields (_rotation_preset, _scale_enabled)
    rather than the raw config keys (rotation_p, rotation_min, etc.). This function
    translates those into the rotation/scale values needed by
    `compute_augmentation_padding`.

    Args:
        aug_form_data: Dict from form_widgets["augmentation"].get_form_data().
        bbox_size: Max bounding box dimension from user-labeled instances,
            measured in the size-matched space and floored at the configured
            `min_crop_size`, as `ModelTrainer._compute_crop_padding` does.

    Returns:
        Padding in pixels, or 0 if no geometric augmentations are enabled.
    """
    preset_to_angle = {"Off": 0.0, "±15°": 15.0, "±180°": 180.0}
    preset = aug_form_data.get("_rotation_preset", "Off")
    if preset == "Custom":
        rot_max = float(aug_form_data.get("_rotation_custom_angle", 0) or 0)
    else:
        rot_max = preset_to_angle.get(preset, 0.0)

    scale_enabled = aug_form_data.get("_scale_enabled", False)
    s_max = (
        float(
            aug_form_data.get(
                "data_config.augmentation_config.geometric.scale_max", 1.0
            )
            or 1.0
        )
        if scale_enabled
        else 1.0
    )

    return compute_augmentation_padding(
        bbox_size, rotation_max=rot_max, scale_max=s_max
    )


#: Head-config leaves that carry the centroid knobs, for the model types that
#: crop. Mirrors the `leaf_paths` map in `ModelTrainer._resolve_crop_centroid`.
CROP_HEAD_LEAVES = (
    "centered_instance.confmaps",
    "multi_class_topdown.confmaps",
    "centered_instance_segmentation.segmentation",
)


def resolve_crop_centroid(
    model_cfg: OmegaConf,
    labels: Optional[sio.Labels] = None,
) -> Tuple[Optional[int], Optional[Text], Text, Optional[Text]]:
    """Resolve the centroid definition the crops will be centered on.

    The GUI twin of `ModelTrainer._resolve_crop_centroid`. The crop center is
    what makes a crop size sufficient or not, so the preview has to resolve it
    the same way training does. Resolution is lenient: an ``anchor_part`` that
    does not name a node in the skeleton degrades to its fallback rather than
    raising, which is what training does too (`degrade_anchor_if_unresolved`).

    The head is identified by which leaf actually carries the knobs rather than
    by `get_head_from_omegaconf`, so a partial config -- which is what the form
    hands over while the user is still filling it in -- resolves instead of
    raising.

    Args:
        model_cfg: Model configuration OmegaConf from the model form.
        labels: Labels whose first skeleton the anchor name is resolved against.

    Returns:
        ``(anchor_ind, anchor_part, method, fallback)``. ``anchor_part`` is
        ``None`` unless the anchor actually took effect, so a caller can report
        the center that is really in use.
    """
    anchor_part = centroid_method = centroid_fallback = None
    for leaf in CROP_HEAD_LEAVES:
        base = f"model_config.head_configs.{leaf}"
        if OmegaConf.select(model_cfg, base, default=None) is None:
            continue
        anchor_part = OmegaConf.select(model_cfg, f"{base}.anchor_part", default=None)
        centroid_method = OmegaConf.select(
            model_cfg, f"{base}.centroid_method", default=None
        )
        centroid_fallback = OmegaConf.select(
            model_cfg, f"{base}.centroid_fallback", default=None
        )
        break

    method, fallback = resolve_centroid_method(
        anchor_part=anchor_part,
        centroid_method=centroid_method,
        centroid_fallback=centroid_fallback,
    )
    anchor_ind = resolve_anchor_ind(labels, anchor_part)
    if anchor_ind is None:
        # The configured anchor is unresolvable, so it is not the center in
        # force; degrade to the fallback and stop naming the node.
        if method == "anchor":
            method, fallback = (fallback or DEFAULT_ANCHOR_FALLBACK), None
        anchor_part = None
    return anchor_ind, anchor_part, method, fallback


def resolve_crop_inputs(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    labels: Optional[sio.Labels] = None,
) -> dict:
    """Resolve everything the crop-size computation needs from the GUI forms.

    Three of these are not on the training form at all (``max_height`` /
    ``max_width``, ``min_crop_size``, ``user_instances_only``), but training
    resolves them from its config defaults and applies them, so the preview
    reads them where present and falls back to the same defaults.

    Args:
        data_cfg: Data configuration OmegaConf from the data form.
        model_cfg: Model configuration OmegaConf from the model form.
        labels: Labels used to resolve the size-matcher target and the anchor.

    Returns:
        A dict of keyword arguments for `find_instance_crop_size` /
        `count_clipped_instances`, plus ``anchor_part`` for reporting.
    """
    pre = "data_config.preprocessing"
    max_hw = resolve_max_hw(
        labels,
        max_height=OmegaConf.select(data_cfg, f"{pre}.max_height", default=None),
        max_width=OmegaConf.select(data_cfg, f"{pre}.max_width", default=None),
    )
    min_crop_size = OmegaConf.select(data_cfg, f"{pre}.min_crop_size", default=None)
    if min_crop_size is None:
        min_crop_size = DEFAULT_MIN_CROP_SIZE
    user_instances_only = OmegaConf.select(
        data_cfg, "data_config.user_instances_only", default=True
    )
    anchor_ind, anchor_part, method, fallback = resolve_crop_centroid(model_cfg, labels)
    return dict(
        max_hw=max_hw,
        min_crop_size=int(min_crop_size),
        user_instances_only=bool(user_instances_only),
        anchor_ind=anchor_ind,
        anchor_part=anchor_part,
        centroid_method=method,
        centroid_fallback=fallback,
    )


def compute_crop_size_from_cfg(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    labels: Optional[sio.Labels] = None,
    aug_form_data: Optional[dict] = None,
) -> Optional[int]:
    """Computes crop size from model configuration.

    When crop_size is not set (None/auto), computes it the way
    `ModelTrainer._setup_preprocessing_config` does: from the reach of every
    labeled instance around the centroid its crop is centered on, measured in
    the size-matched pixel space cropping happens in, floored at
    ``min_crop_size``, plus augmentation padding, rounded up to the backbone's
    max stride.

    Two differences from training remain, and neither is resolvable from the
    dialog: the crop size is sized here from *all* labeled frames rather than
    from the training split (the split is drawn at training time, from a seed),
    and only the first labels file is considered.

    Args:
        data_cfg: Data configuration OmegaConf from the data form.
        model_cfg: Model configuration OmegaConf from the model form.
        labels: Labels object for computing instance bounding boxes.
        aug_form_data: Raw dict from the augmentation form's get_form_data(),
            used to compute augmentation padding from virtual fields.

    Returns:
        The network's input crop size in pixels -- the crop size in size-matched
        pixels, scaled by ``preprocessing.scale``, which is what training
        reports as the network input. ``None`` if it cannot be determined.
    """
    crop_size = data_cfg.data_config.preprocessing.crop_size
    if crop_size is None:
        try:
            backbone = model_cfg["_backbone_name"]
            max_stride = int(
                model_cfg.model_config.backbone_config[backbone].max_stride
            )

            inputs = resolve_crop_inputs(data_cfg, model_cfg, labels)
            inputs.pop("anchor_part")

            padding = 0
            if aug_form_data:
                # Training floors the bbox the padding is derived from at
                # `min_crop_size` before expanding it
                # (`ModelTrainer._compute_crop_padding`).
                bbox_size = max(
                    find_max_instance_bbox_size(
                        labels,
                        max_hw=inputs["max_hw"],
                        user_instances_only=inputs["user_instances_only"],
                    ),
                    inputs["min_crop_size"],
                )
                padding = _compute_padding_from_aug_form(aug_form_data, bbox_size)

            crop_size = find_instance_crop_size(
                labels, padding=padding, maximum_stride=max_stride, **inputs
            )
        except Exception:
            crop_size = None
    if crop_size is not None and data_cfg.data_config.preprocessing.scale is not None:
        crop_size = int(crop_size * data_cfg.data_config.preprocessing.scale)
    return crop_size


def get_first_labeled_frame_and_instance(
    labels: Optional[sio.Labels],
) -> Tuple[Optional[np.ndarray], Optional[sio.Instance], Optional[sio.Video]]:
    """Gets the first frame with ground truth labels and the first instance.

    Args:
        labels: The Labels object containing labeled frames.

    Returns:
        A tuple of (frame_image, instance, video) where frame_image is a numpy
        array, instance is the first user instance and video is the video the
        frame was read from -- needed to know the pixel space the image is in,
        since crop sizes are measured after size matching. Returns
        (None, None, None) if no labeled frames with user instances are found.
    """
    if labels is None:
        return None, None, None

    for lf in labels:
        if lf.user_instances:
            # Get the first user instance
            instance = lf.user_instances[0]
            video = lf.video if hasattr(lf, "video") else labels.videos[0]
            # Get the frame image using sleap-io's Video indexing
            try:
                # sleap-io Video uses __getitem__ for frame access
                frame_image = video[lf.frame_idx]
                return frame_image, instance, video
            except Exception:
                # If we can't load the frame, still return the instance
                # The caller will need to handle the None frame_image
                return None, instance, video

    return None, None, None


def compute_anchor_point(
    instance: Optional[sio.Instance], anchor_part: Optional[Text] = None
) -> Optional[Tuple[float, float]]:
    """Computes the anchor point for an instance.

    Args:
        instance: The instance to compute the anchor point for.
        anchor_part: The name of the body part to use as anchor. If None,
            the mean of all visible keypoints is used.

    Returns:
        A tuple (x, y) representing the anchor point coordinates, or None
        if the anchor cannot be computed.
    """
    if instance is None:
        return None

    # If anchor_part is specified, try to use that node
    if anchor_part:
        for node, point in zip(instance.skeleton.nodes, instance.numpy()):
            if node.name == anchor_part and not np.isnan(point).any():
                return (float(point[0]), float(point[1]))

    # Fall back to mean of all visible keypoints
    points = instance.numpy()
    visible_points = points[~np.isnan(points).any(axis=1)]
    if len(visible_points) > 0:
        mean_point = np.mean(visible_points, axis=0)
        return (float(mean_point[0]), float(mean_point[1]))

    return None


class ReceptiveFieldWidget(QtWidgets.QWidget):
    """
    Widget for previewing receptive field on sample image, with caption.

    Args:
        head_name: If given, then used in caption to show which model the
            preview is for.
        show_crop_box: If True, shows a crop size box centered on anchor point.
            This is intended for centered_instance and multi_class_topdown heads.

    Usage:
        Create, then call `setImage` and `setModelConfig` methods.
        For crop box display, also call `setLabels` and `setCropConfig`.
    """

    def __init__(
        self, head_name: Text = "", show_crop_box: bool = False, *args, **kwargs
    ):
        super(ReceptiveFieldWidget, self).__init__(*args, **kwargs)

        self._show_crop_box = show_crop_box
        self._labels = None
        self._instance = None
        self._preview_video = None
        self._anchor_part = None
        self._crop_size = None
        self._rf_size = None  # Track receptive field size for legend
        self._head_name = head_name

        self.layout = QtWidgets.QVBoxLayout()

        self._field_image_widget = ReceptiveFieldImageWidget()

        # Legend (crop size + receptive field)
        self._legend_widget = QtWidgets.QLabel("")

        # Placeholder layout for button insertion (between legend and explanation)
        self._button_layout = QtWidgets.QVBoxLayout()
        self._button_layout.setContentsMargins(0, 4, 0, 4)

        # Explanation text (below legend and optional button)
        self._explanation_widget = QtWidgets.QLabel("")

        # UNet architecture info (params and channels)
        self._arch_info_widget = QtWidgets.QLabel("")

        self.layout.addWidget(self._field_image_widget)
        self.layout.addWidget(self._legend_widget)
        self.layout.addLayout(self._button_layout)
        self.layout.addWidget(self._explanation_widget)
        self.layout.addWidget(self._arch_info_widget)
        self.layout.addStretch()

        self.setLayout(self.layout)

    def _get_legend_text(self) -> Text:
        """Returns the legend text for crop size and receptive field."""
        result = ""

        # Crop size line (if enabled)
        if self._show_crop_box:
            if self._crop_size:
                result += (
                    f'<span style="color: red;">\u25a0</span> '
                    f"<b>Crop Size:</b> {self._crop_size} px<br/>"
                )
            else:
                result += (
                    '<span style="color: red;">\u25a0</span> <b>Crop Size:</b><br/>'
                )

        # Receptive field line
        if self._rf_size:
            result += (
                f'<span style="color: blue;">\u25a0</span> '
                f"<b>Receptive Field:</b> {self._rf_size} px"
            )
        else:
            result += (
                '<span style="color: blue;">\u25a0</span> '
                "<b>Receptive Field:</b> <i>N/A</i>"
            )

        return result

    def _get_explanation_text(
        self, scale, max_stride, down_blocks, convs_per_block, kernel_size
    ) -> Text:
        """Returns explanatory text about receptive field parameters."""
        return f"""<p>Receptive field size is a function<br />
        of the number of down blocks ({down_blocks}), the<br />
        number of convolutions per block ({convs_per_block}),<br />
        and the convolution kernel size ({kernel_size}).</p>

        <p>You can control the number of down<br />
        blocks by setting the <b>Max Stride</b> ({max_stride}).</p>

        <p>You can also control the receptive<br />
        field size relative to the original<br />
        image by adjusting the <b>Input Scaling</b> ({scale}).</p>"""

    def addButtonWidget(self, widget: QtWidgets.QWidget):
        """Add a widget (typically a button) between the legend and explanation.

        Args:
            widget: The widget to add (e.g., QPushButton for "Analyze Sizes...")
        """
        self._button_layout.addWidget(widget)

    def _get_head_output_channels(self, head_name: str) -> Optional[int]:
        """Get the number of output channels required by a head type.

        Args:
            head_name: Name of the sub-head (e.g., "confmaps", "pafs", "class_vectors")

        Returns:
            Number of output channels needed, or None if cannot be determined.
        """
        if self._labels is None:
            return None

        if not self._labels.skeletons:
            return None
        skeleton = self._labels.skeletons[0]

        if head_name == "confmaps":
            # One channel per keypoint
            return len(skeleton.nodes)
        elif head_name == "pafs":
            # Two channels (x, y) per edge
            return len(skeleton.edges) * 2
        elif head_name in ("class_vectors", "class_maps"):
            # Number of unique classes/tracks - skip validation for now
            return None
        else:
            return None

    def _get_arch_info_text(
        self,
        params_formatted: Optional[str],
        head_features: list,
        backbone_type: Optional[str] = "unet",
        model_type: Optional[str] = None,
    ) -> Text:
        """Returns text showing backbone architecture info (params and channels).

        Args:
            params_formatted: Human-readable param count (e.g., "1.30M")
            head_features: List of (head_name, output_stride, channels) tuples
            backbone_type: Type of backbone (e.g., "unet", "convnext", "swint").
            model_type: For convnext/swint, the model variant (e.g., "tiny").
        """
        if backbone_type is None:
            return ""

        if backbone_type == "unet":
            if params_formatted is None:
                return ""

            result = "<p><b>UNet:</b><br/>"
            result += f"<b>Parameters:</b> ~{params_formatted}<br/>"

            # Show features for each head with validation
            for i, (head_name, stride, backbone_channels) in enumerate(head_features):
                head_output = self._get_head_output_channels(head_name)

                if head_output is not None:
                    if backbone_channels >= head_output:
                        # Good: backbone has enough channels
                        result += (
                            f"<b>Features ({head_name} @ stride {stride}):</b> "
                            f'<span style="color: green;">'
                            f"{backbone_channels}\u2192{head_output} \u2713</span>"
                        )
                    else:
                        # Warning: backbone channels less than head output
                        result += (
                            f"<b>Features ({head_name} @ stride {stride}):</b> "
                            f'<span style="color: red;">'
                            f"{backbone_channels}\u2192{head_output} \u26a0</span>"
                        )
                else:
                    # Can't determine head output, just show backbone channels
                    result += (
                        f"<b>Features ({head_name} @ stride {stride}):</b> "
                        f"{backbone_channels} ch"
                    )

                if i < len(head_features) - 1:
                    result += "<br/>"

            result += "</p>"
            return result

        elif backbone_type == "convnext":
            model_display = model_type.capitalize() if model_type else "Tiny"
            result = f"<p><b>ConvNeXt ({model_display}):</b><br/>"
            result += "<b>Max Stride:</b> 32 (fixed)<br/>"
            if params_formatted:
                result += f"<b>Parameters:</b> ~{params_formatted}<br/>"
            result += "<b>Pretrained:</b> ImageNet weights available</p>"
            return result

        elif backbone_type == "swint":
            model_display = model_type.capitalize() if model_type else "Tiny"
            result = f"<p><b>Swin Transformer ({model_display}):</b><br/>"
            result += "<b>Max Stride:</b> 32 (fixed)<br/>"
            if params_formatted:
                result += f"<b>Parameters:</b> ~{params_formatted}<br/>"
            result += "<b>Pretrained:</b> ImageNet weights available</p>"
            return result

        return ""

    def setModelConfig(self, model_cfg: OmegaConf, scale: float):
        """Updates receptive field preview from model config."""
        rf_info = receptive_field_info_from_model_cfg(model_cfg)

        # Store receptive field size for legend
        self._rf_size = rf_info["size"]

        # Update architecture info (params and channels) - only for supported backbones
        self._arch_info_widget.setText(
            self._get_arch_info_text(
                params_formatted=rf_info["params_formatted"],
                head_features=rf_info["head_features"],
                backbone_type=rf_info["backbone_type"],
                model_type=rf_info.get("model_type"),
            )
        )

        # Update legend (crop size + receptive field)
        self._legend_widget.setText(self._get_legend_text())

        # Update explanation text
        self._explanation_widget.setText(
            self._get_explanation_text(
                scale=scale,
                max_stride=rf_info["max_stride"],
                down_blocks=rf_info["down_blocks"],
                convs_per_block=rf_info["convs_per_block"],
                kernel_size=rf_info["kernel_size"],
            )
        )

        self._field_image_widget._set_field_size(rf_info["size"] or 0, scale)

    def setImage(self, *args, **kwargs):
        """Sets image on which receptive field box will be drawn."""
        self._field_image_widget.setImage(*args, **kwargs)

    def setLabels(self, labels: Optional[sio.Labels], fallback_video=None):
        """Sets labels and displays the first labeled frame.

        This finds the first frame with ground truth labels, displays that frame,
        and stores the instance for crop box anchor point calculation (if enabled).

        Args:
            labels: The Labels object containing labeled frames.
            fallback_video: Video to use for getting test frame if labeled frame
                cannot be loaded.
        """
        self._labels = labels
        frame_image, instance, video = get_first_labeled_frame_and_instance(labels)

        # Store instance for crop box (only used if show_crop_box is True)
        if self._show_crop_box:
            self._instance = instance
            self._preview_video = video

        # Set the image - prefer the labeled frame, fall back to video test frame
        if frame_image is not None:
            self._field_image_widget.setImage(frame_image)
        elif fallback_video is not None:
            self._field_image_widget.setImage(fallback_video.backend.read_test_frame())

    def setCropConfig(
        self,
        crop_size: Optional[int],
        scale: float,
        anchor_part: Optional[Text] = None,
        max_hw: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ):
        """Sets crop box configuration.

        Args:
            crop_size: The crop size in pixels, in the size-matched space that
                training crops in (scaled by ``scale``, as
                `compute_crop_size_from_cfg` returns it).
            scale: The scale factor applied to the image during training.
            anchor_part: The name of the body part to use as anchor.
                If None, the mean of all keypoints is used.
            max_hw: The resolved ``(max_height, max_width)`` size-matcher
                target. The previewed frame is displayed at its video's native
                resolution, so the box has to be divided back out of the
                size-matched space to cover the region it really will.
        """
        if not self._show_crop_box:
            return

        self._anchor_part = anchor_part
        self._crop_size = crop_size

        # Compute anchor point from the instance
        anchor = compute_anchor_point(self._instance, anchor_part)

        # Update the legend to include crop size
        self._legend_widget.setText(self._get_legend_text())

        if crop_size and anchor:
            eff_scale = 1.0
            if max_hw is not None and self._preview_video is not None:
                try:
                    shape = self._preview_video.shape
                except Exception:
                    shape = None
                if shape is not None and len(shape) >= 3:
                    eff_scale = compute_eff_scale(
                        (int(shape[1]), int(shape[2])), max_hw
                    )
            self._field_image_widget._set_crop_size(
                crop_size, scale * (eff_scale or 1.0), anchor
            )


class ReceptiveFieldImageWidget(GraphicsView):
    """Widget for showing image with receptive field and optional crop box."""

    def __init__(self, *args, **kwargs):
        self._widget_size = 200
        self._pen_width = 4
        self._crop_pen_width = 2
        self._box_size = None
        self._scale = None
        self._crop_size = None
        # The crop box keeps its own scale: it is drawn in the previewed video's
        # native pixels, which the size matcher rescales, while the receptive
        # field box only accounts for input scaling.
        self._crop_scale = None
        self._crop_anchor = None  # (x, y) coordinates of anchor point

        # Receptive field box (blue, solid)
        box_pen = QtGui.QPen(QtGui.QColor("blue"), self._pen_width)
        box_pen.setCosmetic(True)

        self.box = QtWidgets.QGraphicsRectItem()
        self.box.setPen(box_pen)

        # Crop box (red, dotted, thinner line)
        crop_pen = QtGui.QPen(QtGui.QColor("red"), self._crop_pen_width)
        crop_pen.setCosmetic(True)
        crop_pen.setStyle(QtCore.Qt.DotLine)

        self.crop_box = QtWidgets.QGraphicsRectItem()
        self.crop_box.setPen(crop_pen)

        super(ReceptiveFieldImageWidget, self).__init__(*args, **kwargs)

        self.setFixedSize(self._widget_size, self._widget_size)
        self.scene.addItem(self.box)
        self.scene.addItem(self.crop_box)

    def viewportEvent(self, event):
        """Re-draw receptive field and crop box when needed."""
        # Update the position and visible size of field
        if isinstance(event, QtGui.QPaintEvent):
            self._set_field_size()
            self._set_crop_size()

        # Now draw the viewport
        return super(ReceptiveFieldImageWidget, self).viewportEvent(event)

    def _set_field_size(self, size: Optional[int] = None, scale: float = 1.0):
        """Draws receptive field preview rect, updating size if needed."""
        if size is not None:
            self._box_size = size
            self._scale = scale if scale else 1.0

        if not self._box_size or not self._scale:
            self.box.hide()
            return

        self.box.show()

        # Adjust box relative to scaling on image that will happen in training
        scaled_box_size = self._box_size // self._scale

        # Calculate offset so that box stays centered in the view
        vis_box_rect = self.mapFromScene(
            0, 0, scaled_box_size, scaled_box_size
        ).boundingRect()
        offset = self._widget_size / 2
        scene_center = self.mapToScene(
            offset - (vis_box_rect.width() / 2), offset - (vis_box_rect.height() / 2)
        )

        self.box.setRect(
            scene_center.x(), scene_center.y(), scaled_box_size, scaled_box_size
        )

    def _set_crop_size(
        self,
        size: Optional[int] = None,
        scale: float = 1.0,
        anchor: Optional[Tuple[float, float]] = None,
    ):
        """Draws crop size preview rect centered in the view.

        The crop box tracks the view center (like the receptive field box) so both
        overlays move together when the view changes. The anchor parameter is stored
        but not used for positioning since this is a size comparison preview.

        Args:
            size: The crop size in pixels. If None, uses previously set value.
            scale: The factor mapping `size` into the displayed image's pixels:
                the training input scale, times the size-matcher scale for the
                previewed video (see `ReceptiveFieldWidget.setCropConfig`).
            anchor: The (x, y) coordinates of the anchor point in scene coordinates.
                If None, uses previously set value. Stored for reference but not
                used for positioning.
        """
        if size is not None:
            self._crop_size = size
            self._crop_scale = scale if scale else 1.0
        if anchor is not None:
            self._crop_anchor = anchor

        if not self._crop_size or not self._crop_scale:
            self.crop_box.hide()
            return

        self.crop_box.show()

        # Adjust box relative to scaling on image that will happen in training
        scaled_crop_size = self._crop_size // self._crop_scale

        # Calculate offset so that box stays centered in the view
        # (same logic as _set_field_size for consistency)
        vis_box_rect = self.mapFromScene(
            0, 0, scaled_crop_size, scaled_crop_size
        ).boundingRect()
        offset = self._widget_size / 2
        scene_center = self.mapToScene(
            offset - (vis_box_rect.width() / 2), offset - (vis_box_rect.height() / 2)
        )

        self.crop_box.setRect(
            scene_center.x(),
            scene_center.y(),
            scaled_crop_size,
            scaled_crop_size,
        )
