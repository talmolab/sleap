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
    """
    model_cfg = cfg.model_config
    model_cfg.backbone_config.unet.max_stride = int(
        model_cfg.backbone_config.unet.max_stride
    )

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
        backbone_type=None,  # e.g., "unet"
    )
    head_type = get_head_from_omegaconf(cfg)

    # Collect output strides for each sub-head
    head_output_strides = []  # List of (sub_head_name, output_stride)
    for k, head_cfg in model_cfg.head_configs[head_type].items():
        if k == "class_vectors":
            head_output_strides.append(
                (k, int(model_cfg.backbone_config.unet.max_stride))
            )
        else:
            head_output_strides.append((k, int(head_cfg.output_stride)))

    output_strides = [s for _, s in head_output_strides]
    output_stride = min(output_strides)
    rf_info["output_stride"] = output_stride

    # Check backbone type - currently only UNet is supported
    # TODO: Add support for other backbones (ConvNext, SwinT, etc.)
    if not hasattr(model_cfg.backbone_config, "unet"):
        return rf_info

    rf_info["backbone_type"] = "unet"

    try:
        _ = np.log2(model_cfg.backbone_config.unet.max_stride / output_stride)
    except ZeroDivisionError:
        # Unable to create model from these config parameters
        return rf_info

    if hasattr(model_cfg.backbone_config.unet, "max_stride"):
        rf_info["max_stride"] = model_cfg.backbone_config.unet.max_stride

    rf_info["convs_per_block"] = 2

    rf_info["kernel_size"] = 3

    stem_stride = None
    stem_blocks = 0
    if hasattr(model_cfg.backbone_config.unet, "stem_stride"):
        cfg_stem_stride = model_cfg.backbone_config.unet.stem_stride
        if cfg_stem_stride is not None:
            stem_stride = int(cfg_stem_stride)
            stem_blocks = np.log2(cfg_stem_stride).astype(int)

    down_blocks = (
        np.log2(model_cfg.backbone_config.unet.max_stride).astype(int) - stem_blocks
    )

    rf_info["down_blocks"] = down_blocks

    if rf_info["down_blocks"] and rf_info["convs_per_block"] and rf_info["kernel_size"]:
        rf_info["size"] = compute_rf(
            down_blocks=rf_info["down_blocks"],
            convs_per_block=rf_info["convs_per_block"],
            kernel_size=rf_info["kernel_size"],
        )

    # Extract UNet config for architecture calculations
    unet_cfg = model_cfg.backbone_config.unet
    filters = int(getattr(unet_cfg, "filters", 32))
    filters_rate = float(getattr(unet_cfg, "filters_rate", 1.5))
    max_stride = int(unet_cfg.max_stride)
    middle_block = bool(getattr(unet_cfg, "middle_block", True))
    up_interpolate = bool(getattr(unet_cfg, "up_interpolate", False))

    # Compute channel counts at each stride
    try:
        # Use min output_stride to get all decoder levels we need
        min_output_stride = min(s for _, s in head_output_strides)
        stride_to_channels = unet_utils.compute_unet_channels(
            filters=filters,
            filters_rate=filters_rate,
            max_stride=max_stride,
            output_stride=min_output_stride,
            stem_stride=stem_stride,
        )
        # Populate head_features for each sub-head
        for head_name, head_stride in head_output_strides:
            channels = stride_to_channels.get(head_stride)
            if channels is not None:
                rf_info["head_features"].append((head_name, head_stride, channels))
    except Exception:
        pass  # Leave as empty list if computation fails

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
        pass  # Leave as None if computation fails

    return rf_info


def find_max_instance_bbox_size(labels: sio.Labels) -> float:
    """Find the maximum bounding box dimension across all instances in labels.

    This is a local implementation that avoids importing sleap_nn (which would
    trigger importing torch, adding ~2s to startup time).

    Args:
        labels: A `sio.Labels` containing user-labeled instances.

    Returns:
        The maximum bounding box dimension (max of width or height) across all
        instances.
    """
    max_length = 0.0
    for lf in labels:
        for inst in lf.instances:
            if not inst.is_empty:
                pts = inst.numpy()
                diff_x = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
                diff_x = 0 if np.isnan(diff_x) else diff_x
                max_length = np.maximum(max_length, diff_x)
                diff_y = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
                diff_y = 0 if np.isnan(diff_y) else diff_y
                max_length = np.maximum(max_length, diff_y)
    return float(max_length)


def find_instance_crop_size(
    labels: sio.Labels,
    padding: int = 0,
    maximum_stride: int = 2,
    min_crop_size: Optional[int] = None,
) -> int:
    """Compute the size of the largest instance bounding box from labels.

    This is a local implementation that avoids importing sleap_nn (which would
    trigger importing torch, adding ~2s to startup time).

    Args:
        labels: A `sio.Labels` containing user-labeled instances.
        padding: Integer number of pixels to add to the bounds as margin padding.
        maximum_stride: Ensure that the returned crop size is divisible by this
            value. Useful for ensuring that the crop size will not be truncated
            in a given architecture.
        min_crop_size: The minimum crop size to return. If this value is already
            divisible by maximum_stride, it is returned directly.

    Returns:
        An integer crop size denoting the length of the side of the bounding
        boxes that will contain the instances when cropped. The returned crop
        size will be larger or equal to the input `min_crop_size`.
    """
    # Check if user-specified crop size is divisible by max stride
    min_crop_size = 0 if min_crop_size is None else min_crop_size
    if (min_crop_size > 0) and (min_crop_size % maximum_stride == 0):
        return min_crop_size

    # Calculate crop size by iterating over all instances
    min_crop_size_no_pad = min_crop_size - padding
    max_length = 0.0
    for lf in labels:
        for inst in lf.instances:
            if not inst.is_empty:
                pts = inst.numpy()
                diff_x = np.nanmax(pts[:, 0]) - np.nanmin(pts[:, 0])
                diff_x = 0 if np.isnan(diff_x) else diff_x
                max_length = np.maximum(max_length, diff_x)
                diff_y = np.nanmax(pts[:, 1]) - np.nanmin(pts[:, 1])
                diff_y = 0 if np.isnan(diff_y) else diff_y
                max_length = np.maximum(max_length, diff_y)
                max_length = np.maximum(max_length, min_crop_size_no_pad)

    max_length += float(padding)
    crop_size = math.ceil(max_length / float(maximum_stride)) * maximum_stride

    return int(crop_size)


# Cache for find_instance_crop_size results to avoid expensive recomputation.
# Key: (labels_id, max_stride), Value: computed crop_size (unscaled)
_crop_size_cache: dict = {}


def compute_crop_size_from_cfg(
    data_cfg: OmegaConf, model_cfg: OmegaConf, labels: Optional[sio.Labels] = None
) -> int:
    """Computes crop size from model configuration.

    Uses a cache to avoid expensive recomputation of find_instance_crop_size(),
    which iterates over all instances in the labels.
    """
    crop_size = data_cfg.data_config.preprocessing.crop_size
    if crop_size is None:
        try:
            backbone = model_cfg["_backbone_name"]
            max_stride = int(
                model_cfg.model_config.backbone_config[backbone].max_stride
            )

            # Check cache first (keyed by labels identity and max_stride)
            cache_key = (id(labels), max_stride)
            if cache_key in _crop_size_cache:
                crop_size = _crop_size_cache[cache_key]
            else:
                # Use local implementation (avoids importing sleap_nn/torch)
                crop_size = find_instance_crop_size(labels, maximum_stride=max_stride)
                _crop_size_cache[cache_key] = crop_size
        except Exception:
            # Handle any errors (e.g., missing backbone config)
            crop_size = None
    if crop_size is not None and data_cfg.data_config.preprocessing.scale is not None:
        crop_size = int(crop_size * data_cfg.data_config.preprocessing.scale)
    return crop_size


def get_first_labeled_frame_and_instance(
    labels: Optional[sio.Labels],
) -> Tuple[Optional[np.ndarray], Optional[sio.Instance]]:
    """Gets the first frame with ground truth labels and the first instance.

    Args:
        labels: The Labels object containing labeled frames.

    Returns:
        A tuple of (frame_image, instance) where frame_image is a numpy array
        and instance is the first user instance. Returns (None, None) if no
        labeled frames with user instances are found.
    """
    if labels is None:
        return None, None

    for lf in labels:
        if lf.user_instances:
            # Get the first user instance
            instance = lf.user_instances[0]
            # Get the frame image using sleap-io's Video indexing
            try:
                video = lf.video if hasattr(lf, "video") else labels.videos[0]
                # sleap-io Video uses __getitem__ for frame access
                frame_image = video[lf.frame_idx]
                return frame_image, instance
            except Exception:
                # If we can't load the frame, still return the instance
                # The caller will need to handle the None frame_image
                return None, instance

    return None, None


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

        skeleton = self._labels.skeleton
        if skeleton is None:
            return None

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
    ) -> Text:
        """Returns text showing backbone architecture info (params and channels).

        Args:
            params_formatted: Human-readable param count (e.g., "1.30M")
            head_features: List of (head_name, output_stride, channels) tuples
            backbone_type: Type of backbone (e.g., "unet"). Only renders for
                supported types.
        """
        # Only render for supported backbone types
        if backbone_type != "unet" or params_formatted is None:
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
        frame_image, instance = get_first_labeled_frame_and_instance(labels)

        # Store instance for crop box (only used if show_crop_box is True)
        if self._show_crop_box:
            self._instance = instance

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
    ):
        """Sets crop box configuration.

        Args:
            crop_size: The crop size in pixels.
            scale: The scale factor applied to the image during training.
            anchor_part: The name of the body part to use as anchor.
                If None, the mean of all keypoints is used.
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
            self._field_image_widget._set_crop_size(crop_size, scale, anchor)


class ReceptiveFieldImageWidget(GraphicsView):
    """Widget for showing image with receptive field and optional crop box."""

    def __init__(self, *args, **kwargs):
        self._widget_size = 200
        self._pen_width = 4
        self._crop_pen_width = 2
        self._box_size = None
        self._scale = None
        self._crop_size = None
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
            scale: The scale factor applied to the image during training.
            anchor: The (x, y) coordinates of the anchor point in scene coordinates.
                If None, uses previously set value. Stored for reference but not
                used for positioning.
        """
        if size is not None:
            self._crop_size = size
            self._scale = scale if scale else 1.0
        if anchor is not None:
            self._crop_anchor = anchor

        if not self._crop_size or not self._scale:
            self.crop_box.hide()
            return

        self.crop_box.show()

        # Adjust box relative to scaling on image that will happen in training
        scaled_crop_size = self._crop_size // self._scale

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
