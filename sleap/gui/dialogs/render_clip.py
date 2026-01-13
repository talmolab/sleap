"""Render Video Clip dialog with live preview using sleap-io.

This dialog replaces the YAML-based FormBuilder approach for exporting
labeled video clips, providing a pure Qt implementation with real-time
preview capabilities using sleap-io's rendering API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from qtpy import QtCore, QtWidgets

from sleap.gui.widgets.rendering_preview import RenderingPreviewWidget

if TYPE_CHECKING:
    import sleap_io as sio


class RenderClipDialog(QtWidgets.QDialog):
    """Dialog for rendering video clips with live preview.

    This dialog provides a pure Qt interface for configuring video rendering
    options with real-time preview using sleap-io's `render_image()` function.

    Usage from SLEAP GUI::

        # Get frame range if clip is selected
        frame_range = None
        if context.state["has_frame_range"]:
            frame_range = tuple(context.state["frame_range"])

        dialog = RenderClipDialog(
            labels=context.state["labels"],
            video=context.state["video"],
            current_frame=context.state["frame_idx"],
            frame_range=frame_range,
            parent=context.app,
        )
        if dialog.exec_():
            params = dialog.get_export_params()
            output_path = dialog.get_output_path()
            # Call sio.render_video(...)

    Attributes:
        labels: The Labels object containing labeled frames.
        video: The current video to render.
    """

    def __init__(
        self,
        labels: "sio.Labels",
        video: "sio.Video | None" = None,
        current_frame: int | None = None,
        frame_range: tuple[int, int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        """Initialize the render clip dialog.

        Args:
            labels: The Labels object containing labeled frames.
            video: The video to render. If None, uses first video in labels.
            current_frame: Initial frame to show in preview. If None, uses
                first labeled frame.
            frame_range: Optional (start, end) tuple for initial frame range.
                If provided, selects "Custom range" and sets start/end values.
                This is typically from the main window's selected clip.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.labels = labels
        self.video = video or (labels.videos[0] if labels.videos else None)
        self._current_frame = current_frame
        self._initial_frame_range = frame_range
        self._output_path: str | None = None

        self._setup_ui()
        self._connect_signals()

        # Initialize frame range from main window's selection
        if frame_range is not None:
            self._set_frame_range(frame_range)

        # Jump to current frame if specified
        if current_frame is not None:
            self._jump_to_frame(current_frame)

        self._update_preview()

    def _set_frame_range(self, frame_range: tuple[int, int]):
        """Set the frame range from an external selection (e.g., main window clip).

        Args:
            frame_range: (start, end) tuple of frame indices.
        """
        start, end = frame_range
        # Select custom range mode
        self.range_custom.setChecked(True)
        # Set the start and end values
        self.start_frame.setValue(start)
        self.end_frame.setValue(end - 1)  # end is exclusive in context, make inclusive

    def _jump_to_frame(self, frame_idx: int):
        """Jump preview to a specific frame index."""
        self.preview.set_frame(frame_idx)

    def _setup_ui(self):
        """Build the dialog UI."""
        self.setWindowTitle("Render Video Clip")
        self.setMinimumSize(1000, 700)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setSpacing(12)

        # Left: Preview widget (2/3 width)
        self.preview = RenderingPreviewWidget(self.labels, self.video)
        main_layout.addWidget(self.preview, stretch=2)

        # Right: Options panel (1/3 width)
        options_panel = self._create_options_panel()
        main_layout.addWidget(options_panel, stretch=1)

    def _create_options_panel(self) -> QtWidgets.QWidget:
        """Build the options panel with grouped controls."""
        # Container for scroll + buttons
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)

        # Scrollable options area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        # Video selection (for multi-video Labels)
        if len(self.labels.videos) > 1:
            layout.addWidget(self._create_video_selector_group())

        # Frame Range
        layout.addWidget(self._create_frame_range_group())

        # Appearance
        layout.addWidget(self._create_appearance_group())

        # Quality
        layout.addWidget(self._create_quality_group())

        # Output
        layout.addWidget(self._create_output_group())

        # Spacer
        layout.addStretch()

        scroll.setWidget(widget)
        container_layout.addWidget(scroll, stretch=1)

        # Buttons (outside scroll area - always visible)
        container_layout.addWidget(self._create_buttons())

        return container

    def _create_video_selector_group(self) -> QtWidgets.QGroupBox:
        """Create video selection group (for multi-video Labels)."""
        group = QtWidgets.QGroupBox("Video")
        layout = QtWidgets.QFormLayout(group)

        self.video_selector = QtWidgets.QComboBox()
        for i, video in enumerate(self.labels.videos):
            # Get video filename
            name = getattr(video, "filename", None) or getattr(video, "backend", None)
            if name:
                name = os.path.basename(str(name))
            else:
                name = f"Video {i}"
            n_frames = len(
                [lf for lf in self.labels.labeled_frames if lf.video == video]
            )
            self.video_selector.addItem(f"{name} ({n_frames} labeled)", video)

        # Select current video
        if self.video in self.labels.videos:
            self.video_selector.setCurrentIndex(self.labels.videos.index(self.video))

        self.video_selector.currentIndexChanged.connect(self._on_video_changed)
        layout.addRow("Source:", self.video_selector)

        return group

    def _on_video_changed(self, index: int):
        """Handle video selection change."""
        self.video = self.video_selector.itemData(index)
        # Update preview widget
        self.preview.video = self.video
        self.preview._labeled_frame_indices = self.preview._get_labeled_frame_indices()
        self.preview._update_slider_range()
        if self.preview._labeled_frame_indices:
            self.preview._current_frame_idx = self.preview._labeled_frame_indices[0]
            self.preview.frame_slider.setValue(0)
        self.preview._do_render()
        # Update frame range limits
        self._update_frame_range_limits()

    def _update_frame_range_limits(self):
        """Update frame range spinboxes based on current video."""
        labeled_frames = [
            lf.frame_idx
            for lf in self.labels.labeled_frames
            if self.video is None or lf.video == self.video
        ]
        if labeled_frames:
            self.start_frame.setValue(min(labeled_frames))
            self.end_frame.setValue(max(labeled_frames))
            self.start_frame.setMaximum(max(labeled_frames))
            self.end_frame.setMaximum(max(labeled_frames))

    def _create_frame_range_group(self) -> QtWidgets.QGroupBox:
        """Create frame range selection group."""
        group = QtWidgets.QGroupBox("Frame Range")
        layout = QtWidgets.QVBoxLayout(group)

        # Radio buttons for range mode
        self.range_all = QtWidgets.QRadioButton("All labeled frames")
        self.range_all.setToolTip("Render all frames with labels")
        self.range_all.setChecked(True)
        layout.addWidget(self.range_all)

        self.range_custom = QtWidgets.QRadioButton("Custom range:")
        self.range_custom.setToolTip("Render a specific frame range")
        layout.addWidget(self.range_custom)

        # Custom range inputs
        range_layout = QtWidgets.QHBoxLayout()
        range_layout.setContentsMargins(20, 0, 0, 0)

        self.start_frame = QtWidgets.QSpinBox()
        self.start_frame.setRange(0, 999999)
        self.start_frame.setEnabled(False)
        range_layout.addWidget(QtWidgets.QLabel("Start:"))
        range_layout.addWidget(self.start_frame)

        self.end_frame = QtWidgets.QSpinBox()
        self.end_frame.setRange(0, 999999)
        self.end_frame.setEnabled(False)
        range_layout.addWidget(QtWidgets.QLabel("End:"))
        range_layout.addWidget(self.end_frame)

        range_layout.addStretch()
        layout.addLayout(range_layout)

        # Connect range mode changes
        self.range_custom.toggled.connect(self.start_frame.setEnabled)
        self.range_custom.toggled.connect(self.end_frame.setEnabled)

        # Set range based on labels
        labeled_frames = [
            lf.frame_idx
            for lf in self.labels.labeled_frames
            if self.video is None or lf.video == self.video
        ]
        if labeled_frames:
            self.start_frame.setValue(min(labeled_frames))
            self.end_frame.setValue(max(labeled_frames))
            self.start_frame.setMaximum(max(labeled_frames))
            self.end_frame.setMaximum(max(labeled_frames))

        return group

    def _create_appearance_group(self) -> QtWidgets.QGroupBox:
        """Create appearance options group."""
        group = QtWidgets.QGroupBox("Appearance")
        layout = QtWidgets.QFormLayout(group)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        # Color by
        self.color_by = QtWidgets.QComboBox()
        self.color_by.addItems(["track", "instance", "node"])
        self.color_by.setToolTip(
            "track: Consistent colors per tracked animal\n"
            "instance: Unique colors per animal per frame\n"
            "node: Unique colors per body part"
        )
        layout.addRow("Color by:", self.color_by)

        # Palette
        self.palette = QtWidgets.QComboBox()
        palettes = [
            "standard",
            "distinct",
            "rainbow",
            "warm",
            "cool",
            "pastel",
            "seaborn",
            "tableau10",
            "viridis",
            "glasbey",
            "glasbey_hv",
            "glasbey_cool",
            "glasbey_warm",
        ]
        self.palette.addItems(palettes)
        self.palette.setCurrentText("tableau10")
        self.palette.setToolTip("Color palette for instances/nodes")
        layout.addRow("Palette:", self.palette)

        # Marker shape
        self.marker_shape = QtWidgets.QComboBox()
        self.marker_shape.addItems(["circle", "square", "diamond", "triangle", "cross"])
        self.marker_shape.setToolTip("Shape of node markers")
        layout.addRow("Marker:", self.marker_shape)

        # Marker size
        self.marker_size = QtWidgets.QDoubleSpinBox()
        self.marker_size.setRange(1.0, 30.0)
        self.marker_size.setValue(4.0)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setToolTip("Size of node markers in pixels")
        layout.addRow("Marker size:", self.marker_size)

        # Line width
        self.line_width = QtWidgets.QDoubleSpinBox()
        self.line_width.setRange(0.5, 15.0)
        self.line_width.setValue(2.0)
        self.line_width.setSingleStep(0.5)
        self.line_width.setToolTip("Width of skeleton edges in pixels")
        layout.addRow("Line width:", self.line_width)

        # Opacity
        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0.1, 1.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(1.0)
        self.alpha.setToolTip("Opacity of skeleton overlay (0.1-1.0)")
        layout.addRow("Opacity:", self.alpha)

        # Show nodes/edges checkboxes
        visibility_layout = QtWidgets.QHBoxLayout()
        self.show_nodes = QtWidgets.QCheckBox("Nodes")
        self.show_nodes.setChecked(True)
        self.show_nodes.setToolTip("Show node markers")
        self.show_edges = QtWidgets.QCheckBox("Edges")
        self.show_edges.setChecked(True)
        self.show_edges.setToolTip("Show skeleton edges")
        visibility_layout.addWidget(self.show_nodes)
        visibility_layout.addWidget(self.show_edges)
        visibility_layout.addStretch()
        layout.addRow("Show:", visibility_layout)

        # Background
        self.background = QtWidgets.QComboBox()
        self.background.addItems(["video", "black", "white", "gray"])
        self.background.setToolTip(
            "Background for rendering\n"
            "video: Use original video frames\n"
            "black/white/gray: Solid color background"
        )
        layout.addRow("Background:", self.background)

        return group

    def _create_quality_group(self) -> QtWidgets.QGroupBox:
        """Create quality/encoding options group."""
        group = QtWidgets.QGroupBox("Quality")
        layout = QtWidgets.QFormLayout(group)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        # Preset
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(
            ["preview (0.25x)", "draft (0.5x)", "final (1.0x)", "custom"]
        )
        self.preset.setCurrentText("final (1.0x)")
        self.preset.setToolTip(
            "Quality preset\n"
            "preview: Fast, 0.25x resolution\n"
            "draft: Medium, 0.5x resolution\n"
            "final: Full resolution"
        )
        layout.addRow("Preset:", self.preset)

        # Scale (only enabled for custom preset)
        self.scale = QtWidgets.QDoubleSpinBox()
        self.scale.setRange(0.1, 2.0)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.1)
        self.scale.setEnabled(False)
        self.scale.setToolTip("Output scale factor (1.0 = original resolution)")
        layout.addRow("Scale:", self.scale)

        # CRF (video quality)
        self.crf = QtWidgets.QSpinBox()
        self.crf.setRange(2, 51)
        self.crf.setValue(23)
        self.crf.setToolTip(
            "Video quality (CRF)\n"
            "Lower = better quality, larger file\n"
            "Recommended: 18-28"
        )
        layout.addRow("CRF:", self.crf)

        # Connect preset to scale
        self.preset.currentTextChanged.connect(self._on_preset_changed)

        return group

    def _create_output_group(self) -> QtWidgets.QGroupBox:
        """Create output options group."""
        group = QtWidgets.QGroupBox("Output")
        layout = QtWidgets.QFormLayout(group)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        # FPS
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(30)
        self.fps.setToolTip("Output video frame rate")
        layout.addRow("FPS:", self.fps)

        # Open when done
        self.open_when_done = QtWidgets.QCheckBox("Open video when done")
        self.open_when_done.setChecked(True)
        layout.addRow("", self.open_when_done)

        return group

    def _create_buttons(self) -> QtWidgets.QWidget:
        """Create dialog buttons (fixed at bottom, outside scroll area)."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)

        layout.addStretch()

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

        self.render_btn = QtWidgets.QPushButton("Render Video...")
        self.render_btn.setDefault(True)
        self.render_btn.clicked.connect(self._on_render)
        layout.addWidget(self.render_btn)

        return widget

    def _connect_signals(self):
        """Connect all control signals to preview updates."""
        # Appearance controls
        self.color_by.currentTextChanged.connect(self._update_preview)
        self.palette.currentTextChanged.connect(self._update_preview)
        self.marker_shape.currentTextChanged.connect(self._update_preview)
        self.marker_size.valueChanged.connect(self._update_preview)
        self.line_width.valueChanged.connect(self._update_preview)
        self.alpha.valueChanged.connect(self._update_preview)
        self.show_nodes.toggled.connect(self._update_preview)
        self.show_edges.toggled.connect(self._update_preview)
        self.background.currentTextChanged.connect(self._update_preview)

        # Quality controls that affect preview
        self.preset.currentTextChanged.connect(self._update_preview)
        self.scale.valueChanged.connect(self._update_preview)

    def _on_preset_changed(self, preset_text: str):
        """Handle preset change - enable/disable scale spinbox."""
        is_custom = preset_text == "custom"
        self.scale.setEnabled(is_custom)

        # Update scale value based on preset
        if not is_custom:
            preset_scales = {
                "preview (0.25x)": 0.25,
                "draft (0.5x)": 0.5,
                "final (1.0x)": 1.0,
            }
            self.scale.setValue(preset_scales.get(preset_text, 1.0))

    def _get_render_params(self, for_preview: bool = True) -> dict:
        """Get current render parameters from controls.

        Args:
            for_preview: If True, always use scale=1.0 for accurate preview.
                If False, use the actual scale setting for export.
        """
        params = {
            "color_by": self.color_by.currentText(),
            "palette": self.palette.currentText(),
            "marker_shape": self.marker_shape.currentText(),
            "marker_size": self.marker_size.value(),
            "line_width": self.line_width.value(),
            "alpha": self.alpha.value(),
            "show_nodes": self.show_nodes.isChecked(),
            "show_edges": self.show_edges.isChecked(),
        }

        # Background
        bg = self.background.currentText()
        if bg != "video":
            params["background"] = bg

        # Scale: always 1.0 for preview (accurate display), use setting for export
        if for_preview:
            # Preview always at full resolution for accurate display
            params["scale"] = 1.0
        else:
            params["scale"] = self.scale.value()

        return params

    def _update_preview(self):
        """Update preview with current settings."""
        params = self._get_render_params()
        self.preview.set_render_params(**params)

    def _on_render(self):
        """Handle render button click - prompt for output file."""
        # Generate default filename
        default_name = "rendered_clip.mp4"
        if self.video:
            video_name = getattr(self.video, "filename", None) or getattr(
                self.video, "backend", None
            )
            if video_name:
                base = os.path.splitext(os.path.basename(str(video_name)))[0]
                default_name = f"{base}_rendered.mp4"

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Rendered Video",
            default_name,
            "MP4 Video (*.mp4);;All Files (*)",
        )

        if not filename:
            return

        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"

        self._output_path = filename
        self.accept()

    def get_output_path(self) -> str | None:
        """Get the selected output file path."""
        return self._output_path

    def get_export_params(self) -> dict:
        """Get all export parameters.

        Returns:
            Dictionary of parameters for `sio.render_video()`.
        """
        # Use actual scale setting for export (not preview's scale=1.0)
        params = self._get_render_params(for_preview=False)
        params["fps"] = self.fps.value()
        params["crf"] = self.crf.value()
        params["open_when_done"] = self.open_when_done.isChecked()

        # Frame range
        if self.range_custom.isChecked():
            params["start"] = self.start_frame.value()
            params["end"] = self.end_frame.value()

        return params

    def get_frame_indices(self) -> list[int]:
        """Get list of frame indices to render.

        Returns:
            List of frame indices to include in the rendered video.
        """
        if self.range_all.isChecked():
            # All labeled frames for current video
            return [
                lf.frame_idx
                for lf in self.labels.labeled_frames
                if self.video is None or lf.video == self.video
            ]
        else:
            # Custom range - all labeled frames within range
            start = self.start_frame.value()
            end = self.end_frame.value()
            return [
                lf.frame_idx
                for lf in self.labels.labeled_frames
                if (self.video is None or lf.video == self.video)
                and start <= lf.frame_idx <= end
            ]
