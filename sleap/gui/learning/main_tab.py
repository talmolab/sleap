"""MainTabWidget for training/inference dialogs.

This module replaces TrainingPipelineWidget's YAML form builder approach
with explicit Qt layout programming.

Layout sections:
- Pipeline Type: dropdown + description + pipeline-specific fields
- Inference Target: FrameTargetSelector
- Input Data: Convert Image To dropdown (training only)
- Tracker: tracker method + options (inference only)
- Performance: paired fields layout
- WandB: paired fields layout (training only)
- Output: grouped checkboxes (training only)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sleap.gui.widgets.frame_target_selector import FrameTargetSelector
from sleap.gui.learning.wandb_utils import (
    check_wandb_login_status,
    get_wandb_api_key_help_text,
)
from sleap.prefs import prefs


# =============================================================================
# Data Definitions
# =============================================================================


@dataclass
class PipelineOption:
    """Defines a pipeline type option."""

    key: str
    label: str
    description: str
    fields: List[Tuple[str, str, Any]] = None  # (key, label, default_value)

    def __post_init__(self):
        if self.fields is None:
            self.fields = []


PIPELINE_OPTIONS_TRAINING = [
    PipelineOption(
        key="multi-animal bottom-up",
        label="multi-animal bottom-up",
        description=(
            "This pipeline uses a single model with two output heads: "
            'a "confidence map" head to predict the nodes for an entire image '
            'and a "part affinity field" head to group the nodes into distinct '
            "animal instances."
        ),
        fields=[
            (
                "model_config.head_configs.bottomup.confmaps.sigma",
                "Sigma for Nodes",
                5.0,
            ),
            (
                "model_config.head_configs.bottomup.pafs.sigma",
                "Sigma for Edges",
                15.0,
            ),
        ],
    ),
    PipelineOption(
        key="multi-animal top-down",
        label="multi-animal top-down",
        description=(
            'This pipeline uses two models: a "centroid" model to locate and '
            'crop around each animal in the frame, and a "centered-instance '
            'confidence map" model for predicted node locations for each '
            "individual animal predicted by the centroid model."
        ),
        fields=[
            (
                "model_config.head_configs.centered_instance.confmaps.anchor_part",
                "Anchor Part",
                None,
            ),
            (
                "model_config.head_configs.centroid.confmaps.sigma",
                "Sigma for Centroids",
                5.0,
            ),
            (
                "model_config.head_configs.centered_instance.confmaps.sigma",
                "Sigma for Nodes",
                5.0,
            ),
        ],
    ),
    PipelineOption(
        key="multi-animal bottom-up-id",
        label="multi-animal bottom-up-id",
        description=(
            'This pipeline uses a single model with two output heads: a "confidence '
            'map" head to predict the nodes for an entire image and a "part '
            'affinity field" head to group the nodes into distinct animal '
            "instances. It also handles classification and tracking."
        ),
        fields=[
            (
                "model_config.head_configs.multi_class_bottomup.confmaps.sigma",
                "Sigma for Nodes",
                5.0,
            ),
            (
                "model_config.head_configs.multi_class_bottomup.class_maps.sigma",
                "Sigma for Edges",
                15.0,
            ),
        ],
    ),
    PipelineOption(
        key="multi-animal top-down-id",
        label="multi-animal top-down-id",
        description=(
            'This pipeline uses two models: a "centroid" model to locate and crop '
            'around each animal in the frame, and a "centered-instance confidence '
            'map" model for predicted node locations for each individual animal '
            "predicted by the centroid model. It also handles classification and "
            "tracking."
        ),
        fields=[
            (
                "model_config.head_configs.multi_class_topdown.confmaps.anchor_part",
                "Anchor Part",
                None,
            ),
            (
                "model_config.head_configs.centroid.confmaps.sigma",
                "Sigma for Centroids",
                5.0,
            ),
            (
                "model_config.head_configs.multi_class_topdown.confmaps.sigma",
                "Sigma for Nodes",
                5.0,
            ),
        ],
    ),
    PipelineOption(
        key="single animal",
        label="single animal",
        description=(
            'This pipeline uses a single "confidence map" model to predict the '
            "nodes for an entire image. It cannot be used for multi-animal data."
        ),
        fields=[
            (
                "model_config.head_configs.single_instance.confmaps.sigma",
                "Sigma for Nodes",
                5.0,
            ),
        ],
    ),
]

PIPELINE_OPTIONS_INFERENCE = PIPELINE_OPTIONS_TRAINING + [
    PipelineOption(
        key="movenet-lightning",
        label="movenet-lightning",
        description=(
            "This pipeline uses a pretrained MoveNet Lightning model to predict the "
            "nodes for an entire image and then groups all of these nodes into a "
            "single instance. Lightning is intended for latency-critical applications. "
            "Note that this model is intended for human pose estimation. There is no "
            "support for videos containing more than one instance."
        ),
    ),
    PipelineOption(
        key="movenet-thunder",
        label="movenet-thunder",
        description=(
            "This pipeline uses a pretrained MoveNet Thunder model to predict the "
            "nodes for an entire image and then groups all of these nodes into a "
            "single instance. Thunder is intended for applications that require high "
            "accuracy. Note that this model is intended for human pose estimation. "
            "There is no support for videos containing more than one instance."
        ),
    ),
    PipelineOption(
        key="tracking-only",
        label="tracking-only",
        description="Run tracking on existing predictions without running pose "
        "estimation.",
    ),
]


# =============================================================================
# MainTabWidget
# =============================================================================


class MainTabWidget(QWidget):
    """Native Qt main tab for training/inference dialogs.

    Replaces TrainingPipelineWidget's YAML form builder approach with
    explicit Qt layout programming.

    Signals:
        updatePipeline: Emitted when pipeline selection changes.
        valueChanged: Emitted when any field value changes.
    """

    updatePipeline = Signal(str)
    valueChanged = Signal()

    # Minimum width for main content boxes
    BOX_MIN_WIDTH = 550

    def __init__(
        self,
        mode: str = "training",
        skeleton: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ):
        """Initialize the main tab widget.

        Args:
            mode: "training" or "inference"
            skeleton: Skeleton object with node_names for anchor part dropdowns.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._mode = mode
        self._skeleton = skeleton
        self._fields: Dict[str, QWidget] = {}
        # Store pipeline-specific fields separately to handle duplicate keys
        # across pipelines (e.g., centroid.sigma in top-down and top-down-id)
        self._pipeline_fields: Dict[str, Dict[str, QWidget]] = {}
        self._pipeline_options = (
            PIPELINE_OPTIONS_TRAINING
            if mode == "training"
            else PIPELINE_OPTIONS_INFERENCE
        )
        self._wandb_api_key_placeholder: Optional[str] = None
        self._setup_ui()
        self._connect_signals()

        # Initialize preferences and status displays
        if mode == "training":
            self._init_training_settings()
            self._update_wandb_status()

    def _setup_ui(self):
        """Build the complete main tab layout."""
        # Main layout with scroll area
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        scroll_content.setObjectName("mainTabScrollContent")
        scroll_content.setStyleSheet(
            "#mainTabScrollContent { background-color: white; }"
        )
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Section 1: Pipeline Type
        main_layout.addWidget(self._create_pipeline_section())

        # Section 2: Frame Target Selector
        self.frame_target_selector = FrameTargetSelector(mode=self._mode)
        self.frame_target_selector.set_compact_mode(True)
        self.frame_target_selector.setMinimumWidth(self.BOX_MIN_WIDTH)
        self.frame_target_selector.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        main_layout.addWidget(self.frame_target_selector)

        # Section 3: Preprocessing / Postprocessing (both modes)
        main_layout.addWidget(self._create_preprocessing_section())

        # Section 4: Tracker (inference only)
        if self._mode == "inference":
            main_layout.addWidget(self._create_tracker_section())

        # Section 5: Performance
        main_layout.addWidget(self._create_performance_section())

        # Section 6: WandB (training only)
        if self._mode == "training":
            main_layout.addWidget(self._create_wandb_section())

            # Section 7: Output (training only)
            main_layout.addWidget(self._create_output_section())

        main_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

    def _connect_signals(self):
        """Connect internal signals."""
        # Pipeline combo
        self._pipeline_combo.currentIndexChanged.connect(self._on_pipeline_changed)

        # Frame target selector
        self.frame_target_selector.valueChanged.connect(self.valueChanged.emit)

        # Connect all pipeline-specific fields (from all pipelines)
        connected_widgets = set()
        for pipeline_fields in self._pipeline_fields.values():
            for widget in pipeline_fields.values():
                if id(widget) not in connected_widgets:
                    self._connect_field_signal(widget)
                    connected_widgets.add(id(widget))

        # Connect non-pipeline fields
        for widget in self._fields.values():
            if id(widget) not in connected_widgets:
                self._connect_field_signal(widget)
                connected_widgets.add(id(widget))

    def _connect_field_signal(self, widget: QWidget):
        """Connect a field widget's change signal to valueChanged."""
        if isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(lambda _: self.valueChanged.emit())
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(lambda _: self.valueChanged.emit())
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(lambda _: self.valueChanged.emit())
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(lambda _: self.valueChanged.emit())

    def _on_pipeline_changed(self, index: int):
        """Handle pipeline selection change."""
        if index >= 0:
            self._pipeline_stack.setCurrentIndex(index)
            # Emit normalized short name (e.g., "top-down" not "multi-animal top-down")
            self.updatePipeline.emit(self.current_pipeline)
            self.valueChanged.emit()

    # -------------------------------------------------------------------------
    # Section Builders
    # -------------------------------------------------------------------------

    def _create_pipeline_section(self) -> QGroupBox:
        """Create the Pipeline Type section."""
        group = QGroupBox("Pipeline Type")
        group.setMinimumWidth(self.BOX_MIN_WIDTH)

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Pipeline dropdown
        self._pipeline_combo = QComboBox()
        for opt in self._pipeline_options:
            self._pipeline_combo.addItem(opt.label, opt.key)
        layout.addWidget(self._pipeline_combo)

        # Stacked widget for pipeline-specific content
        self._pipeline_stack = QStackedWidget()
        self._pipeline_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        for opt in self._pipeline_options:
            page = self._create_pipeline_page(opt)
            self._pipeline_stack.addWidget(page)

        # Calculate and set fixed height based on tallest page
        # Force layout first so word-wrapped labels calculate correct heights
        max_height = 0
        for i in range(self._pipeline_stack.count()):
            page = self._pipeline_stack.widget(i)
            page.adjustSize()
            page_height = page.sizeHint().height()
            max_height = max(max_height, page_height)
        self._pipeline_stack.setFixedHeight(max_height)

        layout.addWidget(self._pipeline_stack)

        return group

    def _create_pipeline_page(self, opt: PipelineOption) -> QWidget:
        """Create a page for a pipeline option."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Description
        desc_label = QLabel(opt.description)
        desc_label.setWordWrap(True)
        desc_label.setMinimumWidth(
            self.BOX_MIN_WIDTH - 24
        )  # Account for group box margins
        desc_label.setStyleSheet("color: #666;")
        layout.addWidget(desc_label)

        # Pipeline-specific fields - all on one row
        if opt.fields:
            # Initialize dict for this pipeline's fields
            self._pipeline_fields[opt.key] = {}

            fields_row = QHBoxLayout()
            fields_row.setSpacing(8)

            for field_key, field_label, default_value in opt.fields:
                label = QLabel(field_label + ":")
                fields_row.addWidget(label)

                widget = self._create_field_widget(field_key, default_value)
                fields_row.addWidget(widget)
                # Store in pipeline-specific dict (prevents overwrites)
                self._pipeline_fields[opt.key][field_key] = widget
                # Also store in _fields for backward compatibility with set_form_data
                self._fields[field_key] = widget

                fields_row.addSpacing(12)

            fields_row.addStretch()
            layout.addLayout(fields_row)

        layout.addStretch()
        return page

    # Standard height for form field widgets (spinboxes, dropdowns)
    FIELD_HEIGHT = 22

    def _create_field_widget(self, key: str, default_value: Any) -> QWidget:
        """Create a widget for a field based on its key and default value."""
        if "anchor_part" in key:
            # Optional list (dropdown populated later or from skeleton)
            widget = QComboBox()
            widget.setFixedHeight(self.FIELD_HEIGHT)
            widget.addItem("", None)  # Empty = use bounding box midpoint
            # Populate with skeleton node names if available
            if self._skeleton and hasattr(self._skeleton, "node_names"):
                for name in self._skeleton.node_names:
                    widget.addItem(name, name)
            return widget
        elif "sigma" in key:
            # Double spin box
            widget = QDoubleSpinBox()
            widget.setFixedHeight(self.FIELD_HEIGHT)
            widget.setRange(0.1, 100.0)
            widget.setSingleStep(0.5)
            widget.setDecimals(2)
            widget.setValue(default_value if default_value is not None else 5.0)
            return widget
        else:
            # Default to line edit
            widget = QLineEdit()
            if default_value is not None:
                widget.setText(str(default_value))
            return widget

    def _create_preprocessing_section(self) -> QGroupBox:
        """Create the Preprocessing / Postprocessing section."""
        group = QGroupBox("Preprocessing / Postprocessing")
        group.setMinimumWidth(self.BOX_MIN_WIDTH)

        layout = QHBoxLayout(group)
        layout.setSpacing(8)

        # Convert Colors (training only shows this)
        if self._mode == "training":
            label = QLabel("Convert Colors:")
            layout.addWidget(label)

            convert_combo = QComboBox()
            convert_combo.addItem("", "")  # No conversion
            convert_combo.addItem("RGB", "RGB")
            convert_combo.addItem("grayscale", "grayscale")
            self._fields["_ensure_channels"] = convert_combo
            layout.addWidget(convert_combo)

            layout.addSpacing(20)

        # Max Instances
        max_label = QLabel("Max Instances:")
        layout.addWidget(max_label)

        max_spinbox = QSpinBox()
        max_spinbox.setRange(1, 100)
        max_spinbox.setValue(1)
        max_spinbox.setMinimumWidth(80)
        self._fields["_max_instances"] = max_spinbox
        layout.addWidget(max_spinbox)

        no_max_cb = QCheckBox("No max")
        no_max_cb.setChecked(True)  # Default to no max
        self._fields["_max_instances_disabled"] = no_max_cb
        layout.addWidget(no_max_cb)

        # Connect checkbox to enable/disable spinbox
        no_max_cb.stateChanged.connect(
            lambda state, sb=max_spinbox: sb.setEnabled(not state)
        )
        max_spinbox.setEnabled(False)  # Start disabled since "No max" is checked

        layout.addStretch()

        return group

    def _create_tracker_section(self) -> QGroupBox:
        """Create the Tracker section (inference only)."""
        group = QGroupBox("Tracker")
        group.setMinimumWidth(self.BOX_MIN_WIDTH)

        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(9, 6, 9, 9)

        # Tracker method row
        method_row = QHBoxLayout()
        method_row.setSpacing(8)

        method_label = QLabel("Tracker Method:")
        method_row.addWidget(method_label)

        tracker_combo = QComboBox()
        tracker_combo.addItem("none", "none")
        tracker_combo.addItem("flow", "flow")
        tracker_combo.addItem("simple", "simple")
        self._fields["tracking.tracker"] = tracker_combo
        method_row.addWidget(tracker_combo)
        method_row.addStretch()

        layout.addLayout(method_row)

        # Stacked widget for tracker-specific options
        self._tracker_stack = QStackedWidget()
        self._tracker_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # None page (empty - minimal height)
        none_page = QWidget()
        none_page.setFixedHeight(0)
        self._tracker_stack.addWidget(none_page)

        # Flow page
        flow_page = self._create_tracker_options_page("flow")
        self._tracker_stack.addWidget(flow_page)

        # Simple page
        simple_page = self._create_tracker_options_page("simple")
        self._tracker_stack.addWidget(simple_page)

        def on_tracker_changed(index):
            self._tracker_stack.setCurrentIndex(index)
            # Resize stacked widget to fit current page
            current = self._tracker_stack.currentWidget()
            self._tracker_stack.setFixedHeight(current.sizeHint().height())

        tracker_combo.currentIndexChanged.connect(on_tracker_changed)
        # Initialize with current page height
        on_tracker_changed(0)

        layout.addWidget(self._tracker_stack)

        return group

    def _create_tracker_options_page(self, tracker_type: str) -> QWidget:
        """Create options page for a tracker type."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        # Description
        if tracker_type == "flow":
            desc = (
                'This tracker "shifts" instances from previous frames using optical '
                "flow before matching instances in each frame to the shifted instances "
                "from prior frames."
            )
        else:
            desc = (
                "This tracker assigns track identities by matching instances from "
                "prior frames to instances on subsequent frames."
            )

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setMinimumWidth(self.BOX_MIN_WIDTH - 24)
        desc_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(desc_label)

        # Form for tracker options - compact layout
        form = QFormLayout()
        form.setSpacing(4)
        form.setContentsMargins(0, 4, 0, 0)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)

        # Max tracks
        max_tracks_row = QHBoxLayout()
        max_tracks_row.setSpacing(6)
        max_tracks = QSpinBox()
        max_tracks.setRange(1, 100)
        max_tracks.setValue(1)
        max_tracks.setFixedWidth(60)
        self._fields[f"tracking.max_tracks.{tracker_type}"] = max_tracks
        max_tracks_row.addWidget(max_tracks)

        no_limit_cb = QCheckBox("No limit")
        no_limit_cb.setChecked(True)
        self._fields[f"tracking.max_tracks_disabled.{tracker_type}"] = no_limit_cb
        max_tracks_row.addWidget(no_limit_cb)
        max_tracks_row.addStretch()

        no_limit_cb.stateChanged.connect(
            lambda state, sb=max_tracks: sb.setEnabled(not state)
        )
        max_tracks.setEnabled(False)

        form.addRow("Max Tracks:", max_tracks_row)

        # Similarity method
        similarity = QComboBox()
        if tracker_type == "flow":
            similarity.addItem("oks", "oks")
            similarity.addItem("iou", "iou")
            similarity.addItem("centroids", "centroids")
        else:
            similarity.addItem("centroid", "centroid")
            similarity.addItem("iou", "iou")
            similarity.addItem("object keypoint", "instance")
        self._fields[f"tracking.similarity.{tracker_type}"] = similarity
        form.addRow("Similarity Method:", similarity)

        # Match method
        match = QComboBox()
        match.addItem("greedy", "greedy")
        match.addItem("hungarian", "hungarian")
        if tracker_type == "simple":
            match.setCurrentIndex(1)  # Default to hungarian for simple
        self._fields[f"tracking.match.{tracker_type}"] = match
        form.addRow("Matching Method:", match)

        # Track window
        track_window = QSpinBox()
        track_window.setRange(1, 100)
        track_window.setValue(5)
        track_window.setFixedWidth(60)
        self._fields[f"tracking.track_window.{tracker_type}"] = track_window
        form.addRow("Elapsed Frame Window:", track_window)

        # Robust quantile of similarity scores
        robust_row = QHBoxLayout()
        robust_row.setSpacing(6)
        robust_spinbox = QDoubleSpinBox()
        robust_spinbox.setRange(0.0, 1.0)
        robust_spinbox.setSingleStep(0.05)
        robust_spinbox.setValue(0.95)
        robust_spinbox.setFixedWidth(60)
        robust_spinbox.setDecimals(2)
        self._fields[f"tracking.robust.{tracker_type}"] = robust_spinbox
        robust_row.addWidget(robust_spinbox)

        use_max_cb = QCheckBox("Use max (non-robust)")
        use_max_cb.setChecked(True)  # Default: disabled (use max)
        self._fields[f"tracking.robust_disabled.{tracker_type}"] = use_max_cb
        robust_row.addWidget(use_max_cb)
        robust_row.addStretch()

        use_max_cb.stateChanged.connect(
            lambda state, sb=robust_spinbox: sb.setEnabled(not state)
        )
        robust_spinbox.setEnabled(False)  # Disabled by default

        form.addRow("Robust Quantile:", robust_row)

        layout.addLayout(form)

        # Post-tracker options - compact
        post_row = QHBoxLayout()
        post_row.setContentsMargins(0, 2, 0, 0)
        post_label = QLabel("Post-tracking:")
        post_label.setStyleSheet("font-weight: bold;")
        post_row.addWidget(post_label)

        connect_breaks = QCheckBox("Connect Single Track Breaks")
        self._fields[f"tracking.post_connect_single_breaks.{tracker_type}"] = (
            connect_breaks
        )
        post_row.addWidget(connect_breaks)
        post_row.addStretch()
        layout.addLayout(post_row)

        return page

    def _create_performance_section(self) -> QGroupBox:
        """Create the Performance section."""
        group = QGroupBox("Performance")
        group.setMinimumWidth(self.BOX_MIN_WIDTH)

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Fixed-width labels for alignment
        LABEL_WIDTH_COL1 = 105  # "Device Accelerator:" width
        DROPDOWN_WIDTH = 140  # Dropdown width for alignment
        LABEL_WIDTH_COL2 = 115  # "Data Loader Workers:" is longest

        # Row 1: Data Pipeline + Workers (training only)
        if self._mode == "training":
            row1 = QHBoxLayout()
            row1.setSpacing(8)

            pipeline_label = QLabel("Data Pipeline:")
            pipeline_label.setFixedWidth(LABEL_WIDTH_COL1)
            row1.addWidget(pipeline_label)

            data_pipeline = QComboBox()
            data_pipeline.addItem("Stream (no caching)", "stream")
            data_pipeline.addItem("Cache in Memory", "cache_memory")
            data_pipeline.addItem("Cache to Disk", "cache_disk")
            data_pipeline.setCurrentIndex(1)  # Default: Cache in Memory
            data_pipeline.setFixedWidth(DROPDOWN_WIDTH)
            self._fields["_data_pipeline_fw"] = data_pipeline
            row1.addWidget(data_pipeline)

            row1.addSpacing(20)

            workers_label = QLabel("Data Loader Workers:")
            workers_label.setFixedWidth(LABEL_WIDTH_COL2)
            row1.addWidget(workers_label)

            workers = QSpinBox()
            workers.setRange(0, 16)
            workers.setValue(0)
            workers.setMinimumWidth(60)
            self._fields["trainer_config.train_data_loader.num_workers"] = workers
            row1.addWidget(workers)

            row1.addStretch()
            layout.addLayout(row1)

        # Row 1 for inference: Batch Size
        elif self._mode == "inference":
            row1 = QHBoxLayout()
            row1.setSpacing(8)

            batch_label = QLabel("Batch Size:")
            batch_label.setFixedWidth(LABEL_WIDTH_COL1)
            row1.addWidget(batch_label)

            batch_size = QSpinBox()
            batch_size.setRange(1, 128)
            batch_size.setValue(4)
            batch_size.setMinimumWidth(60)
            self._fields["_batch_size"] = batch_size
            row1.addWidget(batch_size)

            default_batch_cb = QCheckBox("Default")
            default_batch_cb.setChecked(True)  # Default to using model's batch size
            self._fields["_batch_size_default"] = default_batch_cb
            row1.addWidget(default_batch_cb)

            # Connect checkbox to enable/disable spinbox
            default_batch_cb.stateChanged.connect(
                lambda state, sb=batch_size: sb.setEnabled(not state)
            )
            batch_size.setEnabled(False)  # Start disabled since "Default" is checked

            row1.addStretch()
            layout.addLayout(row1)

        # Row 2: Accelerator + Devices
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        accel_label = QLabel("Device Accelerator:")
        accel_label.setFixedWidth(LABEL_WIDTH_COL1)
        row2.addWidget(accel_label)

        accelerator = QComboBox()
        accelerator.addItem("auto", "auto")
        accelerator.addItem("cuda", "cuda")
        accelerator.addItem("cpu", "cpu")
        accelerator.addItem("mps", "mps")
        accelerator.setFixedWidth(DROPDOWN_WIDTH)
        self._fields["trainer_config.trainer_accelerator"] = accelerator
        row2.addWidget(accelerator)

        row2.addSpacing(20)

        devices_label = QLabel("Number of Devices:")
        devices_label.setFixedWidth(LABEL_WIDTH_COL2)
        row2.addWidget(devices_label)

        devices = QSpinBox()
        devices.setRange(1, 8)
        devices.setValue(1)
        devices.setMinimumWidth(60)
        self._fields["trainer_config.trainer_devices"] = devices
        row2.addWidget(devices)

        auto_devices = QCheckBox("Auto")
        auto_devices.setChecked(True)
        self._fields["_trainer_devices_auto"] = auto_devices
        row2.addWidget(auto_devices)

        # Connect checkbox to enable/disable spinbox
        auto_devices.stateChanged.connect(
            lambda state, sb=devices: sb.setEnabled(not state)
        )
        devices.setEnabled(False)

        row2.addStretch()
        layout.addLayout(row2)

        return group

    def _create_wandb_section(self) -> QGroupBox:
        """Create the WandB section (training only)."""
        group = QGroupBox("WandB")
        group.setMinimumWidth(self.BOX_MIN_WIDTH)

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Row 1: Status + buttons + Enable + Upload Viz
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        row1.addWidget(QLabel("Status:"))

        # Status label (will be updated by _update_wandb_status)
        api_key_status = QLabel()
        api_key_status.setStyleSheet("color: #666;")
        self._api_key_status_label = api_key_status
        row1.addWidget(api_key_status)

        # Copy button (only visible when not logged in)
        copy_btn = QPushButton("📋")
        copy_btn.setFixedSize(24, 24)
        copy_btn.setToolTip("Copy login command to clipboard")
        copy_btn.setFlat(True)
        copy_btn.clicked.connect(self._copy_wandb_login_command)
        self._wandb_copy_btn = copy_btn
        row1.addWidget(copy_btn)

        # Refresh button (only visible when not logged in)
        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.setToolTip("Check WandB login status")
        refresh_btn.setFlat(True)
        refresh_btn.clicked.connect(self._update_wandb_status)
        self._wandb_refresh_btn = refresh_btn
        row1.addWidget(refresh_btn)

        row1.addSpacing(20)

        enable_wandb = QCheckBox("Enable WandB for logging")
        self._fields["trainer_config.use_wandb"] = enable_wandb
        row1.addWidget(enable_wandb)

        row1.addSpacing(12)

        upload_viz = QCheckBox("Upload Viz")
        self._fields["trainer_config.wandb.save_viz_imgs_wandb"] = upload_viz
        row1.addWidget(upload_viz)

        row1.addSpacing(12)

        open_in_browser = QCheckBox("Open in browser")
        self._fields["gui.wandb_open_in_browser"] = open_in_browser
        row1.addWidget(open_in_browser)

        row1.addStretch()
        layout.addLayout(row1)

        # Hidden actual API key field (for form data)
        api_key = QLineEdit()
        api_key.setEchoMode(QLineEdit.Password)
        api_key.setVisible(False)
        self._fields["trainer_config.wandb.api_key"] = api_key

        # Fixed-width labels for alignment
        LABEL_WIDTH_COL1 = 95  # "Previous Run ID:" is longest
        LABEL_WIDTH_COL2 = 80  # "Project Name:" width

        # Row 3: Entity + Project (paired)
        row3 = QHBoxLayout()
        row3.setSpacing(8)

        entity_label = QLabel("Entity Name:")
        entity_label.setFixedWidth(LABEL_WIDTH_COL1)
        row3.addWidget(entity_label)

        entity = QLineEdit()
        entity.setMinimumWidth(120)
        self._fields["trainer_config.wandb.entity"] = entity
        row3.addWidget(entity)

        row3.addSpacing(20)

        project_label = QLabel("Project Name:")
        project_label.setFixedWidth(LABEL_WIDTH_COL2)
        row3.addWidget(project_label)

        project = QLineEdit()
        project.setMinimumWidth(120)
        self._fields["trainer_config.wandb.project"] = project
        row3.addWidget(project)

        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Previous Run ID + Group (paired)
        row4 = QHBoxLayout()
        row4.setSpacing(8)

        prev_run_label = QLabel("Previous Run ID:")
        prev_run_label.setFixedWidth(LABEL_WIDTH_COL1)
        row4.addWidget(prev_run_label)

        prev_run_id = QLineEdit()
        prev_run_id.setMinimumWidth(120)
        self._fields["trainer_config.wandb.prv_runid"] = prev_run_id
        row4.addWidget(prev_run_id)

        row4.addSpacing(20)

        group_name_label = QLabel("Group Name:")
        group_name_label.setFixedWidth(LABEL_WIDTH_COL2)
        row4.addWidget(group_name_label)

        group_name = QLineEdit()
        group_name.setMinimumWidth(120)
        self._fields["trainer_config.wandb.group"] = group_name
        row4.addWidget(group_name)

        row4.addStretch()
        layout.addLayout(row4)

        return group

    def _create_output_section(self) -> QGroupBox:
        """Create the Output section."""
        group = QGroupBox("Output")
        group.setMinimumWidth(self.BOX_MIN_WIDTH)

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # Row 1: Run Name
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        row1.addWidget(QLabel("Run Name:"))
        run_name = QLineEdit()
        self._fields["trainer_config.run_name"] = run_name
        row1.addWidget(run_name)

        layout.addLayout(row1)

        # Row 2: Runs Folder
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        row2.addWidget(QLabel("Runs Folder:"))
        runs_folder = QLineEdit()
        runs_folder.setText("models")
        self._fields["trainer_config.ckpt_dir"] = runs_folder
        row2.addWidget(runs_folder)

        layout.addLayout(row2)

        # Row 3: Checkpoint checkboxes
        row3 = QHBoxLayout()
        row3.setSpacing(8)

        row3.addWidget(QLabel("Checkpoint:"))

        save_best = QCheckBox("Best Model")
        save_best.setChecked(True)
        self._fields["trainer_config.save_ckpt"] = save_best
        row3.addWidget(save_best)

        save_latest = QCheckBox("Latest Model")
        self._fields["trainer_config.model_ckpt.save_last"] = save_latest
        row3.addWidget(save_latest)

        row3.addStretch()
        layout.addLayout(row3)

        # Row 4: Visualization checkboxes
        row4 = QHBoxLayout()
        row4.setSpacing(8)

        row4.addWidget(QLabel("Visualization:"))

        viz_preds = QCheckBox("Visualize Predictions")
        viz_preds.setChecked(True)
        self._fields["trainer_config.visualize_preds_during_training"] = viz_preds
        row4.addWidget(viz_preds)

        keep_viz = QCheckBox("Keep Viz Images")
        self._fields["trainer_config.keep_viz"] = keep_viz
        row4.addWidget(keep_viz)

        row4.addStretch()
        layout.addLayout(row4)

        return group

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    @property
    def fields(self) -> Dict[str, QWidget]:
        """Access field widgets by dotted key name."""
        return self._fields

    @property
    def current_pipeline_key(self) -> str:
        """Get current pipeline selection key (full name for internal use).

        Returns:
            Full pipeline key like "multi-animal top-down", "single animal", etc.
        """
        return self._pipeline_combo.currentData() or ""

    @property
    def current_pipeline(self) -> str:
        """Get current pipeline selection (normalized short name).

        Returns:
            Short pipeline name: "top-down", "bottom-up", "single",
            "top-down-id", or "bottom-up-id".
        """
        label = self._pipeline_combo.currentText()
        if "top-down" in label:
            if "id" not in label:
                return "top-down"
            else:
                return "top-down-id"
        if "bottom-up" in label:
            if "id" not in label:
                return "bottom-up"
            else:
                return "bottom-up-id"
        if "single" in label:
            return "single"
        return ""

    @current_pipeline.setter
    def current_pipeline(self, val: str):
        """Set pipeline by normalized short name.

        Args:
            val: Short pipeline name like "top-down", "bottom-up", etc.
        """
        if val not in (
            "top-down",
            "bottom-up",
            "single",
            "top-down-id",
            "bottom-up-id",
        ):
            return  # Ignore invalid values

        # Match short name to full pipeline name shown in menu
        for i in range(self._pipeline_combo.count()):
            option_text = self._pipeline_combo.itemText(i)
            if val in option_text:
                self._pipeline_combo.setCurrentIndex(i)
                break

    def get_form_data(self) -> Dict[str, Any]:
        """Return all field values as dotted key-value dict."""
        data = {"_pipeline": self.current_pipeline}

        # Get values from pipeline-specific fields (from current pipeline's widgets)
        pipeline_key = self.current_pipeline_key
        if pipeline_key in self._pipeline_fields:
            for key, widget in self._pipeline_fields[pipeline_key].items():
                data[key] = self._get_widget_value(widget)

        # Get values from non-pipeline fields (shared across all pipelines)
        for key, widget in self._fields.items():
            # Skip if already added from pipeline-specific fields
            if key not in data:
                data[key] = self._get_widget_value(widget)

        # Add frame target data
        data.update(self.frame_target_selector.get_form_data())

        # Consolidate tracking parameters based on selected tracker.
        # Form stores suffixed keys (tracking.match.flow), runners expects unsuffixed.
        tracker = data.get("tracking.tracker", "none")
        if tracker in ("flow", "simple"):
            tracking_fields = [
                "tracking.match",
                "tracking.similarity",
                "tracking.track_window",
                "tracking.max_tracks",
                "tracking.max_tracks_disabled",
                "tracking.post_connect_single_breaks",
                "tracking.robust",
                "tracking.robust_disabled",
            ]
            for field in tracking_fields:
                suffixed_key = f"{field}.{tracker}"
                if suffixed_key in data:
                    data[field] = data[suffixed_key]

            # Handle max_tracks: if "no limit" is checked, set to None
            if data.get("tracking.max_tracks_disabled", True):
                data["tracking.max_tracks"] = None

            # Handle robust: if "use max" is checked, set to 1.0 (non-robust)
            if data.get("tracking.robust_disabled", True):
                data["tracking.robust"] = 1.0

        # Handle max_instances: if "no max" is checked, set to None
        if data.get("_max_instances_disabled", False):
            data["_max_instances"] = None

        # Handle batch_size: if "Default" is checked, omit so CLI uses model default
        if data.get("_batch_size_default", True):
            data.pop("_batch_size", None)

        # Strip placeholder from API key if user didn't change it
        api_key = data.get("trainer_config.wandb.api_key")
        if api_key and self._wandb_api_key_placeholder:
            if api_key == self._wandb_api_key_placeholder:
                data["trainer_config.wandb.api_key"] = None

        # Save preferences
        if self._mode == "training":
            self._save_training_preferences(data)

        return data

    def set_form_data(self, data: Dict[str, Any]):
        """Set field values from dotted key-value dict."""
        # Set pipeline first
        if "_pipeline" in data:
            self.current_pipeline = data["_pipeline"]

        for key, value in data.items():
            # Set in pipeline-specific fields (update ALL pipelines that have this key)
            for pipeline_key, fields in self._pipeline_fields.items():
                if key in fields:
                    self._set_widget_value(fields[key], value)

            # Also set in _fields for non-pipeline fields
            if key in self._fields:
                # Only set if not a pipeline-specific field (avoid double-set)
                is_pipeline_field = any(
                    key in fields for fields in self._pipeline_fields.values()
                )
                if not is_pipeline_field:
                    self._set_widget_value(self._fields[key], value)

    def _get_widget_value(self, widget: QWidget) -> Any:
        """Get value from a widget."""
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QLineEdit):
            text = widget.text().strip()
            return text if text else None
        return None

    def _set_widget_value(self, widget: QWidget, value: Any):
        """Set value on a widget."""
        if isinstance(widget, QComboBox):
            idx = widget.findData(value)
            if idx >= 0:
                widget.setCurrentIndex(idx)
            elif isinstance(value, str):
                idx = widget.findText(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            if value is not None:
                widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value) if value is not None else 0.0)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value) if value is not None else "")

    def set_node_options(self, node_names: List[str]):
        """Populate node dropdown fields (for anchor part selection)."""
        for key, widget in self._fields.items():
            if "anchor_part" in key and isinstance(widget, QComboBox):
                current = widget.currentData()
                widget.clear()
                widget.addItem("", None)  # Empty = use bounding box midpoint
                for name in node_names:
                    widget.addItem(name, name)
                # Restore selection if possible
                if current:
                    idx = widget.findData(current)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)

    def emitPipeline(self):
        """Emit updatePipeline signal with current pipeline."""
        self.updatePipeline.emit(self.current_pipeline)

    # -------------------------------------------------------------------------
    # Preferences Management
    # -------------------------------------------------------------------------

    def _update_wandb_status(self):
        """Check and update the WandB login status display."""
        is_logged_in, auth_source, username = check_wandb_login_status()

        if not hasattr(self, "_api_key_status_label"):
            return

        # WandB option fields to enable/disable based on login status
        wandb_option_fields = [
            "trainer_config.use_wandb",
            "trainer_config.wandb.save_viz_imgs_wandb",
            "gui.wandb_open_in_browser",
            "trainer_config.wandb.entity",
            "trainer_config.wandb.project",
            "trainer_config.wandb.prv_runid",
            "trainer_config.wandb.group",
        ]

        if is_logged_in:
            # Show logged in status
            if auth_source == "WANDB_API_KEY environment variable":
                status_text = "Logged in via env var ✓"
            else:
                status_text = "Logged in ✓"

            self._api_key_status_label.setText(status_text)
            self._api_key_status_label.setStyleSheet("color: #2e7d32;")  # Green

            # Hide copy and refresh buttons when logged in
            if hasattr(self, "_wandb_copy_btn"):
                self._wandb_copy_btn.setVisible(False)
            if hasattr(self, "_wandb_refresh_btn"):
                self._wandb_refresh_btn.setVisible(False)

            # Enable WandB option fields when logged in
            for field_name in wandb_option_fields:
                field = self._fields.get(field_name)
                if field is not None:
                    field.setEnabled(True)

            # Update hidden API key field
            api_key_field = self._fields.get("trainer_config.wandb.api_key")
            if api_key_field is not None:
                placeholder = f"(using {auth_source})"
                api_key_field.setText(placeholder)
                api_key_field.setToolTip(
                    get_wandb_api_key_help_text(is_logged_in, auth_source)
                )
                self._wandb_api_key_placeholder = placeholder
        else:
            # Show login instructions
            self._api_key_status_label.setText("Login with: uvx wandb login")
            self._api_key_status_label.setStyleSheet("color: #666;")

            # Show copy and refresh buttons when not logged in
            if hasattr(self, "_wandb_copy_btn"):
                self._wandb_copy_btn.setVisible(True)
            if hasattr(self, "_wandb_refresh_btn"):
                self._wandb_refresh_btn.setVisible(True)

            # Disable WandB option fields when not logged in
            for field_name in wandb_option_fields:
                field = self._fields.get(field_name)
                if field is not None:
                    field.setEnabled(False)

            # Also uncheck the wandb-related checkboxes to prevent
            # wandb being enabled in config when not logged in
            checkbox_fields = [
                "trainer_config.use_wandb",
                "trainer_config.wandb.save_viz_imgs_wandb",
                "gui.wandb_open_in_browser",
            ]
            for field_name in checkbox_fields:
                field = self._fields.get(field_name)
                if field is not None and isinstance(field, QCheckBox):
                    field.setChecked(False)

            # Clear any placeholder
            self._wandb_api_key_placeholder = None
            api_key_field = self._fields.get("trainer_config.wandb.api_key")
            if api_key_field is not None:
                api_key_field.clear()
                api_key_field.setToolTip(get_wandb_api_key_help_text(False, None))

    def _copy_wandb_login_command(self):
        """Copy the WandB login command to clipboard."""
        clipboard = QGuiApplication.clipboard()
        clipboard.setText("uvx wandb login")

    def _init_training_settings(self):
        """Initialize training pipeline settings from preferences."""
        training_prefs = {
            "_data_pipeline_fw": prefs["training data pipeline framework"],
            "trainer_config.train_data_loader.num_workers": prefs[
                "training num workers"
            ],
            "trainer_config.trainer_accelerator": prefs["training accelerator"],
        }
        # trainer_devices is optional_int - only set if not None
        if prefs["training num devices"] is not None:
            training_prefs["trainer_config.trainer_devices"] = prefs[
                "training num devices"
            ]
            # Uncheck Auto if we have a saved devices value
            if "_trainer_devices_auto" in self._fields:
                self._set_widget_value(self._fields["_trainer_devices_auto"], False)

        for key, value in training_prefs.items():
            if value is not None and key in self._fields:
                self._set_widget_value(self._fields[key], value)

    def _save_training_preferences(self, form_data: dict):
        """Save training pipeline settings to preferences."""
        pref_mapping = {
            "_data_pipeline_fw": "training data pipeline framework",
            "trainer_config.train_data_loader.num_workers": "training num workers",
            "trainer_config.trainer_devices": "training num devices",
            "trainer_config.trainer_accelerator": "training accelerator",
        }
        changed = False
        for form_key, pref_key in pref_mapping.items():
            if form_key in form_data:
                value = form_data[form_key]
                if prefs[pref_key] != value:
                    prefs[pref_key] = value
                    changed = True
        if changed:
            prefs.save()
