"""
Dialogs for running training and/or inference in GUI.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Text, cast

from qtpy import QtCore, QtGui, QtWidgets

import sleap
from omegaconf import OmegaConf
from sleap_io import Labels, Video, Skeleton, load_file
from sleap.gui.config_utils import (
    get_omegaconf_from_gui_form,
    apply_cfg_transforms_to_key_val_dict,
    find_backbone_name_from_key_val_dict,
    get_keyval_dict_from_omegaconf,
    filter_cfg,
)
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.dialogs.formbuilder import YamlFormWidget
from sleap.gui.learning import receptivefield, runners, configs
from sleap.gui.learning.configs import TrainingConfigsGetter
from sleap.sleap_io_adaptors.skeleton_utils import (
    cycles,
    is_arborescence,
    root_nodes,
    in_degree_over_one,
)
from sleap.sleap_io_adaptors.lf_labels_utils import instances
from sleap.util import show_sleap_nn_installation_message
from sleap.gui.widgets.frame_target_selector import (
    FrameTargetOption,
    FrameTargetSelection,
)
from sleap.gui.learning.main_tab import MainTabWidget
from sleap.gui.learning.wandb_utils import check_wandb_login_status

# List of fields which should show list of skeleton nodes
NODE_LIST_FIELDS = [
    "model.model_config.head_configs.centered_instance.confmaps.anchor_part",
    "model.model_config.head_configs.centroid.confmaps.anchor_part",
    "model.model_config.head_configs.multi_class_topdown.confmaps.anchor_part",
]


class LearningDialog(QtWidgets.QDialog):
    """
    Dialog for running training and/or inference.

    The dialog shows tabs for configuring the pipeline (
    :py:class:`MainTabWidget`) and, depending on the pipeline, for
    each specific model (:py:class:`TrainingEditorWidget`).

    In training mode, the model hyperpameters are editable unless you're using
    a trained model; they are read-only in inference mode.

    Arguments:
        mode: either "training" or "inference".
        labels_filename: path to labels file, used for default location to
            save models.
        labels: the `Labels` object (can also be loaded from given filename)
        skeleton: the `Skeleton` object (can also be taken from `Labels`), used
            for list of nodes for (e.g.) selecting anchor node
    """

    _handle_learning_finished = QtCore.Signal(int)
    navigate_to_instance = QtCore.Signal(
        int, int, int
    )  # video_idx, frame_idx, instance_idx

    # Class-level cache for pipeline tab form state.
    # Persists across dialog sessions so values are restored after Cancel.
    # Key: (mode, labels_filename, tab_name), Value: form data dict
    _cached_tab_state: Dict[tuple, dict] = {}

    def __init__(
        self,
        mode: Text,
        labels_filename: Text,
        labels: Optional[Labels] = None,
        skeleton: Optional["Skeleton"] = None,
        *args,
        **kwargs,
    ):
        super(LearningDialog, self).__init__()

        # Set window title based on mode
        mode_title = "Training" if mode == "training" else "Inference"
        self.setWindowTitle(
            f"{mode_title} Configuration - SLEAP v{sleap.version.__version__}"
        )

        if labels is None:
            labels = load_file(labels_filename)

        if skeleton is None and labels.skeletons:
            skeleton = labels.skeletons[0]

        self.mode = mode
        self.labels_filename = labels_filename
        self.labels = labels
        self.skeleton = skeleton

        self._frame_selection = None

        self.current_pipeline = ""

        self.tabs: Dict[str, TrainingEditorWidget] = dict()
        self.shown_tab_names = []

        self._cfg_getter = configs.TrainingConfigsGetter.make_from_labels_filename(
            labels_filename=self.labels_filename
        )

        # Layout for buttons
        buttons = QtWidgets.QDialogButtonBox()
        self.copy_button = buttons.addButton(
            "Copy to clipboard", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.save_button = buttons.addButton(
            "Save configuration files...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.export_button = buttons.addButton(
            "Export training job package...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.cancel_button = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.run_button = buttons.addButton("Run", QtWidgets.QDialogButtonBox.ApplyRole)

        self.copy_button.setToolTip("Copy configuration to the clipboard")
        self.save_button.setToolTip("Save scripts and configuration to run pipeline.")
        self.export_button.setToolTip(
            "Export data, configuration, and scripts for remote training and inference."
        )
        self.run_button.setToolTip("Run pipeline locally (GPU recommended).")

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(buttons, alignment=QtCore.Qt.AlignTop)

        buttons_layout_widget = QtWidgets.QWidget()
        buttons_layout_widget.setLayout(buttons_layout)

        self.pipeline_form_widget = MainTabWidget(mode=mode, skeleton=skeleton)
        if mode == "training":
            tab_label = "Training Pipeline"
        elif mode == "inference":
            tab_label = "Inference Pipeline"
        else:
            raise ValueError(f"Invalid LearningDialog mode: {mode}")

        self.tab_widget = QtWidgets.QTabWidget()

        self.tab_widget.addTab(self.pipeline_form_widget, tab_label)
        self.make_tabs()

        self.message_widget = QtWidgets.QLabel("")
        self.message_widget.setWordWrap(True)

        # Frame target selector is now owned by MainTabWidget
        self.frame_target_selector = self.pipeline_form_widget.frame_target_selector
        self._target_selection_user_changed = False

        # Layout for entire dialog - single scrollable area (same for both modes)
        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.addWidget(self.tab_widget)
        content_layout.addWidget(self.message_widget)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # Main layout: scrollable content + buttons
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(scroll_area)
        layout.addWidget(buttons_layout_widget)

        self.adjust_initial_size()

        # Default to most recently trained pipeline (if there is one)
        self.set_default_pipeline_tab()

        # Connect functions to update pipeline tabs when pipeline changes
        self.pipeline_form_widget.updatePipeline.connect(self.set_pipeline)
        self.pipeline_form_widget.emitPipeline()

        self.connect_signals()

        # Track when user changes the frame target selector
        self.frame_target_selector.valueChanged.connect(
            self._on_target_selection_changed
        )

        # Connect actions for buttons
        self.copy_button.clicked.connect(self.copy)
        self.save_button.clicked.connect(lambda: self.save())
        self.export_button.clicked.connect(lambda: self.export_package())
        self.cancel_button.clicked.connect(self.reject)
        self.run_button.clicked.connect(self.run)

    def adjust_initial_size(self):
        """Set initial dialog size based on mode and screen size.

        V9 Layout: Both modes use single-column layout (no side panel)
        - Training: 880x900 (more sections, needs more height)
        - Inference: 880x850
        """
        screen = QtGui.QGuiApplication.primaryScreen().availableGeometry()

        if self.mode == "training":
            max_width = 880
            max_height = 900
        else:  # inference
            max_width = 880
            max_height = 850

        margin = 0.05  # 5% margin from screen edge

        # Calculate target width and height
        target_width = min(screen.width() - screen.width() * margin, max_width)
        target_height = min(screen.height() - screen.height() * margin, max_height)
        # Set the dialog's dimensions
        self.resize(int(target_width), int(target_height))

        # Note: minimum width is set dynamically in _update_minimum_width()
        # after tabs are added, since TrainingEditorWidget tabs may be wider
        # than the main tab.

    def _update_minimum_width(self):
        """Update dialog minimum width based on tab content.

        Called after tabs are added/changed to ensure the dialog cannot be
        resized smaller than its content requires. This is especially important
        on Windows 11 where Qt doesn't always propagate minimum size constraints
        from child widgets through scroll areas.
        """
        # Get minimum width from the tab widget which accounts for all tabs
        # Use sizeHint as it reflects the preferred size including all content
        tab_width = self.tab_widget.sizeHint().width()

        # Add margins for dialog layout and window frame decorations
        # The scroll area and content layout add some padding
        min_width = tab_width + 40

        # Floor at the main tab's requirement as a safety minimum
        min_width = max(min_width, MainTabWidget.BOX_MIN_WIDTH + 50)

        self.setMinimumWidth(min_width)

    def showEvent(self, event):
        """Handle dialog show event.

        Updates minimum width after the dialog is shown and layout is computed.
        This ensures accurate size calculations on all platforms.
        """
        super().showEvent(event)
        # Defer minimum width update to after event processing completes
        # to ensure layout is fully computed
        QtCore.QTimer.singleShot(0, self._update_minimum_width)

    def closeEvent(self, event):
        """Handle dialog close event.

        Saves current form state for all pipeline tabs before closing.
        This allows values to persist across Cancel operations - when the
        dialog is reopened, the cached values are restored.
        """
        # Save form state for all initialized tabs
        for tab_name, tab_widget in self.tabs.items():
            cache_key = (self.mode, self.labels_filename, tab_name)
            LearningDialog._cached_tab_state[cache_key] = tab_widget.get_all_form_data()

        super().closeEvent(event)

    def update_file_lists(self):
        """Update config file lists for all currently shown tabs.

        With lazy tab creation, we only update tabs that are currently visible.
        Tabs that haven't been created yet will get their file lists updated
        when they're first initialized in _ensure_tab_initialized().
        """
        self._cfg_getter.update()
        # Only update tabs that are currently shown (and thus initialized)
        for tab_name in self.shown_tab_names:
            if tab_name in self.tabs:
                self.tabs[tab_name].update_file_list()

    @staticmethod
    def count_total_frames_for_selection_option(
        videos_frames: Dict[Video, List[int]],
    ) -> int:
        if not videos_frames:
            return 0

        count = 0
        for frame_list in videos_frames.values():
            # Check for [X, Y) range given as (X, -Y) tuple
            if len(frame_list) == 2 and frame_list[1] < 0:
                count += -frame_list[1] - frame_list[0]
            elif frame_list != (0, 0):
                count += len(frame_list)

        return count

    @property
    def frame_selection(self) -> Dict[str, Dict[Video, List[int]]]:
        """
        Returns dictionary with frames that user has selected for learning.
        """
        return self._frame_selection

    @frame_selection.setter
    def frame_selection(self, frame_selection: Dict[str, Dict[Video, List[int]]]):
        """Sets options of frames on which to run learning."""
        self._frame_selection = frame_selection

        # Update frame target selector with options
        self._update_frame_target_selector()

    def _update_frame_target_selector(self):
        """Update frame target selector widget with current frame selection options."""
        if self._frame_selection is None:
            return

        # Build options for the new selector
        options = {}

        # Calculate frame counts for each option
        frame_count = self.count_total_frames_for_selection_option(
            self._frame_selection.get("frame", {})
        )
        options["frame"] = FrameTargetOption(
            key="frame",
            label="Current frame",
            description="Predict on just this frame",
            frame_count=frame_count,
        )

        if "clip" in self._frame_selection:
            frame_count = self.count_total_frames_for_selection_option(
                self._frame_selection["clip"]
            )
            options["clip"] = FrameTargetOption(
                key="clip",
                label="Selected clip",
                description="Predict on the frame range you selected",
                frame_count=frame_count,
                available=frame_count > 0,
            )

        if "video" in self._frame_selection:
            frame_count = self.count_total_frames_for_selection_option(
                self._frame_selection["video"]
            )
            options["video"] = FrameTargetOption(
                key="video",
                label="Entire video",
                description="Predict on all frames in current video",
                frame_count=frame_count,
            )

        if "all_videos" in self._frame_selection and len(self.labels.videos) > 1:
            frame_count = self.count_total_frames_for_selection_option(
                self._frame_selection["all_videos"]
            )
            options["all_videos"] = FrameTargetOption(
                key="all_videos",
                label="All videos",
                description="Predict on every frame across all videos",
                frame_count=frame_count,
            )

        # For random sample options, use sampling logic with default sample count
        # The actual count will be updated when settings change
        default_sample_count = 20

        if "random_video" in self._frame_selection:
            frame_count = self._sample_frames_from_pool(
                self._frame_selection["random_video"],
                default_sample_count,
                exclude_user_labeled=False,
            )
            options["random_video"] = FrameTargetOption(
                key="random_video",
                label="Random sample (current video)",
                description="Random frames from current video",
                frame_count=frame_count,
            )

        if "random" in self._frame_selection:
            frame_count = self._sample_frames_from_pool(
                self._frame_selection["random"],
                default_sample_count,
                exclude_user_labeled=False,
            )
            options["random"] = FrameTargetOption(
                key="random",
                label="Random sample (all videos)",
                description="Random frames from all videos",
                frame_count=frame_count,
            )

        if "suggestions" in self._frame_selection:
            frame_count = self.count_total_frames_for_selection_option(
                self._frame_selection["suggestions"]
            )
            if frame_count > 0:
                options["suggestions"] = FrameTargetOption(
                    key="suggestions",
                    label="Suggestions",
                    description="Frames in the Labeling Suggestions list",
                    frame_count=frame_count,
                )

        if "user" in self._frame_selection:
            frame_count = self.count_total_frames_for_selection_option(
                self._frame_selection["user"]
            )
            if frame_count > 0:
                options["user_labeled"] = FrameTargetOption(
                    key="user_labeled",
                    label="User labeled",
                    description="Frames you've annotated (for evaluation)",
                    frame_count=frame_count,
                )

        if "predicted" in self._frame_selection:
            frame_count = self.count_total_frames_for_selection_option(
                self._frame_selection["predicted"]
            )
            if frame_count > 0:
                options["predicted"] = FrameTargetOption(
                    key="predicted",
                    label="Frames with predictions",
                    description="Only frames that already have predictions",
                    frame_count=frame_count,
                )

        # Add "nothing" option for training mode only
        if self.mode == "training":
            options = {
                "nothing": FrameTargetOption(
                    key="nothing",
                    label="Nothing",
                    description="Skip predictions, training only",
                    frame_count=0,
                    training_only=True,
                ),
                **options,
            }

        self.frame_target_selector.set_options(options)

        # Set default selection based on precedence rules
        self._set_frame_target_default(options)

    def _set_frame_target_default(self, options: Dict[str, FrameTargetOption]):
        """Set default frame target selection based on precedence rules.

        Priority order:
        1. User selection this session - preserve user's explicit choice
        2. Selected clip - if a clip is selected in the timeline
        3. Suggestions - if there are suggested frames available
        4. Current frame - fallback for both training and inference
        """
        if self._target_selection_user_changed:
            # User already made a selection - keep it if still available
            current = self.frame_target_selector.get_selection()
            if current.target_key in options:
                return  # Keep current selection

        # Check for selected clip (highest priority after user selection)
        if "clip" in options and options["clip"].frame_count > 0:
            self.frame_target_selector.set_selection(
                FrameTargetSelection(target_key="clip")
            )
            return

        # Check for suggestions
        if "suggestions" in options and options["suggestions"].frame_count > 0:
            self.frame_target_selector.set_selection(
                FrameTargetSelection(target_key="suggestions")
            )
            return

        # Fallback: current frame for both training and inference
        if "frame" in options:
            self.frame_target_selector.set_selection(
                FrameTargetSelection(target_key="frame")
            )

    def _on_target_selection_changed(self):
        """Track when user explicitly changes the frame target selection.

        Also updates frame counts for random sample options based on current
        settings (sample count spinbox, exclude user labeled checkbox).
        """
        self._target_selection_user_changed = True
        self._update_random_sample_frame_counts()

    def _get_user_labeled_frame_indices(self, video: Video) -> Set[int]:
        """Get set of frame indices that have user-labeled instances for a video."""
        return {
            lf.frame_idx for lf in self.labels.user_labeled_frames if lf.video == video
        }

    def _sample_frames_from_pool(
        self,
        candidate_pool: Dict[Video, List[int]],
        sample_count: int,
        exclude_user_labeled: bool,
    ) -> int:
        """Sample frames from a candidate pool and return the sampled count.

        This implements the "filter then sample" approach, which maintains
        the target sample count when possible (unlike "sample then filter").

        Args:
            candidate_pool: Dict mapping videos to lists of candidate frame indices.
            sample_count: Target number of frames to sample.
            exclude_user_labeled: If True, exclude user-labeled frames first.

        Returns:
            The actual number of frames that would be sampled.
        """
        total_sampled = 0

        for video, candidates in candidate_pool.items():
            if not candidates:
                continue

            # Convert to set for efficient filtering
            available = set(candidates)

            # Filter out user-labeled frames if requested
            if exclude_user_labeled:
                user_labeled = self._get_user_labeled_frame_indices(video)
                available = available - user_labeled

            # Sample up to sample_count from available frames
            actual_sample_size = min(sample_count, len(available))
            total_sampled += actual_sample_size

        return total_sampled

    def _update_random_sample_frame_counts(self):
        """Update frame counts for random sample options based on current settings."""
        if self._frame_selection is None:
            return

        selection = self.frame_target_selector.get_selection()
        sample_count = selection.sample_count
        exclude_user_labeled = selection.exclude_user_labeled

        # Update random_video option
        if "random_video" in self._frame_selection:
            count = self._sample_frames_from_pool(
                self._frame_selection["random_video"],
                sample_count,
                exclude_user_labeled,
            )
            self.frame_target_selector.update_option_frame_count("random_video", count)

        # Update random (all videos) option
        if "random" in self._frame_selection:
            count = self._sample_frames_from_pool(
                self._frame_selection["random"],
                sample_count,
                exclude_user_labeled,
            )
            self.frame_target_selector.update_option_frame_count("random", count)

    def connect_signals(self):
        """Connect valueChanged signals for pipeline and any existing tabs.

        Note: With lazy tab creation, tabs may not exist yet at dialog startup.
        Signals for lazily-created tabs are connected in _ensure_tab_initialized().
        """
        self.pipeline_form_widget.valueChanged.connect(self.on_tab_data_change)

        # Only connect signals for tabs that already exist
        for head_name, tab in self.tabs.items():
            tab.valueChanged.connect(lambda n=head_name: self.on_tab_data_change(n))

    def disconnect_signals(self):
        """Disconnect valueChanged signals from pipeline and tabs.

        Uses try/except to handle cases where signals may not be connected
        (e.g., with lazy tab creation, some tabs may not exist yet).
        Warnings are suppressed since Qt prints RuntimeWarning before the
        exception can be caught.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                self.pipeline_form_widget.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signal was not connected

            for head_name, tab in self.tabs.items():
                try:
                    tab.valueChanged.disconnect()
                except (TypeError, RuntimeError):
                    pass  # Signal was not connected

    def make_tabs(self):
        """Initialize tab tracking without creating widgets yet (lazy loading).

        TrainingEditorWidget instances are created on-demand when tabs are first
        shown via add_tab(). This significantly reduces dialog startup time by
        avoiding creation of ~6 complex widgets upfront (~200-230ms each).
        """
        # Define available head types - widgets created lazily in add_tab()
        self._head_types = (
            "single_instance",
            "centroid",
            "centered_instance",
            "bottomup",
            "multi_class_topdown",
            "multi_class_bottomup",
        )
        # tabs dict will be populated lazily as tabs are added

    def _ensure_tab_initialized(self, head_name: str) -> "TrainingEditorWidget":
        """Create TrainingEditorWidget for a head type if not already created.

        This implements lazy tab creation - widgets are only created when the
        tab is first added to the UI, not at dialog startup.

        Args:
            head_name: The head type (e.g., "centroid", "centered_instance")

        Returns:
            The TrainingEditorWidget for this head type.
        """
        if head_name not in self.tabs:
            video = self.labels.videos[0] if self.labels else None
            widget = TrainingEditorWidget(
                video=video,
                skeleton=self.skeleton,
                head=head_name,
                cfg_getter=self._cfg_getter,
                require_trained=(self.mode == "inference"),
                labels=self.labels,
                parent_dialog=self,
            )
            self.tabs[head_name] = widget

            # Connect signals for the newly created tab
            widget.valueChanged.connect(lambda n=head_name: self.on_tab_data_change(n))

            # Update file list for the newly created tab
            widget.update_file_list()

            # Restore cached form state if available (persists across Cancel)
            cache_key = (self.mode, self.labels_filename, head_name)
            if cache_key in LearningDialog._cached_tab_state:
                cached_data = LearningDialog._cached_tab_state[cache_key]
                widget.set_fields_from_key_val_dict(cached_data)

        return self.tabs[head_name]

    def adjust_data_to_update_other_tabs(self, source_data, updated_data=None):
        if updated_data is None:
            updated_data = source_data

        anchor_part = None
        set_anchor = False

        if "model_config.head_configs.centroid.confmaps.anchor_part" in source_data:
            anchor_part = source_data[
                "model_config.head_configs.centroid.confmaps.anchor_part"
            ]
            set_anchor = True
        elif (
            "model_config.head_configs.centered_instance.confmaps.anchor_part"
            in source_data
        ):
            anchor_part = source_data[
                "model_config.head_configs.centered_instance.confmaps.anchor_part"
            ]
            set_anchor = True
        elif (
            "model_config.head_configs.multi_class_topdown.confmaps.anchor_part"
            in source_data
        ):
            anchor_part = source_data[
                "model_config.head_configs.multi_class_topdown.confmaps.anchor_part"
            ]
            set_anchor = True

        # Use None instead of empty string/list
        anchor_part = anchor_part or None

        if set_anchor:
            updated_data["model_config.head_configs.centroid.confmaps.anchor_part"] = (
                anchor_part
            )
            updated_data[
                "model_config.head_configs.centered_instance.confmaps.anchor_part"
            ] = anchor_part
            updated_data[
                "model_config.head_configs.multi_class_topdown.confmaps.anchor_part"
            ] = anchor_part

    def update_tabs_from_pipeline(self, source_data):
        self.adjust_data_to_update_other_tabs(source_data)

        for tab in self.tabs.values():
            tab.set_fields_from_key_val_dict(source_data)

    def update_tabs_from_tab(self, source_data):
        data_to_transfer = dict()
        self.adjust_data_to_update_other_tabs(source_data, data_to_transfer)

        if data_to_transfer:
            for tab in self.tabs.values():
                tab.set_fields_from_key_val_dict(data_to_transfer)

    def on_tab_data_change(self, tab_name=None):
        self.disconnect_signals()

        if tab_name is None:
            # Move data from pipeline tab to other tabs
            source_data = self.pipeline_form_widget.get_form_data()
            self.update_tabs_from_pipeline(source_data)
        else:
            # Get data from head-specific tab
            source_data = self.tabs[tab_name].get_all_form_data()

            self.update_tabs_from_tab(source_data)

            # Update pipeline tab, but filter out run_name to prevent cross-tab
            # contamination (each head has its own run_name from its base config,
            # but the pipeline run_name should remain independent)
            pipeline_data = {
                k: v for k, v in source_data.items() if k != "trainer_config.run_name"
            }
            self.pipeline_form_widget.set_form_data(pipeline_data)

        self._validate_pipeline()

        self.connect_signals()

    def get_most_recent_pipeline_trained(self) -> Text:
        recent_cfg_info = self._cfg_getter.get_first()

        if recent_cfg_info and recent_cfg_info.head_name:
            if recent_cfg_info.head_name in ("multi_class_topdown",):
                return "top-down-id"
            if recent_cfg_info.head_name in ("centroid", "centered_instance"):
                return "top-down"
            if recent_cfg_info.head_name in ("bottomup",):
                return "bottom-up"
            if recent_cfg_info.head_name in ("single_instance",):
                return "single"
            if recent_cfg_info.head_name in ("multi_class_bottomup",):
                return "bottom-up-id"
        return ""

    def _get_head_names_for_pipeline(self, pipeline: str) -> List[str]:
        """Get the head name(s) associated with a pipeline.

        Args:
            pipeline: Pipeline name (e.g., "top-down", "bottom-up").

        Returns:
            List of head names for this pipeline.
        """
        pipeline_to_heads = {
            "top-down": ["centroid", "centered_instance"],
            "bottom-up": ["bottomup"],
            "top-down-id": ["centroid", "multi_class_topdown"],
            "bottom-up-id": ["multi_class_bottomup"],
            "single": ["single_instance"],
        }
        return pipeline_to_heads.get(pipeline, [])

    def _get_trained_config_for_pipeline(
        self, pipeline: str
    ) -> Optional[configs.ConfigFileInfo]:
        """Get the most recent trained config for a pipeline.

        Args:
            pipeline: Pipeline name (e.g., "top-down", "bottom-up").

        Returns:
            ConfigFileInfo if a trained config exists, None otherwise.
        """
        head_names = self._get_head_names_for_pipeline(pipeline)
        for head_name in head_names:
            trained_cfgs = self._cfg_getter.get_filtered_configs(
                head_filter=head_name, only_trained=True
            )
            if trained_cfgs:
                return trained_cfgs[0]
        return None

    def _get_video_channels_default(self) -> str:
        """Determine default image conversion based on video channels.

        Returns:
            "RGB" if all videos are RGB, "grayscale" if all are grayscale,
            empty string if mixed or unknown.
        """
        if not self.labels or not self.labels.videos:
            return ""

        from sleap.sleap_io_adaptors.video_utils import video_get_channels

        channels_set = set()
        for video in self.labels.videos:
            try:
                channels = video_get_channels(video)
                channels_set.add(channels)
            except Exception:
                # If we can't determine channels, skip this video
                pass

        if len(channels_set) == 1:
            channels = channels_set.pop()
            if channels == 1:
                return "grayscale"
            elif channels == 3:
                return "RGB"

        return ""

    def _apply_pipeline_defaults(self, pipeline: str):
        """Apply defaults from previously trained config or video analysis.

        This sets image conversion and WandB defaults based on:
        1. Previously trained config for this pipeline (if exists)
        2. Video channel analysis (for image conversion only)

        Args:
            pipeline: Pipeline name being switched to.
        """
        if self.mode != "training":
            return

        # Try to get a trained config for this pipeline
        trained_cfg = self._get_trained_config_for_pipeline(pipeline)

        defaults_to_apply = {}
        used_trained_config = False

        if trained_cfg and trained_cfg.config:
            cfg = trained_cfg.config

            # Only use OmegaConf if cfg is actually an OmegaConf object
            if OmegaConf.is_config(cfg):
                used_trained_config = True

                # Image conversion from previous config
                ensure_rgb = OmegaConf.select(
                    cfg, "data_config.preprocessing.ensure_rgb", default=None
                )
                ensure_grayscale = OmegaConf.select(
                    cfg, "data_config.preprocessing.ensure_grayscale", default=None
                )
                if ensure_rgb:
                    defaults_to_apply["_ensure_channels"] = "RGB"
                elif ensure_grayscale:
                    defaults_to_apply["_ensure_channels"] = "grayscale"

                # WandB settings from previous config (except run_name)
                # Only apply if user is logged in to prevent enabling wandb
                # when credentials are not available (which would trigger an
                # interactive login prompt that stalls training)
                is_wandb_logged_in, _, _ = check_wandb_login_status()
                if is_wandb_logged_in:
                    use_wandb = OmegaConf.select(
                        cfg, "trainer_config.use_wandb", default=None
                    )
                    if use_wandb is not None:
                        defaults_to_apply["trainer_config.use_wandb"] = use_wandb

                    wandb_entity = OmegaConf.select(
                        cfg, "trainer_config.wandb.entity", default=None
                    )
                    if wandb_entity:
                        defaults_to_apply["trainer_config.wandb.entity"] = wandb_entity

                    wandb_project = OmegaConf.select(
                        cfg, "trainer_config.wandb.project", default=None
                    )
                    if wandb_project:
                        defaults_to_apply["trainer_config.wandb.project"] = (
                            wandb_project
                        )

                    wandb_group = OmegaConf.select(
                        cfg, "trainer_config.wandb.group", default=None
                    )
                    if wandb_group:
                        defaults_to_apply["trainer_config.wandb.group"] = wandb_group

                    save_viz = OmegaConf.select(
                        cfg, "trainer_config.wandb.save_viz_imgs_wandb", default=None
                    )
                    if save_viz is not None:
                        key = "trainer_config.wandb.save_viz_imgs_wandb"
                        defaults_to_apply[key] = save_viz

        if not used_trained_config:
            # No trained config - use video channel analysis for image conversion
            video_default = self._get_video_channels_default()
            if video_default:
                defaults_to_apply["_ensure_channels"] = video_default

        # Apply the defaults
        if defaults_to_apply:
            self.pipeline_form_widget.set_form_data(defaults_to_apply)

    def set_default_pipeline_tab(self):
        recent_pipeline_name = self.get_most_recent_pipeline_trained()
        if recent_pipeline_name:
            self.pipeline_form_widget.current_pipeline = recent_pipeline_name
        else:
            # Set default based on detection of single- vs multi-animal project.
            max_user_instance = 0
            for lf in self.labels:
                max_user_instance = max(max_user_instance, len(lf.user_instances))

            if max_user_instance == 1:
                self.pipeline_form_widget.current_pipeline = "single"
            else:
                self.pipeline_form_widget.current_pipeline = "top-down"

    def add_tab(self, tab_name):
        """Add a tab to the dialog, creating the widget lazily if needed.

        This method is idempotent - calling it multiple times with the same
        tab_name will only add the tab once (prevents issues with signal
        re-entrancy during widget construction).
        """
        # Prevent duplicate additions (can happen due to signal re-entrancy)
        if tab_name in self.shown_tab_names:
            return

        tab_labels = {
            "single_instance": "Single Instance Model Configuration",
            "centroid": "Centroid Model Configuration",
            "centered_instance": "Centered Instance Model Configuration",
            "bottomup": "Bottom-Up Model Configuration",
            "multi_class_topdown": "Top-Down-Id Model Configuration",
            "multi_class_bottomup": "Bottom-Up-Id Model Configuration",
        }
        # Mark as shown first to prevent re-entrancy issues
        self.shown_tab_names.append(tab_name)
        # Lazily create the widget if it doesn't exist yet
        widget = self._ensure_tab_initialized(tab_name)
        self.tab_widget.addTab(widget, tab_labels[tab_name])

    def remove_tabs(self):
        while self.tab_widget.count() > 1:
            self.tab_widget.removeTab(1)
        self.shown_tab_names = []

    def set_pipeline(self, pipeline: str):
        pipeline_changed = pipeline != self.current_pipeline
        if pipeline_changed:
            self.remove_tabs()
            if pipeline == "top-down":
                self.add_tab("centroid")
                self.add_tab("centered_instance")
            elif pipeline == "bottom-up":
                self.add_tab("bottomup")
            elif pipeline == "top-down-id":
                self.add_tab("centroid")
                self.add_tab("multi_class_topdown")
            elif pipeline == "bottom-up-id":
                self.add_tab("multi_class_bottomup")
            elif pipeline == "single":
                self.add_tab("single_instance")

            # Apply defaults from previous trained config or video analysis
            self._apply_pipeline_defaults(pipeline)

            # Update minimum width after tabs change (if dialog is visible)
            if self.isVisible():
                self._update_minimum_width()

        self.current_pipeline = pipeline

        self._validate_pipeline()

    def change_tab(self, tab_idx: int):
        print(tab_idx)

    def merge_pipeline_and_head_config_data(self, head_name, head_data, pipeline_data):
        for key, val in pipeline_data.items():
            # Skip GUI-only fields (not part of sleap-nn config schema)
            if key.startswith("gui."):
                continue
            if key.startswith("model_config.head_configs."):
                key_scope = key.split(".")
                if key_scope[2] != head_name:
                    continue
            head_data[key] = val

    @staticmethod
    def update_loaded_config(
        loaded_cfg: dict, tab_cfg_key_val_dict: dict
    ):  # -> scopedkeydict.ScopedKeyDict:
        """Update a loaded preset config with values from the training editor.

        Args:
            loaded_cfg: Dict from a yaml file that was loaded from a preset or previous
                training run.
            tab_cfg_key_val_dict: A dictionary with the values extracted from the
                training editor GUI tab.

        Returns:
                    A `ScopedKeyDict` with the loaded config values overriden by the
        corresponding ones from the `tab_cfg_key_val_dict`.
        """
        # Replace params exposed in GUI with values from GUI
        for param, value in tab_cfg_key_val_dict.items():
            loaded_cfg[param] = value

        return loaded_cfg

    def get_every_head_config_data(
        self, pipeline_form_data
    ) -> List[configs.ConfigFileInfo]:
        cfg_info_list = []

        # Copy relevant data into linked fields (i.e., anchor part).
        self.adjust_data_to_update_other_tabs(pipeline_form_data)

        for tab_name in self.shown_tab_names:
            trained_cfg_info = self.tabs[tab_name].trained_config_info_to_use
            if self.tabs[tab_name].use_trained and (trained_cfg_info is not None):
                cfg_info_list.append(trained_cfg_info)

            else:
                # Get config data from GUI
                tab_cfg_key_val_dict = self.tabs[tab_name].get_all_form_data()
                self.merge_pipeline_and_head_config_data(
                    head_name=tab_name,
                    head_data=tab_cfg_key_val_dict,
                    pipeline_data=pipeline_form_data,
                )
                apply_cfg_transforms_to_key_val_dict(tab_cfg_key_val_dict)

                if trained_cfg_info is None:
                    # Config could not be loaded, just use the values from the GUI
                    loaded_cfg_scoped: dict = tab_cfg_key_val_dict
                else:
                    # Config was loaded, override with the values from the GUI
                    loaded_cfg_scoped = LearningDialog.update_loaded_config(
                        get_keyval_dict_from_omegaconf(trained_cfg_info.config),
                        tab_cfg_key_val_dict,
                    )

                # Clear wandb.name for new training runs (not resume) so sleap-nn
                # will default it to the new run_name. The wandb.name field is not
                # in the GUI form, so old values from base configs would persist.
                if not self.tabs[tab_name].resume_training:
                    loaded_cfg_scoped["trainer_config.wandb.name"] = None

                # Deserialize merged dict to object
                cfg = get_omegaconf_from_gui_form(loaded_cfg_scoped)

                if len(self.labels.tracks) > 0:
                    # For multiclass topdown, the class vectors output stride
                    # should be the max stride.
                    backbone_name = find_backbone_name_from_key_val_dict(
                        tab_cfg_key_val_dict
                    )
                    max_stride = tab_cfg_key_val_dict[
                        f"model_config.backbone_config.{backbone_name}.max_stride"
                    ]

                    # Classes should be added here to prevent value error in
                    # model since we don't add them in the training config yaml.
                    if (
                        OmegaConf.select(
                            cfg,
                            "model_config.head_configs.multi_class_bottomup",
                            default=None,
                        )
                        is not None
                    ):
                        (
                            cfg.model_config.head_configs.multi_class_bottomup.class_maps.classes
                        ) = [t.name for t in self.labels.tracks]
                    elif (
                        OmegaConf.select(
                            cfg,
                            "model_config.head_configs.multi_class_topdown",
                            default=None,
                        )
                        is not None
                    ):
                        (
                            cfg.model_config.head_configs.multi_class_topdown.class_vectors.classes
                        ) = [t.name for t in self.labels.tracks]
                        (
                            cfg.model_config.head_configs.multi_class_topdown.class_vectors.output_stride
                        ) = max_stride

                cfg_info = configs.ConfigFileInfo(config=cfg, head_name=tab_name)

                cfg_info_list.append(cfg_info)

        return cfg_info_list

    def get_selected_frames_to_predict(
        self, pipeline_form_data
    ) -> Dict[Video, List[int]]:
        """Get frames to predict based on user selection.

        Uses the new FrameTargetSelector widget for selection.

        For random sample options (random, random_video), this method performs
        the actual sampling based on sample_count and exclude_user_labeled settings.
        """
        import random

        frames_to_predict = dict()

        if self._frame_selection is None:
            return frames_to_predict

        # Get selection from new frame target selector
        selection = self.frame_target_selector.get_selection()
        target_key = selection.target_key

        # Map widget keys to frame_selection keys
        key_map = {
            "frame": "frame",
            "clip": "clip",
            "video": "video",
            "all_videos": "all_videos",
            "random": "random",
            "random_video": "random_video",
            "suggestions": "suggestions",
            "user_labeled": "user",
            "predicted": "predicted",
            "nothing": None,  # No frames to predict
        }

        frame_selection_key = key_map.get(target_key)
        if frame_selection_key and frame_selection_key in self._frame_selection:
            candidate_frames = self._frame_selection[frame_selection_key].copy()

            # For random sample options, perform actual sampling
            if target_key in ("random", "random_video"):
                sample_count = selection.sample_count
                exclude_user_labeled = selection.exclude_user_labeled

                frames_to_predict = {}
                for video, candidates in candidate_frames.items():
                    if not candidates:
                        frames_to_predict[video] = []
                        continue

                    # Convert to set for efficient filtering
                    available = set(candidates)

                    # Filter out user-labeled frames if requested
                    if exclude_user_labeled:
                        user_labeled = self._get_user_labeled_frame_indices(video)
                        available = available - user_labeled

                    # Sample from available frames
                    available_list = list(available)
                    actual_sample_size = min(sample_count, len(available_list))
                    if actual_sample_size > 0:
                        sampled = random.sample(available_list, actual_sample_size)
                        frames_to_predict[video] = sorted(sampled)
                    else:
                        frames_to_predict[video] = []
            else:
                frames_to_predict = candidate_frames

        return frames_to_predict

    def get_items_for_inference(self, pipeline_form_data) -> runners.ItemsForInference:
        """Build inference items from current selection.

        Uses the new FrameTargetSelector widget for selection.
        """
        frame_selection = self.get_selected_frames_to_predict(pipeline_form_data)
        frame_count = self.count_total_frames_for_selection_option(frame_selection)

        # Get target key from new widget
        selection = self.frame_target_selector.get_selection()
        target_key = selection.target_key

        if target_key == "user_labeled":
            items_for_inference = runners.ItemsForInference(
                items=[
                    runners.DatasetItemForInference(
                        labels_path=self.labels_filename, frame_filter="user"
                    )
                ],
                total_frame_count=frame_count,
            )
        elif target_key == "suggestions":
            items_for_inference = runners.ItemsForInference(
                items=[
                    runners.DatasetItemForInference(
                        labels_path=self.labels_filename, frame_filter="suggested"
                    )
                ],
                total_frame_count=frame_count,
            )
        elif target_key == "predicted":
            items_for_inference = runners.ItemsForInference(
                items=[
                    runners.DatasetItemForInference(
                        labels_path=self.labels_filename, frame_filter="predicted"
                    )
                ],
                total_frame_count=frame_count,
            )
        else:
            items_for_inference = runners.ItemsForInference.from_video_frames_dict(
                video_frames_dict=frame_selection,
                total_frame_count=frame_count,
                labels_path=self.labels_filename,
                labels=self.labels,
            )
        return items_for_inference

    def _validate_id_model(self) -> bool:
        """Make sure we have instances with tracks set for ID models."""
        if not self.labels.tracks:
            return False

        found_tracks = False
        for inst in instances(labels=self.labels):
            if type(inst) == sleap.Instance and inst.track is not None:
                found_tracks = True
                break

        return found_tracks

    def _validate_pipeline(self):
        can_run = True
        message = ""

        if self.mode == "inference":
            # Make sure we have trained models for each required head.
            untrained = [
                tab_name
                for tab_name in self.shown_tab_names
                if not self.tabs[tab_name].has_trained_config_selected
            ]
            if untrained:
                can_run = False
                message = (
                    "Cannot run inference with untrained models "
                    f"({', '.join(untrained)})."
                )
                can_run = False

        # Make sure we have instances with tracks set for ID models.
        if self.mode == "training" and self.current_pipeline in (
            "top-down-id",
            "bottom-up-id",
        ):
            can_run = self._validate_id_model()
            if not can_run:
                message = "Cannot run ID model training without tracks."

        # Make sure skeleton will be valid for bottom-up inference.
        if self.mode == "training" and self.current_pipeline == "bottom-up":
            skeleton = self.labels.skeletons[0]

            if not is_arborescence(skeleton):
                message += (
                    "Cannot run bottom-up pipeline when skeleton is not an "
                    "arborescence."
                )

                # These functions return node names (strings), not Node objects
                root_names = root_nodes(skeleton)
                over_max_in_degree = in_degree_over_one(skeleton)
                cycles_var = cycles(skeleton)

                if len(root_names) > 1:
                    message += (
                        f" There are multiple root nodes: {', '.join(root_names)} "
                        "(there should be exactly one node which is not a target)."
                    )

                if over_max_in_degree:
                    message += (
                        " There are nodes which are target in multiple edges: "
                        f"{', '.join(over_max_in_degree)} (maximum in-degree should be "
                        "1).</li>"
                    )

                if cycles_var:
                    cycle_strings = []
                    for cycle in cycles_var:
                        # cycles returns node names (strings), not Node objects
                        cycle_strings.append(" &ndash;&gt; ".join(cycle))

                    message += (
                        f" There are cycles in graph: {'; '.join(cycle_strings)}."
                    )

                can_run = False

        if not can_run and message:
            message = f"<b>Unable to run:</b><br />{message}"

        self.message_widget.setText(message)
        self.run_button.setEnabled(can_run)

    def run(self):
        """Run with current dialog settings."""
        # Get selection from new widget
        selection = self.frame_target_selector.get_selection()

        pipeline_form_data = self.pipeline_form_widget.get_form_data()

        # Add prediction mode to pipeline form data for the runner
        pipeline_form_data["_prediction_mode"] = selection.prediction_mode

        items_for_inference = self.get_items_for_inference(pipeline_form_data)

        config_info_list = self.get_every_head_config_data(pipeline_form_data)

        # Close the dialog now that we have the data from it
        self.accept()

        # Run training/learning pipeline using the TrainingJobs
        new_counts = runners.run_learning_pipeline(
            labels_filename=self.labels_filename,
            labels=self.labels,
            config_info_list=config_info_list,
            inference_params=pipeline_form_data,
            items_for_inference=items_for_inference,
        )

        self._handle_learning_finished.emit(new_counts)

        # count < 0 means there was an error and we didn't get any results.
        if new_counts is not None and new_counts >= 0:
            total_count = items_for_inference.total_frame_count
            no_result_count = max(0, total_count - new_counts)

            message = (
                f"Inference ran on {total_count} frames."
                f"\n\nInstances were predicted on {new_counts} frames "
                f"({no_result_count} frame{'s' if no_result_count != 1 else ''} with "
                "no instances found)."
            )

            win = QtWidgets.QMessageBox(text=message)
            win.setWindowTitle("Inference Results")
            win.exec_()

    def copy(self):
        """Copy scripts and configs to clipboard"""

        # Get all info from dialog
        pipeline_form_data = self.pipeline_form_widget.get_form_data()
        config_info_list = self.get_every_head_config_data(pipeline_form_data)

        # Format information for each tab in dialog
        # output = [OmegaConf.to_yaml(pipeline_form_data)] # TODO:cfg:
        output = []
        for config_info in config_info_list:
            config_info = config_info.config
            # convert to sleap-nn cfg (yaml)
            try:
                from sleap_nn.config.training_job_config import verify_training_cfg

                config_info = filter_cfg(config_info)
                cfg = verify_training_cfg(config_info)
                cfg.data_config.train_labels_path = [self.labels_filename]
                output.append(OmegaConf.to_yaml(cfg))
            except ImportError:
                show_sleap_nn_installation_message()
                print(
                    "sleap-nn is not installed. This appears to be GUI-only install."
                    "To enable training, please install SLEAP with the 'nn' dependency."
                    "See the installation guide: https://docs.sleap.ai/latest/installation/"
                )

        output = "\n".join(output)
        # Set the clipboard text
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(output)

    def save(
        self, output_dir: Optional[str] = None, labels_filename: Optional[str] = None
    ):
        """Save scripts and configs to run pipeline."""
        if output_dir is None or not output_dir:
            labels_fn = Path(self.labels_filename)
            models_dir = Path(labels_fn.parent, "models")
            output_dir = FileDialog.openDir(
                None,
                dir=models_dir.as_posix(),
                caption="Select directory to save scripts",
            )

            if not output_dir:
                return

        pipeline_form_data = self.pipeline_form_widget.get_form_data()
        items_for_inference = self.get_items_for_inference(pipeline_form_data)
        config_info_list = self.get_every_head_config_data(pipeline_form_data)

        if labels_filename is None:
            labels_filename = self.labels_filename

        runners.write_pipeline_files(
            output_dir=output_dir,
            labels_filename=labels_filename,
            config_info_list=config_info_list,
            inference_params=pipeline_form_data,
            items_for_inference=items_for_inference,
        )

    def export_package(self, output_path: Optional[str] = None, gui: bool = True):
        """Export training job package."""
        # TODO: Warn if self.mode != "training"?
        if output_path is None or not output_path:
            # Prompt for output path.
            output_path, _ = FileDialog.save(
                caption="Export Training Job Package...",
                dir=f"{self.labels_filename}.training_job.zip",
                filter="Training Job Package (*.zip)",
            )
            if len(output_path) == 0:
                return

        # Create temp dir before packaging.
        tmp_dir = tempfile.TemporaryDirectory()

        # Remove the temp dir when program exits in case something goes wrong.
        # atexit.register(shutil.rmtree, tmp_dir.name, ignore_errors=True)

        # Check if we need to include suggestions.
        include_suggestions = False
        items_for_inference = self.get_items_for_inference(
            self.pipeline_form_widget.get_form_data()
        )
        for item in items_for_inference.items:
            if (
                isinstance(item, runners.DatasetItemForInference)
                and item.frame_filter == "suggested"
            ):
                include_suggestions = True

        # Save dataset with images.
        labels_pkg_filename = str(
            Path(self.labels_filename).with_suffix(".pkg.slp").name
        )
        if gui:
            ret = sleap.gui.commands.export_dataset_gui(
                self.labels,
                tmp_dir.name + "/" + labels_pkg_filename,
                all_labeled=False,
                suggested=include_suggestions,
            )
            if ret == "canceled":
                # Quit if user canceled during export.
                tmp_dir.cleanup()
                return
        else:
            self.labels.save(
                filename=tmp_dir.name + "/" + labels_pkg_filename,
                embed=True,
            )

        # Save config and scripts.
        self.save(tmp_dir.name, labels_filename=labels_pkg_filename)

        # Package everything.
        shutil.make_archive(
            base_name=str(Path(output_path).with_suffix("")),
            format="zip",
            root_dir=tmp_dir.name,
        )

        msg = f"Saved training job package to: {output_path}"
        print(msg)

        # Close training editor.
        self.accept()

        if gui:
            msgBox = QtWidgets.QMessageBox(text="Created training job package.")
            msgBox.setDetailedText(output_path)
            msgBox.setWindowTitle("Training Job Package")
            msgBox.addButton(QtWidgets.QMessageBox.Ok)
            openFolderButton = msgBox.addButton(
                "Open containing folder", QtWidgets.QMessageBox.ActionRole
            )
            colabButton = msgBox.addButton(
                "Go to Colab", QtWidgets.QMessageBox.ActionRole
            )
            msgBox.exec_()

            if msgBox.clickedButton() == openFolderButton:
                sleap.gui.commands.open_file(str(Path(output_path).resolve().parent))
            elif msgBox.clickedButton() == colabButton:
                # TODO: Update this to more workflow-tailored notebook.
                sleap.gui.commands.copy_to_clipboard(output_path)
                sleap.gui.commands.open_website(
                    "https://colab.research.google.com/github/talmolab/sleap/blob/develop/docs/notebooks/Training_and_inference_using_Google_Drive.ipynb"
                )

        tmp_dir.cleanup()


class TrainingEditorWidget(QtWidgets.QWidget):
    """
    Dialog for viewing and modifying training profiles (model hyperparameters).

    Args:
        video: `Video` to use for receptive field preview
        skeleton: `Skeleton` to use for node option list
        head: If given, then only show configs with specified head name
        cfg_getter: Object to use for getting list of config files.
            If given, then menu of config files will be shown so user can
            either copy hyperameters from another profile/model, or use a model
            that was already trained.
        require_trained: If True, then only show configs that are trained,
            and don't allow user to uncheck "use trained" setting. This is set
            when :py:class:`LearningDialog` is in "inference" mode.
    """

    valueChanged = QtCore.Signal()

    def __init__(
        self,
        video: Optional[Video] = None,
        skeleton: Optional["Skeleton"] = None,
        head: Optional[Text] = None,
        cfg_getter: Optional["TrainingConfigsGetter"] = None,
        require_trained: bool = False,
        labels: Optional[Labels] = None,
        parent_dialog: Optional["LearningDialog"] = None,
        *args,
        **kwargs,
    ):
        super(TrainingEditorWidget, self).__init__()

        self._video = video
        self._labels = labels
        self._parent_dialog = parent_dialog
        self._cfg_getter = cfg_getter
        self._cfg_list_widget = None
        self._receptive_field_widget = None
        self._training_mode_group = None
        self._radio_train_scratch = None
        self._radio_use_trained = None
        self._radio_resume = None
        self._require_trained = require_trained
        self.head = head

        yaml_name = "training_editor_form"

        self.form_widgets: Dict[str, YamlFormWidget] = dict()

        for key in ("model", "data", "augmentation", "optimization", "outputs"):
            self.form_widgets[key] = YamlFormWidget.from_name(
                yaml_name, which_form=key, title=key.title()
            )
            self.form_widgets[key].valueChanged.connect(self.emitValueChanged)

        self.form_widgets["model"].valueChanged.connect(self.update_receptive_field)
        self.form_widgets["data"].valueChanged.connect(self.update_receptive_field)

        # Connect overfit mode checkbox to disable validation fraction
        self._setup_overfit_mode_toggle()

        # Connect rotation preset dropdown to enable/disable custom angle field
        self._setup_rotation_preset_toggle()

        # Connect augmentation checkboxes to show/hide their parameter fields
        self._setup_augmentation_param_toggles()

        # Hide crop size for non-cropping model types
        self._setup_crop_size_visibility()

        if hasattr(skeleton, "node_names"):
            for field_name in NODE_LIST_FIELDS:
                form_name = field_name.split(".")[0]
                self.form_widgets[form_name].set_field_options(
                    ".".join(field_name.split(".")[1:]),
                    skeleton.node_names,
                )

        # crop box should be shown for centered_instance/multi_class_topdown
        show_crop_box = head in ("centered_instance", "multi_class_topdown")
        # Use labeled frame image for topdown pipeline (centroid + centered_instance)
        use_labeled_frame = head in (
            "centroid",
            "centered_instance",
            "multi_class_topdown",
        )

        if self._video or (use_labeled_frame and labels):
            self._receptive_field_widget = receptivefield.ReceptiveFieldWidget(
                self.head, show_crop_box=show_crop_box
            )
            # For topdown heads, use labeled frame image for consistency
            if use_labeled_frame and labels:
                self._receptive_field_widget.setLabels(labels, self._video)
            elif self._video:
                self._receptive_field_widget.setImage(
                    self._video.backend.read_test_frame()
                )

        self._set_head()

        # Layout for header and columns
        layout = QtWidgets.QVBoxLayout()

        # Two column layout: Data+Augmentation+Optimization | Model
        col1_layout = QtWidgets.QVBoxLayout()
        col2_layout = QtWidgets.QVBoxLayout()

        col1_layout.addWidget(self.form_widgets["data"])
        col1_layout.addWidget(self.form_widgets["augmentation"])
        col1_layout.addWidget(self.form_widgets["optimization"])
        col2_layout.addWidget(self.form_widgets["model"])

        if self._receptive_field_widget:
            col0_layout = QtWidgets.QVBoxLayout()
            col0_layout.addWidget(self._receptive_field_widget)

            # Add "Analyze Sizes" button for cropping model types
            # Button is inserted into the receptive field widget (below legend)
            if show_crop_box and labels is not None:
                self._analyze_size_button = QtWidgets.QPushButton("Analyze Sizes...")
                self._analyze_size_button.setToolTip(
                    "View the distribution of instance sizes and identify outliers"
                )
                self._analyze_size_button.clicked.connect(self._open_size_distribution)
                self._receptive_field_widget.addButtonWidget(self._analyze_size_button)
        else:
            col0_layout = None

        col_layout = QtWidgets.QHBoxLayout()
        if col0_layout:
            col_layout.addWidget(
                self._layout_widget(col0_layout),
                stretch=0,
                alignment=QtCore.Qt.AlignTop,
            )
        col_layout.addWidget(
            self._layout_widget(col1_layout), stretch=0, alignment=QtCore.Qt.AlignTop
        )
        col_layout.addWidget(
            self._layout_widget(col2_layout), stretch=0, alignment=QtCore.Qt.AlignTop
        )
        col_layout.addStretch(1)  # Push columns left, absorb extra space

        # If we have an object which gets a list of config files,
        # then we'll show a menu to allow selection from the list.
        if self._cfg_getter is not None:
            self._cfg_list_widget = configs.TrainingConfigFilesWidget(
                cfg_getter=self._cfg_getter,
                head_name=cast(str, head),  # Expect head to be a string
                require_trained=require_trained,
            )
            self._cfg_list_widget.onConfigSelection.connect(
                self.acceptSelectedConfigInfo
            )
            # self._cfg_list_widget.setDataDict.connect(
            # self.set_fields_from_key_val_dict
            # )

            layout.addWidget(self._cfg_list_widget)

        if self._require_trained:
            self._update_use_trained()
        elif self._cfg_list_widget is not None:
            # Add radio buttons for training mode selection
            # Three mutually exclusive options for how to use the selected config
            self._training_mode_group = QtWidgets.QButtonGroup(self)

            self._radio_train_scratch = QtWidgets.QRadioButton(
                "Reuse config (train from scratch)"
            )
            self._radio_resume = QtWidgets.QRadioButton("Resume training (fine-tune)")
            self._radio_use_trained = QtWidgets.QRadioButton(
                "Reuse model (don't retrain)"
            )

            # Set IDs for easier identification
            self._training_mode_group.addButton(self._radio_train_scratch, 0)
            self._training_mode_group.addButton(self._radio_resume, 1)
            self._training_mode_group.addButton(self._radio_use_trained, 2)

            # Default to training from scratch
            self._radio_train_scratch.setChecked(True)

            # Last two options only enabled when trained model is available
            self._radio_resume.setEnabled(False)
            self._radio_use_trained.setEnabled(False)

            # Layout radio buttons horizontally with minimal spacing
            radio_layout = QtWidgets.QHBoxLayout()
            radio_layout.setContentsMargins(0, 0, 0, 0)
            radio_layout.setSpacing(12)
            radio_layout.addWidget(self._radio_train_scratch)
            radio_layout.addWidget(self._radio_resume)
            radio_layout.addWidget(self._radio_use_trained)
            radio_layout.addStretch()

            radio_widget = QtWidgets.QWidget()
            radio_widget.setLayout(radio_layout)
            radio_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
            )

            self._training_mode_group.buttonClicked.connect(self._update_use_trained)

            layout.addWidget(radio_widget)

        layout.addWidget(self._layout_widget(col_layout))
        self.setLayout(layout)

    @classmethod
    def from_trained_config(
        cls, cfg_info: configs.ConfigFileInfo, cfg_getter: configs.TrainingConfigsGetter
    ):
        widget = cls(
            require_trained=True, head=cfg_info.head_name, cfg_getter=cfg_getter
        )
        widget.acceptSelectedConfigInfo(cfg_info)
        widget.setWindowTitle(cfg_info.path_dir)
        return widget

    @staticmethod
    def _layout_widget(layout):
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def emitValueChanged(self):
        self.valueChanged.emit()

        # When there's a config getter, we want to inform it that the data
        # has changed so that it can activate/update the "user" config
        # if self._cfg_list_widget:
        #     self._set_user_config()

    def _setup_overfit_mode_toggle(self):
        """Connect overfit mode checkbox to disable validation fraction.

        Follows the same pattern as OptionalSpinWidget.updateState().
        """
        data_form = self.form_widgets["data"]
        overfit_field_name = "data_config.use_same_data_for_val"
        val_frac_field_name = "data_config.validation_fraction"

        overfit_checkbox = data_form.fields.get(overfit_field_name)
        val_frac_widget = data_form.fields.get(val_frac_field_name)

        if overfit_checkbox is not None and val_frac_widget is not None:

            def update_state():
                val_frac_widget.setDisabled(overfit_checkbox.isChecked())

            overfit_checkbox.stateChanged.connect(update_state)
            update_state()

    def _setup_rotation_preset_toggle(self):
        """Connect rotation preset dropdown to show/hide custom angle field.

        When a preset (Off, ±15°, ±180°) is selected, the custom angle field is
        hidden. When "Custom" is selected, the field is shown.
        """
        aug_form = self.form_widgets["augmentation"]
        form_layout = aug_form.form_layout
        preset_field = aug_form.fields.get("_rotation_preset")
        custom_field = aug_form.fields.get("_rotation_custom_angle")

        if preset_field is not None and custom_field is not None:
            # Get the label for the custom field
            custom_label = form_layout.labelForField(custom_field)

            def update_state():
                is_custom = preset_field.value() == "Custom"
                custom_field.setVisible(is_custom)
                if custom_label is not None:
                    custom_label.setVisible(is_custom)

            preset_field.valueChanged.connect(update_state)
            update_state()  # Set initial state

    def _setup_augmentation_param_toggles(self):
        """Connect augmentation checkboxes to show/hide their parameter fields.

        When an augmentation checkbox is unchecked, its parameter fields are hidden
        to reduce visual clutter. The fields are shown when the checkbox is checked.
        """
        aug_form = self.form_widgets["augmentation"]
        form_layout = aug_form.form_layout

        # Define which checkbox controls which parameter fields
        toggle_groups = {
            "_scale_enabled": [
                "data_config.augmentation_config.geometric.scale_min",
                "data_config.augmentation_config.geometric.scale_max",
            ],
            "_uniform_noise_enabled": [
                "data_config.augmentation_config.intensity.uniform_noise_min",
                "data_config.augmentation_config.intensity.uniform_noise_max",
            ],
            "_gaussian_noise_enabled": [
                "data_config.augmentation_config.intensity.gaussian_noise_mean",
                "data_config.augmentation_config.intensity.gaussian_noise_std",
            ],
            "_contrast_enabled": [
                "data_config.augmentation_config.intensity.contrast_min",
                "data_config.augmentation_config.intensity.contrast_max",
            ],
            "_brightness_enabled": [
                "data_config.augmentation_config.intensity.brightness_min",
                "data_config.augmentation_config.intensity.brightness_max",
            ],
        }

        for checkbox_name, param_fields in toggle_groups.items():
            checkbox = aug_form.fields.get(checkbox_name)
            if checkbox is None:
                continue

            # Collect the parameter field widgets and their labels
            param_widgets = []
            for field_name in param_fields:
                field = aug_form.fields.get(field_name)
                if field is not None:
                    # Get the label for this field from the form layout
                    label = form_layout.labelForField(field)
                    param_widgets.append((field, label))

            if not param_widgets:
                continue

            # Create update function that captures the widgets
            def make_update_visibility(widgets):
                def update_visibility(state):
                    visible = bool(state)
                    for field, label in widgets:
                        field.setVisible(visible)
                        if label is not None:
                            label.setVisible(visible)

                return update_visibility

            update_fn = make_update_visibility(param_widgets)
            checkbox.stateChanged.connect(update_fn)
            update_fn(checkbox.isChecked())  # Set initial state

    def _setup_crop_size_visibility(self):
        """Hide crop size field for model types that don't use cropping.

        Crop size is only relevant for centered_instance and multi_class_topdown
        models which crop around detected centroids. Other model types (centroid,
        bottomup, single_instance, multi_class_bottomup) process full images.
        """
        # Only show for models that use instance cropping
        if self.head in ("centered_instance", "multi_class_topdown"):
            return  # Keep visible (default state)

        data_form = self.form_widgets["data"]
        form_layout = data_form.form_layout
        crop_field = data_form.fields.get("data_config.preprocessing.crop_size")

        if crop_field is not None:
            crop_label = form_layout.labelForField(crop_field)
            crop_field.setVisible(False)
            if crop_label is not None:
                crop_label.setVisible(False)

    def acceptSelectedConfigInfo(self, cfg_info: configs.ConfigFileInfo):
        self._load_config(cfg_info)

        has_trained_model = cfg_info.has_trained_model

        # Update radio button states based on whether selected config has trained model
        if self._radio_use_trained is not None:
            # Enable/disable trained model options based on availability
            self._radio_use_trained.setEnabled(has_trained_model)
            self._radio_resume.setEnabled(has_trained_model)

            # If no trained model available, reset to "train from scratch"
            if not has_trained_model:
                self._radio_train_scratch.setChecked(True)

        self.update_receptive_field()

    def update_receptive_field(self):
        data_form_data = get_omegaconf_from_gui_form(
            self.form_widgets["data"].get_form_data()
        )

        model_cfg = get_omegaconf_from_gui_form(
            self.form_widgets["model"].get_form_data()
        )

        rf_image_scale = OmegaConf.select(
            data_form_data, "data_config.preprocessing.scale", default=1.0
        )

        if self._receptive_field_widget:
            self._receptive_field_widget.setModelConfig(model_cfg, scale=rf_image_scale)

            # Update crop box for centered_instance/multi_class_topdown heads
            if (
                self.head in ("centered_instance", "multi_class_topdown")
                and self._labels
            ):
                # Get crop size from config
                crop_size = receptivefield.compute_crop_size_from_cfg(
                    data_form_data, model_cfg, self._labels
                )

                # Get anchor part from the model form data
                anchor_part = None
                if self.head == "centered_instance":
                    anchor_part = OmegaConf.select(
                        model_cfg,
                        "model_config.head_configs.centered_instance.confmaps.anchor_part",
                        default=None,
                    )
                elif self.head == "multi_class_topdown":
                    anchor_part = OmegaConf.select(
                        model_cfg,
                        "model_config.head_configs.multi_class_topdown.confmaps.anchor_part",
                        default=None,
                    )

                self._receptive_field_widget.setCropConfig(
                    crop_size=crop_size,
                    scale=rf_image_scale,
                    anchor_part=anchor_part,
                )

            self._receptive_field_widget.repaint()

    def update_file_list(self):
        self._cfg_list_widget.update()

    def _load_config_or_key_val_dict(self, cfg_data):
        if type(cfg_data) != dict:
            self._load_config(cfg_data)
        else:
            self.set_fields_from_key_val_dict(cfg_data)

    def _load_config(self, cfg_info: configs.ConfigFileInfo):
        if cfg_info is None:
            return

        cfg = cfg_info.config
        key_val_dict = get_keyval_dict_from_omegaconf(cfg)

        # Filter out system-specific settings that should come from preferences,
        # not from the training profile. These are machine-specific (GPU count,
        # workers) and should default to "auto" regardless of what the profile says.
        system_specific_keys = [
            "trainer_config.trainer_devices",
            "trainer_config.train_data_loader.num_workers",
        ]
        for key in system_specific_keys:
            key_val_dict.pop(key, None)

        # Clear run_name - it should be auto-generated for new training runs.
        # This prevents old run_name from base config from leaking through.
        key_val_dict["trainer_config.run_name"] = None

        # Reverse-map rotation_min/rotation_max to _rotation_preset dropdown
        rot_min = key_val_dict.get(
            "data_config.augmentation_config.geometric.rotation_min"
        )
        rot_max = key_val_dict.get(
            "data_config.augmentation_config.geometric.rotation_max"
        )
        rot_p = key_val_dict.get("data_config.augmentation_config.geometric.rotation_p")
        if rot_p is None or rot_p == 0:
            key_val_dict["_rotation_preset"] = "Off"
        elif rot_min is not None and rot_max is not None:
            # Check for symmetric presets
            if rot_min == -15 and rot_max == 15:
                key_val_dict["_rotation_preset"] = "±15°"
            elif rot_min == -180 and rot_max == 180:
                key_val_dict["_rotation_preset"] = "±180°"
            elif rot_min == -rot_max:
                # Symmetric but custom angle
                key_val_dict["_rotation_preset"] = "Custom"
                key_val_dict["_rotation_custom_angle"] = rot_max
            else:
                # Asymmetric (rare) - use Custom with max as angle
                key_val_dict["_rotation_preset"] = "Custom"
                key_val_dict["_rotation_custom_angle"] = max(abs(rot_min), abs(rot_max))

        self.set_fields_from_key_val_dict(key_val_dict)

    # def _set_user_config(self):
    #     cfg_form_data_dict = self.get_all_form_data()
    #     self._cfg_list_widget.setUserConfigData(cfg_form_data_dict)

    def _update_use_trained(self, button=None):
        """Update config GUI based on training mode radio button selection.

        This function is called when a radio button is clicked or when
        _require_trained is set (inference mode).

        Training modes:
        - Train from scratch: All forms editable, train new model
        - Use same model (don't retrain): All forms disabled, use trained model as-is
        - Resume training (fine-tune): Model form disabled, other forms editable

        Args:
            button: The clicked radio button (unused, we check button group state).

        Returns:
            None

        Side Effects:
            Disables/Enables fields based on selected training mode.
        """
        # Get current training mode
        use_trained_params = self.use_trained
        resume_training_params = self.resume_training

        # Enable/disable all form widgets based on mode
        for form in self.form_widgets.values():
            form.set_enabled(not use_trained_params)

        # Get config info if we need to load trained config/model
        cfg_info = None
        if use_trained_params or resume_training_params:
            if self._cfg_list_widget is not None:
                cfg_info = self._cfg_list_widget.getSelectedConfigInfo()

        # Resume training: model form disabled, load model config
        if resume_training_params and cfg_info is not None:
            self.form_widgets["model"].set_enabled(False)

            # Set model form to match config
            cfg = cfg_info.config
            key_val_dict = get_keyval_dict_from_omegaconf(cfg)
            self.set_fields_from_key_val_dict({"model": key_val_dict})

        # Use trained model: all forms disabled, load full config
        if use_trained_params and cfg_info is not None:
            self._load_config(cfg_info)

        self._set_head()

    def _set_head(self):
        if self.head:
            self.set_fields_from_key_val_dict(
                {
                    "_heads_name": self.head,
                }
            )

            self.form_widgets["model"].set_field_enabled("_heads_name", False)

    def set_fields_from_key_val_dict(self, cfg_key_val_dict):
        for form in self.form_widgets.values():
            form.set_form_data(cfg_key_val_dict)

        self._set_backbone_from_key_val_dict(cfg_key_val_dict)

    def _set_backbone_from_key_val_dict(self, cfg_key_val_dict):
        for key, val in cfg_key_val_dict.items():
            if (
                key.startswith("model.model_config.backbone_config.")
                and val is not None
            ):
                backbone_name = key.split(".")[3]
                self.set_fields_from_key_val_dict(dict(_backbone_name=backbone_name))
                break

    @property
    def use_trained(self) -> bool:
        """Check if user wants to use trained model without retraining.

        Returns True when:
        - _require_trained is True (inference mode), OR
        - "Use same model (don't retrain)" radio button is selected
        """
        if self._require_trained:
            return True

        if self._radio_use_trained is not None and self._radio_use_trained.isChecked():
            return True

        return False

    @property
    def resume_training(self) -> bool:
        """Check if user wants to resume/fine-tune training.

        Returns True when "Resume training (fine-tune)" radio button is selected.
        """
        if self._radio_resume is not None and self._radio_resume.isChecked():
            return True
        return False

    @property
    def trained_config_info_to_use(self) -> Optional[configs.ConfigFileInfo]:
        # If `TrainingEditorWidget` was initialized with a config getter, then
        # we expect to have a list of config files
        if self._cfg_list_widget is None:
            return None

        selected_config_info: Optional[configs.ConfigFileInfo] = (
            self._cfg_list_widget.getSelectedConfigInfo()
        )
        if (selected_config_info is None) or (
            not selected_config_info.has_trained_model
        ):
            return None

        trained_config_info = configs.ConfigFileInfo.from_config_file(
            selected_config_info.path
        )
        if self.use_trained:
            trained_config_info.dont_retrain = True
        else:
            # Set certain parameters to defaults
            trained_config = trained_config_info.config
            trained_config.data_config.val_labels_path = None
            trained_config.data_config.test_file_path = None
            trained_config.data_config.skeletons = []
            trained_config.trainer_config.ckpt_dir = None
            trained_config.trainer_config.run_name = None

        if self.resume_training:
            # Get the folder path of trained config and find checkpoint file
            model_dir = Path(cast(str, trained_config_info.path)).parent
            file_list = list(model_dir.iterdir())
            ckpt = None
            if (model_dir / "best.ckpt") in file_list:
                ckpt = "best.ckpt"
            elif (model_dir / "best_model.h5") in file_list:
                ckpt = "best_model.h5"

            if ckpt is not None:
                trained_config_info.config.model_config.pretrained_backbone_weights = (
                    model_dir / ckpt
                ).as_posix()
                trained_config_info.config.model_config.pretrained_head_weights = (
                    trained_config_info.config.model_config.pretrained_backbone_weights
                )
            else:
                # No checkpoint found - proceed without pretrained weights
                trained_config_info.config.model_config.pretrained_backbone_weights = (
                    None
                )
                trained_config_info.config.model_config.pretrained_head_weights = None
        else:
            trained_config_info.config.model_config.pretrained_backbone_weights = None
            trained_config_info.config.model_config.pretrained_head_weights = None

        # Always clear wandb.name so sleap-nn will default it to the new run_name.
        # "Use Trained Model Weights" means use pretrained weights for initialization,
        # not resume the same wandb logging run.
        trained_config_info.config.trainer_config.wandb.name = None

        return trained_config_info

    @property
    def has_trained_config_selected(self) -> bool:
        if self._cfg_list_widget is None:
            return False

        cfg_info = self._cfg_list_widget.getSelectedConfigInfo()
        if cfg_info and cfg_info.has_trained_model:
            return True

        return False

    def get_all_form_data(self) -> dict:
        form_data = dict()
        for form in self.form_widgets.values():
            form_data.update(form.get_form_data())
        return form_data

    def _open_size_distribution(self):
        """Opens the instance size distribution analysis dialog."""
        if self._labels is None:
            return

        from sleap.gui.dialogs.size_distribution import SizeDistributionDialog

        # Create navigate callback that emits signal to parent LearningDialog
        navigate_callback = None
        if self._parent_dialog is not None:

            def navigate_callback(video_idx: int, frame_idx: int, instance_idx: int):
                self._parent_dialog.navigate_to_instance.emit(
                    video_idx, frame_idx, instance_idx
                )

        dialog = SizeDistributionDialog(
            labels=self._labels,
            navigate_callback=navigate_callback,
            parent=self,
        )

        # Sync rotation preset with augmentation settings
        try:
            aug_data = self.form_widgets["augmentation"].get_form_data()
            rotation_preset = aug_data.get("_rotation_preset", "Off")

            # Map form values to widget values
            # Form uses: "Off", "±15°", "±180°", "Custom"
            # Widget expects: "Off", "+/-15", "+/-180", "Custom"
            preset_map = {
                "Off": "Off",
                "±15°": "+/-15",
                "±180°": "+/-180",
                "Custom": "Custom",
            }
            widget_preset = preset_map.get(rotation_preset, "Off")
            dialog.set_rotation_preset(widget_preset)

            # Also sync custom angle if applicable
            if widget_preset == "Custom":
                custom_angle = aug_data.get("_rotation_custom_angle")
                if custom_angle is not None:
                    dialog.set_custom_angle(int(custom_angle))
        except Exception:
            pass  # Use default if we can't read augmentation settings

        # Use show() for non-modal dialog so user can interact with main window
        dialog.show()


def demo_training_dialog():
    app = QtWidgets.QApplication([])

    filename = "tests/data/json_format_v1/centered_pair.json"
    labels = load_file(filename)
    win = LearningDialog("inference", labels_filename=filename, labels=labels)

    win.frame_selection = {"clip": {labels.videos[0]: (1, 2, 3, 4)}}
    # win.training_editor_widget.set_fields_from_key_val_dict({
    #     "_backbone_name": "unet",
    #     "_heads_name": "centered_instance",
    # })
    #
    # win.training_editor_widget.form_widgets["model"].set_field_enabled(
    # "_heads_name", False
    # )

    win.show()
    app.exec_()


if __name__ == "__main__":
    demo_training_dialog()
