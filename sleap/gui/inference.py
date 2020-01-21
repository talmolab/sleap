"""
Module for running training and inference from the main gui application.
"""

import os
import attr
import cattr
import numpy as np

from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple

from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.gui.filedialog import FileDialog
from sleap.gui.training_editor import TrainingEditor
from sleap.gui.formbuilder import YamlFormWidget
from sleap.nn.model import ModelOutputType
from sleap.nn.job import TrainingJob
from sleap import util

from PySide2 import QtWidgets, QtCore


SELECT_FILE_OPTION = "Select training run/model file..."

MENU_NAME_TYPE_MAP = dict(
    confmap=(ModelOutputType.CONFIDENCE_MAP, ModelOutputType.TOPDOWN_CONFIDENCE_MAP,),
    paf=(ModelOutputType.PART_AFFINITY_FIELD,),
    centroid=(ModelOutputType.CENTROIDS,),
)


class InferenceDialog(QtWidgets.QDialog):
    """Training/inference dialog.

    The dialog can be used in different modes:
    * simplified training + inference (fewer controls)
    * expert training + inference (full controls)
    * inference only

    Arguments:
        labels_filename: Path to the dataset where we'll get training data.
        labels: The dataset where we'll get training data and add predictions.
        mode: String which specified mode ("learning", "expert", or "inference").
    """

    learningFinished = QtCore.Signal()

    def __init__(
        self,
        labels_filename: str,
        labels: Labels,
        mode: str = "expert",
        *args,
        **kwargs,
    ):

        super(InferenceDialog, self).__init__(*args, **kwargs)

        self.labels_filename = labels_filename
        self.labels = labels
        self.mode = mode

        self._frame_selection = None
        self._job_filter = None

        if self.mode == "inference":
            self._job_filter = lambda job: job.is_trained

        print(f"Number of frames to train on: {len(labels.user_labeled_frames)}")

        title = dict(
            learning="Training and Inference",
            inference="Inference",
            expert="Inference Pipeline",
        )

        self.form_widget = YamlFormWidget.from_name(
            form_name="inference_forms",
            which_form=self.mode,
            title=title[self.mode] + " Settings",
        )

        self.setWindowTitle(title[self.mode])

        # form ui

        is_confmap_strict = self.mode == "learning"

        job_option_widgets = dict()
        if "_conf_job" in self.form_widget.fields:
            job_option_widgets["confmap"] = self.form_widget.fields["_conf_job"]
        if "_paf_job" in self.form_widget.fields:
            job_option_widgets["paf"] = self.form_widget.fields["_paf_job"]
        if "_centroid_job" in self.form_widget.fields:
            job_option_widgets["centroid"] = self.form_widget.fields["_centroid_job"]

        self.job_menu_manager = JobMenuManager(
            labels_filename,
            job_option_widgets,
            require_trained=(self.mode == "inference"),
            strict_confmap_type=is_confmap_strict,
            menu_selection_callback=self.on_job_menu_selection,
        )

        self.job_menu_manager.rebuild()
        self.job_menu_manager.update_menus(init=True)

        buttons = QtWidgets.QDialogButtonBox()
        self.cancel_button = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.run_button = buttons.addButton(
            "Run " + title[self.mode], QtWidgets.QDialogButtonBox.AcceptRole
        )

        self.status_message = QtWidgets.QLabel("hi!")

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.status_message)
        buttons_layout.addWidget(buttons, alignment=QtCore.Qt.AlignTop)

        buttons_layout_widget = QtWidgets.QWidget()
        buttons_layout_widget.setLayout(buttons_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.form_widget)
        layout.addWidget(buttons_layout_widget)

        self.setLayout(layout)

        # connect actions to buttons

        # TODO: fix

        def edit_conf_profile():
            self._view_profile(self.form_widget["_conf_job"], menu_name="confmap")

        def edit_paf_profile():
            self._view_profile(
                self.form_widget["_paf_job"], menu_name="paf",
            )

        def edit_cent_profile():
            self._view_profile(self.form_widget["_centroid_job"], menu_name="centroid")

        if "_view_conf" in self.form_widget.buttons:
            self.form_widget.buttons["_view_conf"].clicked.connect(edit_conf_profile)
        if "_view_paf" in self.form_widget.buttons:
            self.form_widget.buttons["_view_paf"].clicked.connect(edit_paf_profile)
        if "_view_centoids" in self.form_widget.buttons:
            self.form_widget.buttons["_view_centoids"].clicked.connect(
                edit_cent_profile
            )
        if "_view_datagen" in self.form_widget.buttons:
            self.form_widget.buttons["_view_datagen"].clicked.connect(self.view_datagen)

        self.form_widget.valueChanged.connect(lambda: self.update_gui())

        buttons.accepted.connect(self.run)
        buttons.rejected.connect(self.reject)

        self.update_gui()

    @property
    def frame_selection(self) -> Dict[str, Dict[Video, List[int]]]:
        """
        Returns dictionary with frames that user has selected for inference.
        """
        return self._frame_selection

    @frame_selection.setter
    def frame_selection(self, frame_selection: Dict[str, Dict[Video, List[int]]]):
        """Sets options of frames on which to run inference."""
        self._frame_selection = frame_selection

        if "_predict_frames" in self.form_widget.fields.keys():
            prediction_options = []

            def count_total_frames(videos_frames):
                if not videos_frames:
                    return 0
                count = 0
                for frame_list in videos_frames.values():
                    # Check for range, given as X, -Y
                    if len(frame_list) == 2 and frame_list[1] < 0:
                        count += -frame_list[1] - frame_list[0] + 1
                    else:
                        count += len(frame_list)
                return count

            # Determine which options are available given _frame_selection

            total_random = count_total_frames(self._frame_selection["random"])
            total_suggestions = count_total_frames(self._frame_selection["suggestions"])
            clip_length = count_total_frames(self._frame_selection["clip"])
            video_length = count_total_frames(self._frame_selection["video"])

            # Build list of options

            if self.mode != "inference":
                prediction_options.append("nothing")
            prediction_options.append("current frame")

            option = f"random frames ({total_random} total frames)"
            prediction_options.append(option)
            default_option = option

            if total_suggestions > 0:
                option = f"suggested frames ({total_suggestions} total frames)"
                prediction_options.append(option)
                default_option = option

            if clip_length > 0:
                option = f"selected clip ({clip_length} frames)"
                prediction_options.append(option)
                default_option = option

            prediction_options.append(f"entire video ({video_length} frames)")

            self.form_widget.fields["_predict_frames"].set_options(
                prediction_options, default_option
            )

    def show(self):
        """Shows dialog (we hide rather than close to maintain settings)."""
        super(InferenceDialog, self).show()

        # TODO: keep selection and any items added from training editor

        self.job_menu_manager.rebuild()
        self.job_menu_manager.update_menus()

    def update_gui(self):
        """Updates gui state after user changes to options."""
        form_data = self.form_widget.get_form_data()

        can_run = True

        use_centroids = form_data.get("_use_centroids", False)

        if "_use_centroids" in self.form_widget.fields:
            if form_data.get("_use_trained_centroids", False):
                # you must use centroids if you are using a centroid model
                use_centroids = True
                self.form_widget.set_form_data(dict(_use_centroids=True))
                self.form_widget.fields["_use_centroids"].setEnabled(False)
            else:
                self.form_widget.fields["_use_centroids"].setEnabled(True)

            if use_centroids:
                # you must crop if you are using centroids
                self.form_widget.set_form_data(dict(instance_crop=True))
                self.form_widget.fields["instance_crop"].setEnabled(False)
            else:
                self.form_widget.fields["instance_crop"].setEnabled(True)

        error_messages = []
        if form_data.get("_use_trained_confmaps", False) and form_data.get(
            "_use_trained_pafs", False
        ):
            # make sure trained models are compatible
            conf_job, _ = self.job_menu_manager.get_current_job("confmap")
            paf_job, _ = self.job_menu_manager.get_current_job("paf")

            # only check compatible if we have both profiles
            if conf_job is not None and paf_job is not None:
                if conf_job.trainer.scale != paf_job.trainer.scale:
                    can_run = False
                    error_messages.append(
                        f"training image scale for confmaps ({conf_job.trainer.scale}) does not match pafs ({paf_job.trainer.scale})"
                    )
                if conf_job.trainer.instance_crop != paf_job.trainer.instance_crop:
                    can_run = False
                    crop_model_name = (
                        "confmaps" if conf_job.trainer.instance_crop else "pafs"
                    )
                    error_messages.append(
                        f"exactly one model ({crop_model_name}) was trained on crops"
                    )
                if use_centroids and not conf_job.trainer.instance_crop:
                    can_run = False
                    error_messages.append(
                        f"models used with centroids must be trained on cropped images"
                    )

        message = ""
        if not can_run:
            message = (
                "Unable to run with selected models:\n- "
                + ";\n- ".join(error_messages)
                + "."
            )
        self.status_message.setText(message)

        self.run_button.setEnabled(can_run)

    def _get_model_types_to_use(self):
        """Returns lists of model types which user has enabled."""
        form_data = self.form_widget.get_form_data()
        types_to_use = []

        # TODO: check _grouping_method for confmaps vs topdown

        # always include confidence maps
        if "topdown" in form_data.get("_grouping_method", ""):
            types_to_use.append("topdown")
        else:
            types_to_use.append("confmap")

        # by default we want to use part affinity fields
        do_use_pafs = form_data.get("_use_pafs", True)
        if form_data.get("_dont_use_pafs", False):
            do_use_pafs = False
        elif form_data.get("_multi_instance_mode", "") == "single":
            do_use_pafs = False
        elif "topdown" in form_data.get("_grouping_method", ""):
            do_use_pafs = False

        if do_use_pafs:
            types_to_use.append("paf")

        # by default we want to use centroids
        do_use_centroids = True
        if not form_data.get("_use_centroids", True):
            do_use_centroids = False
        elif form_data.get("_region_proposal_mode", "") == "full frame":
            do_use_centroids = False

        if do_use_centroids:
            types_to_use.append("centroid")

        return types_to_use

    def _get_current_training_jobs(self) -> Dict[ModelOutputType, TrainingJob]:
        """
        Returns all currently selected training jobs.

        Form fields which match job parameters override values in saved jobs.
        """
        form_data = self.form_widget.get_form_data()
        training_jobs = dict()

        default_use_trained = self.mode == "inference"

        for menu_name in self._get_model_types_to_use():
            job, _ = self.job_menu_manager.get_current_job(menu_name)

            if job is None:
                continue

            model_type = job.model.output_type

            if model_type != ModelOutputType.CENTROIDS:
                # update training job from params in form
                trainer = job.trainer
                for key, val in form_data.items():
                    # check if field name is [var]_[model_type] (eg sigma_confmaps)
                    if key.split("_")[-1] == str(model_type):
                        key = "_".join(key.split("_")[:-1])
                    # check if form field matches attribute of Trainer object
                    if key in dir(trainer):
                        setattr(trainer, key, val)
            # Use already trained model if desired
            if form_data.get(f"_use_trained_{str(model_type)}", default_use_trained):
                job.use_trained_model = True
            elif model_type == ModelOutputType.TOPDOWN_CONFIDENCE_MAP:
                if form_data.get(f"_use_trained_confmaps", default_use_trained):
                    job.use_trained_model = True

            # Clear parameters that shouldn't be copied
            job.val_set_filename = None
            job.test_set_filename = None

            training_jobs[model_type] = job

        return training_jobs

    def run(self):
        """Run training (or inference) with current dialog settings."""
        # Collect TrainingJobs and params from form
        form_data = self.form_widget.get_form_data()
        training_jobs = self._get_current_training_jobs()

        # Close the dialog now that we have the data from it
        self.accept()

        frames_to_predict = dict()

        if self._frame_selection is not None:
            predict_frames_choice = form_data.get("_predict_frames", "")
            if predict_frames_choice.startswith("current frame"):
                frames_to_predict = self._frame_selection["frame"]
            elif predict_frames_choice.startswith("random"):
                frames_to_predict = self._frame_selection["random"]
            elif predict_frames_choice.startswith("selected clip"):
                frames_to_predict = self._frame_selection["clip"]
                with_tracking = True
            elif predict_frames_choice.startswith("suggested"):
                frames_to_predict = self._frame_selection["suggestions"]
            elif predict_frames_choice.startswith("entire video"):
                frames_to_predict = self._frame_selection["video"]

        # for key, val in training_jobs.items():
        #     print(key)
        #     print(val)
        #     print()
        # print(form_data)

        # Run training/inference pipeline using the TrainingJobs
        new_counts = run_learning_pipeline(
            labels_filename=self.labels_filename,
            labels=self.labels,
            training_jobs=training_jobs,
            inference_params=form_data,
            frames_to_predict=frames_to_predict,
        )

        self.learningFinished.emit()

        if new_counts >= 0:
            QtWidgets.QMessageBox(
                text=f"Inference has finished. Instances were predicted on {new_counts} frames."
            ).exec_()

    def view_datagen(self):
        """Shows windows with sample visual data that will be used training."""

        from sleap.nn import data
        from sleap.io.video import Video
        from sleap.gui.overlays.confmaps import demo_confmaps
        from sleap.gui.overlays.pafs import demo_pafs

        training_data = data.TrainingData.from_labels(self.labels)
        ds = training_data.to_ds()

        conf_job, _ = self.job_menu_manager.get_current_job("confmap")

        # settings for datagen
        form_data = self.form_widget.get_form_data()
        scale = form_data.get("scale", conf_job.trainer.scale)
        sigma = form_data.get("sigma", None)
        sigma_confmaps = form_data.get("sigma_confmaps", sigma)
        sigma_pafs = form_data.get("sigma_pafs", sigma)
        instance_crop = form_data.get("instance_crop", conf_job.trainer.instance_crop)
        bounding_box_size = form_data.get(
            "bounding_box_size", conf_job.trainer.bounding_box_size
        )
        # negative_samples = form_data.get("negative_samples", 0)

        # Augment dataset
        aug_params = dict(
            # rotate=conf_job.trainer.augment_rotate,
            # rotation_min_angle=-conf_job.trainer.augment_rotation,
            # rotation_max_angle=conf_job.trainer.augment_rotation,
            scale=form_data.get("scale", conf_job.trainer.scale),
            # scale_min=conf_job.trainer.augment_scale_min,
            # scale_max=conf_job.trainer.augment_scale_max,
            # uniform_noise=conf_job.trainer.augment_uniform_noise,
            # min_noise_val=conf_job.trainer.augment_uniform_noise_min_val,
            # max_noise_val=conf_job.trainer.augment_uniform_noise_max_val,
            # gaussian_noise=conf_job.trainer.augment_gaussian_noise,
            # gaussian_noise_mean=conf_job.trainer.augment_gaussian_noise_mean,
            # gaussian_noise_stddev=conf_job.trainer.augment_gaussian_noise_stddev,
            contrast=conf_job.trainer.augment_contrast,
            contrast_min_gamma=conf_job.trainer.augment_contrast_min_gamma,
            contrast_max_gamma=conf_job.trainer.augment_contrast_max_gamma,
            brightness=conf_job.trainer.augment_brightness,
            brightness_val=conf_job.trainer.augment_brightness_val,
        )
        ds = data.augment_dataset(ds, **aug_params)

        if bounding_box_size is None or bounding_box_size <= 0:
            bounding_box_size = data.estimate_instance_crop_size(
                training_data.points,
                min_multiple=conf_job.model.input_min_multiple,
                padding=conf_job.trainer.instance_crop_padding,
            )

        if instance_crop:
            ds = data.instance_crop_dataset(
                ds, box_height=bounding_box_size, box_width=bounding_box_size
            )

        skeleton = self.labels.skeletons[0]

        if conf_job.model.output_type == ModelOutputType.CONFIDENCE_MAP:
            conf_data = data.make_confmap_dataset(
                ds, output_scale=scale, sigma=sigma_confmaps,
            )
        elif conf_job.model.output_type == ModelOutputType.TOPDOWN_CONFIDENCE_MAP:
            conf_data = data.make_instance_confmap_dataset(
                ds, with_ctr_peaks=True, output_scale=scale, sigma=sigma_confmaps,
            )

        imgs = []
        confmaps = []
        for img, confmap in conf_data.take(10):
            if type(confmap) == tuple:
                confmap = confmap[0]
            imgs.append(img)
            confmaps.append(confmap)

        imgs = np.stack(imgs)
        confmaps = np.stack(confmaps)
        conf_vid = Video.from_numpy(imgs * 255)

        conf_win = demo_confmaps(confmaps, conf_vid)
        conf_win.activateWindow()
        conf_win.resize(bounding_box_size + 50, bounding_box_size + 50)
        conf_win.move(200, 200)

        if ModelOutputType.PART_AFFINITY_FIELD in self._get_current_training_jobs():
            paf_data = data.make_paf_dataset(
                ds,
                data.SimpleSkeleton.from_skeleton(skeleton).edges,
                output_scale=scale,
                distance_threshold=sigma_pafs,
            )

            imgs = []
            pafs = []
            for img, paf in paf_data.take(10):
                imgs.append(img)
                pafs.append(paf)

            imgs = np.stack(imgs)
            pafs = np.stack(pafs)
            paf_vid = Video.from_numpy(imgs * 255)

            paf_win = demo_pafs(pafs, paf_vid)
            paf_win.activateWindow()
            paf_win.resize(bounding_box_size + 50, bounding_box_size + 50)
            paf_win.move(220 + conf_win.rect().width(), 200)

        # FIXME: hide dialog so use can see other windows
        # can we show these windows without closing dialog?
        self.hide()

    def _view_profile(self, filename: str, menu_name: str, windows=[]):
        """Opens profile editor in new dialog window."""
        saved_files = []
        win = TrainingEditor(filename, saved_files=saved_files, parent=self)
        windows.append(win)
        win.exec_()

        for new_filename in saved_files:
            self.job_menu_manager.add_job_to_list(new_filename, menu_name)

    def update_fields_from_job(self, job: TrainingJob):
        model_type = job.model.output_type

        training_params = cattr.unstructure(job.trainer)
        training_params_specific = {
            f"{key}_{str(model_type)}": val for key, val in training_params.items()
        }
        # confmap and paf models should share some params shown in dialog (e.g. scale)
        # but centroids does not, so just set any centroid_foo fields from its profile
        if model_type in [ModelOutputType.CENTROIDS]:
            training_params = training_params_specific
        else:
            training_params = {**training_params, **training_params_specific}
        self.form_widget.set_form_data(training_params)

        # is the model already trained?
        is_trained = job.is_trained
        field_name = f"_use_trained_{str(model_type)}"
        # update "use trained" checkbox if present
        if field_name in self.form_widget.fields:
            self.form_widget.fields[field_name].setEnabled(is_trained)
            self.form_widget[field_name] = is_trained

    def on_job_menu_selection(self, menu_name: str, selected_idx: int, field):
        """Handles when user selects an item from model/job menu.

        If selection is a valid job, then update form fields from job.
        If selection is "add", then show the appropriate gui for selecting job.
        """
        if selected_idx == -1:
            return

        job = None
        field_text = field.currentText()
        if field_text == SELECT_FILE_OPTION:
            job = self.job_menu_manager.add_job_gui(menu_name)

        else:
            path, job = self.job_menu_manager.get_menu_item(menu_name, selected_idx)

        if job is not None:
            self.update_fields_from_job(job)


@attr.s(auto_attribs=True)
class JobMenuManager:

    labels_filename: str
    job_option_widgets: dict  # keyed by menu name
    job_options_by_menu: dict = attr.ib(factory=dict)  # keyed by model type
    strict_confmap_type: bool = False
    require_trained: bool = False
    menu_selection_callback: Optional[Callable] = None

    def rebuild(self):
        """
        Rebuilds list of profile options (checking for new profile files).
        """
        # load list of job profiles from directory
        profile_dir = util.get_package_file("sleap/training_profiles")

        self.job_options_by_menu = dict()

        # list any profiles from previous runs
        if self.labels_filename:
            models_dir = os.path.join(os.path.dirname(self.labels_filename), "models")
            if os.path.exists(models_dir):
                self.find_saved_jobs(models_dir, self.job_options_by_menu)
        # list default profiles (without searching subdirs)
        self.find_saved_jobs(profile_dir, self.job_options_by_menu, depth=0)

        # Apply any filters
        if self.require_trained:
            for key, jobs_list in self.job_options_by_menu.items():
                self.job_options_by_menu[key] = [
                    (path, job) for (path, job) in jobs_list if job.is_trained
                ]

    def get_menu_options(self, menu_name: str):
        """Returns the list of (path, TrainingJob) tuples for menu."""
        if menu_name in self.job_options_by_menu:
            return self.job_options_by_menu[menu_name]
        else:
            return []
        # menu_options = []
        # for model_type in MENU_NAME_TYPE_MAP[menu_name]:
        #     if model_type in self.job_options_by_model_type:
        #         menu_options.extend(self.job_options_by_model_type[model_type])
        # return menu_options

    def option_list_from_jobs_list(self, jobs):
        """Returns list of menu options for given model type."""
        option_list = [name for (name, job) in jobs]
        option_list.append("")
        option_list.append("---")
        option_list.append(SELECT_FILE_OPTION)
        return option_list

    def update_menus(self, init: bool = False):
        """Updates the menus with training profile options.

        Args:
            init: Whether this is first time calling (so we should connect
                signals), or we're just updating menus.

        Returns:
            None.
        """

        for menu_name in self.job_option_widgets.keys():
            self.update_menu(menu_name, init=init)

    def update_menu(
        self,
        menu_name,
        select_item: Optional[str] = None,
        init: bool = False,
        signal: bool = False,
    ):
        menu_options = self.get_menu_options(menu_name)
        field = self.job_option_widgets[menu_name]

        if init:

            def menu_action(idx, menu=menu_name, field=field):
                self.menu_selection_callback(menu, idx, field)

            field.currentIndexChanged.connect(menu_action)
        elif not signal:
            # block signals so we can update combobox without overwriting
            # any user data with the defaults from the profile
            field.blockSignals(True)

        field.set_options(self.option_list_from_jobs_list(menu_options), select_item)
        # enable signals again so that choice of profile will update params
        field.blockSignals(False)

    def get_current_job(
        self, menu_name: str
    ) -> Tuple[Optional[TrainingJob], Optional[str]]:
        """Returns training job currently selected for given model type.

        Args:
            model_type: The type of model for which we want data.

        Returns: Tuple of (TrainingJob, path to job profile).
        """

        # by default use the first model for a given type
        idx = 0

        # If there's a menu, then use the selected item
        if menu_name in self.job_option_widgets:
            field = self.job_option_widgets[menu_name]
            idx = field.currentIndex()

        job_filename, job = self.get_menu_item(menu_name, idx)

        return job, job_filename

    def get_menu_item(
        self, menu_name: str, item_idx: int
    ) -> Tuple[Optional[str], Optional[TrainingJob]]:
        menu_options = self.get_menu_options(menu_name)
        if item_idx >= len(menu_options):
            return None, None

        return menu_options[item_idx]

    def insert_menu_item(self, menu_name: str, job_path, job):
        # insert at beginning of list
        self.job_options_by_menu[menu_name].insert(0, (job_path, job))
        self.update_menu(menu_name, select_item=job_path, signal=True)

    def add_job_gui(self, menu_name: str):
        """Allow user to add training profile for given model type."""
        filename, _ = FileDialog.open(
            None,
            dir=None,
            caption="Select training profile...",
            filter="TrainingJob JSON (*.json)",
        )

        self.add_job_to_list(filename, menu_name)

        # If we didn't successfully select a new file, then reset menu selection
        field = self.job_option_widgets[menu_name]
        if field.currentIndex() == field.count() - 1:  # subtract 1 for separator
            field.setCurrentIndex(-1)

    def add_job_to_list(self, filename: str, menu_name: str):
        """Adds selected training profile for given model type."""
        if len(filename):
            try:
                # try to load json as TrainingJob
                job = TrainingJob.load_json(filename)
            except:
                # but do raise any other type of error
                QtWidgets.QMessageBox(
                    text=f"Unable to load a training profile from {filename}."
                ).exec_()
                raise
            else:
                # Get the model type for the model/profile selected by user
                file_model_type = job.model.output_type

                # Make sure this is the right type for this menu
                if file_model_type in MENU_NAME_TYPE_MAP[menu_name]:

                    self.insert_menu_item(menu_name, filename, job)

                else:
                    QtWidgets.QMessageBox(
                        text=f"Profile selected is for training {str(file_model_type)} instead of {menu_name}."
                    ).exec_()

    def find_saved_jobs(
        self, job_dir: str, jobs=None, depth: int = 1
    ) -> Dict[ModelOutputType, List[Tuple[str, TrainingJob]]]:
        """Find all the TrainingJob json files in a given directory.

        Args:
            job_dir: the directory in which to look for json files
            jobs: If given, then the found jobs will be added to this object,
                rather than creating new dict.
        Returns:
            dict of {ModelOutputType: list of (filename, TrainingJob) tuples}
        """

        json_files = util.find_files_by_suffix(job_dir, ".json", depth=depth)

        # Sort files, starting with most recently modified
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        json_paths = [file.path for file in json_files]

        jobs = dict() if jobs is None else jobs
        for full_filename in json_paths:
            try:
                # try to load json as TrainingJob
                job = TrainingJob.load_json(full_filename)
            except Exception as e:
                # Couldn't load as TrainingJob so just ignore this json file
                # probably it's a json file for something else (or an json for a
                # older version of the object with different class attributes).
                print(e)
                pass
            else:
                # we loaded the json as a TrainingJob, so see what type of model it's for
                key = self.menu_name_from_model_type(job.model.output_type)
                if key not in jobs:
                    jobs[key] = []
                jobs[key].append((full_filename, job))

        return jobs

    def menu_name_from_model_type(self, model_type):

        if self.strict_confmap_type:
            conf_types = (ModelOutputType.CONFIDENCE_MAP,)
        else:
            conf_types = (
                ModelOutputType.CONFIDENCE_MAP,
                ModelOutputType.TOPDOWN_CONFIDENCE_MAP,
            )

        if model_type in conf_types:
            return "confmap"
        elif model_type == ModelOutputType.TOPDOWN_CONFIDENCE_MAP:
            return "topdown"

        if model_type == ModelOutputType.CENTROIDS:
            return "centroid"

        if model_type == ModelOutputType.PART_AFFINITY_FIELD:
            return "paf"

        return ""


def run_learning_pipeline(
    labels_filename: str,
    labels: Labels,
    training_jobs: Dict["ModelOutputType", "TrainingJob"],
    inference_params: Dict[str, str],
    frames_to_predict: Dict[Video, List[int]] = None,
) -> int:
    """Run training (as needed) and inference.

    Args:
        labels_filename: Path to already saved current labels object.
        labels: The current labels object; results will be added to this.
        training_jobs: The TrainingJobs with params/hyperparams for training.
        inference_params: Parameters to pass to inference.
        frames_to_predict: Dict that gives list of frame indices for each video.

    Returns:
        Number of new frames added to labels.

    """

    # Set the parameters specific to this run
    for job in training_jobs.values():
        job.labels_filename = labels_filename

    # TODO: only require labels_filename if we're training?
    # save_dir = os.path.join(os.path.dirname(labels_filename), "models")

    # Train the TrainingJobs
    trained_jobs = run_gui_training(labels_filename, training_jobs)

    # Check that all the models were trained
    if None in trained_jobs.values():
        return -1

    trained_job_paths = list(trained_jobs.values())

    # Run the Predictor for suggested frames
    new_labeled_frame_count = run_gui_inference(
        labels=labels,
        trained_job_paths=trained_job_paths,
        inference_params=inference_params,
        frames_to_predict=frames_to_predict,
    )

    return new_labeled_frame_count


def has_jobs_to_train(training_jobs: Dict["ModelOutputType", "TrainingJob"]):
    """Returns whether any of the jobs need to be trained."""
    return any(not getattr(job, "use_trained_model", False) for job in training_jobs)


def run_gui_training(
    labels_filename: str,
    training_jobs: Dict["ModelOutputType", "TrainingJob"],
    gui: bool = True,
) -> Dict["ModelOutputType", str]:
    """
    Run training for each training job.

    Args:
        labels: Labels object from which we'll get training data.
        training_jobs: Dict of the jobs to train.
        save_dir: Path to the directory where we'll save inference results.
        gui: Whether to show gui windows and process gui events.

    Returns:
        Dict of paths to trained jobs corresponding with input training jobs.
    """

    from sleap.nn import training

    trained_jobs = dict()

    if gui:
        from sleap.nn.monitor import LossViewer

        # open training monitor window
        win = LossViewer()
        win.resize(600, 400)
        win.show()

    for model_type, job in training_jobs.items():
        if getattr(job, "use_trained_model", False):
            # set path to TrainingJob already trained from previous run
            # json_name = f"{job.run_name}.json"
            trained_jobs[model_type] = job.run_path
            print(f"Using already trained model: {trained_jobs[model_type]}")

        else:
            # Clear save dir and run name for job we're about to train
            job.save_dir = None
            job.run_name = None

            if gui:
                print("Resetting monitor window.")
                win.reset(what=str(model_type))
                win.setWindowTitle(f"Training Model - {str(model_type)}")

            print(f"Start training {str(model_type)}...")

            def waiting():
                if gui:
                    QtWidgets.QApplication.instance().processEvents()

            # Run training
            trained_job_path, success = training.Trainer.train_subprocess(
                job, labels_filename, waiting
            )

            if success:
                # get the path to the resulting TrainingJob file
                trained_jobs[model_type] = trained_job_path
                print(f"Finished training {str(model_type)}.")
            else:
                if gui:
                    win.close()
                    QtWidgets.QMessageBox(
                        text=f"An error occurred while training {str(model_type)}. Your command line terminal may have more information about the error."
                    ).exec_()
                trained_jobs[model_type] = None

    if gui:
        # close training monitor window
        win.close()

    return trained_jobs


def run_gui_inference(
    labels: Labels,
    trained_job_paths: List[str],
    frames_to_predict: Dict[Video, List[int]],
    inference_params: Dict[str, str],
    gui: bool = True,
) -> int:
    """Run inference on specified frames using models from training_jobs.

    Args:
        labels: The current labels object; results will be added to this.
        trained_job_paths: List of paths to TrainingJobs with trained models.
        frames_to_predict: Dict that gives list of frame indices for each video.
        inference_params: Parameters to pass to inference.
        gui: Whether to show gui windows and process gui events.

    Returns:
        Number of new frames added to labels.
    """
    from sleap.nn import inference

    if gui:
        # show message while running inference
        progress = QtWidgets.QProgressDialog(
            f"Running inference on {len(frames_to_predict)} videos...",
            "Cancel",
            0,
            len(frames_to_predict),
        )
        progress.show()
        QtWidgets.QApplication.instance().processEvents()

    new_lfs = []
    for i, (video, frames) in enumerate(frames_to_predict.items()):

        if len(frames):

            def waiting():
                if gui:
                    QtWidgets.QApplication.instance().processEvents()
                    progress.setValue(i)
                    if progress.wasCanceled():
                        return -1

            # Run inference for desired frames in this video
            predictions_path, success = inference.Predictor.predict_subprocess(
                video=video,
                frames=frames,
                trained_job_paths=trained_job_paths,
                kwargs=inference_params,
                waiting_callback=waiting,
            )

            if success:
                predictions_labels = Labels.load_file(predictions_path, match_to=labels)
                new_lfs.extend(predictions_labels.labeled_frames)
            else:
                if gui:
                    progress.close()
                    QtWidgets.QMessageBox(
                        text=f"An error occcured during inference. Your command line terminal may have more information about the error."
                    ).exec_()
                return -1

    # Remove any frames without instances
    new_lfs = list(filter(lambda lf: len(lf.instances), new_lfs))

    # Merge predictions into current labels dataset
    _, _, new_conflicts = Labels.complex_merge_between(
        labels,
        new_labels=Labels(new_lfs),
        unify=False,  # since we used match_to when loading predictions file
    )

    # new predictions should replace old ones
    Labels.finish_complex_merge(labels, new_conflicts)

    # close message window
    if gui:
        progress.close()

    # return total_new_lf_count
    return len(new_lfs)


if __name__ == "__main__":
    import sys

    #     labels_filename = "/Volumes/fileset-mmurthy/nat/shruthi/labels-mac.json"
    labels_filename = sys.argv[1]
    labels = Labels.load_file(labels_filename)

    app = QtWidgets.QApplication()
    win = InferenceDialog(
        labels=labels, labels_filename=labels_filename, mode="inference"
    )
    win.show()
    app.exec_()

#     labeled_frames = run_active_learning_pipeline(labels_filename)
#     print(labeled_frames)
