import os
import cattr

from pkg_resources import Requirement, resource_filename

from sleap.io.dataset import Labels
from sleap.gui.training_editor import TrainingEditor

from PySide2 import QtWidgets

class ActiveLearningDialog(QtWidgets.QDialog):

    def __init__(self, labels_filename: str, labels: Labels, *args, **kwargs):
        super(ActiveLearningDialog, self).__init__(*args, **kwargs)

        self.labels_filename = labels_filename
        self.labels = labels

        from sleap.gui.formbuilder import YamlFormWidget
        from sleap.nn.model import ModelOutputType
        from sleap.nn.training import TrainingJob

        learning_yaml = resource_filename(Requirement.parse("sleap"),"config/active.yaml")
        self.form_widget = YamlFormWidget(yaml_file=learning_yaml, title="Active Learning Settings")

        # load list of job profiles from directory
        self.job_options = find_saved_jobs(resource_filename(Requirement.parse("sleap"),"training_profiles"))

        # form ui

        self.training_profile_widgets = {
                ModelOutputType.CONFIDENCE_MAP: self.form_widget.fields["conf_job"],
                ModelOutputType.PART_AFFINITY_FIELD: self.form_widget.fields["paf_job"],
                }

        for model_type, field in self.training_profile_widgets.items():
            if model_type not in self.job_options:
                self.job_options[model_type] = []
            field.currentIndexChanged.connect(lambda idx, mt=model_type: self.select_job(mt, idx))
            field.set_options(self.option_list_from_jobs(model_type))

        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        buttons.addButton("Run Active Learning", QtWidgets.QDialogButtonBox.AcceptRole)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.form_widget)
        layout.addWidget(buttons)

        self.setLayout(layout)

        # connect actions to buttons
        self.form_widget.buttons["_view_conf"].clicked.connect(lambda: self.view_profile(self.form_widget["conf_job"]))
        self.form_widget.buttons["_view_paf"].clicked.connect(lambda: self.view_profile(self.form_widget["paf_job"]))
        buttons.accepted.connect(self.run)
        buttons.rejected.connect(self.reject)

    def run(self):
        # Collect TrainingJobs and params from form
        form_data = self.form_widget.get_form_data()
        training_jobs = dict()
        for model_type, field in self.training_profile_widgets.items():
            idx = field.currentIndex()
            job_filename, job = self.job_options[model_type][idx]
            training_jobs[model_type] = job
            # update training job from params in form
            trainer = job.trainer
            for key, val in form_data.items():
                # check if form field matches attribute of Trainer object
                if key in dir(trainer):
                    setattr(trainer, key, val)

        # Close the dialog now that we have the data from it
        self.accept()

        # Run active learning pipeline using the TrainingJobs
        new_lfs = run_active_learning_pipeline(self.labels_filename, self.labels, training_jobs)

        # Update labels with results of active learning
        self.labels.labeled_frames.extend(new_lfs)

    # open profile editor in new dialog window
    def view_profile(self, filename, windows=[]):
        win = TrainingEditor(filename, parent=self)
        windows.append(win)
        win.exec_()

    def option_list_from_jobs(self, model_type):
        jobs = self.job_options[model_type]
        option_list = [name for (name, job) in jobs]
        option_list.append("---")
        option_list.append("Select a training profile file...")
        return option_list

    def add_job_file(self, model_type):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None, dir=None,
                        caption="Select training profile...",
                        filter="TrainingJob JSON (*.json)")
        if len(filename):
            try:
                # try to load json as TrainingJob
                job = TrainingJob.load_json(filename)
            except:
                # but do raise any other type of error
                QtWidgets.QMessageBox(text=f"Unable to load a training profile from {filename}.").exec_()
                raise
            else:
                # we loaded the json as a TrainingJob, so see what type of model it's for
                file_model_type = job.model.output_type
                # make sure the users selected a file with the right model type
                if model_type == file_model_type:
                    self.job_options[model_type].append((filename, job))
                    # update ui list
                    field = self.training_profile_widgets[model_type]
                    field.set_options(self.option_list_from_jobs(model_type), filename)
                else:
                    QtWidgets.QMessageBox(text=f"Profile selected is for training {str(file_model_type)} instead of {str(model_type)}.").exec_()
        field = self.training_profile_widgets[model_type]
        # if we didn't successfully select a new file, then clear selection
        if field.currentIndex() == field.count()-1: # subtract 1 for separator
            field.setCurrentIndex(-1)

    def select_job(self, model_type, idx):
        jobs = self.job_options[model_type]
        if idx == -1: return
        if idx < len(jobs):
            name, job = jobs[idx]
            training_params = cattr.unstructure(job.trainer)
            self.form_widget.set_form_data(training_params)
        else:
            # last item is "select file..."
            self.add_job_file(model_type)


def make_default_training_jobs():
    from sleap.nn.model import Model, ModelOutputType
    from sleap.nn.training import Trainer, TrainingJob
    from sleap.nn.architectures import unet, leap

    # Build Models (wrapper for Keras model with some metadata)

    models = dict()
    models[ModelOutputType.CONFIDENCE_MAP] = Model(
            output_type=ModelOutputType.CONFIDENCE_MAP,
            backbone=unet.UNet(num_filters=32))
    models[ModelOutputType.PART_AFFINITY_FIELD] = Model(
            output_type=ModelOutputType.PART_AFFINITY_FIELD,
            backbone=leap.LeapCNN(num_filters=64))

    # Build Trainers

    defaults = dict()
    defaults["shared"] = dict(
            instance_crop = True,
            val_size = 0.1,
            augment_rotation=180,
            batch_size=4,
            learning_rate = 1e-4,
            reduce_lr_factor=0.5,
            reduce_lr_cooldown=3,
            reduce_lr_min_delta=1e-6,
            reduce_lr_min_lr = 1e-10,
            amsgrad = True,
            shuffle_every_epoch=True,
            save_every_epoch = False,
#             val_batches_per_epoch = 10,
#             upsampling_layers = True,
#             depth = 3,
    )
    defaults[ModelOutputType.CONFIDENCE_MAP] = dict(
            **defaults["shared"],
            num_epochs=100,
            steps_per_epoch=200,
            reduce_lr_patience=5,
            )

    defaults[ModelOutputType.PART_AFFINITY_FIELD] = dict(
            **defaults["shared"],
            num_epochs=75,
            steps_per_epoch = 100,
            reduce_lr_patience=8,
            )

    trainers = dict()
    for type in models.keys():
        trainers[type] = Trainer(**defaults[type])

    # Build TrainingJobs from Models and Trainers

    training_jobs = dict()
    for type in models.keys():
        training_jobs[type] = TrainingJob(models[type], trainers[type])

    return training_jobs

def find_saved_jobs(job_dir):
    """Find all the TrainingJob json files in a given directory.

    Args:
        job_dir: the directory in which to look for json files
    Returns:
        dict of {ModelOutputType: list of (filename, TrainingJob) tuples}
    """
    from sleap.nn.training import TrainingJob

    files = os.listdir(job_dir)

    json_files = [f for f in files if f.endswith(".json")]

    jobs = dict()
    for f in json_files:
        full_filename = os.path.join(job_dir, f)
        try:
            # try to load json as TrainingJob
            job = TrainingJob.load_json(full_filename)
        except ValueError:
            # couldn't load as TrainingJob so just ignore this json file
            # probably it's a json file for something else
            pass
        except:
            # but do raise any other type of error
            raise
        else:
            # we loaded the json as a TrainingJob, so see what type of model it's for
            model_type = job.model.output_type
            if model_type not in jobs:
                jobs[model_type] = []
            jobs[model_type].append((full_filename, job))

    return jobs

def run_active_learning_pipeline(labels_filename, labels=None, training_jobs=None, skip_learning=False):
    # Imports here so we don't load TensorFlow before necessary
    from sleap.nn.monitor import LossViewer
    from sleap.nn.training import TrainingJob
    from sleap.nn.model import ModelOutputType
    from sleap.nn.inference import Predictor

    from PySide2 import QtWidgets

    labels = labels or Labels.load_json(labels_filename)

    # If the video frames are large, determine factor to downsample for active learning
#     scale = 1
#     rescale_under = 1024
#     h, w = labels.videos[0].height, labels.videos[0].width
#     largest_dim = max(h, w)
#     while largest_dim/scale > rescale_under:
#         scale += 1

    # Prepare our TrainingJobs

    # Load the defaults we use for active learning
    if training_jobs is None:
        training_jobs = make_default_training_jobs()

    # Set the parameters specific to this run
    for job in training_jobs.values():
        job.labels_filename = labels_filename
#         job.trainer.scale = scale

    # Run the TrainingJobs

    save_dir = os.path.dirname(labels_filename)

    for model_type, job in training_jobs.items():
        run_name = f"models_{model_type}"
        # use line below if we want to load models already trained from previous run
        # training_jobs[model_type] = os.path.join(save_dir, run_name+".json")

        # open monitor window
        win = LossViewer()
        win.resize(600, 400)
        win.show()

        if not skip_learning:
            # run training
            pool, result = job.trainer.train_async(model=job.model, labels=labels,
                                    save_dir=save_dir, run_name=run_name)

            while not result.ready():
                QtWidgets.QApplication.instance().processEvents()
                result.wait(.1)

            # get the result
            training_jobs[model_type] = result.get()

        # close monitor used for this trainer
        win.close()

    if not skip_learning:
        for model_type, job in training_jobs.items():
            # load job from json
            training_jobs[model_type] = TrainingJob.load_json(training_jobs[model_type])

    if not skip_learning:
        # Create Predictor from the results of training
        predictor = Predictor.from_training_jobs(training_jobs, labels)

    # Run the Predictor for suggested frames
    # We want to predict for suggested frames that don't already have user instances

    new_labeled_frames = []
    user_labeled_frames = labels.user_labeled_frames

    # show message while running inference
    win = QtWidgets.QProgressDialog()
    win.setLabelText("    Running inference on selected frames...    ")
    win.show()
    QtWidgets.QApplication.instance().processEvents()

    for video in labels.videos:
        frames = labels.get_video_suggestions(video)
        if len(frames):
            # remove frames that already have user instances
            video_user_labeled_frame_idxs = [lf.frame_idx for lf in user_labeled_frames
                                                if lf.video == video]
            frames = list(set(frames) - set(video_user_labeled_frame_idxs))

            if not skip_learning:
                # run predictions for desired frames in this video
                video_lfs = predictor.predict(input_video=video, frames=frames)
                # FIXME: code below makes the last training job process run again
                # pool, result = predictor.predict_async(input_video=video.filename, frames=frames)

                # while not result.ready():
                #     QtWidgets.QApplication.instance().processEvents()
                #     result.wait(.1)
                # video_lfs = result.get()
            else:
                import time
                time.sleep(1)
                video_lfs = []

            new_labeled_frames.extend(video_lfs)

    # close message window
    win.close()

    return new_labeled_frames

if __name__ == "__main__":
    import sys

#     labels_filename = "/Volumes/fileset-mmurthy/nat/shruthi/labels-mac.json"
    labels_filename = sys.argv[1]

    labeled_frames = run_active_learning_pipeline(labels_filename)
    print(labeled_frames)
