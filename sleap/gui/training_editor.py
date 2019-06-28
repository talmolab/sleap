import os
import attr
import cattr
from typing import Optional

from pkg_resources import Requirement, resource_filename

from PySide2 import QtWidgets

from sleap.io.dataset import Labels
from sleap.gui.formbuilder import YamlFormWidget

class TrainingEditor(QtWidgets.QDialog):

    def __init__(self, profile_filename: Optional[str]=None, saved_files: list=[], *args, **kwargs):
        super(TrainingEditor, self).__init__()

        form_yaml = resource_filename(Requirement.parse("sleap"),"sleap/config/training_editor.yaml")

        self.form_widgets = dict()
        self.form_widgets["model"] = YamlFormWidget(form_yaml, "model", "Network Architecture")
        self.form_widgets["datagen"] = YamlFormWidget(form_yaml, "datagen", "Data Generation/Preprocessing")
        self.form_widgets["trainer"] = YamlFormWidget(form_yaml, "trainer", "Trainer")
        self.form_widgets["output"] = YamlFormWidget(form_yaml, "output",)
        self.form_widgets["buttons"] = YamlFormWidget(form_yaml, "buttons")

        self.form_widgets["buttons"].mainAction.connect(self._save_as)

        col1_layout = QtWidgets.QVBoxLayout()
        col2_layout = QtWidgets.QVBoxLayout()

        col1_layout.addWidget(self.form_widgets["model"])
        col1_layout.addWidget(self.form_widgets["datagen"])
        col1_layout.addWidget(self.form_widgets["output"])

        col2_layout.addWidget(self.form_widgets["trainer"])
        col2_layout.addWidget(self.form_widgets["buttons"])

        col_layout = QtWidgets.QHBoxLayout()
        col_layout.addWidget(self._layout_widget(col1_layout))
        col_layout.addWidget(self._layout_widget(col2_layout))

        self.setLayout(col_layout)

        self.profile_filename = profile_filename
        self.saved_files = saved_files

    @property
    def profile_filename(self):
        return self._profile_filename

    @profile_filename.setter
    def profile_filename(self, val):
        self._profile_filename = val
        # set window title
        self.setWindowTitle(self.profile_filename)
        # load file
        if self.profile_filename:
            self._load_profile(self.profile_filename)

    @staticmethod
    def _layout_widget(layout):
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def _load_profile(self, profile_filename:str):
        from sleap.nn.model import ModelOutputType
        from sleap.nn.training import TrainingJob

        self.training_job = TrainingJob.load_json(profile_filename)

        job_dict = cattr.unstructure(self.training_job)

        job_dict["model"]["arch"] = job_dict["model"]["backbone_name"]
        job_dict["model"]["output_type"] = str(self.training_job.model.output_type)

        self.form_widgets["model"].set_form_data(job_dict["model"])
        self.form_widgets["model"].set_form_data(job_dict["model"]["backbone"])
        for name in "datagen,trainer,output".split(","):
            self.form_widgets[name].set_form_data(job_dict["trainer"])

    def _update_profile(self):
        # update training job from params in form
        trainer = job.trainer
        for key, val in form_data.items():
            # check if form field matches attribute of Trainer object
            if key in dir(trainer):
                setattr(trainer, key, val)

    def _save_as(self):

        # Show "Save" dialog
        save_filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, caption="Save As...", dir=None, filter="Profile JSON (*.json)")

        if len(save_filename):
            from sleap.nn.model import Model, ModelOutputType
            from sleap.nn.training import Trainer, TrainingJob
            from sleap.nn.architectures import unet, leap, hourglass

            # Construct Model
            model_data = self.form_widgets["model"].get_form_data()
            arch = dict(LeapCNN=leap.LeapCNN,
                        StackedHourglass=hourglass.StackedHourglass,
                        UNet=unet.UNet,
                        StackedUNet=unet.StackedUNet,
                        )[model_data["arch"]]

            output_type = dict(confmaps=ModelOutputType.CONFIDENCE_MAP,
                               pafs=ModelOutputType.PART_AFFINITY_FIELD,
                                )[model_data["output_type"]]

            backbone_kwargs = {key:val for key, val in model_data.items()
                                if key in attr.fields_dict(arch).keys()}

            model = Model(output_type=output_type, backbone=arch(**backbone_kwargs))

            # Construct Trainer
            trainer_data = {**self.form_widgets["datagen"].get_form_data(),
                            **self.form_widgets["output"].get_form_data(),
                            **self.form_widgets["trainer"].get_form_data(),
                            }

            trainer_kwargs = {key:val for key, val in trainer_data.items()
                                if key in attr.fields_dict(Trainer).keys()}
            trainer = Trainer(**trainer_kwargs)

            # Construct TrainingJob
            training_job_kwargs = {key:val for key, val in trainer_data.items()
                                    if key in attr.fields_dict(TrainingJob).keys()}
            training_job = TrainingJob(model, trainer, **training_job_kwargs)

            # Write the file
            TrainingJob.save_json(training_job, save_filename)

            self.saved_files.append(save_filename)
            # print(cattr.unstructure(training_job))
            self.profile_filename = save_filename

        self.close()

if __name__ == "__main__":
    import sys

    profile_filename = None if len(sys.argv) <= 1 else sys.argv[1]

    # profile_filename = "training_profiles/default_confmaps.json"

    app = QtWidgets.QApplication([])
    win = TrainingEditor(profile_filename)
    win.show()
    app.exec_()