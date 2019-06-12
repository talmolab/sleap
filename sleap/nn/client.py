import numpy as np
import multiprocessing as mp
from multiprocessing import sharedctypes
import threading
import zmq
import jsonpickle
import yaml

from PySide2 import QtCore, QtWidgets

from sleap.gui.formbuilder import YamlFormWidget
from sleap.nn.monitor import LossViewer

class TrainingDialog(QtWidgets.QMainWindow):

    def __init__(self, zmq_context=None, server="127.0.0.1", *args, **kwargs):
        super(TrainingDialog, self).__init__(*args, **kwargs)

        self.zmq_context = zmq_context

        # Controller
        if self.zmq_context is not None:
            self.zmq_ctrl = self.zmq_context.socket(zmq.PUB)
            self.zmq_ctrl.bind("tcp://*:9000")

        # Data
        self.labels_file = "tests/data/json_format_v1/centered_pair.json"
        self.default_scale = 1
        self.is_training = False
        self.training_data = dict(ready=False, times=None, update=threading.Event())

        # UI
        self.form_widget = YamlFormWidget(yaml_file="sleap/nn/training-forms.yaml", title="Training Parameters")
        self.form_widget.mainAction.connect(self.run_training)
        self.form_widget.valueChanged.connect(self.update_ui)

        self.setCentralWidget(self.form_widget)

        # Default values for testing
        
        default_data = dict(
            num_filters = 32,
            augment_rotation = 0,
            val_size = 0.1,
            num_epochs = 20,
            batch_size = 16,
            steps_per_epoch = 100,
            val_batches_per_epoch = 10,
            shuffle_every_epoch = False,
            learning_rate = 1e-4,
            reduce_lr_factor = 0.1,
            reduce_lr_patience = 2,
            reduce_lr_cooldown = 0,
            reduce_lr_min_delta = 1e-5,
            reduce_lr_min_lr = 1e-10,
            upsampling_layers = True,
            save_every_epoch = False,
            amsgrad = True,
            depth = 3,
            )

        unet_data = dict(
            arch='unet',
            num_filters=32,
            num_epochs=100,
            steps_per_epoch=200,
            batch_size=4,
            shuffle_every_epoch=True,
            augment_rotation=180,
            reduce_lr_patience=5,
            reduce_lr_factor=0.5,
            reduce_lr_cooldown=3,
            reduce_lr_min_delta=1e-6,
            )

        leap_cnn_data = dict(
            num_filters=64,
            num_epochs=75,
            reduce_lr_patience=8,
            reduce_lr_factor=0.5,
            reduce_lr_cooldown=3,
            reduce_lr_min_delta=1e-6,
            batch_size=4,
            shuffle_every_epoch=True,
            augment_rotation=180,
            arch="leap_cnn",
            )

        self.form_widget.set_form_data(default_data)
        # self.form_widget.set_form_data(unet_data)

        self.update_ui()

    def update_ui(self, *args):
        current_form_data = self.form_widget.get_form_data()
        self.labels_file = current_form_data["_labels_file"]

        has_labels = self.labels_file != ""

        run_button = self.form_widget.buttons["run_button"]
        run_button.setEnabled(has_labels)

        training_button_text = "Stop Training" if self.is_training else "Start Training"
        run_button.setText(training_button_text)

    def run_training(self, *args):
        if not self.is_training:
            # Get data from form fields
            training_params = self.form_widget.get_form_data()

            # Datagen for images and points
            # confmaps/pafs are generated live during training

            from sleap.io.dataset import Labels
            from sleap.nn.datagen import generate_images, generate_points

            labels = Labels.load_json(training_params["_labels_file"])

            # TODO: support multiple skeletons
            skeleton = labels.skeletons[0]
            imgs = generate_images(labels)
            points = generate_points(labels)

            # Make any adjustments to params we'll pass to training

            del training_params["_labels_file"]
            if training_params.get("save_dir", "") == "":
                training_params["save_dir"] = None

            # Start training

            from sleap.nn.training import train
            mp.Process(target=train,
                args=(imgs, points, skeleton),
                kwargs=training_params
                ).start()

            # TODO: open training monitor(s)
            loss_viewer = LossViewer(zmq_context=self.zmq_context, parent=self)
            loss_viewer.resize(600, 400)
            loss_viewer.show()
            timer = QtCore.QTimer()
            timer.timeout.connect(loss_viewer.check_messages)
            timer.start(0)

            self.is_training = True
        else:
            if self.zmq_context is not None:
                # send command to stop training
                print("Sending command to stop training")
                self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="stop",)))
            self.is_training = False

        self.update_ui()

    def check_messages(self, *args):
        pass


if __name__ == "__main__":

    server_address = "127.0.0.1"

    print(f"starting client to {server_address}")

    ctx = None
    ctx = zmq.Context()

    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(True)

    training_window = TrainingDialog(zmq_context=ctx, server=server_address)
    training_window.show()

    # timer = QtCore.QTimer()
    # timer.timeout.connect(training_window.check_messages)
    # timer.start(0)

    app.exec_()
