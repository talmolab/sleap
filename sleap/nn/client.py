import numpy as np
import multiprocessing as mp
from multiprocessing import sharedctypes
import threading
import zmq
import jsonpickle
import yaml

from PySide2 import QtCore, QtWidgets

from sleap.gui.slider import VideoSlider
from sleap.gui.formbuilder import FormBuilderLayout
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

        # UI Widgets

        with open("sleap/nn/training-forms.yaml", 'r') as forms_yaml:
            items_to_create = yaml.load(forms_yaml, Loader=yaml.SafeLoader)

        # Data Gen form
        data_gen_form = FormBuilderLayout(items_to_create["data_gen"])
        data_gen_form.valueChanged.connect(self.refresh)

        self.data_gen_button = QtWidgets.QPushButton("Generate Data")
        self.data_gen_button.clicked.connect(self.generateData)
        self.data_gen_button.clicked.connect(self.refresh)
        self.debug_button = QtWidgets.QPushButton("Debug")
        self.debug_button.clicked.connect(self.debug)
        data_gen_form.addRow(self.data_gen_button)
        data_gen_form.addRow(self.debug_button)

        # Training form
        training_form = FormBuilderLayout(items_to_create["training"])

        self.training_button = QtWidgets.QPushButton("Start Training")
        self.training_button.clicked.connect(self.runTraining)
        training_form.addRow(self.training_button)

        # UI Group Widgets

        self.gen_group = QtWidgets.QGroupBox("Data Generation")
        self.gen_group.setLayout(data_gen_form)

        self.training_group = QtWidgets.QGroupBox("Training")
        self.training_group.setLayout(training_form)

        # UI Layout

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.gen_group)
        self.layout.addWidget(self.training_group)

#         self.setLayout(self.layout)
        self.main_widget = QtWidgets.QWidget()
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        self.refresh()

    def refresh(self, *args):
        print("refresh called")

        self.labels_file = self.gen_group.layout().get_form_data()["labels_file"]

        has_labels = self.labels_file != ""
        has_data = self.training_data["ready"]
        print(f"has_data: {has_data}")

        self.data_gen_button.setEnabled(has_labels)

        training_button_text = "Stop Training" if self.is_training else "Start Training"
#         training_button_text = str(has_data)
        self.training_button.setText(training_button_text)
        self.training_button.setEnabled(has_data)

    def generateData(self, *args):
#         run_data_gen(self.labels_file, self.training_data)
        threading.Thread(target=run_data_gen, args=(self.labels_file, self.training_data)).start()
#         cmd = dict(command="data_gen", labels_file=self.labels_file)
#         self.zmq_ctrl.send_string(jsonpickle.encode(cmd))

    def runTraining(self, *args):
        if not self.is_training:
            if True:
                # Get data from form fields
                training_params = self.training_group.layout().get_form_data()
                # Adjust raw data from form fields
                if training_params.get("save_dir", "") == "":
                    training_params["save_dir"] = None
                # Start training process
                from sleap.nn.training import train
                mp.Process(target=train,
                    args=(self.training_data["imgs"], self.training_data["confmaps"]),
                    kwargs=training_params
                    ).start()
            else:
                print("SET TO NOT RUN TRAINING")
                print(self.training_group.layout().get_form_data())

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

        self.refresh()

#         self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="train",)))
#         print("sent train command")

    def preview(self):
        from sleap.io.video import Video
        from sleap.gui.video import QtVideoPlayer
        from sleap.gui.confmapsplot import ConfMapsPlot
        from sleap.gui.quiverplot import MultiQuiverPlot
        print("in preview")
        imgs = self.training_data["imgs"]
        confmaps = self.training_data["confmaps"]
        print(imgs.shape)
        print(confmaps.shape)
        vid = Video.from_numpy(imgs * 255)
        conf_window = QtVideoPlayer(video=vid)
        conf_window.show()

        def plot_confmaps(parent, item_idx):
            frame_conf_map = ConfMapsPlot(confmaps[parent.frame_idx,...])
            conf_window.view.scene.addItem(frame_conf_map)

        conf_window.changedPlot.connect(plot_confmaps)
        conf_window.plot()

    def check_messages(self, *args):
         if self.training_data["update"].is_set():
            self.training_data["update"].clear()
#             training_window.training_data["imgs"] = np.ctypeslib.as_array(training_window.training_data["imgs_raw"])
#             training_window.training_data["confmaps"] = np.ctypeslib.as_array(training_window.training_data["confmaps_raw"])
            self.preview()
            self.refresh()

    def debug(self, *args):
        # print(self.gen_group.layout().get_form_data())
        # print(self.training_group.layout().get_form_data())
        self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="test command",)))
        # self.preview()

def run_data_gen(labels_file, results):
#     local_data = threading.local()
#     local_data.results = dict()
#     results = dict()

#     ctx = zmq.Context()
#     pub = ctx.socket(zmq.PUB)
#     pub.bind("tcp://*:9001")
#     pub.send_string(jsonpickle.encode(dict(event="starting data_gen",)))
#     pub.send(umsgpack.packb(dict(event="starting data_gen")))

    from time import time, sleep
    results["ready"] = False
    results["times"] = dict()
    results["times"]["start_time"] = time()

    results["times"]["start_io"] = time()
    from sleap.io.dataset import Labels
    labels = Labels.load_json(labels_file)
    results["times"]["end_io"] = time()

    # TESTING: just use a few frames
    labels.labeled_frames = labels.labeled_frames[0:2]

    from sleap.nn.datagen import generate_images, generate_confidence_maps

    results["times"]["start_imgs"] = time()
    imgs = generate_images(labels)
    results["times"]["end_imgs"] = time()

    results["times"]["start_conf"] = time()
    confmaps = generate_confidence_maps(labels, sigma=5)
    results["times"]["end_conf"] = time()

    results["times"]["end_time"] = time()

    results["times"]["total"] = results["times"]["end_time"] - results["times"]["start_time"]
    results["times"]["io"] = results["times"]["end_io"] - results["times"]["start_io"]
    results["times"]["imgs"] = results["times"]["end_imgs"] - results["times"]["start_imgs"]
    results["times"]["conf"] = results["times"]["end_conf"] - results["times"]["start_conf"]

#     imgs_ctypes = np.ctypeslib.as_ctypes(imgs)
#     imgs_raw = sharedctypes.RawArray(imgs_ctypes._type_, imgs_ctypes)
#     confmaps_ctypes = np.ctypeslib.as_ctypes(confmaps)
#     confmaps_raw = sharedctypes.RawArray(confmaps_ctypes._type_, confmaps_ctypes)

    results["imgs"] = imgs#_raw
    results["confmaps"] = confmaps#_raw

    results["ready"] = True
    results["update"].set()

#     pub.send_string(jsonpickle.encode(dict(event="data_gen done",results=results)))
#     pub.send(umsgpack.packb(dict(event="data_gen done",results=results)))
#     pub.close()

    print("done with data_gen")


if __name__ == "__main__":

    server_address = "127.0.0.1"

    print(f"starting client to {server_address}")

    ctx = None
    ctx = zmq.Context()

    app = QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(True)

    training_window = TrainingDialog(zmq_context=ctx, server=server_address)
    training_window.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(training_window.check_messages)
    timer.start(0)

    app.exec_()
