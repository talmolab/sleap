import multiprocessing as mp
import threading
import zmq
import jsonpickle
import umsgpack

from PySide2 import QtCore, QtWidgets

class TrainingDialog(QtWidgets.QWidget):

    def __init__(self, zmq_context, server="127.0.0.1", *args, **kwargs):
        super(TrainingDialog, self).__init__(*args, **kwargs)

        self.zmq_context = zmq_context

        # Controller
        self.zmq_ctrl = self.zmq_context.socket(zmq.PUB)
        self.zmq_ctrl.connect(f"tcp://{server}:9000")

        # Data

        self.labels_file = "tests/data/json_format_v1/centered_pair.json"
        self.default_scale = 1
        self.is_training = False
        self.training_data = dict(ready=False, times=None)

        # UI Widgets

        self.labels_status = QtWidgets.QLabel()

        self.fetch_labels_button = QtWidgets.QPushButton("Load Labels")
        self.fetch_labels_button.clicked.connect(self.fetchLabelsFile)
        self.fetch_labels_button.clicked.connect(self.refresh)

        self.scale_label = QtWidgets.QLabel()
        self.scale_label.setText("Scale:")
        self.scale = QtWidgets.QSpinBox()
        self.scale.setValue(self.default_scale)
#         self.scale.setRange(1, 3)
        self.data_gen_button = QtWidgets.QPushButton("Generate Data")
        self.data_gen_button.clicked.connect(self.generateData)
        self.data_gen_button.clicked.connect(self.refresh)
        self.debug_button = QtWidgets.QPushButton("Debug")
        self.debug_button.clicked.connect(self.debug)
        
        self.training_button = QtWidgets.QPushButton("Start Training")
        self.training_button.clicked.connect(self.runTraining)

        # UI Group Widgets
        
        gb = QtWidgets.QGroupBox("Training Labels")
        vb = QtWidgets.QVBoxLayout()
        vb.addWidget(self.labels_status)
        vb.addWidget(self.fetch_labels_button)
        gb.setLayout(vb)
        self.labels_group = gb

        gb = QtWidgets.QGroupBox("Data Generation")
        vb = QtWidgets.QVBoxLayout()
        vb.addWidget(self.scale_label)
        vb.addWidget(self.scale)
        vb.addWidget(self.data_gen_button)
        vb.addWidget(self.debug_button)
        gb.setLayout(vb)
        self.gen_group = gb

        gb = QtWidgets.QGroupBox("Training")
        vb = QtWidgets.QVBoxLayout()
        vb.addWidget(self.training_button)
        gb.setLayout(vb)
        self.training_group = gb

        # UI Layout

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.labels_group)
        self.layout.addWidget(self.gen_group)
        self.layout.addWidget(self.training_group)

        self.setLayout(self.layout)

        self.refresh()

    def refresh(self, *args):
        print("refresh called")

        has_labels = self.labels_file != ""
        has_data = self.training_data["ready"]
        print(f"has_data: {has_data}")
        labels_status = self.labels_file if has_labels else "[no labels loaded]"
        self.labels_status.setText(labels_status)

        self.data_gen_button.setEnabled(has_labels)
        
        training_button_text = "Stop Training" if self.is_training else "Start Training"
#         training_button_text = str(has_data)
        self.training_button.setText(training_button_text)
        self.training_button.setEnabled(has_data)

    def fetchLabelsFile(self, *args):
        filters = ["JSON Labels (*.json)"]
        filename, selected_filter = QtWidgets.QFileDialog.getOpenFileName(None, directory=None, caption="Open File", filter=";;".join(filters))
    
        if len(filename):
            self.labels_file = filename
            self.training_data = dict(ready=False)

    def generateData(self, *args):
#         mp.Process(target=data_gen, args=(self.labels_file,)).start()
        cmd = dict(command="data_gen", labels_file=self.labels_file)
        self.zmq_ctrl.send_string(jsonpickle.encode(cmd))

    def runTraining(self, *args):
        self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="train",)))
        print("sent train command")

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
        app.processEvents()

        def plot_confmaps(parent, item_idx):
            frame_conf_map = ConfMapsPlot(confmaps[parent.frame_idx,...])
            conf_window.view.scene.addItem(frame_conf_map)

        conf_window.changedPlot.connect(plot_confmaps)
        conf_window.plot()

    def debug(self, *args):
        if "imgs" in self.training_data:
            print(self.training_data["imgs"].shape)
        print(self.training_data["times"])
        self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="debug",)))
        print("sent zmq message")
#         print([t.getName() for t in threading.enumerate()])

if __name__ == "__main__":

    
#     server_address = "127.0.0.1"
#     server_address = "10.9.111.77" # me (temp)
    server_address = "128.112.217.175" # talmo

    print(f"starting client to {server_address}")

    ctx = zmq.Context()

    app = QtWidgets.QApplication([])
    window = TrainingDialog(zmq_context=ctx, server=server_address)
    window.show()

    app.setQuitOnLastWindowClosed(True)
    app.processEvents()

    # Result monitoring
    sub = ctx.socket(zmq.SUB)
    sub.subscribe("")
    sub.connect(f"tcp://{server_address}:9001")

    def poll(timeout=10):
        if sub.poll(timeout, zmq.POLLIN):
#             return umsgpack.unpackb(sub.recv())
            return jsonpickle.decode(sub.recv_string())
        return None

    epoch = 0
    while True:
        msg = poll()
        if msg is not None:
            print(f"msg event: {msg['event']}")
            if msg["event"] == "data_gen done":
                for key in msg["results"].keys():
                    window.training_data[key] = msg["results"][key]
                # window.preview()
                pass
            window.refresh()

        app.processEvents()

    # Stop training
    ctrl.send_string(jsonpickle.encode(dict(command="stop")))

    while True:
        app.processEvents()
