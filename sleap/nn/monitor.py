import numpy as np
from time import time, sleep
import zmq
import jsonpickle

from PySide2 import QtCore, QtWidgets, QtGui, QtCharts

class LossViewer(QtWidgets.QMainWindow):
    def __init__(self, zmq_context=None, show_controller=True, parent=None):
        super(LossViewer, self).__init__(parent)

        self.show_controller = show_controller

        self.reset()
        self.setup_zmq(zmq_context)

    def __del__(self):
        # close the zmq socket
        self.sub.close()
        self.sub = None
        if self.zmq_ctrl is not None:
            url = self.zmq_ctrl.LAST_ENDPOINT
            self.zmq_ctrl.unbind(url)
            self.zmq_ctrl.close()
            self.zmq_ctrl = None
        # if we started out own zmq context, terminate it
        if not self.ctx_given:
            self.ctx.term()

    def reset(self):
        self.chart = QtCharts.QtCharts.QChart()

        self.series = dict()
        self.color = dict()

        self.series["batch"] = QtCharts.QtCharts.QScatterSeries()
        self.series["epoch_loss"] = QtCharts.QtCharts.QLineSeries()
        self.series["val_loss"] = QtCharts.QtCharts.QLineSeries()


        self.color["batch"] = QtGui.QColor("blue")
        self.color["epoch_loss"] = QtGui.QColor("green")
        self.color["val_loss"] = QtGui.QColor("red")

        for s in self.series:
            self.series[s].pen().setColor(self.color[s])
        self.series["batch"].setMarkerSize(8.)

        self.chart.addSeries(self.series["batch"])
        self.chart.addSeries(self.series["epoch_loss"])
        self.chart.addSeries(self.series["val_loss"])

        self.chart.createDefaultAxes()
        self.chart.axisX().setLabelFormat("%d")
        self.chart.axisX().setTitleText("Batches")
        self.chart.axisY().setTitleText("Loss")

        self.chart.legend().hide()

        self.chartView = QtCharts.QtCharts.QChartView(self.chart)
        self.chartView.setRenderHint(QtGui.QPainter.Antialiasing)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chartView)

        if self.show_controller:
            btn = QtWidgets.QPushButton("Stop Training")
            btn.clicked.connect(self.stop)
            layout.addWidget(btn)

        wid = QtWidgets.QWidget()
        wid.setLayout(layout)
        self.setCentralWidget(wid)

        self.X = []
        self.Y = []
        self.t0 = None
        self.current_job_output_type = ""
        self.epoch = 0
        self.epoch_size = 1
        self.last_batch_number = 0
        self.is_running = False

    def setup_zmq(self, zmq_context):
        # Progress monitoring
        self.ctx_given = (zmq_context is not None)
        self.ctx = zmq.Context() if zmq_context is None else zmq_context
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.subscribe("")
        self.sub.connect("tcp://127.0.0.1:9001")
        # Controller
        self.zmq_ctrl = None
        if self.show_controller:
            self.zmq_ctrl = self.ctx.socket(zmq.PUB)
            self.zmq_ctrl.bind("tcp://127.0.0.1:9000")
        # Set timer to poll for messages
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_messages)
        self.timer.start(0)

    def stop(self):
        if self.zmq_ctrl is not None:
            # send command to stop training
            print("Sending command to stop training")
            self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="stop",)))

    def add_datapoint(self, x, y, which="batch"):
        self.series[which].append(x, y)

        self.X.append(x)
        self.Y.append(y)

        dx = 0.5
        dy = np.ptp(self.Y) * 0.02
        self.chart.axisX().setRange(min(self.X) - dx, max(self.X) + dx)
        self.chart.axisY().setRange(min(self.Y) - dy, max(self.Y) + dy)

    def set_start_time(self, t0):
        self.t0 = t0
        self.is_running = True

    def set_end(self):
        self.is_running = False
        self.timer.stop()

    def update_runtime(self):
        if self.t0 is not None and self.is_running:
            dt = time() - self.t0
            dt_min, dt_sec = divmod(dt, 60)
            self.chart.setTitle(f"Runtime: {int(dt_min):02}:{int(dt_sec):02}")

    def check_messages(self, timeout=10):
        if self.sub and self.sub.poll(timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.sub.recv_string())

            # print(msg)
            if msg["event"] == "train_begin":
                self.set_start_time(time())
                self.current_job_output_type = msg["what"]

            # make sure message matches current training job
            if msg.get("what", "") == self.current_job_output_type:
                if msg["event"] == "train_end":
                    self.set_end()
                elif msg["event"] == "epoch_begin":
                    self.epoch = msg["epoch"]
                elif msg["event"] == "epoch_end":
                    self.epoch_size = max(self.epoch_size, self.last_batch_number + 1)
                    self.add_datapoint((self.epoch+1)*self.epoch_size, msg["logs"]["loss"], "epoch_loss")
                    if "val_loss" in msg["logs"].keys():
                        self.add_datapoint((self.epoch+1)*self.epoch_size, msg["logs"]["val_loss"], "val_loss")
                elif msg["event"] == "batch_end":
                    self.last_batch_number = msg["logs"]["batch"]
                    self.add_datapoint((self.epoch * self.epoch_size) + msg["logs"]["batch"], msg["logs"]["loss"])
            else:
                pass
                # debug print the message that doesn't match our currently running job
                # print(f"{self.current_job_output_type} != {msg}")

        self.update_runtime()
