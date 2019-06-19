import numpy as np
from time import time, sleep
import zmq
import jsonpickle

from PySide2 import QtCore, QtWidgets, QtGui, QtCharts

class LossViewer(QtWidgets.QMainWindow):
    def __init__(self, zmq_context=None, parent=None):
        super(LossViewer, self).__init__(parent)

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
        self.chartView.setRenderHint(QtGui.QPainter.Antialiasing);
        self.setCentralWidget(self.chartView)

        self.X = []
        self.Y = []
        self.t0 = None
        self.epoch = 0
        self.epoch_size = 1
        self.last_batch_number = 0
        self.is_running = False
        
        # Progress monitoring
        self.ctx_given = (zmq_context is not None)
        self.ctx = zmq.Context() if zmq_context is None else zmq_context
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.subscribe("")
        self.sub.connect("tcp://127.0.0.1:9001")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_messages)
        self.timer.start(0)

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
        # close the zmq socket
        self.sub.close()
        self.sub = None
        # if we started out own zmq context, terminate it
        if not self.ctx_given:
            self.ctx.term()

    def update_runtime(self):
        if self.t0 is not None and self.is_running:
            dt = time() - self.t0
            dt_min, dt_sec = divmod(dt, 60)
            self.chart.setTitle(f"Runtime: {int(dt_min):02}:{int(dt_sec):02}")

    def check_messages(self, timeout=10):
        if self.sub and self.sub.poll(timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.sub.recv_string())

            print(msg)
            if msg["event"] == "train_begin":
                self.set_start_time(time())
            if msg["event"] == "train_end":
                self.set_end()
            elif msg["event"] == "epoch_begin":
                self.epoch = msg["epoch"]
            elif msg["event"] == "epoch_end":
                self.epoch_size = max(self.epoch_size, self.last_batch_number + 1)
                self.add_datapoint((self.epoch+1)*self.epoch_size, msg["logs"]["loss"], "epoch_loss")
                self.add_datapoint((self.epoch+1)*self.epoch_size, msg["logs"]["val_loss"], "val_loss")
            elif msg["event"] == "batch_end":
                self.last_batch_number = msg["logs"]["batch"]
                self.add_datapoint((self.epoch * self.epoch_size) + msg["logs"]["batch"], msg["logs"]["loss"])

        self.update_runtime()
