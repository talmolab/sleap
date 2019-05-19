import numpy as np
from time import time, sleep
import zmq
import jsonpickle

from PySide2 import QtCore, QtWidgets, QtGui, QtCharts

class LossViewer(QtWidgets.QMainWindow):
    def __init__(self, zmq_context=None, parent=None):
        super(LossViewer, self).__init__(parent)

        self.chart = QtCharts.QtCharts.QChart()

        self.series = QtCharts.QtCharts.QScatterSeries()
        pen = self.series.pen()
        pen.setWidth(3)
        # pen.setColor(r,g,b,alpha)
        self.chart.addSeries(self.series)

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
        
        # Progress monitoring
        self.ctx = zmq.Context() if zmq_context is None else zmq_context
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.subscribe("")
        self.sub.connect("tcp://127.0.0.1:9001")

    def add_datapoint(self, x, y):
        self.series.append(x, y)

        self.X.append(x)
        self.Y.append(y)

        dx = 0.5
        dy = np.ptp(self.Y) * 0.02
        self.chart.axisX().setRange(min(self.X) - dx, max(self.X) + dx)
        self.chart.axisY().setRange(min(self.Y) - dy, max(self.Y) + dy)

    def set_start_time(self, t0):
        self.t0 = t0

    def update_runtime(self):
        if self.t0 is not None:
            dt = time() - t0
            dt_min, dt_sec = divmod(dt, 60)
            self.chart.setTitle(f"Runtime: {int(dt_min):02}:{int(dt_sec):02}")

    def check_messages(self, timeout=10):
        if sub.poll(timeout, zmq.POLLIN):
            msg = jsonpickle.decode(sub.recv_string())

            print(msg)
            if msg["event"] == "train_begin":
                self.set_start_time(time())
            elif msg["event"] == "epoch_begin":
                self.epoch = msg["epoch"]
            elif msg["event"] == "batch_end":
                self.add_datapoint((self.epoch * 100) + msg["logs"]["batch"], msg["logs"]["loss"])

        self.update_runtime()
