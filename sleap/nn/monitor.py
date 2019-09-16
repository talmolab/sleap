from collections import deque
import numpy as np
from time import time, sleep
import zmq
import jsonpickle
import logging
logger = logging.getLogger(__name__)

from PySide2 import QtCore, QtWidgets, QtGui, QtCharts

class LossViewer(QtWidgets.QMainWindow):
    def __init__(self, zmq_context=None, show_controller=True, parent=None):
        super(LossViewer, self).__init__(parent)

        self.show_controller = show_controller
        self.stop_button = None

        self.reset()
        self.setup_zmq(zmq_context)

    def __del__(self):
        # close the zmq socket
        self.sub.unbind(self.sub.LAST_ENDPOINT)
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

    def reset(self, what=""):
        self.chart = QtCharts.QtCharts.QChart()

        self.series = dict()
        self.color = dict()

        self.series["batch"] = QtCharts.QtCharts.QScatterSeries()
        self.series["epoch_loss"] = QtCharts.QtCharts.QLineSeries()
        self.series["val_loss"] = QtCharts.QtCharts.QLineSeries()

        self.series["batch"].setName("Batch Training Loss")
        self.series["epoch_loss"].setName("Epoch Training Loss")
        self.series["val_loss"].setName("Epoch Validation Loss")

        self.color["batch"] = QtGui.QColor("blue")
        self.color["epoch_loss"] = QtGui.QColor("green")
        self.color["val_loss"] = QtGui.QColor("red")

        for s in self.series:
            self.series[s].pen().setColor(self.color[s])
        self.series["batch"].setMarkerSize(8.)

        self.chart.addSeries(self.series["batch"])
        self.chart.addSeries(self.series["epoch_loss"])
        self.chart.addSeries(self.series["val_loss"])

        # self.chart.createDefaultAxes()
        axisX = QtCharts.QtCharts.QValueAxis()
        axisX.setLabelFormat("%d")
        axisX.setTitleText("Batches")
        self.chart.addAxis(axisX, QtCore.Qt.AlignBottom)

        axisY = QtCharts.QtCharts.QLogValueAxis()
        axisY.setLabelFormat("%f")
        axisY.setLabelsVisible(True)
        axisY.setMinorTickCount(1)
        axisY.setTitleText("Loss")
        axisY.setBase(10)
        self.chart.addAxis(axisY, QtCore.Qt.AlignLeft)

        for series in self.chart.series():
            series.attachAxis(axisX)
            series.attachAxis(axisY)

        # self.chart.legend().hide()
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(QtCore.Qt.AlignTop)

        self.chartView = QtCharts.QtCharts.QChartView(self.chart)
        self.chartView.setRenderHint(QtGui.QPainter.Antialiasing)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chartView)

        if self.show_controller:
            self.stop_button = QtWidgets.QPushButton("Stop Training")
            self.stop_button.clicked.connect(self.stop)
            layout.addWidget(self.stop_button)

        wid = QtWidgets.QWidget()
        wid.setLayout(layout)
        self.setCentralWidget(wid)

        # Only show that last 2000 batch values
        self.X = deque(maxlen=2000)
        self.Y = deque(maxlen=2000)

        self.t0 = None
        self.current_job_output_type = what
        self.epoch = 0
        self.epoch_size = 1
        self.last_epoch_val_loss = None
        self.last_batch_number = 0
        self.is_running = False

    def setup_zmq(self, zmq_context):
        # Progress monitoring
        self.ctx_given = (zmq_context is not None)
        self.ctx = zmq.Context() if zmq_context is None else zmq_context
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.subscribe("")
        self.sub.bind("tcp://127.0.0.1:9001")
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
            logger.info("Sending command to stop training")
            self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="stop",)))
        if self.stop_button is not None:
            self.stop_button.setText("Stopping...")
            self.stop_button.setEnabled(False)


    def add_datapoint(self, x, y, which="batch"):

        # Keep track of all batch points
        if which == "batch":
            self.X.append(x)
            self.Y.append(y)

            # Redraw batch ever 40 points (faster than plotting each)
            if x % 40 == 0:
                xs, ys = self.X, self.Y
                points = [QtCore.QPointF(x, y) for x, y in zip(xs, ys) if y > 0]
                self.series["batch"].replace(points)

                # Set X scale to show all points
                dx = 0.5
                self.chart.axisX().setRange(min(self.X) - dx, max(self.X) + dx)

                # Set Y scale to exclude outliers
                q1, q3 = np.quantile(self.Y, (.25, .75))
                iqr = q3-q1 # interquartile range
                low = q1 - iqr * 1.5
                high = q3 + iqr * 1.5

                low = max(low, min(self.Y) - .2) # keep within range of data
                low = max(low, 1e-5) # for log scale, low cannot be 0

                high = min(high, max(self.Y) + .2)

                self.chart.axisY().setRange(low, high)
        else:
            self.series[which].append(x, y)

    def set_start_time(self, t0):
        self.t0 = t0
        self.is_running = True

    def set_end(self):
        self.is_running = False

    def update_runtime(self):
        if self.t0 is not None and self.is_running:
            dt = time() - self.t0
            dt_min, dt_sec = divmod(dt, 60)
            title = f"Training Epoch <b>{self.epoch}</b> / "
            title += f"Runtime: <b>{int(dt_min):02}:{int(dt_sec):02}</b>"
            if self.last_epoch_val_loss is not None:
                title += f"<br />Last Epoch Validation Loss: <b>{self.last_epoch_val_loss:.3e}</b>"
            self.chart.setTitle(title)

    def check_messages(self, timeout=10):
        if self.sub and self.sub.poll(timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.sub.recv_string())

            # logger.info(msg)

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
                        self.last_epoch_val_loss = msg["logs"]["val_loss"]
                        self.add_datapoint((self.epoch+1)*self.epoch_size, msg["logs"]["val_loss"], "val_loss")
                elif msg["event"] == "batch_end":
                    self.last_batch_number = msg["logs"]["batch"]
                    self.add_datapoint((self.epoch * self.epoch_size) + msg["logs"]["batch"], msg["logs"]["loss"])

        self.update_runtime()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = LossViewer()
    win.show()

    def test_point(x=[0]):
        x[0] += 1
        i = x[0]+1
        win.add_datapoint(i, i%30)

    t = QtCore.QTimer()
    t.timeout.connect(test_point)
    t.start(0)

    app.exec_()