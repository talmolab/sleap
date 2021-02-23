"""GUI for monitoring training progress interactively."""

from collections import deque
import numpy as np
from time import time, sleep
import zmq
import jsonpickle
import logging
from typing import Optional

from PySide2 import QtCore, QtWidgets, QtGui, QtCharts

logger = logging.getLogger(__name__)


class LossViewer(QtWidgets.QMainWindow):
    """Qt window for showing in-progress training metrics sent over ZMQ."""

    on_epoch = QtCore.Signal()

    def __init__(
        self,
        zmq_context: Optional[zmq.Context] = None,
        show_controller=True,
        parent=None,
    ):
        super(LossViewer, self).__init__(parent)

        self.show_controller = show_controller
        self.stop_button = None
        self.cancel_button = None
        self.canceled = False

        self.redraw_batch_interval = 40
        self.batches_to_show = -1  # -1 to show all
        self.ignore_outliers = False
        self.log_scale = True

        self.reset()
        self.setup_zmq(zmq_context)

    def __del__(self):
        self.unbind()

    def close(self):
        self.unbind()
        super(LossViewer, self).close()

    def unbind(self):
        # close the zmq socket
        if self.sub is not None:
            self.sub.unbind(self.sub.LAST_ENDPOINT)
            self.sub.close()
            self.sub = None
        if self.zmq_ctrl is not None:
            url = self.zmq_ctrl.LAST_ENDPOINT
            self.zmq_ctrl.unbind(url)
            self.zmq_ctrl.close()
            self.zmq_ctrl = None
        # if we started out own zmq context, terminate it
        if not self.ctx_given and self.ctx is not None:
            self.ctx.term()
            self.ctx = None

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
        self.series["batch"].setMarkerSize(8.0)
        self.series["batch"].setBorderColor(QtGui.QColor(255, 255, 255, 25))

        self.chart.addSeries(self.series["batch"])
        self.chart.addSeries(self.series["epoch_loss"])
        self.chart.addSeries(self.series["val_loss"])

        axisX = QtCharts.QtCharts.QValueAxis()
        axisX.setLabelFormat("%d")
        axisX.setTitleText("Batches")
        self.chart.addAxis(axisX, QtCore.Qt.AlignBottom)

        # create the different Y axes that can be used
        self.axisY = dict()

        self.axisY["log"] = QtCharts.QtCharts.QLogValueAxis()
        self.axisY["log"].setBase(10)

        self.axisY["linear"] = QtCharts.QtCharts.QValueAxis()

        # settings that apply to all Y axes
        for axisY in self.axisY.values():
            axisY.setLabelFormat("%f")
            axisY.setLabelsVisible(True)
            axisY.setMinorTickCount(1)
            axisY.setTitleText("Loss")

        # use the default Y axis
        axisY = self.axisY["log"] if self.log_scale else self.axisY["linear"]

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
            control_layout = QtWidgets.QHBoxLayout()

            field = QtWidgets.QCheckBox("Log Scale")
            field.setChecked(self.log_scale)
            field.stateChanged.connect(lambda x: self.toggle("log_scale"))
            control_layout.addWidget(field)

            field = QtWidgets.QCheckBox("Ignore Outliers")
            field.setChecked(self.ignore_outliers)
            field.stateChanged.connect(lambda x: self.toggle("ignore_outliers"))
            control_layout.addWidget(field)

            control_layout.addWidget(QtWidgets.QLabel("Batches to Show:"))

            # add field for how many batches to show in chart
            field = QtWidgets.QComboBox()
            # add options
            self.batch_options = "200,1000,5000,All".split(",")
            for opt in self.batch_options:
                field.addItem(opt)
            # set field to currently set value
            cur_opt_str = (
                "All" if self.batches_to_show < 0 else str(self.batches_to_show)
            )
            if cur_opt_str in self.batch_options:
                field.setCurrentText(cur_opt_str)
            # connection action for when user selects another option
            field.currentIndexChanged.connect(
                lambda x: self.set_batches_to_show(self.batch_options[x])
            )
            # store field as property and add to layout
            self.batches_to_show_field = field
            control_layout.addWidget(self.batches_to_show_field)

            control_layout.addStretch(1)

            self.stop_button = QtWidgets.QPushButton("Stop Early")
            self.stop_button.clicked.connect(self.stop)
            control_layout.addWidget(self.stop_button)
            self.cancel_button = QtWidgets.QPushButton("Cancel Training")
            self.cancel_button.clicked.connect(self.cancel)
            control_layout.addWidget(self.cancel_button)

            widget = QtWidgets.QWidget()
            widget.setLayout(control_layout)
            layout.addWidget(widget)

        wid = QtWidgets.QWidget()
        wid.setLayout(layout)
        self.setCentralWidget(wid)

        self.X = []
        self.Y = []

        self.t0 = None
        self.current_job_output_type = what
        self.epoch = 0
        self.epoch_size = 1
        self.last_epoch_val_loss = None
        self.last_batch_number = 0
        self.is_running = False

    def toggle(self, what):
        if what == "log_scale":
            self.log_scale = not self.log_scale
            self.update_y_axis()
        elif what == "ignore_outliers":
            self.ignore_outliers = not self.ignore_outliers
        elif what == "entire_history":
            if self.batches_to_show > 0:
                self.batches_to_show = -1
            else:
                self.batches_to_show = 200

    def set_batches_to_show(self, val):
        if val.isdigit():
            self.batches_to_show = int(val)
        else:
            self.batches_to_show = -1

    def update_y_axis(self):
        to = "log" if self.log_scale else "linear"
        # remove other axes
        for name, axisY in self.axisY.items():
            if name != to:
                if axisY in self.chart.axes():
                    self.chart.removeAxis(axisY)
                for series in self.chart.series():
                    if axisY in series.attachedAxes():
                        series.detachAxis(axisY)
        # add axis
        axisY = self.axisY[to]
        self.chart.addAxis(axisY, QtCore.Qt.AlignLeft)
        for series in self.chart.series():
            series.attachAxis(axisY)

    def setup_zmq(self, zmq_context: Optional[zmq.Context]):

        # Keep track of whether we're using an existing context (which we won't
        # close when done) or are creating our own (which we should close).
        self.ctx_given = zmq_context is not None
        self.ctx = zmq.Context() if zmq_context is None else zmq_context

        # Progress monitoring, SUBSCRIBER
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.subscribe("")
        self.sub.bind("tcp://127.0.0.1:9001")

        # Controller, PUBLISHER
        self.zmq_ctrl = None
        if self.show_controller:
            self.zmq_ctrl = self.ctx.socket(zmq.PUB)
            self.zmq_ctrl.bind("tcp://127.0.0.1:9000")

        # Set timer to poll for messages every 20 milliseconds
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_messages)
        self.timer.start(20)

    def cancel(self):
        """Set the cancel flag."""
        self.canceled = True
        if self.cancel_button is not None:
            self.cancel_button.setText("Canceling...")
            self.cancel_button.setEnabled(False)

    def stop(self):
        """Action to stop training."""

        if self.zmq_ctrl is not None:
            # send command to stop training
            logger.info("Sending command to stop training.")
            self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="stop")))

        # Disable the button
        if self.stop_button is not None:
            self.stop_button.setText("Stopping...")
            self.stop_button.setEnabled(False)

    def add_datapoint(self, x, y, which="batch"):
        """
        Adds data point to graph.

        Args:
            x: typically the batch number (out of all epochs, not just current)
            y: typically the loss value
            which: type of data point we're adding, possible values are
                * batch (loss for batch)
                * epoch_loss (loss for entire epoch)
                * val_loss (validation loss for for epoch)
        """

        # Keep track of all batch points
        if which == "batch":
            self.X.append(x)
            self.Y.append(y)

            # Redraw batch at intervals (faster than plotting each)
            if x % self.redraw_batch_interval == 0:

                if self.batches_to_show < 0 or len(self.X) < self.batches_to_show:
                    xs, ys = self.X, self.Y
                else:
                    xs, ys = (
                        self.X[-self.batches_to_show :],
                        self.Y[-self.batches_to_show :],
                    )

                points = [QtCore.QPointF(x, y) for x, y in zip(xs, ys) if y > 0]
                self.series["batch"].replace(points)

                # Set X scale to show all points
                dx = 0.5
                self.chart.axisX().setRange(min(xs) - dx, max(xs) + dx)

                if self.ignore_outliers:
                    dy = np.ptp(ys) * 0.02
                    # Set Y scale to exclude outliers
                    q1, q3 = np.quantile(ys, (0.25, 0.75))
                    iqr = q3 - q1  # interquartile range
                    low = q1 - iqr * 1.5
                    high = q3 + iqr * 1.5

                    low = max(low, min(ys) - dy)  # keep within range of data
                    high = min(high, max(ys) + dy)
                else:
                    # Set Y scale to show all points
                    dy = np.ptp(ys) * 0.02
                    low = min(ys) - dy
                    high = max(ys) + dy

                if self.log_scale:
                    low = max(low, 1e-5)  # for log scale, low cannot be 0

                self.chart.axisY().setRange(low, high)

        else:
            self.series[which].append(x, y)

    def set_start_time(self, t0):
        self.t0 = t0
        self.is_running = True

    def set_end(self):
        self.is_running = False

    def update_runtime(self):
        if self.is_timer_running():
            dt = time() - self.t0
            dt_min, dt_sec = divmod(dt, 60)
            title = f"Training Epoch <b>{self.epoch+1}</b> / "
            title += f"Runtime: <b>{int(dt_min):02}:{int(dt_sec):02}</b>"
            if self.last_epoch_val_loss is not None:
                title += f"<br />Last Epoch Validation Loss: <b>{self.last_epoch_val_loss:.3e}</b>"
            self.set_message(title)

    def is_timer_running(self):
        return self.t0 is not None and self.is_running

    def set_message(self, text):
        self.chart.setTitle(text)

    def check_messages(
        self, timeout=10, times_to_check: int = 10, do_update: bool = True
    ):
        """
        Polls for ZMQ messages and adds any received data to graph.

        The message is a dictionary encoded as JSON:
            * event - options include
                * train_begin
                * train_end
                * epoch_begin
                * epoch_end
                * batch_end
            * what - this should match the type of model we're training and
                ensures that we ignore old messages when we start monitoring
                a new training session (when we're training multiple types
                of models in a sequence, as for the top-down pipeline).
            * logs - dictionary with data relevant for plotting, can include
                * loss
                * val_loss

        """
        if self.sub and self.sub.poll(timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.sub.recv_string())

            # logger.info(msg)

            if msg["event"] == "train_begin":
                self.set_start_time(time())
                self.current_job_output_type = msg["what"]

            # make sure message matches current training job
            if msg.get("what", "") == self.current_job_output_type:

                if not self.is_timer_running():
                    # We must have missed the train_begin message, so start timer now
                    self.set_start_time(time())

                if msg["event"] == "train_end":
                    self.set_end()
                elif msg["event"] == "epoch_begin":
                    self.epoch = msg["epoch"]
                elif msg["event"] == "epoch_end":
                    self.epoch_size = max(self.epoch_size, self.last_batch_number + 1)
                    self.add_datapoint(
                        (self.epoch + 1) * self.epoch_size,
                        msg["logs"]["loss"],
                        "epoch_loss",
                    )
                    if "val_loss" in msg["logs"].keys():
                        self.last_epoch_val_loss = msg["logs"]["val_loss"]
                        self.add_datapoint(
                            (self.epoch + 1) * self.epoch_size,
                            msg["logs"]["val_loss"],
                            "val_loss",
                        )
                    self.on_epoch.emit()
                elif msg["event"] == "batch_end":
                    self.last_batch_number = msg["batch"]
                    self.add_datapoint(
                        (self.epoch * self.epoch_size) + msg["batch"],
                        msg["logs"]["loss"],
                    )

            # Check for messages again (up to times_to_check times)
            if times_to_check:
                self.check_messages(
                    timeout=timeout, times_to_check=times_to_check - 1, do_update=False
                )

        if do_update:
            self.update_runtime()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = LossViewer()
    win.show()

    def test_point(x=[0]):
        x[0] += 1
        i = x[0] + 1
        win.add_datapoint(i, i % 30 + 1)

    t = QtCore.QTimer()
    t.timeout.connect(test_point)
    t.start(20)

    win.set_message("Waiting for 3 seconds...")
    t2 = QtCore.QTimer()
    t2.timeout.connect(lambda: win.set_message("Running demo..."))
    t2.start(3000)

    app.exec_()
