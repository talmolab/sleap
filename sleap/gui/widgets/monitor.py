"""GUI for monitoring training progress interactively."""

import numpy as np
from time import perf_counter
from sleap.nn.config.training_job import TrainingJobConfig
import zmq
import jsonpickle
import logging
from typing import Optional
from PySide2 import QtCore, QtWidgets, QtGui
from PySide2.QtCharts import QtCharts
import attr


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
        super().__init__(parent)

        self.show_controller = show_controller
        self.stop_button = None
        self.cancel_button = None
        self.canceled = False

        self.batches_to_show = -1  # -1 to show all
        self.ignore_outliers = False
        self.log_scale = True
        self.message_poll_time_ms = 20  # ms
        self.redraw_batch_time_ms = 500  # ms
        self.last_redraw_batch = None

        self.reset()
        self.setup_zmq(zmq_context)

    def __del__(self):
        self.unbind()

    def close(self):
        """Disconnect from ZMQ ports and close the window."""
        self.unbind()
        super().close()

    def unbind(self):
        """Disconnect from all ZMQ sockets."""
        if self.sub is not None:
            self.sub.unbind(self.sub.LAST_ENDPOINT)
            self.sub.close()
            self.sub = None

        if self.zmq_ctrl is not None:
            url = self.zmq_ctrl.LAST_ENDPOINT
            self.zmq_ctrl.unbind(url)
            self.zmq_ctrl.close()
            self.zmq_ctrl = None

        # If we started out own zmq context, terminate it.
        if not self.ctx_given and self.ctx is not None:
            self.ctx.term()
            self.ctx = None

    def reset(
        self,
        what: str = "",
        config: TrainingJobConfig = attr.ib(factory=TrainingJobConfig),
    ):
        """Reset all chart series.

        Args:
            what: String identifier indicating which job type the current run
                corresponds to.
        """
        self.chart = QtCharts.QChart()

        self.series = dict()

        COLOR_TRAIN = (18, 158, 220)
        COLOR_VAL = (248, 167, 52)
        COLOR_BEST_VAL = (151, 204, 89)

        self.series["batch"] = QtCharts.QScatterSeries()
        self.series["batch"].setName("Batch Training Loss")
        self.series["batch"].setColor(QtGui.QColor(*COLOR_TRAIN, 48))
        self.series["batch"].setMarkerSize(8.0)
        self.series["batch"].setBorderColor(QtGui.QColor(255, 255, 255, 25))
        self.chart.addSeries(self.series["batch"])

        self.series["epoch_loss"] = QtCharts.QLineSeries()
        self.series["epoch_loss"].setName("Epoch Training Loss")
        self.series["epoch_loss"].setColor(QtGui.QColor(*COLOR_TRAIN, 255))
        pen = self.series["epoch_loss"].pen()
        pen.setWidth(4)
        self.series["epoch_loss"].setPen(pen)
        self.chart.addSeries(self.series["epoch_loss"])

        self.series["epoch_loss_scatter"] = QtCharts.QScatterSeries()
        self.series["epoch_loss_scatter"].setColor(QtGui.QColor(*COLOR_TRAIN, 255))
        self.series["epoch_loss_scatter"].setMarkerSize(12.0)
        self.series["epoch_loss_scatter"].setBorderColor(
            QtGui.QColor(255, 255, 255, 25)
        )
        self.chart.addSeries(self.series["epoch_loss_scatter"])

        self.series["val_loss"] = QtCharts.QLineSeries()
        self.series["val_loss"].setName("Epoch Validation Loss")
        self.series["val_loss"].setColor(QtGui.QColor(*COLOR_VAL, 255))
        pen = self.series["val_loss"].pen()
        pen.setWidth(4)
        self.series["val_loss"].setPen(pen)
        self.chart.addSeries(self.series["val_loss"])

        self.series["val_loss_scatter"] = QtCharts.QScatterSeries()
        self.series["val_loss_scatter"].setColor(QtGui.QColor(*COLOR_VAL, 255))
        self.series["val_loss_scatter"].setMarkerSize(12.0)
        self.series["val_loss_scatter"].setBorderColor(QtGui.QColor(255, 255, 255, 25))
        self.chart.addSeries(self.series["val_loss_scatter"])

        self.series["val_loss_best"] = QtCharts.QScatterSeries()
        self.series["val_loss_best"].setName("Best Validation Loss")
        self.series["val_loss_best"].setColor(QtGui.QColor(*COLOR_BEST_VAL, 255))
        self.series["val_loss_best"].setMarkerSize(12.0)
        self.series["val_loss_best"].setBorderColor(QtGui.QColor(32, 32, 32, 25))
        self.chart.addSeries(self.series["val_loss_best"])

        axisX = QtCharts.QValueAxis()
        axisX.setLabelFormat("%d")
        axisX.setTitleText("Batches")
        self.chart.addAxis(axisX, QtCore.Qt.AlignBottom)

        # Create the different Y axes that can be used.
        self.axisY = dict()

        self.axisY["log"] = QtCharts.QLogValueAxis()
        self.axisY["log"].setBase(10)

        self.axisY["linear"] = QtCharts.QValueAxis()

        # Apply settings that apply to all Y axes.
        for axisY in self.axisY.values():
            axisY.setLabelFormat("%f")
            axisY.setLabelsVisible(True)
            axisY.setMinorTickCount(1)
            axisY.setTitleText("Loss")

        # Use the default Y axis.
        axisY = self.axisY["log"] if self.log_scale else self.axisY["linear"]

        # Add axes to chart and series.
        self.chart.addAxis(axisY, QtCore.Qt.AlignLeft)
        for series in self.chart.series():
            series.attachAxis(axisX)
            series.attachAxis(axisY)

        # Setup legend.
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(QtCore.Qt.AlignTop)
        self.chart.legend().setMarkerShape(QtCharts.QLegend.MarkerShapeCircle)

        # Hide scatters for epoch and val loss from legend.
        for s in ("epoch_loss_scatter", "val_loss_scatter"):
            self.chart.legend().markers(self.series[s])[0].setVisible(False)

        self.chartView = QtCharts.QChartView(self.chart)
        self.chartView.setRenderHint(QtGui.QPainter.Antialiasing)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.chartView)

        if self.show_controller:
            control_layout = QtWidgets.QHBoxLayout()

            field = QtWidgets.QCheckBox("Log Scale")
            field.setChecked(self.log_scale)
            field.stateChanged.connect(self.toggle_log_scale)
            control_layout.addWidget(field)

            field = QtWidgets.QCheckBox("Ignore Outliers")
            field.setChecked(self.ignore_outliers)
            field.stateChanged.connect(self.toggle_ignore_outliers)
            control_layout.addWidget(field)

            control_layout.addWidget(QtWidgets.QLabel("Batches to Show:"))

            # Add field for how many batches to show in chart.
            field = QtWidgets.QComboBox()
            self.batch_options = "200,1000,5000,All".split(",")
            for opt in self.batch_options:
                field.addItem(opt)
            cur_opt_str = (
                "All" if self.batches_to_show < 0 else str(self.batches_to_show)
            )
            if cur_opt_str in self.batch_options:
                field.setCurrentText(cur_opt_str)

            # Set connection action for when user selects another option.
            field.currentIndexChanged.connect(
                lambda x: self.set_batches_to_show(self.batch_options[x])
            )

            # Store field as property and add to layout.
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

        self.config = config
        self.X = []
        self.Y = []
        self.best_val_x = None
        self.best_val_y = None

        self.t0 = None
        self.mean_epoch_time_min = None
        self.mean_epoch_time_sec = None
        self.eta_ten_epochs_min = None

        self.current_job_output_type = what
        self.epoch = 0
        self.epoch_size = 1
        self.epochs_in_plateau = 0
        self.last_epoch_val_loss = None
        self.penultimate_epoch_val_loss = None
        self.epoch_in_plateau_flag = False
        self.last_batch_number = 0
        self.is_running = False

    def toggle_ignore_outliers(self):
        """Toggles whether to ignore outliers in chart scaling."""
        self.ignore_outliers = not self.ignore_outliers

    def toggle_log_scale(self):
        """Toggle whether to use log-scaled y-axis."""
        self.log_scale = not self.log_scale
        self.update_y_axis()

    def set_batches_to_show(self, batches: str):
        """Set the number of batches to show on the x-axis.

        Args:
            batches: Number of batches as a string. If numeric, this will be converted
                to an integer. If non-numeric string (e.g., "All"), then all batches
                will be shown.
        """
        if batches.isdigit():
            self.batches_to_show = int(batches)
        else:
            self.batches_to_show = -1

    def update_y_axis(self):
        """Update the y-axis when scale changes."""
        to = "log" if self.log_scale else "linear"

        # Remove other axes.
        for name, axisY in self.axisY.items():
            if name != to:
                if axisY in self.chart.axes():
                    self.chart.removeAxis(axisY)
                for series in self.chart.series():
                    if axisY in series.attachedAxes():
                        series.detachAxis(axisY)

        # Add axis.
        axisY = self.axisY[to]
        self.chart.addAxis(axisY, QtCore.Qt.AlignLeft)
        for series in self.chart.series():
            series.attachAxis(axisY)

    def setup_zmq(self, zmq_context: Optional[zmq.Context] = None):
        """Connect to ZMQ ports that listen to commands and updates.

        Args:
            zmq_context: The `zmq.Context` object to use for connections. A new one is
                created if not specified and will be closed when the monitor exits. If
                an existing one is provided, it will NOT be closed.
        """
        # Keep track of whether we're using an existing context (which we won't close
        # when done) or are creating our own (which we should close).
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

        # Set timer to poll for messages.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check_messages)
        self.timer.start(self.message_poll_time_ms)

    def cancel(self):
        """Set the cancel flag."""
        self.canceled = True
        if self.cancel_button is not None:
            self.cancel_button.setText("Canceling...")
            self.cancel_button.setEnabled(False)

    def stop(self):
        """Send command to stop training."""
        if self.zmq_ctrl is not None:
            # Send command to stop training.
            logger.info("Sending command to stop training.")
            self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="stop")))

        # Disable the button to prevent double messages.
        if self.stop_button is not None:
            self.stop_button.setText("Stopping...")
            self.stop_button.setEnabled(False)

    def add_datapoint(self, x: int, y: float, which: str):
        """Add a data point to graph.

        Args:
            x: The batch number (out of all epochs, not just current), or epoch.
            y: The loss value.
            which: Type of data point we're adding. Possible values are:
                * "batch" (loss for the batch)
                * "epoch_loss" (loss for the entire epoch)
                * "val_loss" (validation loss for the epoch)
        """
        if which == "batch":
            self.X.append(x)
            self.Y.append(y)

            # Redraw batch at intervals (faster than plotting every batch).
            draw_batch = False
            if self.last_redraw_batch is None:
                draw_batch = True
            else:
                dt = perf_counter() - self.last_redraw_batch
                draw_batch = (dt * 1000) >= self.redraw_batch_time_ms

            if draw_batch:
                self.last_redraw_batch = perf_counter()
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
                    low = max(low, 1e-8)  # for log scale, low cannot be 0

                self.chart.axisY().setRange(low, high)

        else:
            if which == "epoch_loss":
                self.series["epoch_loss"].append(x, y)
                self.series["epoch_loss_scatter"].append(x, y)
            elif which == "val_loss":
                self.series["val_loss"].append(x, y)
                self.series["val_loss_scatter"].append(x, y)
                if self.best_val_y is None or y < self.best_val_y:
                    self.best_val_x = x
                    self.best_val_y = y
                    self.series["val_loss_best"].replace([QtCore.QPointF(x, y)])

    def set_start_time(self, t0: float):
        """Mark the start flag and time of the run.

        Args:
            t0: Start time in seconds.
        """
        self.t0 = t0
        self.is_running = True

    def set_end(self):
        """Mark the end of the run."""
        self.is_running = False

    def update_runtime(self):
        """Update the title text with the current running time."""
        if self.is_timer_running:
            dt = perf_counter() - self.t0
            dt_min, dt_sec = divmod(dt, 60)
            title = f"Training Epoch <b>{self.epoch + 1}</b> / "
            title += f"Runtime: <b>{int(dt_min):02}:{int(dt_sec):02}</b>"
            if self.last_epoch_val_loss is not None:
                if self.penultimate_epoch_val_loss is not None:
                    title += (
                        f"<br />Mean Time per Epoch: "
                        f"<b>{int(self.mean_epoch_time_min):02}:{int(self.mean_epoch_time_sec):02}</b> / "
                        f"ETA Next 10 Epochs: <b>{int(self.eta_ten_epochs_min)} min</b>"
                    )
                    if self.epoch_in_plateau_flag:
                        title += (
                            f"<br />Epochs in Plateau: "
                            f"<b>{self.epochs_in_plateau} / "
                            f"{self.config.optimization.early_stopping.plateau_patience}</b>"
                        )
                title += (
                    f"<br />Last Epoch Validation Loss: "
                    f"<b>{self.last_epoch_val_loss:.3e}</b>"
                )
            if self.best_val_x is not None:
                best_epoch = (self.best_val_x // self.epoch_size) + 1
                title += (
                    f"<br />Best Epoch Validation Loss: "
                    f"<b>{self.best_val_y:.3e}</b> (epoch <b>{best_epoch}</b>)"
                )
            self.set_message(title)

    @property
    def is_timer_running(self) -> bool:
        """Return True if the timer has started."""
        return self.t0 is not None and self.is_running

    def set_message(self, text: str):
        """Set the chart title text."""
        self.chart.setTitle(text)

    def check_messages(
        self, timeout: int = 10, times_to_check: int = 10, do_update: bool = True
    ):
        """Poll for ZMQ messages and adds any received data to graph.

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

        Args:
            timeout: Message polling timeout in milliseconds. This is how often we will
                check for new command messages.
            times_to_check: How many times to check for new messages in the queue before
                going back to polling with a timeout. Helps to clear backlogs of
                messages if necessary.
            do_update: If True (the default), update the GUI text.
        """
        if self.sub and self.sub.poll(timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.sub.recv_string())

            if msg["event"] == "train_begin":
                self.set_start_time(perf_counter())
                self.current_job_output_type = msg["what"]

            # Make sure message matches current training job.
            if msg.get("what", "") == self.current_job_output_type:

                if not self.is_timer_running:
                    # We must have missed the train_begin message, so start timer now.
                    self.set_start_time(perf_counter())

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
                        # update variables and add points to plot
                        self.penultimate_epoch_val_loss = self.last_epoch_val_loss
                        self.last_epoch_val_loss = msg["logs"]["val_loss"]
                        self.add_datapoint(
                            (self.epoch + 1) * self.epoch_size,
                            msg["logs"]["val_loss"],
                            "val_loss",
                        )
                        # calculate timing and flags at new epoch
                        if self.penultimate_epoch_val_loss is not None:
                            mean_epoch_time = (perf_counter() - self.t0) / (
                                self.epoch + 1
                            )
                            self.mean_epoch_time_min, self.mean_epoch_time_sec = divmod(
                                mean_epoch_time, 60
                            )
                            self.eta_ten_epochs_min = (mean_epoch_time * 10) // 60

                            val_loss_delta = (
                                self.penultimate_epoch_val_loss
                                - self.last_epoch_val_loss
                            )
                            self.epoch_in_plateau_flag = (
                                val_loss_delta
                                < self.config.optimization.early_stopping.plateau_min_delta
                            ) or (self.best_val_y < self.last_epoch_val_loss)
                            self.epochs_in_plateau = (
                                self.epochs_in_plateau + 1
                                if self.epoch_in_plateau_flag
                                else 0
                            )
                    self.on_epoch.emit()
                elif msg["event"] == "batch_end":
                    self.last_batch_number = msg["batch"]
                    self.add_datapoint(
                        (self.epoch * self.epoch_size) + msg["batch"],
                        msg["logs"]["loss"],
                        "batch",
                    )

            # Check for messages again (up to times_to_check times).
            if times_to_check > 0:
                self.check_messages(
                    timeout=timeout, times_to_check=times_to_check - 1, do_update=False
                )

        if do_update:
            self.update_runtime()
