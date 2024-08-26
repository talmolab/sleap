"""GUI for monitoring training progress interactively."""

import logging
from time import perf_counter
from typing import Dict, Optional, Tuple

import attr
import jsonpickle
import numpy as np
import zmq
from matplotlib.collections import PathCollection
import matplotlib.transforms as mtransforms
from qtpy import QtCore, QtWidgets

from sleap.gui.utils import is_port_free, select_zmq_port
from sleap.gui.widgets.mpl import MplCanvas
from sleap.nn.config.training_job import TrainingJobConfig

logger = logging.getLogger(__name__)


class LossPlot(MplCanvas):
    """Matplotlib canvas for diplaying training and validation loss curves."""

    def __init__(
        self,
        width: int = 5,
        height: int = 4,
        dpi: int = 100,
        log_scale: bool = True,
        ignore_outliers: bool = False,
    ):
        super().__init__(width=width, height=height, dpi=dpi)

        self._log_scale: bool = log_scale

        self.ignore_outliers = ignore_outliers

        # Initialize the series for the plot
        self.series: dict = {}
        COLOR_TRAIN = (18, 158, 220)
        COLOR_VAL = (248, 167, 52)
        COLOR_BEST_VAL = (151, 204, 89)

        # Initialize scatter series for batch training loss
        self.series["batch"] = self._init_series(
            series_type=self.axes.scatter,
            name="Batch Training Loss",
            color=COLOR_TRAIN + (48,),
            border_color=(255, 255, 255, 25),
        )

        # Initialize line series for epoch training loss
        self.series["epoch_loss"] = self._init_series(
            series_type=self.axes.plot,
            name="Epoch Training Loss",
            color=COLOR_TRAIN + (255,),
            line_width=3.0,
        )

        # Initialize line series for epoch validation loss
        self.series["val_loss"] = self._init_series(
            series_type=self.axes.plot,
            name="Epoch Validation Loss",
            color=COLOR_VAL + (255,),
            line_width=3.0,
            zorder=4,  # Below best validation loss series
        )

        # Initialize scatter series for best epoch validation loss
        self.series["val_loss_best"] = self._init_series(
            series_type=self.axes.scatter,
            name="Best Validation Loss",
            color=COLOR_BEST_VAL + (255,),
            border_color=(255, 255, 255, 25),
            zorder=5,  # Above epoch validation loss series
        )

        # Set the x and y positions for the xy labels (as fraction of figure size)
        self.ypos_xlabel = 0.1
        self.xpos_ylabel = 0.05

        # Padding between the axes and the xy labels
        self.xpos_padding = 0.2
        self.ypos_padding = 0.1

        # Set up the major gridlines
        self._setup_major_gridlines()

        # Set up the x-axis
        self._setup_x_axis()

        # Set up the y-axis
        self._set_up_y_axis()

        # Set up the legend
        self.legend_width, legend_height = self._setup_legend()

        # Set up the title space
        self.ypos_title = None
        title_height = self._set_title_space()
        self.ypos_title = 1 - title_height - self.ypos_padding

        # Determine the top height of the plot
        top_height = max(title_height, legend_height)

        # Adjust the figure layout
        self.xpos_left_plot = self.xpos_ylabel + self.xpos_padding
        self.xpos_right_plot = 0.97
        self.ypos_bottom_plot = self.ypos_xlabel + self.ypos_padding
        self.ypos_top_plot = 1 - top_height - self.ypos_padding

        # Adjust the top parameters as needed
        self.fig.subplots_adjust(
            left=self.xpos_left_plot,
            right=self.xpos_right_plot,
            top=self.ypos_top_plot,
            bottom=self.ypos_bottom_plot,
        )

    @property
    def log_scale(self):
        """Returns True if the plot has a log scale for y-axis."""

        return self._log_scale

    @log_scale.setter
    def log_scale(self, val):
        """Sets the scale of the y axis to log if True else linear."""

        if isinstance(val, bool):
            self._log_scale = val

        y_scale = "log" if self._log_scale else "linear"
        self.axes.set_yscale(y_scale)
        self.redraw_plot()

    def set_data_on_scatter(self, xs, ys, which):
        """Set data on a scatter plot.

        Not to be used with line plots.

        Args:
            xs: The x-coordinates of the data points.
            ys: The y-coordinates of the data points.
            which: The type of data point. Possible values are:
                * "batch"
                * "val_loss_best"
        """

        offsets = np.column_stack((xs, ys))
        self.series[which].set_offsets(offsets)

    def add_data_to_plot(self, x, y, which):
        """Add data to a line plot.

        Not to be used with scatter plots.

        Args:
            x: The x-coordinate of the data point.
            y: The y-coordinate of the data point.
            which: The type of data point. Possible values are:
                * "epoch_loss"
                * "val_loss"
        """

        x_data, y_data = self.series[which].get_data()
        self.series[which].set_data(np.append(x_data, x), np.append(y_data, y))

    def resize_axes(self, x, y):
        """Resize axes to fit data.

        This is only called when plotting batches.

        Args:
            x: The x-coordinates of the data points.
            y: The y-coordinates of the data points.
        """

        # Set X scale to show all points
        x_min, x_max = self._calculate_xlim(x)
        self.axes.set_xlim(x_min, x_max)

        # Set Y scale, ensuring that y_min and y_max do not lead to sngular transform
        y_min, y_max = self._calculate_ylim(y)
        y_min, y_max = self.axes.yaxis.get_major_locator().nonsingular(y_min, y_max)
        self.axes.set_ylim(y_min, y_max)

        # Add gridlines at midpoint between major ticks (major gridlines are automatic)
        self._add_midpoint_gridlines()

        # Redraw the plot
        self.redraw_plot()

    def redraw_plot(self):
        """Redraw the plot."""

        self.fig.canvas.draw_idle()

    def set_title(self, title, color=None):
        """Set the title of the plot.

        Args:
            title: The title text to display.
        """

        if color is None:
            color = "black"

        self.axes.set_title(
            title, fontweight="light", fontsize="small", color=color, x=0.55, y=1.03
        )

    def update_runtime_title(
        self,
        epoch: int,
        dt_min: int,
        dt_sec: int,
        last_epoch_val_loss: float = None,
        penultimate_epoch_val_loss: float = None,
        mean_epoch_time_min: int = None,
        mean_epoch_time_sec: int = None,
        eta_ten_epochs_min: int = None,
        epochs_in_plateau: int = None,
        plateau_patience: int = None,
        epoch_in_plateau_flag: bool = False,
        best_val_x: int = None,
        best_val_y: float = None,
        epoch_size: int = None,
    ):

        # Add training epoch and runtime info
        title = self._get_training_epoch_and_runtime_text(epoch, dt_min, dt_sec)

        if last_epoch_val_loss is not None:

            if penultimate_epoch_val_loss is not None:
                # Add mean epoch time and ETA for next 10 epochs
                eta_text = self._get_eta_text(
                    mean_epoch_time_min, mean_epoch_time_sec, eta_ten_epochs_min
                )
                title = self._add_with_newline(title, eta_text)

                # Add epochs in plateau if flag is set
                if epoch_in_plateau_flag:
                    plateau_text = self._get_epochs_in_plateau_text(
                        epochs_in_plateau, plateau_patience
                    )
                    title = self._add_with_newline(title, plateau_text)

            # Add last epoch validation loss
            last_val_text = self._get_last_validation_loss_text(last_epoch_val_loss)
            title = self._add_with_newline(title, last_val_text)

            # Add best epoch validation loss if available
            if best_val_x is not None:
                best_epoch = (best_val_x // epoch_size) + 1
                best_val_text = self._get_best_validation_loss_text(
                    best_val_y, best_epoch
                )
                title = self._add_with_newline(title, best_val_text)

        self.set_title(title)

    @staticmethod
    def _get_training_epoch_and_runtime_text(epoch: int, dt_min: int, dt_sec: int):
        """Get the training epoch and runtime text to display in the plot.

        Args:
            epoch: The current epoch.
            dt_min: The number of minutes since training started.
            dt_sec: The number of seconds since training started.
        """

        runtime_text = (
            r"Training Epoch $\mathbf{" + str(epoch + 1) + r"}$ / "
            r"Runtime: $\mathbf{" + f"{int(dt_min):02}:{int(dt_sec):02}" + r"}$"
        )

        return runtime_text

    @staticmethod
    def _get_eta_text(mean_epoch_time_min, mean_epoch_time_sec, eta_ten_epochs_min):
        """Get the mean time and ETA text to display in the plot.

        Args:
            mean_epoch_time_min: The mean time per epoch in minutes.
            mean_epoch_time_sec: The mean time per epoch in seconds.
            eta_ten_epochs_min: The estimated time for the next ten epochs in minutes.
        """

        runtime_text = (
            r"Mean Time per Epoch: $\mathbf{"
            + f"{int(mean_epoch_time_min):02}:{int(mean_epoch_time_sec):02}"
            + r"}$ / "
            r"ETA Next 10 Epochs: $\mathbf{" + f"{int(eta_ten_epochs_min)}" + r"}$ min"
        )

        return runtime_text

    @staticmethod
    def _get_epochs_in_plateau_text(epochs_in_plateau, plateau_patience):
        """Get the epochs in plateau text to display in the plot.

        Args:
            epochs_in_plateau: The number of epochs in plateau.
            plateau_patience: The number of epochs to wait before stopping training.
        """

        plateau_text = (
            r"Epochs in Plateau: $\mathbf{" + f"{epochs_in_plateau}" + r"}$ / "
            r"$\mathbf{" + f"{plateau_patience}" + r"}$"
        )

        return plateau_text

    @staticmethod
    def _get_last_validation_loss_text(last_epoch_val_loss):
        """Get the last epoch validation loss text to display in the plot.

        Args:
            last_epoch_val_loss: The validation loss from the last epoch.
        """

        last_val_loss_text = (
            "Last Epoch Validation Loss: "
            r"$\mathbf{" + f"{last_epoch_val_loss:.3e}" + r"}$"
        )

        return last_val_loss_text

    @staticmethod
    def _get_best_validation_loss_text(best_val_y, best_epoch):
        """Get the best epoch validation loss text to display in the plot.

        Args:
            best_val_x: The epoch number of the best validation loss.
            best_val_y: The best validation loss.
        """

        best_val_loss_text = (
            r"Best Epoch Validation Loss: $\mathbf{"
            + f"{best_val_y:.3e}"
            + r"}$ (epoch $\mathbf{"
            + str(best_epoch)
            + r"}$)"
        )

        return best_val_loss_text

    @staticmethod
    def _add_with_newline(old_text: str, new_text: str):
        """Add a new line to the text.

        Args:
            old_text: The existing text.
            new_text: The text to add on a new line.
        """

        return old_text + "\n" + new_text

    @staticmethod
    def _calculate_xlim(x: np.ndarray, dx: float = 0.5):
        """Calculates x-axis limits.

        Args:
            x: Array of x data to fit the limits to.
            dx: The padding to add to the limits.

        Returns:
            Tuple of the minimum and maximum x-axis limits.
        """

        x_min = min(x) - dx
        x_min = x_min if x_min > 0 else 0
        x_max = max(x) + dx

        return x_min, x_max

    def _calculate_ylim(self, y: np.ndarray, dy: float = 0.02):
        """Calculates y-axis limits.

        Args:
            y: Array of y data to fit the limits to.
            dy: The padding to add to the limits.

        Returns:
            Tuple of the minimum and maximum y-axis limits.
        """

        if self.ignore_outliers:
            dy = np.ptp(y) * 0.02
            # Set Y scale to exclude outliers
            q1, q3 = np.quantile(y, (0.25, 0.75))
            iqr = q3 - q1  # Interquartile range
            y_min = q1 - iqr * 1.5
            y_max = q3 + iqr * 1.5

            # Keep within range of data
            y_min = max(y_min, min(y) - dy)
            y_max = min(y_max, max(y) + dy)
        else:
            # Set Y scale to show all points
            dy = np.ptp(y) * 0.02
            y_min = min(y) - dy
            y_max = max(y) + dy

        # For log scale, low cannot be 0
        if self.log_scale:
            y_min = max(y_min, 1e-8)

        return y_min, y_max

    def _set_title_space(self):
        """Set up the title space.

        Returns:
            The height of the title space as a decimal fraction of the total figure height.
        """

        # Set a dummy title of the plot
        n_lines = 5  # Number of lines in the title
        title_str = "\n".join(
            [r"Number: $\mathbf{" + str(n) + r"}$" for n in range(n_lines + 1)]
        )
        self.set_title(
            title_str, color="white"
        )  # Set the title color to white so it's not visible

        # Draw the canvas to ensure the title is created
        self.fig.canvas.draw()

        # Get the title Text object
        title = self.axes.title

        # Get the bounding box of the title in display coordinates
        bbox = title.get_window_extent()

        # Transform the bounding box to figure coordinates
        bbox = bbox.transformed(self.fig.transFigure.inverted())

        # Calculate the height of the title as a percentage of the total figure height
        title_height = bbox.height

        return title_height

    def _setup_x_axis(self):
        """Set up the x axis.

        This includes setting the label, limits, and bottom/right adjustment.
        """

        self.axes.set_xlim(0, 1)
        self.axes.set_xlabel("Batches", fontweight="bold", fontsize="small")

        # Set the x-label in the center of the axes and some amount above the bottom of the figure
        blended_transform = mtransforms.blended_transform_factory(
            self.axes.transAxes, self.fig.transFigure
        )
        self.axes.xaxis.set_label_coords(
            0.5, self.ypos_xlabel, transform=blended_transform
        )

    def _set_up_y_axis(self):
        """Set up the y axis.

        This includes setting the label, limits, scaling, and left adjustment.
        """

        # Set the minimum value of the y-axis depending on scaling
        if self.log_scale:
            yscale = "log"
            y_min = 0.001
        else:
            yscale = "linear"
            y_min = 0
        self.axes.set_ylim(bottom=y_min)
        self.axes.set_yscale(yscale)

        # Set the y-label name, size, wight, and position
        self.axes.set_ylabel("Loss", fontweight="bold", fontsize="small")
        self.axes.yaxis.set_label_coords(
            self.xpos_ylabel, 0.5, transform=self.fig.transFigure
        )

    def _setup_legend(self):
        """Set up the legend.

        Returns:
            Tuple of the width and height of the legend as a decimal fraction of the total figure width and height.
        """

        # Move the legend outside the plot on the upper left
        legend = self.axes.legend(
            loc="upper left",
            fontsize="small",
            bbox_to_anchor=(0, 1),
            bbox_transform=self.fig.transFigure,
        )

        # Draw the canvas to ensure the legend is created
        self.fig.canvas.draw()

        # Get the bounding box of the legend in display coordinates
        bbox = legend.get_window_extent()

        # Transform the bounding box to figure coordinates
        bbox = bbox.transformed(self.fig.transFigure.inverted())

        # Calculate the width and height of the legend as a percentage of the total figure width and height
        return bbox.width, bbox.height

    def _setup_major_gridlines(self):

        # Set the outline color of the plot to gray
        for spine in self.axes.spines.values():
            spine.set_edgecolor("#d3d3d3")  # Light gray color

        # Remove the top and right axis spines
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)

        # Set the tick markers color to light gray, but not the tick labels
        self.axes.tick_params(
            axis="both", which="both", color="#d3d3d3", labelsize="small"
        )

        # Add gridlines at the tick labels
        self.axes.grid(True, which="major", linewidth=0.5, color="#d3d3d3")

    def _add_midpoint_gridlines(self):
        # Clear existing minor vertical lines
        for line in self.axes.get_lines():
            if line.get_linestyle() == ":":
                line.remove()

        # Add gridlines at midpoint between major ticks
        major_ticks = self.axes.yaxis.get_majorticklocs()
        if len(major_ticks) > 1:
            prev_major_tick = major_ticks[0]
            for major_tick in major_ticks[:-1]:
                midpoint = (major_tick + prev_major_tick) / 2
                self.axes.axhline(
                    midpoint, linestyle=":", linewidth=0.5, color="#d3d3d3"
                )
                prev_major_tick = major_tick

    def _init_series(
        self,
        series_type,
        color,
        name: Optional[str] = None,
        line_width: Optional[float] = None,
        border_color: Optional[Tuple[int, int, int]] = None,
        zorder: Optional[int] = None,
    ):

        # Set the color
        color = [c / 255.0 for c in color]  # Normalize color values to [0, 1]

        # Create the series
        series = series_type(
            [],
            [],
            color=color,
            label=name,
            marker="o",
            zorder=zorder,
        )

        # ax.plot returns a list of PathCollections, so we need to get the first one
        if not isinstance(series, PathCollection):
            series = series[0]

        if line_width is not None:
            series.set_linewidth(line_width)

        # Set the border color (edge color)
        if border_color is not None:
            border_color = [
                c / 255.0 for c in border_color
            ]  # Normalize color values to [0, 1]
            series.set_edgecolor(border_color)

        return series


class LossViewer(QtWidgets.QMainWindow):
    """Qt window for showing in-progress training metrics sent over ZMQ."""

    on_epoch = QtCore.Signal()

    def __init__(
        self,
        zmq_ports: Dict = None,
        zmq_context: Optional[zmq.Context] = None,
        show_controller=True,
        parent=None,
    ):
        super().__init__(parent)

        self.show_controller = show_controller
        self.stop_button = None
        self.cancel_button = None
        self.canceled = False

        # Set up ZMQ ports for communication.
        zmq_ports = zmq_ports or dict()
        zmq_ports["publish_port"] = zmq_ports.get("publish_port", 9001)
        zmq_ports["controller_port"] = zmq_ports.get("controller_port", 9000)
        self.zmq_ports = zmq_ports

        self.batches_to_show = -1  # -1 to show all
        self._ignore_outliers = False
        self._log_scale = True
        self.message_poll_time_ms = 20  # ms
        self.redraw_batch_time_ms = 500  # ms
        self.last_redraw_batch = None

        self.canvas = None
        self.reset()
        self._setup_zmq(zmq_context)

    def __del__(self):
        self._unbind()

    @property
    def is_timer_running(self) -> bool:
        """Return True if the timer has started."""
        return self.t0 is not None and self.is_running

    @property
    def log_scale(self):
        """Returns True if the plot has a log scale for y-axis."""

        return self._log_scale

    @log_scale.setter
    def log_scale(self, val):
        """Sets the scale of the y axis to log if True else linear."""

        if isinstance(val, bool):
            self._log_scale = val

        # Set the log scale on the canvas
        self.canvas.log_scale = self._log_scale

    @property
    def ignore_outliers(self):
        """Returns True if the plot ignores outliers."""

        return self._ignore_outliers

    @ignore_outliers.setter
    def ignore_outliers(self, val):
        """Sets whether to ignore outliers in the plot."""

        if isinstance(val, bool):
            self._ignore_outliers = val

        # Set the ignore_outliers on the canvas
        self.canvas.ignore_outliers = self._ignore_outliers

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
        self.canvas = LossPlot(
            width=5,
            height=4,
            dpi=100,
            log_scale=self.log_scale,
            ignore_outliers=self.ignore_outliers,
        )

        self.mp_series = dict()
        self.mp_series["batch"] = self.canvas.series["batch"]
        self.mp_series["epoch_loss"] = self.canvas.series["epoch_loss"]
        self.mp_series["val_loss"] = self.canvas.series["val_loss"]
        self.mp_series["val_loss_best"] = self.canvas.series["val_loss_best"]

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)

        if self.show_controller:
            control_layout = QtWidgets.QHBoxLayout()

            field = QtWidgets.QCheckBox("Log Scale")
            field.setChecked(self.log_scale)
            field.stateChanged.connect(self._toggle_log_scale)
            control_layout.addWidget(field)

            field = QtWidgets.QCheckBox("Ignore Outliers")
            field.setChecked(self.ignore_outliers)
            field.stateChanged.connect(self._toggle_ignore_outliers)
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
                lambda x: self._set_batches_to_show(self.batch_options[x])
            )

            # Store field as property and add to layout.
            self.batches_to_show_field = field
            control_layout.addWidget(self.batches_to_show_field)

            control_layout.addStretch(1)

            self.stop_button = QtWidgets.QPushButton("Stop Early")
            self.stop_button.clicked.connect(self._stop)
            control_layout.addWidget(self.stop_button)
            self.cancel_button = QtWidgets.QPushButton("Cancel Training")
            self.cancel_button.clicked.connect(self._cancel)
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

    def set_message(self, text: str):
        """Set the chart title text."""
        self.canvas.set_title(text)

    def close(self):
        """Disconnect from ZMQ ports and close the window."""
        self._unbind()
        super().close()

    def _setup_zmq(self, zmq_context: Optional[zmq.Context] = None):
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

        def find_free_port(port: int, zmq_context: zmq.Context):
            """Find free port to bind to.

            Args:
                port: The port to start searching from.
                zmq_context: The ZMQ context to use.

            Returns:
                The free port.
            """
            attempts = 0
            max_attempts = 10
            while not is_port_free(port=port, zmq_context=zmq_context):
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Could not find free port to display training progress after "
                        f"{max_attempts} attempts. Please check your network settings "
                        "or use the CLI `sleap-train` command."
                    )
                port = select_zmq_port(zmq_context=self.ctx)
                attempts += 1

            return port

        # Find a free port and bind to it.
        self.zmq_ports["publish_port"] = find_free_port(
            port=self.zmq_ports["publish_port"], zmq_context=self.ctx
        )
        publish_address = f"tcp://127.0.0.1:{self.zmq_ports['publish_port']}"
        self.sub.bind(publish_address)

        # Controller, PUBLISHER
        self.zmq_ctrl = None
        if self.show_controller:
            self.zmq_ctrl = self.ctx.socket(zmq.PUB)

            # Find a free port and bind to it.
            self.zmq_ports["controller_port"] = find_free_port(
                port=self.zmq_ports["controller_port"], zmq_context=self.ctx
            )
            controller_address = f"tcp://127.0.0.1:{self.zmq_ports['controller_port']}"
            self.zmq_ctrl.bind(controller_address)

        # Set timer to poll for messages.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._check_messages)
        self.timer.start(self.message_poll_time_ms)

    def _set_batches_to_show(self, batches: str):
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

    def _set_start_time(self, t0: float):
        """Mark the start flag and time of the run.

        Args:
            t0: Start time in seconds.
        """
        self.t0 = t0
        self.is_running = True

    def _update_runtime(self):
        """Update the title text with the current running time."""

        if self.is_timer_running:
            dt = perf_counter() - self.t0
            dt_min, dt_sec = divmod(dt, 60)

            self.canvas.update_runtime_title(
                epoch=self.epoch,
                dt_min=dt_min,
                dt_sec=dt_sec,
                last_epoch_val_loss=self.last_epoch_val_loss,
                penultimate_epoch_val_loss=self.penultimate_epoch_val_loss,
                mean_epoch_time_min=self.mean_epoch_time_min,
                mean_epoch_time_sec=self.mean_epoch_time_sec,
                eta_ten_epochs_min=self.eta_ten_epochs_min,
                epochs_in_plateau=self.epochs_in_plateau,
                plateau_patience=self.config.optimization.early_stopping.plateau_patience,
                epoch_in_plateau_flag=self.epoch_in_plateau_flag,
                best_val_x=self.best_val_x,
                best_val_y=self.best_val_y,
                epoch_size=self.epoch_size,
            )

    def _check_messages(
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
                self._set_start_time(perf_counter())
                self.current_job_output_type = msg["what"]

            # Make sure message matches current training job.
            if msg.get("what", "") == self.current_job_output_type:

                if not self.is_timer_running:
                    # We must have missed the train_begin message, so start timer now.
                    self._set_start_time(perf_counter())

                if msg["event"] == "train_end":
                    self._set_end()
                elif msg["event"] == "epoch_begin":
                    self.epoch = msg["epoch"]
                elif msg["event"] == "epoch_end":
                    self.epoch_size = max(self.epoch_size, self.last_batch_number + 1)
                    self._add_datapoint(
                        (self.epoch + 1) * self.epoch_size,
                        msg["logs"]["loss"],
                        "epoch_loss",
                    )
                    if "val_loss" in msg["logs"].keys():
                        # update variables and add points to plot
                        self.penultimate_epoch_val_loss = self.last_epoch_val_loss
                        self.last_epoch_val_loss = msg["logs"]["val_loss"]
                        self._add_datapoint(
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
                    self._add_datapoint(
                        (self.epoch * self.epoch_size) + msg["batch"],
                        msg["logs"]["loss"],
                        "batch",
                    )

            # Check for messages again (up to times_to_check times).
            if times_to_check > 0:
                self._check_messages(
                    timeout=timeout, times_to_check=times_to_check - 1, do_update=False
                )

        if do_update:
            self._update_runtime()

    def _add_datapoint(self, x: int, y: float, which: str):
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

                # Set data, resize and redraw the plot
                self._set_data_on_scatter(xs, ys, which)
                self._resize_axes(xs, ys)

        else:

            if which == "val_loss":
                if self.best_val_y is None or y < self.best_val_y:
                    self.best_val_x = x
                    self.best_val_y = y
                    self._set_data_on_scatter([x], [y], "val_loss_best")

            # Add data and redraw the plot
            self._add_data_to_plot(x, y, which)
            self._redraw_plot()

    def _set_data_on_scatter(self, xs, ys, which):
        """Add data to a scatter plot.

        Not to be used with line plots.

        Args:
            xs: The x-coordinates of the data points.
            ys: The y-coordinates of the data points.
            which: The type of data point. Possible values are:
                * "batch"
                * "val_loss_best"
        """

        self.canvas.set_data_on_scatter(xs, ys, which)

    def _add_data_to_plot(self, x, y, which):
        """Add data to a line plot.

        Not to be used with scatter plots.

        Args:
            x: The x-coordinate of the data point.
            y: The y-coordinate of the data point.
            which: The type of data point. Possible values are:
                * "epoch_loss"
                * "val_loss"
        """

        self.canvas.add_data_to_plot(x, y, which)

    def _redraw_plot(self):
        """Redraw the plot."""

        self.canvas.redraw_plot()

    def _resize_axes(self, x, y):
        """Resize axes to fit data.

        This is only called when plotting batches.

        Args:
            x: The x-coordinates of the data points.
            y: The y-coordinates of the data points.
        """
        self.canvas.resize_axes(x, y)

    def _toggle_ignore_outliers(self):
        """Toggles whether to ignore outliers in chart scaling."""

        self.ignore_outliers = not self.ignore_outliers

    def _toggle_log_scale(self):
        """Toggle whether to use log-scaled y-axis."""

        self.log_scale = not self.log_scale

    def _stop(self):
        """Send command to stop training."""
        if self.zmq_ctrl is not None:
            # Send command to stop training.
            logger.info("Sending command to stop training.")
            self.zmq_ctrl.send_string(jsonpickle.encode(dict(command="stop")))

        # Disable the button to prevent double messages.
        if self.stop_button is not None:
            self.stop_button.setText("Stopping...")
            self.stop_button.setEnabled(False)

    def _cancel(self):
        """Set the cancel flag."""
        self.canceled = True
        if self.cancel_button is not None:
            self.cancel_button.setText("Canceling...")
            self.cancel_button.setEnabled(False)

    def _unbind(self):
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

    def _set_end(self):
        """Mark the end of the run."""
        self.is_running = False
