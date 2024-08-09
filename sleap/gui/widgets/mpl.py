"""
Widget which wraps Matplotlib canvas.

Currently this is used for plotting metrics graphs in GUI.
"""

from qtpy import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
import matplotlib

matplotlib.use("QtAgg")


class MplCanvas(Canvas):
    """Matplotlib canvas."""

    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        super(MplCanvas, self).__init__(self.fig)

        Canvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        Canvas.updateGeometry(self)
        self.series = {}

    def add_scatter(self, key, x, y, label, color, markersize):
        if key in self.series:
            self.series[key].remove()
        scatter = self.axes.scatter(x, y, label=label, color=color, s=markersize)
        self.series[key] = scatter
        self.draw()

    def add_line(self, key, x, y, label, color, width):
        if key in self.series:
            self.series[key].remove()
        (line,) = self.axes.plot(x, y, label=label, color=color, linewidth=width)
        self.series[key] = line
        self.draw()

    def clear(self):
        self.axes.cla()
        self.series.clear()
        self.draw()
