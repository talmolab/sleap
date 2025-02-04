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
