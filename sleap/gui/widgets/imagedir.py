"""
Qt widget for showing images from a directory.

There's a seekbar to move between images, and an optional drop-down menu
to select different filename filters (e.g., "validation.*.png").

The typical use-case is to show predictions for each training epoch on the
training/validation images. For this use-case, there's a factory method which
creates widgets with relevant filters from a given training run path.
"""
from sleap import Video
from sleap.gui.widgets.video import QtVideoPlayer

from PySide2 import QtWidgets

import glob
import os
from typing import List, Optional, Text, Tuple


class QtImageDirectoryWidget(QtVideoPlayer):
    """
    Qt widget (window) for showing images from directory with seekbar.

    Call `poll()` method to check directory for new files.

    Arguments:
        directory: The path for which to search for image files.
        filters: Filename filters, given as (display name, filter)-tuples,
            e.g., ("Validation", "validation.*.png")
    """

    def __init__(
        self, directory: Text, filters: Optional[List[Tuple[Text, Text]]] = None
    ):
        self.directory = directory
        self.filters = filters
        self.files = []

        super(QtImageDirectoryWidget, self).__init__()
        self.seekbar.tick_index_offset = 0  # show frame numbers indexed by 0
        self.changedPlot.connect(
            lambda vp, idx, select_idx: self.setWindowTitle(
                self._get_win_title_for_frame(idx)
            )
        )

        self.resize(360, 400)

        if self.filters:
            self.filter_menu = QtWidgets.QComboBox()
            self.filter_menu.addItems([filter[0] for filter in self.filters])
            self.filter_menu.currentIndexChanged.connect(self.setFilter)
            self.layout.addWidget(self.filter_menu)

            self.setFilter(self.filter_menu.currentIndex())
        else:
            self.poll()

    def _get_win_title_for_frame(self, frame_idx: int) -> Text:
        """Get window title to use based on specified frame."""
        return (
            os.path.basename(self.files[frame_idx])
            if frame_idx < len(self.files)
            else ""
        )

    def setFilter(self, filter_idx: int):
        """Set filter (by number) for which files in directory to show."""
        self.filter_idx = filter_idx
        self.poll()

    @property
    def _current_filter_mask(self) -> Text:
        if not self.filters:
            return "*"
        return self.filters[self.filter_idx][1]

    def poll(self):
        """Re-scans directory (using current filter) and updates widget."""
        path = os.path.join(self.directory, self._current_filter_mask)
        print(f"Polling: {path}")

        files = glob.glob(path)
        files.sort()

        if not files:
            return

        if files != self.files:
            was_on_last_image = False
            if self.video is None:
                was_on_last_image = True
                self.show()
            elif self.state["frame_idx"] == self.video.last_frame_idx:
                was_on_last_image = True

            self.files = files
            self.video = Video.from_image_filenames(filenames=files)
            self.load_video(video=self.video)

            if was_on_last_image:
                self.state["frame_idx"] = self.video.last_frame_idx
            elif self.state["frame_idx"]:
                self.state["frame_idx"] = min(
                    self.state["frame_idx"], self.video.last_frame_idx
                )

    @classmethod
    def make_training_vizualizer(cls, run_path: Text) -> "QtImageDirectoryWidget":
        """
        Factory method for only currently use-case of this widget.

        Args:
            run_path: The run path directory for model, should contain viz
                subdirectory.

        Returns:
            Instance of `QtImageDirectoryWidget` widget.
        """
        dir = os.path.join(run_path, "viz")
        filters = [("Validation", "validation.*.png"), ("Training", "train.*.png")]
        win = QtImageDirectoryWidget(dir, filters=filters)
        return win


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication

    app = QApplication([])

    # run_path = "tests/data/json_format_v1/models/200420_100503.centroid.70"
    # window = QtImageDirectoryWidget.make_training_vizualizer(run_path)
    window = QtImageDirectoryWidget(
        directory="tests/data/videos/",
        filters=[("JPEG", "*.jpg"), ("Robot 1", "*1.jpg")],
    )

    window.show()
    window.plot()

    app.exec_()
