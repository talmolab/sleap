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
            lambda vp, idx, select_idx: self.setWindowTitle(self.getFrameTitle(idx))
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

    def getFrameTitle(self, frame_idx):
        return (
            os.path.basename(self.files[frame_idx])
            if frame_idx < len(self.files)
            else ""
        )

    def setFilter(self, filter_idx):
        self.filter_idx = filter_idx
        self.poll()

    def getFilterMask(self):
        if not self.filters:
            return "*"
        return self.filters[self.filter_idx][1]

    def poll(self):
        path = os.path.join(self.directory, self.getFilterMask())
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
    def make_training_vizualizer(cls, run_path: Text):
        dir = os.path.join(run_path, "viz")
        filters = [("Validation", "validation.*.png"), ("Training", "train.*.png")]
        win = QtImageDirectoryWidget(dir, filters=filters)
        return win


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication

    app = QApplication([])

    run_path = "tests/data/json_format_v1/models/200130_155013.UNet.centroids/"
    window = QtImageDirectoryWidget.make_training_viz(run_path)

    window.show()
    window.plot()

    app.exec_()
