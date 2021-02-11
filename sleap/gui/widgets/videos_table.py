"""General purpose video selector widget. Useful for prompting user for input videos."""

from typing import Optional, List, Dict

import sleap
from sleap.gui.dataviews import (
    GenericCheckableTableModel,
    GenericTableView,
)
from sleap.gui.dialogs.importvideos import ImportVideos

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
)


class VideosTableModel(GenericCheckableTableModel):
    properties = (
        "Path",
        "Frames",
        "Image size",
        "First frame",
        "Last frame",
    )
    sort_as_string = ("Path",)
    row_name = "videos"

    def item_to_data(self, obj, video):
        item_data = {
            "Path": video.filename,
            "Frames": video.frames,
            "Image size": str(video.shape[1:]),
            "First frame": 1,
            "Last frame": video.frames,
        }
        return item_data

    def can_set(self, item, key):
        return key in ["First frame", "Last frame"]

    def set_item(self, video, key, value):
        row = self.original_items.index(video)
        item = self.items[row]
        try:
            val = int(value)
        except:
            return
        if key == "First frame":
            val = min(val, len(video), item["Last frame"])
            val = max(val, 1)
            # self.items[row]["First frame"] = val
            self._data[row]["First frame"] = val
        elif key == "Last frame":
            val = min(val, len(video))
            val = max(val, 1, item["First frame"])
            # self.items[row]["Last frame"] = val
            self._data[row]["Last frame"] = val


class VideosTableView(GenericTableView):
    row_name = "videos"
    is_activatable = True
    is_sortable = True
    resize_mode = "contents"


class VideosTableWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.table_model = VideosTableModel()
        self.table_view = VideosTableView(model=self.table_model)

        layout = QVBoxLayout()
        self.setLayout(layout)

        add_button = QPushButton("Add videos...")
        add_button.clicked.connect(lambda: self.add_videos())

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.table_model.remove_checked)
        remove_button.setEnabled(False)

        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(self.table_model.check_all)
        select_all_button.setEnabled(False)

        select_none_button = QPushButton("Select none")
        select_none_button.clicked.connect(self.table_model.check_none)
        select_none_button.setEnabled(False)

        hl = QHBoxLayout()
        hl.addWidget(add_button)
        hl.addWidget(remove_button)
        hl.addWidget(select_all_button)
        hl.addWidget(select_none_button)
        layout.addLayout(hl)

        layout.addWidget(self.table_view)

        status_label = QLabel("0 files selected.")
        layout.addWidget(status_label)

        def update_status_label():
            videos = self.checked_videos
            if len(videos) == 0:
                status_label.setText("0 videos selected.")
            else:
                n_frames = 0
                for video in videos:
                    item = self.table_model.items[
                        self.table_model.original_items.index(video)
                    ]
                    n_frames += item["Last frame"] - item["First frame"] + 1
                status_label.setText(
                    f"{len(videos)} videos selected. " f"Total: {n_frames:,} frames"
                )

        self.table_model.checked.connect(lambda x: remove_button.setEnabled(len(x) > 0))
        self.table_model.items_changed.connect(
            lambda x: select_all_button.setEnabled(len(x) > 0)
        )
        self.table_model.items_changed.connect(
            lambda x: select_none_button.setEnabled(len(x) > 0)
        )
        self.table_model.checked.connect(update_status_label)
        self.table_model.dataChanged.connect(update_status_label)

    @property
    def videos(self):
        return self.table_model.original_items

    @property
    def checked_videos(self):
        return self.table_model.checked_items

    @property
    def video_paths(self):
        return [video.filename for video in self.videos]

    def add_videos(self, video_paths: Optional[str] = None):
        if video_paths is None:
            videos = ImportVideos().ask_and_return_videos()

        if video_paths is not None:
            if isinstance(video_paths, str):
                video_paths = [video_paths]
            videos = [sleap.load_video(p) for p in video_paths]

        videos = [v for v in videos if v.filename not in self.video_paths]

        if len(videos) > 0:
            self.table_model.items = self.videos + videos
            self.table_model.set_checked(videos, True)
            self.table_view.resizeColumnsToContents()
