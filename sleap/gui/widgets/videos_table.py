"""General purpose video selector widget. Useful for prompting user for input videos."""

from typing import Optional, List, Union

import sleap
from sleap.gui.dataviews import (
    GenericCheckableTableModel,
    GenericTableView,
)
from sleap.gui.dialogs.importvideos import ImportVideos

from PySide2.QtWidgets import (
    QWidget,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
)

from sleap.util import frame_list


class VideosTableModel(GenericCheckableTableModel):
    properties = (
        "Path",
        "Frames",
        "Image size",
        "Selected frames",
    )
    sort_as_string = ("Path",)
    row_name = "videos"

    def item_to_data(self, obj, video):
        item_data = {
            "Path": video.filename,
            "Frames": video.frames,
            "Image size": str(video.shape[1:]),
            "Selected frames": f"{1}-{video.frames}",
        }
        return item_data

    def can_set(self, item, key):
        return key in ["Selected frames"]

    def set_item(self, video, key, value):
        row = self.original_items.index(video)
        if key == "Selected frames":
            if not value or len(frame_list(value)) == 0:
                value = f"{1}-{video.frames}"
            self._data[row]["Selected frames"] = value
        else:
            raise ValueError(f"Unknown property {key}")


class VideosTableView(GenericTableView):
    row_name = "videos"
    is_activatable = True
    is_sortable = True
    resize_mode = "contents"


class VideosTableWidget(QWidget):
    def __init__(self, table_model: VideosTableModel):
        super().__init__()

        self.table_model = table_model
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
                    n_frames += len(frame_list(self.selected_frames(video)))
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

    def selected_frames(self, video):
        item = self.table_model.items[
            self.table_model.original_items.index(video)
        ]
        return item["Selected frames"]

    @property
    def videos(self):
        return self.table_model.original_items

    @property
    def video_paths(self):
        return [video.filename for video in self.videos]

    @property
    def checked_videos(self):
        return self.table_model.checked_items

    @property
    def checked_video_paths(self):
        return [video.filename for video in self.checked_videos]

    @property
    def checked_video_frames(self):
        return [self.selected_frames(video) for video in self.checked_videos]

    def add_videos(self, video_paths: Optional[Union[str, List[str]]] = None):
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

    def set_videos(self, video_paths: List[str], selected_frames: Optional[List[str]] = None):
        videos = [sleap.load_video(p) for p in video_paths]
        self.table_model.items = videos
        if selected_frames:
            for v, f in zip(video_paths, selected_frames):
                self.table_model.set_item(v, "Selected frames", f)
        self.table_model.set_checked(videos, True)
        self.table_view.resizeColumnsToContents()
