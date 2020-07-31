"""
Base classes for overlays.

Overlays are used for showing additional visuals on top of a video frame (i.e.,
a `QtVideoPlayer` object). Overlay objects are created in the main GUI app,
which then automatically calls the `add_to_scene` for each loaded overlay after
drawing a frame (i.e., when user navigates to a new frame or something changes
so that current frame must be redrawn).
"""

from PySide2 import QtWidgets

import attr
import abc
import numpy as np
from typing import Sequence, Union

from sleap import Labels, Video
from sleap.gui.widgets.video import QtVideoPlayer
from sleap.nn.data.providers import VideoReader
from sleap.nn.inference import VisualPredictor


@attr.s(auto_attribs=True)
class BaseOverlay(abc.ABC):
    """
    Abstract base class for overlays.

    Most overlays need rely on the `Labels` from which to get data and need the
    `QtVideoPlayer` to which a `QGraphicsObject` item will be added, so these
    attributes are included in the base class.
    """

    labels: Labels = None
    player: QtVideoPlayer = None

    @abc.abstractmethod
    def add_to_scene(self, video: Video, frame_idx: int):
        pass


@attr.s(auto_attribs=True)
class ModelData(Sequence):
    """Sequence-type object which generates predictions for specified frames."""

    predictor: VisualPredictor
    result_key: str
    video: Video
    output_scale: float = 1.0
    adjust_vals: bool = True

    def __getitem__(self, i: int) -> np.ndarray:
        """Data data for frame i from predictor."""
        # Get predictions for frame i
        frame_result = self.predictor.predict(VideoReader(self.video, [i]))

        # We just want the single image results
        frame_result = frame_result[0][self.result_key]

        if self.adjust_vals:
            frame_result = np.clip(frame_result, 0, 1)

        # Determine output scale by comparing original image with model output
        self.output_scale = self.video.height / frame_result.shape[0]

        return frame_result

    def __len__(self):
        return self.video.num_frames


@attr.s(auto_attribs=True)
class DataOverlay(BaseOverlay):
    """
    Base class for confidence maps/part affinity fields overlays.

    These overlays use a `ModelData` class which provides the confidence maps/
    part affinity fields for the frame (by running inference with a model).
    They could easily be modified to use another "data" class, e.g., one
    which load saved confidence maps/part affinity fields from a file.

    Attributes:
        data: instance of a class such that you can use `data[frame_idx]`
            to get the data (e.g., confmaps) for a given frame.
        overlay_class: determines how the data will be shown, i.e., as
            confidence maps or as a quiver plot (for part affinity fields).
    """

    data: Sequence = None
    overlay_class: Union["ConfMapsPlot", "MultiQuiverPlot", None] = None

    def add_to_scene(self, video: Video, frame_idx: int):
        if self.data is None:
            return

        if self.overlay_class is None:
            return

        img_data = self.data[frame_idx]
        img_scale = self.data.output_scale

        self._add(
            to=self.player.view.scene,
            what=self.overlay_class(img_data, scale=img_scale),
        )

    def _add(
        self,
        to: QtWidgets.QGraphicsScene,
        what: QtWidgets.QGraphicsObject,
        where: tuple = (0, 0),
    ):
        to.addItem(what)
        what.setPos(*where)

    @classmethod
    def make_predictor(cls, filename: str) -> VisualPredictor:
        return VisualPredictor.from_trained_models(filename)

    @classmethod
    def from_model(cls, filename: str, *args, **kwargs):
        return cls.from_predictor(cls.make_predictor(filename), *args, **kwargs)

    @classmethod
    def from_predictor(
        cls, predictor: VisualPredictor, video: Video, show_pafs: bool = False, **kwargs
    ):
        # imports here so we don't get circular dependencies
        from sleap.gui.overlays.confmaps import ConfMapsPlot
        from sleap.gui.overlays.pafs import MultiQuiverPlot

        if show_pafs:
            result_key = predictor.part_affinity_fields_key_name
        else:
            result_key = predictor.confidence_maps_key_name

        data_object = ModelData(predictor=predictor, result_key=result_key, video=video)

        # Determine whether to use confmap or paf overlay
        if show_pafs:
            overlay_class = MultiQuiverPlot
        else:
            overlay_class = ConfMapsPlot

        return cls(data=data_object, overlay_class=overlay_class, **kwargs)


h5_colors = [
    [204, 81, 81],
    [81, 204, 204],
    [51, 127, 127],
    [127, 51, 51],
    [142, 204, 81],
    [89, 127, 51],
    [142, 81, 204],
    [89, 51, 127],
    [204, 173, 81],
    [127, 108, 51],
    [81, 204, 112],
    [51, 127, 70],
    [81, 112, 204],
    [51, 70, 127],
    [204, 81, 173],
    [127, 51, 108],
    [204, 127, 81],
    [127, 79, 51],
    [188, 204, 81],
    [117, 127, 51],
    [96, 204, 81],
    [60, 127, 51],
    [81, 204, 158],
    [51, 127, 98],
    [81, 158, 204],
    [51, 98, 127],
    [96, 81, 204],
    [60, 51, 127],
    [188, 81, 204],
    [117, 51, 127],
    [204, 81, 127],
    [127, 51, 79],
    [204, 104, 81],
    [127, 65, 51],
    [204, 150, 81],
    [127, 94, 51],
    [204, 196, 81],
    [127, 122, 51],
    [165, 204, 81],
    [103, 127, 51],
    [119, 204, 81],
    [74, 127, 51],
    [81, 204, 89],
    [51, 127, 55],
    [81, 204, 135],
    [51, 127, 84],
    [81, 204, 181],
    [51, 127, 113],
    [81, 181, 204],
    [51, 113, 127],
]
