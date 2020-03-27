"""Base class for overlays."""

from PySide2 import QtWidgets

import attr
import numpy as np
from typing import Sequence, Union

from sleap.io.video import Video
from sleap.gui.widgets.video import QtVideoPlayer
from sleap.nn.data.providers import VideoReader
from sleap.nn.inference import VisualPredictor


@attr.s(auto_attribs=True)
class ModelData:
    predictor: VisualPredictor
    video: Video
    do_rescale: bool = False
    output_scale: float = 1.0
    adjust_vals: bool = True

    def __getitem__(self, i: int):
        """Data data for frame i from predictor."""
        # Get predictions for frame i
        frame_result = self.predictor.predict(VideoReader(self.video, [i]))

        # We just want the single image results
        # todo: support for pafs
        frame_result = frame_result[0][self.predictor.confidence_maps_key_name()]

        if self.adjust_vals:
            frame_result = np.clip(frame_result, 0, 1)

        # Determine output scale by comparing original image with model output
        self.output_scale = self.video.height / frame_result.shape[0]

        return frame_result


@attr.s(auto_attribs=True)
class DataOverlay:

    data: Sequence = None
    player: QtVideoPlayer = None
    overlay_class: Union["ConfMapsPlot", "MultiQuiverPlot", None] = None

    def add_to_scene(self, video, frame_idx):
        if self.data is None:
            return

        if self.overlay_class is None:
            return

        img_data = self.data[frame_idx]

        self._add(
            to=self.player.view.scene,
            what=self.overlay_class(img_data, scale=self.data.output_scale),
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
    def from_model(cls, filename, video, **kwargs):
        # Construct the ModelData object that runs inference
        data_object = ModelData(
            predictor=VisualPredictor.from_trained_models(filename), video=video
        )

        # Determine whether to use confmap or paf overlay
        # todo: make this selectable by user for bottom up model w/ both outputs
        from sleap.gui.overlays.confmaps import ConfMapsPlot
        from sleap.gui.overlays.pafs import MultiQuiverPlot

        # todo: support for pafs
        # if model_output_type == ModelOutputType.PART_AFFINITY_FIELD:
        #     overlay_class = MultiQuiverPlot
        # else:
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
