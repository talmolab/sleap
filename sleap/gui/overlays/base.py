"""Base class for overlays."""

from PySide2 import QtWidgets

import attr
import numpy as np
from typing import Any, Sequence, Union

import sleap
from sleap.io.video import Video
from sleap.gui.video import QtVideoPlayer


@attr.s(auto_attribs=True)
class ModelData:
    inference_object: Union[
        "sleap.nn.peak_finding.ConfmapPeakFinder", "sleap.nn.paf_grouping.PAFGrouper"
    ]
    video: Video
    do_rescale: bool = False
    output_scale: float = 1.0
    adjust_vals: bool = True

    def __getitem__(self, i):
        """Data data for frame i from predictor."""
        frame_img = self.video[i]

        frame_result = self.inference_object.inference(
            self.inference_object.preproc(frame_img)
        ).numpy()

        # We just want the single image results
        if type(i) != slice:
            frame_result = frame_result[0]

        if self.adjust_vals:
            frame_result = np.clip(frame_result, 0, 1)

        return frame_result


@attr.s(auto_attribs=True)
class DataOverlay:

    data: Sequence = None
    player: QtVideoPlayer = None
    overlay_class: QtWidgets.QGraphicsObject = None

    def add_to_scene(self, video, frame_idx):
        if self.data is None:
            return

        img_data = self.data[frame_idx]
        # print(img_data.shape, np.ptp(img_data))
        self._add(
            self.player.view.scene,
            self.overlay_class(img_data, scale=1.0 / self.data.output_scale),
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
        from sleap.nn.model import ModelOutputType, InferenceModel
        from sleap.nn import job

        # Load the trained model
        trained_job = job.TrainingJob.load_json(filename)
        inference_model = InferenceModel.from_training_job(trained_job)
        model_output_type = trained_job.model.output_type

        if trained_job.model.output_type == ModelOutputType.PART_AFFINITY_FIELD:
            from sleap.nn import paf_grouping

            inference_object = paf_grouping.PAFGrouper(inference_model=inference_model)
        else:
            from sleap.nn import peak_finding

            inference_object = peak_finding.ConfmapPeakFinder(
                inference_model=inference_model
            )

        # Construct the ModelData object that runs inference
        data_object = ModelData(
            inference_object, video, output_scale=inference_model.output_scale
        )

        # Determine whether to use confmap or paf overlay
        from sleap.gui.overlays.confmaps import ConfMapsPlot
        from sleap.gui.overlays.pafs import MultiQuiverPlot

        if model_output_type == ModelOutputType.PART_AFFINITY_FIELD:
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
