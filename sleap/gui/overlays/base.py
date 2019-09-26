"""Base class for overlays."""

from PySide2 import QtWidgets

import attr
import numpy as np
from typing import Sequence

from sleap.io.video import Video, HDF5Video
from sleap.gui.video import QtVideoPlayer
from sleap.nn.transform import DataTransform


class HDF5Data(HDF5Video):
    def __getitem__(self, i):
        """Get data for frame i from `HDF5Video` object."""
        x = self.get_frame(i)
        return np.clip(x, 0, 1)


@attr.s(auto_attribs=True)
class ModelData:
    # TODO: Unify this class with inference.Predictor or InferenceModel
    model: "keras.Model"
    video: Video
    do_rescale: bool = False
    output_scale: float = 1.0
    adjust_vals: bool = True

    def __getitem__(self, i):
        """Data data for frame i from predictor."""
        frame_img = self.video[i]

        # Trim to size that works for model
        frame_img = frame_img[
            :, : self.video.height // 8 * 8, : self.video.width // 8 * 8, :
        ]

        inference_transform = DataTransform()
        if self.do_rescale:
            # Scale input image if model trained on scaled images
            frame_img = inference_transform.scale_to(
                imgs=frame_img, target_size=self.model.input_shape[1:3]
            )

        # Get predictions
        frame_result = self.model.predict(frame_img.astype("float32") / 255)
        if self.do_rescale or self.output_scale != 1.0:
            inference_transform.scale *= self.output_scale
            frame_result = inference_transform.invert_scale(frame_result)

        # We just want the single image results
        if type(i) != slice:
            frame_result = frame_result[0]

        if self.adjust_vals:
            # If max value is below 1, amplify values so max is 1.
            # This allows us to visualize model with small ptp value
            # even though this model may not give us adequate predictions.
            max_val = np.max(frame_result)
            if max_val < 1:
                frame_result = frame_result / np.max(frame_result)

            # Clip values to ensure that they're within [0, 1]
            frame_result = np.clip(frame_result, 0, 1)

        return frame_result


@attr.s(auto_attribs=True)
class DataOverlay:

    data: Sequence = None
    player: QtVideoPlayer = None
    overlay_class: QtWidgets.QGraphicsObject = None
    transform: DataTransform = None

    def add_to_scene(self, video, frame_idx):
        if self.data is None:
            return

        # Check if video matches video for ModelData object
        if hasattr(self.data, "video") and self.data.video != video:
            video_shape = (video.height, video.width, video.channels)
            prior_shape = (
                self.data.video.height,
                self.data.video.width,
                self.data.video.channels,
            )
            # Check if the videos are both compatible with the loaded model
            if video_shape == prior_shape:
                # Shapes match so we can apply model to this video
                self.data.video = video
            else:
                # Shapes don't match so don't do anything with this video
                return

        if self.transform is None:
            self._add(self.player.view.scene, self.overlay_class(self.data[frame_idx]))

        else:
            # If data indices are different than frame indices, use data
            # index; otherwise just use frame index.
            idxs = (
                self.transform.get_data_idxs(frame_idx)
                if self.transform.frame_idxs
                else [frame_idx]
            )

            # Loop over indices, in case there's more than one for frame
            for idx in idxs:
                if idx in self.transform.bounding_boxes:
                    x, y, *_ = self.transform.bounding_boxes[idx]
                else:
                    x, y = 0, 0

                overlay_object = self.overlay_class(
                    self.data[idx], scale=self.transform.scale
                )

                self._add(self.player.view.scene, overlay_object, (x, y))

    def _add(
        self,
        to: QtWidgets.QGraphicsScene,
        what: QtWidgets.QGraphicsObject,
        where: tuple = (0, 0),
    ):
        to.addItem(what)
        what.setPos(*where)

    @classmethod
    def from_h5(cls, filename, dataset, input_format="channels_last", **kwargs):
        import h5py as h5

        with h5.File(filename, "r") as f:
            frame_idxs = np.asarray(f["frame_idxs"], dtype="int")
            bounding_boxes = np.asarray(f["bounds"])

        transform = DataTransform(frame_idxs=frame_idxs, bounding_boxes=bounding_boxes)

        data_object = HDF5Data(
            filename, dataset, input_format=input_format, convert_range=False
        )

        return cls(data=data_object, transform=transform, **kwargs)

    @classmethod
    def from_model(cls, filename, video, **kwargs):
        from sleap.nn.model import ModelOutputType
        from sleap.nn.loadmodel import load_model, get_model_data
        from sleap.nn.training import TrainingJob

        # Load the trained model

        trainingjob = TrainingJob.load_json(filename)

        input_size = (video.height // 8 * 8, video.width // 8 * 8, video.channels)
        model_output_type = trainingjob.model.output_type

        model = load_model(
            sleap_models={model_output_type: trainingjob},
            input_size=input_size,
            output_types=[model_output_type],
        )

        model_data = get_model_data(
            sleap_models={model_output_type: trainingjob},
            output_types=[model_output_type],
        )

        # Here we determine if the input should be scaled. If so, then
        # the output of the model will also be rescaled accordingly.

        do_rescale = model_data["scale"] < 1

        # Determine how the output from the model should be scaled
        img_output_scale = 1.0  # image rescaling
        obj_output_scale = 1.0  # scale to pass to overlay object

        if model_output_type == ModelOutputType.PART_AFFINITY_FIELD:
            obj_output_scale = model_data["multiscale"]
        else:
            img_output_scale = model_data["multiscale"]

        # Construct the ModelData object that runs inference

        data_object = ModelData(
            model, video, do_rescale=do_rescale, output_scale=img_output_scale
        )

        # Determine whether to use confmap or paf overlay

        from sleap.gui.overlays.confmaps import ConfMapsPlot
        from sleap.gui.overlays.pafs import MultiQuiverPlot

        if model_output_type == ModelOutputType.PART_AFFINITY_FIELD:
            overlay_class = MultiQuiverPlot
        else:
            overlay_class = ConfMapsPlot

        # We use the transform scale for *multiscale* models
        # with full-scale input and lower-scale output.
        # This doesn't require rescaling the input, and the "scale"
        # will be passed to the overlay object to do its own upscaling
        # (at least for pafs).

        transform = DataTransform(scale=obj_output_scale)

        return cls(
            data=data_object, transform=transform, overlay_class=overlay_class, **kwargs
        )


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
