# import sleap
# import tensorflow as tf
# import attr
# from sleap.nn.inference import Predictor
# from typing import Optional, Text, Iterator, Dict, List
# from sleap.nn.data.pipelines import Pipeline, Provider
# from sleap.nn.config import DataConfig

# # from sleap.nn.model import Model
# import numpy as np


# MOVENET_MODELS = {
#     "lightning": {
#         "model_path": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
#         "image_size": 192,
#     },
#     "thunder": {
#         "model_path": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
#         # "model_path": "models/movenet_singlepose_thunder",
#         "image_size": 256,
#     },
# }

# MOVENET_SKELETON = sleap.Skeleton.from_names_and_edge_inds(
#     [
#         "nose",
#         "left_eye",
#         "right_eye",
#         "left_ear",
#         "right_ear",
#         "left_shoulder",
#         "right_shoulder",
#         "left_elbow",
#         "right_elbow",
#         "left_wrist",
#         "right_wrist",
#         "left_hip",
#         "right_hip",
#         "left_knee",
#         "right_knee",
#         "left_ankle",
#         "right_ankle",
#     ],
#     [
#         (10, 8),
#         (8, 6),
#         (6, 5),
#         (5, 7),
#         (7, 9),
#         (6, 12),
#         (5, 11),
#         (12, 14),
#         (14, 16),
#         (11, 13),
#         (13, 15),
#         (4, 2),
#         (2, 0),
#         (0, 1),
#         (1, 3),
#     ],
# )


# def load_movenet_model(model_name: str) -> tf.keras.Model:
#     """Load a MoveNet model by name.

#     Args:
#         model_name: Name of the model ("lightning" or "thunder")

#     Returns:
#         A tf.keras.Model ready for inference.
#     """
#     model_path = MOVENET_MODELS[model_name]["model_path"]
#     image_size = MOVENET_MODELS[model_name]["image_size"]

#     x_in = tf.keras.layers.Input([image_size, image_size, 3], name="image")

#     x = tf.keras.layers.Lambda(
#         lambda x: tf.cast(x, dtype=tf.int32), name="cast_to_int32"
#     )(x_in)
#     layer = hub.KerasLayer(
#         model_path,
#         signature="serving_default",
#         output_key="output_0",
#         name="movenet_layer",
#     )
#     x = layer(x)

#     def split_outputs(x):
#         x_ = tf.reshape(x, [-1, 17, 3])
#         keypoints = tf.gather(x_, [1, 0], axis=-1)
#         keypoints *= image_size
#         scores = tf.squeeze(tf.gather(x_, [2], axis=-1), axis=-1)
#         return keypoints, scores

#     x = tf.keras.layers.Lambda(split_outputs, name="keypoints_and_scores")(x)
#     model = tf.keras.Model(x_in, x)
#     return model


# class MoveNetInferenceLayer(sleap.nn.inference.InferenceLayer):
#     def __init__(self, model_name="lightning"):
#         self.keras_model = load_movenet_model(model_name)
#         self.model_name = model_name
#         self.image_size = MOVENET_MODELS[model_name]["image_size"]
#         super().__init__(
#             keras_model=self.keras_model,
#             input_scale=1.0,
#             pad_to_stride=1,
#             ensure_grayscale=False,
#             ensure_float=False,
#         )

#     def call(self, ex):
#         if type(ex) == dict:
#             img = ex["image"]

#         else:
#             img = ex

#         points, confidences = super().call(img)
#         points = tf.expand_dims(points, axis=1)  # (batch, 1, nodes, 2)
#         confidences = tf.expand_dims(confidences, axis=1)  # (batch, 1, nodes)
#         return {"instance_peaks": points, "confidences": confidences}


# class MoveNetInferenceModel(sleap.nn.inference.InferenceModel):
#     def __init__(self, inference_layer, **kwargs):
#         super().__init__(**kwargs)
#         self.inference_layer = inference_layer

#     @property
#     def model_name(self):
#         return self.inference_layer.model_name

#     @property
#     def image_size(self):
#         return self.inference_layer.image_size

#     def call(self, x):
#         return self.inference_layer(x)


# @attr.s(auto_attribs=True)
# class MoveNetPredictor(Predictor):

#     inference_model: Optional[MoveNetInferenceModel] = attr.ib(default=None)
#     pipeline: Optional[Pipeline] = attr.ib(default=None, init=False)
#     peak_threshold: float = 0.2
#     batch_size: int = 1
#     model_name: str = "lightning"

#     def _initialize_inference_model(self):
#         """Initialize the inference model from the trained model and configuration."""
#         # Force batch size to be 1 since that's what the underlying model expects.
#         self.batch_size = 1
#         self.inference_model = MoveNetInferenceModel(
#             MoveNetInferenceLayer(
#                 model_name=self.model_name,
#             )
#         )

#     @property
#     def data_config(self) -> DataConfig:

#         if self.inference_model is None:
#             self._initialize_inference_model()

#         data_config = DataConfig()
#         data_config.preprocessing.resize_and_pad_to_target = True
#         data_config.preprocessing.target_height = self.inference_model.image_size
#         data_config.preprocessing.target_width = self.inference_model.image_size
#         return data_config

#     @property
#     def is_grayscale(self) -> bool:
#         """Return whether the model expects grayscale inputs."""
#         return False

#     @classmethod
#     def from_trained_models(
#         cls, model_name: Text, peak_threshold: float = 0.2, batch_size: int = 1
#     ) -> "MoveNetPredictor":

#         obj = cls(
#             model_name=model_name,
#             peak_threshold=peak_threshold,
#             batch_size=1,
#         )
#         obj._initialize_inference_model()
#         return obj

#     def _make_labeled_frames_from_generator(
#         self, generator: Iterator[Dict[str, np.ndarray]], data_provider: Provider
#     ) -> List[sleap.LabeledFrame]:

#         skeleton = MOVENET_SKELETON

#         # Loop over batches.
#         predicted_frames = []
#         for ex in generator:

#             # Loop over frames.
#             for video_ind, frame_ind, points, confidences in zip(
#                 ex["video_ind"],
#                 ex["frame_ind"],
#                 ex["instance_peaks"],
#                 ex["confidences"],
#             ):
#                 # Filter out points with low confidences
#                 points[confidences < self.peak_threshold] = np.nan

#                 # Create predicted instances from MoveNet predictions
#                 if np.isnan(points).all():
#                     predicted_instances = []
#                 else:
#                     predicted_instances = [
#                         sleap.instance.PredictedInstance.from_arrays(
#                             points=points[0],  # (nodes, 2)
#                             point_confidences=confidences[0],  # (nodes,)
#                             instance_score=np.nansum(confidences[0]),  # ()
#                             skeleton=skeleton,
#                         )
#                     ]

#                 predicted_frames.append(
#                     sleap.LabeledFrame(
#                         video=data_provider.videos[video_ind],
#                         frame_idx=frame_ind,
#                         instances=predicted_instances,
#                     )
#                 )

#         return predicted_frames
