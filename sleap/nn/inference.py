"""Inference pipelines and utilities."""

import os
from collections import defaultdict

import tensorflow as tf

import attr
from typing import Text

from sleap.nn.config import TrainingJobConfig
from sleap.nn.model import Model
from sleap.nn.data.pipelines import (
    Provider,
    Pipeline,
    LabelsReader,
    VideoReader,
    Normalizer,
    Resizer,
    Prefetcher,
    KerasModelPredictor,
    LocalPeakFinder,
    PredictedInstanceCropper,
    GlobalPeakFinder,
    KeyFilter,
    PredictedCenterInstanceNormalizer,
    PartAffinityFieldInstanceGrouper,
)


def group_examples(examples):
    grouped_examples = defaultdict(list)
    for example in examples:
        video_ind = example["video_ind"].numpy()
        frame_ind = example["frame_ind"].numpy()
        grouped_examples[(video_ind, frame_ind)].append(example)
    return grouped_examples


@attr.s(auto_attribs=True)
class TopdownPredictor:
    centroid_config: TrainingJobConfig
    centroid_model: Model
    confmap_config: TrainingJobConfig
    confmap_model: Model

    @classmethod
    def from_trained_models(
        cls, centroid_model_path: Text, confmap_model_path: Text
    ) -> "TopdownPredictor":
        """Create predictor from saved models."""
        # Load centroid model.
        centroid_config = TrainingJobConfig.load_json(centroid_model_path)
        centroid_keras_model_path = os.path.join(centroid_model_path, "best_model.h5")
        centroid_model = Model.from_config(centroid_config.model)
        centroid_model.keras_model = tf.keras.models.load_model(
            centroid_keras_model_path, compile=False
        )

        # Load confmap model.
        confmap_config = TrainingJobConfig.load_json(confmap_model_path)
        confmap_keras_model_path = os.path.join(confmap_model_path, "best_model.h5")
        confmap_model = Model.from_config(confmap_config.model)
        confmap_model.keras_model = tf.keras.models.load_model(
            confmap_keras_model_path, compile=False
        )

        return cls(
            centroid_config=centroid_config,
            centroid_model=centroid_model,
            confmap_config=confmap_config,
            confmap_model=confmap_model,
        )

    def make_pipeline(self, data_provider: Provider) -> Pipeline:
        pipeline = Pipeline(providers=data_provider)
        pipeline += Normalizer.from_config(self.centroid_config.data.preprocessing)
        pipeline += Resizer.from_config(
            self.centroid_config.data.preprocessing,
            keep_full_image=True,
            points_key=None,
        )

        pipeline += Prefetcher()

        pipeline += KerasModelPredictor(
            keras_model=self.centroid_model.keras_model,
            model_input_keys="image",
            model_output_keys="predicted_centroid_confidence_maps",
        )

        pipeline += LocalPeakFinder(
            confmaps_stride=self.centroid_model.heads[0].output_stride,
            peak_threshold=0.2,
            confmaps_key="predicted_centroid_confidence_maps",
            peaks_key="predicted_centroids",
            peak_vals_key="predicted_centroid_confidences",
            peak_sample_inds_key="predicted_centroid_sample_inds",
            peak_channel_inds_key="predicted_centroid_channel_inds",
            keep_confmaps=False,
        )

        pipeline += PredictedInstanceCropper(
            crop_width=self.confmap_config.data.instance_cropping.crop_size,
            crop_height=self.confmap_config.data.instance_cropping.crop_size,
            centroids_key="predicted_centroids",
            full_image_key="full_image",
        )

        pipeline += KerasModelPredictor(
            keras_model=self.confmap_model.keras_model,
            model_input_keys="instance_image",
            model_output_keys="predicted_instance_confidence_maps",
        )
        pipeline += GlobalPeakFinder(
            confmaps_key="predicted_instance_confidence_maps",
            peaks_key="predicted_center_instance_points",
            confmaps_stride=self.confmap_model.heads[0].output_stride,
            peak_threshold=0.2,
        )

        pipeline += KeyFilter(
            keep_keys=[
                "bbox",
                "center_instance_ind",
                "centroid",
                "scale",
                "video_ind",
                "frame_ind",
                "center_instance_ind",
                "predicted_center_instance_points",
                "predicted_center_instance_confidences",
            ]
        )

        pipeline += PredictedCenterInstanceNormalizer(
            centroids_key="centroid",
            peaks_key="predicted_center_instance_points",
            new_centroid_key="predicted_centroid",
            new_peaks_key="predicted_instance",
        )

        return pipeline
