"""This module defines high level pipeline configurations from providers/transformers.

The `Pipeline` class has the capability to create sequences of data I/O and processing
operations wrapped in a `tf.data`-based pipeline.

This allows for convenient ways to configure individual variants of common pipelines, as
well as to define training vs inference versions based on the same configurations.
"""

import tensorflow as tf
import numpy as np
import attr
from typing import Sequence, Text, Optional, List, Tuple, Union, TypeVar, Dict

import sleap
from sleap.nn.data.providers import LabelsReader, VideoReader
from sleap.nn.data.augmentation import AugmentationConfig, ImgaugAugmenter
from sleap.nn.data.normalization import Normalizer
from sleap.nn.data.resizing import Resizer, PointsRescaler
from sleap.nn.data.instance_centroids import InstanceCentroidFinder
from sleap.nn.data.instance_cropping import InstanceCropper, PredictedInstanceCropper
from sleap.nn.data.confidence_maps import (
    MultiConfidenceMapGenerator,
    InstanceConfidenceMapGenerator,
    SingleInstanceConfidenceMapGenerator,
)
from sleap.nn.data.edge_maps import PartAffinityFieldsGenerator
from sleap.nn.data.dataset_ops import (
    Shuffler,
    Batcher,
    Unbatcher,
    Repeater,
    Prefetcher,
    Preloader,
    LambdaFilter,
)
from sleap.nn.data.training import KeyMapper
from sleap.nn.data.general import KeyFilter, KeyRenamer, KeyDeviceMover
from sleap.nn.data.inference import (
    KerasModelPredictor,
    GlobalPeakFinder,
    MockGlobalPeakFinder,
    LocalPeakFinder,
    PredictedCenterInstanceNormalizer,
)
from sleap.nn.paf_grouping import PartAffinityFieldInstanceGrouper
from sleap.nn.data.utils import ensure_list

from sleap.nn.config import DataConfig, OptimizationConfig
from sleap.nn.heads import (
    MultiInstanceConfmapsHead,
    PartAffinityFieldsHead,
    CentroidConfmapsHead,
    CenteredInstanceConfmapsHead,
    SingleInstanceConfmapsHead,
)


PROVIDERS = (LabelsReader, VideoReader)
TRANSFORMERS = (
    ImgaugAugmenter,
    Normalizer,
    Resizer,
    InstanceCentroidFinder,
    InstanceCropper,
    MultiConfidenceMapGenerator,
    InstanceConfidenceMapGenerator,
    PartAffinityFieldsGenerator,
    SingleInstanceConfidenceMapGenerator,
    Shuffler,
    Batcher,
    Unbatcher,
    Repeater,
    Prefetcher,
    Preloader,
    LambdaFilter,
    KeyMapper,
    KerasModelPredictor,
    GlobalPeakFinder,
    MockGlobalPeakFinder,
    LocalPeakFinder,
    PredictedInstanceCropper,
    PredictedCenterInstanceNormalizer,
    KeyFilter,
    KeyRenamer,
    KeyDeviceMover,
    PartAffinityFieldInstanceGrouper,
    PointsRescaler,
)
Provider = TypeVar("Provider", *PROVIDERS)
Transformer = TypeVar("Transformer", *TRANSFORMERS)


@attr.s(auto_attribs=True)
class Pipeline:
    """Pipeline composed of providers and transformers.

    Attributes:
        providers: A single or a list of data providers.
        transformers: A single or a list of transformers.
    """

    providers: List[Provider] = attr.ib(converter=ensure_list, factory=list)
    transformers: List[Transformer] = attr.ib(converter=ensure_list, factory=list)

    @classmethod
    def from_blocks(
        cls,
        blocks: Union[
            Union[Provider, Transformer], Sequence[Union[Provider, Transformer]]
        ],
    ) -> "Pipeline":
        """Create a pipeline from a sequence of providers and transformers.

        Args:
            sequence: List or tuple of providers and transformer instances.

        Returns:
            An instantiated pipeline with all blocks chained.
        """
        if isinstance(blocks, PROVIDERS + TRANSFORMERS):
            blocks = [blocks]
        providers = []
        transformers = []
        for i, block in enumerate(blocks):
            if isinstance(block, PROVIDERS):
                providers.append(block)
            elif isinstance(block, TRANSFORMERS):
                transformers.append(block)
            else:
                raise ValueError(
                    f"Unrecognized pipeline block type (index = {i}): {type(block)}"
                )
        return cls(providers=providers, transformers=transformers)

    @classmethod
    def from_pipelines(cls, pipelines: Sequence["Pipeline"]) -> "Pipeline":
        """Create a new pipeline instance by chaining together multiple pipelines.

        Args:
            pipelines: A sequence of `Pipeline` instances.

        Returns:
            A new `Pipeline` instance formed by concatenating the individual pipelines.
        """
        blocks = []
        for pipeline in pipelines:
            if isinstance(pipeline, PROVIDERS + TRANSFORMERS):
                pipeline = cls.from_blocks(pipeline)
            blocks.extend(pipeline.providers)
            blocks.extend(pipeline.transformers)
        return cls.from_blocks(blocks)

    def __add__(self, other: "Pipeline") -> "Pipeline":
        """Overload for + operator concatenation."""
        return self.from_pipelines([self, other])

    def __or__(self, other: "Pipeline") -> "Pipeline":
        """Overload for | operator concatenation."""
        return self.from_pipelines([self, other])

    def append(self, other: Union["Pipeline", Transformer, List[Transformer]]):
        """Append one or more blocks to this pipeline instance.

        Args:
            other: A single `Pipeline`, `Transformer` or list of `Transformer`s to
                append to the end of this pipeline.

        Raises:
            ValueError: If blocks provided are not a `Pipeline`, `Transformer` or list
                of `Transformer`s.
        """
        if isinstance(other, TRANSFORMERS):
            self.transformers.append(other)
        elif isinstance(other, list):
            if all(isinstance(block, TRANSFORMERS) for block in other):
                self.transformers.extend(other)
            else:
                raise ValueError(
                    "Cannot append blocks that are not pipelines or transformers."
                )
        elif hasattr(other, "providers") and hasattr(other, "transformers"):
            self.providers.extend(other.providers)
            self.transformers.extend(other.transformers)
        else:
            raise ValueError(
                "Cannot append blocks that are not pipelines or transformers."
            )

    def __iadd__(self, other: Union["Pipeline", Transformer, List[Transformer]]):
        """Overload for += for appending blocks to existing instance."""
        self.append(other)
        return self

    def __ior__(self, other: Union["Pipeline", Transformer]):
        """Overload for |= for appending blocks to existing instance."""
        self.append(other)
        return self

    def validate_pipeline(self) -> List[Text]:
        """Check that all pipeline blocks meet the data requirements.

        Returns:
            The final keys that will be present in each example.

        Raises:
            ValueError: If keys required for a block are dropped at some point in the
                pipeline.
        """
        example_keys = []
        for provider in self.providers:
            example_keys.extend(provider.output_keys)

        for i, transformer in enumerate(self.transformers):
            # Required keys that are in the example:
            input_keys_in_example = list(
                set(example_keys) & set(transformer.input_keys)
            )

            # Required keys that are missing from the example:
            input_keys_not_in_example = list(
                set(transformer.input_keys) - set(example_keys)
            )

            # Keys in the example that are not required by transformer:
            extra_example_keys = list(set(example_keys) - set(transformer.output_keys))

            # Keys that the transformer will output:
            output_keys = transformer.output_keys

            # Check that all the required inputs are in the example.
            if len(input_keys_not_in_example) > 0:
                raise ValueError(
                    f"Missing required keys for transformer (index = {i}, "
                    f"type = {type(transformer)}): {input_keys_not_in_example}.\n"
                    f"Available: {extra_example_keys}"
                )

            # The new example keys will be the outputs of the transformer and the
            # previous extraneous keys.
            example_keys = output_keys + extra_example_keys

        return example_keys

    @property
    def output_keys(self) -> List[Text]:
        """Return the keys in examples from a dataset generated from this pipeline."""
        return self.validate_pipeline()

    def make_dataset(self) -> tf.data.Dataset:
        """Create a dataset instance that generates examples from the pipeline.

        Returns:
            The instantiated `tf.data.Dataset` pipeline that generates examples with the
            keys in the `output_keys` attribute.
        """
        # Check that the pipeline can be instantiated.
        self.validate_pipeline()

        # Create providers.
        # TODO: Multi-provider pipelines by merging the example dictionaries.
        #       Need something like an optional side-packet into in providers. Or a
        #       transformer that just merges all the keys after a Dataset.zip?
        ds = self.providers[0].make_dataset()

        # Apply transformers.
        for transformer in self.transformers:
            ds = transformer.transform_dataset(ds)

        return ds

    def run(self) -> List[Dict[Text, tf.Tensor]]:
        """Build and evaluate the pipeline.

        Returns:
            List of example dictionaries after processing the pipeline.
        """
        return list(self.make_dataset())


@attr.s(auto_attribs=True)
class BottomUpPipeline:
    """Pipeline builder for confidence maps + part affinity fields models.

    Attributes:
        data_config: Data-related configuration.
        optimization_config: Optimization-related configuration.
        confmaps_head: Instantiated head describing the output confidence maps tensor.
        pafs_head: Instantiated head describing the output PAFs tensor.
    """

    data_config: DataConfig
    optimization_config: OptimizationConfig
    confmaps_head: MultiInstanceConfmapsHead
    pafs_head: PartAffinityFieldsHead

    def make_base_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create base pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce input examples.
        """
        pipeline = Pipeline(providers=data_provider)
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)
        return pipeline

    def make_training_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create full training pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce all data keys required for
            training.

        Notes:
            This does not remap keys to model outputs. Use `KeyMapper` to pull out keys
            with the appropriate format for the instantiated `tf.keras.Model`.
        """
        pipeline = Pipeline(providers=data_provider)

        if self.optimization_config.preload_data:
            pipeline += Preloader()

        if self.optimization_config.online_shuffling:
            pipeline += Shuffler(self.optimization_config.shuffle_buffer_size)

        pipeline += ImgaugAugmenter.from_config(
            self.optimization_config.augmentation_config
        )
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)

        pipeline += MultiConfidenceMapGenerator(
            sigma=self.confmaps_head.sigma,
            output_stride=self.confmaps_head.output_stride,
            centroids=False,
        )
        pipeline += PartAffinityFieldsGenerator(
            sigma=self.pafs_head.sigma,
            output_stride=self.pafs_head.output_stride,
            skeletons=self.data_config.labels.skeletons,
            flatten_channels=True,
        )

        if len(data_provider) >= self.optimization_config.batch_size:
            # Batching before repeating is preferred since it preserves epoch boundaries
            # such that no sample is repeated within the epoch. But this breaks if there
            # are fewer samples than the batch size.
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )
            pipeline += Repeater()

        else:
            pipeline += Repeater()
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )

        if self.optimization_config.prefetch:
            pipeline += Prefetcher()

        return pipeline

    def make_viz_pipeline(
        self, data_provider: Provider, keras_model: tf.keras.Model
    ) -> Pipeline:
        """Create visualization pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            keras_model: A `tf.keras.Model` that can be used for inference.

        Returns:
            A `Pipeline` instance configured to fetch data and run inference to generate
            predictions useful for visualization during training.
        """
        pipeline = self.make_base_pipeline(data_provider=data_provider)
        pipeline += Prefetcher()
        pipeline += Repeater()
        pipeline += KerasModelPredictor(
            keras_model=keras_model,
            model_input_keys="image",
            model_output_keys=[
                "predicted_confidence_maps",
                "predicted_part_affinity_fields",
            ],
        )
        pipeline += LocalPeakFinder(
            confmaps_stride=self.confmaps_head.output_stride,
            peak_threshold=0.2,
            confmaps_key="predicted_confidence_maps",
            peaks_key="predicted_peaks",
            peak_vals_key="predicted_peak_confidences",
            peak_sample_inds_key="predicted_peak_sample_inds",
            peak_channel_inds_key="predicted_peak_channel_inds",
        )
        # TODO: PAF grouping inference
        return pipeline


@attr.s(auto_attribs=True)
class CentroidConfmapsPipeline:
    """Pipeline builder for centroid confidence map models.

    Attributes:
        data_config: Data-related configuration.
        optimization_config: Optimization-related configuration.
        centroid_confmap_head: Instantiated head describing the output centroid
            confidence maps tensor.
    """

    data_config: DataConfig
    optimization_config: OptimizationConfig
    centroid_confmap_head: CentroidConfmapsHead

    def make_base_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create base pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce input examples.
        """
        pipeline = Pipeline(providers=data_provider)
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)
        pipeline += InstanceCentroidFinder.from_config(
            self.data_config.instance_cropping,
            skeletons=self.data_config.labels.skeletons,
        )
        return pipeline

    def make_training_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create full training pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce all data keys required for
            training.

        Notes:
            This does not remap keys to model outputs. Use `KeyMapper` to pull out keys
            with the appropriate format for the instantiated `tf.keras.Model`.
        """
        pipeline = Pipeline(providers=data_provider)

        if self.optimization_config.preload_data:
            pipeline += Preloader()

        if self.optimization_config.online_shuffling:
            pipeline += Shuffler(self.optimization_config.shuffle_buffer_size)

        pipeline += ImgaugAugmenter.from_config(
            self.optimization_config.augmentation_config
        )
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)

        pipeline += InstanceCentroidFinder.from_config(
            self.data_config.instance_cropping,
            skeletons=self.data_config.labels.skeletons,
        )
        pipeline += MultiConfidenceMapGenerator(
            sigma=self.centroid_confmap_head.sigma,
            output_stride=self.centroid_confmap_head.output_stride,
            centroids=True,
        )

        if len(data_provider) >= self.optimization_config.batch_size:
            # Batching before repeating is preferred since it preserves epoch boundaries
            # such that no sample is repeated within the epoch. But this breaks if there
            # are fewer samples than the batch size.
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )
            pipeline += Repeater()

        else:
            pipeline += Repeater()
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )

        if self.optimization_config.prefetch:
            pipeline += Prefetcher()

        return pipeline

    def make_viz_pipeline(
        self, data_provider: Provider, keras_model: tf.keras.Model
    ) -> Pipeline:
        """Create visualization pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            keras_model: A `tf.keras.Model` that can be used for inference.

        Returns:
            A `Pipeline` instance configured to fetch data and run inference to generate
            predictions useful for visualization during training.
        """
        pipeline = self.make_base_pipeline(data_provider=data_provider)
        pipeline += Prefetcher()
        pipeline += Repeater()
        pipeline += KerasModelPredictor(
            keras_model=keras_model,
            model_input_keys="image",
            model_output_keys="predicted_centroid_confidence_maps",
        )
        pipeline += LocalPeakFinder(
            confmaps_stride=self.centroid_confmap_head.output_stride,
            peak_threshold=0.2,
            confmaps_key="predicted_centroid_confidence_maps",
            peaks_key="predicted_centroids",
            peak_vals_key="predicted_centroid_confidences",
            peak_sample_inds_key="predicted_centroid_sample_inds",
            peak_channel_inds_key="predicted_centroid_channel_inds",
        )
        return pipeline


@attr.s(auto_attribs=True)
class TopdownConfmapsPipeline:
    """Pipeline builder for instance-centered confidence map models.

    Attributes:
        data_config: Data-related configuration.
        optimization_config: Optimization-related configuration.
        instance_confmap_head: Instantiated head describing the output centered
            confidence maps tensor.
    """

    data_config: DataConfig
    optimization_config: OptimizationConfig
    instance_confmap_head: CenteredInstanceConfmapsHead

    def make_base_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create base pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce input examples.
        """
        pipeline = Pipeline(providers=data_provider)
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)
        pipeline += InstanceCentroidFinder.from_config(
            self.data_config.instance_cropping,
            skeletons=self.data_config.labels.skeletons,
        )
        pipeline += InstanceCropper.from_config(self.data_config.instance_cropping)
        return pipeline

    def make_training_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create full training pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce all data keys required for
            training.

        Notes:
            This does not remap keys to model outputs. Use `KeyMapper` to pull out keys
            with the appropriate format for the instantiated `tf.keras.Model`.
        """
        pipeline = Pipeline(providers=data_provider)

        if self.optimization_config.preload_data:
            pipeline += Preloader()

        if self.optimization_config.online_shuffling:
            pipeline += Shuffler(self.optimization_config.shuffle_buffer_size)

        pipeline += ImgaugAugmenter.from_config(
            self.optimization_config.augmentation_config
        )
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)

        pipeline += InstanceCentroidFinder.from_config(
            self.data_config.instance_cropping,
            skeletons=self.data_config.labels.skeletons,
        )
        pipeline += InstanceCropper.from_config(self.data_config.instance_cropping)
        pipeline += InstanceConfidenceMapGenerator(
            sigma=self.instance_confmap_head.sigma,
            output_stride=self.instance_confmap_head.output_stride,
            all_instances=False,
        )

        if len(data_provider) >= self.optimization_config.batch_size:
            # Batching before repeating is preferred since it preserves epoch boundaries
            # such that no sample is repeated within the epoch. But this breaks if there
            # are fewer samples than the batch size.
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )
            pipeline += Repeater()

        else:
            pipeline += Repeater()
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )

        if self.optimization_config.prefetch:
            pipeline += Prefetcher()

        return pipeline

    def make_viz_pipeline(
        self, data_provider: Provider, keras_model: tf.keras.Model
    ) -> Pipeline:
        """Create visualization pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            keras_model: A `tf.keras.Model` that can be used for inference.

        Returns:
            A `Pipeline` instance configured to fetch data and run inference to generate
            predictions useful for visualization during training.
        """
        pipeline = self.make_base_pipeline(data_provider=data_provider)
        pipeline += Prefetcher()
        pipeline += Repeater()
        pipeline += KerasModelPredictor(
            keras_model=keras_model,
            model_input_keys="instance_image",
            model_output_keys="predicted_instance_confidence_maps",
        )
        pipeline += GlobalPeakFinder(
            confmaps_key="predicted_instance_confidence_maps",
            peaks_key="predicted_center_instance_points",
            confmaps_stride=self.instance_confmap_head.output_stride,
            peak_threshold=0.2,
        )
        return pipeline


@attr.s(auto_attribs=True)
class SingleInstanceConfmapsPipeline:
    """Pipeline builder for single-instance confidence map models.

    Attributes:
        data_config: Data-related configuration.
        optimization_config: Optimization-related configuration.
        single_instance_confmap_head: Instantiated head describing the output confidence
            maps tensor.
    """

    data_config: DataConfig
    optimization_config: OptimizationConfig
    single_instance_confmap_head: SingleInstanceConfmapsHead

    def make_base_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create base pipeline with input data only.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce input examples.
        """
        pipeline = Pipeline(providers=data_provider)
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)
        return pipeline

    def make_training_pipeline(self, data_provider: Provider) -> Pipeline:
        """Create full training pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.

        Returns:
            A `Pipeline` instance configured to produce all data keys required for
            training.

        Notes:
            This does not remap keys to model outputs. Use `KeyMapper` to pull out keys
            with the appropriate format for the instantiated `tf.keras.Model`.
        """
        pipeline = Pipeline(providers=data_provider)

        if self.optimization_config.preload_data:
            pipeline += Preloader()

        if self.optimization_config.online_shuffling:
            pipeline += Shuffler(self.optimization_config.shuffle_buffer_size)

        pipeline += ImgaugAugmenter.from_config(
            self.optimization_config.augmentation_config
        )
        pipeline += Normalizer.from_config(self.data_config.preprocessing)
        pipeline += Resizer.from_config(self.data_config.preprocessing)

        pipeline += SingleInstanceConfidenceMapGenerator(
            sigma=self.single_instance_confmap_head.sigma,
            output_stride=self.single_instance_confmap_head.output_stride,
        )

        if len(data_provider) >= self.optimization_config.batch_size:
            # Batching before repeating is preferred since it preserves epoch boundaries
            # such that no sample is repeated within the epoch. But this breaks if there
            # are fewer samples than the batch size.
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )
            pipeline += Repeater()

        else:
            pipeline += Repeater()
            pipeline += Batcher(
                batch_size=self.optimization_config.batch_size, drop_remainder=True
            )

        if self.optimization_config.prefetch:
            pipeline += Prefetcher()

        return pipeline

    def make_viz_pipeline(
        self, data_provider: Provider, keras_model: tf.keras.Model
    ) -> Pipeline:
        """Create visualization pipeline.

        Args:
            data_provider: A `Provider` that generates data examples, typically a
                `LabelsReader` instance.
            keras_model: A `tf.keras.Model` that can be used for inference.

        Returns:
            A `Pipeline` instance configured to fetch data and run inference to generate
            predictions useful for visualization during training.
        """
        pipeline = self.make_base_pipeline(data_provider=data_provider)
        pipeline += Prefetcher()
        pipeline += Repeater()
        pipeline += KerasModelPredictor(
            keras_model=keras_model,
            model_input_keys="image",
            model_output_keys="predicted_confidence_maps",
        )
        pipeline += GlobalPeakFinder(
            confmaps_key="predicted_confidence_maps",
            peaks_key="predicted_points",
            peak_vals_key="predicted_confidences",
            confmaps_stride=self.single_instance_confmap_head.output_stride,
            peak_threshold=0.2,
        )
        return pipeline
