"""This module defines high level pipeline configurations from providers/transformers.

The `Pipeline` class has the capability to create sequences of data I/O and processing
operations wrapped in a `tf.data`-based pipeline.

This allows for convenient ways to configure individual variants of common pipelines, as
well as to define training vs inference versions based on the same configurations.
"""

import tensorflow as tf
import numpy as np
import attr
from typing import Sequence, Text, Optional, List, Tuple, Union, TypeVar


import sleap
from sleap.nn.data.providers import LabelsReader, VideoReader
from sleap.nn.data.augmentation import AugmentationConfig, ImgaugAugmenter
from sleap.nn.data.normalization import Normalizer
from sleap.nn.data.resizing import Resizer
from sleap.nn.data.instance_centroids import InstanceCentroidFinder
from sleap.nn.data.instance_cropping import InstanceCropper
from sleap.nn.data.confidence_maps import (
    MultiConfidenceMapGenerator,
    InstanceConfidenceMapGenerator,
)
from sleap.nn.data.edge_maps import PartAffinityFieldsGenerator
from sleap.nn.data.dataset_ops import Shuffler, Batcher, Repeater, Prefetcher, Preloader
from sleap.nn.data.utils import ensure_list


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
    Shuffler,
    Batcher,
    Repeater,
    Prefetcher,
    Preloader,
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

    providers: List[Provider] = attr.ib(converter=ensure_list)
    transformers: List[Transformer] = attr.ib(converter=ensure_list)

    @classmethod
    def from_sequence(
        cls, sequence: Sequence[Union[Provider, Transformer]]
    ) -> "Pipeline":
        """Create a pipeline from a sequence of providers and transformers.

        Args:
            sequence: List or tuple of providers and transformer instances.

        Returns:
            An instantiated pipeline with all blocks chained.
        """
        providers = []
        transformers = []
        for i, block in enumerate(sequence):
            if isinstance(block, PROVIDERS):
                providers.append(block)
            elif isinstance(block, TRANSFORMERS):
                transformers.append(block)
            else:
                raise ValueError(
                    f"Unrecognized pipeline block type (index = {i}): {type(block)}"
                )
        return cls(providers=providers, transformers=transformers)

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


@attr.s(auto_attribs=True)
class BottomUpPipeline:
    """Standard bottom up pipeline."""

    data_provider: Provider
    shuffler: Shuffler
    augmenter: ImgaugAugmenter
    normalizer: Normalizer
    resizer: Resizer
    multi_confmap_generator: MultiConfidenceMapGenerator
    # multi_confmap_predictor: MultiConfidenceMapPredictor
    paf_generator: PartAffinityFieldsGenerator
    # paf_matcher: PAFMatcher
    batcher: Batcher
    repeater: Repeater
    prefetcher: Prefetcher

    def make_training_pipeline(self) -> Pipeline:
        """Make training pipeline."""
        return Pipeline.from_sequence(
            [
                self.data_provider,
                Preloader(),
                self.shuffler,
                self.augmenter,
                self.normalizer,
                self.resizer,
                self.multi_confmap_generator,
                self.paf_generator,
                self.batcher,
                self.repeater,
                self.prefetcher,
            ]
        )

    def make_training_dataset(self) -> tf.data.Dataset:
        """Instantiate the training pipeline to create a dataset."""
        return self.make_training_pipeline().make_dataset()

    def make_inference_pipeline(self) -> Pipeline:
        """Make inference pipeline."""
        return Pipeline.from_sequence(
            [
                self.data_provider,
                self.normalizer,
                self.resizer,
                self.batcher,
                self.prefetcher,
                # TODO:
                # self.multi_confmap_predictor,
                # self.paf_matcher,
            ]
        )

    def make_inference_dataset(self) -> tf.data.Dataset:
        """Instantiate the inference pipeline to create a dataset."""
        return self.make_inference_pipeline().make_dataset()


@attr.s(auto_attribs=True)
class TopDownPipeline:
    """Standard top down pipeline."""

    data_provider: Provider
    shuffler: Shuffler
    augmenter: ImgaugAugmenter
    normalizer: Normalizer
    resizer: Resizer
    centroid_finder: InstanceCentroidFinder
    # centroid_confmap_generator: MultiConfidenceMapGenerator
    # centroid_predictor: InstanceCentroidPredictor
    instance_cropper: InstanceCropper
    instance_confmap_generator: InstanceConfidenceMapGenerator
    # instance_confmap_predictor: InstanceConfidenceMapPredictor
    batcher: Batcher
    repeater: Repeater
    prefetcher: Prefetcher

    def make_training_pipeline(self, centroid: bool = False) -> Pipeline:
        """Make training pipeline."""
        if centroid:
            middle_blocks = [self.centroid_confmap_generator]
        else:
            middle_blocks = [self.instance_cropper, self.instance_confmap_generator]
        return Pipeline.from_sequence(
            [
                self.data_provider,
                Preloader(),
                self.shuffler,
                self.augmenter,
                self.normalizer,
                self.resizer,
                self.centroid_finder,
            ]
            + middle_blocks
            + [
            self.batcher,
            self.prefetcher,
            self.repeater
            ]
        )

    def make_training_dataset(self) -> tf.data.Dataset:
        """Instantiate the training pipeline to create a dataset."""
        return self.make_training_pipeline().make_dataset()

    def make_inference_pipeline(self) -> Pipeline:
        """Make inference pipeline."""
        return Pipeline.from_sequence(
            [
                self.data_provider,
                self.normalizer,
                self.resizer,
                self.batcher,
                self.prefetcher,
                # TODO:
                # self.centroid_predictor,
                self.instance_cropper,  # TODO: update to work with batches as well
                # self.single_part_predictor,
            ]
        )

    def make_inference_dataset(self) -> tf.data.Dataset:
        """Instantiate the inference pipeline to create a dataset."""
        return self.make_inference_pipeline().make_dataset()
