"""Data providers for pipeline I/O."""

import numpy as np
import tensorflow as tf
import attr
from typing import Text, Optional, List, Sequence, Union
import sleap


@attr.s(auto_attribs=True)
class LabelsReader:
    """Data provider from a `sleap.Labels` instance.

    This class can generate `tf.data.Dataset`s from a set of labels for use in data
    pipelines. Each element in the dataset will contain the data contained in a single
    `LabeledFrame`.

    Attributes:
        labels: The `sleap.Labels` instance to generate data from.
        example_indices: List or numpy array of ints with the labeled frame indices to
            use when iterating over the labels. Use this to specify subsets of the
            labels to use. Particularly handy for creating data splits. If not provided,
            the entire labels dataset will be read. These indices will be applicable to
            the labeled frames in `labels` attribute, which may have changed in ordering
            or filtered.
    """

    labels: sleap.Labels
    example_indices: Optional[Union[Sequence[int], np.ndarray]] = None

    @classmethod
    def from_user_instances(cls, labels: sleap.Labels) -> "LabelsReader":
        """Create a `LabelsReader` using the user instances in a `Labels` set.

        Args:
            labels: A `sleap.Labels` instance containing user instances.

        Returns:
            A `LabelsReader` instance that can create a dataset for pipelining. Note
            that the examples may change in ordering relative to the input `labels`, so
            be sure to use the `labels` attribute in the returned instance.
        """
        user_labels = sleap.Labels(
            [
                sleap.LabeledFrame(lf.video, lf.frame_idx, lf.training_instances)
                for lf in labels.user_labeled_frames
            ]
        )
        return cls(labels=user_labels)

    @classmethod
    def from_filename(
        cls, filename: Text, user_instances: bool = True
    ) -> "LabelsReader":
        """Create a `LabelsReader` from a saved labels file.

        Args:
            filename: Path to a saved labels file.
            user_instances: If True, will use only labeled frames with user instances.

        Returns:
            A `LabelsReader` instance that can create a dataset for pipelining.
        """
        labels = sleap.Labels.load_file(filename)
        if user_instances:
            return cls.from_user_instances(labels)
        else:
            return cls(labels=labels)

    def __len__(self) -> int:
        """Return the number of elements in the dataset."""
        if self.example_indices is None:
            return len(self.labels)
        else:
            return len(self.example_indices)

    @property
    def output_keys(self) -> List[Text]:
        """Return the output keys that the dataset will produce."""
        return [
            "image",
            "raw_image_size",
            "example_ind",
            "video_ind",
            "frame_ind",
            "scale",
            "instances",
            "skeleton_inds",
        ]

    @property
    def videos(self) -> List[sleap.Video]:
        """Return the list of videos that `video_ind` in examples match up with."""
        return self.labels.videos

    def make_dataset(
        self, ds_index: Optional[tf.data.Dataset] = None
    ) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` whose elements are data from labeled frames.

        Returns:
            A dataset whose elements are dictionaries with the loaded data associated
            with a single `LabeledFrame`. Items will be converted to tensors. These are:
                "image": Tensors of shape (height, width, channels) containing the full
                    raw frame image. The dtype is determined by the input data.
                "raw_image_size": The image size when it was first read as a tf.int32
                    tensor of shape (3,) representing [height, width, channels]. This is
                    useful for keeping track of absolute image coordinates if downstream
                    processing modules resize, crop or pad the image.
                "example_ind": Index of the individual labeled frame within the labels
                    stored in the `labels` attribute of this reader.
                "video_ind": Index of the video within the `Labels.videos` list that the
                    labeled frame comes from. Tensor will be a scalar of dtype tf.int32.
                "frame_ind": Index of the frame within the video that the labeled frame
                    comes from. Tensor will be a scalar of dtype tf.int64.
                "scale": The relative scaling factor of each image dimension specified
                    as a tf.float32 tensor of shape (2,) representing the
                    (x_scale, y_scale) of the example. This is always (1.0, 1.0) when
                    the images are initially read, but may be modified downstream in
                    order to keep track of scaling operations. This is especially
                    important to keep track of changes to the aspect ratio of the image
                    grid in order to properly map points to image coordinates.
                "instances": Tensor of shape (n_instances, n_nodes, 2) of dtype float32
                    containing all of the instances in the frame.
                "skeleton_inds": Tensor of shape (n_instances,) of dtype tf.int32 that
                    specifies the index of the skeleton used for each instance.
        """
        # Grab an image to test for the dtype.
        test_lf = self.labels[0]
        test_image = tf.convert_to_tensor(test_lf.image)
        image_dtype = test_image.dtype

        def py_fetch_lf(ind):
            """Local function that will not be autographed."""
            lf = self.labels[int(ind.numpy())]
            video_ind = np.array(self.videos.index(lf.video)).astype("int32")
            frame_ind = np.array(lf.frame_idx).astype("int64")
            raw_image = lf.image
            raw_image_size = np.array(raw_image.shape).astype("int32")
            instances = np.stack(
                [inst.points_array.astype("float32") for inst in lf.instances], axis=0
            )
            skeleton_inds = np.array(
                [self.labels.skeletons.index(inst.skeleton) for inst in lf.instances]
            ).astype("int32")
            return (
                raw_image,
                raw_image_size,
                instances,
                video_ind,
                frame_ind,
                skeleton_inds,
            )

        def fetch_lf(ind):
            """Local function that fetches a sample given the index."""
            ind = tf.cast(ind, tf.int64)
            (
                image,
                raw_image_size,
                instances,
                video_ind,
                frame_ind,
                skeleton_inds,
            ) = tf.py_function(
                py_fetch_lf,
                [ind],
                [image_dtype, tf.int32, tf.float32, tf.int32, tf.int64, tf.int32],
            )

            return {
                "image": image,
                "raw_image_size": raw_image_size,
                "example_ind": ind,
                "video_ind": video_ind,
                "frame_ind": frame_ind,
                "scale": tf.ones([2], dtype=tf.float32),
                "instances": instances,
                "skeleton_inds": skeleton_inds,
            }

        if self.example_indices is None:
            # Create default indexing dataset.
            ds_index = tf.data.Dataset.range(len(self))
        else:
            # Create indexing dataset from provided indices.
            ds_index = tf.data.Dataset.from_tensor_slices(self.example_indices)

        # Create reader dataset.
        # Note: We don't parallelize here for thread safety.
        ds_reader = ds_index.map(fetch_lf)

        return ds_reader


@attr.s(auto_attribs=True)
class VideoReader:
    """Data provider from a `sleap.Video` instance.

    This class can generate `tf.data.Dataset`s from a video for use in data pipelines.
    Each element in the dataset will contain the image data from a single frame.

    Attributes:
        video: The `sleap.Video` instance to generate data from.
        example_indices: List or numpy array of ints with the frame indices to use when
            iterating over the video. Use this to specify subsets of the video to read.
            If not provided, the entire video will be read.
        video_ind: Scalar index of video to keep with each example. Helpful when running
            inference across videos.
    """

    video: sleap.Video
    example_indices: Optional[Union[Sequence[int], np.ndarray]] = None

    @classmethod
    def from_filepath(
        cls,
        filename: Text,
        example_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
        **kwargs
    ) -> "VideoReader":
        """Create a `LabelsReader` from a saved labels file.

        Args:
            filename: Path to a video file.
            example_indices: List or numpy array of ints with the frame indices to use
                when iterating over the video. Use this to specify subsets of the video
                to read. If not provided, the entire video will be read.
            **kwargs: Any other video keyword argument (e.g., grayscale, dataset).

        Returns:
            A `VideoReader` instance that can create a dataset for pipelining.
        """
        video = sleap.Video.from_filename(filename, **kwargs)
        return cls(video=video, example_indices=example_indices)

    def __len__(self) -> int:
        """Return the number of elements in the dataset."""
        if self.example_indices is None:
            return len(self.video)
        else:
            return len(self.example_indices)

    @property
    def videos(self) -> List[sleap.Video]:
        """Return the list of videos that `video_ind` in examples match up with."""
        return [self.video]

    @property
    def output_keys(self) -> List[Text]:
        """Return the output keys that the dataset will produce."""
        return ["image", "raw_image_size", "video_ind", "frame_ind", "scale"]

    def make_dataset(self) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` whose elements are data from video frames.

        Returns:
            A dataset whose elements are dictionaries with the loaded data associated
            with a single video frame. Items will be converted to tensors. These are:
                "image": Tensors of shape (height, width, channels) containing the full
                    raw frame image.
                "raw_image_size": The image size when it was first read as a tf.int32
                    tensor of shape (3,) representing [height, width, channels]. This is
                    useful for keeping track of absolute image coordinates if downstream
                    processing modules resize, crop or pad the image.
                "video_ind": Index of the video (always 0). Can be used to index into
                    the `videos` attribute of the provider.
                "frame_ind": Index of the frame within the video that the frame comes
                    from. This is the same as the input index, but is also provided for
                    convenience in downstream processing.
                "scale": The relative scaling factor of each image dimension specified
                    as a tf.float32 tensor of shape (2,) representing the
                    (x_scale, y_scale) of the example. This is always (1.0, 1.0) when
                    the images are initially read, but may be modified downstream in
                    order to keep track of scaling operations. This is especially
                    important to keep track of changes to the aspect ratio of the image
                    grid in order to properly map points to image coordinates.
        """
        # Grab an image to test for the dtype.
        test_image = tf.convert_to_tensor(self.video.test_frame)
        image_dtype = test_image.dtype

        def py_fetch_frame(ind):
            """Local function that will not be autographed."""
            frame_ind = int(ind.numpy())
            raw_image = self.video.get_frame(frame_ind)
            raw_image_size = np.array(raw_image.shape).astype("int32")
            return raw_image, raw_image_size, np.array(frame_ind).astype("int64")

        def fetch_frame(ind):
            """Local function that fetches a sample given the index."""
            ind = tf.cast(ind, tf.int64)
            image, raw_image_size, frame_ind = tf.py_function(
                py_fetch_frame, [ind], [image_dtype, tf.int32, tf.int64]
            )

            return {
                "image": image,
                "raw_image_size": raw_image_size,
                "video_ind": 0,
                "frame_ind": frame_ind,
                "scale": tf.ones([2], dtype=tf.float32),
            }

        if self.example_indices is None:
            # Create default indexing dataset.
            ds_index = tf.data.Dataset.range(len(self))
        else:
            # Create indexing dataset from provided indices.
            ds_index = tf.data.Dataset.from_tensor_slices(self.example_indices)

        # Create reader dataset.
        # Note: We don't parallelize here for thread safety.
        ds_reader = ds_index.map(fetch_frame)

        return ds_reader
