"""Data providers for pipeline I/O."""

import numpy as np
import tensorflow as tf
import attr
from typing import Text, Optional, List, Sequence, Union, Tuple
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
        user_instances_only: If `True`, load only user labeled instances. If `False`,
            all instances will be loaded.
        with_track_only: If `True`, load only instances that have a track assigned.
            Useful when training supervised ID models.
    """

    labels: sleap.Labels
    example_indices: Optional[Union[Sequence[int], np.ndarray]] = None
    user_instances_only: bool = False
    with_track_only: bool = False

    @classmethod
    def from_user_instances(
        cls, labels: sleap.Labels, with_track_only: bool = False
    ) -> "LabelsReader":
        """Create a `LabelsReader` using the user instances in a `Labels` set.
        Args:
            labels: A `sleap.Labels` instance containing user instances.
            with_track_only: If `True`, load only instances that have a track assigned.
                Useful when training supervised ID models.

        Returns:
            A `LabelsReader` instance that can create a dataset for pipelining.

        Notes:
            This will remove "empty" instances, i.e., instances with no visible points,
            in the original labels. Make a copy of the original labels if needed as they
            will be modified in place.
        """
        labels.remove_empty_instances(keep_empty_frames=False)
        obj = cls.from_user_labeled_frames(labels)
        obj.user_instances_only = True
        obj.with_track_only = with_track_only
        return obj

    @classmethod
    def from_user_labeled_frames(cls, labels: sleap.Labels) -> "LabelsReader":
        """Create a `LabelsReader` using the user labeled frames in a `Labels` set.
        Args:
            labels: A `sleap.Labels` instance containing user labeled frames.
        Returns:
            A `LabelsReader` instance that can create a dataset for pipelining.
            Note that this constructor will load ALL instances in frames that have user
            instances. To load only user labeled instances, use
            `LabelsReader.from_user_instances`.
        """
        return cls(labels=labels, example_indices=labels.user_labeled_frame_inds)

    @classmethod
    def from_unlabeled_suggestions(cls, labels: sleap.Labels) -> "LabelsReader":
        """Create a `LabelsReader` using the unlabeled suggestions in a `Labels` set.
        Args:
            labels: A `sleap.Labels` instance containing unlabeled suggestions.
        Returns:
            A `LabelsReader` instance that can create a dataset for pipelining.
        """
        inds = labels.get_unlabeled_suggestion_inds()
        return cls(labels=labels, example_indices=inds)

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
        labels = sleap.load_file(filename)
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
            "track_inds",
            "n_tracks",
        ]

    @property
    def videos(self) -> List[sleap.Video]:
        """Return the list of videos that `video_ind` in examples match up with."""
        return self.labels.videos

    @property
    def tracks(self) -> List[sleap.Track]:
        """Return the list of tracks that `track_inds` in examples match up with."""
        return self.labels.tracks

    @property
    def max_height_and_width(self) -> Tuple[int, int]:
        """Return `(height, width)` that is the maximum of all videos."""
        return max(video.shape[1] for video in self.videos), max(
            video.shape[2] for video in self.videos
        )

    @property
    def is_from_multi_size_videos(self) -> bool:
        """Return `True` if labels contain videos with different sizes."""
        return (
            len(set(v.shape[1] for v in self.videos)) > 1
            or len(set(v.shape[2] for v in self.videos)) > 1
        )

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
                "track_inds": Tensor of shape (n_instance,) of dtype tf.int32 that
                    specifies the index of the instance track identity. If not
                    specified, in the labels, this is set to -1.
        """
        # Grab the first image to capture dtype and number of color channels.
        first_image = tf.convert_to_tensor(self.labels[0].image)
        image_dtype = first_image.dtype
        image_num_channels = first_image.shape[-1]

        def py_fetch_lf(ind):
            """Local function that will not be autographed."""
            lf = self.labels[int(ind.numpy())]

            video_ind = np.array(self.videos.index(lf.video)).astype("int32")
            frame_ind = np.array(lf.frame_idx).astype("int64")

            raw_image = lf.image
            raw_image_size = np.array(raw_image.shape).astype("int32")

            if self.user_instances_only:
                insts = lf.user_instances
            else:
                insts = lf.instances
            insts = [inst for inst in insts if len(inst) > 0]
            if self.with_track_only:
                insts = [inst for inst in insts if inst.track is not None]
            n_instances = len(insts)
            n_nodes = len(insts[0].skeleton) if n_instances > 0 else 0

            instances = np.full((n_instances, n_nodes, 2), np.nan, dtype="float32")
            for i, instance in enumerate(insts):
                instances[i] = instance.numpy()

            skeleton_inds = np.array(
                [self.labels.skeletons.index(inst.skeleton) for inst in insts]
            ).astype("int32")
            track_inds = np.array(
                [
                    self.tracks.index(inst.track) if inst.track is not None else -1
                    for inst in insts
                ]
            ).astype("int32")
            n_tracks = np.array(len(self.tracks)).astype("int32")
            return (
                raw_image,
                raw_image_size,
                instances,
                video_ind,
                frame_ind,
                skeleton_inds,
                track_inds,
                n_tracks,
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
                track_inds,
                n_tracks,
            ) = tf.py_function(
                py_fetch_lf,
                [ind],
                [
                    image_dtype,
                    tf.int32,
                    tf.float32,
                    tf.int32,
                    tf.int64,
                    tf.int32,
                    tf.int32,
                    tf.int32,
                ],
            )

            # Ensure shape with constant or variable height/width, based on whether or
            # not the videos have mixed sizes.
            if self.is_from_multi_size_videos:
                image = tf.ensure_shape(image, (None, None, image_num_channels))
            else:
                image = tf.ensure_shape(image, first_image.shape)

            instances = tf.ensure_shape(instances, tf.TensorShape([None, None, 2]))
            skeleton_inds = tf.ensure_shape(skeleton_inds, tf.TensorShape([None]))
            track_inds = tf.ensure_shape(track_inds, tf.TensorShape([None]))

            return {
                "image": image,
                "raw_image_size": raw_image_size,
                "example_ind": ind,
                "video_ind": video_ind,
                "frame_ind": frame_ind,
                "scale": tf.ones([2], dtype=tf.float32),
                "instances": instances,
                "skeleton_inds": skeleton_inds,
                "track_inds": track_inds,
                "n_tracks": n_tracks,
            }

        if self.example_indices is None:
            # Create default indexing dataset.
            ds_index = tf.data.Dataset.range(len(self))
        else:
            # Create indexing dataset from provided indices.
            if isinstance(self.example_indices, range):
                ds_index = tf.data.Dataset.from_tensor_slices(
                    list(self.example_indices)
                )
            else:
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
    def max_height_and_width(self) -> Tuple[int, int]:
        """Return `(height, width)` that is the maximum of all videos."""
        return max(video.shape[1] for video in self.videos), max(
            video.shape[2] for video in self.videos
        )

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
        test_image = tf.convert_to_tensor(
            self.video.get_frame(self.video.last_frame_idx)
        )
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
            image = tf.ensure_shape(image, test_image.shape)

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
            if isinstance(self.example_indices, range):
                ds_index = tf.data.Dataset.from_tensor_slices(
                    list(self.example_indices)
                )
            else:
                ds_index = tf.data.Dataset.from_tensor_slices(self.example_indices)

        # Create reader dataset.
        # Note: We don't parallelize here for thread safety.
        ds_reader = ds_index.map(fetch_frame)

        return ds_reader
