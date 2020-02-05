"""Data providers for pipeline I/O."""

import tensorflow as tf
import attr
from typing import Text, Optional, List
import sleap


@attr.s(auto_attribs=True)
class LabelsReader:
    """Data provider from a `sleap.Labels` instance.
    
    This class can generate `tf.data.Dataset`s from a set of labels for use in data
    pipelines. Each element in the dataset will contain the data contained in a single
    `LabeledFrame`.
    
    Attributes:
        labels: The `sleap.Labels` instance to generate data from.
    """

    labels: sleap.Labels

    @classmethod
    def from_user_instances(cls, labels: sleap.Labels) -> "LabelsReader":
        """Create a `LabelsReader` using the user instances in a `Labels` set.
        
        Args:
            labels: A `sleap.Labels` instance containing user instances.
        
        Returns:
            A `LabelsReader` instance that can create a dataset for pipelining.
        """
        user_labels = sleap.Labels(
            [
                sleap.LabeledFrame(lf.video, lf.frame_idx, lf.user_instances)
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
        return len(self.labels)

    @property
    def output_keys(self) -> List[Text]:
        """Return the output keys that the dataset will produce."""
        return [
            "image",
            "video_ind",
            "frame_ind",
            "scale",
            "instances",
            "skeleton_inds",
        ]

    def make_dataset(
        self, ds_index: Optional[tf.data.Dataset] = None
    ) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` whose elements are data from labeled frames.
        
        Args:
            ds_index: If provided, the returned dataset will be mapped onto this
                dataset. This can be used to provide an explicit range of samples to
                read. Elements of this dataset must be integers.
                If not provided, an indexing dataset will be created containing all of
                the indices available, i.e., `tf.data.Dataset.range(len(self))`.

        Returns:
            A dataset whose elements are dictionaries with the loaded data associated
            with a single `LabeledFrame`. Items will be converted to tensors. These are:
                "image": Tensors of shape (height, width, channels) containing the full
                    raw frame image. The dtype is determined by the input data.
                "video_ind": Index of the video within the `Labels.videos` list that the
                    labeled frame comes from. Tensor will be a scalar of dtype tf.int32.
                "frame_ind": Index of the frame within the video that the labeled frame
                    comes from. Tensor will be a scalar of dtype tf.int64.
                "scale": The relative scale of the image and points. This is always 1.0
                    when reading the raw data, but can be adjusted by downstream
                    processing. Tensor will be a scalar of dtype tf.int32.
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
            video_ind = self.labels.videos.index(lf.video)
            frame_ind = lf.frame_idx
            image = tf.convert_to_tensor(lf.image)
            instances = [
                tf.convert_to_tensor(inst.points_array, dtype=tf.float32)
                for inst in lf.instances
            ]
            skeleton_inds = [
                self.labels.skeletons.index(inst.skeleton) for inst in lf.instances
            ]
            return image, instances, video_ind, frame_ind, skeleton_inds

        def fetch_lf(ind):
            """Local function that fetches a sample given the index."""
            image, instances, video_ind, frame_ind, skeleton_inds = tf.py_function(
                py_fetch_lf,
                [ind],
                [image_dtype, tf.float32, tf.int32, tf.int64, tf.int32],
            )

            return {
                "image": image,
                "video_ind": video_ind,
                "frame_ind": frame_ind,
                "scale": 1.0,
                "instances": instances,
                "skeleton_inds": skeleton_inds,
            }

        if ds_index is None:
            # Create default indexing dataset.
            ds_index = tf.data.Dataset.range(len(self))

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
    """

    video: sleap.Video

    @classmethod
    def from_filepath(cls, filename: Text, **kwargs) -> "VideoReader":
        """Create a `LabelsReader` from a saved labels file.
        
        Args:
            filename: Path to a video file.
            **kwargs: Any other video keyword argument (e.g., grayscale, dataset).
        
        Returns:
            A `VideoReader` instance that can create a dataset for pipelining.
        """
        video = sleap.Video.from_filename(filename, **kwargs)
        return cls(video=video)

    def __len__(self) -> int:
        """Return the number of elements in the dataset."""
        return len(self.video)

    @property
    def output_keys(self) -> List[Text]:
        """Return the output keys that the dataset will produce."""
        return ["image", "frame_ind", "scale"]

    def make_dataset(
        self, ds_index: Optional[tf.data.Dataset] = None
    ) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` whose elements are data from labeled frames.
        
        Args:
            ds_index: If provided, the returned dataset will be mapped onto this
                dataset. This can be used to provide an explicit range of samples to
                read. Elements of this dataset must be integers.
                If not provided, an indexing dataset will be created containing all of
                the indices available, i.e., `tf.data.Dataset.range(len(self))`.

        Returns:
            A dataset whose elements are dictionaries with the loaded data associated
            with a single video frame. Items will be converted to tensors. These are:
                "image": Tensors of shape (height, width, channels) containing the full
                    raw frame image.
                "frame_ind": Index of the frame within the video that the frame comes
                    from. This is the same as the input index, but is also provided for
                    convenience in downstream processing.
                "scale": This is always 1.0 when reading the raw data, but can be
                    adjusted by downstream processing.
        """
        # Grab an image to test for the dtype.
        test_image = tf.convert_to_tensor(self.video.get_frame(0))
        image_dtype = test_image.dtype

        def py_fetch_frame(ind):
            """Local function that will not be autographed."""
            frame_ind = int(ind.numpy())
            image = tf.convert_to_tensor(self.video.get_frame(frame_ind))
            return image, frame_ind

        def fetch_frame(ind):
            """Local function that fetches a sample given the index."""
            image, frame_ind = tf.py_function(
                py_fetch_frame, [ind], [image_dtype, tf.int64]
            )

            return {"image": image, "frame_ind": frame_ind, "scale": 1.0}

        if ds_index is None:
            # Create default indexing dataset.
            ds_index = tf.data.Dataset.range(len(self))

        # Create reader dataset.
        # Note: We don't parallelize here for thread safety.
        ds_reader = ds_index.map(fetch_frame)

        return ds_reader
