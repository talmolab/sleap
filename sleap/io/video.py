""" Video reading and writing interfaces for different formats. """

import os

import h5py as h5
import cv2
import numpy as np
import attr
import cattr

from typing import Iterable, Union

from sleap.util import try_open_file


@attr.s(auto_attribs=True, cmp=False)
class Video:
    """
    The top-level interface to any Video data used by sLEAP is represented by
    the :class:`.Video` class. This class provides a common interface for
    various supported video data backends. It provides the bare minimum of
    properties and methods that any video data needs to support in order to
    function with other sLEAP components. This interface currently only supports
    reading of video data, there is no write support. Unless one is creating a new video
    backend, this class should be instantiated from its various class methods
    for different formats. For example:

    >>> video = Video.from_hdf5(file='test.h5', dataset='box')
    >>> video = Video.from_media(file='test.mp4')

    Args:
        backend: A backend is and object that implements the following basic
        required methods and properties

        * Properties

            * :code:`frames`: The number of frames in the video
            * :code:`channels`: The number of channels in the video (e.g. 1 for grayscale, 3 for RGB)
            * :code:`width`: The width of each frame in pixels
            * :code:`height`: The height of each frame in pixels

        * Methods

            * :code:`get_frame(frame_index: int) -> np.ndarray(shape=(width, height, channels)`:
            Get a single frame from the underlying video data

    """

    _backend: object = attr.ib()

    @property
    def frames(self) -> int:
        """The number of frames in the video"""
        return self._backend.frames

    @property
    def num_frames(self) -> int:
        """The number of frames in the video. Just an alias for frames property."""
        return self.frames

    @property
    def width(self) -> int:
        """The width of the video in pixels"""
        return self._backend.width

    @property
    def height(self) -> int:
        """The height of the video in pixels"""
        return self._backend.height

    @property
    def channels(self) -> int:
        """The number of channels in the video (e.g. 1 for grayscale, 3 for RGB)"""
        return self._backend.channels

    @property
    def dtype(self) -> np.dtype:
        """The numpy datatype of each frame ndarray"""
        return self._backend.dtype

    @property
    def shape(self):
        """Utitlity property for :code:`(frames, height, width, channels)`"""
        return (self.frames, self.height, self.width, self.channels)

    def __str__(self):
        """ Informal string representation (for print or format) """
        return type(self).__name__ + "([%d x %d x %d x %d])" % self.shape

    def __len__(self):
        """
        The length of the video should be the number of frames.

        Returns:
            The number of frames in the video.
        """
        return self.frames

    def get_frame(self, idx: int) -> np.ndarray:
        """
        Return a single frame of video from the underlying video data.

        Args:
            idx: The index of the video frame

        Returns:
            The video frame with shape (width, height, channels)
        """
        return self._backend.get_frame(idx)

    def get_frames(self, idxs: Union[int, Iterable[int]]) -> np.ndarray:
        """
        Return a collection of video frames from the underlying video data.

        Args:
            idxs: An iterable object that contains the indices of frames.

        Returns:
            The requested video frames with shape (len(idxs), width, height, channels)
        """
        if np.isscalar(idxs):
            idxs = [idxs,]
        return np.stack([self.get_frame(idx) for idx in idxs], axis=0)

    def __getitem__(self, idxs):
        if isinstance(idxs, slice):
            start, stop, step = idxs.indices(self.num_frames)
            idxs = range(start, stop, step)
        return self.get_frames(idxs)

    @classmethod
    def from_hdf5(cls, file: str, dataset: Union[str, h5.Dataset], input_format: str = "channels_last"):
        """
        Create an instance of a video object from an HDF5 file and dataset. This
        is a helper method that invokes the HDF5Video backend.

        Args:
            file:
            dataset:
            input_format:

        Returns:

        """
        if type(file) is str:
            backend = HDF5Video.from_file(filename=file, dataset_name=dataset, input_format=input_format)
        else:
            backend = HDF5Video.from_dataset(dataset=dataset, input_format=input_format)

        return cls(backend=backend)

    @classmethod
    def from_media(cls, file: str, *args, **kwargs):
        """
        Create an instance of a video object from a typical media file (e.g. .mp4, .avi).

        Args:
            file: The name of the file


        Returns:

        """
        backend = MediaVideo(filename=file, *args, **kwargs)
        return cls(backend=backend)

    @staticmethod
    def make_cattr():
        _cattr = cattr.Converter()
        _cattr.register_unstructure_hook(h5.File, lambda x: None)
        _cattr.register_unstructure_hook(h5.Dataset, lambda x: None)
        _cattr.register_unstructure_hook(h5.Group, lambda x: None)
        _cattr.register_unstructure_hook(cv2.VideoCapture, lambda x: None)
        _cattr.register_unstructure_hook(np.bool_, bool)

        return _cattr

@attr.s(auto_attribs=True, cmp=False)
class HDF5Video:
    """
    Video data stored as 4D datasets in HDF5 files can be imported into
    the sLEAP system with this class.

    Args:
        file: The name of the HDF5 file where the dataset with video data is stored.
        dataset: The name of the HDF5 dataset where the video data is stored.
        file_h5: The h5.File object that the underlying dataset is stored.
        dataset_h5: The h5.Dataset object that the underlying data is stored.
        input_format: A string value equal to either "channels_last" or "channels_first".
        This specifies whether the underlying video data is stored as:

            * "channels_first": shape = (frames, channels, width, height)
            * "channels_last": shape = (frames, width, height, channels)
    """

    file: str = attr.ib(default=None)
    dataset: str = attr.ib(default=None)
    _file_h5: h5.File = attr.ib(default=None)
    _dataset_h5: h5.Dataset = attr.ib(default=None)
    input_format: str = attr.ib(default="channels_last")

    def __attrs_post_init__(self):
        if isinstance(self._file_h5, h5.File):
            self.file = self._file_h5.filename
        elif self._file_h5 is None:
            try:
                self._file_h5 = h5.File(self.file, 'r')
            except OSError as ex:
                raise FileNotFoundError(f"Could not find HDF5 file {self.file}") from ex

        if isinstance(self._dataset_h5, h5.Dataset):
            self.dataset = self._dataset_h5.name
        elif self._dataset_h5 is None:
            self._dataset_h5 = self._file_h5[self.dataset]


    @input_format.validator
    def check(self, attribute, value):
        if value not in ["channels_first", "channels_last"]:
            raise ValueError(f"HDF5Video input_format={value} invalid.")

        if value == "channels_first":
            self.__channel_idx = 1
            self.__width_idx = 2
            self.__height_idx = 3
        else:
            self.__channel_idx = 3
            self.__width_idx = 1
            self.__height_idx = 2

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def frames(self):
        return self._dataset_h5.shape[0]

    @property
    def channels(self):
        return self._dataset_h5.shape[self.__channel_idx]

    @property
    def width(self):
        return self._dataset_h5.shape[self.__width_idx]

    @property
    def height(self):
        return self._dataset_h5.shape[self.__height_idx]

    @property
    def dtype(self):
        return self._dataset_h5.dtype

    def get_frame(self, idx) -> np.ndarray:
        """
        Get a frame from the underlying HDF5 video data.

        Args:
            idx: The index of the frame to get.

        Returns:
            The numpy.ndarray representing the video frame data.
        """
        frame = self._dataset_h5[idx]

        if self.input_format == "channels_first":
            frame = np.transpose(frame, (2, 1, 0))

        return frame

    @classmethod
    def from_file(cls, filename: str, dataset_name: str, *args, **kwargs):
        """
        Create and HDF5Video object from a dataset name and HDF5 file name.

        Args:
            filename: The name of the HDF5 file
            dataset_name:  The name of the dataset.

        Returns:
            The instantiated HDF5Video object.
        """
        return cls(file=filename, dataset=dataset_name, *args, **kwargs)

    @classmethod
    def from_dataset(cls, dataset: h5.Dataset) -> 'HDF5Video':
        """
        Create an HDF5Video from an HDF5 dataset object

        Args:
            dataset: The dataset that contains the underlying video data.

        Returns:

        """
        return cls(file_h5=dataset.file, dataset_h5=dataset)


@attr.s(auto_attribs=True, cmp=False)
class MediaVideo:
    """
    Video data stored in traditional media formats readable by FFMPEG can be loaded
    with this class. This class provides bare minimum read only interface on top of
    OpenCV's VideoCapture class.

    Args:
        filename: The name of the fiel
    """
    filename: str = attr.ib()
    _reader: cv2.VideoCapture = attr.ib(default=None)
    grayscale: bool = attr.ib(default=None)

    def __attrs_post_init__(self):

        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"Could not file video file named {self.filename}")

        # Try and open the file either locally in current directory or with full path
        self._reader = cv2.VideoCapture(self.filename)

        # Lets grab a test frame to help us figure things out about the video
        self.__test_frame = self.get_frame(0, grayscale=False)

        # If the user specified None for grayscale bool, figure it out based on the
        # the first frame of data.
        if self.grayscale is None:
            self.grayscale = np.alltrue(self.__test_frame[..., 0] == self.__test_frame[..., -1])

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def frames(self):
        return int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frames_float(self):
        return self._reader.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def channels(self):
        if self.grayscale:
            return 1
        else:
            return self.__test_frame.shape[0]

    @property
    def width(self):
        return self.__test_frame.shape[1]

    @property
    def height(self):
        return self.__test_frame.shape[2]

    @property
    def dtype(self):
        return self.__test_frame.dtype

    def get_frame(self, idx, grayscale=None):
        if grayscale is None:
            grayscale = self.grayscale

        if self._reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self._reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = self._reader.read()

        if grayscale:
            frame = frame[...,0][...,None]

        return frame

