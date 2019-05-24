""" Video reading and writing interfaces for different formats. """

import os

import h5py as h5
import cv2
import numpy as np
import attr
import cattr

from typing import Iterable, Union


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
        convert_range: Whether we should convert data to [0, 255]-range
    """

    file: str = attr.ib(default=None)
    dataset: str = attr.ib(default=None)
    input_format: str = attr.ib(default="channels_last")
    convert_range: bool = attr.ib(default=True)

    def __attrs_post_init__(self):

        # Handle cases where the user feeds in h5.File objects instead of filename
        if isinstance(self.file, h5.File):
            self.__file_h5 = self.file
            self.file = self.__file_h5.filename
        elif type(self.file) is str:
            try:
                self.__file_h5 = h5.File(self.file, 'r')
            except OSError as ex:
                raise FileNotFoundError(f"Could not find HDF5 file {self.file}") from ex
        else:
            self.__file_h5 = None

        # Handle the case when h5.Dataset is passed in
        if isinstance(self.dataset, h5.Dataset):
            self.__dataset_h5 = self.dataset
            self.__file_h5 = self.__dataset_h5.file
            self.dataset = self.__dataset_h5.name
        elif self.dataset is not None and type(self.dataset) is str:
            self.__dataset_h5 = self.__file_h5[self.dataset]
        else:
            self.__dataset_h5 = None


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
            self.__width_idx = 2
            self.__height_idx = 1

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def frames(self):
        return self.__dataset_h5.shape[0]

    @property
    def channels(self):
        return self.__dataset_h5.shape[self.__channel_idx]

    @property
    def width(self):
        return self.__dataset_h5.shape[self.__width_idx]

    @property
    def height(self):
        return self.__dataset_h5.shape[self.__height_idx]

    @property
    def dtype(self):
        return self.__dataset_h5.dtype

    @property
    def filename(self):
        return self.file

    def get_frame(self, idx):# -> np.ndarray:
        """
        Get a frame from the underlying HDF5 video data.

        Args:
            idx: The index of the frame to get.

        Returns:
            The numpy.ndarray representing the video frame data.
        """
        frame = self.__dataset_h5[idx]

        if self.input_format == "channels_first":
            frame = np.transpose(frame, (2, 1, 0))

        if self.convert_range and np.max(frame) <= 1.:
            frame = (frame * 255).astype(int)

        return frame


@attr.s(auto_attribs=True, cmp=False)
class MediaVideo:
    """
    Video data stored in traditional media formats readable by FFMPEG can be loaded
    with this class. This class provides bare minimum read only interface on top of
    OpenCV's VideoCapture class.

    Args:
        filename: The name of the file (.mp4, .avi, etc)
        grayscale: Whether the video is grayscale or not. "auto" means detect
        based on first frame.
    """
    filename: str = attr.ib()
    # grayscale: bool = attr.ib(default=None, converter=bool)
    grayscale: bool = attr.ib()
    bgr: bool = attr.ib(default=True)
    _detect_grayscale = False

    @grayscale.default
    def __grayscale_default__(self):
        self._detect_grayscale = True
        return False

    def __attrs_post_init__(self):

        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"Could not find file video file named {self.filename}")

        # Try and open the file either locally in current directory or with full path
        self.__reader = cv2.VideoCapture(self.filename)

        # Lets grab a test frame to help us figure things out about the video
        self.__test_frame = self.get_frame(0, grayscale=False)

        # If the user specified None for grayscale bool, figure it out based on the
        # the first frame of data.
        if self._detect_grayscale is True:
            self.grayscale = bool(np.alltrue(self.__test_frame[..., 0] == self.__test_frame[..., -1]))

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def frames(self):
        return int(self.__reader.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frames_float(self):
        return self.__reader.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def channels(self):
        if self.grayscale:
            return 1
        else:
            return self.__test_frame.shape[2]

    @property
    def width(self):
        return self.__test_frame.shape[1]

    @property
    def height(self):
        return self.__test_frame.shape[0]

    @property
    def dtype(self):
        return self.__test_frame.dtype

    def get_frame(self, idx, grayscale=None):
        if grayscale is None:
            grayscale = self.grayscale

        if self.__reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self.__reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = self.__reader.read()

        if grayscale:
            frame = frame[...,0][...,None]

        if self.bgr:
            frame = frame[...,::-1]

        return frame


@attr.s(auto_attribs=True, cmp=False)
class NumpyVideo:
    """
    Video data stored as Numpy array.

    Args:
        file: Either a filename to load or a numpy array of the data.

        * numpy data shape: (frames, width, height, channels)
    """
    file: attr.ib()

    def __attrs_post_init__(self):

        self.__frame_idx = 0
        self.__width_idx = 1
        self.__height_idx = 2
        self.__channel_idx = 3

        # Handle cases where the user feeds in np.array instead of filename
        if isinstance(self.file, np.ndarray):
            self.__data = self.file
            self.file = "Raw Video Data"
        elif type(self.file) is str:
            try:
                self.__data = np.load(self.file)
            except OSError as ex:
                raise FileNotFoundError(f"Could not find file {self.file}") from ex
        else:
            self.__data = None

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def filename(self):
        return self.file

    @property
    def frames(self):
        return self.__data.shape[self.__frame_idx]

    @property
    def channels(self):
        return self.__data.shape[self.__channel_idx]

    @property
    def width(self):
        return self.__data.shape[self.__width_idx]

    @property
    def height(self):
        return self.__data.shape[self.__height_idx]

    @property
    def dtype(self):
        return self.__data.dtype

    def get_frame(self, idx):
        return self.__data[idx]


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

    backend: Union[HDF5Video, NumpyVideo, MediaVideo] = attr.ib()

    # Delegate to the backend
    def __getattr__(self, item):
        return getattr(self.backend, item)

    @property
    def num_frames(self) -> int:
        """The number of frames in the video. Just an alias for frames property."""
        return self.frames

    @property
    def shape(self):
        return (self.frames, self.height, self.width, self.channels)

    def __str__(self):
        """ Informal string representation (for print or format) """
        return type(self).__name__ + " ([%d x %d x %d x %d])" % self.shape

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
        return self.backend.get_frame(idx)

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
    def from_hdf5(cls, dataset: Union[str, h5.Dataset],
                  file: Union[str, h5.File] = None,
                  input_format: str = "channels_last",
                  convert_range: bool = True):
        """
        Create an instance of a video object from an HDF5 file and dataset. This
        is a helper method that invokes the HDF5Video backend.

        Args:
            dataset: The name of the dataset or and h5.Dataset object. If file is
            h5.File, dataset must be a str of the dataset name.
            file: The name of the HDF5 file or and open h5.File object.
            input_format: Whether the data is oriented with "channels_first" or "channels_last"
            convert_range: Whether we should convert data to [0, 255]-range

        Returns:
            A Video object with HDF5Video backend.
        """
        backend = HDF5Video(
                    file=file,
                    dataset=dataset,
                    input_format=input_format,
                    convert_range=convert_range
                    )
        return cls(backend=backend)

    @classmethod
    def from_numpy(cls, file, *args, **kwargs):
        """
        Create an instance of a video object from a numpy array.

        Args:
            file: The numpy array or the name of the file

        Returns:
            A Video object with a NumpyVideo backend
        """
        backend = NumpyVideo(file=file, *args, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_media(cls, file: str, *args, **kwargs):
        """
        Create an instance of a video object from a typical media file (e.g. .mp4, .avi).

        Args:
            file: The name of the file

        Returns:
            A Video object with a MediaVideo backend
        """
        backend = MediaVideo(filename=file, *args, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_filename(cls, file: str, *args, **kwargs):
        """
        Create an instance of a video object from a filename, auto-detecting the backend.

        Args:
            file: The path to the video file

        Returns:
            A Video object with the detected backend
        """
        if file.lower().endswith(("h5", "hdf5")):
            return cls(backend=HDF5Video(file=file, *args, **kwargs))
        elif file.endswith(("npy")):
            return cls(backend=NumpyVideo(file=file, *args, **kwargs))
        elif file.lower().endswith(("mp4", "avi")):
            return cls(backend=MediaVideo(filename=file, *args, **kwargs))
        else:
            raise ValueError("Could not detect backend for specified filename.")

    @classmethod
    def to_numpy(cls, frame_data: np.array, file_name: str):
        np.save(file_name, frame_data, 'w')
