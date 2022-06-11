""" Video reading and writing interfaces for different formats. """

import os
import shutil

import h5py as h5
import cv2
import imgstore
import numpy as np
import attr
import cattr
import logging
import multiprocessing

from typing import Iterable, List, Optional, Tuple, Union, Text

from sleap.util import json_loads, json_dumps

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True, eq=False, order=False)
class DummyVideo:
    """
    Fake video backend,returns frames with all zeros.

    This can be useful when you want to look at labels for a dataset but don't
    have access to the real video.
    """

    filename: str = ""
    height: int = 2000
    width: int = 2000
    frames: int = 10000
    channels: int = 1
    dummy: bool = True

    @property
    def test_frame(self):
        return self.get_frame(0)

    def get_frame(self, idx) -> np.ndarray:
        return np.zeros((self.height, self.width, self.channels))


@attr.s(auto_attribs=True, eq=False, order=False)
class HDF5Video:
    """
    Video data stored as 4D datasets in HDF5 files.

    Args:
        filename: The name of the HDF5 file where the dataset with video data
            is stored.
        dataset: The name of the HDF5 dataset where the video data is stored.
        file_h5: The h5.File object that the underlying dataset is stored.
        dataset_h5: The h5.Dataset object that the underlying data is stored.
        input_format: A string value equal to either "channels_last" or
            "channels_first".
            This specifies whether the underlying video data is stored as:

                * "channels_first": shape = (frames, channels, height, width)
                * "channels_last": shape = (frames, height, width, channels)
        convert_range: Whether we should convert data to [0, 255]-range
    """

    filename: str = attr.ib(default=None)
    dataset: str = attr.ib(default=None)
    input_format: str = attr.ib(default="channels_last")
    convert_range: bool = attr.ib(default=True)

    def __attrs_post_init__(self):
        """Called by attrs after __init__()."""

        self.enable_source_video = True
        self._test_frame_ = None
        self.__original_to_current_frame_idx = dict()
        self.__dataset_h5 = None
        self.__tried_to_load = False

    @input_format.validator
    def check(self, attribute, value):
        """Called by attrs to validates input format."""
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

    def _load(self):
        if self.__tried_to_load:
            return

        self.__tried_to_load = True

        # Handle cases where the user feeds in h5.File objects instead of filename
        if isinstance(self.filename, h5.File):
            self.__file_h5 = self.filename
            self.filename = self.__file_h5.filename
        elif type(self.filename) is str:
            try:
                self.__file_h5 = h5.File(self.filename, "r")
            except OSError as ex:
                raise FileNotFoundError(
                    f"Could not find HDF5 file {self.filename}"
                ) from ex
        else:
            self.__file_h5 = None

        # Handle the case when h5.Dataset is passed in
        if isinstance(self.dataset, h5.Dataset):
            self.__dataset_h5 = self.dataset
            self.__file_h5 = self.__dataset_h5.file
            self.dataset = self.__dataset_h5.name

        # File loaded and dataset name given, so load dataset
        elif isinstance(self.dataset, str) and (self.__file_h5 is not None):
            # dataset = "video0" passed:
            if self.dataset + "/video" in self.__file_h5:
                self.__dataset_h5 = self.__file_h5[self.dataset + "/video"]
                base_dataset_path = self.dataset
            else:
                # dataset = "video0/video" passed:
                self.__dataset_h5 = self.__file_h5[self.dataset]
                base_dataset_path = "/".join(self.dataset.split("/")[:-1])

            # Check for frame_numbers dataset corresponding to video
            framenum_dataset = f"{base_dataset_path}/frame_numbers"
            if framenum_dataset in self.__file_h5:
                original_idx_lists = self.__file_h5[framenum_dataset]
                # Create map from idx in original video to idx in current
                for current_idx in range(len(original_idx_lists)):
                    original_idx = original_idx_lists[current_idx]
                    self.__original_to_current_frame_idx[original_idx] = current_idx

            source_video_group = f"{base_dataset_path}/source_video"
            if source_video_group in self.__file_h5:
                d = json_loads(
                    self.__file_h5.require_group(source_video_group).attrs["json"]
                )

                self._source_video = Video.cattr().structure(d, Video)

    @property
    def __dataset_h5(self) -> h5.Dataset:
        if self.__loaded_dataset is None and not self.__tried_to_load:
            self._load()
        return self.__loaded_dataset

    @__dataset_h5.setter
    def __dataset_h5(self, val):
        self.__loaded_dataset = val

    @property
    def test_frame(self):
        # Load if not already loaded
        if self._test_frame_ is None:
            # Lets grab a test frame to help us figure things out about the video
            self._test_frame_ = self.get_frame(self.last_frame_idx)

        # Return stored test frame
        return self._test_frame_

    @property
    def enable_source_video(self) -> bool:
        """If set to `True`, will attempt to read from original video for frames not
        saved in the file."""
        return self._enable_source_video

    @enable_source_video.setter
    def enable_source_video(self, val: bool):
        self._enable_source_video = val

    @property
    def has_embedded_images(self) -> bool:
        """Return `True` if the file was saved with cached frame images."""
        self._load()
        return len(self.__original_to_current_frame_idx) > 0

    @property
    def embedded_frame_inds(self) -> List[int]:
        """Return list of frame indices with embedded images."""
        self._load()
        return list(self.__original_to_current_frame_idx.keys())

    @property
    def source_video_available(self) -> bool:
        """Return `True` if the source file is available for reading uncached frames."""
        self._load()
        return (
            self.enable_source_video
            and hasattr(self, "_source_video")
            and self._source_video
        )

    @property
    def source_video(self) -> "Video":
        """Return the source video if available, otherwise return `None`."""
        if self.source_video_available:
            return self._source_video
        return None

    def matches(self, other: "HDF5Video") -> bool:
        """
        Check if attributes match those of another video.

        Args:
            other: The other video to compare with.

        Returns:
            True if attributes match, False otherwise.
        """
        return (
            self.filename == other.filename
            and self.dataset == other.dataset
            and self.convert_range == other.convert_range
            and self.input_format == other.input_format
        )

    def close(self):
        """Close the HDF5 file object (if it's open)."""
        try:
            self.__file_h5.close()
        except:
            pass
        self.__file_h5 = None

    def __del__(self):
        """Releases file object."""
        self.close()

    def _try_frame_from_source_video(self, idx) -> np.ndarray:
        try:
            return self.source_video.get_frame(idx)
        except:
            raise IndexError(f"Frame index {idx} not in original index.")

    # The properties and methods below complete our contract with the higher level
    # Video interface.

    @property
    def frames(self):
        """See :class:`Video`."""
        return self.__dataset_h5.shape[0]

    @property
    def channels(self):
        """See :class:`Video`."""
        if "channels" in self.__dataset_h5.attrs:
            return int(self.__dataset_h5.attrs["channels"])
        return self.__dataset_h5.shape[self.__channel_idx]

    @property
    def width(self):
        """See :class:`Video`."""
        if "width" in self.__dataset_h5.attrs:
            return int(self.__dataset_h5.attrs["width"])
        return self.__dataset_h5.shape[self.__width_idx]

    @property
    def height(self):
        """See :class:`Video`."""
        if "height" in self.__dataset_h5.attrs:
            return int(self.__dataset_h5.attrs["height"])
        return self.__dataset_h5.shape[self.__height_idx]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.test_frame.dtype

    @property
    def last_frame_idx(self) -> int:
        """
        The idx number of the last frame.

        Overrides method of base :class:`Video` class for videos with
        select frames indexed by number from original video, since the last
        frame index here will not match the number of frames in video.
        """
        # Ensure that video is loaded since we'll need data from loading
        self._load()

        if self.__original_to_current_frame_idx:
            last_key = sorted(self.__original_to_current_frame_idx.keys())[-1]
            return last_key
        return self.frames - 1

    def reset(self):
        """Reloads the video."""
        # TODO
        pass

    def get_frame(self, idx) -> np.ndarray:
        """
        Get a frame from the underlying HDF5 video data.

        Args:
            idx: The index of the frame to get.

        Returns:
            The numpy.ndarray representing the video frame data.
        """
        # Ensure that video is loaded since we'll need data from loading
        self._load()

        # If we only saved some frames from a video, map to idx in dataset.
        if self.__original_to_current_frame_idx:
            if idx in self.__original_to_current_frame_idx:
                idx = self.__original_to_current_frame_idx[idx]
            else:
                return self._try_frame_from_source_video(idx)

        frame = self.__dataset_h5[idx]

        if self.__dataset_h5.attrs.get("format", ""):
            frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)

            # Add dimension for single channel (dropped by opencv).
            if frame.ndim == 2:
                frame = frame[..., np.newaxis]

        if self.input_format == "channels_first":
            frame = np.transpose(frame, (2, 1, 0))

        if self.convert_range and np.max(frame) <= 1.0:
            frame = (frame * 255).astype(int)

        return frame


@attr.s(auto_attribs=True, eq=False, order=False)
class MediaVideo:
    """
    Video data stored in traditional media formats readable by FFMPEG

    This class provides bare minimum read only interface on top of
    OpenCV's VideoCapture class.

    Args:
        filename: The name of the file (.mp4, .avi, etc)
        grayscale: Whether the video is grayscale or not. "auto" means detect
            based on first frame.
        bgr: Whether color channels ordered as (blue, green, red).
    """

    filename: str = attr.ib()
    grayscale: bool = attr.ib()
    bgr: bool = attr.ib(default=True)

    # Unused attributes still here so we don't break deserialization
    dataset: str = attr.ib(default="")
    input_format: str = attr.ib(default="")

    _detect_grayscale = False
    _reader_ = None
    _test_frame_ = None

    @property
    def __lock(self):
        if not hasattr(self, "_lock"):
            self._lock = multiprocessing.RLock()
        return self._lock

    @grayscale.default
    def __grayscale_default__(self):
        self._detect_grayscale = True
        return False

    @property
    def __reader(self):
        # Load if not already loaded
        if self._reader_ is None:
            if not os.path.isfile(self.filename):
                raise FileNotFoundError(
                    f"Could not find filename video filename named {self.filename}"
                )

            # Try and open the file either locally in current directory or with full
            # path
            self._reader_ = cv2.VideoCapture(self.filename)

            # If the user specified None for grayscale bool, figure it out based on the
            # the first frame of data.
            if self._detect_grayscale is True:
                self.grayscale = bool(
                    np.alltrue(self.test_frame[..., 0] == self.test_frame[..., -1])
                )

        # Return cached reader
        return self._reader_

    @property
    def __frames_float(self):
        return self.__reader.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def test_frame(self):
        # Load if not already loaded
        if self._test_frame_ is None:
            # Lets grab a test frame to help us figure things out about the video
            self._test_frame_ = self.get_frame(0, grayscale=False)

        # Return stored test frame
        return self._test_frame_

    def matches(self, other: "MediaVideo") -> bool:
        """
        Check if attributes match those of another video.

        Args:
            other: The other video to compare with.

        Returns:
            True if attributes match, False otherwise.
        """
        return (
            self.filename == other.filename
            and self.grayscale == other.grayscale
            and self.bgr == other.bgr
        )

    @property
    def fps(self) -> float:
        """Returns frames per second of video."""
        return self.__reader.get(cv2.CAP_PROP_FPS)

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def frames(self):
        """See :class:`Video`."""
        return int(self.__frames_float)

    @property
    def channels(self):
        """See :class:`Video`."""
        if self.grayscale:
            return 1
        else:
            return self.test_frame.shape[2]

    @property
    def width(self):
        """See :class:`Video`."""
        return self.test_frame.shape[1]

    @property
    def height(self):
        """See :class:`Video`."""
        return self.test_frame.shape[0]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.test_frame.dtype

    def reset(self):
        """Reloads the video."""
        self._reader_ = None

    def get_frame(self, idx: int, grayscale: bool = None) -> np.ndarray:
        """See :class:`Video`."""

        with self.__lock:
            if self.__reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
                self.__reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

            success, frame = self.__reader.read()

        if not success or frame is None:
            raise KeyError(f"Unable to load frame {idx} from {self}.")

        if grayscale is None:
            grayscale = self.grayscale

        if grayscale:
            frame = frame[..., 0][..., None]

        if self.bgr:
            frame = frame[..., ::-1]

        return frame


@attr.s(auto_attribs=True, eq=False, order=False)
class NumpyVideo:
    """
    Video data stored as Numpy array.

    Args:
        filename: Either a file to load or a numpy array of the data.

        * numpy data shape: (frames, height, width, channels)
    """

    filename: Union[str, np.ndarray] = attr.ib()

    def __attrs_post_init__(self):

        self.__frame_idx = 0
        self.__height_idx = 1
        self.__width_idx = 2
        self.__channel_idx = 3

        # Handle cases where the user feeds in np.array instead of filename
        if isinstance(self.filename, np.ndarray):
            self.__data = self.filename
            self.filename = "Raw Video Data"
        elif type(self.filename) is str:
            try:
                self.__data = np.load(self.filename)
            except OSError as ex:
                raise FileNotFoundError(
                    f"Could not find filename {self.filename}"
                ) from ex
        else:
            self.__data = None

    def set_video_ndarray(self, data: np.ndarray):
        self.__data = data

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def test_frame(self):
        return self.get_frame(0)

    def matches(self, other: "NumpyVideo") -> np.ndarray:
        """
        Check if attributes match those of another video.

        Args:
            other: The other video to compare with.

        Returns:
            True if attributes match, False otherwise.
        """
        return np.all(self.__data == other.__data)

    @property
    def frames(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__frame_idx]

    @property
    def channels(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__channel_idx]

    @property
    def width(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__width_idx]

    @property
    def height(self):
        """See :class:`Video`."""
        return self.__data.shape[self.__height_idx]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.__data.dtype

    def reset(self):
        """Reload the video."""
        # TODO
        pass

    def get_frame(self, idx):
        """See :class:`Video`."""
        return self.__data[idx]

    @property
    def is_missing(self) -> bool:
        """Return True if the video comes from a file and is missing."""
        if self.filename == "Raw Video Data":
            return False
        return not os.path.exists(self.filename)


@attr.s(auto_attribs=True, eq=False, order=False)
class ImgStoreVideo:
    """
    Video data stored as an ImgStore dataset.

    See: https://github.com/loopbio/imgstore
    This class is just a lightweight wrapper for reading such datasets as
    video sources for SLEAP.

    Args:
        filename: The name of the file or directory to the imgstore.
        index_by_original: ImgStores are great for storing a collection of
            selected frames from an larger video. If the index_by_original is
            set to True then the get_frame function will accept the original
            frame numbers of from original video. If False, then it will
            accept the frame index from the store directly.
            Default to True so that we can use an ImgStoreVideo in a dataset
            to replace another video without having to update all the frame
            indices on :class:`LabeledFrame` objects in the dataset.
    """

    filename: str = attr.ib(default=None)
    index_by_original: bool = attr.ib(default=True)
    _store_ = None
    _img_ = None

    def __attrs_post_init__(self):

        # If the filename does not contain metadata.yaml, append it to the filename
        # assuming that this is a directory that contains the imgstore.
        if "metadata.yaml" not in self.filename:
            # Use "/" since this works on Windows and posix
            self.filename = self.filename + "/metadata.yaml"

        # Make relative path into absolute, ImgStores don't work properly it seems
        # without full paths if we change working directories. Video.fixup_path will
        # fix this later when loading these datasets.
        self.filename = os.path.abspath(self.filename)

        self.__store = None

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    def matches(self, other):
        """
        Check if attributes match.

        Args:
            other: The instance to comapare with.

        Returns:
            True if attributes match, False otherwise
        """
        return (
            self.filename == other.filename
            and self.index_by_original == other.index_by_original
        )

    @property
    def __store(self):
        if self._store_ is None:
            self.open()
        return self._store_

    @__store.setter
    def __store(self, val):
        self._store_ = val

    @property
    def __img(self):
        if self._img_ is None:
            self.open()
        return self._img_

    @property
    def frames(self):
        """See :class:`Video`."""
        return self.__store.frame_count

    @property
    def channels(self):
        """See :class:`Video`."""
        if len(self.__img.shape) < 3:
            return 1
        else:
            return self.__img.shape[2]

    @property
    def width(self):
        """See :class:`Video`."""
        return self.__img.shape[1]

    @property
    def height(self):
        """See :class:`Video`."""
        return self.__img.shape[0]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.__img.dtype

    @property
    def last_frame_idx(self) -> int:
        """
        The idx number of the last frame.

        Overrides method of base :class:`Video` class for videos with
        select frames indexed by number from original video, since the last
        frame index here will not match the number of frames in video.
        """
        if self.index_by_original:
            return self.__store.frame_max
        return self.frames - 1

    def reset(self):
        """Reloads the video."""
        # TODO
        pass

    def get_frame(self, frame_number: int) -> np.ndarray:
        """
        Get a frame from the underlying ImgStore video data.

        Args:
            frame_number: The number of the frame to get. If
                index_by_original is set to True, then this number should
                actually be a frame index within the imgstore. That is,
                if there are 4 frames in the imgstore, this number should be
                be from 0 to 3.

        Returns:
            The numpy.ndarray representing the video frame data.
        """

        # Check if we need to open the imgstore and do it if needed
        if not self._store_:
            self.open()

        if self.index_by_original:
            img, (frame_number, frame_timestamp) = self.__store.get_image(frame_number)
        else:
            img, (frame_number, frame_timestamp) = self.__store.get_image(
                frame_number=None, frame_index=frame_number
            )

        # If the frame has one channel, add a singleton channel as it seems other
        # video implementations do this.
        if img.ndim == 2:
            img = img[:, :, None]

        return img

    @property
    def imgstore(self):
        """
        Get the underlying ImgStore object for this Video.

        Returns:
            The imgstore that is backing this video object.
        """
        return self.__store

    def open(self):
        """
        Open the image store if it isn't already open.

        Returns:
            None
        """
        if not self._store_:
            # Open the imgstore
            self._store_ = imgstore.new_for_filename(self.filename)

            # Read a frame so we can compute shape an such
            self._img_, (frame_number, frame_timestamp) = self._store_.get_next_image()

    def close(self):
        """
        Close the imgstore if it isn't already closed.

        Returns:
            None
        """
        if self.imgstore:
            # Open the imgstore
            self.__store.close()
            self.__store = None


@attr.s(auto_attribs=True, eq=False, order=False)
class SingleImageVideo:
    """
    Video wrapper for individual image files.

    Args:
        filenames: Files to load as video.
    """

    filename: Optional[str] = attr.ib(default=None)
    filenames: Optional[List[str]] = attr.ib(factory=list)
    height_: Optional[int] = attr.ib(default=None)
    width_: Optional[int] = attr.ib(default=None)
    channels_: Optional[int] = attr.ib(default=None)

    def __attrs_post_init__(self):
        if not self.filename and self.filenames:
            self.filename = self.filenames[0]
        elif self.filename and not self.filenames:
            self.filenames = [self.filename]

        self.__data = dict()
        self.test_frame_ = None

    def _load_idx(self, idx):
        img = cv2.imread(self._get_filename(idx))

        if img.shape[2] == 3:
            # OpenCV channels are in BGR order, so we should convert to RGB
            img = img[:, :, ::-1]
        return img

    def _get_filename(self, idx: int) -> str:
        f = self.filenames[idx]
        if os.path.exists(f):
            return f

        # Try the directory from the "video" file (this works if all the images
        # are in the same directory with distinctive filenames).
        f = os.path.join(os.path.dirname(self.filename), os.path.basename(f))
        if os.path.exists(f):
            return f

        raise FileNotFoundError(f"Unable to locate file {idx}: {self.filenames[idx]}")

    def _load_test_frame(self):
        if self.test_frame_ is None:
            self.test_frame_ = self._load_idx(0)

            if self.height_ is None:
                self.height_ = self.test_frame.shape[0]
            if self.width_ is None:
                self.width_ = self.test_frame.shape[1]
            if self.channels_ is None:
                self.channels_ = self.test_frame.shape[2]

    def get_idx_from_filename(self, filename: str) -> int:
        try:
            return self.filenames.index(filename)
        except IndexError:
            return None

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def test_frame(self) -> np.ndarray:
        self._load_test_frame()
        return self.test_frame_

    def matches(self, other: "SingleImageVideo") -> bool:
        """
        Check if attributes match those of another video.

        Args:
            other: The other video to compare with.

        Returns:
            True if attributes match, False otherwise.
        """
        return self.filenames == other.filenames

    @property
    def frames(self):
        """See :class:`Video`."""
        return len(self.filenames)

    @property
    def channels(self):
        """See :class:`Video`."""
        if self.channels_ is None:
            self._load_test_frame()

        return self.channels_

    @property
    def width(self):
        """See :class:`Video`."""
        if self.width_ is None:
            self._load_test_frame()

        return self.width_

    @width.setter
    def width(self, val):
        self.width_ = val

    @property
    def height(self):
        """See :class:`Video`."""
        if self.height_ is None:
            self._load_test_frame()

        return self.height_

    @height.setter
    def height(self, val):
        self.height_ = val

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.__data.dtype

    def reset(self):
        """Reloads the video."""
        # TODO
        pass

    def get_frame(self, idx):
        """See :class:`Video`."""
        if idx not in self.__data:
            self.__data[idx] = self._load_idx(idx)

        return self.__data[idx]


@attr.s(auto_attribs=True, eq=False, order=False)
class Video:
    """
    The top-level interface to any Video data used by SLEAP.

    This class provides a common interface for various supported video data
    backends. It provides the bare minimum of properties and methods that
    any video data needs to support in order to function with other SLEAP
    components. This interface currently only supports reading of video
    data, there is no write support. Unless one is creating a new video
    backend, this class should be instantiated from its various class methods
    for different formats. For example: ::

       >>> video = Video.from_hdf5(filename="test.h5", dataset="box")
       >>> video = Video.from_media(filename="test.mp4")

    Or we can use auto-detection based on filename: ::

       >>> video = Video.from_filename(filename="test.mp4")

    Args:
        backend: A backend is an object that implements the following basic
            required methods and properties

        * Properties

            * :code:`frames`: The number of frames in the video
            * :code:`channels`: The number of channels in the video
              (e.g. 1 for grayscale, 3 for RGB)
            * :code:`width`: The width of each frame in pixels
            * :code:`height`: The height of each frame in pixels

        * Methods

            * :code:`get_frame(frame_index: int) -> np.ndarray`:
              Get a single frame from the underlying video data with
              output shape=(height, width, channels).

    """

    backend: Union[
        HDF5Video, NumpyVideo, MediaVideo, ImgStoreVideo, SingleImageVideo, DummyVideo
    ] = attr.ib()

    # Delegate to the backend
    def __getattr__(self, item):
        return getattr(self.backend, item)

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the video."""
        return self.frames

    @property
    def last_frame_idx(self) -> int:
        """Return the index number of the last frame. Usually `num_frames - 1`."""
        if hasattr(self.backend, "last_frame_idx"):
            return self.backend.last_frame_idx
        return self.frames - 1

    @property
    def shape(
        self,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Return tuple of (frame count, height, width, channels)."""
        try:
            return (self.frames, self.height, self.width, self.channels)
        except:
            return (None, None, None, None)

    def __str__(self) -> str:
        """Informal string representation (for print or format)."""
        return (
            "Video("
            f"filename={self.filename}, "
            f"shape={self.shape}, "
            f"backend={type(self.backend).__name__}"
            ")"
        )

    def __len__(self) -> int:
        """Return the length of the video as the number of frames."""
        return self.frames

    @property
    def is_missing(self) -> bool:
        """Return True if the video is a file and is not present."""
        if not hasattr(self.backend, "filename"):
            return True
        elif hasattr(self.backend, "is_missing"):
            return self.backend.is_missing
        else:
            return not os.path.exists(self.backend.filename)

    def get_frame(self, idx: int) -> np.ndarray:
        """
        Return a single frame of video from the underlying video data.

        Args:
            idx: The index of the video frame

        Returns:
            The video frame with shape (height, width, channels)
        """
        return self.backend.get_frame(idx)

    def get_frames(self, idxs: Union[int, Iterable[int]]) -> np.ndarray:
        """Return a collection of video frames from the underlying video data.

        Args:
            idxs: An iterable object that contains the indices of frames.

        Returns:
            The requested video frames with shape (len(idxs), height, width, channels).
        """
        if np.isscalar(idxs):
            idxs = [idxs]
        return np.stack([self.get_frame(idx) for idx in idxs], axis=0)

    def get_frames_safely(self, idxs: Iterable[int]) -> Tuple[List[int], np.ndarray]:
        """Return list of frame indices and frames which were successfully loaded.

        idxs: An iterable object that contains the indices of frames.

        Returns: A tuple of (frame indices, frames), where
            * frame indices is a subset of the specified idxs, and
            * frames has shape (len(frame indices), height, width, channels).
            If zero frames were loaded successfully, then frames is None.
        """
        frames = []
        idxs_found = []

        for idx in idxs:
            try:
                frame = self.get_frame(idx)
            except Exception as e:
                print(e)
                # ignore frames which we couldn't load
                frame = None

            if frame is not None:
                frames.append(frame)
                idxs_found.append(idx)

        if frames:
            frames = np.stack(frames, axis=0)
        else:
            frames = None

        return idxs_found, frames

    def __getitem__(self, idxs):
        if isinstance(idxs, slice):
            start, stop, step = idxs.indices(self.num_frames)
            idxs = range(start, stop, step)
        return self.get_frames(idxs)

    @classmethod
    def from_hdf5(
        cls,
        dataset: Union[str, h5.Dataset],
        filename: Union[str, h5.File] = None,
        input_format: str = "channels_last",
        convert_range: bool = True,
    ) -> "Video":
        """
        Create an instance of a video object from an HDF5 file and dataset.

        This is a helper method that invokes the HDF5Video backend.

        Args:
            dataset: The name of the dataset or and h5.Dataset object. If
                filename is h5.File, dataset must be a str of the dataset name.
            filename: The name of the HDF5 file or and open h5.File object.
            input_format: Whether the data is oriented with "channels_first"
                or "channels_last"
            convert_range: Whether we should convert data to [0, 255]-range

        Returns:
            A Video object with HDF5Video backend.
        """
        filename = Video.fixup_path(filename)
        backend = HDF5Video(
            filename=filename,
            dataset=dataset,
            input_format=input_format,
            convert_range=convert_range,
        )
        return cls(backend=backend)

    @classmethod
    def from_numpy(cls, filename: Union[str, np.ndarray], *args, **kwargs) -> "Video":
        """
        Create an instance of a video object from a numpy array.

        Args:
            filename: The numpy array or the name of the file
            args: Arguments to pass to :class:`NumpyVideo`
            kwargs: Arguments to pass to :class:`NumpyVideo`

        Returns:
            A Video object with a NumpyVideo backend
        """
        filename = Video.fixup_path(filename)
        backend = NumpyVideo(filename=filename, *args, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_media(cls, filename: str, *args, **kwargs) -> "Video":
        """Create an instance of a video object from a typical media file.

        For example, mp4, avi, or other types readable by FFMPEG.

        Args:
            filename: The name of the file
            args: Arguments to pass to :class:`MediaVideo`
            kwargs: Arguments to pass to :class:`MediaVideo`

        Returns:
            A Video object with a MediaVideo backend
        """
        filename = Video.fixup_path(filename)
        backend = MediaVideo(filename=filename, *args, **kwargs)
        return cls(backend=backend)

    @classmethod
    def from_image_filenames(
        cls,
        filenames: List[str],
        height: Optional[int] = None,
        width: Optional[int] = None,
        *args,
        **kwargs,
    ) -> "Video":
        """Create an instance of a SingleImageVideo from individual image file(s)."""
        backend = SingleImageVideo(filenames=filenames)
        if height:
            backend.height = height
        if width:
            backend.width = width
        return cls(backend=backend)

    @classmethod
    def from_filename(cls, filename: str, *args, **kwargs) -> "Video":
        """Create an instance of a video object, auto-detecting the backend.

        Args:
            filename: The path to the video filename.
                Currently supported types are:

                * Media Videos - AVI, MP4, etc. handled by OpenCV directly
                * HDF5 Datasets - .h5 files
                * Numpy Arrays - npy files
                * imgstore datasets - produced by loopbio's Motif recording
                    system. See: https://github.com/loopbio/imgstore.

            args: Arguments to pass to :class:`NumpyVideo`
            kwargs: Arguments to pass to :class:`NumpyVideo`

        Returns:
            A Video object with the detected backend.
        """
        filename = Video.fixup_path(filename)

        if filename.lower().endswith(("h5", "hdf5", "slp")):
            backend_class = HDF5Video
        elif filename.endswith(("npy")):
            backend_class = NumpyVideo
        elif filename.lower().endswith(("mp4", "avi", "mov")):
            backend_class = MediaVideo
            kwargs["dataset"] = ""  # prevent serialization from breaking
        elif os.path.isdir(filename) or "metadata.yaml" in filename:
            backend_class = ImgStoreVideo
        else:
            raise ValueError("Could not detect backend for specified filename.")

        kwargs["filename"] = filename

        return cls(backend=cls.make_specific_backend(backend_class, kwargs))

    @classmethod
    def imgstore_from_filenames(
        cls, filenames: list, output_filename: str, *args, **kwargs
    ) -> "Video":
        """Create an imgstore from a list of image files.

        Args:
            filenames: List of filenames for the image files.
            output_filename: Filename for the imgstore to create.

        Returns:
            A `Video` object for the new imgstore.
        """
        # get the image size from the first file
        first_img = cv2.imread(filenames[0], flags=cv2.IMREAD_COLOR)
        img_shape = first_img.shape

        # create the imgstore
        store = imgstore.new_for_format(
            "png", mode="w", basedir=output_filename, imgshape=img_shape
        )

        # read each frame and write it to the imgstore
        # unfortunately imgstore doesn't let us just add the file
        for i, img_filename in enumerate(filenames):
            img = cv2.imread(img_filename, flags=cv2.IMREAD_COLOR)
            store.add_image(img, i, i)

        store.close()

        # Return an ImgStoreVideo object referencing this new imgstore.
        return cls(backend=ImgStoreVideo(filename=output_filename))

    def to_imgstore(
        self,
        path: str,
        frame_numbers: List[int] = None,
        format: str = "png",
        index_by_original: bool = True,
    ) -> "Video":
        """Convert frames from arbitrary video backend to ImgStoreVideo.

        This should facilitate conversion of any video to a loopbio imgstore.

        Args:
            path: Filename or directory name to store imgstore.
            frame_numbers: A list of frame numbers from the video to save.
                If None save the entire video.
            format: By default it will create a DirectoryImgStore with lossless
                PNG format unless the frame_indices = None, in which case,
                it will default to 'mjpeg/avi' format for video.
            index_by_original: ImgStores are great for storing a collection of
                selected frames from an larger video. If the index_by_original
                is set to True then the get_frame function will accept the
                original frame numbers of from original video. If False,
                then it will accept the frame index from the store directly.
                Default to True so that we can use an ImgStoreVideo in a
                dataset to replace another video without having to update
                all the frame indices on :class:`LabeledFrame` objects in the dataset.

        Returns:
            A new Video object that references the imgstore.
        """
        # If the user has not provided a list of frames to store, store them all.
        if frame_numbers is None:
            frame_numbers = range(self.num_frames)

            # We probably don't want to store all the frames as the PNG default,
            # lets use MJPEG by default.
            format = "mjpeg/avi"

        # Delete the imgstore if it already exists.
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

        # If the video is already an imgstore, we just need to copy it
        # if type(self) is ImgStoreVideo:
        #     new_backend = self.backend.copy_to(path)
        #     return self.__class__(backend=new_backend)

        store = imgstore.new_for_format(
            format,
            mode="w",
            basedir=path,
            imgshape=(self.height, self.width, self.channels),
            chunksize=1000,
        )

        # Write the JSON for the original video object to the metadata
        # of the imgstore for posterity
        store.add_extra_data(source_sleap_video_obj=Video.cattr().unstructure(self))

        import time

        for frame_num in frame_numbers:
            store.add_image(self.get_frame(frame_num), frame_num, time.time())

        # If there are no frames to save for this video, add a dummy frame
        # since we can't save an empty imgstore.
        if len(frame_numbers) == 0:
            store.add_image(
                np.zeros((self.height, self.width, self.channels)), 0, time.time()
            )

        store.close()

        # Return an ImgStoreVideo object referencing this new imgstore.
        return self.__class__(
            backend=ImgStoreVideo(filename=path, index_by_original=index_by_original)
        )

    def to_hdf5(
        self,
        path: str,
        dataset: str,
        frame_numbers: List[int] = None,
        format: str = "",
        index_by_original: bool = True,
    ):
        """Convert frames from arbitrary video backend to HDF5Video.

        Used for building an HDF5 that holds all data needed for training.

        Args:
            path: Filename to HDF5 (which could already exist).
            dataset: The HDF5 dataset in which to store video frames.
            frame_numbers: A list of frame numbers from the video to save.
                If None save the entire video.
            format: If non-empty, then encode images in format before saving.
                Otherwise, save numpy matrix of frames.
            index_by_original: If the index_by_original is set to True then
                the get_frame function will accept the original frame
                numbers of from original video.
                If False, then it will accept the frame index directly.
                Default to True so that we can use resulting video in a
                dataset to replace another video without having to update
                all the frame indices in the dataset.

        Returns:
            A new Video object that references the HDF5 dataset.
        """
        # If the user has not provided a list of frames to store, store them all.
        if frame_numbers is None:
            frame_numbers = range(self.num_frames)

        if frame_numbers:
            frame_data = self.get_frames(frame_numbers)
        else:
            frame_data = np.zeros((1, 1, 1, 1))

        frame_numbers_data = np.array(list(frame_numbers), dtype=int)

        with h5.File(path, "a") as f:

            if format:

                def encode(img):
                    _, encoded = cv2.imencode("." + format, img)
                    return np.squeeze(encoded)

                dtype = h5.special_dtype(vlen=np.dtype("int8"))
                dset = f.create_dataset(
                    dataset + "/video", (len(frame_numbers),), dtype=dtype
                )
                dset.attrs["format"] = format
                dset.attrs["channels"] = self.channels
                dset.attrs["height"] = self.height
                dset.attrs["width"] = self.width

                for i in range(len(frame_numbers)):
                    dset[i] = encode(frame_data[i])
            else:
                f.create_dataset(
                    dataset + "/video",
                    data=frame_data,
                    compression="gzip",
                    compression_opts=9,
                )

            if index_by_original:
                f.create_dataset(dataset + "/frame_numbers", data=frame_numbers_data)

            source_video_group = f.require_group(dataset + "/source_video")
            source_video_dict = Video.cattr().unstructure(self)
            source_video_group.attrs["json"] = json_dumps(source_video_dict)

        return self.__class__(
            backend=HDF5Video(
                filename=path,
                dataset=dataset + "/video",
                input_format="channels_last",
                convert_range=False,
            )
        )

    def to_pipeline(
        self,
        batch_size: Optional[int] = None,
        prefetch: bool = True,
        frame_indices: Optional[List[int]] = None,
    ) -> "sleap.pipelines.Pipeline":
        """Create a pipeline for reading the video.

        Args:
            batch_size: If not `None`, the video frames will be batched into rank-4
                tensors. Otherwise, single rank-3 images will be returned.
            prefetch: If `True`, pipeline will include prefetching.
            frame_indices: Frame indices to limit the pipeline reader to. If not
                specified (default), pipeline will read the entire video.

        Returns:
            A `sleap.pipelines.Pipeline` that builds `tf.data.Dataset` for high
            throughput I/O during inference.

        See also: sleap.pipelines.VideoReader
        """
        from sleap.nn.data import pipelines

        pipeline = pipelines.Pipeline(
            pipelines.VideoReader(self, example_indices=frame_indices)
        )
        if batch_size is not None:
            pipeline += pipelines.Batcher(
                batch_size=batch_size, drop_remainder=False, unrag=False
            )

        pipeline += pipelines.Prefetcher()
        return pipeline

    @staticmethod
    def make_specific_backend(backend_class, kwargs):
        # Only pass through the kwargs that match attributes for the backend
        attribute_kwargs = {
            key: val
            for (key, val) in kwargs.items()
            if key in attr.fields_dict(backend_class).keys()
        }

        return backend_class(**attribute_kwargs)

    @staticmethod
    def cattr():
        """Return a cattr converter for serialiazing/deserializing Video objects.

        Returns:
            A cattr converter.
        """

        # When we are structuring video backends, try to fixup the video file paths
        # in case they are coming from a different computer or the file has been moved.
        def fixup_video(x, cl):
            if "filename" in x:
                x["filename"] = Video.fixup_path(x["filename"])
            if "file" in x:
                x["file"] = Video.fixup_path(x["file"])

            return Video.make_specific_backend(cl, x)

        vid_cattr = cattr.Converter()

        # Check the type hint for backend and register the video path
        # fixup hook for each type in the Union.
        for t in attr.fields(Video).backend.type.__args__:
            vid_cattr.register_structure_hook(t, fixup_video)

        return vid_cattr

    @staticmethod
    def fixup_path(
        path: str, raise_error: bool = False, raise_warning: bool = False
    ) -> str:
        """Try to locate video if the given path doesn't work.

        Given a path to a video try to find it. This is attempt to make the
        paths serialized for different video objects portable across multiple
        computers. The default behavior is to store whatever path is stored
        on the backend object. If this is an absolute path it is almost
        certainly wrong when transferred when the object is created on
        another computer. We try to find the video by looking in the current
        working directory as well.

        Note that when loading videos during the process of deserializing a
        saved :class:`Labels` dataset, it's usually preferable to fix video
        paths using a `video_search` callback or path list.

        Args:
            path: The path the video asset.
            raise_error: Whether to raise error if we cannot find video.
            raise_warning: Whether to raise warning if we cannot find video.

        Raises:
            FileNotFoundError: If file still cannot be found and raise_error
                is True.

        Returns:
            The fixed up path
        """
        # If path is not a string then just return it and assume the backend
        # knows what to do with it.
        if type(path) is not str:
            return path

        if os.path.exists(path):
            return path

        # Strip the directory and lets see if the file is in the current working
        # directory.
        elif os.path.exists(os.path.basename(path)):
            return os.path.basename(path)

        # Special case: this is an ImgStore path! We cant use
        # basename because it will strip the directory name off
        elif path.endswith("metadata.yaml"):

            # Get the parent dir of the YAML file.
            img_store_dir = os.path.basename(os.path.split(path)[0])

            if os.path.exists(img_store_dir):
                return img_store_dir

        if raise_error:
            raise FileNotFoundError(f"Cannot find a video file: {path}")
        else:
            if raise_warning:
                logger.warning(f"Cannot find a video file: {path}")
            return path


def load_video(
    filename: str,
    grayscale: Optional[bool] = None,
    dataset=Optional[None],
    channels_first: bool = False,
) -> Video:
    """Open a video from disk.

    Args:
        filename: Path to a video file. The video reader backend will be determined by
            the file extension. Support extensions include: `.mp4`, `.avi`, `.h5`,
            `.hdf5` and `.slp` (for embedded images in a labels file). If the path to a
            folder is provided, images within that folder will be treated as video
            frames.
        grayscale: Read frames as a single channel grayscale images. If `None` (the
            default), this will be auto-detected.
        dataset: Name of the dataset that contains the video if loading a video stored
            in an HDF5 file. This has no effect for non-HDF5 inputs.
        channels_first: If `False` (the default), assume the data in the HDF5 dataset
            are formatted in `(frames, height, width, channels)` order. If `False`,
            assume the data are in `(frames, channels, width, height)` format. This has
            no effect for non-HDF5 inputs.

    Returns:
        A `sleap.Video` instance with the appropriate backend for its format.

        This enables numpy-like access to video data.

    Example: ::

        >>> video = sleap.load_video("centered_pair_small.mp4")
        >>> video.shape
        (1100, 384, 384, 1)
        >>> imgs = video[0:3]
        >>> imgs.shape
        (3, 384, 384, 1)

    See also:
        sleap.io.video.Video
    """
    kwargs = {}
    if grayscale is not None:
        kwargs["grayscale"] = grayscale
    if dataset is not None:
        kwargs["dataset"] = dataset
    kwargs["input_format"] = "channels_first" if channels_first else "channels_last"
    return Video.from_filename(filename, **kwargs)
