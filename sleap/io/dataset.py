"""A LEAP Dataset represents annotated (labeled) video data.

A LEAP Dataset stores almost all data required for training of a model.
This includes, raw video frame data, labelled instances of skeleton _points,
confidence maps, part affinity fields, and skeleton data. A LEAP :class:`.Dataset`
is a high level API to these data structures that abstracts away their underlying
storage format.

"""

import logging
import h5py
import os
import numpy as np

from time import time
from abc import ABC, abstractmethod

from sleap.skeleton import Skeleton
from sleap.instance import Instance


class Dataset(ABC):
    """
    The LEAP Dataset class represents an API for accessing labelled video
    frames and other associated metadata. This class is front-end for all
    interactions with loading, writing, and modifying a dataset. The actual
    storage backend for the data is mostly abstracted away from the main
    interface.
    """

    def __init__(self, path: str):
        """
        The constructor for any subclass of :class:`.DatasetBackend` should call
        this constructor to initiate loading of each component of a dataset.

        Args:
            path: The path to the file or folder containing the dataset.
        """

        self.path = path

        # Load all the components of the dataset.
        #self._load_frames()
        self._load_instance_data()
        self._load_skeleton()

    @abstractmethod
    def _load_frames(self):
        raise NotImplementedError()

    @abstractmethod
    def _load_skeleton(self):
        raise NotImplementedError()

    @abstractmethod
    def _load_instance_data(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str, format: str = 'auto', create:bool = True):
        """
        Construct the :class:`.Dataset` from the file path. This may be a
        single file or directory depending on the storage backend.

        Args:
            path: The filasytem path to the file or folder that stores the dataset.
            format: The format the dataset is stored in. LEAP supports the following
            formats for its dataset files:
            create: Should the dataset be created if it does not exist?

            * HDF5

            The default value of auto will attempt to detect the format automatically
            based on filename and contents. If it fails it will throw and exception.
        """

        # First things first, check if the dataset exists, if not, throw exception if
        # create is False
        if not create and os.path.isfile(path):
            raise FileNotFoundError(f"Could not load dataset {path}!")

        # If format is auto, we need to try to detect the format automatically.
        # For now, lets look at the file extension
        path_ext = os.path.splitext(path)[-1].lower()

        if path_ext == '.h5' or path_ext == '.hdf5' or format.tolower() == 'hdf5':
            return DatasetHDF5(path=path)
        else:
            raise ValueError("Can't automatically find dataset file format. " +
                             "Are you sure this is a LEAP dataset?")

    @abstractmethod
    def save(self):
        """
        Save the dataset to HDF5 file specified in self.path.

        Returns:
            None
        """
        raise NotImplementedError()


class DatasetHDF5(Dataset):
    """
    The :class:`.DatasetHDF5` class provides for reading and writing of HDF5 backed
    LEAP datasets.
    """

    # Class level constants that define the dataset paths within
    # the HDF5 data.
    skeleton_group_name = "skeleton"  # HDF5 dataset name for skeleton data
    points_group_name = "points"      # HDF5 dataset name labeled _points
    frames_group_name = "frames"      # HDF5 dataset name video frames

    def __init__(self, path: str):

        # Open the HDF5 file for reading and call the base constructor to load all the
        # parts of the dataset.
        with h5py.File(path) as self._h5_file:
            super(DatasetHDF5, self).__init__(path)

    def _load_frames(self):
        """
        Loads and normalizes the video frame data from the HDF5 dataset.

        Returns:
            None
        """

        try:
            # Load
            t0 = time()
            self.frames = self._h5_file[DatasetHDF5._frames_dataset_name][:]
            logging.info("Loaded %d video frames [%.1fs]" % (len(self.frames), time() - t0))

            # Adjust dimensions
            t0 = time()
            self.frames = self._preprocess(self.frames, permute = (0, 3, 2, 1))
            logging.info("Permuted and normalized video frame data. [%.1fs]" % (time() - t0))
        except Exception as e:
            raise ValueError("HDF5 format data did not have valid video frames data!") from e


    @staticmethod
    def _preprocess(x: np.ndarray, permute: tuple = None) -> np.ndarray:
        """
        Normalizes input data. Handles things like single images and unsigned integers.

        Args:
            x: A 4-D numpy array
            permute: A tuple specifying how to shift the dimensions of the array. None means leave be.

        Returns:
            The resulting data.
        """

        # Add singleton dim for single images
        if x.ndim == 3:
            x = x[None, ...]

        # Adjust dimensions
        if permute is not None:
            x = np.transpose(x, permute)

        # Normalize
        if x.dtype == "uint8":
            x = x.astype("float32") / 255

        return x

    def _load_skeleton(self):
        """
        Load the skeleton data into a skeleton object models.

        Returns:
            None
        """
        try:
            self.skeletons = Skeleton.load_all_hdf5(self._h5_file, return_dict=True)
        except KeyError:
            self.skeletons = {}

    def _load_instance_data(self):
        """
        Load the instance data.
        """
        try:
            self.instances = Instance.load_hdf5(file=self._h5_file)
        except KeyError:
            self.instances = []

    def save(self):
        """
        Save the dataset to HDF5 file specified in self.path.

        Returns:
            None
        """
        if hasattr(self, 'instances') and self.instances:
            Instance.save_hdf5(file=self.path, instances=self.instances)

        if hasattr(self, 'skeletons') and self.skeletons:
            Skeleton.save_all_hdf5(file=self._h5_file, skeletons=self.skeletons)

