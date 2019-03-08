"""A LEAP Dataset represents annotated (labeled) video data.

A LEAP Dataset stores almost all data required for training of a model.
This includes, raw video frame data, labelled instances of skeleton points,
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
        self._load_frames()
        self._load_instance_data()
        self._load_confidence_maps()
        self._load_skeleton()
        self._load_pafs()

        # Check if confidence maps were found, if not we will need to compute them
        # based on the point data.
        if not hasattr(self, 'confmaps') or self.confmaps is not None:
            logging.warning("Confidence maps not found in dataset. Need to compute them.")

        # Check if part affinity fields were found, if not we will need to compute
        # them based on points and skeleton data.
        if not hasattr(self, 'pafs') or self.pafs is not None:
            logging.warning("Part affinity fields not found in dataset. Need to compute them.")

    @abstractmethod
    def _load_frames(self):
        raise NotImplementedError()

    @abstractmethod
    def _load_confidence_maps(self):
        raise NotImplementedError()

    @abstractmethod
    def _load_skeleton(self):
        raise NotImplementedError()

    @abstractmethod
    def _load_instance_data(self):
        raise NotImplementedError()

    @abstractmethod
    def _load_pafs(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str, format: str = 'auto'):
        """
        Construct the :class:`.Dataset` from the file path. This may be a
        single file or directory depending on the storage backend.

        Args:
            path: The filasytem path to the file or folder that stores the dataset.
            format: The format the dataset is stored in. LEAP supports the following
            formats for its dataset files:

            * HDF5

            The default value of auto will attempt to detect the format automatically
            based on filename and contents. If it fails it will throw and exception.
        """

        # If format is auto, we need to try to detect the format automatically.
        # For now, lets look at the file extension
        path_ext = os.path.splitext(path)[-1].lower()

        if path_ext == '.h5' or path_ext == '.hdf5' or format.tolower() == 'hdf5':
            return DatasetHDF5(path=path)
        else:
            raise ValueError("Can't automatically find dataset file format. " +
                             "Are you sure this is a LEAP dataset?")

class DatasetHDF5(Dataset):
    """
    The :class:`.DatasetHDF5` class provides for reading and writing of HDF5 backed
    LEAP datasets.
    """

    # Class level constants that define the dataset paths within
    # the HDF5 data.
    _frames_dataset_name = "box"  # HDF5 dataset name for video frame data
    _confmaps_dataset_name = "confmaps"  # HDF5 dataset name for confidence map data
    _pafs_dataset_name = "pafs"  # HDF5 dataset name for part affinity field data
    _skelton_dataset_name = "skeleton"  # HDF5 dataset name for skeleton data
    _points_dataset_name = "points"  # HDF5 dataset name labeled points

    def __init__(self, path: str):

        # Open the HDF5 file for reading and call the base constructor to load all the
        # parts of the dataset.
        with h5py.File(path, "r") as self._h5_file:
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

    def _load_confidence_maps(self):
        """
        Loads and normalizes the confidence map data from the HDF5 dataset.

        Returns:
            None
        """
        try:
            # Load
            t0 = time()
            self.confmaps = self._h5_file[DatasetHDF5._confmaps_dataset_name][:]
            logging.info("Loaded %d frames of confidence map data [%.1fs]" % (len(self.frames), time() - t0))

            # Adjust dimensions
            t0 = time()
            self.confmaps = self._preprocess(self.confmaps, permute = (0, 3, 2, 1))
            logging.info("Permuted and normalized the confidence map data. [%.1fs]" % (time() - t0))

        except:
            # Part affinity field data might not be pre-computed, ignore exceptions.
            pass

    def _load_pafs(self):
        """
        Loads and the part affinity fields from the HDF5 dataset.

        Returns:
            None
        """
        try:
            # Load
            t0 = time()
            self.pafs = self._h5_file[DatasetHDF5._pafs_dataset_name][:]
            logging.info("Loaded %d frames of part affinity field (PAF) data [%.1fs]" % (len(self.frames), time() - t0))

            # Adjust dimensions
            t0 = time()
            self.pafs = self._preprocess(self.pafs, permute = (0, 3, 2, 1))
            logging.info("Permuted and normalized the part affinity field (PAF) map data. [%.1fs]" % (time() - t0))

        except:
            # Part affinity field data might not be pre-computed, ignore exceptions.
            pass


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
        self.skeleton = Skeleton.load_hdf5(self._h5_file[self. _skelton_dataset_name])

    def _load_instance_data(self):
        pass
