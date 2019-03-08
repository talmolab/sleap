""" Video reading and writing interfaces for different formats. """

from abc import ABC, abstractmethod
import h5py
import cv2
import numpy as np

class Video(ABC):
    """ Generic interface for video reading across formats. """
    def __init__(*args, **kwargs):
        pass

    @property
    def frames(self):
        return self.num_frames

    @property
    def shape(self):
        return (self.num_frames, self.height, self.width, self.channels)
    
    def __repr__(self):
        """ Formal string representation (Python code-like) """
        return type(self).__name__ + "([%d x %d x %d x %d])" % self.shape
    
    def __str__(self):
        """ Informal string representation (for print or format) """
        return type(self).__name__ + "([%d x %d x %d x %d])" % self.shape

    @abstractmethod
    def get_frame(self, idx):
        pass

    def get_frames(self, idxs):
        if np.isscalar(idxs):
            idxs = [idxs,]
        return np.stack([self.get_frame(idx) for idx in idxs], axis=0)

    def __getitem__(self, idxs):
        if isinstance(idxs, slice):
            start, stop, step = idxs.indices(self.num_frames)
            idxs = range(start, stop, step)
        return self.get_frames(idxs)


class HDF5Video(Video):
    """ Video reading for movies stored in HDF5 datasets. """
    def __init__(self, file_path, dset, input_format="channels_last", *args, **kwargs):
        super(Video, self).__init__(*args, **kwargs)

        if isinstance(file_path, h5py.File):
            self._file = file_path
        else:
            self._file = h5py.File(file_path, "r")
        self.dset = dset
        self._dataset = self._file[self.dset]
        self.input_format = input_format

        test_frame = self.get_frame(0)
        self.height, self.width, self.channels = test_frame.shape

        self.num_frames = len(self._dataset)

    def get_frame(self, idx):
        frame = self._dataset[idx]

        if self.input_format == "channels_first":
            frame = np.transpose(frame, (2, 1, 0))

        return frame


class MediaVideo(Video):
    """ Video reading for movies stored in traditional media files via OpenCV. """
    def __init__(self, file_path, grayscale="auto", *args, **kwargs):
        super(Video, self).__init__(*args, **kwargs)

        self._reader = cv2.VideoCapture(file_path)
        
        self.grayscale = grayscale

        test_frame = self.get_frame(0, grayscale=False)

        if self.grayscale == "auto":
            self.grayscale = np.alltrue(test_frame[...,0] == test_frame[...,-1])

        self.height, self.width, self.channels = test_frame.shape
        if self.grayscale:
            self.channels = 1

        self.num_frames = self._reader.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_frame(self, idx, grayscale=None):
        if grayscale is None:
            grayscale = self.grayscale

        if self._reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
            self._reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = self._reader.read()

        if grayscale:
            frame = frame[...,0][...,None]

        return frame


