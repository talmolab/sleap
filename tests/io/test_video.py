import pytest
import os

import numpy as np

from sleap.io.video import Video
from tests.fixtures.videos import TEST_H5_FILE, TEST_SMALL_ROBOT_MP4_FILE

# FIXME:
# Parameterizing fixtures with fixtures is annoying so this leads to a lot
# of redundant test code here.
# See: https://github.com/pytest-dev/pytest/issues/349

def test_hdf5_get_shape(hdf5_vid):
    assert(hdf5_vid.shape == (42, 512, 512, 1))


def test_hdf5_len(hdf5_vid):
    assert(len(hdf5_vid) == 42)


def test_hdf5_dtype(hdf5_vid):
    assert(hdf5_vid.dtype == np.uint8)


def test_hdf5_get_frame(hdf5_vid):
    assert(hdf5_vid.get_frame(0).shape == (512, 512, 1))


def test_hdf5_get_frames(hdf5_vid):
    assert(hdf5_vid.get_frames(0).shape == (1, 512, 512, 1))
    assert(hdf5_vid.get_frames([0,1]).shape == (2, 512, 512, 1))


def test_hdf5_get_item(hdf5_vid):
    assert(hdf5_vid[0].shape == (1, 512, 512, 1))
    assert(np.alltrue(hdf5_vid[1:10:3] == hdf5_vid.get_frames([1, 4, 7])))

def test_hd5f_file_not_found():
    with pytest.raises(FileNotFoundError):
        Video.from_hdf5("non-existent-filename.h5", 'dataset_name')

def test_mp4_get_shape(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.shape == (166, 320, 560, 3))


def test_mp4_len(small_robot_mp4_vid):
    assert(len(small_robot_mp4_vid) == 166)


def test_mp4_dtype(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.dtype == np.uint8)


def test_mp4_get_frame(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.get_frame(0).shape == (320, 560, 3))


def test_mp4_get_frames(small_robot_mp4_vid):
    assert(small_robot_mp4_vid.get_frames(0).shape == (1, 320, 560, 3))
    assert(small_robot_mp4_vid.get_frames([0,1]).shape == (2, 320, 560, 3))


def test_mp4_get_item(small_robot_mp4_vid):
    assert(small_robot_mp4_vid[0].shape == (1, 320, 560, 3))
    assert(np.alltrue(small_robot_mp4_vid[1:10:3] == small_robot_mp4_vid.get_frames([1, 4, 7])))

def test_mp4_file_not_found():
    with pytest.raises(FileNotFoundError):
        vid = Video.from_media("non-existent-filename.mp4")
        vid.channels

def test_numpy_frames(small_robot_mp4_vid):
    clip_frames = small_robot_mp4_vid.get_frames((3,7,9))
    np_vid = Video.from_numpy(clip_frames)

    assert np.all(np.equal(np_vid.get_frame(1), small_robot_mp4_vid.get_frame(7)))

@pytest.mark.parametrize("format", ['png', 'jpg', "mjpeg/avi"])
def test_imgstore_video(small_robot_mp4_vid, tmpdir, format):

    path = os.path.join(tmpdir, 'test_imgstore')

    # If format is video, test saving all the frames.
    if format == "mjpeg/avi":
       frame_indices = None
    else:
        frame_indices = [0, 1, 5]

    # Save and imgstore version of the first few frames of this
    # video.
    if format == "png":
        # Check that the default format is "png"
        imgstore_vid = small_robot_mp4_vid.to_imgstore(path, frame_numbers=frame_indices)
    else:
        imgstore_vid = small_robot_mp4_vid.to_imgstore(path, frame_numbers=frame_indices, format=format)

    if frame_indices is None:
        assert small_robot_mp4_vid.num_frames == imgstore_vid.num_frames

        # Make sure we can the first 10 frames, takes to long to read them all.
        for i in range(10):
            assert type(imgstore_vid.get_frame(i)) == np.ndarray

    else:
        assert(imgstore_vid.num_frames == len(frame_indices))

        # Make sure we can read arbitrary frames by imgstore frame number
        for i in frame_indices:
            assert type(imgstore_vid.get_frame(i)) == np.ndarray

    assert(imgstore_vid.channels == 3)
    assert(imgstore_vid.height == 320)
    assert(imgstore_vid.width == 560)

    # Check the image data is exactly the same when lossless is used.
    if format == "png":
        assert np.allclose(imgstore_vid.get_frame(0), small_robot_mp4_vid.get_frame(0), rtol=0.91)


def test_imgstore_indexing(small_robot_mp4_vid, tmpdir):
    """
    Test different types of indexing (by frame number or index) supported
    by only imgstore videos.
    """
    path = os.path.join(tmpdir, 'test_imgstore')

    frame_indices = [20, 40, 15]

    imgstore_vid = small_robot_mp4_vid.to_imgstore(path, frame_numbers=frame_indices, index_by_original=False)

    # Index by frame index in imgstore
    frames = imgstore_vid.get_frames([0, 1, 2])
    assert frames.shape == (3, 320, 560, 3)

    with pytest.raises(ValueError):
        imgstore_vid.get_frames(frame_indices)

    # Now re-create the imgstore with frame number indexing, (the default)
    imgstore_vid = small_robot_mp4_vid.to_imgstore(path, frame_numbers=frame_indices)

    # Index by frame index in imgstore
    frames = imgstore_vid.get_frames(frame_indices)
    assert frames.shape == (3, 320, 560, 3)

    with pytest.raises(ValueError):
        imgstore_vid.get_frames([0, 1, 2])
