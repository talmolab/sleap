import pytest
import os
import h5py

import numpy as np

from sleap.io.video import Video, HDF5Video, MediaVideo, DummyVideo, load_video
from tests.fixtures.videos import (
    TEST_H5_FILE,
    TEST_SMALL_ROBOT_MP4_FILE,
    TEST_H5_DSET,
    TEST_H5_INPUT_FORMAT,
    TEST_SMALL_CENTERED_PAIR_VID,
)

# FIXME:
# Parameterizing fixtures with fixtures is annoying so this leads to a lot
# of redundant test code here.
# See: https://github.com/pytest-dev/pytest/issues/349


def test_from_filename():
    assert type(Video.from_filename(TEST_H5_FILE).backend) == HDF5Video
    assert type(Video.from_filename(TEST_SMALL_ROBOT_MP4_FILE).backend) == MediaVideo


def test_backend_extra_kwargs():
    Video.from_filename(TEST_H5_FILE, grayscale=True, another_kwarg=False)
    Video.from_filename(
        TEST_SMALL_ROBOT_MP4_FILE, dataset="no dataset", fake_kwarg=True
    )


def test_grayscale_video():
    assert Video.from_filename(TEST_SMALL_ROBOT_MP4_FILE, grayscale=True).shape[-1] == 1


def test_hdf5_get_shape(hdf5_vid):
    assert hdf5_vid.shape == (42, 512, 512, 1)


def test_hdf5_len(hdf5_vid):
    assert len(hdf5_vid) == 42


def test_hdf5_dtype(hdf5_vid):
    assert hdf5_vid.dtype == np.uint8


def test_hdf5_get_frame(hdf5_vid):
    assert hdf5_vid.get_frame(0).shape == (512, 512, 1)


def test_hdf5_get_frames(hdf5_vid):
    assert hdf5_vid.get_frames(0).shape == (1, 512, 512, 1)
    assert hdf5_vid.get_frames([0, 1]).shape == (2, 512, 512, 1)


def test_hdf5_get_item(hdf5_vid):
    assert hdf5_vid[0].shape == (1, 512, 512, 1)
    assert np.alltrue(hdf5_vid[1:10:3] == hdf5_vid.get_frames([1, 4, 7]))


def test_hd5f_file_not_found():
    with pytest.raises(FileNotFoundError):
        Video.from_hdf5("non-existent-filename.h5", "dataset_name").height


def test_mp4_get_shape(small_robot_mp4_vid):
    assert small_robot_mp4_vid.shape == (166, 320, 560, 3)


def test_mp4_fps(small_robot_mp4_vid):
    assert small_robot_mp4_vid.fps == 30.0


def test_mp4_len(small_robot_mp4_vid):
    assert len(small_robot_mp4_vid) == 166


def test_mp4_dtype(small_robot_mp4_vid):
    assert small_robot_mp4_vid.dtype == np.uint8


def test_mp4_get_frame(small_robot_mp4_vid):
    assert small_robot_mp4_vid.get_frame(0).shape == (320, 560, 3)


def test_mp4_get_frames(small_robot_mp4_vid):
    assert small_robot_mp4_vid.get_frames(0).shape == (1, 320, 560, 3)
    assert small_robot_mp4_vid.get_frames([0, 1]).shape == (2, 320, 560, 3)


def test_mp4_get_item(small_robot_mp4_vid):
    assert small_robot_mp4_vid[0].shape == (1, 320, 560, 3)
    assert np.alltrue(
        small_robot_mp4_vid[1:10:3] == small_robot_mp4_vid.get_frames([1, 4, 7])
    )


def test_mp4_file_not_found():
    with pytest.raises(FileNotFoundError):
        vid = Video.from_media("non-existent-filename.mp4")
        vid.channels


def test_numpy_frames(small_robot_mp4_vid):
    clip_frames = small_robot_mp4_vid.get_frames((3, 7, 9))
    np_vid = Video.from_numpy(clip_frames)

    assert np.all(np.equal(np_vid.get_frame(1), small_robot_mp4_vid.get_frame(7)))


def test_is_missing():
    vid = Video.from_media(TEST_SMALL_ROBOT_MP4_FILE)
    assert not vid.is_missing
    vid = Video.from_media("non-existent-filename.mp4")
    assert vid.is_missing
    vid = Video.from_numpy(
        Video.from_media(TEST_SMALL_ROBOT_MP4_FILE).get_frames((3, 7, 9))
    )
    assert not vid.is_missing


@pytest.mark.parametrize("format", ["png", "jpg", "mjpeg/avi"])
def test_imgstore_video(small_robot_mp4_vid, tmpdir, format):

    path = os.path.join(tmpdir, "test_imgstore")

    # If format is video, test saving all the frames.
    if format == "mjpeg/avi":
        frame_indices = None
    else:
        frame_indices = [0, 1, 5]

    # Save and imgstore version of the first few frames of this
    # video.
    if format == "png":
        # Check that the default format is "png"
        imgstore_vid = small_robot_mp4_vid.to_imgstore(
            path, frame_numbers=frame_indices
        )
    else:
        imgstore_vid = small_robot_mp4_vid.to_imgstore(
            path, frame_numbers=frame_indices, format=format
        )

    if frame_indices is None:
        assert small_robot_mp4_vid.num_frames == imgstore_vid.num_frames

        # Make sure we can the first 10 frames, takes to long to read them all.
        for i in range(10):
            assert type(imgstore_vid.get_frame(i)) == np.ndarray

    else:
        assert imgstore_vid.num_frames == len(frame_indices)

        # Make sure we can read arbitrary frames by imgstore frame number
        for i in frame_indices:
            assert type(imgstore_vid.get_frame(i)) == np.ndarray

    assert imgstore_vid.channels == 3
    assert imgstore_vid.height == 320
    assert imgstore_vid.width == 560

    # Check the image data is exactly the same when lossless is used.
    if format == "png":
        assert np.allclose(
            imgstore_vid.get_frame(0), small_robot_mp4_vid.get_frame(0), rtol=0.91
        )


def test_imgstore_indexing(small_robot_mp4_vid, tmpdir):
    """
    Test different types of indexing (by frame number or index) supported
    by only imgstore videos.
    """
    path = os.path.join(tmpdir, "test_imgstore")

    frame_indices = [20, 40, 15]

    imgstore_vid = small_robot_mp4_vid.to_imgstore(
        path, frame_numbers=frame_indices, index_by_original=False
    )

    # Index by frame index in imgstore
    frames = imgstore_vid.get_frames([0, 1, 2])
    assert frames.shape == (3, 320, 560, 3)

    assert imgstore_vid.last_frame_idx == len(frame_indices) - 1

    with pytest.raises(ValueError):
        imgstore_vid.get_frames(frame_indices)

    # Now re-create the imgstore with frame number indexing, (the default)
    imgstore_vid = small_robot_mp4_vid.to_imgstore(path, frame_numbers=frame_indices)

    # Index by frame index in imgstore
    frames = imgstore_vid.get_frames(frame_indices)
    assert frames.shape == (3, 320, 560, 3)

    assert imgstore_vid.last_frame_idx == max(frame_indices)

    with pytest.raises(ValueError):
        imgstore_vid.get_frames([0, 1, 2])


def test_imgstore_deferred_loading(small_robot_mp4_vid, tmpdir):
    path = os.path.join(tmpdir, "test_imgstore")
    frame_indices = [20, 40, 15]
    vid = small_robot_mp4_vid.to_imgstore(path, frame_numbers=frame_indices)

    # This is actually testing that the __img will be loaded when needed,
    # since we use __img to get dtype.
    assert vid.dtype == np.dtype("uint8")


def test_imgstore_single_channel(centered_pair_vid, tmpdir):
    path = os.path.join(tmpdir, "test_imgstore")
    frame_indices = [20, 40, 15]
    vid = centered_pair_vid.to_imgstore(path, frame_numbers=frame_indices)

    assert vid.channels == 1


def test_imgstore_no_frames(small_robot_mp4_vid, tmpdir):
    path = os.path.join(tmpdir, "test_imgstore")
    frame_indices = []
    vid = small_robot_mp4_vid.to_imgstore(path, frame_numbers=frame_indices)

    # This is actually testing that the __img will be loaded when needed,
    # since we use __img to get dtype.
    assert vid.dtype == np.dtype("uint8")


def test_empty_hdf5_video(small_robot_mp4_vid, tmpdir):
    path = os.path.join(tmpdir, "test_to_hdf5")
    hdf5_vid = small_robot_mp4_vid.to_hdf5(path, "testvid", frame_numbers=[])


@pytest.mark.parametrize("format", ["", "png", "jpg"])
def test_hdf5_inline_video(small_robot_mp4_vid, tmpdir, format):

    path = os.path.join(tmpdir, f"test_to_hdf5_{format}")
    frame_indices = [0, 1, 5]

    # Save hdf5 version of the first few frames of this video.
    hdf5_vid = small_robot_mp4_vid.to_hdf5(
        path, "testvid", format=format, frame_numbers=frame_indices
    )
    assert hdf5_vid.num_frames == len(frame_indices)

    # Make sure we can read arbitrary frames by imgstore frame number
    for i in frame_indices:
        assert type(hdf5_vid.get_frame(i)) == np.ndarray

    assert hdf5_vid.channels == 3
    assert hdf5_vid.height == 320
    assert hdf5_vid.width == 560

    # Try loading a frame from the source video that's not in the inline video
    assert hdf5_vid.get_frame(3).shape == (320, 560, 3)

    # Check the image data is exactly the same when lossless is used.
    if format in ("", "png"):
        assert np.allclose(
            hdf5_vid.get_frame(0), small_robot_mp4_vid.get_frame(0), rtol=0.91
        )


def test_hdf5_indexing(small_robot_mp4_vid, tmpdir):
    """
    Test different types of indexing (by frame number or index).
    """
    path = os.path.join(tmpdir, "test_to_hdf5")

    frame_indices = [20, 40, 15]

    hdf5_vid = small_robot_mp4_vid.to_hdf5(
        path, dataset="testvid2", frame_numbers=frame_indices, index_by_original=False
    )

    # Index by frame index in newly saved video
    frames = hdf5_vid.get_frames([0, 1, 2])
    assert frames.shape == (3, 320, 560, 3)

    assert hdf5_vid.last_frame_idx == len(frame_indices) - 1

    # Disable loading frames from the original source video
    hdf5_vid.backend.enable_source_video = False

    with pytest.raises(IndexError):
        hdf5_vid.get_frames(frame_indices)

    # We have to close file before we can add another video dataset.
    hdf5_vid.close()

    # Now re-create the imgstore with frame number indexing (the default)
    hdf5_vid2 = small_robot_mp4_vid.to_hdf5(
        path, dataset="testvid3", frame_numbers=frame_indices
    )

    # Disable loading frames from the original source video
    assert hdf5_vid2.has_embedded_images
    assert hdf5_vid2.source_video_available
    hdf5_vid2.backend.enable_source_video = False
    assert hdf5_vid2.has_embedded_images
    assert not hdf5_vid2.source_video_available

    # Index by frame index in original video
    frames = hdf5_vid2.get_frames(frame_indices)
    assert frames.shape == (3, 320, 560, 3)

    assert hdf5_vid2.embedded_frame_inds == frame_indices

    assert hdf5_vid2.last_frame_idx == max(frame_indices)

    with pytest.raises(IndexError):
        hdf5_vid2.get_frames([0, 1, 2])


def test_hdf5_vid_from_open_dataset():
    with h5py.File(TEST_H5_FILE, "r") as f:
        dataset = f[TEST_H5_DSET]

        vid = Video(
            backend=HDF5Video(
                filename=f, dataset=dataset, input_format=TEST_H5_INPUT_FORMAT
            )
        )

        assert vid.shape == (42, 512, 512, 1)


def test_dummy_video():
    vid = Video(backend=DummyVideo("foo", 10, 20, 30, 3))

    assert vid.filename == "foo"
    assert vid.height == 10
    assert vid.width == 20
    assert vid.frames == 30
    assert vid.channels == 3

    assert vid[0].shape == (1, 10, 20, 3)


def test_images_video():
    filenames = [f"tests/data/videos/robot{i}.jpg" for i in range(3)]
    vid = Video.from_image_filenames(filenames)

    assert vid.frames == len(filenames)
    assert vid.height == 320
    assert vid.width == 560
    assert vid.channels == 3

    assert vid[0].shape == (1, 320, 560, 3)


def test_imgstore_from_filenames(tmpdir):
    temp_filename = os.path.join(tmpdir, "test_imgstore")
    filenames = [f"tests/data/videos/robot{i}.jpg" for i in range(3)]

    vid = Video.imgstore_from_filenames(filenames, temp_filename)

    assert vid.frames == len(filenames)
    assert vid.height == 320
    assert vid.width == 560
    assert vid.channels == 3

    assert vid[0].shape == (1, 320, 560, 3)


def test_safe_frame_loading(small_robot_mp4_vid):
    vid = small_robot_mp4_vid

    frame_count = vid.frames

    with pytest.raises(KeyError):
        vid.get_frames([1, 2, frame_count + 5])

    idxs, frames = vid.get_frames_safely([1, 2, frame_count + 5])

    assert idxs == [1, 2]
    assert len(frames) == 2


def test_numpy_video_backend():
    vid = Video.from_numpy(np.zeros((1, 2, 3, 1)))
    assert vid.test_frame.shape == (2, 3, 1)

    vid.backend.set_video_ndarray(np.ones((2, 3, 4, 1)))
    assert vid.get_frame(1).shape == (3, 4, 1)


def test_safe_frame_loading_all_invalid():
    vid = Video.from_filename("video_that_does_not_exist.mp4")

    idxs, frames = vid.get_frames_safely(list(range(10)))

    assert idxs == []
    assert frames is None


def test_load_video():
    video = load_video(TEST_SMALL_CENTERED_PAIR_VID)
    assert video.shape == (1100, 384, 384, 1)
    assert video[:3].shape == (3, 384, 384, 1)
