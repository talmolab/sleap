"""Tier 0.5 verify-only regression tests for sleap-io 0.8.0 metadata handling.

These lock in three coupled behaviors introduced in sleap-io 0.8.0:

- #483: ``save_slp(prefer_metadata=True)`` is now the DEFAULT. When a video's
  shape/grayscale/fps are recorded in ``backend_metadata`` (e.g. loaded from a
  ``.slp``), serialization prefers those recorded values instead of decoding a
  frame through the live backend.
- #490: a resolution-changing relink (``Video.replace_filename``) INVALIDATES the
  now-stale recorded shape/grayscale/fps so the new resolution is recomputed from
  the new backend. Without this fix, ``prefer_metadata=True`` (#483) would persist
  the OLD file's shape under the NEW filename.
- #495: flipping the grayscale flag after load updates the recorded grayscale but
  not the recorded shape; serializing them independently would emit a
  self-inconsistent entry (e.g. a 3-channel shape with ``grayscale=true``). The
  channel count is reconciled with the flag on save.

These tests drive the REAL sleap GUI relink path,
``sleap/sleap_io_adaptors/video_utils.py::video_util_reset`` (used by
``ReplaceVideo`` and the colormode toggle in ``sleap/gui/commands.py``), for both
the filename relink and the grayscale flip. The relink branch was fixed alongside
this upgrade to call ``Video.replace_filename(..., open=True)``: with the previous
``open=False`` the live backend kept serving the old file, so a relink-then-save
under the new ``prefer_metadata=True`` default (#483) persisted the OLD video's
resolution/grayscale under the new filename. These tests guard that fix.
"""

import inspect
import os

import pytest

import sleap_io as sio

from sleap.sleap_io_adaptors.video_utils import video_util_reset


def _videos_dir(any_video_path: str) -> str:
    """Return the directory holding the test video fixtures."""
    return os.path.dirname(any_video_path)


def _true_shape(path: str) -> list:
    """Return the on-disk shape of a video as a plain list (frames, H, W, C)."""
    return list(sio.Video.from_filename(path).shape)


def _build_and_reload(path: str, tmp_path, name: str) -> sio.Labels:
    """Create a Labels referencing ``path``, save (.slp) and reload it.

    Reloading is what stamps the video's shape/grayscale/fps into
    ``backend_metadata`` -- i.e. the realistic "opened a project" state in which
    ``prefer_metadata=True`` (#483) would serialize the recorded values.
    """
    slp_path = os.path.join(str(tmp_path), name)
    labels = sio.Labels(videos=[sio.Video.from_filename(path)])
    sio.save_file(labels, slp_path)
    return sio.load_file(slp_path)


def _persisted_shape(labels: sio.Labels) -> list:
    """Return the shape recorded in the (single) video's backend_metadata."""
    shape = labels.videos[0].backend_metadata.get("shape")
    return None if shape is None else list(shape)


def test_save_slp_prefer_metadata_default_is_true():
    """#483: prefer_metadata defaults to True on every save entry point."""
    from sleap_io.io.slp import write_labels
    from sleap_io.io.main import save_slp

    assert (
        inspect.signature(write_labels).parameters["prefer_metadata"].default is True
    )
    assert inspect.signature(save_slp).parameters["prefer_metadata"].default is True


def test_relink_invalidates_stale_resolution_fixture_pair(
    centered_pair_vid_path, small_robot_mp4_path, tmp_path
):
    """#490 + #483: relinking to a different-resolution video via the real GUI
    helper (video_util_reset) does NOT persist the stale recorded shape under the
    new filename.

    Reproduces the realistic flow: open a project (so the OLD shape is recorded in
    backend_metadata), Replace Video with a different-resolution file, then save
    with the default prefer_metadata=True. The channel count is re-read from the
    new file too, but this asserts on the resolution dims (frames, height, width).
    Guards the video_util_reset open=True fix: with open=False the live backend
    kept serving the old file and the save persisted A's [1100, 384, 384].
    """
    a_shape = _true_shape(centered_pair_vid_path)  # (1100, 384, 384, 1)
    b_shape = _true_shape(small_robot_mp4_path)  # (166, 320, 560, 3)
    # Guard the premise of the test: the two fixtures really differ in resolution.
    assert a_shape[1:3] != b_shape[1:3]

    # Realistic "opened project" state: the recorded shape is A's (this is what
    # #483 would otherwise persist verbatim if the relink left a stale backend).
    labels = _build_and_reload(centered_pair_vid_path, tmp_path, "labels_A.slp")
    assert _persisted_shape(labels)[1:3] == a_shape[1:3]

    # Replace Video with B via the real GUI helper (open=True reopens the backend).
    video_util_reset(labels.videos[0], filename=small_robot_mp4_path)

    out_path = os.path.join(str(tmp_path), "relinked.slp")
    sio.save_file(labels, out_path)  # DEFAULT prefer_metadata (no kwarg).
    persisted = _persisted_shape(sio.load_file(out_path))

    # The persisted resolution is the NEW video's, not the stale old one.
    assert persisted[1:3] == b_shape[1:3]
    assert persisted[1:3] != a_shape[1:3]
    # Frame count is also the new video's.
    assert persisted[0] == b_shape[0]


def test_relink_full_shape_matches_new_video_same_channels(
    small_robot_mp4_path, tmp_path
):
    """#490 + #483: with a same-channel relink via the real GUI helper the FULL
    persisted shape exactly matches the new video (no channel carryover to confound
    the comparison).

    small_robot.mp4 (166, 320, 560, 3) -> dance.mp4 (RGB, different resolution): so
    the entire recorded shape must be recomputed from the new file.
    """
    dance_path = os.path.join(_videos_dir(small_robot_mp4_path), "dance.mp4")
    if not os.path.exists(dance_path):
        pytest.skip("dance.mp4 fixture not present in this checkout")

    a_shape = _true_shape(small_robot_mp4_path)  # (166, 320, 560, 3)
    b_shape = _true_shape(dance_path)
    assert a_shape != b_shape
    assert a_shape[-1] == b_shape[-1] == 3  # same channel count

    labels = _build_and_reload(small_robot_mp4_path, tmp_path, "labels_sr.slp")
    assert _persisted_shape(labels) == a_shape  # recorded as small_robot

    # Replace Video with dance via the real GUI helper.
    video_util_reset(labels.videos[0], filename=dance_path)

    out_path = os.path.join(str(tmp_path), "relinked_full.slp")
    sio.save_file(labels, out_path)  # DEFAULT prefer_metadata.
    persisted = _persisted_shape(sio.load_file(out_path))

    # The full persisted shape equals the new video exactly, not the stale one.
    assert persisted == b_shape
    assert persisted != a_shape


def test_grayscale_flip_reconciles_channel_count(small_robot_mp4_path, tmp_path):
    """#495 + #483: flipping grayscale post-load (via the sleap video_util_reset
    path) yields a persisted shape whose channel count is reconciled with the
    grayscale flag, instead of an inconsistent 3-channel-shape + grayscale=True.
    """
    labels = _build_and_reload(small_robot_mp4_path, tmp_path, "labels_rgb.slp")
    video = labels.videos[0]

    # Reloaded as RGB: recorded shape has 3 channels, grayscale flag False.
    assert _persisted_shape(labels)[-1] == 3
    assert video.backend_metadata.get("grayscale") is False

    # Flip to grayscale via the real sleap helper (the colormode-toggle path).
    video_util_reset(video, grayscale=True)

    # In-memory the recorded shape is now self-inconsistent: it still has 3
    # channels even though the grayscale flag was flipped to True. This is the
    # exact stale state #495 reconciles at serialization time.
    assert video.backend_metadata.get("shape")[-1] == 3
    assert video.backend_metadata.get("grayscale") is True

    out_path = os.path.join(str(tmp_path), "grayscaled.slp")
    sio.save_file(labels, out_path)  # DEFAULT prefer_metadata.
    reloaded = sio.load_file(out_path)
    rvideo = reloaded.videos[0]

    # The persisted entry is consistent: a single channel matching grayscale=True.
    persisted = _persisted_shape(reloaded)
    assert persisted[-1] == 1
    assert rvideo.backend_metadata.get("grayscale") is True
    # And the live reloaded video reads as grayscale (single channel) too.
    assert rvideo.shape[-1] == 1
    # The spatial resolution is unchanged by the grayscale flip.
    assert persisted[1:3] == _true_shape(small_robot_mp4_path)[1:3]
