"""Tests for render-clip GUI plumbing.

Covers the bugs surfaced by rendering a multi-video predictions file:
  - Duplicated skeletons when a user-corrected ``Instance`` coexists with the
    ``PredictedInstance`` it was derived from.
  - Out-of-order output when ``Labels.labeled_frames`` is not stored in frame
    order (``sio.render_video`` writes frames in the order of ``frame_inds``).
"""

from __future__ import annotations

import numpy as np
import pytest
import sleap_io as sio

from sleap.gui.commands import _labels_with_visible_instances


def _skeleton() -> sio.Skeleton:
    return sio.Skeleton([sio.Node("a"), sio.Node("b")])


def _pred(skel: sio.Skeleton, xy: tuple[float, float]) -> sio.PredictedInstance:
    points = np.array([[xy[0], xy[1]], [xy[0] + 1.0, xy[1] + 1.0]])
    return sio.PredictedInstance.from_numpy(points, skeleton=skel, score=0.9)


def _user(
    skel: sio.Skeleton,
    xy: tuple[float, float],
    from_predicted: sio.PredictedInstance | None = None,
) -> sio.Instance:
    points = np.array([[xy[0], xy[1]], [xy[0] + 1.0, xy[1] + 1.0]])
    inst = sio.Instance.from_numpy(points, skeleton=skel)
    if from_predicted is not None:
        inst.from_predicted = from_predicted
    return inst


def test_labels_with_visible_instances_hides_used_predictions():
    skel = _skeleton()
    video_a = sio.Video(filename="a.mp4")
    video_b = sio.Video(filename="b.mp4")

    # Frame 0: prediction corrected by a user instance -> prediction hidden.
    pred_used = _pred(skel, (10.0, 20.0))
    user_corrected = _user(skel, (11.0, 21.0), from_predicted=pred_used)
    lf_corrected = sio.LabeledFrame(
        video=video_a, frame_idx=0, instances=[pred_used, user_corrected]
    )

    # Frame 1: only a raw prediction -> kept.
    pred_orphan = _pred(skel, (30.0, 40.0))
    lf_orphan = sio.LabeledFrame(video=video_a, frame_idx=1, instances=[pred_orphan])

    # Frame 2 (different video): prediction + user pair, must also be filtered
    # because we pass video=None (means "filter all videos").
    pred_used_b = _pred(skel, (50.0, 60.0))
    user_corrected_b = _user(skel, (51.0, 61.0), from_predicted=pred_used_b)
    lf_other_video = sio.LabeledFrame(
        video=video_b,
        frame_idx=5,
        instances=[pred_used_b, user_corrected_b],
    )

    labels = sio.Labels(
        labeled_frames=[lf_corrected, lf_orphan, lf_other_video],
        videos=[video_a, video_b],
        skeletons=[skel],
    )

    # Filter only video_a: video_b frame should pass through unchanged.
    filtered = _labels_with_visible_instances(labels, video_a)

    assert filtered is not labels
    assert filtered.labeled_frames is not labels.labeled_frames
    # Original must not be mutated.
    assert [type(i).__name__ for i in lf_corrected.instances] == [
        "PredictedInstance",
        "Instance",
    ]

    # video_a/frame 0: prediction hidden, only user instance visible.
    out_corrected = next(
        lf
        for lf in filtered.labeled_frames
        if lf.video is video_a and lf.frame_idx == 0
    )
    assert len(out_corrected.instances) == 1
    assert isinstance(out_corrected.instances[0], sio.Instance)
    assert not isinstance(out_corrected.instances[0], sio.PredictedInstance)

    # video_a/frame 1: orphan prediction kept.
    out_orphan = next(
        lf
        for lf in filtered.labeled_frames
        if lf.video is video_a and lf.frame_idx == 1
    )
    assert len(out_orphan.instances) == 1
    assert isinstance(out_orphan.instances[0], sio.PredictedInstance)

    # video_b frame: passthrough (same LabeledFrame object).
    out_other = next(lf for lf in filtered.labeled_frames if lf.video is video_b)
    assert out_other is lf_other_video


def test_labels_with_visible_instances_uses_track_when_present():
    """With tracks, the "used" check goes through tracks, not ``from_predicted``."""
    skel = _skeleton()
    video = sio.Video(filename="a.mp4")
    track = sio.Track(name="t0")

    pred = _pred(skel, (10.0, 20.0))
    pred.track = track
    user = _user(skel, (11.0, 21.0))
    user.track = track

    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred, user])
    labels = sio.Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skel],
        tracks=[track],
    )

    filtered = _labels_with_visible_instances(labels, video)

    out = filtered.labeled_frames[0]
    assert len(out.instances) == 1
    assert isinstance(out.instances[0], sio.Instance)
    assert not isinstance(out.instances[0], sio.PredictedInstance)


def test_labels_with_visible_instances_preserves_top_level_state():
    skel = _skeleton()
    video = sio.Video(filename="a.mp4")
    track = sio.Track(name="t0")
    pred = _pred(skel, (0.0, 0.0))
    user = _user(skel, (0.1, 0.1), from_predicted=pred)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[pred, user])
    labels = sio.Labels(
        labeled_frames=[lf],
        videos=[video],
        skeletons=[skel],
        tracks=[track],
        provenance={"source": "test"},
    )

    filtered = _labels_with_visible_instances(labels, video)

    # Shared references preserved; only labeled_frames is new.
    assert filtered.videos is labels.videos
    assert filtered.skeletons is labels.skeletons
    assert filtered.tracks is labels.tracks
    assert filtered.provenance == labels.provenance


def test_get_frame_indices_sorts_output(centered_pair_predictions):
    """``RenderClipDialog.get_frame_indices()`` must return frame indices in
    ascending order — ``Labels.labeled_frames`` storage is not guaranteed to
    be in frame order, and ``sio.render_video`` writes frames in the order it
    receives them.
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions

    # Shuffle labeled_frames to mimic the wild — sleap-io load order is not
    # guaranteed to be frame-sorted on all files (confirmed with multi-video
    # prediction files exported via the GUI).
    lfs = list(labels.labeled_frames)
    shuffled = [lfs[i] for i in (len(lfs) // 2, 0, *range(1, len(lfs) // 2))]
    shuffled += [lf for lf in lfs if lf not in shuffled]
    labels.labeled_frames = shuffled

    video = labels.videos[0]
    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        # Only the "all labeled frames" radio is exercised here — custom range
        # uses the same comprehension and sort.
        dialog.range_all.setChecked(True)
        indices = dialog.get_frame_indices()
    finally:
        dialog.deleteLater()

    assert indices == sorted(indices)
    assert len(indices) == sum(1 for lf in labels.labeled_frames if lf.video == video)


def test_match_source_fps_checkbox(centered_pair_predictions):
    """Dialog's match-source-fps checkbox should pin the FPS spinbox to the
    source video's reported FPS when checked, disable the spinbox, and restore
    manual editing when unchecked. Pre-corrections shipped a hardcoded
    ``fps=30`` default that silently slowed high-fps source footage (120+ fps
    in this project).
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions
    video = labels.videos[0]
    src_fps = video.backend.fps

    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        assert dialog.match_source_fps.isChecked()
        assert dialog.fps.value() == int(round(src_fps))
        assert not dialog.fps.isEnabled()

        # Unchecking re-enables the spinbox without changing its value.
        dialog.match_source_fps.setChecked(False)
        assert dialog.fps.isEnabled()
        dialog.fps.setValue(15)
        assert dialog.get_export_params()["fps"] == 15

        # Re-checking pins back to source fps.
        dialog.match_source_fps.setChecked(True)
        assert dialog.fps.value() == int(round(src_fps))
        assert not dialog.fps.isEnabled()
        assert dialog.get_export_params()["fps"] == int(round(src_fps))
    finally:
        dialog.deleteLater()


def test_include_unlabeled_returns_none_frame_indices(centered_pair_predictions):
    """When "Include unlabeled frames" is checked, ``get_frame_indices()``
    must return ``None`` so that ``sio.render_video()`` enumerates frames
    from the video instead of restricting output to the labeled-only list.
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions
    video = labels.videos[0]

    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        # Default is unchecked -> falls back to labeled-only behavior.
        assert not dialog.include_unlabeled.isChecked()
        assert isinstance(dialog.get_frame_indices(), list)

        # Once checked, the dialog hands over driver responsibility to sleap-io.
        dialog.include_unlabeled.setChecked(True)
        assert dialog.get_frame_indices() is None
    finally:
        dialog.deleteLater()


def test_include_unlabeled_export_params_forward_range(centered_pair_predictions):
    """Checking "Include unlabeled frames" should add ``include_unlabeled=True``
    to the export params, and a custom range should be forwarded with an
    exclusive ``end`` (sleap-io's convention) so the user's inclusive UI value
    maps correctly.
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions
    video = labels.videos[0]

    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        # Unchecked path: no include_unlabeled key, no start/end.
        params = dialog.get_export_params()
        assert "include_unlabeled" not in params
        assert "start" not in params
        assert "end" not in params

        # Checked + all-frames radio: include_unlabeled=True, no start/end so
        # sleap-io renders the whole target video.
        dialog.include_unlabeled.setChecked(True)
        dialog.range_all.setChecked(True)
        params = dialog.get_export_params()
        assert params["include_unlabeled"] is True
        assert "start" not in params
        assert "end" not in params

        # Checked + custom range: include_unlabeled=True with start/end+1 so
        # the inclusive UI bound maps to sleap-io's exclusive end.
        dialog.range_custom.setChecked(True)
        dialog.start_frame.setValue(5)
        dialog.end_frame.setValue(17)
        params = dialog.get_export_params()
        assert params["include_unlabeled"] is True
        assert params["start"] == 5
        assert params["end"] == 18
    finally:
        dialog.deleteLater()


def test_trail_params_absent_when_disabled(centered_pair_predictions):
    """With trails off (the default), no trail kwargs are forwarded so both the
    preview and export keep sleap-io's ``show_trails=False`` default.
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions
    video = labels.videos[0]

    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        assert not dialog.show_trails.isChecked()
        params = dialog.get_export_params()
        for key in (
            "show_trails",
            "trail_length",
            "trail_node",
            "trail_width",
            "trail_alpha_fade",
            "trail_alpha",
            "trail_color",
        ):
            assert key not in params
    finally:
        dialog.deleteLater()


def test_trail_params_forwarded_when_enabled(centered_pair_predictions):
    """Enabling trails forwards the full sleap-io trail kwarg set with values
    read from the widgets. ``trail_color`` is omitted for the "Match poses"
    default (sleap-io then colors trails by track/instance).
    """
    pytest.importorskip("qtpy.QtWidgets")

    from qtpy import QtWidgets

    from sleap.gui.dialogs.render_clip import RenderClipDialog

    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    labels: sio.Labels = centered_pair_predictions
    video = labels.videos[0]

    dialog = RenderClipDialog(labels=labels, video=video)
    try:
        # Node picker offers "Centroid" plus every skeleton node.
        node_items = [
            dialog.trail_node.itemData(i) for i in range(dialog.trail_node.count())
        ]
        assert node_items[0] == "centroid"
        for node in labels.skeletons[0].nodes:
            assert node.name in node_items

        # Sub-controls disabled until the master toggle is on.
        assert not dialog.trail_length.isEnabled()
        dialog.show_trails.setChecked(True)
        assert dialog.trail_length.isEnabled()

        dialog.trail_length.setValue(25)
        dialog.trail_node.setCurrentIndex(1)  # first real node
        dialog.trail_width.setValue(3.5)
        dialog.trail_alpha.setValue(0.5)
        dialog.trail_fade.setChecked(False)

        params = dialog.get_export_params()
        assert params["show_trails"] is True
        assert params["trail_length"] == 25
        assert params["trail_node"] == labels.skeletons[0].nodes[0].name
        assert params["trail_width"] == 3.5
        assert params["trail_alpha"] == 0.5
        assert params["trail_alpha_fade"] is False
        # "Match poses" -> no uniform color forwarded.
        assert "trail_color" not in params

        # A named color is forwarded verbatim.
        idx = dialog.trail_color.findData("red")
        dialog.trail_color.setCurrentIndex(idx)
        params = dialog.get_export_params()
        assert params["trail_color"] == "red"
    finally:
        dialog.deleteLater()


def test_trail_params_render_without_tracks():
    """The trail params the dialog produces must actually render — including for
    single-instance / untracked data, where sleap-io keys trails by instance
    position (no tracks required). This guards the end-to-end passthrough into
    ``sio.render_image``.
    """
    skel = _skeleton()
    video = sio.Video(filename="a.mp4")

    # Single moving instance across several frames, NO tracks assigned.
    lfs = [
        sio.LabeledFrame(
            video=video, frame_idx=i, instances=[_pred(skel, (10.0 + i, 20.0 + i))]
        )
        for i in range(6)
    ]
    labels = sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel])
    assert len(labels.tracks) == 0

    # Provide a solid background so no real video decode is needed.
    img = sio.render_image(
        labels,
        video=video,
        frame_idx=5,
        background="black",
        show_trails=True,
        trail_length=5,
        trail_node="centroid",
        trail_width=2.0,
        trail_alpha_fade=True,
        trail_alpha=1.0,
    )
    assert img is not None
    assert img.shape[-1] in (3, 4)
