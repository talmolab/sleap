"""Integration tests for the B2 non-GMM channel wiring in LabelQCDetector.

These cover the two B2 channel modules once they are wired into the detector:

- the appearance-outlier channel (detector (e)): scored against a per-node
  image-appearance model built at fit time from decoded frames, surfaced via
  ``QCResults.channel_scores["appearance"]`` and labeled "Appearance outlier",
- the in-sample model-prediction channel (detector (f), Tier-2): a single
  batched ``run_insample_prediction`` call run after the per-instance loop,
  surfaced via ``QCResults.channel_scores["prediction"]`` and labeled "Model
  expects a labeled part here".

Both are non-GMM channels (default-OFF / experimental), wired exactly like the
existing missing-node channel. They MUST NOT change the fixed 22-wide feature
vector (they are channels, not GMM features). Real model inference is never run
here -- ``run_insample_prediction`` is monkeypatched, and the appearance model
is exercised either against tiny real decodable frames or via a monkeypatched
``fit_appearance``/``score_appearance`` to test the wiring in isolation.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import sleap_io as sio
from sleap_io import LabeledFrame
from sleap_io.model.instance import Instance

import sleap.qc.detector as detector_mod
from sleap.qc import LabelQCDetector, QCConfig, QCResults
from sleap.qc.detector import V3_FEATURE_NAMES
from sleap.qc.features.baseline import BASELINE_FEATURE_NAMES
from sleap.qc.results import CHANNEL_ISSUE_LABELS, InstanceKey


# ---------------------------------------------------------------------------
# Skeleton / pose / labels fixtures (mirror the B1 integration helpers)
# ---------------------------------------------------------------------------


def _line_skeleton(n: int = 6) -> sio.Skeleton:
    """A simple ``n``-node line graph 0-1-...-(n-1)."""
    names = [f"n{i}" for i in range(n)]
    skel = sio.Skeleton(names)
    skel.add_edges([(f"n{i}", f"n{i + 1}") for i in range(n - 1)])
    return skel


def _line_base(n: int = 6, spacing: float = 10.0) -> np.ndarray:
    return np.array([[i * spacing, 0.0] for i in range(n)], dtype=float)


def _labels_from_poses(skeleton: sio.Skeleton, poses: list[np.ndarray]) -> sio.Labels:
    """Build a single-video Labels with one instance per frame."""
    video = sio.Video.from_filename("test_video.mp4")
    labels = sio.Labels()
    for frame_idx, pts in enumerate(poses):
        inst = Instance.from_numpy(pts, skeleton=skeleton)
        labels.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst]))
    return labels


# ---------------------------------------------------------------------------
# Real decodable frames for the appearance channel
# ---------------------------------------------------------------------------


def _make_image_video(images: list[np.ndarray], tmpdir: str) -> sio.Video:
    """Write grayscale image frames to PNGs and wrap them in an ImageVideo.

    Decoding ``video[idx]`` returns an ``(H, W, 1)`` uint8 array, exactly the
    decoded-frame shape the appearance module expects, so this exercises the
    real ``fit()`` decode path without depending on a video codec.
    """
    try:
        import imageio.v3 as iio
    except Exception:  # pragma: no cover - older imageio
        import imageio as iio

    paths = []
    for i, img in enumerate(images):
        path = os.path.join(tmpdir, f"frame_{i:04d}.png")
        iio.imwrite(path, np.asarray(img, dtype=np.uint8))
        paths.append(path)
    return sio.Video.from_filename(paths)


def _appearance_labels_real_frames(tmpdir: str):
    """Build a Labels over real decodable frames with one planted outlier.

    A 3-node line skeleton is placed at fixed pixel locations in a dark
    background. For most frames every node sits on dark pixels; in the final
    (outlier) frame node 1 is dragged onto a bright square (the "wrong object"),
    so its appearance descriptor is far from the learned dark-pixel model.

    Returns ``(labels, skeleton, outlier_frame_idx, node_xy)``.
    """
    height, width = 40, 60
    n_frames = 40
    rng = np.random.default_rng(7)

    # Fixed node pixel locations (kept away from borders so full patches fit).
    node_xy = np.array([[12.0, 20.0], [30.0, 20.0], [48.0, 20.0]], dtype=float)

    images = []
    poses = []
    for _ in range(n_frames):
        # Dark, low-contrast background under every node.
        img = rng.integers(0, 25, size=(height, width), dtype=np.uint8)
        images.append(img)
        # Tiny jitter so descriptors are not perfectly identical.
        poses.append(node_xy + rng.normal(0, 0.3, size=node_xy.shape))

    # Outlier frame: paint a bright block under node 1's location.
    outlier_idx = n_frames
    out_img = rng.integers(0, 25, size=(height, width), dtype=np.uint8)
    cx, cy = int(round(node_xy[1, 0])), int(round(node_xy[1, 1]))
    out_img[cy - 5 : cy + 6, cx - 5 : cx + 6] = 250  # bright "wrong object"
    images.append(out_img)
    poses.append(node_xy.copy())

    skel = _line_skeleton(3)
    video = _make_image_video(images, tmpdir)
    labels = sio.Labels(videos=[video], skeletons=[skel])
    for frame_idx, pts in enumerate(poses):
        inst = Instance.from_numpy(pts, skeleton=skel)
        labels.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst]))
    return labels, skel, outlier_idx, node_xy


# ---------------------------------------------------------------------------
# Width invariance: the B2 channels must not change the 22-wide feature vector
# ---------------------------------------------------------------------------


class TestFeatureWidthInvariance:
    """The B2 channels are channels, not GMM features -> width stays 22."""

    EXPECTED_WIDTH = 22  # 12 baseline + 10 v3.

    @pytest.mark.parametrize(
        "config",
        [
            QCConfig(),  # both B2 channels OFF
            QCConfig(use_appearance=True),
            QCConfig(use_insample_prediction=True),
            QCConfig(use_appearance=True, use_insample_prediction=True),
        ],
    )
    def test_feature_vector_stays_22_with_b2_channels(self, config):
        """fit() + a single _extract_features still yield width-22 vectors.

        Holds with either or both B2 channels toggled on. (use_appearance with
        the synthetic .mp4 video here decodes nothing, so the appearance model
        is empty -- which is exactly fine for the width check.)
        """
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(0)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]

        detector = LabelQCDetector(config)
        detector.fit(_labels_from_poses(skel, poses))

        assert len(detector.feature_names) == self.EXPECTED_WIDTH
        assert V3_FEATURE_NAMES == [
            "max_curvature",
            "curvature_std",
            "visibility_pattern_score",
            "nn_distance",
            "hull_area_zscore",
            "hull_compactness",
            "chirality_wrong_fraction",
            "pose_split_score",
            "order_inversion_rate",
            "chain_intersection_count",
        ]
        feats = detector._extract_features(base)
        assert feats.shape == (self.EXPECTED_WIDTH,)
        total = len(BASELINE_FEATURE_NAMES) + len(V3_FEATURE_NAMES)
        assert total == self.EXPECTED_WIDTH


# ---------------------------------------------------------------------------
# Appearance channel
# ---------------------------------------------------------------------------


class TestAppearanceChannelRealFrames:
    """End-to-end appearance channel against tiny real decodable frames."""

    def test_outlier_populates_appearance_channel_and_surfaces_label(self):
        """A node dragged onto a bright block is flagged 'Appearance outlier'.

        Exercises the genuine fit-time decode path (``video[frame_idx]``), the
        per-node appearance model, and the score-time channel population +
        get_flagged label surfacing -- no mocking of the appearance module.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            labels, _skel, outlier_idx, _xy = _appearance_labels_real_frames(tmpdir)

            # min_samples low enough that 40 frames (40 patches/node) model it.
            config = QCConfig(use_appearance=True, appearance_min_samples=20)
            detector = LabelQCDetector(config)
            detector.fit(labels)

            # The appearance model actually learned at least node 1.
            assert detector._appearance_model is not None
            assert 1 in detector._appearance_model["node_models"]

            results = detector.score(labels)

            assert "appearance" in results.channel_scores
            chan = results.channel_scores["appearance"]
            out_key = InstanceKey(0, outlier_idx, 0)
            assert out_key in chan
            assert chan[out_key] > 0.0

            # The clean training frames sit on dark pixels -> near-zero outlier
            # score, so they should out-score the planted bright-block frame.
            assert chan[out_key] > max(
                (chan.get(InstanceKey(0, i, 0), 0.0) for i in range(40)),
                default=0.0,
            )

            # When the appearance channel wins for the outlier, get_flagged
            # surfaces it with the channel's human-readable label.
            flagged = detector.score(labels).get_flagged(
                threshold=min(chan[out_key], 0.99)
            )
            by_key = {f.instance_key: f for f in flagged}
            assert out_key in by_key
            assert by_key[out_key].top_issue == CHANNEL_ISSUE_LABELS["appearance"]


class TestAppearanceChannelWiring:
    """Appearance channel wiring tested with a monkeypatched module (no I/O)."""

    def test_channel_populated_via_monkeypatched_module(self, monkeypatch):
        """With canned fit/score values, the appearance channel is populated.

        Monkeypatches ``fit_appearance``/``score_appearance`` as imported into
        ``sleap.qc.detector`` so the test is independent of any real frame
        decoding (the synthetic .mp4 video here cannot be decoded). This pins
        the WIRING: a non-empty model is built, frames are decoded once, and a
        positive ``appearance_outlier_score`` lands in channel_scores.
        """
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(1)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(8)]
        outlier_idx = len(poses)
        poses.append(base.copy())
        labels = _labels_from_poses(skel, poses)

        # Canned model + a frame stub so the decode try/except yields non-None.
        canned_model = {"node_models": {0: {}}, "patch_size": 7}
        monkeypatch.setattr(
            detector_mod, "fit_appearance", lambda *a, **k: canned_model
        )

        # Make every frame decode to a tiny non-None array.
        fake_frame = np.zeros((4, 4, 1), dtype=np.uint8)
        for video in labels.videos:
            monkeypatch.setattr(
                type(video), "__getitem__", lambda self, idx: fake_frame, raising=False
            )

        # Only the outlier frame gets a positive score; the rest score 0.
        def fake_score(frame, points, model, patch_size=None):
            is_outlier = bool(np.allclose(np.nan_to_num(points), base))
            return {
                "appearance_outlier_score": 0.92 if is_outlier else 0.0,
                "worst_node": 0 if is_outlier else -1,
                "node_scores": {0: 0.92} if is_outlier else {},
            }

        monkeypatch.setattr(detector_mod, "score_appearance", fake_score)

        detector = LabelQCDetector(QCConfig(use_appearance=True))
        detector.fit(labels)
        assert detector._appearance_model is canned_model

        results = detector.score(labels)
        assert "appearance" in results.channel_scores
        out_key = InstanceKey(0, outlier_idx, 0)
        assert results.channel_scores["appearance"][out_key] == pytest.approx(0.92)
        # A zero-scoring frame is never recorded (mirrors missing_node).
        assert InstanceKey(0, 0, 0) not in results.channel_scores["appearance"]

    def test_appearance_off_by_default_no_model_no_channel(self, monkeypatch):
        """Default config builds no appearance model and never decodes frames."""
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(2)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(8)]
        labels = _labels_from_poses(skel, poses)

        # fit_appearance/score_appearance must never be called with the default.
        def _boom(*a, **k):  # pragma: no cover - asserts non-invocation
            raise AssertionError("appearance module called with use_appearance=False")

        monkeypatch.setattr(detector_mod, "fit_appearance", _boom)
        monkeypatch.setattr(detector_mod, "score_appearance", _boom)

        detector = LabelQCDetector(QCConfig())  # appearance OFF
        detector.fit(labels)
        assert detector._appearance_model is None

        results = detector.score(labels)
        assert "appearance" not in results.channel_scores


# ---------------------------------------------------------------------------
# In-sample model-prediction channel (run_insample_prediction is MOCKED)
# ---------------------------------------------------------------------------


class TestInsamplePredictionChannel:
    """In-sample prediction channel wiring; never runs real inference."""

    def test_channel_populated_via_monkeypatched_inference(self, monkeypatch):
        """Canned run_insample_prediction output lands in channel_scores.

        Monkeypatches ``run_insample_prediction`` (as imported into the
        detector) to return a fixed result, so no torch/sleap_nn or real
        inference is involved. Pins that the single batched call's
        ``instance_scores`` are copied into the ``"prediction"`` channel.
        """
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(3)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(8)]
        labels = _labels_from_poses(skel, poses)

        captured = {}

        def fake_run(labels_arg, **kwargs):
            captured["labels"] = labels_arg
            captured["kwargs"] = kwargs
            return {
                "ran": True,
                "reason": "",
                "instance_scores": {(0, 0, 0): 0.9},
                "records": [],
            }

        monkeypatch.setattr(detector_mod, "run_insample_prediction", fake_run)

        config = QCConfig(
            use_insample_prediction=True, insample_model_path="/fake/model"
        )
        detector = LabelQCDetector(config)
        detector.fit(labels)
        results = detector.score(labels)

        # Called exactly once, in-sample, with the labels + configured params.
        assert captured["labels"] is labels
        assert captured["kwargs"]["model_path"] == "/fake/model"
        assert captured["kwargs"]["peak_threshold"] == config.insample_peak_threshold
        assert captured["kwargs"]["min_confidence"] == config.insample_min_confidence
        assert captured["kwargs"]["device"] == config.insample_device

        # The canned score is surfaced under the "prediction" channel.
        assert "prediction" in results.channel_scores
        pred_key = InstanceKey(0, 0, 0)
        assert results.channel_scores["prediction"][pred_key] == pytest.approx(0.9)

    def test_insample_off_by_default_not_called(self, monkeypatch):
        """Default config never calls run_insample_prediction; no channel.

        This is the key safety guard: with use_insample_prediction=False (the
        default) the expensive inference path is not even invoked, and behavior
        is unchanged (no "prediction" channel).
        """
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(5)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(8)]
        labels = _labels_from_poses(skel, poses)

        called = {"count": 0}

        def fake_run(*a, **k):  # pragma: no cover - asserts non-invocation
            called["count"] += 1
            raise AssertionError("run_insample_prediction called with channel OFF")

        monkeypatch.setattr(detector_mod, "run_insample_prediction", fake_run)

        detector = LabelQCDetector(QCConfig())  # in-sample prediction OFF
        detector.fit(labels)
        results = detector.score(labels)

        assert called["count"] == 0
        assert "prediction" not in results.channel_scores
        # Behavior unchanged: every instance still scored as before.
        assert len(results.instance_scores) == len(poses)


# ---------------------------------------------------------------------------
# get_flagged label surfacing for the B2 channels
# ---------------------------------------------------------------------------


class TestB2ChannelLabelSurfacing:
    """A B2-channel-dominant key surfaces via get_flagged with the right label.

    Built on a hand-constructed QCResults (mirrors the B1 missing-node label
    test) so the channel cleanly out-scores the GMM. On the tiny synthetic
    datasets used elsewhere the GMM's normalized score saturates near 1.0, which
    would otherwise mask the channel label in an end-to-end run; the detector
    population of these channels is covered by the wiring tests above.
    """

    @pytest.mark.parametrize(
        "channel",
        ["appearance", "prediction"],
    )
    def test_channel_dominant_key_uses_channel_label(self, channel):
        results = QCResults(feature_names=["max_edge_zscore"])
        gmm_key = InstanceKey(0, 0, 0)  # GMM-only
        chan_only_key = InstanceKey(0, 1, 0)  # channel-only, no GMM score
        both_key = InstanceKey(0, 2, 0)  # both, channel dominant

        results.instance_scores = {gmm_key: 0.9, both_key: 0.4}
        results.feature_contributions = {
            gmm_key: {"max_edge_zscore": 5.0},
            both_key: {"max_edge_zscore": 1.0},
        }
        results.channel_scores = {channel: {chan_only_key: 0.85, both_key: 0.95}}

        flagged = results.get_flagged(threshold=0.7)
        by_key = {f.instance_key: f for f in flagged}

        # Channel-only key: flagged on the channel, with the channel label, and
        # absent feature contributions are tolerated (empty dict).
        assert chan_only_key in by_key
        assert by_key[chan_only_key].score == pytest.approx(0.85)
        assert by_key[chan_only_key].top_issue == CHANNEL_ISSUE_LABELS[channel]
        assert by_key[chan_only_key].feature_contributions == {}

        # Both present, channel wins -> channel label and the higher score.
        assert by_key[both_key].score == pytest.approx(0.95)
        assert by_key[both_key].top_issue == CHANNEL_ISSUE_LABELS[channel]

        # GMM-only key keeps its inferred (feature-based) issue.
        assert by_key[gmm_key].top_issue != CHANNEL_ISSUE_LABELS[channel]
