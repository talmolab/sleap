"""Tests for sleap.qc.features.appearance.

These tests cover the appearance-outlier detector (e): "points placed on
occluders / wrong object". The detector learns a per-node image-patch
descriptor distribution and flags nodes whose local pixels do not match what
that node usually sits on (e.g. a keypoint dragged onto bright cotton bedding
over a dark mouse).

Synthetic frames give us precise control over appearance: a node placed on a
region matching its learned appearance should score ~0, while a node moved onto
a visually-distinct patch (a bright square on a dark background) should score
high. Degenerate cases (too few samples, all-NaN, edge patches, undecodable
frames) must be safe. A final opt-in smoke test exercises the real grayscale
``train.slp`` end-to-end.
"""

import os

import numpy as np
import pytest

from sleap.qc.features.appearance import (
    DEFAULT_MIN_SAMPLES,
    extract_patch_descriptor,
    fit_appearance,
    score_appearance,
)


# ---------------------------------------------------------------------------
# Synthetic-frame helpers.
# ---------------------------------------------------------------------------

H, W = 64, 64
N_NODES = 3

# Per-node "home" location and intensity (grayscale background is dark = 20).
# Each node sits on a distinct-but-modest grey level so descriptors differ.
BG = 20
NODE_LOCS = [(16, 16), (32, 40), (48, 24)]  # (y, x)
NODE_GREY = [80, 130, 60]
BLOB_HALF = 5  # half-size of the painted region around each node


def _make_grayscale_frame(jitter: int = 0, seed: int = 0) -> np.ndarray:
    """Build a (H, W, 1) uint8 frame with a grey blob at each node's home.

    A small amount of additive noise (controlled by ``jitter``) makes the
    per-node descriptors non-degenerate so a real covariance can be fit.
    """
    rng = np.random.default_rng(seed)
    frame = np.full((H, W, 1), BG, dtype=np.float64)
    for (y, x), grey in zip(NODE_LOCS, NODE_GREY):
        frame[
            y - BLOB_HALF : y + BLOB_HALF + 1,
            x - BLOB_HALF : x + BLOB_HALF + 1,
            0,
        ] = grey
    if jitter:
        frame += rng.normal(0.0, jitter, size=frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _home_points() -> np.ndarray:
    """(N_NODES, 2) array placing every node on its home blob (x, y order)."""
    pts = np.full((N_NODES, 2), np.nan)
    for i, (y, x) in enumerate(NODE_LOCS):
        pts[i] = [x, y]
    return pts


def _build_training_set(n_frames: int = 30, seed0: int = 0) -> list[tuple]:
    """List of (frame, home_points) pairs for fitting."""
    pairs = []
    for k in range(n_frames):
        frame = _make_grayscale_frame(jitter=3, seed=seed0 + k)
        pairs.append((frame, _home_points()))
    return pairs


# ---------------------------------------------------------------------------
# extract_patch_descriptor.
# ---------------------------------------------------------------------------


class TestExtractPatchDescriptor:
    def test_grayscale_descriptor_length_two(self):
        frame = _make_grayscale_frame()
        desc = extract_patch_descriptor(frame, x=16, y=16, patch_size=7)
        assert desc is not None
        assert desc.shape == (2,)  # [mean, std]
        # Centre of node-0 blob -> mean near its grey level, std ~0.
        assert desc[0] == pytest.approx(NODE_GREY[0], abs=1.0)
        assert desc[1] == pytest.approx(0.0, abs=1.0)

    def test_rgb_descriptor_length_six(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # red channel high
        desc = extract_patch_descriptor(frame, x=16, y=16, patch_size=5)
        assert desc is not None
        assert desc.shape == (6,)  # [mean_rgb..., std_rgb...]
        assert desc[0] == pytest.approx(200.0, abs=1.0)
        assert desc[1] == pytest.approx(0.0, abs=1.0)

    def test_2d_frame_promoted_to_single_channel(self):
        frame2d = np.full((32, 32), 100, dtype=np.uint8)
        desc = extract_patch_descriptor(frame2d, x=10, y=10, patch_size=5)
        assert desc is not None
        assert desc.shape == (2,)
        assert desc[0] == pytest.approx(100.0, abs=1e-6)

    def test_bright_patch_distinct_from_dark(self):
        frame = _make_grayscale_frame()
        # Background (dark) vs node blob (brighter) -> clearly different means.
        dark = extract_patch_descriptor(frame, x=2, y=2, patch_size=7)
        bright = extract_patch_descriptor(frame, x=16, y=16, patch_size=7)
        assert dark[0] < bright[0]

    def test_edge_patch_is_clipped_not_error(self):
        frame = _make_grayscale_frame()
        # Corner node: patch extends off the top-left but should still return.
        desc = extract_patch_descriptor(frame, x=0, y=0, patch_size=7)
        assert desc is not None
        assert desc.shape == (2,)
        assert np.all(np.isfinite(desc))

    def test_nan_coordinate_returns_none(self):
        frame = _make_grayscale_frame()
        assert extract_patch_descriptor(frame, x=np.nan, y=10) is None
        assert extract_patch_descriptor(frame, x=10, y=np.nan) is None

    def test_centre_fully_outside_returns_none(self):
        frame = _make_grayscale_frame()
        assert extract_patch_descriptor(frame, x=1000, y=1000, patch_size=7) is None
        assert extract_patch_descriptor(frame, x=-50, y=-50, patch_size=7) is None

    def test_none_or_empty_frame_returns_none(self):
        assert extract_patch_descriptor(None, x=1, y=1) is None
        assert extract_patch_descriptor(np.zeros((0, 0, 1)), x=0, y=0) is None


# ---------------------------------------------------------------------------
# fit_appearance.
# ---------------------------------------------------------------------------


class TestFitAppearance:
    def test_models_nodes_with_enough_samples(self):
        # 30 frames -> 30 samples per node, above the default of 20.
        pairs = _build_training_set(n_frames=30)
        model = fit_appearance(pairs, n_nodes=N_NODES, patch_size=7)

        assert set(model["node_models"].keys()) == {0, 1, 2}
        for node_idx in range(N_NODES):
            assert model["node_sample_counts"][node_idx] == 30
            nm = model["node_models"][node_idx]
            assert nm["center"].shape == (2,)
            assert nm["inv_cov"].shape == (2, 2)
            assert nm["dist_scale"] > 0
            assert nm["n_samples"] == 30
        assert model["descriptor_dim"] == 2
        assert model["patch_size"] == 7
        assert model["n_nodes"] == N_NODES

    def test_node_below_min_samples_not_modeled(self):
        # Only 5 frames -> 5 samples/node, below default 20 -> no models.
        pairs = _build_training_set(n_frames=5)
        model = fit_appearance(pairs, n_nodes=N_NODES, patch_size=7)
        assert model["node_models"] == {}
        for node_idx in range(N_NODES):
            assert model["node_sample_counts"][node_idx] == 5

    def test_custom_min_samples_threshold(self):
        pairs = _build_training_set(n_frames=10)
        model = fit_appearance(pairs, n_nodes=N_NODES, patch_size=7, min_samples=5)
        # 10 >= 5 -> all nodes modeled.
        assert set(model["node_models"].keys()) == {0, 1, 2}

    def test_invisible_node_contributes_no_samples(self):
        # Node 1 is NaN in every frame -> never sampled, never modeled.
        pairs = []
        for k in range(30):
            frame = _make_grayscale_frame(jitter=3, seed=k)
            pts = _home_points()
            pts[1] = [np.nan, np.nan]
            pairs.append((frame, pts))
        model = fit_appearance(pairs, n_nodes=N_NODES, patch_size=7)
        assert model["node_sample_counts"][1] == 0
        assert 1 not in model["node_models"]
        assert {0, 2}.issubset(model["node_models"].keys())

    def test_none_frames_skipped(self):
        pairs = _build_training_set(n_frames=25)
        pairs.append((None, _home_points()))  # undecodable frame
        pairs.append(None)  # malformed pair
        model = fit_appearance(pairs, n_nodes=N_NODES, patch_size=7)
        # The bad entries are ignored; valid frames still produce models.
        assert set(model["node_models"].keys()) == {0, 1, 2}
        assert model["node_sample_counts"][0] == 25

    def test_empty_training_set(self):
        model = fit_appearance([], n_nodes=N_NODES, patch_size=7)
        assert model["node_models"] == {}
        assert model["descriptor_dim"] is None
        assert model["n_nodes"] == N_NODES

    def test_default_min_samples_constant(self):
        assert DEFAULT_MIN_SAMPLES == 20


# ---------------------------------------------------------------------------
# score_appearance.
# ---------------------------------------------------------------------------


class TestScoreAppearance:
    def _fit(self, n_frames=30):
        return fit_appearance(
            _build_training_set(n_frames=n_frames), n_nodes=N_NODES, patch_size=7
        )

    def test_matching_appearance_scores_low(self):
        model = self._fit()
        frame = _make_grayscale_frame(jitter=3, seed=999)
        result = score_appearance(frame, _home_points(), model)
        # All nodes on their learned blobs -> low outlier score.
        assert result["appearance_outlier_score"] < 0.3
        assert set(result["node_scores"].keys()) == {0, 1, 2}
        assert result["worst_node"] in (0, 1, 2)

    def test_node_on_bright_anomaly_scores_high(self):
        model = self._fit()
        # Paint a bright square far from any blob and drag node 0 onto it.
        frame = _make_grayscale_frame(jitter=3, seed=7)
        frame[4:14, 4:14, 0] = 255  # bright cotton-like patch on dark bg
        pts = _home_points()
        pts[0] = [9, 9]  # move node 0 onto the bright square

        result = score_appearance(frame, pts, model)
        assert result["worst_node"] == 0
        assert result["node_scores"][0] > 0.7
        assert result["appearance_outlier_score"] > 0.7
        # The untouched nodes (1, 2) remain low.
        assert result["node_scores"][1] < 0.3
        assert result["node_scores"][2] < 0.3

    def test_score_is_max_over_nodes(self):
        model = self._fit()
        frame = _make_grayscale_frame(jitter=3, seed=11)
        frame[4:14, 4:14, 0] = 255
        pts = _home_points()
        pts[2] = [9, 9]  # only node 2 is on the anomaly

        result = score_appearance(frame, pts, model)
        assert result["worst_node"] == 2
        assert result["appearance_outlier_score"] == pytest.approx(
            max(result["node_scores"].values())
        )

    def test_all_nan_instance_scores_zero(self):
        model = self._fit()
        frame = _make_grayscale_frame()
        pts = np.full((N_NODES, 2), np.nan)
        result = score_appearance(frame, pts, model)
        assert result["appearance_outlier_score"] == 0.0
        assert result["worst_node"] == -1
        assert result["node_scores"] == {}

    def test_none_frame_scores_zero(self):
        model = self._fit()
        result = score_appearance(None, _home_points(), model)
        assert result["appearance_outlier_score"] == 0.0
        assert result["worst_node"] == -1

    def test_empty_model_scores_zero(self):
        # Model with no node models (e.g. too few training samples).
        empty_model = fit_appearance(
            _build_training_set(n_frames=3), n_nodes=N_NODES, patch_size=7
        )
        assert empty_model["node_models"] == {}
        result = score_appearance(_make_grayscale_frame(), _home_points(), empty_model)
        assert result["appearance_outlier_score"] == 0.0
        assert result["worst_node"] == -1

    def test_none_model_scores_zero(self):
        result = score_appearance(_make_grayscale_frame(), _home_points(), None)
        assert result["appearance_outlier_score"] == 0.0

    def test_only_modeled_visible_nodes_scored(self):
        model = self._fit()
        frame = _make_grayscale_frame(jitter=3, seed=42)
        pts = _home_points()
        pts[1] = [np.nan, np.nan]  # node 1 invisible in this instance
        result = score_appearance(frame, pts, model)
        # Node 1 was modeled but is invisible here -> not scored.
        assert set(result["node_scores"].keys()) == {0, 2}

    def test_edge_node_scored_safely(self):
        model = self._fit()
        frame = _make_grayscale_frame(jitter=3, seed=5)
        pts = _home_points()
        pts[0] = [0, 0]  # node 0 dragged to the corner (clipped patch)
        result = score_appearance(frame, pts, model)
        # Must return a finite score without raising on the clipped patch.
        assert np.isfinite(result["appearance_outlier_score"])
        assert 0.0 <= result["appearance_outlier_score"] <= 1.0

    def test_scores_clamped_to_unit_interval(self):
        model = self._fit()
        frame = _make_grayscale_frame(jitter=3, seed=3)
        frame[4:14, 4:14, 0] = 255
        pts = _home_points()
        pts[0] = [9, 9]
        result = score_appearance(frame, pts, model)
        for s in result["node_scores"].values():
            assert 0.0 <= s <= 1.0
        assert 0.0 <= result["appearance_outlier_score"] <= 1.0

    def test_patch_size_defaults_to_model(self):
        model = self._fit()
        frame = _make_grayscale_frame(jitter=3, seed=8)
        # Not passing patch_size should reuse the fit-time size and work.
        result = score_appearance(frame, _home_points(), model)
        assert set(result["node_scores"].keys()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Real-data smoke test (opt-in; needs the local train.slp).
# ---------------------------------------------------------------------------

REAL_SLP = "/home/talmolab/als2h_0922CVATNN/train.slp"


@pytest.mark.skipif(not os.path.exists(REAL_SLP), reason="real train.slp not available")
def test_real_data_smoke():
    """End-to-end on real grayscale frames: fit on a subset, score a couple."""
    import sleap_io as sio

    labels = sio.load_slp(REAL_SLP)
    n_nodes = len(labels.skeletons[0].nodes)

    # Decode a handful of labeled frames (cap decode cost).
    pairs = []
    decoded = []
    for lf in labels:
        if not lf.user_instances:
            continue
        try:
            frame = lf.video[lf.frame_idx]
        except Exception:
            continue
        for inst in lf.user_instances:
            pairs.append((frame, inst.numpy(invisible_as_nan=True)))
        decoded.append((frame, lf))
        if len(decoded) >= 5:
            break

    assert pairs, "expected at least one labeled instance to decode"
    # Confirm real frames are grayscale (H, W, 1) uint8 as documented.
    assert decoded[0][0].ndim == 3

    # Lower min_samples so a 5-frame subset can actually model some nodes.
    model = fit_appearance(pairs, n_nodes=n_nodes, patch_size=7, min_samples=3)

    # Score the instances of the first couple decoded frames.
    n_scored = 0
    for frame, lf in decoded[:2]:
        for inst in lf.user_instances:
            res = score_appearance(frame, inst.numpy(invisible_as_nan=True), model)
            assert np.isfinite(res["appearance_outlier_score"])
            assert 0.0 <= res["appearance_outlier_score"] <= 1.0
            n_scored += 1
    assert n_scored > 0
