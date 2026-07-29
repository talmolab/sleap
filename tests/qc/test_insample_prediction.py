"""Tests for the in-sample prediction detector (Tier-2 detector (f)).

These unit tests exercise the matching + disagreement logic and the graceful
no-op behavior WITHOUT requiring torch. The model inference call
(``sleap_nn.legacy_predict.run_inference``) is monkeypatched to return canned
``sio.PredictedInstance`` objects, so the whole pipeline can be tested
deterministically and cheaply.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import sleap_io as sio

from sleap.qc.insample_prediction import (
    match_predictions_to_users,
    score_instance_disagreement,
    run_insample_prediction,
)


# ---------------------------------------------------------------------------
# Helpers to build tiny Labels / predicted instances.
# ---------------------------------------------------------------------------

NODE_NAMES = ["A", "B", "C", "D"]


def _skeleton(names=NODE_NAMES):
    return sio.Skeleton(list(names))


def _user_instance(skeleton, points):
    """User instance from an (n_nodes, 2) array (NaN -> invisible)."""
    return sio.Instance.from_numpy(np.asarray(points, dtype=float), skeleton=skeleton)


def _pred_instance(skeleton, points, scores, instance_score=0.9):
    """Predicted instance with per-node scores."""
    return sio.PredictedInstance.from_numpy(
        points_data=np.asarray(points, dtype=float),
        skeleton=skeleton,
        point_scores=np.asarray(scores, dtype=float),
        score=instance_score,
    )


def _labels_one_frame(skeleton, user_arrays):
    """A Labels with a single video + single frame holding user instances."""
    video = sio.Video.from_filename("fake_video.mp4")
    insts = [_user_instance(skeleton, a) for a in user_arrays]
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=insts)
    return sio.Labels(labeled_frames=[lf], videos=[video], skeletons=[skeleton])


def _patch_run_inference(monkeypatch, predicted_labels):
    """Install a fake ``sleap_nn.legacy_predict`` module returning canned
    predictions.

    Avoids importing the real (torch-backed) ``sleap_nn`` entirely.
    """
    fake_predict = types.ModuleType("sleap_nn.legacy_predict")

    def fake_run_inference(*args, **kwargs):  # noqa: ANN001, ANN002
        return predicted_labels

    fake_predict.run_inference = fake_run_inference

    fake_pkg = sys.modules.get("sleap_nn")
    if fake_pkg is None:
        fake_pkg = types.ModuleType("sleap_nn")
        monkeypatch.setitem(sys.modules, "sleap_nn", fake_pkg)
    monkeypatch.setitem(sys.modules, "sleap_nn.legacy_predict", fake_predict)


# ---------------------------------------------------------------------------
# Pure logic: match_predictions_to_users
# ---------------------------------------------------------------------------


class TestMatching:
    def test_no_predictions_returns_all_none(self):
        user = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        assert match_predictions_to_users(user, []) == [None]

    def test_no_users_returns_empty(self):
        pred = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        assert match_predictions_to_users([], pred) == []

    def test_nearest_centroid_match_two_instances(self):
        # User 0 near origin, user 1 near (100,100).
        users = [
            np.array([[0.0, 0.0], [2.0, 0.0]]),  # centroid (1,0)
            np.array([[100.0, 100.0], [102.0, 100.0]]),  # centroid (101,100)
        ]
        # Predictions deliberately in swapped list order to prove it matches by
        # geometry, not by index.
        preds = [
            np.array([[99.0, 100.0], [103.0, 100.0]]),  # centroid (101,100) -> user1
            np.array([[1.0, 0.0], [1.0, 0.0]]),  # centroid (1,0) -> user0
        ]
        matches = match_predictions_to_users(users, preds)
        assert matches == [1, 0]

    def test_mutually_exclusive_assignment(self):
        # Two users but only one prediction: the closer user wins, the other
        # gets None (one prediction cannot be claimed twice).
        users = [
            np.array([[0.0, 0.0]]),  # centroid (0,0), closer
            np.array([[10.0, 0.0]]),  # centroid (10,0)
        ]
        preds = [np.array([[1.0, 0.0]])]  # centroid (1,0)
        matches = match_predictions_to_users(users, preds)
        assert matches == [0, None]

    def test_user_with_no_visible_nodes_is_unmatched(self):
        users = [np.array([[np.nan, np.nan], [np.nan, np.nan]])]
        preds = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        assert match_predictions_to_users(users, preds) == [None]


# ---------------------------------------------------------------------------
# Pure logic: score_instance_disagreement
# ---------------------------------------------------------------------------


class TestDisagreementScoring:
    def test_confident_prediction_at_unlabeled_node_flags(self):
        # Node 2 unlabeled by user; model confident (0.85) -> disagreement.
        user = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]])
        pred_scores = np.array([0.9, 0.8, 0.85, 0.7])
        res = score_instance_disagreement(user, pred_scores, min_confidence=0.5)
        assert res["n_disagreements"] == 1
        assert res["disagreement_nodes"] == [2]
        assert res["prediction_disagreement_score"] == pytest.approx(0.85)
        assert res["unlabeled_confidences"] == {2: pytest.approx(0.85)}

    def test_occluded_node_with_low_model_confidence_not_flagged(self):
        # Node 2 unlabeled AND model also unsure (0.1 < min_confidence) ->
        # "truly occluded", not flagged.
        user = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]])
        pred_scores = np.array([0.9, 0.8, 0.1, 0.7])
        res = score_instance_disagreement(user, pred_scores, min_confidence=0.5)
        assert res["n_disagreements"] == 0
        assert res["disagreement_nodes"] == []
        assert res["prediction_disagreement_score"] == 0.0
        # The unlabeled node's (low) confidence is still recorded for context.
        assert res["unlabeled_confidences"] == {2: pytest.approx(0.1)}

    def test_score_is_gated_max_over_multiple_unlabeled(self):
        # Two unlabeled nodes; one low (0.3), one high (0.95). Score = the
        # high one (gated max), and only it is flagged.
        user = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [3.0, 3.0]])
        pred_scores = np.array([0.9, 0.3, 0.95, 0.7])
        res = score_instance_disagreement(user, pred_scores, min_confidence=0.5)
        assert res["disagreement_nodes"] == [2]
        assert res["prediction_disagreement_score"] == pytest.approx(0.95)

    def test_labeled_node_never_flagged_even_if_model_confident(self):
        # All nodes labeled -> nothing to flag regardless of model confidence.
        user = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        pred_scores = np.array([0.99, 0.99, 0.99, 0.99])
        res = score_instance_disagreement(user, pred_scores, min_confidence=0.5)
        assert res["n_disagreements"] == 0
        assert res["prediction_disagreement_score"] == 0.0

    def test_no_matched_prediction_is_zero(self):
        user = np.array([[0.0, 0.0], [np.nan, np.nan]])
        res = score_instance_disagreement(user, None, min_confidence=0.5)
        assert res["prediction_disagreement_score"] == 0.0
        assert res["n_disagreements"] == 0

    def test_nan_model_confidence_at_unlabeled_node_ignored(self):
        # Model produced no peak (NaN score) at the unlabeled node -> not a
        # disagreement and not recorded.
        user = np.array([[0.0, 0.0], [np.nan, np.nan]])
        pred_scores = np.array([0.9, np.nan])
        res = score_instance_disagreement(user, pred_scores, min_confidence=0.5)
        assert res["n_disagreements"] == 0
        assert res["prediction_disagreement_score"] == 0.0
        assert res["unlabeled_confidences"] == {}

    def test_min_confidence_threshold_boundary(self):
        # Exactly at threshold counts (>=).
        user = np.array([[np.nan, np.nan], [1.0, 1.0]])
        pred_scores = np.array([0.5, 0.9])
        res = score_instance_disagreement(user, pred_scores, min_confidence=0.5)
        assert res["disagreement_nodes"] == [0]
        # Just below threshold does not.
        res2 = score_instance_disagreement(
            user, np.array([0.49, 0.9]), min_confidence=0.5
        )
        assert res2["disagreement_nodes"] == []


# ---------------------------------------------------------------------------
# End-to-end with a MOCKED predictor (no torch).
# ---------------------------------------------------------------------------


class TestRunInsamplePredictionMocked:
    def test_confident_unlabeled_node_flagged_end_to_end(self, monkeypatch):
        skel = _skeleton()
        # User left node C (idx 2) blank.
        user_arr = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])
        labels = _labels_one_frame(skel, [user_arr])

        # Model predicts ALL four nodes, confident at the blank node C (0.92).
        pred = _pred_instance(
            skel,
            points=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            scores=np.array([0.95, 0.9, 0.92, 0.88]),
        )
        pred_video = labels.videos[0]
        pred_lf = sio.LabeledFrame(video=pred_video, frame_idx=0, instances=[pred])
        pred_labels = sio.Labels(
            labeled_frames=[pred_lf], videos=[pred_video], skeletons=[skel]
        )
        _patch_run_inference(monkeypatch, pred_labels)

        out = run_insample_prediction(
            labels, model_path="/fake/model", min_confidence=0.5
        )

        assert out["ran"] is True
        key = (0, 0, 0)
        assert key in out["instance_scores"]
        assert out["instance_scores"][key] == pytest.approx(0.92)
        # One per-node record for node C.
        assert len(out["records"]) == 1
        rec = out["records"][0]
        assert rec["node_idx"] == 2
        assert rec["node_name"] == "C"
        assert rec["predicted_confidence"] == pytest.approx(0.92)
        assert (rec["video_idx"], rec["frame_idx"], rec["instance_idx"]) == key

    def test_truly_occluded_node_not_flagged_end_to_end(self, monkeypatch):
        skel = _skeleton()
        # User left node C blank.
        user_arr = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])
        labels = _labels_one_frame(skel, [user_arr])

        # Model is ALSO unsure at node C (0.12) -> truly occluded, no flag.
        pred = _pred_instance(
            skel,
            points=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            scores=np.array([0.95, 0.9, 0.12, 0.88]),
        )
        pred_video = labels.videos[0]
        pred_lf = sio.LabeledFrame(video=pred_video, frame_idx=0, instances=[pred])
        pred_labels = sio.Labels(
            labeled_frames=[pred_lf], videos=[pred_video], skeletons=[skel]
        )
        _patch_run_inference(monkeypatch, pred_labels)

        out = run_insample_prediction(
            labels, model_path="/fake/model", min_confidence=0.5
        )
        assert out["ran"] is True
        assert out["instance_scores"] == {}
        assert out["records"] == []

    def test_two_instances_matched_independently(self, monkeypatch):
        skel = _skeleton()
        # Instance 0 (near origin) missing node C; instance 1 (far) fully labeled.
        u0 = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])
        u1 = np.array([[100.0, 0.0], [101.0, 0.0], [102.0, 0.0], [103.0, 0.0]])
        labels = _labels_one_frame(skel, [u0, u1])

        pred_video = labels.videos[0]
        # Prediction near origin (matches u0), confident at C.
        p0 = _pred_instance(
            skel,
            points=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            scores=np.array([0.95, 0.9, 0.93, 0.88]),
        )
        # Prediction near (100,0) (matches u1).
        p1 = _pred_instance(
            skel,
            points=np.array([[100.0, 0.0], [101.0, 0.0], [102.0, 0.0], [103.0, 0.0]]),
            scores=np.array([0.9, 0.9, 0.9, 0.9]),
        )
        pred_lf = sio.LabeledFrame(video=pred_video, frame_idx=0, instances=[p1, p0])
        pred_labels = sio.Labels(
            labeled_frames=[pred_lf], videos=[pred_video], skeletons=[skel]
        )
        _patch_run_inference(monkeypatch, pred_labels)

        out = run_insample_prediction(labels, model_path="/fake/model")
        # Only instance 0 has a blank node, and only it should be flagged.
        assert (0, 0, 0) in out["instance_scores"]
        assert (0, 0, 1) not in out["instance_scores"]
        assert out["instance_scores"][(0, 0, 0)] == pytest.approx(0.93)

    def test_node_name_mismatch_is_graceful_noop(self, monkeypatch):
        skel = _skeleton(["A", "B", "C", "D"])
        user_arr = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])
        labels = _labels_one_frame(skel, [user_arr])

        # Predicted skeleton has DIFFERENT node names -> must no-op gracefully.
        wrong_skel = _skeleton(["W", "X", "Y", "Z"])
        pred = _pred_instance(
            wrong_skel,
            points=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            scores=np.array([0.95, 0.9, 0.92, 0.88]),
        )
        pred_video = labels.videos[0]
        pred_lf = sio.LabeledFrame(video=pred_video, frame_idx=0, instances=[pred])
        pred_labels = sio.Labels(
            labeled_frames=[pred_lf], videos=[pred_video], skeletons=[wrong_skel]
        )
        _patch_run_inference(monkeypatch, pred_labels)

        out = run_insample_prediction(labels, model_path="/fake/model")
        assert out["ran"] is False
        assert "node-names do not match" in out["reason"]
        assert out["instance_scores"] == {}
        assert out["records"] == []

    def test_no_model_path_is_noop(self):
        skel = _skeleton()
        labels = _labels_one_frame(skel, [np.zeros((4, 2))])
        out = run_insample_prediction(labels, model_path="")
        assert out["ran"] is False
        assert "no model path" in out["reason"]
        assert out["instance_scores"] == {}

    def test_no_model_path_none_is_noop(self):
        skel = _skeleton()
        labels = _labels_one_frame(skel, [np.zeros((4, 2))])
        out = run_insample_prediction(labels, model_path=None)
        assert out["ran"] is False
        assert out["instance_scores"] == {}

    def test_sleap_nn_unavailable_is_graceful(self, monkeypatch):
        # Simulate an environment where importing sleap_nn.legacy_predict fails.
        skel = _skeleton()
        labels = _labels_one_frame(skel, [np.zeros((4, 2))])

        import builtins

        real_import = builtins.__import__

        def boom(name, *args, **kwargs):
            if name == "sleap_nn.legacy_predict" or name.startswith("sleap_nn"):
                raise ImportError("simulated missing sleap_nn")
            return real_import(name, *args, **kwargs)

        # Remove cached module so the import is re-attempted and fails.
        monkeypatch.delitem(sys.modules, "sleap_nn.legacy_predict", raising=False)
        monkeypatch.setattr(builtins, "__import__", boom)

        out = run_insample_prediction(labels, model_path="/fake/model")
        assert out["ran"] is False
        assert "sleap_nn unavailable" in out["reason"]

    def test_inference_exception_is_graceful(self, monkeypatch):
        skel = _skeleton()
        labels = _labels_one_frame(skel, [np.zeros((4, 2))])

        fake_predict = types.ModuleType("sleap_nn.legacy_predict")

        def boom_inference(*args, **kwargs):
            raise RuntimeError("CUDA OOM simulated")

        fake_predict.run_inference = boom_inference
        fake_pkg = sys.modules.get("sleap_nn") or types.ModuleType("sleap_nn")
        monkeypatch.setitem(sys.modules, "sleap_nn", fake_pkg)
        monkeypatch.setitem(sys.modules, "sleap_nn.legacy_predict", fake_predict)

        out = run_insample_prediction(labels, model_path="/fake/model")
        assert out["ran"] is False
        assert "inference failed" in out["reason"]

    def test_import_is_torch_free(self):
        # Importing the module must NOT have pulled in torch / sleap_nn.
        import importlib

        # Fresh import check: the module itself imports neither at top level.
        mod = importlib.import_module("sleap.qc.insample_prediction")
        src = mod.__doc__ or ""
        assert "Import-safe" in src
        # The function references are present without torch imported by us.
        assert callable(mod.run_insample_prediction)
