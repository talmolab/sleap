"""In-sample model prediction: labelable-but-unlabeled points (Tier-2).

This module implements the *model-based* (Tier-2) variant of detector (f),
"labelable points left unlabeled" -- the maintainer's active-learning idea.

The geometry/visibility-only sibling
(:mod:`sleap.qc.features.missing_node`) can only catch *outlier* drops: an
instance missing a node its co-visible peers usually keep. It is blind to
dataset-wide systematic under-labeling (e.g. nobody ever labels the tail tip),
because the project's own visibility statistics never expect a node that is
rarely labeled.

This detector instead runs a *trained model* on the ALREADY-labeled frames
(in-sample inference). For each node a human left UNLABELED/invisible, it reads
the model's predicted confidence at the matched predicted instance's node and
flags the cases where the model *confidently* localizes a part the human left
blank. That distinguishes:

    * "truly occluded" -- the model is also unsure (low confidence), and
    * "labelable but skipped" -- the model is confident the part is there.

Design / scope:
    * **Import-safe.** Importing this module never imports ``torch`` or
      ``sleap_nn``; those heavy imports happen lazily inside
      :func:`run_insample_prediction`.
    * **Graceful.** If ``sleap_nn``/the model is unavailable, or the model's
      skeleton node-names do not match ``labels.skeletons[0]``, the function
      logs a reason and returns an empty/zero result instead of crashing.
    * **Non-GMM channel.** The per-instance ``prediction_disagreement_score`` is
      surfaced via :attr:`QCResults.channel_scores` (like the missing-node
      channel), not as a GMM feature. Default-OFF / experimental.

The matching + scoring logic (:func:`match_predictions_to_users` and
:func:`score_instance_disagreement`) is a pure, model-free core so it can be
unit-tested with canned predicted instances and without any torch dependency.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import sleap_io as sio

logger = logging.getLogger(__name__)


# Default score for an instance with no disagreement (or that could not be
# evaluated). Surfaced as the channel score; 0.0 means "nothing to flag".
_NO_DISAGREEMENT = 0.0


def _visible_mask(points: np.ndarray) -> np.ndarray:
    """Boolean ``(n_nodes,)`` mask of nodes with finite (visible) coordinates.

    A node is "visible/labeled" when both of its coordinates are finite. NaN
    (the convention for invisible/unlabeled nodes) -> ``False``.
    """
    points = np.asarray(points, dtype=float)
    return ~np.isnan(points[:, :2]).any(axis=1)


def _instance_centroid(points: np.ndarray) -> np.ndarray:
    """Mean of the visible node coordinates, or NaNs if none are visible."""
    points = np.asarray(points, dtype=float)
    mask = _visible_mask(points)
    if not mask.any():
        return np.array([np.nan, np.nan])
    return np.nanmean(points[mask, :2], axis=0)


def _predicted_scores(pred_points: np.ndarray, n_nodes: int) -> np.ndarray:
    """Extract the per-node confidence column from a predicted-instance array.

    ``sio.PredictedInstance.numpy(scores=True)`` returns an ``(n_nodes, 3)``
    array of ``(x, y, score)``. This pulls out the score column and pads/truncates
    defensively to ``n_nodes`` so a malformed prediction can never index out of
    range against the user skeleton. Missing entries -> ``NaN`` (treated as "no
    prediction").

    Args:
        pred_points: ``(n_nodes, 3)`` predicted array of ``(x, y, score)``.
        n_nodes: Expected node count (from the user skeleton).

    Returns:
        ``(n_nodes,)`` float array of per-node confidences (``NaN`` where the
        model produced no usable score).
    """
    pred_points = np.asarray(pred_points, dtype=float)
    scores = np.full(n_nodes, np.nan, dtype=float)
    if pred_points.ndim != 2 or pred_points.shape[1] < 3:
        return scores
    k = min(n_nodes, pred_points.shape[0])
    scores[:k] = pred_points[:k, 2]
    return scores


def match_predictions_to_users(
    user_points: list[np.ndarray],
    pred_points: list[np.ndarray],
) -> list[Optional[int]]:
    """Match each USER instance to the nearest PREDICTED instance in a frame.

    Bottom-up (and top-down) models emit predicted instances with no guaranteed
    correspondence to the user instances in the same frame, so we associate them
    by spatial proximity: each user instance is paired with the predicted
    instance whose visible-node centroid is closest (greedy nearest, one
    predicted instance per user instance, mutually exclusive).

    The matching is symmetric in spirit to "match each predicted instance to the
    nearest user instance" -- a greedy global nearest-centroid assignment -- but
    is indexed by user instance so the caller can directly look up the prediction
    for a given user instance.

    Args:
        user_points: List of ``(n_nodes, 2)`` user pose arrays (NaN = invisible).
        pred_points: List of ``(n_nodes, >=2)`` predicted pose arrays. Only the
            first two columns (x, y) are used for centroid matching.

    Returns:
        A list parallel to ``user_points``: ``match[i]`` is the index into
        ``pred_points`` of the predicted instance matched to user instance ``i``,
        or ``None`` if no prediction could be matched (no predictions in the
        frame, or no usable centroid).
    """
    n_user = len(user_points)
    matches: list[Optional[int]] = [None] * n_user
    if n_user == 0 or len(pred_points) == 0:
        return matches

    user_centroids = [_instance_centroid(p) for p in user_points]
    pred_centroids = [_instance_centroid(p) for p in pred_points]

    # Build all valid (distance, user_idx, pred_idx) pairs, then assign greedily
    # from the closest pair outward. This is O(U*P) which is tiny per frame.
    pairs: list[tuple[float, int, int]] = []
    for ui, uc in enumerate(user_centroids):
        if np.isnan(uc).any():
            continue
        for pi, pc in enumerate(pred_centroids):
            if np.isnan(pc).any():
                continue
            dist = float(np.hypot(uc[0] - pc[0], uc[1] - pc[1]))
            pairs.append((dist, ui, pi))

    pairs.sort(key=lambda t: t[0])
    used_pred: set[int] = set()
    assigned_user: set[int] = set()
    for _dist, ui, pi in pairs:
        if ui in assigned_user or pi in used_pred:
            continue
        matches[ui] = pi
        assigned_user.add(ui)
        used_pred.add(pi)

    return matches


def score_instance_disagreement(
    user_points: np.ndarray,
    pred_scores: Optional[np.ndarray],
    min_confidence: float = 0.5,
) -> dict:
    """Score how strongly a model disagrees with a user's *unlabeled* nodes.

    For each node the user left invisible/unlabeled (NaN), look up the matched
    predicted instance's confidence at that node. A node is a *disagreement* when
    the model is confident the part is present::

        disagreement at node k  <=>  user node k is invisible
                                      AND pred_scores[k] >= min_confidence

    The per-instance ``prediction_disagreement_score`` is the maximum predicted
    confidence over the instance's unlabeled nodes, but only counting nodes whose
    confidence clears ``min_confidence`` (so it is gated -- a model that is mildly
    unsure about every blank node yields 0.0). The result is therefore in
    ``[0, 1]`` and 0.0 when the model never confidently fills a human-left blank.

    Args:
        user_points: ``(n_nodes, 2)`` user pose array (NaN = invisible/unlabeled).
        pred_scores: ``(n_nodes,)`` per-node predicted confidence for the matched
            predicted instance, or ``None`` if the user instance had no matched
            prediction. ``NaN`` entries mean "model produced no peak there".
        min_confidence: Confidence at/above which a model prediction at an
            unlabeled node counts as a disagreement. Defaults to ``0.5``.

    Returns:
        Dictionary with:

        - ``prediction_disagreement_score``: ``float`` in ``[0, 1]`` (gated max,
          see above).
        - ``disagreement_nodes``: ``list[int]`` of flagged node indices
          (ascending) -- nodes the user left blank that the model localized at
          ``>= min_confidence``.
        - ``unlabeled_confidences``: ``dict[int, float]`` mapping every unlabeled
          node index to the model's confidence there (NaN-confidence nodes
          omitted). Useful for explanations / per-node records.
        - ``n_disagreements``: ``int`` count of flagged nodes.
    """
    user_points = np.asarray(user_points, dtype=float)
    n_nodes = user_points.shape[0]
    invisible = ~_visible_mask(user_points)
    invisible_idx = np.where(invisible)[0]

    empty = {
        "prediction_disagreement_score": _NO_DISAGREEMENT,
        "disagreement_nodes": [],
        "unlabeled_confidences": {},
        "n_disagreements": 0,
    }

    # No unlabeled nodes, or no matched prediction -> nothing to evaluate.
    if len(invisible_idx) == 0 or pred_scores is None:
        return empty

    pred_scores = np.asarray(pred_scores, dtype=float)
    if pred_scores.shape[0] != n_nodes:
        # Shape mismatch (should not happen given _predicted_scores padding);
        # be defensive and report nothing rather than mis-index.
        return empty

    unlabeled_confidences: dict[int, float] = {}
    disagreement_nodes: list[int] = []
    confident_values: list[float] = []
    for k in invisible_idx:
        conf = pred_scores[k]
        if not np.isfinite(conf):
            continue
        unlabeled_confidences[int(k)] = float(conf)
        if conf >= min_confidence:
            disagreement_nodes.append(int(k))
            confident_values.append(float(conf))

    score = float(np.max(confident_values)) if confident_values else _NO_DISAGREEMENT
    # Clamp defensively: predicted confidences are nominally in [0, 1] but
    # refinement could overshoot by a hair.
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "prediction_disagreement_score": score,
        "disagreement_nodes": sorted(disagreement_nodes),
        "unlabeled_confidences": unlabeled_confidences,
        "n_disagreements": len(disagreement_nodes),
    }


def _skeleton_node_names(skeleton) -> list[str]:
    """Best-effort list of node names from a sleap-io skeleton."""
    nodes = getattr(skeleton, "nodes", None)
    if nodes is None:
        return []
    names: list[str] = []
    for n in nodes:
        name = getattr(n, "name", None)
        names.append(name if isinstance(name, str) else str(n))
    return names


def _video_key(video, video_idx: int) -> str:
    """Stable, hashable identifier for a video (mirrors LabelQCDetector).

    ``Video.filename`` is a list for image-sequence backends, which is
    unhashable, so fall back to the video index.
    """
    filename = getattr(video, "filename", None)
    if isinstance(filename, str) and filename:
        return filename
    return str(video_idx)


def _empty_result(reason: str) -> dict:
    """Build a no-op result and log why we bailed out."""
    logger.info("In-sample prediction detector: %s", reason)
    return {
        "instance_scores": {},
        "records": [],
        "ran": False,
        "reason": reason,
    }


def run_insample_prediction(
    labels: "sio.Labels",
    model_path: str,
    peak_threshold: float = 0.2,
    min_confidence: float = 0.5,
    device: str = "auto",
    progress_callback=None,
) -> dict:
    """Flag labelable-but-unlabeled points via in-sample model prediction.

    Runs a trained ``sleap_nn`` model on the ALREADY-labeled frames of
    ``labels`` (in-sample), matches each predicted instance to the nearest user
    instance, and -- for every node the user left invisible -- reads the matched
    prediction's confidence there. A confident prediction at a blank node is a
    "disagreement" (the model expects a labeled part the human skipped).

    This is the model-based (Tier-2) variant of detector (f). It is import-safe
    (no top-level torch/``sleap_nn`` import) and graceful: any failure to load
    or run the model, or a skeleton node-name mismatch, returns a zero result
    with ``ran=False`` and a logged ``reason`` rather than raising.

    Args:
        labels: Labels with user-annotated instances to evaluate (in-sample).
        model_path: Path to a trained ``sleap_nn`` model directory (containing
            ``best.ckpt`` + ``training_config.yaml``). If falsy (``None``/empty),
            the detector no-ops -- callers should not run this stage without a
            configured model path.
        peak_threshold: Minimum peak confidence for the model's peak finding.
            Lower values let the model report weaker peaks (more candidate
            disagreements). Passed through to inference. Defaults to ``0.2``.
        min_confidence: Confidence at/above which a model prediction at an
            unlabeled node counts as a disagreement (gates the per-instance
            score). Defaults to ``0.5``.
        device: Torch device for inference (``"auto"``/``"cpu"``/``"cuda"``/
            ``"mps"``). Defaults to ``"auto"``.
        progress_callback: Optional callable ``(step, fraction, detail)`` for
            progress reporting (matches ``LabelQCDetector`` callbacks). May be
            ``None``.

    Returns:
        Dictionary with:

        - ``instance_scores``: ``dict[(video_idx, frame_idx, instance_idx),
          float]`` of per-user-instance ``prediction_disagreement_score`` in
          ``[0, 1]``. Only instances with a non-zero score are included (so the
          integration layer can copy them straight into
          ``QCResults.channel_scores["prediction"]``).
        - ``records``: ``list[dict]`` of per-node disagreement records, one per
          ``(video_idx, frame_idx, instance_idx, node_idx)`` where the user left
          the node blank and the model was confident. Each record has keys
          ``video_idx, frame_idx, instance_idx, node_idx, node_name,
          predicted_confidence``.
        - ``ran``: ``bool`` -- whether inference actually ran.
        - ``reason``: ``str`` -- why it did not run (empty when ``ran`` is True).
    """

    def _report(step: str, fraction: float, detail: Optional[str] = None) -> None:
        if progress_callback is not None:
            progress_callback(step, fraction, detail)

    if not model_path:
        return _empty_result("no model path configured; skipping (no-op)")

    if not labels.skeletons:
        return _empty_result("labels have no skeleton; skipping")

    user_skeleton = labels.skeletons[0]
    user_node_names = _skeleton_node_names(user_skeleton)
    n_nodes = len(user_node_names)
    if n_nodes == 0:
        return _empty_result("user skeleton has no nodes; skipping")

    # --- Lazy, guarded model load + inference. All torch/sleap_nn touching code
    # lives below this point so importing this module stays cheap and safe. ---
    _report("In-sample prediction", 0.0, "Loading model")
    try:
        from sleap_nn.legacy_predict import run_inference
    except Exception as e:  # pragma: no cover - environment-dependent
        return _empty_result(f"sleap_nn unavailable ({e!r}); skipping")

    # ``run_inference`` ALWAYS writes the predicted Labels to disk when
    # make_labels=True (defaulting to ``./results.predictions.slp``), with no
    # flag to disable it. We are only after the in-memory predictions, so direct
    # the output into a throwaway temp dir and remove it afterward to avoid
    # littering the user's working directory.
    try:
        with tempfile.TemporaryDirectory(prefix="sleap_qc_insample_") as tmpdir:
            out_path = os.path.join(tmpdir, "insample.predictions.slp")
            predicted_labels = run_inference(
                input_labels=labels,
                model_paths=[model_path],
                only_labeled_frames=True,
                exclude_user_labeled=False,
                peak_threshold=peak_threshold,
                make_labels=True,
                device=device,
                output_path=out_path,
            )
    except Exception as e:
        return _empty_result(f"inference failed ({e!r}); skipping")

    if predicted_labels is None or not getattr(predicted_labels, "skeletons", None):
        return _empty_result("inference returned no predictions; skipping")

    # Node-name match guard: the predicted skeleton must line up with the user
    # skeleton or the per-node confidence lookup would be meaningless.
    pred_node_names = _skeleton_node_names(predicted_labels.skeletons[0])
    if pred_node_names != user_node_names:
        return _empty_result(
            "model skeleton node-names do not match labels.skeletons[0] "
            f"(model={pred_node_names!r} vs labels={user_node_names!r}); skipping"
        )

    _report("In-sample prediction", 0.5, "Matching predictions")

    # Index predicted frames by (video_idx, frame_idx) so we can align them to
    # the user frames regardless of frame ordering in the returned Labels.
    # Map predicted videos back to user-video indices by identifier so a
    # video_idx is stable across the two Labels objects.
    user_video_key_to_idx = {_video_key(v, i): i for i, v in enumerate(labels.videos)}

    def _resolve_video_idx(video, fallback_idx: int) -> int:
        return user_video_key_to_idx.get(_video_key(video, fallback_idx), fallback_idx)

    pred_by_frame: dict[tuple[int, int], list] = {}
    for pv_idx, pvideo in enumerate(predicted_labels.videos):
        v_idx = _resolve_video_idx(pvideo, pv_idx)
        for lf in predicted_labels:
            if lf.video is not pvideo:
                continue
            pred_by_frame.setdefault((v_idx, lf.frame_idx), []).extend(
                list(lf.instances)
            )

    instance_scores: dict[tuple[int, int, int], float] = {}
    records: list[dict] = []

    for video_idx, video in enumerate(labels.videos):
        labeled_frames = [lf for lf in labels if lf.video == video]
        for lf in labeled_frames:
            frame_idx = lf.frame_idx
            user_instances = list(lf.user_instances)
            if not user_instances:
                continue

            user_points = [inst.numpy(invisible_as_nan=True) for inst in user_instances]
            pred_instances = pred_by_frame.get((video_idx, frame_idx), [])
            pred_points = [
                _predicted_scores_array(pi, n_nodes) for pi in pred_instances
            ]

            matches = match_predictions_to_users(
                user_points, [pp["xy"] for pp in pred_points]
            )

            for inst_idx, upts in enumerate(user_points):
                match_idx = matches[inst_idx]
                pred_scores = (
                    pred_points[match_idx]["scores"] if match_idx is not None else None
                )
                result = score_instance_disagreement(
                    upts, pred_scores, min_confidence=min_confidence
                )

                score = result["prediction_disagreement_score"]
                if score > 0.0:
                    instance_scores[(video_idx, frame_idx, inst_idx)] = score

                confs = result["unlabeled_confidences"]
                for node_idx in result["disagreement_nodes"]:
                    records.append(
                        {
                            "video_idx": video_idx,
                            "frame_idx": frame_idx,
                            "instance_idx": inst_idx,
                            "node_idx": node_idx,
                            "node_name": user_node_names[node_idx]
                            if node_idx < len(user_node_names)
                            else str(node_idx),
                            "predicted_confidence": confs.get(node_idx, float("nan")),
                        }
                    )

    _report(
        "In-sample prediction",
        1.0,
        f"{len(instance_scores)} instances with disagreements",
    )

    return {
        "instance_scores": instance_scores,
        "records": records,
        "ran": True,
        "reason": "",
    }


def _predicted_scores_array(pred_instance, n_nodes: int) -> dict:
    """Extract ``{"xy": (n,2), "scores": (n,)}`` from a predicted instance.

    Uses ``PredictedInstance.numpy(scores=True)`` -> ``(n_nodes, 3)`` of
    ``(x, y, score)``. Splits out the coordinate columns (for centroid matching)
    and the score column (for the disagreement lookup). Defensive against odd
    shapes / instances that don't support ``scores=True``.
    """
    try:
        arr = np.asarray(pred_instance.numpy(scores=True), dtype=float)
    except TypeError:
        # Older/odd instance without a scores kwarg: fall back to coords only,
        # treating confidence as "present but unknown" -> NaN (never flags).
        arr = np.asarray(pred_instance.numpy(), dtype=float)
        xy = np.full((n_nodes, 2), np.nan)
        k = min(n_nodes, arr.shape[0])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            xy[:k] = arr[:k, :2]
        return {"xy": xy, "scores": np.full(n_nodes, np.nan)}

    xy = np.full((n_nodes, 2), np.nan)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        k = min(n_nodes, arr.shape[0])
        xy[:k] = arr[:k, :2]
    scores = _predicted_scores(arr, n_nodes)
    return {"xy": xy, "scores": scores}
