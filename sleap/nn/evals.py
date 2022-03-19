"""
Evaluation utilities for measuring pose estimation accuracy.

To generate metrics, you'll need two `Labels` datasets, one with ground truth
data and one with predicted data. The video paths in the datasets must match.
Load both datasets and call `evaluate`, like so: ::

   > labels_gt = Labels.load_file("path/to/ground/truth.slp")
   > labels_pr = Labels.load_file("path/to/predictions.slp")
   > metrics = evaluate(labels_gt, labels_pr)

`evaluate` returns a dictionary, keys are strings which name the metric,
values are either floats or numpy arrays.

A good place to start if you want to understand how well your models are
performing is to look at:

    * oks_voc.mAP
    * vis.precision
    * vis.recall
    * dist.p95
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Text, Tuple, Union
import logging
import sleap
from sleap import Labels, LabeledFrame, Instance, PredictedInstance
from sleap.nn.config import (
    TrainingJobConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfig,
    MultiClassBottomUpConfig,
    MultiClassTopDownConfig,
    SingleInstanceConfmapsHeadConfig,
)
from sleap.nn.model import Model
from sleap.nn.data.pipelines import LabelsReader
from sleap.nn.inference import (
    TopDownPredictor,
    BottomUpPredictor,
    BottomUpMultiClassPredictor,
    TopDownMultiClassPredictor,
    SingleInstancePredictor,
)

logger = logging.getLogger(__name__)


def replace_path(video_list: List[dict], new_paths: List[Text]):
    """Replace video paths in unstructured video objects."""
    if isinstance(new_paths, str):
        new_paths = [new_paths] * len(video_list)

    for video, new_path in zip(video_list, new_paths):
        video["backend"]["filename"] = new_path


def find_frame_pairs(
    labels_gt: Labels, labels_pr: Labels, user_labels_only: bool = True
) -> List[Tuple[LabeledFrame, LabeledFrame]]:
    """Find corresponding frames across two sets of labels.

    Args:
        labels_gt: A `sleap.Labels` instance with ground truth instances.
        labels_pr: A `sleap.Labels` instance with predicted instances.
        user_labels_only: If False, frames with predicted instances in `labels_gt` will
            also be considered for matching.

    Returns:
        A list of pairs of `sleap.LabeledFrame`s in the form `(frame_gt, frame_pr)`.
    """
    frame_pairs = []
    for video_gt in labels_gt.videos:

        # Find matching video instance in predictions.
        video_pr = None
        for video in labels_pr.videos:
            if isinstance(video.backend, type(video_gt.backend)) and video.matches(
                video_gt
            ):
                video_pr = video
                break

        if video_pr is None:
            continue

        # Find labeled frames in this video.
        labeled_frames_gt = labels_gt.find(video_gt)
        if user_labels_only:
            labeled_frames_gt = [
                lf for lf in labeled_frames_gt if lf.has_user_instances
            ]

        # Attempt to match each labeled frame in the ground truth.
        for labeled_frame_gt in labeled_frames_gt:
            labeled_frames_pr = labels_pr.find(
                video_pr, frame_idx=labeled_frame_gt.frame_idx
            )

            if not labeled_frames_pr:
                # No match
                continue
            elif len(labeled_frames_pr) == 1:
                # Match!
                frame_pairs.append((labeled_frame_gt, labeled_frames_pr[0]))
            else:
                # Too many matches.
                raise ValueError("More than one labeled frame found in predictions.")

    return frame_pairs


def compute_instance_area(points: np.ndarray) -> np.ndarray:
    """Compute the area of the bounding box of a set of keypoints.

    Args:
        points: A numpy array of coordinates.

    Returns:
        The area of the bounding box of the points.
    """
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)

    min_pt = np.nanmin(points, axis=-2)
    max_pt = np.nanmax(points, axis=-2)

    return np.prod(max_pt - min_pt, axis=-1)


def compute_oks(
    points_gt: np.ndarray,
    points_pr: np.ndarray,
    scale: Optional[float] = None,
    stddev: float = 0.025,
) -> np.ndarray:
    """Compute the object keypoints similarity between sets of points.

    Args:
        points_gt: Ground truth instances of shape (n_gt, n_nodes, n_ed),
            where n_nodes is the number of body parts/keypoint types, and n_ed
            is the number of Euclidean dimensions (typically 2 or 3). Keypoints
            that are missing/not visible should be represented as NaNs.
        points_pr: Predicted instance of shape (n_pr, n_nodes, n_ed).
        scale: Size scaling factor to use when weighing the scores, typically
            the area of the bounding box of the instance (in pixels). This
            should be of the length n_gt. If a scalar is provided, the same
            number is used for all ground truth instances. If set to None, the
            bounding box area of the ground truth instances will be calculated.
        stddev: The standard deviation associated with the spread in the
            localization accuracy of each node/keypoint type. This should be of
            the length n_nodes. "Easier" keypoint types will have lower values
            to reflect the smaller spread expected in localizing it.

    Returns:
        The object keypoints similarity between every pair of ground truth and
        predicted instance, a numpy array of of shape (n_gt, n_pr) in the range
        of [0, 1.0], with 1.0 denoting a perfect match.

    Notes:
        It's important to set the stddev appropriately when accounting for the
        difficulty of each keypoint type. For reference, the median value for
        all keypoint types in COCO is 0.072. The "easiest" keypoint is the left
        eye, with stddev of 0.025, since it is easy to precisely locate the
        eyes when labeling. The "hardest" keypoint is the left hip, with stddev
        of 0.107, since it's hard to locate the left hip bone without external
        anatomical features and since it is often occluded by clothing.

        The implementation here is based off of the descriptions in:
        Ronch & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose
        Estimation." ICCV (2017).
    """
    if points_gt.ndim != 3 or points_pr.ndim != 3:
        raise ValueError(
            "Points must be rank-3 with shape (n_instances, n_nodes, n_ed)."
        )

    if scale is None:
        scale = compute_instance_area(points_gt)

    n_gt, n_nodes, n_ed = points_gt.shape  # n_ed = 2 or 3 (euclidean dimensions)
    n_pr = points_pr.shape[0]

    # If scalar scale was provided, use the same for each ground truth instance.
    if np.isscalar(scale):
        scale = np.full(n_gt, scale)

    # If scalar standard deviation was provided, use the same for each node.
    if np.isscalar(stddev):
        stddev = np.full(n_nodes, stddev)

    # Compute displacement between each pair.
    displacement = np.reshape(points_gt, (n_gt, 1, n_nodes, n_ed)) - np.reshape(
        points_pr, (1, n_pr, n_nodes, n_ed)
    )
    assert displacement.shape == (n_gt, n_pr, n_nodes, n_ed)

    # Convert to pairwise Euclidean distances.
    distance = (displacement ** 2).sum(axis=-1)  # (n_gt, n_pr, n_nodes)
    assert distance.shape == (n_gt, n_pr, n_nodes)

    # Compute the normalization factor per keypoint.
    spread_factor = (2 * stddev) ** 2
    scale_factor = 2 * (scale + np.spacing(1))
    normalization_factor = np.reshape(spread_factor, (1, 1, n_nodes)) * np.reshape(
        scale_factor, (n_gt, 1, 1)
    )
    assert normalization_factor.shape == (n_gt, 1, n_nodes)

    # Since a "miss" is considered as KS < 0.5, we'll set the
    # distances for predicted points that are missing to inf.
    missing_pr = np.any(np.isnan(points_pr), axis=-1)  # (n_pr, n_nodes)
    assert missing_pr.shape == (n_pr, n_nodes)
    distance[:, missing_pr] = np.inf

    # Compute the keypoint similarity as per the top of Eq. 1.
    ks = np.exp(-(distance / normalization_factor))  # (n_gt, n_pr, n_nodes)
    assert ks.shape == (n_gt, n_pr, n_nodes)

    # Set the KS for missing ground truth points to 0.
    # This is equivalent to the visibility delta function of the bottom
    # of Eq. 1.
    missing_gt = np.any(np.isnan(points_gt), axis=-1)  # (n_gt, n_nodes)
    assert missing_gt.shape == (n_gt, n_nodes)
    ks[np.expand_dims(missing_gt, axis=1)] = 0

    # Compute the OKS.
    n_visible_gt = np.sum(
        (~missing_gt).astype("float64"), axis=-1, keepdims=True
    )  # (n_gt, 1)
    oks = np.sum(ks, axis=-1) / n_visible_gt
    assert oks.shape == (n_gt, n_pr)

    return oks


def match_instances(
    frame_gt: LabeledFrame,
    frame_pr: LabeledFrame,
    stddev: float = 0.025,
    scale: Optional[float] = None,
    threshold: float = 0,
    user_labels_only: bool = True,
) -> Tuple[List[Tuple[Instance, PredictedInstance, float]], List[Instance]]:
    """Match pairs of instances between ground truth and predictions in a frame.

    Args:
        frame_gt: A `sleap.LabeledFrame` with ground truth instances.
        frame_pr: A `sleap.LabeledFrame` with predicted instances.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.
        user_labels_only: If False, predicted instances in the ground truth frame may be
            considered for matching.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 3-tuples of the form
        `(instance_gt, instance_pr, oks)` containing the matched pair of instances and
        their OKS.

        `false_negatives` is a list of ground truth `sleap.Instance`s that could not be
        matched.

    Notes:
        This function uses the approach from the PASCAL VOC scoring procedure. Briefly,
        predictions are sorted descending by their instance-level prediction scores and
        greedily matched to ground truth instances which are then removed from the pool
        of available instances.

        Ground truth instances that remain unmatched are considered false negatives.
    """
    # Sort predicted instances by score.
    scores_pr = np.array(
        [
            instance.score
            for instance in frame_pr.instances
            if hasattr(instance, "score")
        ]
    )
    idxs_pr = np.argsort(-scores_pr, kind="mergesort")  # descending
    scores_pr = scores_pr[idxs_pr]

    if user_labels_only:
        available_instances_gt = frame_gt.user_instances
    else:
        available_instances_gt = frame_gt.instances
    available_instances_gt_idxs = list(range(len(available_instances_gt)))

    positive_pairs = []
    for idx_pr in idxs_pr:
        # Pull out predicted instance.
        instance_pr = frame_pr.instances[idx_pr]

        # Convert instances to point arrays.
        points_pr = np.expand_dims(instance_pr.numpy(), axis=0)
        points_gt = np.stack(
            [
                available_instances_gt[idx].numpy()
                for idx in available_instances_gt_idxs
            ],
            axis=0,
        )

        # Find the best match by computing OKS.
        oks = compute_oks(points_gt, points_pr, stddev=stddev, scale=scale)
        oks = np.squeeze(oks, axis=1)
        assert oks.shape == (len(points_gt),)

        oks[oks <= threshold] = np.nan
        best_match_gt_idx = np.argsort(-oks, kind="mergesort")[0]
        best_match_oks = oks[best_match_gt_idx]
        if np.isnan(best_match_oks):
            continue

        # Remove matched ground truth instance and add as a positive pair.
        instance_gt_idx = available_instances_gt_idxs.pop(best_match_gt_idx)
        instance_gt = available_instances_gt[instance_gt_idx]
        positive_pairs.append((instance_gt, instance_pr, best_match_oks))

        # Stop matching lower scoring instances if we run out of candidates in the
        # ground truth.
        if not available_instances_gt_idxs:
            break

    # Any remaining ground truth instances are considered false negatives.
    false_negatives = [
        available_instances_gt[idx] for idx in available_instances_gt_idxs
    ]

    return positive_pairs, false_negatives


def match_frame_pairs(
    frame_pairs: List[Tuple[LabeledFrame, LabeledFrame]],
    stddev: float = 0.025,
    scale: Optional[float] = None,
    threshold: float = 0,
    user_labels_only: bool = True,
) -> Tuple[List[Tuple[Instance, PredictedInstance, float]], List[Instance]]:
    """Match all ground truth and predicted instances within each pair of frames.

    This is a wrapper for `match_instances()` but operates on lists of frames.

    Args:
        frame_pairs: A list of pairs of `sleap.LabeledFrame`s in the form
            `(frame_gt, frame_pr)`. These can be obtained with `find_frame_pairs()`.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.
        user_labels_only: If False, predicted instances in the ground truth frame may be
            considered for matching.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 3-tuples of the form
        `(instance_gt, instance_pr, oks)` containing the matched pair of instances and
        their OKS.

        `false_negatives` is a list of ground truth `sleap.Instance`s that could not be
        matched.
    """
    positive_pairs = []
    false_negatives = []
    for frame_gt, frame_pr in frame_pairs:
        positive_pairs_frame, false_negatives_frame = match_instances(
            frame_gt,
            frame_pr,
            stddev=stddev,
            scale=scale,
            threshold=threshold,
            user_labels_only=user_labels_only,
        )
        positive_pairs.extend(positive_pairs_frame)
        false_negatives.extend(false_negatives_frame)

    return positive_pairs, false_negatives


def compute_generalized_voc_metrics(
    positive_pairs: List[Tuple[Instance, PredictedInstance, Any]],
    false_negatives: List[Instance],
    match_scores: List[float],
    match_score_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),  # 0.5:0.05:0.95
    recall_thresholds: np.ndarray = np.linspace(0, 1, 101),  # 0.0:0.01:1.00
    name: Text = "gvoc",
) -> Dict[Text, Any]:
    """Compute VOC metrics given matched pairs of instances.

    Args:
        positive_pairs: A list of tuples of the form `(instance_gt, instance_pr, _)`
            containing the matched pair of instances.
        false_negatives: A list of unmatched instances.
        match_scores: The score obtained in the matching procedure for each matched pair
            (e.g., OKS).
        match_score_thresholds: Score thresholds at which to consider matches as a true
            positive match.
        recall_thresholds: Recall thresholds at which to evaluate Average Precision.
        name: Name to use to prefix returned metric keys.

    Returns:
        A dictionary of VOC metrics.
    """
    detection_scores = np.array([pp[1].score for pp in positive_pairs])

    inds = np.argsort(-detection_scores, kind="mergesort")
    detection_scores = detection_scores[inds]
    match_scores = match_scores[inds]

    precisions = []
    recalls = []

    npig = len(positive_pairs) + len(false_negatives)  # total number of GT instances

    for match_score_threshold in match_score_thresholds:

        tp = np.cumsum(match_scores >= match_score_threshold)
        fp = np.cumsum(match_scores < match_score_threshold)

        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))

        recall = rc[-1]  # best recall at this OKS threshold

        # Ensure strictly decreasing precisions.
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        # Find best precision at each recall threshold.
        rc_inds = np.searchsorted(rc, recall_thresholds, side="left")
        precision = np.zeros(rc_inds.shape)
        is_valid_rc_ind = rc_inds < len(pr)
        precision[is_valid_rc_ind] = pr[rc_inds[is_valid_rc_ind]]

        precisions.append(precision)
        recalls.append(recall)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    AP = precisions.mean(
        axis=1
    )  # AP = average precision over fixed set of recall thresholds
    AR = recalls  # AR = max recall given a fixed number of detections per image

    mAP = precisions.mean()  # mAP = mean over all OKS thresholds
    mAR = recalls.mean()  # mAR = mean over all OKS thresholds

    return {
        name + ".match_score_thresholds": match_score_thresholds,
        name + ".recall_thresholds": recall_thresholds,
        name + ".match_scores": match_scores,
        name + ".precisions": precisions,
        name + ".recalls": recalls,
        name + ".AP": AP,
        name + ".AR": AR,
        name + ".mAP": mAP,
        name + ".mAR": mAR,
    }


def compute_dists(
    positive_pairs: List[Tuple[Instance, PredictedInstance, Any]]
) -> np.ndarray:
    """Compute Euclidean distances between matched pairs of instances.

    Args:
        positive_pairs: A list of tuples of the form `(instance_gt, instance_pr, _)`
            containing the matched pair of instances.

    Returns:
        An array of pairwise distances of shape `(n_positive_pairs, n_nodes)`.
    """
    dists = []
    for instance_gt, instance_pr, _ in positive_pairs:
        points_gt = instance_gt.points_array
        points_pr = instance_pr.points_array

        dists.append(np.linalg.norm(points_pr - points_gt, axis=-1))
    dists = np.array(dists)

    return dists


def compute_dist_metrics(dists: np.ndarray) -> Dict[Text, np.ndarray]:
    """Compute the Euclidean distance error at different percentiles.

    Args:
        dists: An array of pairwise distances of shape `(n_positive_pairs, n_nodes)`.

    Returns:
        A dictionary of distance metrics.
    """
    results = {
        "dist.dists": dists,
        "dist.avg": np.nanmean(dists),
        "dist.p50": np.nan,
        "dist.p75": np.nan,
        "dist.p90": np.nan,
        "dist.p95": np.nan,
        "dist.p99": np.nan,
    }

    is_non_nan = ~np.isnan(dists)
    if np.any(is_non_nan):
        non_nans = dists[is_non_nan]
        for ptile in (50, 75, 90, 95, 99):
            results[f"dist.p{ptile}"] = np.percentile(non_nans, ptile)

    return results


def compute_pck_metrics(
    dists: np.ndarray, thresholds: np.ndarray = np.linspace(1, 10, 10)
) -> Dict[Text, np.ndarray]:
    """Compute PCK across a range of thresholds.

    Args:
        dists: An array of pairwise distances of shape `(n_positive_pairs, n_nodes)`.
        thresholds: A list of distance thresholds in pixels.

    Returns:
        A dictionary of PCK metrics evaluated at each threshold.
    """
    dists = np.copy(dists)
    dists[np.isnan(dists)] = np.inf
    pcks = np.expand_dims(dists, -1) < np.reshape(thresholds, (1, 1, -1))
    mPCK_parts = pcks.mean(axis=0).mean(axis=-1)
    mPCK = mPCK_parts.mean()

    return {
        "pck.thresholds": thresholds,
        "pck.pcks": pcks,
        "pck.mPCK_parts": mPCK_parts,
        "pck.mPCK": mPCK,
    }


def compute_visibility_conf(
    positive_pairs: List[Tuple[Instance, Instance, Any]]
) -> Dict[Text, float]:
    """Compute node visibility metrics.

    Args:
        positive_pairs: A list of tuples of the form `(instance_gt, instance_pr, _)`
            containing the matched pair of instances.

    Returns:
        A dictionary of visibility metrics, including the confusion matrix.
    """
    vis_tp = 0
    vis_fn = 0
    vis_fp = 0
    vis_tn = 0

    for instance_gt, instance_pr, _ in positive_pairs:
        missing_nodes_gt = np.isnan(instance_gt.points_array).any(axis=-1)
        missing_nodes_pr = np.isnan(instance_pr.points_array).any(axis=-1)

        vis_tn += ((missing_nodes_gt) & (missing_nodes_pr)).sum()
        vis_fn += ((~missing_nodes_gt) & (missing_nodes_pr)).sum()
        vis_fp += ((missing_nodes_gt) & (~missing_nodes_pr)).sum()
        vis_tp += ((~missing_nodes_gt) & (~missing_nodes_pr)).sum()

    return {
        "vis.tp": vis_tp,
        "vis.fp": vis_fp,
        "vis.tn": vis_tn,
        "vis.fn": vis_fn,
        "vis.precision": vis_tp / (vis_tp + vis_fp) if (vis_tp + vis_fp) else np.nan,
        "vis.recall": vis_tp / (vis_tp + vis_fn) if (vis_tp + vis_fn) else np.nan,
    }


def evaluate(
    labels_gt: Labels,
    labels_pr: Labels,
    oks_stddev: float = 0.025,
    oks_scale: Optional[float] = None,
    match_threshold: float = 0,
    user_labels_only: bool = True,
) -> Dict[Text, Union[float, np.ndarray]]:
    """Calculate all metrics from ground truth and predicted labels.

    Args:
        labels_gt: The `Labels` dataset object with ground truth labels.
        labels_pr: The `Labels` dataset object with predicted labels.
        oks_stddev: The standard deviation to use for calculating object
            keypoint similarity; see `compute_oks` function for details.
        oks_scale: The scale to use for calculating object
            keypoint similarity; see `compute_oks` function for details.
        match_threshold: The threshold to use on oks scores when determining
            which instances match between ground truth and predicted frames.
        user_labels_only: If False, predicted instances in the ground truth frame may be
            considered for matching.

    Returns:
        Dict, keys are strings, values are metrics (floats or ndarrays).
    """
    metrics = dict()

    frame_pairs = find_frame_pairs(
        labels_gt, labels_pr, user_labels_only=user_labels_only
    )

    if not frame_pairs:
        return metrics

    positive_pairs, false_negatives = match_frame_pairs(
        frame_pairs,
        stddev=oks_stddev,
        scale=oks_scale,
        threshold=match_threshold,
        user_labels_only=user_labels_only,
    )
    dists = compute_dists(positive_pairs)

    metrics.update(compute_visibility_conf(positive_pairs))
    metrics.update(compute_dist_metrics(dists))
    metrics.update(compute_pck_metrics(dists))

    pair_oks = np.array([oks for _, _, oks in positive_pairs])
    pair_pck = metrics["pck.pcks"].mean(axis=-1).mean(axis=-1)

    metrics["oks.mOKS"] = pair_oks.mean()
    metrics.update(
        compute_generalized_voc_metrics(
            positive_pairs, false_negatives, match_scores=pair_oks, name="oks_voc"
        )
    )
    metrics.update(
        compute_generalized_voc_metrics(
            positive_pairs, false_negatives, match_scores=pair_pck, name="pck_voc"
        )
    )

    return metrics


def evaluate_model(
    cfg: TrainingJobConfig,
    labels_reader: LabelsReader,
    model: Model,
    save: bool = True,
    split_name: Text = "test",
) -> Tuple[Labels, Dict[Text, Any]]:
    """Evaluate a trained model and save metrics and predictions.

    Args:
        cfg: The `TrainingJobConfig` associated with the model.
        labels_reader: A `LabelsReader` pipeline generator that reads the ground truth
            data to evaluate.
        model: The `sleap.nn.model.Model` instance to evaluate.
        save: If True, save the predictions and metrics to the model folder.
        split_name: String name to append to the saved filenames.

    Returns:
        A tuple of `(labels_pr, metrics)`.

        `labels_pr` will contain the predicted labels.

        `metrics` will contain the evaluated metrics given the predictions, or None if
        the metrics failed to be computed.
    """
    # Setup predictor for evaluation.
    head_config = cfg.model.heads.which_oneof()
    if isinstance(head_config, CentroidsHeadConfig):
        predictor = TopDownPredictor(
            centroid_config=cfg,
            centroid_model=model,
            confmap_config=None,
            confmap_model=None,
        )
    elif isinstance(head_config, CenteredInstanceConfmapsHeadConfig):
        predictor = TopDownPredictor(
            centroid_config=None,
            centroid_model=None,
            confmap_config=cfg,
            confmap_model=model,
        )
    elif isinstance(head_config, MultiInstanceConfig):
        predictor = BottomUpPredictor(bottomup_config=cfg, bottomup_model=model)
    elif isinstance(head_config, SingleInstanceConfmapsHeadConfig):
        predictor = SingleInstancePredictor(confmap_config=cfg, confmap_model=model)
    elif isinstance(head_config, MultiClassBottomUpConfig):
        predictor = BottomUpMultiClassPredictor(
            config=cfg,
            model=model,
        )
    elif isinstance(head_config, MultiClassTopDownConfig):
        predictor = TopDownMultiClassPredictor(
            centroid_config=None,
            centroid_model=None,
            confmap_config=cfg,
            confmap_model=model,
        )
    else:
        raise ValueError("Unrecognized model type:", head_config)

    # Predict.
    labels_pr = predictor.predict(labels_reader, make_labels=True)

    # Compute metrics.
    try:
        metrics = evaluate(labels_reader.labels, labels_pr)
    except:
        logger.warning("Failed to compute metrics.")
        metrics = None

    # Save.
    if save:
        labels_pr_path = os.path.join(
            cfg.outputs.run_path, f"labels_pr.{split_name}.slp"
        )
        Labels.save_file(labels_pr, labels_pr_path)
        logger.info("Saved predictions: %s", labels_pr_path)

        if metrics is not None:
            metrics_path = os.path.join(
                cfg.outputs.run_path, f"metrics.{split_name}.npz"
            )
            np.savez_compressed(metrics_path, **{"metrics": metrics})
            logger.info("Saved metrics: %s", metrics_path)

    if metrics is not None:
        logger.info("OKS mAP: %f", metrics["oks_voc.mAP"])

    return labels_pr, metrics


def load_metrics(model_path: str, split: str = "val") -> Dict[str, Any]:
    """Load metrics for a model.

    Args:
        model_path: Path to a model folder or metrics file (.npz).
        split: Name of the split to load the metrics for. Must be `"train"`, `"val"` or
            `"test"` (default: `"val"`). Ignored if a path to a metrics NPZ file is
            provided.

    Returns:
        The loaded metrics as a dictionary with keys:

        - `"vis.tp"`: Visibility - True Positives
        - `"vis.fp"`: Visibility - False Positives
        - `"vis.tn"`: Visibility - True Negatives
        - `"vis.fn"`: Visibility - False Negatives
        - `"vis.precision"`: Visibility - Precision
        - `"vis.recall"`: Visibility - Recall
        - `"dist.avg"`: Average Distance (ground truth vs prediction)
        - `"dist.p50"`: Distance for 50th percentile
        - `"dist.p75"`: Distance for 75th percentile
        - `"dist.p90"`: Distance for 90th percentile
        - `"dist.p95"`: Distance for 95th percentile
        - `"dist.p99"`: Distance for 99th percentile
        - `"dist.dists"`: All distances
        - `"pck.mPCK"`: Mean Percentage of Correct Keypoints (PCK)
        - `"oks.mOKS"`: Mean Object Keypoint Similarity (OKS)
        - `"oks_voc.mAP"`: VOC with OKS scores - mean Average Precision (mAP)
        - `"oks_voc.mAR"`: VOC with OKS scores - mean Average Recall (mAR)
        - `"pck_voc.mAP"`: VOC with PCK scores - mean Average Precision (mAP)
        - `"pck_voc.mAR"`: VOC with PCK scores - mean Average Recall (mAR)
    """
    if os.path.isdir(model_path):
        metrics_path = os.path.join(model_path, f"metrics.{split}.npz")
    else:
        metrics_path = model_path
    with np.load(metrics_path, allow_pickle=True) as data:
        return data["metrics"].item()
