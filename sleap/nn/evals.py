"""This module contains evaluation utilities for measuring pose estimation accuracy."""

import os
import numpy as np


def replace_path(video_list, new_paths):
    if isinstance(new_paths, str):
        new_paths = [new_paths] * len(video_list)

    for video, new_path in zip(video_list, new_paths):
        video["backend"]["filename"] = new_path


def find_frame_pairs(labels_gt, labels_pr):
    frame_pairs = []
    for video_gt in labels_gt.videos:

        # Find matching video instance in predictions.
        video_pr = None
        for video in labels_pr.videos:
            if isinstance(video.backend, type(video_gt.backend)) and video.matches(video_gt):
                video_pr = video
                break

        if video_pr is None:
            continue

        # Find user labeled frames in this video.
        labeled_frames_gt = [
            lf for lf in labels_gt.find(video_gt) if lf.has_user_instances
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


def compute_instance_area(points):
    """Computes the area of the bounding box of a set of keypoints."""

    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)

    min_pt = np.nanmin(points, axis=-2)
    max_pt = np.nanmax(points, axis=-2)

    return np.prod(max_pt - min_pt, axis=-1)


def compute_oks(points_gt, points_pr, scale=None, stddev=0.025):
    """Computes the object keypoints similarity between sets of points.

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


def match_instances(frame_gt, frame_pr, stddev=0.025, scale=None, threshold=0):
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

    available_instances_gt = frame_gt.user_instances
    positive_pairs = []
    for idx_pr in idxs_pr:
        # Pull out predicted instance.
        instance_pr = frame_pr.instances[idx_pr]

        # Convert instances to point arrays.
        points_pr = np.expand_dims(instance_pr.points_array, axis=0)
        points_gt = np.stack(
            [instance.points_array for instance in available_instances_gt], axis=0
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
        instance_gt = available_instances_gt.pop(best_match_gt_idx)
        positive_pairs.append((instance_gt, instance_pr, best_match_oks))

        # Stop matching lower scoring instances if we run out of
        # candidates in the ground truth.
        if not available_instances_gt:
            break

    # Any remaining ground truth instances are considered false negatives.
    false_negatives = available_instances_gt

    return positive_pairs, false_negatives


def match_frame_pairs(frame_pairs, stddev=0.025, scale=None, threshold=0):
    # Match instances within each frame pair.
    positive_pairs = []
    false_negatives = []
    for frame_gt, frame_pr in frame_pairs:
        positive_pairs_frame, false_negatives_frame = match_instances(
            frame_gt, frame_pr, stddev=stddev, scale=scale, threshold=threshold
        )
        positive_pairs.extend(positive_pairs_frame)
        false_negatives.extend(false_negatives_frame)

    return positive_pairs, false_negatives


def compute_generalized_voc_metrics(
    positive_pairs,
    false_negatives,
    match_scores,
    match_score_thresholds=np.linspace(0.5, 0.95, 10),  # 0.5:0.05:0.95
    recall_thresholds=np.linspace(0, 1, 101),  # 0.0:0.01:1.00
    name="gvoc",
):

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


def compute_dists(positive_pairs):
    dists = []
    for instance_gt, instance_pr, _ in positive_pairs:
        points_gt = instance_gt.points_array
        points_pr = instance_pr.points_array

        dists.append(np.linalg.norm(points_pr - points_gt, axis=-1))
    dists = np.array(dists)

    return dists


def compute_dist_metrics(dists):
    return {
        "dist.dists": dists,
        "dist.avg": np.nanmean(dists),
        "dist.p50": np.percentile(dists[~np.isnan(dists)], 50),
        "dist.p75": np.percentile(dists[~np.isnan(dists)], 75),
        "dist.p90": np.percentile(dists[~np.isnan(dists)], 90),
        "dist.p95": np.percentile(dists[~np.isnan(dists)], 95),
        "dist.p99": np.percentile(dists[~np.isnan(dists)], 99),
    }


def compute_pck_metrics(dists, thresholds=np.linspace(1, 10, 10)):

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


def compute_visibility_conf(positive_pairs):

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
        "vis.precision": vis_tp / (vis_tp + vis_fp),
        "vis.recall": vis_tp / (vis_tp + vis_fn),
    }


def evaluate(labels_gt, labels_pr, oks_stddev=0.025, oks_scale=None, match_threshold=0):

    frame_pairs = find_frame_pairs(labels_gt, labels_pr)
    positive_pairs, false_negatives = match_frame_pairs(
        frame_pairs, stddev=oks_stddev, scale=oks_scale, threshold=match_threshold
    )
    dists = compute_dists(positive_pairs)

    metrics = {}
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
