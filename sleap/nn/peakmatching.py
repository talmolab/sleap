import numpy as np
from typing import Dict, List, Union, Optional, Tuple

from sleap.instance import LabeledFrame, PredictedPoint, PredictedInstance
from sleap.info.metrics import calculate_pairwise_cost


def match_single_peaks_frame(points_array, skeleton, transform, img_idx):
    """
    Make instance from points array returned by single peak finding.
    This is for the pipeline that assumes there's exactly one instance
    per frame.

    Returns:
        PredictedInstance, or None if no points.
    """
    if points_array.shape[0] == 0:
        return None

    # apply inverse transform to points
    points_array[..., 0:2] = transform.invert(img_idx, points_array[..., 0:2])

    pts = dict()
    for i, node in enumerate(skeleton.nodes):
        if not any(np.isnan(points_array[i])):
            x, y, score = points_array[i]
            # FIXME: is score just peak value or something else?
            pt = PredictedPoint(x=x, y=y, score=score)
            pts[node] = pt

    matched_instance = None
    if len(pts) > 0:
        # FIXME: how should we calculate score for instance?
        inst_score = np.sum(points_array[..., 2]) / len(pts)
        matched_instance = PredictedInstance(
            skeleton=skeleton, points=pts, score=inst_score
        )

    return matched_instance


def match_single_peaks_all(points_arrays, skeleton, video, transform):
    """
    Make labeled frames for the results of single peak finding.
    This is for the pipeline that assumes there's exactly one instance
    per frame.

    Returns:
        list of LabeledFrames
    """
    predicted_frames = []
    for img_idx, points_array in enumerate(points_arrays):
        inst = match_single_peaks_frame(points_array, skeleton, transform, img_idx)
        if inst is not None:
            frame_idx = transform.get_frame_idxs(img_idx)
            new_lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst])
            predicted_frames.append(new_lf)
    return predicted_frames


def improfile(I, p0, p1, max_points=None):
    """
    Returns values of the image I evaluated along the line formed
    by points p0 and p1.

    Parameters
    ----------
    I : 2d array
        Image to get values from
    p0, p1 : 1d array with 2 elements
        Start and end coordinates of the line

    Returns
    -------
    vals : 1d array
        Vector with the images values along the line formed by p0 and p1
    """
    # Make sure image is 2d
    I = np.squeeze(I)

    # Find number of points to extract
    n = np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
    n = max(n, 1)
    if max_points is not None:
        n = min(n, max_points)
    n = int(n)

    # Compute coordinates
    x = np.floor(np.linspace(p0[0], p1[0], n)).astype("int32")
    y = np.floor(np.linspace(p0[1], p1[1], n)).astype("int32")

    # Extract values and concatenate into vector
    vals = np.stack([I[yi, xi] for xi, yi in zip(x, y)])
    return vals


def match_peaks_frame(
    peaks_t,
    peak_vals_t,
    pafs_t,
    skeleton,
    transform,
    img_idx,
    min_score_to_node_ratio=0.4,
    min_score_midpts=0.05,
    min_score_integral=0.8,
    add_last_edge=False,
    single_per_crop=True,
):
    """
    Matches single frame
    """

    # Effectively the original implementation:
    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/demo_video.py#L107

    # Dumb
    peak_ids = []
    idx = 0
    for i in range(len(peaks_t)):
        idx_i = []
        for j in range(len(peaks_t[i])):
            idx_i.append(idx)
            idx += 1
        peak_ids.append(idx_i)

    # Score each edge
    special_k = []
    connection_all = []
    for k, edge in enumerate(skeleton.edge_names):
        src_node_idx = skeleton.node_to_index(edge[0])
        dst_node_idx = skeleton.node_to_index(edge[1])
        paf_x = pafs_t[..., 2 * k]
        paf_y = pafs_t[..., 2 * k + 1]

        # Make sure matrix has rows for these nodes
        if len(peaks_t) <= src_node_idx or len(peaks_t) <= dst_node_idx:
            special_k.append(k)
            connection_all.append([])
            continue

        peaks_src = peaks_t[src_node_idx]
        peaks_dst = peaks_t[dst_node_idx]
        peak_vals_src = peak_vals_t[src_node_idx]
        peak_vals_dst = peak_vals_t[dst_node_idx]

        if len(peaks_src) == 0 or len(peaks_dst) == 0:
            special_k.append(k)
            connection_all.append([])
        else:
            connection_candidates = []
            for i, peak_src in enumerate(peaks_src):
                for j, peak_dst in enumerate(peaks_dst):
                    # Vector between peaks
                    vec = peak_dst - peak_src

                    # Euclidean distance between points
                    norm = np.sqrt(np.sum(vec ** 2))

                    # Failure if points overlap
                    if norm == 0:
                        continue

                    # Convert to unit vector
                    vec = vec / norm

                    # Get PAF values along edge
                    vec_x = improfile(paf_x, peak_src, peak_dst)
                    vec_y = improfile(paf_y, peak_src, peak_dst)

                    # Compute score
                    score_midpts = vec_x * vec[0] + vec_y * vec[1]
                    score_with_dist_prior = np.mean(score_midpts) + min(
                        0.5 * paf_x.shape[0] / norm - 1, 0
                    )
                    score_integral = np.mean(score_midpts > min_score_midpts)
                    if (
                        score_with_dist_prior > 0
                        and score_integral > min_score_integral
                    ):
                        connection_candidates.append([i, j, score_with_dist_prior])

            # Sort candidates for current edge by descending score
            connection_candidates = sorted(
                connection_candidates, key=lambda x: x[2], reverse=True
            )

            # Add to list of candidates for next step
            connection = np.zeros((0, 5))  # src_id, dst_id, paf_score, i, j
            for candidate in connection_candidates:
                i, j, score = candidate
                # Add to connections if node is not already included
                if (i not in connection[:, 3]) and (j not in connection[:, 4]):
                    id_i = peak_ids[src_node_idx][i]
                    id_j = peak_ids[dst_node_idx][j]
                    connection = np.vstack([connection, [id_i, id_j, score, i, j]])

                    # Stop when reached the max number of matches possible
                    if len(connection) >= min(len(peaks_src), len(peaks_dst)):
                        break
            connection_all.append(connection)

    # Greedy matching of each edge candidate set
    subset = -1 * np.ones(
        (0, len(skeleton.nodes) + 2)
    )  # ids, overall score, number of parts
    candidate = np.array([y for x in peaks_t for y in x])  # flattened set of all points
    candidate_scores = np.array(
        [y for x in peak_vals_t for y in x]
    )  # flattened set of all peak scores
    for k, edge in enumerate(skeleton.edge_names):
        # No matches for this edge
        if k in special_k:
            continue

        # Get IDs for current connection
        partAs = connection_all[k][:, 0]
        partBs = connection_all[k][:, 1]

        # Get edge
        indexA, indexB = (
            skeleton.node_to_index(edge[0]),
            skeleton.node_to_index(edge[1]),
        )

        # Loop through all candidates for current edge
        for i in range(len(connection_all[k])):
            found = 0
            subset_idx = [-1, -1]

            # Search for current candidates in matched subset
            for j in range(len(subset)):
                if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                    subset_idx[found] = j
                    found += 1

            # One of the two candidate points found in matched subset
            if found == 1:
                j = subset_idx[0]
                if subset[j][indexB] != partBs[i]:  # did we already assign this part?
                    subset[j][indexB] = partBs[i]  # assign part
                    subset[j][-1] += 1  # increment instance part counter
                    subset[j][-2] += (
                        candidate_scores[int(partBs[i])] + connection_all[k][i][2]
                    )  # add peak + edge score

            # Both candidate points found in matched subset
            elif found == 2:
                j1, j2 = subset_idx  # get indices in matched subset
                membership = (
                    (subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int)
                )[
                    :-2
                ]  # count number of instances per body parts
                # All body parts are disjoint, merge them
                if np.all(membership < 2):
                    subset[j1][:-2] += subset[j2][:-2] + 1
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, axis=0)

                # Treat them separately
                else:
                    subset[j1][indexB] = partBs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += (
                        candidate_scores[partBs[i].astype(int)]
                        + connection_all[k][i][2]
                    )

            # Neither point found, create a new subset (if not the last edge)
            elif found == 0 and (add_last_edge or (k < (len(skeleton.edges) - 1))):
                row = -1 * np.ones(len(skeleton.nodes) + 2)
                row[indexA] = partAs[i]  # ID
                row[indexB] = partBs[i]  # ID
                row[-1] = 2  # initial count
                row[-2] = (
                    sum(candidate_scores[connection_all[k][i, :2].astype(int)])
                    + connection_all[k][i][2]
                )  # score
                subset = np.vstack([subset, row])  # add to matched subset

    # Filter small instances
    score_to_node_ratio = subset[:, -2] / subset[:, -1]
    subset = subset[score_to_node_ratio > min_score_to_node_ratio, :]

    # Apply inverse transform to points to return to full resolution, uncropped image coordinates
    if candidate.shape[0] > 0:
        candidate[..., 0:2] = transform.invert(img_idx, candidate[..., 0:2])

    # Done with all the matching! Gather the data
    matched_instances_t = []
    for match in subset:

        # Get the predicted points for this predicted instance
        pts = dict()
        for i, node_name in enumerate(skeleton.node_names):
            if match[i] >= 0:
                match_idx = int(match[i])
                pt = PredictedPoint(
                    x=candidate[match_idx, 0],
                    y=candidate[match_idx, 1],
                    score=candidate_scores[match_idx],
                )
                pts[node_name] = pt

        if len(pts):
            matched_instances_t.append(
                PredictedInstance(skeleton=skeleton, points=pts, score=match[-2])
            )

    # For centroid crop just return instance closest to centroid
    # if single_per_crop and len(matched_instances_t) > 1 and transform.is_cropped:

    # crop_centroid = np.array(((transform.crop_size//2, transform.crop_size//2),)) # center of crop box
    # crop_centroid = transform.invert(img_idx, crop_centroid) # relative to original image

    # # sort by distance from crop centroid
    # matched_instances_t.sort(key=lambda inst: np.linalg.norm(inst.centroid - crop_centroid))

    # # logger.debug(f"SINGLE_INSTANCE_PER_CROP: crop has {len(matched_instances_t)} instances, filter to 1.")

    # # just use closest
    # matched_instances_t = matched_instances_t[0:1]

    if single_per_crop and len(matched_instances_t) > 1 and transform.is_cropped:
        # Just keep highest scoring instance
        matched_instances_t = [matched_instances_t[0]]

    return matched_instances_t


def match_peaks_paf(
    peaks,
    peak_vals,
    pafs,
    skeleton,
    video,
    transform,
    min_score_to_node_ratio=0.4,
    min_score_midpts=0.05,
    min_score_integral=0.8,
    add_last_edge=False,
    single_per_crop=True,
    **kwargs
):
    """ Computes PAF-based peak matching via greedy assignment """

    # Process each frame
    predicted_frames = []
    for img_idx, (peaks_t, peak_vals_t, pafs_t) in enumerate(
        zip(peaks, peak_vals, pafs)
    ):
        instances = match_peaks_frame(
            peaks_t,
            peak_vals_t,
            pafs_t,
            skeleton,
            transform,
            img_idx,
            min_score_to_node_ratio=min_score_to_node_ratio,
            min_score_midpts=min_score_midpts,
            min_score_integral=min_score_integral,
            add_last_edge=add_last_edge,
            single_per_crop=single_per_crop,
        )
        frame_idx = transform.get_frame_idxs(img_idx)
        predicted_frames.append(
            LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )

    # Combine LabeledFrame objects for the same video frame
    predicted_frames = LabeledFrame.merge_frames(predicted_frames, video=video)

    return predicted_frames


def match_peaks_paf_par(
    peaks,
    peak_vals,
    pafs,
    skeleton,
    video,
    transform,
    min_score_to_node_ratio=0.4,
    min_score_midpts=0.05,
    min_score_integral=0.8,
    add_last_edge=False,
    single_per_crop=True,
    pool=None,
    **kwargs
):
    """ Parallel version of PAF peak matching """

    if pool is None:
        import multiprocessing

        pool = multiprocessing.Pool()

    futures = []
    for img_idx, (peaks_t, peak_vals_t, pafs_t) in enumerate(
        zip(peaks, peak_vals, pafs)
    ):
        future = pool.apply_async(
            match_peaks_frame,
            [peaks_t, peak_vals_t, pafs_t, skeleton],
            dict(
                transform=transform,
                img_idx=img_idx,
                min_score_to_node_ratio=min_score_to_node_ratio,
                min_score_midpts=min_score_midpts,
                min_score_integral=min_score_integral,
                add_last_edge=add_last_edge,
                single_per_crop=single_per_crop,
            ),
        )
        futures.append(future)

    predicted_frames = []
    for img_idx, future in enumerate(futures):
        instances = future.get()
        frame_idx = transform.get_frame_idxs(img_idx)
        # Ok, since we are doing this in parallel. Objects are getting serialized and sent
        # back and forth. This causes all instances to have a different skeleton object
        # This will cause problems later because checking if two skeletons are equal is
        # an expensive operation.
        for i in range(len(instances)):
            points = {node.name: point for node, point in instances[i].nodes_points}
            instances[i] = PredictedInstance(
                skeleton=skeleton, points=points, score=instances[i].score
            )

        predicted_frames.append(
            LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )

    # Combine LabeledFrame objects for the same video frame
    predicted_frames = LabeledFrame.merge_frames(predicted_frames, video=video)

    return predicted_frames


def instances_nms(
    instances: List[PredictedInstance], thresh: float = 4
) -> List[PredictedInstance]:
    """Remove overlapping instances from list."""
    if len(instances) <= 1:
        return

    # Look for overlapping instances
    overlap_matrix = calculate_pairwise_cost(
        instances,
        instances,
        cost_function=lambda x: np.nan if all(np.isnan(x)) else np.nanmean(x),
    )

    # Set diagonals over threshold since an instance doesn't overlap with itself
    np.fill_diagonal(overlap_matrix, thresh + 1)
    overlap_matrix[np.isnan(overlap_matrix)] = thresh + 1

    instances_to_remove = []

    def sort_funct(inst_idx):
        # sort by number of points in instance, then by prediction score (desc)
        return (
            len(instances[inst_idx].nodes),
            -getattr(instances[inst_idx], "score", 0),
        )

    while np.nanmin(overlap_matrix) < thresh:
        # Find the pair of instances with greatest overlap
        idx_a, idx_b = np.unravel_index(overlap_matrix.argmin(), overlap_matrix.shape)

        # Keep the instance with the most points (or the highest score if tied)
        idxs = sorted([idx_a, idx_b], key=sort_funct)
        pick_idx = idxs[0]
        keep_idx = idxs[-1]

        # Remove this instance from overlap matrix
        overlap_matrix[pick_idx, :] = thresh + 1
        overlap_matrix[:, pick_idx] = thresh + 1

        # Add to list of instances that we'll remove.
        # We'll remove these later so list index doesn't change now.
        instances_to_remove.append(instances[pick_idx])

    # Remove selected instances from list
    # Note that we're modifying the original list in place
    for inst in instances_to_remove:
        instances.remove(inst)
