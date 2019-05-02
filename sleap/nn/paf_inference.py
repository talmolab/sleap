import numpy as np
import multiprocessing

from scipy.optimize import linear_sum_assignment


def improfile(I, p0, p1, max_points=None):
    """ Returns values of the image I evaluated along the line formed by points p0 and p1.

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
    n = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    n = max(n, 1)
    if max_points is not None:
        n = min(n, max_points)
    n = int(n)

    # Compute coordinates
    x = np.round(np.linspace(p0[0], p1[0], n)).astype("int32")
    y = np.round(np.linspace(p0[1], p1[1], n)).astype("int32")

    # Extract values and concatenate into vector
    vals = np.stack([I[yi,xi] for xi, yi in zip(x,y)])
    return vals


def match_peaks_frame(peaks_t, peak_vals_t, pafs_t, skeleton,
    min_score_to_node_ratio=0.4, min_score_midpts=0.05, min_score_integral=0.8, add_last_edge=False):
    """ Matches single frame """
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
        paf_x = pafs_t[...,2*k]
        paf_y = pafs_t[...,2*k+1]
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
                    score_with_dist_prior = np.mean(score_midpts) + min(0.5 * paf_x.shape[0] / norm - 1, 0)
                    score_integral = np.mean(score_midpts > min_score_midpts)
                    if score_with_dist_prior > 0 and score_integral > min_score_integral:
                        connection_candidates.append([i, j, score_with_dist_prior])

            # Sort candidates for current edge by descending score
            connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)

            # Add to list of candidates for next step
            connection = np.zeros((0,5)) # src_id, dst_id, paf_score, i, j
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
    subset = -1 * np.ones((0, len(skeleton.nodes)+2)) # ids, overall score, number of parts
    candidate = np.array([y for x in peaks_t for y in x]) # flattened set of all points
    candidate_scores = np.array([y for x in peak_vals_t for y in x]) # flattened set of all peak scores
    for k, edge in enumerate(skeleton.edge_names):
        # No matches for this edge
        if k in special_k:
            continue

        # Get IDs for current connection
        partAs = connection_all[k][:,0]
        partBs = connection_all[k][:,1]

        # Get edge
        indexA, indexB = (skeleton.node_to_index(edge[0]), skeleton.node_to_index(edge[1]))

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
                if subset[j][indexB] != partBs[i]: # did we already assign this part?
                    subset[j][indexB] = partBs[i] # assign part
                    subset[j][-1] += 1 # increment instance part counter
                    subset[j][-2] += candidate_scores[int(partBs[i])] + connection_all[k][i][2] # add peak + edge score

            # Both candidate points found in matched subset
            elif found == 2:
                j1, j2 = subset_idx # get indices in matched subset
                membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2] # count number of instances per body parts
                # All body parts are disjoint, merge them
                if np.all(membership < 2):
                    subset[j1][:-2] += (subset[j2][:-2] + 1)
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, axis=0)

                # Treat them separately
                else:
                    subset[j1][indexB] = partBs[i]
                    subset[j1][-1] += 1
                    subset[j1][-2] += candidate_scores[partBs[i].astype(int)] + connection_all[k][i][2]

            # Neither point found, create a new subset (if not the last edge)
            elif found == 0 and (add_last_edge or (k < (len(skeleton.edges)-1))):
                row = -1 * np.ones(len(skeleton.nodes)+2)
                row[indexA] = partAs[i] # ID
                row[indexB] = partBs[i] # ID
                row[-1] = 2 # initial count
                row[-2] = sum(candidate_scores[connection_all[k][i, :2].astype(int)]) + connection_all[k][i][2] # score
                subset = np.vstack([subset, row]) # add to matched subset

    # Filter small instances
    score_to_node_ratio = subset[:,-2] / subset[:,-1]
    subset = subset[score_to_node_ratio > min_score_to_node_ratio, :]

    # Done with all the matching! Gather the data
    matched_instances_t = []
    match_scores_t = []
    matched_peak_vals_t = []
    for match in subset:
        pts = np.full((len(skeleton.nodes), 2), np.nan)
        peak_vals = np.full((len(skeleton.nodes),), np.nan)
        for i in range(len(pts)):
            if match[i] >= 0:
                pts[i,:] = candidate[int(match[i]),:2]
                peak_vals[i] = candidate_scores[int(match[i])]
        matched_instances_t.append(pts)
        match_scores_t.append(match[-2]) # score
        matched_peak_vals_t.append(peak_vals)

    return matched_instances_t, match_scores_t, matched_peak_vals_t

def match_peaks_paf(peaks, peak_vals, pafs, skeleton,
    min_score_to_node_ratio=0.4, min_score_midpts=0.05, min_score_integral=0.8, add_last_edge=False):
    """ Computes PAF-based peak matching via greedy assignment and other such dragons """

    # Process each frame
    matched_instances = []
    match_scores = []
    matched_peak_vals = []
    for peaks_t, peak_vals_t, pafs_t in zip(peaks, peak_vals, pafs):
        matched_instances_i, match_scores_i, matched_peak_vals_i = match_peaks_frame(peaks_t, peak_vals_t, pafs_t, skeleton,
            min_score_to_node_ratio=min_score_to_node_ratio, min_score_midpts=min_score_midpts, min_score_integral=min_score_integral, add_last_edge=add_last_edge)

        matched_instances.append(matched_instances_i)
        match_scores.append(match_scores_i)
        matched_peak_vals.append(matched_peak_vals_i)

    return matched_instances, match_scores, matched_peak_vals

def match_peaks_paf_par(peaks, peak_vals, pafs, skeleton,
    min_score_to_node_ratio=0.4, min_score_midpts=0.05, min_score_integral=0.8, add_last_edge=False, pool=None):
    """ Parallel version of PAF peak matching """

    if pool is None:
        pool = multiprocessing.Pool()
    # with multiprocessing.Pool() as pool:

    futures = []
    for peaks_t, peak_vals_t, pafs_t in zip(peaks, peak_vals, pafs):
        future = pool.apply_async(match_peaks_frame, [peaks_t, peak_vals_t, pafs_t, skeleton], dict(min_score_to_node_ratio=min_score_to_node_ratio, min_score_midpts=min_score_midpts, min_score_integral=min_score_integral, add_last_edge=add_last_edge))
        futures.append(future)

    matched_instances = []
    match_scores = []
    matched_peak_vals = []
    for future in futures:
        matched_instances_i, match_scores_i, matched_peak_vals_i = future.get()

        matched_instances.append(matched_instances_i)
        match_scores.append(match_scores_i)
        matched_peak_vals.append(matched_peak_vals_i)

    return matched_instances, match_scores, matched_peak_vals


