import numpy as np

from sleap import Labels
from sleap.info.metrics import (
    match_instance_lists_nodewise,
    matched_instance_distances,
)


def test_matched_instance_distances(centered_pair_labels, centered_pair_predictions):
    labels_gt = centered_pair_labels
    labels_pr = centered_pair_predictions

    # Match each ground truth instance node to the closest corresponding node
    # from any predicted instance in the same frame.

    inst_matching_func = match_instance_lists_nodewise

    # Calculate distances
    frame_idxs, D, points_gt, points_pr = matched_instance_distances(
        labels_gt, labels_pr, inst_matching_func
    )

    # Show mean difference for each node
    node_names = labels_gt.skeletons[0].node_names
    expected_values = {
        "head": 0.872426920709296,
        "neck": 0.8016280746914615,
        "thorax": 0.8602021363390538,
        "abdomen": 1.01012200038258,
        "wingL": 1.1297727023475939,
        "wingR": 1.0869857897008424,
        "forelegL1": 0.780584225081443,
        "forelegL2": 1.170805798894702,
        "forelegL3": 1.1020486509389473,
        "forelegR1": 0.9014698776116817,
        "forelegR2": 0.9448001033112047,
        "forelegR3": 1.308385214215777,
        "midlegL1": 0.9095691623265347,
        "midlegL2": 1.2203595627907582,
        "midlegL3": 0.9813843358470163,
        "midlegR1": 0.9871017182813739,
        "midlegR2": 1.0209829335569256,
        "midlegR3": 1.0990681234096988,
        "hindlegL1": 1.0005335192834348,
        "hindlegL2": 1.273539518539708,
        "hindlegL3": 1.1752245985832817,
        "hindlegR1": 1.1402833959265248,
        "hindlegR2": 1.3143221301212737,
        "hindlegR3": 1.0441458592503365,
    }

    for node_idx, node_name in enumerate(node_names):
        mean_d = np.nanmean(D[..., node_idx])
        assert np.isclose(mean_d, expected_values[node_name], atol=1e-6)
