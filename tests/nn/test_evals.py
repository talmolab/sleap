import numpy as np

from typing import List, Tuple

import sleap

from sleap import Instance, PredictedInstance
from sleap.instance import Point
from sleap.nn.evals import (
    compute_dists,
    compute_dist_metrics,
    compute_oks,
    load_metrics,
)


sleap.use_cpu_only()


def test_compute_oks():
    inst_gt = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)

    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 2 / 3)

    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [2, 2]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)

    inst_gt = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    inst_pr = np.array([[0, 0], [1, 1], [np.nan, np.nan]]).astype("float32")
    oks = compute_oks(inst_gt, inst_pr)
    np.testing.assert_allclose(oks, 1)


def test_compute_dists(instances, predicted_instances):
    # Make some changes to the instances
    error_start = 10
    error_end = 20
    expected_dists = []
    for offset, zipped_insts in enumerate(
        zip(
            instances[error_start:error_end], predicted_instances[error_start:error_end]
        )
    ):

        inst, pred_inst = zipped_insts
        for node_name in inst.skeleton.node_names:
            pred_point = pred_inst[node_name]
            if pred_point != np.NaN:
                inst[node_name] = Point(
                    pred_point.x + offset, pred_point.y + offset + 1
                )

        error = ((offset ** 2) + (offset + 1) ** 2) ** (1 / 2)
        expected_dists.append(error)

    best_match_oks = np.NaN
    positive_pairs: List[Tuple[Instance, PredictedInstance]] = [
        (inst, pred_inst, best_match_oks)
        for inst, pred_inst in zip(instances, predicted_instances)
    ]

    dists_dict = compute_dists(positive_pairs=positive_pairs)
    dists = dists_dict["dists"]

    # Replace nan to 0
    dists_no_nan = np.nan_to_num(dists, nan=0)
    np.testing.assert_allclose(dists_no_nan[0:10], 0)

    # Replace nan to negative (which we never see in a norm)
    dists_no_nan = np.nan_to_num(dists, nan=-1)

    # Check distances are as expected
    for idx, error in enumerate(expected_dists):
        idx += error_start
        dists_idx = dists_no_nan[idx]
        dists_idx = dists_idx[dists_idx >= 0]
        np.testing.assert_allclose(dists_idx, error)

    # Check instances are as expected
    dists_metric = compute_dist_metrics(dists_dict)
    for idx, zipped_insts in enumerate(
        zip(dists_metric["dist.instances.gt"], dists_metric["dist.instances.pr"])
    ):
        inst, pred_inst = zipped_insts
        assert inst == instances[idx]
        assert pred_inst == predicted_instances[idx]


def test_load_metrics(min_centered_instance_model_path):
    model_path = min_centered_instance_model_path

    metrics = load_metrics(f"{model_path}/metrics.val.npz")
    assert "oks_voc.mAP" in metrics

    metrics = load_metrics(model_path, split="val")
    assert "oks_voc.mAP" in metrics

    metrics = load_metrics(model_path, split="train")
    assert "oks_voc.mAP" in metrics
