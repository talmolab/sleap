from pathlib import Path
import numpy as np
import tensorflow as tf

from typing import List, Tuple

import sleap

from sleap import Instance, PredictedInstance
from sleap.instance import Point
from sleap.nn.config.training_job import TrainingJobConfig
from sleap.nn.data.providers import LabelsReader
from sleap.nn.evals import (
    compute_dists,
    compute_dist_metrics,
    load_metrics,
    evaluate_model,
)
from sleap.nn.model import Model


sleap.use_cpu_only()


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
    for idx, zipped_metrics in enumerate(
        zip(dists_metric["dist.frame_idxs"], dists_metric["dist.video_paths"])
    ):
        frame_idx, video_path = zipped_metrics
        assert frame_idx == instances[idx].frame.frame_idx
        assert video_path == instances[idx].frame.video.backend.filename


def test_evaluate_model(min_labels_slp, min_bottomup_model_path):

    labels_reader = LabelsReader(labels=min_labels_slp, user_instances_only=True)
    model_dir: str = min_bottomup_model_path
    cfg = TrainingJobConfig.load_json(str(Path(model_dir, "training_config.json")))
    model = Model.from_config(
        config=cfg.model,
        skeleton=labels_reader.labels.skeletons[0],
        tracks=labels_reader.labels.tracks,
        update_config=True,
    )
    model.keras_model = tf.keras.models.load_model(
        Path(model_dir) / "best_model.h5", compile=False
    )

    labels_pr, metrics = evaluate_model(
        cfg=cfg,
        labels_gt=labels_reader,
        model=model,
        save=True,
        split_name="test",
    )
    assert metrics is not None  # If metrics is None, then the metrics were not saved


def test_load_metrics(min_centered_instance_model_path):
    model_path = min_centered_instance_model_path

    metrics = load_metrics(f"{model_path}/metrics.val.npz")
    assert "oks_voc.mAP" in metrics

    metrics = load_metrics(model_path, split="val")
    assert "oks_voc.mAP" in metrics

    metrics = load_metrics(model_path, split="train")
    assert "oks_voc.mAP" in metrics
