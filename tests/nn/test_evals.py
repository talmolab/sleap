import numpy as np
import sleap
from sleap.nn.evals import load_metrics


sleap.use_cpu_only()


def test_load_metrics(min_centered_instance_model_path):
    model_path = min_centered_instance_model_path

    metrics = load_metrics(f"{model_path}/metrics.val.npz")
    assert "oks_voc.mAP" in metrics

    metrics = load_metrics(model_path, split="val")
    assert "oks_voc.mAP" in metrics

    metrics = load_metrics(model_path, split="train")
    assert "oks_voc.mAP" in metrics
