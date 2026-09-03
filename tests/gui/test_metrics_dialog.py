"""Tests for the Evaluation Metrics dialog (sleap/gui/dialogs/metrics.py)."""

import json

from sleap.gui.dialogs.metrics import _legacy_backbone_name


class _FakeConfig:
    """Minimal stand-in for ConfigFileInfo with just a path."""

    def __init__(self, path):
        self.path = path


def test_legacy_backbone_name_pretrained_encoder(tmp_path):
    """A legacy config with pretrained_encoder returns its name."""
    config = {
        "model": {
            "backbone": {
                "leap": None,
                "unet": None,
                "hourglass": None,
                "resnet": None,
                "pretrained_encoder": {
                    "encoder": "efficientnetb0",
                    "pretrained": True,
                },
            }
        }
    }
    p = tmp_path / "training_config.json"
    p.write_text(json.dumps(config))
    assert _legacy_backbone_name(_FakeConfig(str(p))) == "pretrained_encoder (legacy)"


def test_legacy_backbone_name_hourglass(tmp_path):
    config = {
        "model": {
            "backbone": {
                "leap": None,
                "unet": None,
                "hourglass": {"stem_stride": 4, "max_stride": 64},
                "resnet": None,
                "pretrained_encoder": None,
            }
        }
    }
    p = tmp_path / "training_config.json"
    p.write_text(json.dumps(config))
    assert _legacy_backbone_name(_FakeConfig(str(p))) == "hourglass (legacy)"


def test_legacy_backbone_name_returns_none_for_yaml():
    """Non-JSON configs return None (they're sleap-nn configs, not legacy)."""
    assert _legacy_backbone_name(_FakeConfig("model/training_config.yaml")) is None


def test_legacy_backbone_name_returns_none_for_missing_file():
    cfg = _FakeConfig("/nonexistent/training_config.json")
    assert _legacy_backbone_name(cfg) is None


def test_legacy_backbone_name_returns_none_for_all_null(tmp_path):
    """If every backbone entry is null, returns None (truly unknown)."""
    config = {"model": {"backbone": {"unet": None}}}
    p = tmp_path / "training_config.json"
    p.write_text(json.dumps(config))
    assert _legacy_backbone_name(_FakeConfig(str(p))) is None
