"""Tests for edge cases and error handling.

This module tests error handling scenarios including:
- Bottom-up skeleton validation (arborescence check)
- Config file loading errors
- Subprocess error handling
"""

import pytest
from unittest.mock import patch, MagicMock

from omegaconf import OmegaConf, DictConfig

from sleap_io import Skeleton, Labels

from sleap.sleap_io_adaptors.skeleton_utils import (
    is_arborescence,
    root_nodes,
    in_degree_over_one,
    cycles,
)
from sleap.gui.learning.configs import ConfigFileInfo, _quick_scan_yaml_metadata
from sleap.gui.learning.runners import InferenceTask


# =============================================================================
# Bottom-up Skeleton Validation Tests
# =============================================================================


class TestSkeletonValidation:
    """Tests for skeleton validation functions used by bottom-up pipeline."""

    @pytest.fixture
    def valid_arborescence_skeleton(self):
        """Create a valid arborescence skeleton (tree with single root)."""
        # A -> B -> C (linear chain is valid arborescence)
        skeleton = Skeleton(nodes=["A", "B", "C"], name="valid_tree")
        skeleton.add_edge("A", "B")
        skeleton.add_edge("B", "C")
        return skeleton

    @pytest.fixture
    def skeleton_with_multiple_roots(self):
        """Create a skeleton with multiple roots (invalid for bottom-up)."""
        # A -> B, C -> D (two disconnected trees = two roots)
        skeleton = Skeleton(nodes=["A", "B", "C", "D"], name="multi_root")
        skeleton.add_edge("A", "B")
        skeleton.add_edge("C", "D")
        return skeleton

    @pytest.fixture
    def skeleton_with_cycle(self):
        """Create a skeleton with a cycle (invalid for bottom-up)."""
        # A -> B -> C -> A (circular)
        skeleton = Skeleton(nodes=["A", "B", "C"], name="cycle")
        skeleton.add_edge("A", "B")
        skeleton.add_edge("B", "C")
        skeleton.add_edge("C", "A")
        return skeleton

    @pytest.fixture
    def skeleton_with_high_in_degree(self):
        """Create a skeleton where a node has in-degree > 1 (invalid)."""
        # A -> C, B -> C (C has in-degree 2)
        skeleton = Skeleton(nodes=["A", "B", "C"], name="high_in_degree")
        skeleton.add_edge("A", "C")
        skeleton.add_edge("B", "C")
        return skeleton

    def test_valid_arborescence(self, valid_arborescence_skeleton):
        """Valid arborescence should return True."""
        assert is_arborescence(valid_arborescence_skeleton) is True

    def test_multiple_roots_not_arborescence(self, skeleton_with_multiple_roots):
        """Skeleton with multiple roots should not be an arborescence."""
        assert is_arborescence(skeleton_with_multiple_roots) is False

    def test_cycle_not_arborescence(self, skeleton_with_cycle):
        """Skeleton with cycle should not be an arborescence."""
        assert is_arborescence(skeleton_with_cycle) is False

    def test_high_in_degree_not_arborescence(self, skeleton_with_high_in_degree):
        """Skeleton with in-degree > 1 should not be an arborescence."""
        assert is_arborescence(skeleton_with_high_in_degree) is False

    def test_root_nodes_single_root(self, valid_arborescence_skeleton):
        """Valid skeleton should have exactly one root node."""
        roots = root_nodes(valid_arborescence_skeleton)
        assert len(roots) == 1
        # root_nodes returns node names (strings), not Node objects
        assert roots[0] == "A"

    def test_root_nodes_multiple_roots(self, skeleton_with_multiple_roots):
        """Skeleton with multiple roots should report all of them."""
        roots = root_nodes(skeleton_with_multiple_roots)
        # root_nodes returns node names (strings), not Node objects
        assert len(roots) == 2
        assert "A" in roots
        assert "C" in roots

    def test_in_degree_over_one_valid(self, valid_arborescence_skeleton):
        """Valid skeleton should have no nodes with in-degree > 1."""
        high_in_degree = in_degree_over_one(valid_arborescence_skeleton)
        assert len(high_in_degree) == 0

    def test_in_degree_over_one_invalid(self, skeleton_with_high_in_degree):
        """Skeleton with convergent edges should report affected nodes."""
        high_in_degree = in_degree_over_one(skeleton_with_high_in_degree)
        assert len(high_in_degree) == 1
        # in_degree_over_one returns node names (strings), not Node objects
        assert high_in_degree[0] == "C"

    def test_cycles_none(self, valid_arborescence_skeleton):
        """Valid skeleton should have no cycles."""
        found_cycles = cycles(valid_arborescence_skeleton)
        assert len(found_cycles) == 0

    def test_cycles_found(self, skeleton_with_cycle):
        """Skeleton with cycle should report the cycle."""
        found_cycles = cycles(skeleton_with_cycle)
        assert len(found_cycles) >= 1
        # cycles returns node names (strings), not Node objects
        assert set(found_cycles[0]) == {"A", "B", "C"}


class TestBottomUpValidationInDialog:
    """Tests for bottom-up validation in LearningDialog."""

    @pytest.fixture
    def mock_cfg_getter(self):
        """Create a mock TrainingConfigsGetter."""
        getter = MagicMock()
        getter.get_first.return_value = None
        getter.update.return_value = None
        getter.get_filtered.return_value = []
        return getter

    def test_validate_pipeline_bottom_up_invalid_skeleton(
        self, qtbot, tmp_path, mock_cfg_getter
    ):
        """Bottom-up with invalid skeleton should disable run button."""
        from sleap.gui.learning.dialog import LearningDialog

        # Create skeleton with cycle (invalid for bottom-up)
        skeleton = Skeleton(nodes=["A", "B", "C"], name="cycle")
        skeleton.add_edge("A", "B")
        skeleton.add_edge("B", "C")
        skeleton.add_edge("C", "A")

        labels = Labels(skeletons=[skeleton])
        labels_file = tmp_path / "test.slp"
        labels_file.touch()

        with patch(
            "sleap.gui.learning.dialog.configs.TrainingConfigsGetter.make_from_labels_filename",
            return_value=mock_cfg_getter,
        ):
            dialog = LearningDialog(
                mode="training",
                labels_filename=str(labels_file),
                labels=labels,
                skeleton=skeleton,
            )
            qtbot.addWidget(dialog)

            # Set to bottom-up pipeline
            dialog.pipeline_form_widget.current_pipeline = "bottom-up"
            dialog.set_pipeline("bottom-up")

            # Validate pipeline
            dialog._validate_pipeline()

            # Run button should be disabled
            assert dialog.run_button.isEnabled() is False
            # Message should mention arborescence
            assert "arborescence" in dialog.message_widget.text().lower()

    def test_validate_pipeline_bottom_up_valid_skeleton(
        self, qtbot, tmp_path, mock_cfg_getter
    ):
        """Bottom-up with valid skeleton should enable run button."""
        from sleap.gui.learning.dialog import LearningDialog

        # Create valid arborescence skeleton
        skeleton = Skeleton(nodes=["A", "B", "C"], name="valid")
        skeleton.add_edge("A", "B")
        skeleton.add_edge("B", "C")

        labels = Labels(skeletons=[skeleton])
        labels_file = tmp_path / "test.slp"
        labels_file.touch()

        with patch(
            "sleap.gui.learning.dialog.configs.TrainingConfigsGetter.make_from_labels_filename",
            return_value=mock_cfg_getter,
        ):
            dialog = LearningDialog(
                mode="training",
                labels_filename=str(labels_file),
                labels=labels,
                skeleton=skeleton,
            )
            qtbot.addWidget(dialog)

            # Set to bottom-up pipeline
            dialog.pipeline_form_widget.current_pipeline = "bottom-up"
            dialog.set_pipeline("bottom-up")

            # Validate pipeline
            dialog._validate_pipeline()

            # Run button should be enabled (no validation errors)
            assert dialog.run_button.isEnabled() is True


# =============================================================================
# Config File Loading Error Tests
# =============================================================================


class TestConfigFileLoadingErrors:
    """Tests for config file loading error handling."""

    def test_load_nonexistent_config_file(self):
        """Loading nonexistent config should handle error gracefully."""
        config_info = ConfigFileInfo(path="/nonexistent/path/config.yaml")

        # Access config property should attempt to load
        config = config_info.config

        # Should return None, not raise exception
        assert config is None

    def test_load_malformed_yaml_config(self, tmp_path):
        """Loading malformed YAML should handle error gracefully."""
        # Create malformed YAML file
        config_file = tmp_path / "malformed.yaml"
        config_file.write_text("this is not: valid: yaml: content: [[[")

        config_info = ConfigFileInfo(path=str(config_file))

        # Access config property should attempt to load
        config = config_info.config

        # Should return None, not raise exception
        assert config is None

    def test_load_empty_config_file(self, tmp_path):
        """Loading empty config file should handle gracefully."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config_info = ConfigFileInfo(path=str(config_file))

        # Should not raise, may return empty config or None
        config = config_info.config
        # Empty YAML loads as empty DictConfig in OmegaConf
        assert config is None or isinstance(config, (dict, DictConfig))

    def test_config_info_is_loaded_property(self, tmp_path):
        """is_loaded should reflect actual loading state."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("trainer_config:\n  run_name: test")

        config_info = ConfigFileInfo(path=str(config_file))

        # Before accessing config
        assert config_info.is_loaded is False

        # Access config to trigger load
        _ = config_info.config

        # After accessing config
        assert config_info.is_loaded is True

    def test_config_info_path_none(self):
        """ConfigFileInfo with None path should handle gracefully."""
        config_info = ConfigFileInfo(path=None)

        # Should return None, not raise
        assert config_info.config is None
        assert config_info.is_loaded is False

    def test_quick_scan_yaml_invalid_file(self, tmp_path):
        """Quick scan of invalid YAML should return None, None."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("this: is: not: valid: [[")

        head_type, run_name = _quick_scan_yaml_metadata(str(config_file))

        assert head_type is None
        assert run_name is None

    def test_quick_scan_yaml_nonexistent_file(self):
        """Quick scan of nonexistent file should return None, None."""
        head_type, run_name = _quick_scan_yaml_metadata("/nonexistent/config.yaml")

        assert head_type is None
        assert run_name is None

    def test_quick_scan_yaml_valid_file(self, tmp_path):
        """Quick scan of valid YAML should extract metadata."""
        config_file = tmp_path / "valid.yaml"
        config_file.write_text(
            """
model_config:
  head_configs:
    bottomup:
      confmaps:
        sigma: 5.0
trainer_config:
  run_name: my_run
"""
        )

        head_type, run_name = _quick_scan_yaml_metadata(str(config_file))

        assert head_type == "bottomup"
        assert run_name == "my_run"

    def test_has_trained_model_no_checkpoint(self, tmp_path):
        """has_trained_model should return False when no checkpoint exists."""
        config_file = tmp_path / "training_config.yaml"
        config_file.write_text("trainer_config:\n  run_name: test")

        config_info = ConfigFileInfo(path=str(config_file))

        assert config_info.has_trained_model is False

    def test_has_trained_model_with_checkpoint(self, tmp_path):
        """has_trained_model should return True when best.ckpt exists."""
        config_file = tmp_path / "training_config.yaml"
        config_file.write_text("trainer_config:\n  run_name: test")

        # Create fake checkpoint
        (tmp_path / "best.ckpt").touch()

        config_info = ConfigFileInfo(path=str(config_file))

        assert config_info.has_trained_model is True


# =============================================================================
# Subprocess Error Handling Tests
# =============================================================================


class TestSubprocessErrorHandling:
    """Tests for subprocess error handling in runners."""

    @pytest.fixture
    def mock_labels(self):
        """Create mock labels."""
        labels = MagicMock(spec=Labels)
        labels.videos = []
        return labels

    def test_predict_subprocess_success(self, mock_labels, tmp_path):
        """Successful subprocess should return 'success'."""
        labels_file = tmp_path / "labels.slp"
        task = InferenceTask(
            trained_job_paths=[str(tmp_path / "model")],
            inference_params={},
            labels=mock_labels,
            labels_filename=str(labels_file),
        )

        # Mock subprocess.Popen
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]  # First poll returns None, then 0
        mock_proc.returncode = 0
        mock_proc.stdout.readline.return_value = b""
        mock_proc.__enter__ = MagicMock(return_value=mock_proc)
        mock_proc.__exit__ = MagicMock(return_value=False)

        mock_item = MagicMock()
        mock_item.cli_args = ["--data_path", str(tmp_path / "data")]
        mock_item.path = str(tmp_path / "data")

        with patch(
            "sleap.gui.learning.runners.subprocess.Popen", return_value=mock_proc
        ):
            output_path, result = task.predict_subprocess(
                mock_item, append_results=False
            )

        assert result == "success"

    def test_predict_subprocess_failure(self, mock_labels, tmp_path):
        """Failed subprocess should return error code."""
        labels_file = tmp_path / "labels.slp"
        task = InferenceTask(
            trained_job_paths=[str(tmp_path / "model")],
            inference_params={},
            labels=mock_labels,
            labels_filename=str(labels_file),
        )

        # Mock subprocess.Popen with failure
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [
            None,
            1,
        ]  # First poll returns None, then 1 (error)
        mock_proc.returncode = 1
        mock_proc.stdout.readline.return_value = b""
        mock_proc.__enter__ = MagicMock(return_value=mock_proc)
        mock_proc.__exit__ = MagicMock(return_value=False)

        mock_item = MagicMock()
        mock_item.cli_args = ["--data_path", str(tmp_path / "data")]
        mock_item.path = str(tmp_path / "data")

        with patch(
            "sleap.gui.learning.runners.subprocess.Popen", return_value=mock_proc
        ):
            output_path, result = task.predict_subprocess(
                mock_item, append_results=False
            )

        # Should return the error code
        assert result == 1

    def test_predict_subprocess_canceled(self, mock_labels, tmp_path):
        """Canceled subprocess should return 'canceled'."""
        labels_file = tmp_path / "labels.slp"
        task = InferenceTask(
            trained_job_paths=[str(tmp_path / "model")],
            inference_params={},
            labels=mock_labels,
            labels_filename=str(labels_file),
        )

        # Mock subprocess.Popen
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Never finishes naturally
        mock_proc.pid = 12345
        mock_proc.stdout.readline.return_value = b""
        mock_proc.__enter__ = MagicMock(return_value=mock_proc)
        mock_proc.__exit__ = MagicMock(return_value=False)

        mock_item = MagicMock()
        mock_item.cli_args = ["--data_path", str(tmp_path / "data")]
        mock_item.path = str(tmp_path / "data")

        # Callback that returns cancel
        def cancel_callback(**kwargs):
            return "cancel"

        with patch(
            "sleap.gui.learning.runners.subprocess.Popen", return_value=mock_proc
        ):
            with patch("sleap.gui.learning.runners.kill_process"):
                output_path, result = task.predict_subprocess(
                    mock_item, append_results=False, waiting_callback=cancel_callback
                )

        assert result == "canceled"

    def test_train_subprocess_sleap_nn_not_installed(self, mock_labels):
        """train_subprocess should handle ImportError for sleap-nn."""
        from sleap.gui.learning.runners import train_subprocess

        mock_config = OmegaConf.create(
            {
                "trainer_config": {
                    "ckpt_dir": "/path/to/models",
                    "run_name": "test_run",
                }
            }
        )

        # Mock ImportError for sleap_nn
        with patch(
            "sleap.gui.learning.runners.filter_cfg",
            side_effect=ImportError("No module named 'sleap_nn'"),
        ):
            with patch("sleap.gui.learning.runners.show_sleap_nn_installation_message"):
                run_path, result = train_subprocess(
                    job_config=mock_config,
                    labels_filename="/path/to/labels.slp",
                    inference_params={
                        "controller_port": 5555,
                        "publish_port": 5556,
                    },
                )

        assert result == "error"

    def test_inference_task_json_parsing_error_handled(self, mock_labels, tmp_path):
        """JSON parsing errors in subprocess output should be handled."""
        labels_file = tmp_path / "labels.slp"
        task = InferenceTask(
            trained_job_paths=[str(tmp_path / "model")],
            inference_params={},
            labels=mock_labels,
            labels_filename=str(labels_file),
        )

        # Mock subprocess with invalid JSON output
        mock_proc = MagicMock()
        # Return invalid JSON then finish
        mock_proc.poll.side_effect = [None, None, 0]
        mock_proc.returncode = 0
        mock_proc.stdout.readline.side_effect = [
            b"{invalid json",  # Invalid JSON starting with {
            b"regular output",  # Regular output
            b"",
        ]
        mock_proc.__enter__ = MagicMock(return_value=mock_proc)
        mock_proc.__exit__ = MagicMock(return_value=False)

        mock_item = MagicMock()
        mock_item.cli_args = ["--data_path", str(tmp_path / "data")]
        mock_item.path = str(tmp_path / "data")

        with patch(
            "sleap.gui.learning.runners.subprocess.Popen", return_value=mock_proc
        ):
            # Should not raise, should handle invalid JSON gracefully
            output_path, result = task.predict_subprocess(
                mock_item, append_results=False
            )

        assert result == "success"


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_config_info_path_dir_yaml(self, tmp_path):
        """path_dir should return parent directory for YAML files."""
        config_file = tmp_path / "subdir" / "training_config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.touch()

        config_info = ConfigFileInfo(path=str(config_file))

        assert config_info.path_dir == str(tmp_path / "subdir")

    def test_config_info_path_dir_directory(self, tmp_path):
        """path_dir should return the path itself for directories."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        config_info = ConfigFileInfo(path=str(model_dir))

        assert config_info.path_dir == str(model_dir)

    def test_has_trained_model_cached(self, tmp_path):
        """has_trained_model result should be cached."""
        config_file = tmp_path / "training_config.yaml"
        config_file.write_text("trainer_config:\n  run_name: test")

        config_info = ConfigFileInfo(path=str(config_file))

        # First call
        result1 = config_info.has_trained_model
        assert result1 is False

        # Create checkpoint after first check
        (tmp_path / "best.ckpt").touch()

        # Second call should return cached value (False)
        result2 = config_info.has_trained_model
        assert result2 is False  # Cached value

        # Clear cache and check again
        config_info._has_trained_model_cache = None
        result3 = config_info.has_trained_model
        assert result3 is True  # Fresh check finds the checkpoint
