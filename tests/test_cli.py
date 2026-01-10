"""Tests for the SLEAP CLI module.

This module tests the main CLI entry point and sleap-io command integration.
"""

import pytest
from click.testing import CliRunner

from sleap.cli import cli


class TestCLIBasics:
    """Tests for basic CLI functionality."""

    def test_cli_help(self):
        """Verify main CLI help displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SLEAP" in result.output
        assert "label" in result.output
        assert "doctor" in result.output

    def test_cli_version(self):
        """Verify version flag works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "sleap" in result.output.lower()

    def test_label_help(self):
        """Verify label command help displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["label", "--help"])
        assert result.exit_code == 0
        assert "Launch the SLEAP labeling GUI" in result.output

    def test_doctor_help(self):
        """Verify doctor command help displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "diagnostics" in result.output.lower()


class TestSleapIoIntegration:
    """Tests for sleap-io CLI command integration.

    These tests verify that sleap-io commands are properly inherited and branded.
    """

    def test_sleap_io_commands_are_registered(self):
        """Verify all sleap-io commands are registered on the CLI."""
        expected_commands = ["show", "convert", "split", "filenames", "render"]
        registered_commands = list(cli.commands.keys())

        for cmd in expected_commands:
            assert cmd in registered_commands, (
                f"Command '{cmd}' not found in CLI. "
                f"Available commands: {registered_commands}."
            )

    def test_show_command_registered(self):
        """Verify show command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["show", "--help"])
        assert result.exit_code == 0
        assert "Print labels file summary" in result.output

    def test_show_help_has_sleap_branding(self):
        """Verify show help uses sleap, not sio, in examples."""
        runner = CliRunner()
        result = runner.invoke(cli, ["show", "--help"])
        assert "$ sleap show" in result.output
        assert "$ sio show" not in result.output

    def test_convert_command_registered(self):
        """Verify convert command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Convert between pose data formats" in result.output

    def test_convert_help_has_sleap_branding(self):
        """Verify convert help uses sleap, not sio, in examples."""
        runner = CliRunner()
        result = runner.invoke(cli, ["convert", "--help"])
        assert "$ sleap convert" in result.output
        assert "$ sio convert" not in result.output

    def test_split_command_registered(self):
        """Verify split command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["split", "--help"])
        assert result.exit_code == 0
        assert "Split labels" in result.output or "train/val/test" in result.output

    def test_filenames_command_registered(self):
        """Verify filenames command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["filenames", "--help"])
        assert result.exit_code == 0
        assert "video" in result.output.lower()

    def test_render_command_registered(self):
        """Verify render command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["render", "--help"])
        assert result.exit_code == 0
        assert "Render" in result.output

    def test_main_help_shows_all_commands(self):
        """Verify main help lists all commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Native commands
        assert "label" in result.output
        assert "doctor" in result.output
        # Inherited commands
        assert "show" in result.output
        assert "convert" in result.output
        assert "split" in result.output
        assert "filenames" in result.output
        assert "render" in result.output

    def test_show_with_file(self, tmp_path):
        """Verify show command works with an actual file."""
        # Create a minimal labels file using sleap-io
        from sleap_io import Labels, Skeleton

        skeleton = Skeleton(["A", "B"])
        labels = Labels(skeletons=[skeleton])
        labels_path = tmp_path / "test.slp"
        labels.save(str(labels_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["show", str(labels_path)])
        assert result.exit_code == 0
        assert "test.slp" in result.output


class TestDefaultGroupBehavior:
    """Tests for DefaultGroup functionality."""

    def test_unrecognized_command_falls_back_to_label(self):
        """Verify unrecognized commands fall back to label."""
        runner = CliRunner()
        # This should try to open "nonexistent.slp" with the label command
        # It will fail because the file doesn't exist, but the error should
        # come from the label command, not a "command not found" error
        result = runner.invoke(cli, ["nonexistent.slp"])
        # The label command should be invoked (might fail for other reasons)
        # We mainly check it doesn't say "No such command"
        assert "No such command" not in result.output
