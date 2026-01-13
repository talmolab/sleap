"""Tests for the SLEAP CLI module.

This module tests the main CLI entry point and sleap-io command integration.
"""

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

    def test_doctor_runs(self):
        """Verify doctor command runs and produces output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        # Check for expected sections in output
        assert "Platform" in result.output
        assert "Python" in result.output
        assert "GPU" in result.output or "CUDA" in result.output
        assert "Packages" in result.output

    def test_doctor_json(self):
        """Verify doctor --json outputs valid JSON."""
        import json

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert "sleap_version" in data
        assert "platform" in data
        assert "python" in data
        assert "packages" in data

    def test_doctor_json_has_new_fields(self):
        """Verify doctor --json includes new diagnostic fields."""
        import json

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        # Check new fields
        assert "gpu" in data
        assert "binaries" in data
        assert "path_entries" in data
        assert "path_conflicts" in data
        # Check platform has new fields
        assert "ram_used" in data["platform"]
        assert "disk_used" in data["platform"]

    def test_doctor_has_new_sections(self):
        """Verify doctor has new consolidated sections."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        # Check for new sections
        assert "[Platform]" in result.output
        assert "[Python]" in result.output
        assert "[Packages]" in result.output
        assert "[PATH" in result.output
        # Check for new output format
        assert "Generated:" in result.output
        assert "RAM:" in result.output or "Disk:" in result.output

    def test_doctor_output_to_file(self, tmp_path):
        """Verify doctor -o writes to file."""
        output_file = tmp_path / "doctor.txt"

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "-o", str(output_file)])
        assert result.exit_code == 0

        # File should exist and have content
        assert output_file.exists()
        content = output_file.read_text()
        assert "SLEAP System Diagnostics" in content
        assert "[Platform]" in content
        assert "[Python]" in content

    def test_doctor_output_auto_filename(self, tmp_path, monkeypatch):
        """Verify doctor -o auto creates auto-timestamped file."""
        # Change to tmp_path so auto file is created there
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        # Use "-o auto" since Click requires an argument for the option
        result = runner.invoke(cli, ["doctor", "-o", "auto"])
        assert result.exit_code == 0
        assert "Saved to:" in result.output

        # Find the auto-generated file
        import glob

        files = glob.glob(str(tmp_path / "sleap-doctor-*.txt"))
        assert len(files) == 1
        content = open(files[0]).read()
        assert "SLEAP System Diagnostics" in content

    def test_doctor_file_content_has_all_sections(self, tmp_path):
        """Verify saved file has all diagnostic sections."""
        output_file = tmp_path / "doctor.txt"

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "-o", str(output_file)])
        assert result.exit_code == 0

        content = output_file.read_text()
        # Verify all sections are present
        assert "[Platform]" in content
        assert "[Python]" in content
        assert "[Packages]" in content
        assert "[PATH" in content

    def test_doctor_shows_tip_when_not_saving(self):
        """Verify doctor shows -o tip when not saving to file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "Tip:" in result.output
        assert "sleap doctor -o" in result.output


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
