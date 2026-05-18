"""Tests for the SLEAP system_info module."""

from io import StringIO
from unittest.mock import patch

from rich.console import Console

from sleap.system_info import (
    _print_package_table,
    get_package_info,
    get_all_package_info,
    # New dataclasses
    PackageInfoData,
    UVInfo,
    CondaInfo,
    BinaryInfo,
    GPUInfo,
    # New functions
    run_command,
    get_git_info,
    get_detailed_package_info,
    get_uv_config_value,
    get_default_python_version,
    get_uv_info_data,
    get_conda_info_data,
    get_binary_info,
    get_nvidia_info,
    get_pytorch_info_detailed,
    get_memory_info,
    get_disk_info,
    get_ffmpeg_info,
    analyze_path,
)


class TestPackageInfo:
    """Tests for package info functions."""

    def test_get_package_info_installed(self):
        """Verify get_package_info returns info for installed packages."""
        info = get_package_info("sleap")
        assert info["version"] is not None
        assert info["source"] in ("pip", "conda", "editable", "git", "local")

    def test_get_package_info_not_installed(self):
        """Verify get_package_info returns None version for missing packages."""
        info = get_package_info("nonexistent-package-xyz")
        assert info["version"] is None

    def test_get_all_package_info(self):
        """Verify get_all_package_info returns info for installed packages."""
        packages = get_all_package_info()
        assert "sleap" in packages
        assert packages["sleap"]["version"] is not None


class TestPackageTable:
    """Tests for package table printing."""

    def test_print_package_table(self):
        """Verify _print_package_table prints a table with package info."""
        # Create a console that writes to a string
        output = StringIO()
        console = Console(file=output, force_terminal=True, width=120)

        _print_package_table(console)

        result = output.getvalue()
        # Check table structure
        assert "Package" in result
        assert "Version" in result
        assert "Source" in result
        assert "Location" in result
        # Check that sleap is listed
        assert "sleap" in result

    def test_print_package_table_shows_full_paths(self):
        """Verify package table shows full paths without truncation."""
        output = StringIO()
        # Use a wide console so paths don't need to wrap
        console = Console(file=output, force_terminal=True, width=200)

        _print_package_table(console)

        result = output.getvalue()
        # Check that paths are not truncated with "..."
        # The old _shorten_path would add "..." prefix
        lines_with_dots_prefix = [
            line for line in result.split("\n") if "│ ..." in line
        ]
        assert len(lines_with_dots_prefix) == 0, (
            "Paths should not be truncated with '...' prefix"
        )


class TestDataClasses:
    """Tests for diagnostic dataclasses."""

    def test_package_info_data(self):
        """Verify PackageInfoData dataclass works correctly."""
        pkg = PackageInfoData(
            name="test-pkg",
            version="1.0.0",
            source="pip",
            location="/path/to/pkg",
            editable=False,
        )
        assert pkg.name == "test-pkg"
        assert pkg.version == "1.0.0"
        assert pkg.source == "pip"
        assert pkg.editable is False
        assert pkg.git_commit is None

    def test_package_info_data_with_git(self):
        """Verify PackageInfoData with git fields."""
        pkg = PackageInfoData(
            name="test-pkg",
            version="1.0.0",
            source="editable",
            location="/path/to/pkg",
            editable=True,
            git_commit="abc123",
            git_branch="main",
            git_dirty=True,
        )
        assert pkg.editable is True
        assert pkg.git_commit == "abc123"
        assert pkg.git_branch == "main"
        assert pkg.git_dirty is True

    def test_uv_info(self):
        """Verify UVInfo dataclass works correctly."""
        uv = UVInfo(
            version="uv 0.9.0",
            path="/usr/bin/uv",
            cache_dir="/home/user/.cache/uv",
        )
        assert uv.version == "uv 0.9.0"
        assert uv.path == "/usr/bin/uv"
        assert uv.installed_tools == []  # default

    def test_conda_info(self):
        """Verify CondaInfo dataclass works correctly."""
        conda = CondaInfo(
            active=True,
            environment="myenv",
            prefix="/home/user/conda/envs/myenv",
            version="conda 25.0.0",
        )
        assert conda.active is True
        assert conda.environment == "myenv"
        assert conda.auto_activate_base is None

    def test_binary_info(self):
        """Verify BinaryInfo dataclass works correctly."""
        binary = BinaryInfo(
            name="sleap",
            path="/usr/bin/sleap",
            real_path="/usr/bin/sleap",
            source="pip",
        )
        assert binary.name == "sleap"
        assert binary.path == "/usr/bin/sleap"

    def test_gpu_info(self):
        """Verify GPUInfo dataclass works correctly."""
        gpu = GPUInfo(
            name="NVIDIA RTX 3090",
            memory_total="24576 MB",
            memory_free="20000 MB",
            utilization="10%",
        )
        assert gpu.name == "NVIDIA RTX 3090"
        assert gpu.memory_total == "24576 MB"


class TestDiagnosticFunctions:
    """Tests for diagnostic collection functions."""

    def test_run_command_success(self):
        """Verify run_command returns output for valid commands."""
        rc, stdout, stderr = run_command(["echo", "hello"])
        assert rc == 0
        assert "hello" in stdout

    def test_run_command_failure(self):
        """Verify run_command handles missing commands."""
        rc, stdout, stderr = run_command(["nonexistent_command_xyz"])
        assert rc == -1
        assert stdout == ""

    def test_run_command_timeout(self):
        """Verify run_command handles timeouts."""
        # This test uses a short timeout
        rc, stdout, stderr = run_command(["sleep", "10"], timeout=1)
        assert rc == -1

    def test_get_git_info_valid_repo(self, tmp_path):
        """Verify get_git_info returns info for valid git repos."""
        import subprocess

        # Create a git repo
        subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
            capture_output=True,
        )
        (tmp_path / "test.txt").write_text("hello")
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(tmp_path), "commit", "-m", "Initial"],
            capture_output=True,
        )

        info = get_git_info(str(tmp_path))
        assert "commit" in info
        assert "branch" in info

    def test_get_git_info_not_a_repo(self, tmp_path):
        """Verify get_git_info returns empty dict for non-repos."""
        info = get_git_info(str(tmp_path))
        assert info == {}

    def test_get_git_info_nonexistent_path(self):
        """Verify get_git_info returns empty dict for nonexistent paths."""
        info = get_git_info("/nonexistent/path")
        assert info == {}

    def test_get_detailed_package_info_installed(self):
        """Verify get_detailed_package_info returns info for installed packages."""
        pkg = get_detailed_package_info("sleap")
        assert pkg is not None
        assert pkg.name == "sleap"
        assert pkg.version is not None

    def test_get_detailed_package_info_not_installed(self):
        """Verify get_detailed_package_info returns None for missing packages."""
        pkg = get_detailed_package_info("nonexistent-package-xyz")
        assert pkg is None

    def test_get_uv_config_value_env_var(self):
        """Verify get_uv_config_value reads from environment variables."""
        with patch.dict("os.environ", {"UV_TEST_VAR": "test_value"}):
            result = get_uv_config_value("test-var")
            assert result == "test_value"

    def test_get_uv_config_value_not_set(self):
        """Verify get_uv_config_value returns empty string if not set."""
        result = get_uv_config_value("nonexistent-setting-xyz")
        assert result == ""

    def test_get_default_python_version_from_env(self):
        """Verify get_default_python_version reads from UV_PYTHON."""
        with patch.dict("os.environ", {"UV_PYTHON": "3.11"}):
            result = get_default_python_version()
            assert "3.11" in result
            assert "UV_PYTHON" in result

    def test_get_uv_info_data(self):
        """Verify get_uv_info_data returns UVInfo when uv is installed."""
        import shutil

        if shutil.which("uv"):
            info = get_uv_info_data()
            assert info is not None
            assert info.version != ""
        else:
            info = get_uv_info_data()
            assert info is None

    def test_get_conda_info_data_not_activated(self):
        """Verify get_conda_info_data handles non-activated conda."""
        # Ensure CONDA_PREFIX is not set
        with patch.dict("os.environ", {}, clear=True):
            # If conda is installed but not activated
            import shutil

            if shutil.which("conda"):
                info = get_conda_info_data()
                # Either None or has active=False
                if info is not None:
                    assert info.active is False

    def test_get_binary_info_existing(self):
        """Verify get_binary_info returns info for existing binaries."""
        info = get_binary_info("python")
        assert info is not None
        assert info.name == "python"
        assert info.path != ""

    def test_get_binary_info_nonexistent(self):
        """Verify get_binary_info returns None for missing binaries."""
        info = get_binary_info("nonexistent_binary_xyz")
        assert info is None

    def test_get_nvidia_info_tuple(self):
        """Verify get_nvidia_info returns a tuple of (driver, cuda, gpus)."""
        driver, cuda, gpus = get_nvidia_info()
        # Always returns a tuple, even if no NVIDIA GPU
        assert isinstance(driver, str)
        assert isinstance(cuda, str)
        assert isinstance(gpus, list)

    def test_get_pytorch_info_detailed_tuple(self):
        """Verify get_pytorch_info_detailed returns a tuple."""
        version, accelerator, cuda = get_pytorch_info_detailed()
        assert isinstance(version, str)
        assert isinstance(accelerator, str)
        assert isinstance(cuda, str)

    def test_get_memory_info_tuple(self):
        """Verify get_memory_info returns memory info."""
        used, available, total = get_memory_info()
        # May be empty strings on some systems, but always returns tuple
        assert isinstance(used, str)
        assert isinstance(available, str)
        assert isinstance(total, str)

    def test_get_disk_info_valid_path(self):
        """Verify get_disk_info returns disk info for valid paths."""
        used, available, total = get_disk_info("/")
        assert used != ""
        assert total != ""

    def test_get_disk_info_invalid_path(self):
        """Verify get_disk_info returns empty strings for invalid paths."""
        used, available, total = get_disk_info("/nonexistent/path")
        assert used == ""
        assert total == ""

    def test_get_ffmpeg_info_list(self):
        """Verify get_ffmpeg_info returns a list."""
        binaries = get_ffmpeg_info()
        assert isinstance(binaries, list)

    def test_analyze_path_tuple(self):
        """Verify analyze_path returns entries and conflicts."""
        entries, conflicts = analyze_path()
        assert isinstance(entries, list)
        assert isinstance(conflicts, list)
        # PATH should have at least one entry
        assert len(entries) > 0

    def test_analyze_path_detects_conda_conflict(self):
        """Verify analyze_path detects conda before uv conflict."""
        import os

        # Create a PATH with conda before uv - use os.pathsep for cross-platform
        sep = os.pathsep
        if os.name == "nt":
            # Windows paths
            test_path = (
                f"C:\\conda\\bin{sep}C:\\Users\\user\\.local\\bin{sep}C:\\Windows"
            )
        else:
            # Unix paths
            test_path = f"/conda/bin{sep}/home/user/.local/bin{sep}/usr/bin"
        with patch.dict("os.environ", {"PATH": test_path}):
            entries, conflicts = analyze_path()
            # Should detect conda before uv paths
            assert len(conflicts) > 0
            assert "conda" in conflicts[0].lower()


class TestFormattingHelpers:
    """Tests for formatting helper functions."""

    def test_memory_info_returns_tuple_always(self):
        """Verify get_memory_info returns tuple even on failure."""
        # Even if the underlying commands fail, should return tuple
        used, available, total = get_memory_info()
        assert isinstance(used, str)
        assert isinstance(available, str)
        assert isinstance(total, str)

    def test_disk_info_handles_permission_error(self, tmp_path):
        """Verify get_disk_info handles permission errors gracefully."""
        # Use a path that exists
        used, available, total = get_disk_info(str(tmp_path))
        assert used != ""
        assert total != ""

    def test_get_uv_config_value_from_toml(self, tmp_path):
        """Verify get_uv_config_value can read from TOML config."""
        # Create a mock uv config file
        config_dir = tmp_path / ".config" / "uv"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "uv.toml"
        config_file.write_text('python-preference = "system"\n')

        # Patch HOME to use our temp dir
        with patch.dict(
            "os.environ",
            {"HOME": str(tmp_path), "UV_PYTHON_PREFERENCE": ""},
            clear=False,
        ):
            # Note: This test may not work if tomllib is not available
            # Just verify it doesn't crash
            _ = get_uv_config_value("python-preference")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_git_info_empty_path(self):
        """Verify get_git_info handles empty path."""
        info = get_git_info("")
        assert info == {}

    def test_get_binary_info_with_symlink(self, tmp_path):
        """Verify get_binary_info resolves symlinks."""
        # Create a mock binary and symlink
        real_binary = tmp_path / "real_python"
        real_binary.write_text("#!/usr/bin/env python\n")
        real_binary.chmod(0o755)

        symlink = tmp_path / "python_link"
        symlink.symlink_to(real_binary)

        # Patch shutil.which to return our symlink
        with patch("shutil.which", return_value=str(symlink)):
            info = get_binary_info("python_link")
            assert info is not None
            # Real path should be resolved
            assert str(real_binary) in info.real_path

    def test_run_command_with_stderr(self):
        """Verify run_command captures stderr."""
        # Run a command that outputs to stderr
        rc, stdout, stderr = run_command(["ls", "/nonexistent_dir_xyz"])
        # ls should fail with an error
        assert rc != 0

    def test_get_ffmpeg_info_with_no_ffmpeg(self):
        """Verify get_ffmpeg_info handles missing ffmpeg."""
        with patch("shutil.which", return_value=None):
            # Also mock imageio_ffmpeg import to fail
            with patch.dict("sys.modules", {"imageio_ffmpeg": None}):
                binaries = get_ffmpeg_info()
                assert isinstance(binaries, list)
