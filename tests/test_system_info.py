"""Tests for the SLEAP system_info module."""

from io import StringIO

from rich.console import Console

from sleap.system_info import (
    _print_package_table,
    get_package_info,
    get_all_package_info,
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
