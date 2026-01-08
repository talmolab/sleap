"""Tests for the update checker dialog."""

import pytest
from unittest.mock import patch, MagicMock
from packaging.version import parse as parse_version

from sleap.gui.dialogs.update_checker import (
    PACKAGES,
    VersionFetchWorker,
    UpdateCheckerDialog,
)


class TestPackagesConfig:
    """Tests for the PACKAGES configuration."""

    def test_packages_contains_expected_packages(self):
        """Test that all expected packages are configured."""
        expected = ["sleap", "sleap-io", "sleap-nn"]
        for pkg in expected:
            assert pkg in PACKAGES

    def test_packages_have_valid_structure(self):
        """Test that each package has display_name and repo."""
        for pkg_name, (display_name, repo) in PACKAGES.items():
            assert isinstance(display_name, str)
            assert len(display_name) > 0
            assert isinstance(repo, str)
            assert "/" in repo  # Should be "owner/repo" format


class TestVersionComparison:
    """Tests for version comparison logic used in the dialog."""

    @pytest.mark.parametrize(
        "installed,latest,expect_update",
        [
            ("1.0.0", "1.0.1", True),  # Patch update available
            ("1.0.0", "1.1.0", True),  # Minor update available
            ("1.0.0", "2.0.0", True),  # Major update available
            ("1.0.1", "1.0.0", False),  # Installed is newer
            ("1.0.0", "1.0.0", False),  # Same version
            ("1.4.0a1", "1.3.4", False),  # Pre-release > stable numerically
            ("1.3.4", "1.4.0a1", True),  # Stable < pre-release
            ("0.5.7", "0.5.8", True),  # Typical sleap-io update
            ("0.0.6", "0.0.5", False),  # Dev version ahead of release
        ],
    )
    def test_version_comparison(self, installed, latest, expect_update):
        """Test that version comparison works correctly."""
        installed_v = parse_version(installed)
        latest_v = parse_version(latest)
        has_update = latest_v > installed_v
        assert has_update == expect_update


class TestVersionFetchWorker:
    """Tests for the VersionFetchWorker thread."""

    def test_worker_initialization(self):
        """Test worker initializes with packages."""
        worker = VersionFetchWorker(PACKAGES)
        assert worker.packages == PACKAGES

    @patch("sleap.gui.dialogs.update_checker.requests.get")
    def test_worker_fetches_versions(self, mock_get, qtbot):
        """Test worker fetches versions from GitHub API."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "v1.5.0"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        worker = VersionFetchWorker({"sleap": ("sleap", "talmolab/sleap")})

        # Collect emitted signals
        results = []
        worker.versionFetched.connect(lambda *args: results.append(args))

        # Run worker
        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

        assert len(results) == 1
        pkg_name, version, error = results[0]
        assert pkg_name == "sleap"
        assert version == "1.5.0"  # "v" prefix stripped
        assert error == ""

    @patch("sleap.gui.dialogs.update_checker.requests.get")
    def test_worker_handles_api_error(self, mock_get, qtbot):
        """Test worker handles API errors gracefully."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        worker = VersionFetchWorker({"sleap": ("sleap", "talmolab/sleap")})

        results = []
        worker.versionFetched.connect(lambda *args: results.append(args))

        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

        assert len(results) == 1
        pkg_name, version, error = results[0]
        assert pkg_name == "sleap"
        assert version == ""
        assert "Network error" in error


class TestUpdateCheckerDialog:
    """Tests for the UpdateCheckerDialog UI."""

    def test_dialog_creation(self, qtbot):
        """Test dialog can be created."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            assert dialog.windowTitle() == "Check for Updates"
            assert dialog.minimumWidth() >= 500
            assert dialog.minimumHeight() >= 250

    def test_dialog_table_structure(self, qtbot):
        """Test dialog table has correct structure."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            assert dialog.table.columnCount() == 4
            assert dialog.table.rowCount() == len(PACKAGES)

            # Check headers
            headers = [
                dialog.table.horizontalHeaderItem(i).text() for i in range(4)
            ]
            assert headers == ["Package", "Installed", "Latest", ""]

    def test_dialog_has_buttons(self, qtbot):
        """Test dialog has refresh and close buttons."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            assert dialog.refresh_button is not None
            assert dialog.close_button is not None
            assert dialog.refresh_button.text() == "Refresh"
            assert dialog.close_button.text() == "Close"

    def test_dialog_has_status_label(self, qtbot):
        """Test dialog has status label."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            assert dialog.status_label is not None

    def test_dialog_populates_installed_versions(self, qtbot):
        """Test dialog populates installed versions for packages."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            # Check each row has package name and installed version
            for row in range(dialog.table.rowCount()):
                name_item = dialog.table.item(row, 0)
                installed_item = dialog.table.item(row, 1)

                assert name_item is not None
                assert name_item.text() in [
                    display for display, _ in PACKAGES.values()
                ]

                assert installed_item is not None
                # Should be a version string or "Not installed"
                assert len(installed_item.text()) > 0

    @patch("sleap.gui.dialogs.update_checker.importlib.metadata.version")
    def test_dialog_handles_missing_package(self, mock_version, qtbot):
        """Test dialog handles packages that aren't installed."""
        import importlib.metadata

        mock_version.side_effect = importlib.metadata.PackageNotFoundError()

        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            # At least one row should show "Not installed"
            found_not_installed = False
            for row in range(dialog.table.rowCount()):
                installed_item = dialog.table.item(row, 1)
                if installed_item.text() == "Not installed":
                    found_not_installed = True
                    break
            assert found_not_installed

    def test_on_version_fetched_shows_update_available(self, qtbot):
        """Test that update indicator shows when update is available."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            with patch(
                "sleap.gui.dialogs.update_checker.importlib.metadata.version",
                return_value="1.0.0",
            ):
                dialog = UpdateCheckerDialog()
                qtbot.addWidget(dialog)

                # Simulate receiving a newer version
                dialog._on_version_fetched("sleap", "2.0.0", "")

                # Find the sleap row
                row = list(PACKAGES.keys()).index("sleap")
                status_item = dialog.table.item(row, 3)

                assert status_item.text() == "Update"

    def test_on_version_fetched_shows_up_to_date(self, qtbot):
        """Test that up-to-date indicator shows when versions match."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            with patch(
                "sleap.gui.dialogs.update_checker.importlib.metadata.version",
                return_value="1.0.0",
            ):
                dialog = UpdateCheckerDialog()
                qtbot.addWidget(dialog)

                # Simulate receiving same version
                dialog._on_version_fetched("sleap", "1.0.0", "")

                row = list(PACKAGES.keys()).index("sleap")
                status_item = dialog.table.item(row, 3)

                assert status_item.text() == "✓"

    def test_on_fetch_finished_updates_status(self, qtbot):
        """Test that status label updates after fetch completes."""
        with patch.object(UpdateCheckerDialog, "_fetch_latest_versions"):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            # Simulate no updates
            dialog._on_fetch_finished()

            assert "up to date" in dialog.status_label.text().lower()

    def test_refresh_button_triggers_fetch(self, qtbot):
        """Test that refresh button triggers version fetch."""
        with patch.object(
            UpdateCheckerDialog, "_fetch_latest_versions"
        ) as mock_fetch:
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            # Clear the call from __init__
            mock_fetch.reset_mock()

            # Click refresh
            dialog.refresh_button.click()

            mock_fetch.assert_called_once()
