"""Tests for the update checker dialog."""

import pytest
from unittest.mock import patch, MagicMock
from packaging.version import parse as parse_version

from sleap.gui.dialogs import update_checker
from sleap.gui.dialogs.update_checker import (
    PACKAGES,
    COL_PACKAGE,
    COL_INSTALLED,
    COL_STABLE,
    COL_LATEST,
    COL_DEVELOPMENT,
    COL_STATUS,
    UpdateFetchWorker,
    UpdateCheckerDialog,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the module-level cache before each test for isolation."""
    update_checker._cache.clear()
    yield
    update_checker._cache.clear()


class TestPackagesConfig:
    """Tests for the PACKAGES configuration."""

    def test_packages_contains_expected_packages(self):
        """Test that all expected packages are configured."""
        expected = ["sleap", "sleap-io", "sleap-nn"]
        for pkg in expected:
            assert pkg in PACKAGES

    def test_packages_have_valid_structure(self):
        """Test that each package has display_name, repo, and default_branch."""
        for pkg_name, (display_name, repo, default_branch) in PACKAGES.items():
            assert isinstance(display_name, str)
            assert len(display_name) > 0
            assert isinstance(repo, str)
            assert "/" in repo  # Should be "owner/repo" format
            assert isinstance(default_branch, str)
            assert default_branch in ["main", "develop"]


class TestVersionComparison:
    """Tests for version comparison logic used in the dialog."""

    @pytest.mark.parametrize(
        "installed,stable,expect_upgrade,expect_bleeding_edge",
        [
            ("1.0.0", "1.0.1", True, False),  # Patch update available
            ("1.0.0", "1.1.0", True, False),  # Minor update available
            ("1.0.0", "2.0.0", True, False),  # Major update available
            ("1.0.1", "1.0.0", False, True),  # Installed is newer (bleeding edge)
            ("1.0.0", "1.0.0", False, False),  # Same version (up to date)
            ("1.4.0a1", "1.3.4", False, True),  # Pre-release > stable
            ("1.3.4", "1.4.0a1", True, False),  # Stable < pre-release
            ("0.5.7", "0.5.8", True, False),  # Typical sleap-io update
            ("0.0.6", "0.0.5", False, True),  # Dev version ahead
        ],
    )
    def test_version_comparison(
        self, installed, stable, expect_upgrade, expect_bleeding_edge
    ):
        """Test that version comparison works correctly."""
        installed_v = parse_version(installed)
        stable_v = parse_version(stable)

        has_upgrade = installed_v < stable_v
        is_bleeding_edge = installed_v > stable_v

        assert has_upgrade == expect_upgrade
        assert is_bleeding_edge == expect_bleeding_edge


class TestUpdateFetchWorker:
    """Tests for the UpdateFetchWorker thread."""

    def test_worker_initialization(self):
        """Test worker initializes with packages."""
        worker = UpdateFetchWorker(PACKAGES)
        assert worker.packages == PACKAGES

    @patch("sleap.gui.dialogs.update_checker.requests.get")
    def test_worker_fetches_versions_and_branches(self, mock_get, qtbot):
        """Test worker fetches both version and branch info."""

        # Mock responses for releases and compare APIs
        def mock_response_factory(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            if "releases" in url:
                mock_resp.json.return_value = [
                    {
                        "tag_name": "v1.5.0",
                        "prerelease": False,
                        "html_url": "https://github.com/repo/releases/v1.5.0",
                    },
                    {
                        "tag_name": "v1.6.0a1",
                        "prerelease": True,
                        "html_url": "https://github.com/repo/releases/v1.6.0a1",
                    },
                ]
            elif "compare" in url:
                mock_resp.json.return_value = {
                    "ahead_by": 42,
                    "commits": [
                        {
                            "sha": "abc",
                            "commit": {"committer": {"date": "2026-01-08T12:00:00Z"}},
                        }
                    ],
                }
            return mock_resp

        mock_get.side_effect = mock_response_factory

        worker = UpdateFetchWorker({"sleap": ("sleap", "talmolab/sleap", "develop")})

        version_results = []
        branch_results = []
        worker.versionFetched.connect(lambda *args: version_results.append(args))
        worker.branchFetched.connect(lambda *args: branch_results.append(args))

        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

        # Check version results
        assert len(version_results) == 1
        pkg_name, stable_ver, stable_url, latest_ver, latest_url, error = (
            version_results[0]
        )
        assert pkg_name == "sleap"
        assert stable_ver == "1.5.0"
        assert latest_ver == "1.6.0a1"
        assert error == ""

        # Check branch results
        assert len(branch_results) == 1
        pkg_name, ahead_count, latest_date, repo_url, error = branch_results[0]
        assert pkg_name == "sleap"
        assert ahead_count == 42
        assert latest_date == "2026-01-08"

    @patch("sleap.gui.dialogs.update_checker.requests.get")
    def test_worker_handles_api_error(self, mock_get, qtbot):
        """Test worker handles API errors gracefully."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        worker = UpdateFetchWorker({"sleap": ("sleap", "talmolab/sleap", "develop")})

        version_results = []
        branch_results = []
        worker.versionFetched.connect(lambda *args: version_results.append(args))
        worker.branchFetched.connect(lambda *args: branch_results.append(args))

        with qtbot.waitSignal(worker.finished, timeout=5000):
            worker.start()

        assert len(version_results) == 1
        assert "Network error" in version_results[0][5]  # error field

        # Branch fetch also fails since no tag available
        assert len(branch_results) == 1
        assert "No release tag" in branch_results[0][4]  # error field


@pytest.fixture
def mock_dialog_fetch():
    """Fixture to prevent network calls during dialog tests.

    Pre-populates the cache so the dialog uses cached data instead of
    making network requests. This avoids patching Qt methods which can
    cause issues with Qt's metaclass system.
    """
    # Pre-populate cache with mock data
    update_checker._cache = {
        "sleap": {
            "version": {
                "stable_version": "1.0.0",
                "stable_url": "https://example.com/stable",
                "latest_version": "1.1.0",
                "latest_url": "https://example.com/latest",
                "error": "",
            },
            "branch": {
                "ahead_count": 5,
                "latest_date": "2026-01-01",
                "repo_url": "https://github.com/talmolab/sleap",
                "error": "",
            },
        },
        "sleap-io": {
            "version": {
                "stable_version": "0.5.0",
                "stable_url": "https://example.com/stable",
                "latest_version": "0.5.0",
                "latest_url": "https://example.com/latest",
                "error": "",
            },
            "branch": {
                "ahead_count": 0,
                "latest_date": "",
                "repo_url": "https://github.com/talmolab/sleap-io",
                "error": "",
            },
        },
        "sleap-nn": {
            "version": {
                "stable_version": "0.1.0",
                "stable_url": "https://example.com/stable",
                "latest_version": "0.1.0",
                "latest_url": "https://example.com/latest",
                "error": "",
            },
            "branch": {
                "ahead_count": 0,
                "latest_date": "",
                "repo_url": "https://github.com/talmolab/sleap-nn",
                "error": "",
            },
        },
    }
    yield
    update_checker._cache.clear()


class TestUpdateCheckerDialog:
    """Tests for the UpdateCheckerDialog UI."""

    def test_dialog_creation(self, qtbot, mock_dialog_fetch):
        """Test dialog can be created."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        assert dialog.windowTitle() == "Check for Updates"

    def test_dialog_table_structure(self, qtbot, mock_dialog_fetch):
        """Test dialog table has correct 6-column structure."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        assert dialog.table.columnCount() == 6
        assert dialog.table.rowCount() == len(PACKAGES)

        # Check headers
        headers = [dialog.table.horizontalHeaderItem(i).text() for i in range(6)]
        assert headers == [
            "Package",
            "Installed",
            "Stable",
            "Latest",
            "Development",
            "Status",
        ]

    def test_dialog_has_buttons(self, qtbot, mock_dialog_fetch):
        """Test dialog has refresh and close buttons."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        assert dialog.refresh_button is not None
        assert dialog.close_button is not None
        assert dialog.refresh_button.text() == "Refresh"
        assert dialog.close_button.text() == "Close"

    def test_dialog_has_tip_label(self, qtbot, mock_dialog_fetch):
        """Test dialog has tip label for double-click."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        assert dialog.tip_label is not None
        assert "Double-click" in dialog.tip_label.text()

    def test_dialog_populates_installed_versions(self, qtbot, mock_dialog_fetch):
        """Test dialog populates installed versions for packages."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        for row in range(dialog.table.rowCount()):
            name_item = dialog.table.item(row, COL_PACKAGE)
            installed_item = dialog.table.item(row, COL_INSTALLED)

            assert name_item is not None
            assert name_item.text() in [display for display, _, _ in PACKAGES.values()]

            assert installed_item is not None
            assert len(installed_item.text()) > 0

    def test_dialog_handles_missing_package(self, qtbot, mock_dialog_fetch):
        """Test dialog handles packages that aren't installed."""
        import importlib.metadata

        with patch(
            "sleap.gui.dialogs.update_checker.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError(),
        ):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            found_not_installed = False
            for row in range(dialog.table.rowCount()):
                installed_item = dialog.table.item(row, COL_INSTALLED)
                if installed_item.text() == "Not installed":
                    found_not_installed = True
                    break
            assert found_not_installed

    def test_on_version_fetched_shows_upgrade_available(self, qtbot, mock_dialog_fetch):
        """Test that upgrade emoji shows when update is available."""
        with patch(
            "sleap.gui.dialogs.update_checker.importlib.metadata.version",
            return_value="1.0.0",
        ):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            # Simulate receiving newer stable version
            dialog._on_version_fetched(
                "sleap", "2.0.0", "https://url", "2.0.0", "https://url", ""
            )

            row = list(PACKAGES.keys()).index("sleap")
            status_item = dialog.table.item(row, COL_STATUS)

            assert status_item.text() == "\u2b06\ufe0f"  # ⬆️

    def test_on_version_fetched_shows_up_to_date(self, qtbot, mock_dialog_fetch):
        """Test that up-to-date emoji shows when versions match."""
        with patch(
            "sleap.gui.dialogs.update_checker.importlib.metadata.version",
            return_value="1.0.0",
        ):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            dialog._on_version_fetched(
                "sleap", "1.0.0", "https://url", "1.0.0", "https://url", ""
            )

            row = list(PACKAGES.keys()).index("sleap")
            status_item = dialog.table.item(row, COL_STATUS)

            assert status_item.text() == "\u2705"  # ✅

    def test_on_version_fetched_shows_bleeding_edge(self, qtbot, mock_dialog_fetch):
        """Test that bleeding edge emoji shows when installed > stable."""
        with patch(
            "sleap.gui.dialogs.update_checker.importlib.metadata.version",
            return_value="2.0.0",
        ):
            dialog = UpdateCheckerDialog()
            qtbot.addWidget(dialog)

            dialog._on_version_fetched(
                "sleap", "1.0.0", "https://url", "1.0.0", "https://url", ""
            )

            row = list(PACKAGES.keys()).index("sleap")
            status_item = dialog.table.item(row, COL_STATUS)

            assert status_item.text() == "\U0001f52a"  # 🔪

    def test_on_branch_fetched_shows_commits_ahead(self, qtbot, mock_dialog_fetch):
        """Test that development column shows commits ahead."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        dialog._on_branch_fetched(
            "sleap", 42, "2026-01-08", "https://github.com/repo", ""
        )

        row = list(PACKAGES.keys()).index("sleap")
        dev_item = dialog.table.item(row, COL_DEVELOPMENT)

        assert dev_item.text() == "+42 (2026-01-08)"

    def test_on_branch_fetched_shows_dash_when_no_commits(
        self, qtbot, mock_dialog_fetch
    ):
        """Test that development column shows dash when no commits ahead."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        dialog._on_branch_fetched("sleap", 0, "", "https://github.com/repo", "")

        row = list(PACKAGES.keys()).index("sleap")
        dev_item = dialog.table.item(row, COL_DEVELOPMENT)

        assert dev_item.text() == "—"

    def test_on_all_fetches_finished_enables_refresh(self, qtbot, mock_dialog_fetch):
        """Test that refresh button is re-enabled after all fetches complete."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        dialog.refresh_button.setEnabled(False)
        dialog._on_all_fetches_finished()

        assert dialog.refresh_button.isEnabled()

    def test_refresh_button_triggers_fetch(self, qtbot, mock_dialog_fetch):
        """Test that refresh button triggers version fetch."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        with patch.object(dialog, "_fetch_latest_versions") as mock_fetch:
            dialog.refresh_button.click()

            mock_fetch.assert_called_once()

    def test_double_click_stores_url_data(self, qtbot, mock_dialog_fetch):
        """Test that version fetched stores URL in cell data."""
        dialog = UpdateCheckerDialog()
        qtbot.addWidget(dialog)

        dialog._on_version_fetched(
            "sleap",
            "1.5.0",
            "https://stable-url",
            "1.6.0a1",
            "https://latest-url",
            "",
        )

        row = list(PACKAGES.keys()).index("sleap")

        from qtpy.QtCore import Qt

        stable_item = dialog.table.item(row, COL_STABLE)
        latest_item = dialog.table.item(row, COL_LATEST)

        assert stable_item.data(Qt.UserRole) == "https://stable-url"
        assert latest_item.data(Qt.UserRole) == "https://latest-url"
