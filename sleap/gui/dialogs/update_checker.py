"""
Dialog for checking package updates from GitHub releases.
"""

import importlib.metadata
from typing import Dict, Optional

import requests
from packaging.version import parse as parse_version
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import Qt
from qtpy.QtGui import QDesktopServices
from qtpy.QtCore import QUrl


# Package configuration: name -> (display_name, github_repo, default_branch)
PACKAGES = {
    "sleap": ("sleap", "talmolab/sleap", "develop"),
    "sleap-io": ("sleap-io", "talmolab/sleap-io", "main"),
    "sleap-nn": ("sleap-nn", "talmolab/sleap-nn", "main"),
}

# Column indices
COL_PACKAGE = 0
COL_INSTALLED = 1
COL_STABLE = 2
COL_LATEST = 3
COL_DEVELOPMENT = 4
COL_STATUS = 5

# Module-level cache to persist data between dialog opens
_cache: Dict[str, dict] = {}


class UpdateFetchWorker(QtCore.QThread):
    """Worker thread to fetch version and branch info from GitHub API."""

    versionFetched = QtCore.Signal(
        str, str, str, str, str, str
    )  # pkg, stable_ver, stable_url, latest_ver, latest_url, error
    branchFetched = QtCore.Signal(
        str, int, str, str, str
    )  # pkg, ahead_count, latest_date, repo_url, error

    def __init__(self, packages: Dict[str, tuple], parent=None):
        super().__init__(parent)
        self.packages = packages

    def run(self):
        """Fetch versions and branch info for all packages."""
        latest_tags = {}  # Collect tags for branch comparison

        # Phase 1: Fetch release versions
        for pkg_name, (display_name, repo, _default_branch) in self.packages.items():
            try:
                url = f"https://api.github.com/repos/{repo}/releases?per_page=30"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                releases = response.json()

                if not releases:
                    self.versionFetched.emit(
                        pkg_name, "", "", "", "", "No releases found"
                    )
                    continue

                # Find stable (highest non-prerelease) and latest (highest overall)
                stable_version = None
                stable_url = ""
                latest_version = None
                latest_url = ""

                for release in releases:
                    tag = release.get("tag_name", "").lstrip("v")
                    is_prerelease = release.get("prerelease", False)
                    html_url = release.get("html_url", "")

                    try:
                        ver = parse_version(tag)
                    except Exception:
                        continue

                    # Track latest (highest version overall)
                    if latest_version is None or ver > latest_version:
                        latest_version = ver
                        latest_url = html_url

                    # Track stable (highest non-prerelease)
                    if not is_prerelease:
                        if stable_version is None or ver > stable_version:
                            stable_version = ver
                            stable_url = html_url

                stable_str = str(stable_version) if stable_version else ""
                latest_str = str(latest_version) if latest_version else ""

                # Store tag for branch comparison
                if stable_version:
                    latest_tags[pkg_name] = f"v{stable_version}"
                elif latest_version:
                    latest_tags[pkg_name] = f"v{latest_version}"

                self.versionFetched.emit(
                    pkg_name, stable_str, stable_url, latest_str, latest_url, ""
                )

            except requests.exceptions.RequestException as e:
                self.versionFetched.emit(pkg_name, "", "", "", "", str(e))
            except Exception as e:
                self.versionFetched.emit(pkg_name, "", "", "", "", str(e))

        # Phase 2: Fetch branch comparisons
        for pkg_name, (display_name, repo, default_branch) in self.packages.items():
            latest_tag = latest_tags.get(pkg_name)
            repo_url = f"https://github.com/{repo}"

            if not latest_tag:
                self.branchFetched.emit(pkg_name, 0, "", repo_url, "No release tag")
                continue

            try:
                url = f"https://api.github.com/repos/{repo}/compare/{latest_tag}...{default_branch}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                ahead_by = data.get("ahead_by", 0)

                # Get the date of the most recent commit
                commits = data.get("commits", [])
                latest_date = ""
                if commits:
                    last_commit = commits[-1]
                    commit_info = last_commit.get("commit", {})
                    committer = commit_info.get("committer", {})
                    date_str = committer.get("date", "")
                    if date_str:
                        latest_date = date_str[:10]

                self.branchFetched.emit(pkg_name, ahead_by, latest_date, repo_url, "")

            except requests.exceptions.RequestException as e:
                self.branchFetched.emit(pkg_name, 0, "", repo_url, str(e))
            except Exception as e:
                self.branchFetched.emit(pkg_name, 0, "", repo_url, str(e))


class UpdateCheckerDialog(QtWidgets.QDialog):
    """Dialog for checking if SLEAP packages have available updates."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Check for Updates")
        self.setMinimumWidth(370)
        self.setMaximumHeight(200)

        self._worker: Optional[UpdateFetchWorker] = None

        self._setup_ui()
        self._populate_installed_versions()

        # Use cached data if available, otherwise fetch
        if _cache:
            self._populate_from_cache()
        else:
            self._fetch_latest_versions()

    def _setup_ui(self):
        """Create the dialog UI components."""
        layout = QtWidgets.QVBoxLayout()

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Package", "Installed", "Stable", "Latest", "Development", "Status"]
        )
        self.table.setRowCount(len(PACKAGES))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(40)
        # Start with ResizeToContents; balanced after data loads
        for col in range(self.table.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)

        # Connect double-click
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        layout.addWidget(self.table)

        # Tip label
        self.tip_label = QtWidgets.QLabel(
            "<b>Tip:</b> Double-click to see more details about a release."
        )
        self.tip_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.tip_label)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._fetch_latest_versions)
        button_layout.addWidget(self.refresh_button)

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _populate_installed_versions(self):
        """Populate the table with installed package versions."""
        for row, (pkg_name, (display_name, _repo, _branch)) in enumerate(
            PACKAGES.items()
        ):
            # Package name
            name_item = QtWidgets.QTableWidgetItem(display_name)
            self.table.setItem(row, COL_PACKAGE, name_item)

            # Installed version
            try:
                installed_version = importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                installed_version = "Not installed"

            installed_item = QtWidgets.QTableWidgetItem(installed_version)
            self.table.setItem(row, COL_INSTALLED, installed_item)

            # Initialize loading states for other columns
            for col in [COL_STABLE, COL_LATEST, COL_DEVELOPMENT]:
                item = QtWidgets.QTableWidgetItem("Loading...")
                item.setForeground(QtGui.QColor("gray"))
                self.table.setItem(row, col, item)

            # Empty status column
            status_item = QtWidgets.QTableWidgetItem("")
            self.table.setItem(row, COL_STATUS, status_item)

    def _populate_from_cache(self):
        """Populate table from cached data."""
        for pkg_name, data in _cache.items():
            if "version" in data:
                self._on_version_fetched(pkg_name, **data["version"])
            if "branch" in data:
                self._on_branch_fetched(pkg_name, **data["branch"])

    def _fetch_latest_versions(self):
        """Start fetching latest versions from GitHub."""
        self.refresh_button.setEnabled(False)

        # Reset all loading columns
        for row in range(self.table.rowCount()):
            for col in [COL_STABLE, COL_LATEST, COL_DEVELOPMENT]:
                item = self.table.item(row, col)
                item.setText("Loading...")
                item.setForeground(QtGui.QColor("gray"))
                item.setData(Qt.UserRole, None)  # Clear URL data
            self.table.item(row, COL_STATUS).setText("")

        # Wait for any existing worker to finish
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()

        # Start fetch worker (parent=self ensures proper Qt lifecycle)
        self._worker = UpdateFetchWorker(PACKAGES, parent=self)
        self._worker.versionFetched.connect(self._on_version_fetched)
        self._worker.branchFetched.connect(self._on_branch_fetched)
        self._worker.finished.connect(self._on_all_fetches_finished)
        self._worker.start()

    def _on_version_fetched(
        self,
        pkg_name: str,
        stable_version: str,
        stable_url: str,
        latest_version: str,
        latest_url: str,
        error: str,
    ):
        """Handle a fetched version result."""
        # Cache the result
        if pkg_name not in _cache:
            _cache[pkg_name] = {}
        _cache[pkg_name]["version"] = {
            "stable_version": stable_version,
            "stable_url": stable_url,
            "latest_version": latest_version,
            "latest_url": latest_url,
            "error": error,
        }

        row = list(PACKAGES.keys()).index(pkg_name)

        stable_item = self.table.item(row, COL_STABLE)
        latest_item = self.table.item(row, COL_LATEST)
        status_item = self.table.item(row, COL_STATUS)

        if error:
            stable_item.setText("Error")
            stable_item.setForeground(QtGui.QColor("red"))
            stable_item.setToolTip(error)
            latest_item.setText("Error")
            latest_item.setForeground(QtGui.QColor("red"))
            latest_item.setToolTip(error)
            return

        # Stable column
        if stable_version:
            stable_item.setText(stable_version)
            stable_item.setForeground(self.palette().text().color())
            stable_item.setToolTip("Click to view release")
            stable_item.setData(Qt.UserRole, stable_url)
        else:
            stable_item.setText("N/A")
            stable_item.setForeground(QtGui.QColor("gray"))
            stable_item.setToolTip("No stable release found")

        # Latest column
        if latest_version:
            latest_item.setText(latest_version)
            latest_item.setForeground(self.palette().text().color())
            latest_item.setToolTip("Click to view release")
            latest_item.setData(Qt.UserRole, latest_url)
        else:
            latest_item.setText("N/A")
            latest_item.setForeground(QtGui.QColor("gray"))
            latest_item.setToolTip("No release found")

        # Status column - compare Installed vs Stable
        installed_item = self.table.item(row, COL_INSTALLED)
        installed_text = installed_item.text()

        if installed_text == "Not installed":
            status_item.setText("—")
            status_item.setToolTip("Package not installed")
        elif stable_version:
            try:
                installed_v = parse_version(installed_text)
                stable_v = parse_version(stable_version)

                if installed_v < stable_v:
                    status_item.setText("\u2b06\ufe0f")  # ⬆️
                    status_item.setToolTip("Upgrade available")
                elif installed_v == stable_v:
                    status_item.setText("\u2705")  # ✅
                    status_item.setToolTip("Up to date")
                else:
                    status_item.setText("\U0001f52a")  # 🔪
                    status_item.setToolTip("Bleeding edge")
            except Exception:
                status_item.setText("?")
                status_item.setToolTip("Could not compare versions")
        else:
            status_item.setText("?")
            status_item.setToolTip("No stable version to compare")

    def _on_branch_fetched(
        self,
        pkg_name: str,
        ahead_count: int,
        latest_date: str,
        repo_url: str,
        error: str,
    ):
        """Handle a fetched branch comparison result."""
        # Cache the result
        if pkg_name not in _cache:
            _cache[pkg_name] = {}
        _cache[pkg_name]["branch"] = {
            "ahead_count": ahead_count,
            "latest_date": latest_date,
            "repo_url": repo_url,
            "error": error,
        }

        row = list(PACKAGES.keys()).index(pkg_name)
        dev_item = self.table.item(row, COL_DEVELOPMENT)

        if error:
            dev_item.setText("N/A")
            dev_item.setForeground(QtGui.QColor("gray"))
            dev_item.setToolTip(error)
            dev_item.setData(Qt.UserRole, repo_url)
            return

        if ahead_count > 0:
            if latest_date:
                dev_item.setText(f"+{ahead_count} ({latest_date})")
            else:
                dev_item.setText(f"+{ahead_count}")
            dev_item.setForeground(self.palette().text().color())
            dev_item.setToolTip(f"{ahead_count} commits ahead of latest release")
        else:
            dev_item.setText("—")
            dev_item.setForeground(QtGui.QColor("gray"))
            dev_item.setToolTip("No commits ahead of latest release")

        dev_item.setData(Qt.UserRole, repo_url)

    def _on_all_fetches_finished(self):
        """Handle completion of all fetches (versions + branches)."""
        self.refresh_button.setEnabled(True)

    def _on_cell_double_clicked(self, row: int, col: int):
        """Handle double-click on a cell to open URL."""
        # Only handle Stable, Latest, Development columns
        if col not in [COL_STABLE, COL_LATEST, COL_DEVELOPMENT]:
            return

        item = self.table.item(row, col)
        if item is None:
            return

        url = item.data(Qt.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def closeEvent(self, event):
        """Clean up worker thread on close."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        super().closeEvent(event)
