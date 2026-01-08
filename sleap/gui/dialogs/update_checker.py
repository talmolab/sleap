"""
Dialog for checking package updates from GitHub releases.
"""

import importlib.metadata
from typing import Dict, Optional

import requests
from packaging.version import parse as parse_version
from qtpy import QtCore, QtWidgets, QtGui


# Package configuration: name -> (display_name, github_repo)
PACKAGES = {
    "sleap": ("sleap", "talmolab/sleap"),
    "sleap-io": ("sleap-io", "talmolab/sleap-io"),
    "sleap-nn": ("sleap-nn", "talmolab/sleap-nn"),
}


class VersionFetchWorker(QtCore.QThread):
    """Worker thread to fetch latest versions from GitHub API."""

    versionFetched = QtCore.Signal(str, str, str)
    finished = QtCore.Signal()

    def __init__(self, packages: Dict[str, tuple]):
        super().__init__()
        self.packages = packages

    def run(self):
        """Fetch latest versions for all packages."""
        for pkg_name, (display_name, repo) in self.packages.items():
            try:
                url = f"https://api.github.com/repos/{repo}/releases/latest"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                tag_name = data.get("tag_name", "")
                latest_version = tag_name.lstrip("v")
                self.versionFetched.emit(pkg_name, latest_version, "")
            except requests.exceptions.RequestException as e:
                self.versionFetched.emit(pkg_name, "", str(e))
            except Exception as e:
                self.versionFetched.emit(pkg_name, "", str(e))
        self.finished.emit()


class UpdateCheckerDialog(QtWidgets.QDialog):
    """Dialog for checking if SLEAP packages have available updates."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Check for Updates")
        self.setMinimumWidth(500)
        self.setMinimumHeight(250)

        self._worker: Optional[VersionFetchWorker] = None
        self._latest_versions: Dict[str, str] = {}

        self._setup_ui()
        self._populate_installed_versions()
        self._fetch_latest_versions()

    def _setup_ui(self):
        """Create the dialog UI components."""
        layout = QtWidgets.QVBoxLayout()

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Package", "Installed", "Latest", ""])
        self.table.setRowCount(len(PACKAGES))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)

        layout.addWidget(self.table)

        self.status_label = QtWidgets.QLabel("Checking for updates...")
        layout.addWidget(self.status_label)

        releases_label = QtWidgets.QLabel(
            '<a href="https://docs.sleap.ai/latest/installation/">See the installation docs for upgrade instructions.</a>'
        )
        releases_label.setOpenExternalLinks(True)
        layout.addWidget(releases_label)

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
        for row, (pkg_name, (display_name, _)) in enumerate(PACKAGES.items()):
            name_item = QtWidgets.QTableWidgetItem(display_name)
            self.table.setItem(row, 0, name_item)

            try:
                installed_version = importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                installed_version = "Not installed"

            installed_item = QtWidgets.QTableWidgetItem(installed_version)
            self.table.setItem(row, 1, installed_item)

            latest_item = QtWidgets.QTableWidgetItem("Loading...")
            latest_item.setForeground(QtGui.QColor("gray"))
            self.table.setItem(row, 2, latest_item)

            status_item = QtWidgets.QTableWidgetItem("")
            self.table.setItem(row, 3, status_item)

    def _fetch_latest_versions(self):
        """Start fetching latest versions from GitHub."""
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Checking for updates...")
        self._latest_versions.clear()

        for row in range(self.table.rowCount()):
            latest_item = self.table.item(row, 2)
            latest_item.setText("Loading...")
            latest_item.setForeground(QtGui.QColor("gray"))
            self.table.item(row, 3).setText("")

        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()

        self._worker = VersionFetchWorker(PACKAGES)
        self._worker.versionFetched.connect(self._on_version_fetched)
        self._worker.finished.connect(self._on_fetch_finished)
        self._worker.start()

    def _on_version_fetched(self, pkg_name: str, latest_version: str, error: str):
        """Handle a fetched version result.

        Args:
            pkg_name: The package name.
            latest_version: The latest version string, empty if error.
            error: Error message if fetch failed, empty otherwise.
        """
        row = list(PACKAGES.keys()).index(pkg_name)

        latest_item = self.table.item(row, 2)
        status_item = self.table.item(row, 3)

        if error:
            latest_item.setText("Error")
            latest_item.setForeground(QtGui.QColor("red"))
            latest_item.setToolTip(error)
            return

        self._latest_versions[pkg_name] = latest_version
        latest_item.setText(latest_version)
        latest_item.setForeground(self.palette().text().color())
        latest_item.setToolTip("")

        installed_item = self.table.item(row, 1)
        installed_text = installed_item.text()

        if installed_text == "Not installed":
            status_item.setText("—")
            status_item.setToolTip("Package not installed")
        elif latest_version:
            try:
                installed_v = parse_version(installed_text)
                latest_v = parse_version(latest_version)

                if latest_v > installed_v:
                    status_item.setText("Update")
                    status_item.setForeground(QtGui.QColor("yellow"))
                    status_item.setToolTip("Update available")
                else:
                    status_item.setText("✓")
                    status_item.setForeground(QtGui.QColor("green"))
                    status_item.setToolTip("Up to date")
            except Exception:
                status_item.setText("?")
                status_item.setToolTip("Could not compare versions")

    def _on_fetch_finished(self):
        """Handle completion of all version fetches."""
        self.refresh_button.setEnabled(True)

        updates_available = 0
        for row, pkg_name in enumerate(PACKAGES.keys()):
            status_item = self.table.item(row, 3)
            if status_item and status_item.text() == "Update":
                updates_available += 1

        if updates_available > 0:
            self.status_label.setText(
                f"{updates_available} update(s) available. "
                "See the installation docs for upgrade instructions."
            )
        else:
            self.status_label.setText("All installed packages are up to date!")

    def closeEvent(self, event):
        """Clean up worker thread on close."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        super().closeEvent(event)


