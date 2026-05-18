#!/usr/bin/env python
"""Qt widget screenshot capture utility.

Captures screenshots of Qt widgets without displaying them on screen.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt

# Output directory
OUTPUT_DIR = Path("scratch/.qt-screenshots")


def init_qt() -> QApplication:
    """Initialize Qt application (required once per process)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def timestamp() -> str:
    """Generate timestamp for filename."""
    return datetime.now().strftime("%Y-%m-%d.%H-%M-%S")


def capture_widget(
    widget: QWidget,
    description: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Capture a screenshot of a widget without displaying it on screen.

    Args:
        widget: The QWidget to capture
        description: Short description for filename (use underscores, no spaces)
        output_dir: Override output directory (default: scratch/.qt-screenshots)

    Returns:
        Path to saved screenshot
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Configure for invisible rendering
    widget.setAttribute(Qt.WA_DontShowOnScreen, True)
    widget.show()
    QApplication.processEvents()

    # Capture
    pixmap = widget.grab()
    if pixmap.isNull():
        raise RuntimeError("Failed to capture widget - pixmap is null")

    # Save with timestamp
    desc_clean = description.replace(" ", "_").replace("/", "-")
    filename = f"{timestamp()}_{desc_clean}.png"
    filepath = out / filename

    if not pixmap.save(str(filepath)):
        raise RuntimeError(f"Failed to save screenshot to {filepath}")

    # Don't close - caller may want to interact further
    widget.hide()

    return filepath


def capture_and_click(
    widget: QWidget,
    x: int,
    y: int,
    description: str,
) -> tuple[Path, Optional[QWidget]]:
    """
    Click at coordinates and capture the result.

    Args:
        widget: Parent widget
        x, y: Coordinates to click (in widget coordinate space)
        description: Description for filename

    Returns:
        Tuple of (screenshot path, clicked widget or None)
    """
    widget.setAttribute(Qt.WA_DontShowOnScreen, True)
    widget.show()
    QApplication.processEvents()

    # Find and click widget at coordinates
    target = widget.childAt(x, y)
    if target is not None:
        if hasattr(target, "click"):
            target.click()
        elif hasattr(target, "toggle"):
            target.toggle()
        QApplication.processEvents()

    # Capture result
    path = capture_widget(widget, description)
    return path, target


# Self-test when run directly
if __name__ == "__main__":
    from PySide6.QtWidgets import QPushButton, QVBoxLayout, QLabel

    app = init_qt()

    # Create test widget
    widget = QWidget()
    widget.setWindowTitle("Qt Capture Test")
    widget.setFixedSize(300, 150)

    layout = QVBoxLayout(widget)
    layout.addWidget(QLabel("Qt Capture Test Widget"))
    btn = QPushButton("Test Button")
    layout.addWidget(btn)

    # Capture it
    path = capture_widget(widget, "self_test")
    print(f"Captured: {path}")

    widget.close()
    print("Self-test complete!")
