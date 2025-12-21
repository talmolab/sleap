---
name: qt-testing
description: Capture and visually inspect Qt GUI widgets using screenshots. Use when asked to verify GUI rendering, test widget appearance, check layouts, or visually inspect any PySide6/Qt component. Enables Claude to "see" Qt interfaces by capturing offscreen screenshots and analyzing them with vision.
---

# Qt GUI Testing

Capture screenshots of Qt widgets for visual inspection without displaying windows on screen.

## Quick Start

```python
# Capture any widget
from scripts.qt_capture import capture_widget
path = capture_widget(my_widget, "description_here")
# Then read the screenshot with the Read tool
```

## Core Script

Run `scripts/qt_capture.py` or import `capture_widget` from it:

```bash
# Standalone test
uv run --with PySide6 python .claude/skills/qt-testing/scripts/qt_capture.py
```

## Output Location

All screenshots save to: `scratch/.qt-screenshots/`

Naming: `{YYYY-MM-DD.HH-MM-SS}_{description}.png`

## Workflow

1. Create/obtain the widget to test
2. Call `capture_widget(widget, "description")`
3. Read the saved screenshot with the Read tool
4. Analyze with vision to verify correctness

## Interaction Pattern

To interact with widgets (click buttons, etc.):

```python
# Find widget at coordinates (from vision analysis)
target = widget.childAt(x, y)

# Trigger it directly (not mouse events)
if hasattr(target, 'click'):
    target.click()
    QApplication.processEvents()

# Capture result
capture_widget(widget, "after_click")
```

## Example: Test a Dialog

```python
import sys
from PySide6.QtWidgets import QApplication
from sleap.gui.learning.dialog import TrainingEditorDialog

# Add skill scripts to path
sys.path.insert(0, ".claude/skills/qt-testing")
from scripts.qt_capture import capture_widget, init_qt

app = init_qt()
dialog = TrainingEditorDialog()
path = capture_widget(dialog, "training_dialog")
dialog.close()
print(f"Inspect: {path}")
```

## Key Points

- Uses `Qt.WA_DontShowOnScreen` - no window popup
- Renders identically to on-screen display (verified)
- Call `processEvents()` after interactions before capture
- Use `childAt(x, y)` to map vision coordinates to widgets
- Direct method calls (`.click()`) work; simulated mouse events don't
