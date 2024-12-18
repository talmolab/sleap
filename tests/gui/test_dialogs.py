"""Module to test the dialogs of the GUI (contained in sleap/gui/dialogs)."""

import os
import importlib
from pathlib import Path
import re

import pytest
import qtpy.QtWidgets as QtWidgets
from qtpy.QtWidgets import QComboBox

import sleap
from sleap.skeleton import Skeleton
from sleap.io.dataset import Labels
from sleap.gui.commands import OpenSkeleton
from sleap.gui.dialogs.merge import ReplaceSkeletonTableDialog
from sleap.gui.dialogs.export_clip import ExportClipAndLabelsDialog, ExportClipDialog
from sleap.io.videowriter import VideoWriter
import sleap.gui.dialogs.export_clip as export_clip_dialogs
from sleap.gui.app import MainWindow


def test_ReplaceSkeletonTableDialog(
    qtbot, centered_pair_labels: Labels, flies13_skeleton: Skeleton
):
    """Test ReplaceSkeletonTableDialog."""

    def get_combo_box_items(combo_box: QComboBox) -> set:
        return set([combo_box.itemText(i) for i in range(combo_box.count())])

    def predict_combo_box_items(
        combo_box: QComboBox, base=None, include=None, exclude=None
    ) -> set:
        if isinstance(include, str):
            include = [include]
        if isinstance(exclude, str):
            exclude = [exclude]
        predicted = set([combo_box.currentText(), ""])
        predicted = predicted if base is None else predicted | set(base)
        predicted = predicted if include is None else predicted | set(include)
        predicted = predicted if exclude is None else predicted - set(exclude)
        return predicted

    labels = centered_pair_labels
    skeleton = labels.skeletons[0]

    skeleton_new = flies13_skeleton
    rename_nodes, delete_nodes, add_nodes = OpenSkeleton.compare_skeletons(
        skeleton, skeleton_new
    )

    win = ReplaceSkeletonTableDialog(
        rename_nodes,
        delete_nodes=[],
        add_nodes=[],
    )

    assert win.table is None

    win = ReplaceSkeletonTableDialog(
        rename_nodes,
        delete_nodes,
        add_nodes,
    )

    # Check that all nodes are in the table
    assert win.table.rowCount() == len(rename_nodes) + len(add_nodes)

    # Check table initialized correctly
    for i in range(win.table.rowCount()):
        table_item = win.table.item(i, 0)
        combo_box: QComboBox = win.table.cellWidget(i, 1)

        # Expect combo box to contain all `add_nodes` plus current text and `""`
        combo_box_text: str = combo_box.currentText()
        combo_box_items = get_combo_box_items(combo_box)
        expected_combo_box_items = predict_combo_box_items(combo_box, base=delete_nodes)
        assert combo_box_items == expected_combo_box_items

        # Expect rename nodes to be preset to combo with same node name
        if table_item.text() in rename_nodes:
            assert combo_box_text == table_item.text()
        else:
            assert table_item.text() in add_nodes
            assert combo_box_text == ""

    assert win.result() == {}

    # Change combo box for one row
    combo_box: QComboBox = win.table.cellWidget(0, 1)
    combo_box_text = combo_box.currentText()
    new_text = combo_box.itemText(len(rename_nodes))
    combo_box.setCurrentText(new_text)

    # Check that combo boxes update correctly
    assert get_combo_box_items(combo_box) == predict_combo_box_items(
        combo_box, base=delete_nodes, include=combo_box_text
    )
    for i in range(1, win.table.rowCount()):
        combo_box: QComboBox = win.table.cellWidget(i, 1)
        assert get_combo_box_items(combo_box) == predict_combo_box_items(
            combo_box, base=delete_nodes, include=combo_box_text, exclude=new_text
        )

    # Check that error occurs if trying to ONLY rename nodes to existing node names
    assert win.table.item(0, 0).text() in skeleton.node_names
    with pytest.raises(ValueError):
        data = win.result()

    # Change combo box of a delete node to a new node
    combo_box: QComboBox = win.table.cellWidget(len(rename_nodes), 1)
    combo_box_text = combo_box.currentText()
    new_text = combo_box.itemText(3)
    combo_box.setCurrentText(new_text)

    # This operation should be allowed since we are linking old nodes to new nodes
    # (not just renaming)
    assert win.table.item(len(rename_nodes), 0).text() not in skeleton.node_names
    data = win.result()
    assert data == {"head1": "forelegL2", "forelegL1": "forelegR3"}

def test_ExportClipDialog_message_ffmpeg_available():
    """
    Test ExportClipDialog displays the correct message when ffmpeg is available.
    """
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    dialog = ExportClipDialog()
    dialog.show()

    # Find QLabel widgets and extract their text
    label_texts = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]

    # Check that the correct ffmpeg message appears
    assert any(
        "MP4" in text and "ffmpeg" in text for text in label_texts
    ), "Expected message indicating MP4 encoding using ffmpeg."

    dialog.close()

def strip_html_tags(text: str) -> str:
    """Utility to remove HTML tags from a string."""
    return re.sub(r"<[^>]*>", "", text)

def test_ExportClipDialog_message_ffmpeg_unavailable():
    """
    Test ExportClipDialog displays the correct message when ffmpeg is unavailable.
    """
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    # Override can_use_ffmpeg temporarily
    VideoWriter.can_use_ffmpeg = lambda: False

    try:
        dialog = ExportClipDialog()
        dialog.show()
        dialog.layout().activate()

        # Extract QLabel texts and strip HTML tags
        label_texts = [strip_html_tags(label.text()) for label in dialog.findChildren(QtWidgets.QLabel)]
        print("Extracted QLabel texts:", label_texts)

        # Verify fallback message
        assert any(
            "Unable to use ffmpeg" in text and "AVI" in text for text in label_texts
        ), f"Expected AVI fallback message, got: {label_texts}"
    finally:
        VideoWriter.can_use_ffmpeg = lambda: True
    dialog.close()

def test_ExportClipAndLabelsDialog_initial_values():
    """
    Test ExportClipAndLabelsDialog initializes FPS input and checkbox correctly.
    """
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    video_fps = 25
    dialog = ExportClipAndLabelsDialog(video_fps=video_fps)
    dialog.show()

    # Verify FPS input initialization
    assert dialog.fps_input.value() == video_fps, \
        f"FPS input should initialize to {video_fps}."

    # Verify checkbox default state
    assert not dialog.open_when_done.isChecked(), \
        "Checkbox 'Open file after saving' should be unchecked by default."
    dialog.close()

def test_ExportClipAndLabelsDialog_get_form_results():
    """
    Test ExportClipAndLabelsDialog retrieves form results correctly.
    """
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    dialog = ExportClipAndLabelsDialog(video_fps=30)
    dialog.show()

    # Set form values
    dialog.fps_input.setValue(60)
    dialog.open_when_done.setChecked(True)

    # Retrieve form results
    results = dialog.get_results()

    # Verify form results
    assert results["fps"] == 60, "FPS value should match the input."
    assert results["open_when_done"] is True, "Checkbox value should be True when checked."
    dialog.close()

def test_ExportClipAndLabelsDialog_on_accept():
    """
    Test ExportClipAndLabelsDialog 'on_accept' method stores results correctly.
    """
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    dialog = ExportClipAndLabelsDialog(video_fps=30)
    dialog.show()

    # Set form values
    dialog.fps_input.setValue(45)
    dialog.open_when_done.setChecked(False)

    # Simulate dialog acceptance
    dialog.on_accept()

    # Verify stored results
    assert dialog._results["fps"] == 45, "Stored FPS should match the input value."
    assert dialog._results["open_when_done"] is False, "Stored checkbox state should be False."
    dialog.close()
