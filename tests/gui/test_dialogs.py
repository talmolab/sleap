"""Module to test the dialogs of the GUI (contained in sleap/gui/dialogs)."""


import os
from pathlib import Path

import pytest
from PySide2.QtWidgets import QComboBox

import sleap
from sleap.skeleton import Skeleton
from sleap.io.dataset import Labels
from sleap.gui.commands import OpenSkeleton
from sleap.gui.dialogs.merge import ReplaceSkeletonTableDialog


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
