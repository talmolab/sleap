from sleap.gui.widgets.multicheck import MultiCheckWidget

import PySide2.QtCore as QtCore


def test_gui_video(qtbot):
    cs = MultiCheckWidget(count=10, title="Test", default=True)

    cs.show()
    qtbot.addWidget(cs)

    assert cs.getSelected() == list(range(10))

    for btn in cs.check_group.buttons():
        # click all the odd buttons to uncheck them
        if cs.check_group.id(btn) % 2 == 1:
            qtbot.mouseClick(btn, QtCore.Qt.LeftButton)
    assert cs.getSelected() == list(range(0, 10, 2))

    cs.setSelected([1, 2, 3])
    assert cs.getSelected() == [1, 2, 3]

    # Watch for the app.worker.finished signal, then start the worker.
    with qtbot.waitSignal(cs.selectionChanged, timeout=10):
        qtbot.mouseClick(cs.check_group.buttons()[0], QtCore.Qt.LeftButton)

    assert cs.close()
