from sleap.nn.monitor import LossViewer


def test_monitor_release(qtbot):
    win = LossViewer()
    win.show()
    win.close()

    # Make sure the first monitor released its zmq socket
    win2 = LossViewer()
    win2.show()

    # Make sure batches to show field is working correction

    # It should default to "All"
    assert win2.batches_to_show == -1
    assert win2.batches_to_show_field.currentText() == "All"

    # And it should update batches_to_show property
    win2.batches_to_show_field.setCurrentText("200")
    assert win2.batches_to_show == 200

    win2.close()
