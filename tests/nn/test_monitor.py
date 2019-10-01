from sleap.nn.monitor import LossViewer


def test_monitor_release(qtbot):
    win = LossViewer()
    win.show()
    win.close()

    # Make sure the first monitor released its zmq socket
    win2 = LossViewer()
    win2.show()
    win2.close()
