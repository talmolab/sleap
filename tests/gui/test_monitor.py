import numpy as np

from sleap.gui.widgets.monitor import LossPlot, LossViewer


def test_monitor_release(qtbot, min_centroid_model_path):
    win = LossViewer()
    win.show()

    # Ensure win loads config correctly
    win.reset(what="Model Type", plateau_patience=10, plateau_min_delta=1e-06)
    assert win.plateau_patience == 10
    assert win.plateau_min_delta == 1e-06

    # Ensure zmq port is set correctly
    assert win.zmq_ports["controller_port"] == 9000
    assert win.zmq_ports["publish_port"] == 9001
    # Ensure all lines of update_runtime() are run error-free
    win.is_running = True
    win.t0 = 0
    # Enter "last_epoch_val_loss is not None" conditional
    win.last_epoch_val_loss = win.plateau_min_delta
    # Enter "penultimate_epoch_val_loss is not None" conditional
    win.penultimate_epoch_val_loss = win.last_epoch_val_loss
    win.mean_epoch_time_min = 0
    win.mean_epoch_time_sec = 10
    win.eta_ten_epochs_min = 2
    # Enter "epoch_in_plateau_flag" conditional
    win.epoch_in_plateau_flag = True
    win.epochs_in_plateau = 1
    # Enter "bes_val_x" conditional
    win.best_val_x = 0
    win.best_val_y = win.last_epoch_val_loss
    win._update_runtime()

    win.close()

    # Make sure the first monitor released its zmq socket
    controller_port = 9191
    zmq_ports = dict(controller_port=controller_port)
    win2 = LossViewer(zmq_ports=zmq_ports)
    win2.show()
    assert win2.zmq_ports["controller_port"] == controller_port
    assert win2.zmq_ports["publish_port"] == 9001

    # Make sure batches to show field is working correction

    # It should default to "All"
    assert win2.batches_to_show == -1
    assert win2.batches_to_show_field.currentText() == "All"

    # And it should update batches_to_show property
    win2.batches_to_show_field.setCurrentText("200")
    assert win2.batches_to_show == 200

    win2.close()

    # Ensure zmq port is set correctly
    controller_port = 9191
    publish_port = 9101
    zmq_ports = dict(controller_port=controller_port, publish_port=publish_port)
    win3 = LossViewer(zmq_ports=zmq_ports)
    win3.show()
    assert win3.zmq_ports["controller_port"] == controller_port
    assert win3.zmq_ports["publish_port"] == publish_port

    win3.close()


def test_ylim_log_scale_tracks_data(qtbot):
    """Log-scale y-limits should hug the data, not floor at 1e-8."""
    plot = LossPlot(log_scale=True, ignore_outliers=False)
    y = np.geomspace(1e-3, 1e-1, 200)
    y_min, y_max = plot._calculate_ylim(y)

    # Lower bound should be near min(y), not the legacy 1e-8 floor.
    assert y_min > 1e-4
    assert y_min < y.min()
    # Upper bound should be just above max(y).
    assert y_max > y.max()
    assert y_max < y.max() * 2


def test_ylim_log_scale_ignore_outliers(qtbot):
    """Outlier rejection should run in log space and bracket the IQR."""
    plot = LossPlot(log_scale=True, ignore_outliers=True)
    bulk = np.geomspace(1e-3, 1e-2, 200)
    # Add a single extreme outlier; IQR rejection in log space should clip it.
    y = np.concatenate([bulk, [10.0]])
    y_min, y_max = plot._calculate_ylim(y)

    assert y_min > 1e-4
    assert y_max < 1.0  # outlier excluded


def test_ylim_log_scale_all_nonpositive_fallback(qtbot):
    """All non-positive values on log scale should fall back to a safe range."""
    plot = LossPlot(log_scale=True, ignore_outliers=False)
    y = np.array([0.0, -1.0, 0.0])
    y_min, y_max = plot._calculate_ylim(y)
    assert y_min == 1e-8
    assert y_max == 1.0


def test_ylim_linear_scale_unchanged(qtbot):
    """Linear scale path should pad relative to the data range."""
    plot = LossPlot(log_scale=False, ignore_outliers=False)
    y = np.linspace(0.5, 1.5, 100)
    y_min, y_max = plot._calculate_ylim(y)
    # 2% of range = 0.02; expect padding within that order of magnitude.
    assert y.min() - 0.05 < y_min < y.min()
    assert y.max() < y_max < y.max() + 0.05


def test_batch_subsample_field_defaults_and_updates(qtbot):
    """The new batch-subsample dropdown should default to 1 and update state."""
    win = LossViewer()
    win.show()
    try:
        assert win.batch_subsample == 1
        assert win.batch_subsample_field.currentText() == "1"

        win.batch_subsample_field.setCurrentText("10")
        assert win.batch_subsample == 10

        win.batch_subsample_field.setCurrentText("100")
        assert win.batch_subsample == 100
    finally:
        win.close()


def test_batch_subsample_thins_scatter_data(qtbot):
    """When subsample > 1, scatter should receive a strided subset that ends on last."""
    win = LossViewer()
    win.show()
    try:
        win.batch_subsample = 10
        # Force a redraw on the very next batch by clearing throttle state.
        win.last_redraw_batch = None

        n = 105
        for i in range(n):
            win._add_datapoint(x=i, y=1.0 / (i + 1), which="batch")
            # Reset throttle so every point triggers a redraw — keeps the test
            # deterministic regardless of wall-clock perf.
            win.last_redraw_batch = None

        # Stored data is preserved (no subsampling on the underlying buffers).
        assert len(win.X) == n
        assert len(win.Y) == n

        # The scatter on the canvas should reflect the strided view.
        offsets = win.canvas.series["batch"].get_offsets()
        xs = np.asarray(offsets[:, 0])
        # Last point should always be visible.
        assert xs[-1] == n - 1
        # Roughly n/stride points (with last point pinned).
        expected = (n + 9) // 10
        assert len(xs) == expected
    finally:
        win.close()
