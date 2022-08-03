from turtle import title
from sleap.gui.widgets.monitor import LossViewer
from sleap import TrainingJobConfig


def test_monitor_release(qtbot, min_centroid_model_path):
    win = LossViewer()
    win.show()

    # Ensure win loads config correctly
    config = TrainingJobConfig.load_json(min_centroid_model_path, False)
    win.reset(what="Model Type", config=config)
    assert win.config.optimization.early_stopping.plateau_patience == 10

    # Ensure all lines of update_runtime() are run error-free
    win.is_running = True
    win.t0 = 0
    # Enter "last_epoch_val_loss is not None" conditional
    win.last_epoch_val_loss = win.config.optimization.early_stopping.plateau_min_delta
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
    win.update_runtime()

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
