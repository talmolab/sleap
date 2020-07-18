from sleap.gui.commands import CommandContext
from sleap.gui.dialogs.delete import DeleteDialog


def test_delete_user_dialog(centered_pair_labels, qtbot):
    context = CommandContext.from_labels(centered_pair_labels)
    context.state["frame_idx"] = 123
    context.state["video"] = centered_pair_labels.videos[0]
    context.state["has_frame_range"] = True
    context.state["frame_range"] = (10, 20)

    win = DeleteDialog(context=context)

    assert len(win.get_frames_instances("user", "current video", "any")) == 140
    assert len(win.get_frames_instances("predicted", "current video", "any")) == 0

    assert len(win.get_frames_instances("user", "selected clip", "any")) == 2

    lf_inst_list = win.get_frames_instances(
        "user", "current video except for selected clip", "any"
    )
    assert len(lf_inst_list) == 138


def test_delete_predictions_dialog(centered_pair_predictions, qtbot):
    context = CommandContext.from_labels(centered_pair_predictions)
    context.state["frame_idx"] = 123
    context.state["video"] = centered_pair_predictions.videos[0]
    context.state["has_frame_range"] = True
    context.state["frame_range"] = (10, 20)

    win = DeleteDialog(context=context)

    assert len(win.get_frames_instances("user", "current video", "any")) == 0
    assert len(win.get_frames_instances("predicted", "current video", "any")) == 2274

    assert len(win.get_frames_instances("predicted", "selected clip", "any")) == 20

    lf_inst_list = win.get_frames_instances(
        "predicted", "current video except for selected clip", "any"
    )
    assert len(lf_inst_list) == 2274 - 20

    win.tracks_menu.setCurrentIndex(3)
    assert len(win.get_frames_instances("predicted", "selected clip", "")) == 10

    assert len(win.get_frames_instances("predicted", "selected clip", "no")) == 0


def test_delete_all(centered_pair_predictions, qtbot):
    context = CommandContext.from_labels(centered_pair_predictions)
    context.state["frame_idx"] = 123
    context.state["video"] = centered_pair_predictions.videos[0]
    context.state["has_frame_range"] = True
    context.state["frame_range"] = (10, 20)

    win = DeleteDialog(context=context)

    # Get all instances
    lf_inst_list = win.get_frames_instances("all", "current video", "any")
    assert len(lf_inst_list) == 2274

    # Delete them
    win._delete(lf_inst_list)

    # Make sure the all empty frames were also deleted
    assert centered_pair_predictions.labeled_frames == []
