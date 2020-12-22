from sleap.gui.commands import CommandContext, ImportDeepLabCutFolder


def test_delete_user_dialog(centered_pair_predictions):
    context = CommandContext.from_labels(centered_pair_predictions)
    context.state["labeled_frame"] = centered_pair_predictions.find(
        centered_pair_predictions.videos[0], frame_idx=123
    )[0]

    # No user instances, just predicted
    assert len(context.state["labeled_frame"].user_instances) == 0
    assert len(context.state["labeled_frame"].predicted_instances) == 2

    context.addUserInstancesFromPredictions()

    # Make sure we now have user instances
    assert len(context.state["labeled_frame"].user_instances) == 2


def test_import_labels_from_dlc_folder():
    csv_files = ImportDeepLabCutFolder.find_dlc_files_in_folder('tests/data/dlc_multiple_datasets')
    assert csv_files == [
        'tests/data/dlc_multiple_datasets/video2/dlc_dataset_2.csv',
        'tests/data/dlc_multiple_datasets/video1/dlc_dataset_1.csv',
    ]

    labels = ImportDeepLabCutFolder.import_labels_from_dlc_files(csv_files)

    assert len(labels) == 6
    assert len(labels.videos) == 2
    assert len(labels.skeletons) == 1
    assert len(labels.nodes) == 3
    assert len(labels.tracks) == 0

    assert [l.video.backend.filename for l in labels.labeled_frames] == [
        'tests/data/dlc_multiple_datasets/video2/img000.png',
        'tests/data/dlc_multiple_datasets/video2/img000.png',
        'tests/data/dlc_multiple_datasets/video2/img000.png',
        'tests/data/dlc_multiple_datasets/video1/img000.png',
        'tests/data/dlc_multiple_datasets/video1/img000.png',
        'tests/data/dlc_multiple_datasets/video1/img000.png',
    ]

    assert [l.frame_idx for l in labels.labeled_frames] == [0, 1, 2, 0, 1, 2]