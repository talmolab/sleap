from sleap.gui.commands import (
    CommandContext,
    ImportDeepLabCutFolder,
    get_new_version_filename,
)
from sleap.io.pathutils import fix_path_separator
from pathlib import PurePath


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
    csv_files = ImportDeepLabCutFolder.find_dlc_files_in_folder(
        "tests/data/dlc_multiple_datasets"
    )
    assert set([fix_path_separator(f) for f in csv_files]) == {
        "tests/data/dlc_multiple_datasets/video2/dlc_dataset_2.csv",
        "tests/data/dlc_multiple_datasets/video1/dlc_dataset_1.csv",
    }

    labels = ImportDeepLabCutFolder.import_labels_from_dlc_files(csv_files)

    assert len(labels) == 3
    assert len(labels.videos) == 2
    assert len(labels.skeletons) == 1
    assert len(labels.nodes) == 3
    assert len(labels.tracks) == 0

    assert set(
        [fix_path_separator(l.video.backend.filename) for l in labels.labeled_frames]
    ) == {
        "tests/data/dlc_multiple_datasets/video2/img002.jpg",
        "tests/data/dlc_multiple_datasets/video1/img000.jpg",
        "tests/data/dlc_multiple_datasets/video1/img000.jpg",
    }

    assert set([l.frame_idx for l in labels.labeled_frames]) == {0, 0, 1}


def test_get_new_version_filename():
    assert get_new_version_filename("labels.slp") == "labels copy.slp"
    assert get_new_version_filename("labels.v0.slp") == "labels.v1.slp"
    assert get_new_version_filename("/a/b/labels.slp") == str(
        PurePath("/a/b/labels copy.slp")
    )
    assert get_new_version_filename("/a/b/labels.v0.slp") == str(
        PurePath("/a/b/labels.v1.slp")
    )
    assert get_new_version_filename("/a/b/labels.v01.slp") == str(
        PurePath("/a/b/labels.v02.slp")
    )
