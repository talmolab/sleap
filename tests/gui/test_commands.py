from sleap.gui.commands import (
    CommandContext,
    ImportDeepLabCutFolder,
    get_new_version_filename,
    OpenSkeleton,
)
from sleap.io.pathutils import fix_path_separator
from sleap.io.dataset import Labels
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


def test_open_skeleton(min_labels):
    """Ensure CommandContext.OpenSkeleton only allows one skeleton in Labels.skeletons."""

    # Load in labels with a single skeleton.
    labels: Labels = min_labels
    assert len(labels.skeletons) == 1
    assert labels.skeleton.name == "Skeleton-0"

    # Create command context and params. Load skeleton.
    commands: CommandContext = CommandContext.from_labels(labels)
    skeleton_filename = "tests/data/skeleton/fly_skeleton_legs.json"
    params: dict = {"filename": skeleton_filename}

    # Check that new skeleton replaced skeleton instead of appending to skeletons list.
    OpenSkeleton().do_action(context=commands, params=params)
    assert len(labels.skeletons) == 1
    assert labels.skeleton.name == "skeleton_legs.mat"

    # Check that skeletons, nodes, and edges for instances are updated to new skeleton.
    # TODO: Instance skeletons are unchanged and need to be merged.
    #   Possibly reuse code from MergeDialog.__init__, Labels.complex_merge_between,
    #   LabeledFrame.complex_merge_between, and LabeledFrame.complex_frame_merge.
    #       Code could be implemented in OpenSkeleton.do_action or in callback
    #       MainWindow.on_data_update.
    print(f"\n\nlabels.skeleton.nodes = {labels.skeleton.nodes}")
    print(f"\nlabels.nodes = {labels.nodes}")
    print(f"\n\nlabels.skeleton.name = {labels.skeleton.name}")
    print(
        f"\nlabels.labeled_frames[0].instances[0].skeleton.name = "
        "{labels.labeled_frames[0].instances[0].skeleton.name}"
    )
    assert labels.labeled_frames[0].instances[0].skeleton.name == labels.skeleton.name
    assert labels.nodes == labels.skeleton.nodes
