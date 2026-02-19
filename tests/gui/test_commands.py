import pytest
import shutil
import sys
import time

import numpy as np
from pathlib import PurePath, Path
from qtpy import QtCore
from typing import List

from sleap_io import (
    Skeleton,
    Track,
    PredictedInstance,
    Labels,
    LabeledFrame,
    Instance,
)
from sleap.gui.app import MainWindow
from sleap.gui.commands import (
    AddInstance,
    CommandContext,
    DeleteNode,
    ExportAnalysisFile,
    ExportDatasetWithImages,
    ExportFullPackage,
    ExportVideoClip,
    ImportDeepLabCutFolder,
    RemoveVideo,
    ReplaceVideo,
    OpenSkeleton,
    SaveProjectAs,
    DeleteFrameLimitPredictions,
    get_new_version_filename,
)
from sleap.io.convert import default_analysis_filename
from sleap.io.format.adaptor import Adaptor
from sleap.io.pathutils import fix_path_separator
from sleap_io import Video
from sleap.util import get_package_file

# These imports cause trouble when running `pytest.main()` from within the file
# Comment out to debug tests file via VSCode's "Debug Python File"
from tests.info.test_h5 import extract_meta_hdf5
from sleap.sleap_io_adaptors.video_utils import get_last_frame_idx
from sleap.sleap_io_adaptors.lf_labels_utils import (
    add_suggestion,
    labels_load_file,
    remove_video,
    labels_add_instance,
)
from sleap.sleap_io_adaptors.instance_utils import instance_same_pose_as_compat

from copy import deepcopy


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
    from sleap.sleap_io_adaptors.lf_labels_utils import labels_get_nodes

    assert len(labels_get_nodes(labels)) == 3
    assert len(labels.tracks) == 2

    # sleap-io returns video.filename as list for image sequences
    all_filenames = []
    for lf in labels.labeled_frames:
        if isinstance(lf.video.filename, list):
            # For image sequences, we need to get the specific frame
            if lf.frame_idx < len(lf.video.filename):
                all_filenames.append(
                    fix_path_separator(lf.video.filename[lf.frame_idx])
                )
            else:
                # If frame_idx is out of bounds, use the first image
                all_filenames.append(fix_path_separator(lf.video.filename[0]))
        else:
            all_filenames.append(fix_path_separator(lf.video.filename))

    assert set(all_filenames) == {
        "tests/data/dlc_multiple_datasets/video2/img002.jpg",
        "tests/data/dlc_multiple_datasets/video1/img000.jpg",
        "tests/data/dlc_multiple_datasets/video1/img001.jpg",
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


def test_RemoveVideo(
    centered_pair_predictions: Labels,
    small_robot_mp4_vid: Video,
    centered_pair_vid: Video,
):
    def ask(obj: RemoveVideo, context: CommandContext, params: dict) -> bool:
        return True

    RemoveVideo.ask = ask

    from sleap.sleap_io_adaptors.lf_labels_utils import labels_copy, labels_add_video

    labels = labels_copy(centered_pair_predictions)
    labels_add_video(labels, small_robot_mp4_vid)
    labels_add_video(labels, centered_pair_vid)

    all_videos = labels.videos
    assert len(all_videos) == 3

    video_idxs = [1, 2]
    videos_to_remove = [labels.videos[i] for i in video_idxs]

    context = CommandContext.from_labels(labels)
    context.state["selected_batch_video"] = video_idxs
    context.state["video"] = labels.videos[1]

    context.removeVideo()

    assert len(labels.videos) == 1
    assert context.state["video"] not in videos_to_remove


@pytest.mark.parametrize("out_suffix", ["h5", "csv"])
def test_ExportAnalysisFile(
    centered_pair_predictions: Labels,
    centered_pair_predictions_hdf5_path: str,
    small_robot_mp4_vid: Video,
    out_suffix: str,
    tmpdir,
):
    if out_suffix == "csv":
        csv = True
    else:
        csv = False

    def ExportAnalysisFile_ask(context: CommandContext, params: dict):
        """Taken from ExportAnalysisFile.ask()"""

        def ask_for_filename(default_name: str) -> str:
            """Allow user to specify the filename"""
            # MODIFIED: Does not open dialog.
            return default_name

        labels = context.labels
        if len(labels.labeled_frames) == 0:
            raise ValueError("No labeled frames in project. Nothing to export.")

        if params["all_videos"]:
            all_videos = context.labels.videos
        else:
            all_videos = [context.state["video"] or context.labels.videos[0]]

        from sleap.sleap_io_adaptors.lf_labels_utils import labels_get

        # Check for labeled frames in each video
        videos = [video for video in all_videos if len(labels_get(labels, video)) != 0]
        if len(videos) == 0:
            raise ValueError("No labeled frames in video(s). Nothing to export.")

        default_name = "labels"
        fn = PurePath(tmpdir, default_name)
        if len(videos) == 1:
            # Allow user to specify the filename
            use_default = False
            dirname = str(fn.parent)
        else:
            # Allow user to specify directory, but use default filenames
            use_default = True
            dirname = str(fn.parent)  # MODIFIED: Does not open dialog.
            if len(dirname) == 0:
                return False

        output_paths = []
        analysis_videos = []
        for video in videos:
            # Create the filename
            default_name = default_analysis_filename(
                labels=labels,
                video=video,
                output_path=dirname,
                output_prefix=str(fn.stem),
                format_suffix=out_suffix,
            )
            filename = default_name if use_default else ask_for_filename(default_name)

            if len(filename) != 0:
                analysis_videos.append(video)
                output_paths.append(filename)

        if len(output_paths) == 0:
            return False

        params["analysis_videos"] = zip(output_paths, videos)
        params["eval_analysis_videos"] = zip(output_paths, videos)
        return True

    def assert_videos_written(num_videos: int, labels_path: str = None):
        output_paths = []
        for output_path, video in params["eval_analysis_videos"]:
            assert Path(output_path).exists()
            output_paths.append(output_path)

            if labels_path is not None and not params["csv"]:
                meta_reader = extract_meta_hdf5
                labels_key = "labels_path" if out_suffix == "h5" else "project"
                read_meta = meta_reader(output_path, dset_names_in=["labels_path"])
                assert read_meta[labels_key] == labels_path

        assert len(output_paths) == num_videos, "Wrong number of outputs written"
        assert len(set(output_paths)) == num_videos, "Some output paths overwritten"

    tmpdir = Path(tmpdir)

    from sleap.sleap_io_adaptors.lf_labels_utils import labels_copy

    labels = labels_copy(centered_pair_predictions)
    context = CommandContext.from_labels(labels)
    context.state["filename"] = None

    if csv:
        context.state["filename"] = centered_pair_predictions_hdf5_path

        params = {"all_videos": True, "csv": csv}
        okay = ExportAnalysisFile_ask(context=context, params=params)
        assert okay
        ExportAnalysisFile.do_action(context=context, params=params)
        assert_videos_written(num_videos=1, labels_path=context.state["filename"])

        return

    # Test with all_videos False (single video)
    params = {"all_videos": False, "csv": csv}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add labels path and test with all_videos True (single video)
    context.state["filename"] = str(tmpdir.with_name("path.to.labels"))
    params = {"all_videos": True, "csv": csv}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add a video (no labels) and test with all_videos True
    from sleap.sleap_io_adaptors.lf_labels_utils import labels_add_video

    labels_add_video(labels, small_robot_mp4_vid)

    params = {"all_videos": True, "csv": csv}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add labels and test with all_videos False
    labeled_frame = labels.find(video=labels.videos[1], frame_idx=0, return_new=True)[0]
    instance = Instance(
        points=Instance.empty(labels.skeleton).points, skeleton=labels.skeleton
    )
    labels_add_instance(labels, labeled_frame, instance)
    labels.append(labeled_frame)

    params = {"all_videos": False, "csv": csv}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add specific video and test with all_videos False
    context.state["videos"] = labels.videos[1]

    params = {"all_videos": False, "csv": csv}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Test with all videos True
    params = {"all_videos": True, "csv": csv}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=2, labels_path=context.state["filename"])

    # Test with videos with the same filename
    (tmpdir / "session1").mkdir()
    (tmpdir / "session2").mkdir()
    shutil.copy(
        centered_pair_predictions.video.filename,
        tmpdir / "session1" / "video.mp4",
    )
    shutil.copy(small_robot_mp4_vid.filename, tmpdir / "session2" / "video.mp4")

    labels.videos[0].filename = str(tmpdir / "session1" / "video.mp4")
    labels.videos[1].filename = str(tmpdir / "session2" / "video.mp4")

    params = {"all_videos": True, "csv": csv}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=2, labels_path=context.state["filename"])

    # Remove all videos and test
    all_videos = list(labels.videos)
    for video in all_videos:
        remove_video(labels, labels.videos[-1])

    params = {"all_videos": True, "csv": csv}
    with pytest.raises(ValueError):
        okay = ExportAnalysisFile_ask(context=context, params=params)


def assert_video_params(
    video: Video,
    filename: str = None,
    filenames: List[str] = None,
    grayscale: bool = None,
    bgr: bool = None,
    height: int = None,
    width: int = None,
    channels: int = None,
    reset: bool = False,
):
    if filename is not None:
        assert video.filename == filename

    if grayscale is not None:
        assert video.grayscale == grayscale

    # TODO: Align reset behavior?
    # if reset and isinstance(video.backend, MediaVideo):
    #     assert video.backend._reader_ is None
    #     assert video.backend._test_frame_ is None

    # Getting the channels will assert some of the above are not None
    if grayscale is not None:
        assert video.shape[3] == 3 ** (not grayscale)


def test_ToggleGrayscale(centered_pair_predictions: Labels):
    """Test functionality for ToggleGrayscale on mp4/avi video"""
    labels = centered_pair_predictions
    video = labels.video
    grayscale = video.grayscale
    filename = video.filename

    context = CommandContext.from_labels(labels)
    context.state["video"] = video

    # Toggle grayscale to "not grayscale"
    context.toggleGrayscale()
    assert_video_params(video=video, filename=filename, grayscale=(not grayscale))

    # Toggle grayscale back to "grayscale"
    context.toggleGrayscale()
    assert_video_params(video=video, filename=filename, grayscale=grayscale)


def test_ReplaceVideo(
    centered_pair_predictions: Labels, small_robot_mp4_vid: Video, hdf5_vid: Video
):
    """Test functionality for ToggleGrayscale on mp4/avi video"""

    def get_last_lf_in_video(labels, video):
        from sleap.sleap_io_adaptors.lf_labels_utils import labels_get

        lfs: List[LabeledFrame] = list(labels_get(labels, video))
        lfs.sort(key=lambda lf: lf.frame_idx)
        return lfs[-1].frame_idx

    def replace_video(
        new_video: Video, videos_to_replace: List[Video], context: CommandContext
    ):
        # Video to be imported
        new_video_filename = new_video.filename

        # Replace the video
        import_item_list = [
            {"params": {"filename": new_video_filename, "grayscale": True}}
        ]
        params = {"import_list": zip(import_item_list, videos_to_replace)}
        ReplaceVideo.do_action(context=context, params=params)
        return new_video_filename

    # Labels and video to be replaced
    labels = centered_pair_predictions
    context = CommandContext.from_labels(labels)
    videos = labels.videos
    get_last_lf_in_video(labels, videos[0])

    # Replace the video
    new_video_filename = replace_video(small_robot_mp4_vid, videos, context)

    # Ensure video backend was replaced
    video = labels.video
    assert len(labels.videos) == 1
    assert video.grayscale
    assert video.filename == new_video_filename

    # Ensure labels were truncated (Original video was fully labeled)
    new_last_lf_frame = get_last_lf_in_video(labels, video)
    # Original video was fully labeled
    assert new_last_lf_frame == get_last_frame_idx(labels.video)

    # Attempt to replace an mp4 with an hdf5 video
    with pytest.raises(TypeError):
        replace_video(hdf5_vid, labels.videos, context)


@pytest.mark.skip(reason="Test is freezing. Underlying functionality will be replaced.")
def test_exportNWB(centered_pair_predictions, tmpdir):
    """Test that exportNWB command writes an nwb file."""

    def SaveProjectAs_ask(context: CommandContext, params: dict) -> bool:
        """Replica of SaveProject.ask without the GUI element."""
        default_name = context.state["filename"]
        if "adaptor" in params:
            adaptor: Adaptor = params["adaptor"]
            default_name += f".{adaptor.default_ext}"
            filters = [f"(*.{ext})" for ext in adaptor.all_exts]
            filters[0] = f"{adaptor.name} {filters[0]}"
        else:
            filters = ["SLEAP labels dataset (*.slp)"]
            if default_name:
                default_name = get_new_version_filename(default_name)
            else:
                default_name = "labels.v000.slp"

        # Original function opens GUI here
        filename = default_name

        if len(filename) == 0:
            return False

        params["filename"] = filename
        return True

    # Set-up Labels and context
    labels: Labels = centered_pair_predictions
    context = CommandContext.from_labels(centered_pair_predictions)
    # Add fake method required by SaveProjectAs.do_action
    context.app.__setattr__("plotFrame", lambda: None)
    fn = PurePath(tmpdir, "test_nwb.slp")
    context.state["filename"] = str(fn)
    context.state["labels"] = labels

    # Ensure ".nwb" extension is appended to filename
    params = {"adaptor": "nwb"}
    SaveProjectAs_ask(context, params=params)
    assert PurePath(params["filename"]).suffix == ".nwb"

    # Ensure file was created
    SaveProjectAs.do_action(context=context, params=params)
    assert Path(params["filename"]).exists()

    # Test import nwb
    read_labels = Labels.load_nwb(params["filename"])
    assert len(read_labels.labeled_frames) == len(labels.labeled_frames)
    assert len(read_labels.videos) == len(labels.videos)
    assert read_labels.skeleton.node_names == labels.skeleton.node_names
    assert read_labels.skeleton.edge_inds == labels.skeleton.edge_inds
    assert len(read_labels.tracks) == len(labels.tracks)


def test_OpenSkeleton(
    centered_pair_predictions: Labels, stickman: Skeleton, fly_legs_skeleton_json: str
):
    def assert_skeletons_match(new_skeleton: Skeleton, skeleton: Skeleton):
        print(skeleton)
        print(new_skeleton)
        print(skeleton.symmetries)
        print(new_skeleton.symmetries)
        assert skeleton.matches(new_skeleton)
        # # Node names match
        # assert len(set(new_skeleton.nodes) - set(skeleton.nodes))
        # # Edges match
        # for (new_src, new_dst), (src, dst) in zip(new_skeleton.edges, skeleton.edges):
        #     assert new_src.name == src.name
        #     assert new_dst.name == dst.name

        # # Symmetries match
        # for (new_src, new_dst), (src, dst) in zip(
        #     new_skeleton.symmetries, skeleton.symmetries
        # ):
        #     assert new_src.name == src.name
        #     assert new_dst.name == dst.name

    def OpenSkeleton_ask(context: CommandContext, params: dict) -> bool:
        """Implement `OpenSkeleton.ask` without GUI elements."""
        template = (
            context.app.currentText
        )  # Original function uses `QComboBox.currentText()`
        if template == "Custom":
            # Original function opens FileDialog here
            filename = params["filename_in"]
        else:
            filename = get_package_file(f"skeletons/{template}.json")
        if len(filename) == 0:
            return False

        okay = True
        if len(context.labels.skeletons) > 0:
            # Ask user permission to merge skeletons
            okay = False
            skeleton: Skeleton = context.labels.skeleton  # Assumes single skeleton

            # Load new skeleton and compare
            new_skeleton = OpenSkeleton.load_skeleton(filename)
            (rename_nodes, delete_nodes, add_nodes) = OpenSkeleton.compare_skeletons(
                skeleton, new_skeleton
            )

            # Original function shows pop-up warning here
            if (len(delete_nodes) > 0) or (len(add_nodes) > 0):
                linked_nodes = {
                    "abdomen": "body",
                    "wingL": "left-arm",
                    "wingR": "right-arm",
                }
                delete_nodes = list(set(delete_nodes) - set(linked_nodes.values()))
                add_nodes = list(set(add_nodes) - set(linked_nodes.keys()))
                params["linked_nodes"] = linked_nodes

            params["delete_nodes"] = delete_nodes
            params["add_nodes"] = add_nodes

        params["filename"] = filename
        return okay

    labels = centered_pair_predictions
    skeleton = labels.skeleton
    skeleton.add_symmetry(skeleton.nodes[0].name, skeleton.nodes[1].name)
    print(skeleton.symmetries)
    context = CommandContext.from_labels(labels)
    context.app.__setattr__("currentText", "Custom")
    # Add multiple skeletons to and ensure the unused skeleton is removed
    labels.skeletons.append(stickman)
    print("line: 578:", skeleton.symmetries)

    # Run without OpenSkeleton.ask()
    params = {"filename": fly_legs_skeleton_json}
    new_skeleton = OpenSkeleton.load_skeleton(fly_legs_skeleton_json)
    OpenSkeleton.do_action(context, params)
    print("line: 586:", skeleton.symmetries)
    assert len(labels.skeletons) == 1

    # State is updated
    assert context.state["skeleton"] == skeleton

    # Structure is identical
    print("line: 591:", skeleton.symmetries)
    assert_skeletons_match(new_skeleton, skeleton)

    # Run again with OpenSkeleton_ask()
    labels.skeletons = [stickman]
    params = {"filename_in": fly_legs_skeleton_json}
    OpenSkeleton_ask(context, params)
    assert params["filename"] == fly_legs_skeleton_json
    assert len(set(params["delete_nodes"]) & set(params["linked_nodes"])) == 0
    assert len(set(params["add_nodes"]) & set(params["linked_nodes"])) == 0
    OpenSkeleton.do_action(context, params)
    assert_skeletons_match(new_skeleton, stickman)

    # Run again with template set
    context.app.currentText = "fly32"
    fly32_json = get_package_file("skeletons/fly32.json")
    OpenSkeleton_ask(context, params)
    assert params["filename"] == fly32_json
    fly32_skeleton = OpenSkeleton.load_skeleton(fly32_json)
    OpenSkeleton.do_action(context, params)
    assert_skeletons_match(labels.skeleton, fly32_skeleton)


def test_DeleteNode_updates_instances(centered_pair_predictions: Labels, tmpdir):
    """Test that DeleteNode properly updates instance point data.

    This is a regression test for GitHub Discussion #2500 where removing nodes from
    a skeleton did not update instance point data, causing file corruption that
    prevented the file from being loaded.

    The bug occurred because DeleteNode called skeleton.remove_node() directly,
    which only updates the skeleton but NOT the instance point arrays. The fix
    is to use Labels.remove_nodes() which properly calls Instance.update_skeleton()
    to update all instances.
    """
    from sleap.sleap_io_adaptors.lf_labels_utils import labels_copy, labels_load_file

    # Create a copy to avoid mutating the fixture
    labels = labels_copy(centered_pair_predictions)

    # Get original state
    original_skeleton = labels.skeleton
    original_num_nodes = len(original_skeleton.nodes)

    # Pick a node to delete (use a middle node to test index handling)
    node_to_delete = original_skeleton.nodes[1]  # Second node
    node_name_to_delete = node_to_delete.name

    # Set up command context
    context = CommandContext.from_labels(labels)
    context.state["skeleton"] = original_skeleton
    context.state["selected_node"] = node_to_delete

    # Execute the DeleteNode command
    DeleteNode.do_action(context, params={})

    # Verify skeleton was updated
    assert len(labels.skeleton.nodes) == original_num_nodes - 1
    assert node_name_to_delete not in labels.skeleton.node_names

    # Verify ALL instances have updated point arrays that match the new skeleton
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            # Instance points should have the same number of points as skeleton nodes
            assert len(inst.points) == len(labels.skeleton.nodes), (
                f"Instance has {len(inst.points)} points but skeleton has "
                f"{len(labels.skeleton.nodes)} nodes. "
                "Instance point data was not updated when node was deleted!"
            )
            # Point names should match skeleton node names
            assert list(inst.points["name"]) == labels.skeleton.node_names, (
                "Instance point names do not match skeleton node names!"
            )

    # Save and reload to verify file is not corrupted
    save_path = Path(tmpdir) / "test_delete_node.slp"
    labels.save(str(save_path))

    # This should NOT raise an error - the bug caused a ValueError here
    reloaded_labels = labels_load_file(str(save_path))

    # Verify reloaded data is correct
    assert len(reloaded_labels.skeleton.nodes) == original_num_nodes - 1
    assert node_name_to_delete not in reloaded_labels.skeleton.node_names
    assert len(reloaded_labels.labeled_frames) == len(labels.labeled_frames)

    # Verify all reloaded instances have correct point counts
    for lf in reloaded_labels.labeled_frames:
        for inst in lf.instances:
            assert len(inst.points) == len(reloaded_labels.skeleton.nodes)


def test_DeleteNode_without_labels():
    """Test that DeleteNode still works when no Labels object is available.

    When editing a skeleton without a Labels context (e.g., in a skeleton-only
    editor), the command should fall back to calling skeleton.remove_node()
    directly.
    """
    # Create a standalone skeleton (not associated with Labels)
    skeleton = Skeleton(
        nodes=["head", "neck", "tail"],
        edges=[("head", "neck"), ("neck", "tail")],
    )

    # Set up command context without labels
    context = CommandContext.from_labels(None)
    context.state["skeleton"] = skeleton
    context.state["selected_node"] = skeleton.nodes[1]  # "neck"

    # Execute the DeleteNode command
    DeleteNode.do_action(context, params={})

    # Verify skeleton was updated
    assert len(skeleton.nodes) == 2
    assert "neck" not in skeleton.node_names
    assert skeleton.node_names == ["head", "tail"]
    # Edge from head to neck should be removed
    assert len(skeleton.edges) == 0


def test_SaveProjectAs(centered_pair_predictions: Labels, tmpdir):
    """Test that project can be saved as default slp extension"""

    context = CommandContext.from_labels(centered_pair_predictions)
    # Add fake method required by SaveProjectAs.do_action
    context.app.__setattr__("plotFrame", lambda: None)
    params = {}
    fn = PurePath(tmpdir, "test_save-project-as.slp")
    params["filename"] = str(fn)
    context.state["labels"] = centered_pair_predictions

    SaveProjectAs.do_action(context=context, params=params)
    assert Path(params["filename"]).exists()


def test_SetSelectedInstanceTrack(centered_pair_predictions: Labels):
    """Test that setting new track on instance also sets track on linked prediction."""
    # Extract labeled frame and instance
    labels = centered_pair_predictions
    lf: LabeledFrame = labels[0]
    pred_inst = lf.instances[0]

    # Set-up command context
    context: CommandContext = CommandContext.from_labels(labels)
    context.state["labeled_frame"] = lf
    context.state["frame_idx"] = lf.frame_idx
    context.state["skeleton"] = labels.skeleton
    context.state["video"] = labels.videos[0]

    # Remove all tracks
    from sleap.sleap_io_adaptors.lf_labels_utils import remove_all_tracks

    remove_all_tracks(labels)

    # Create instance from predicted instance
    context.newInstance(copy_instance=pred_inst, mark_complete=False)

    # Set track on new instance
    new_instance = [inst for inst in lf.instances if inst.from_predicted is not None][0]
    context.state["instance"] = new_instance
    track = Track(name="test_track")
    context.setInstanceTrack(new_track=track)

    # Ensure that both instance and predicted instance have same track
    assert new_instance.track == track
    assert pred_inst.track == new_instance.track


def test_DeleteMultipleTracks(min_tracks_2node_labels: Labels):
    """Test that deleting multiple tracks works as expected."""
    labels = min_tracks_2node_labels
    tracks = labels.tracks
    tracks.append(Track(name="unused"))
    assert len(tracks) == 3

    # Set-up command context
    context: CommandContext = CommandContext.from_labels(labels)
    context.state["labels"] = labels

    # Delete unused tracks
    context.deleteMultipleTracks(delete_all=False)
    assert len(labels.tracks) == 2

    # Add back an unused track and delete all tracks
    tracks.append(Track(name="unused"))
    assert len(tracks) == 3
    context.deleteMultipleTracks(delete_all=True)
    assert len(labels.tracks) == 0


def test_CopyInstance(min_tracks_2node_labels: Labels):
    """Test that copying an instance works as expected."""
    labels = min_tracks_2node_labels
    instance = labels[0].instances[0]

    # Set-up command context
    context: CommandContext = CommandContext.from_labels(labels)

    # Copy instance
    assert context.state["instance"] is None
    context.copyInstance()
    assert context.state["clipboard_instance"] is None

    # Copy instance
    context.state["instance"] = instance
    context.copyInstance()
    assert context.state["clipboard_instance"] == instance


def test_PasteInstance(min_tracks_2node_labels: Labels):
    """Test that pasting an instance works as expected."""
    labels = min_tracks_2node_labels
    lf_to_copy: LabeledFrame = labels.labeled_frames[0]
    instance: Instance = lf_to_copy.instances[0]

    # Set-up command context
    context: CommandContext = CommandContext.from_labels(labels)

    def paste_instance(
        lf_to_paste: LabeledFrame, assertions_pre_paste, assertions_post_paste
    ):
        """Helper function to test pasting an instance."""
        instances_checkpoint = list(lf_to_paste.instances)
        assertions_pre_paste(instance, lf_to_copy)

        context.pasteInstance()
        assertions_post_paste(instances_checkpoint, lf_to_copy, lf_to_paste)

    # Case 1: No instance copied, but frame selected

    def assertions_prior(*args):
        assert context.state["clipboard_instance"] is None

    def assertions_post(instances_checkpoint, lf_to_copy, *args):
        assert instances_checkpoint == lf_to_copy.instances

    context.state["labeled_frame"] = lf_to_copy
    paste_instance(lf_to_copy, assertions_prior, assertions_post)

    # Case 2: No frame selected, but instance copied

    def assertions_prior(*args):
        assert context.state["labeled_frame"] is None

    context.state["labeled_frame"] = None
    context.state["clipboard_instance"] = instance
    paste_instance(lf_to_copy, assertions_prior, assertions_post)

    # Case 3: Instance copied and current frame selected

    def assertions_prior(instance, lf_to_copy, *args):
        assert context.state["clipboard_instance"] == instance
        assert context.state["labeled_frame"] == lf_to_copy

    def assertions_post(instances_checkpoint, lf_to_copy, lf_to_paste, *args):
        lf_checkpoint_tracks = [
            inst.track for inst in instances_checkpoint if inst.track is not None
        ]
        lf_to_copy_tracks = [
            inst.track for inst in lf_to_copy.instances if inst.track is not None
        ]
        assert len(lf_checkpoint_tracks) == len(lf_to_copy_tracks)
        assert len(lf_to_paste.instances) == len(instances_checkpoint) + 1
        assert instance_same_pose_as_compat(lf_to_paste.instances[-1], instance)

    context.state["labeled_frame"] = lf_to_copy
    context.state["clipboard_instance"] = instance
    paste_instance(lf_to_copy, assertions_prior, assertions_post)

    # Case 4: Instance copied and different frame selected, but new frame has same track

    def assertions_prior(instance, lf_to_copy, *args):
        assert context.state["clipboard_instance"] == instance
        assert context.state["labeled_frame"] != lf_to_copy
        lf_to_paste = context.state["labeled_frame"]
        tracks_in_lf_to_paste = [
            inst.track for inst in lf_to_paste.instances if inst.track is not None
        ]
        assert instance.track in tracks_in_lf_to_paste

    lf_to_paste = labels.labeled_frames[1]
    context.state["labeled_frame"] = lf_to_paste
    paste_instance(lf_to_paste, assertions_prior, assertions_post)

    # Case 5: Instance copied and different frame selected, and track not in new frame

    def assertions_prior(instance, lf_to_copy, *args):
        assert context.state["clipboard_instance"] == instance
        assert context.state["labeled_frame"] != lf_to_copy
        lf_to_paste = context.state["labeled_frame"]
        tracks_in_lf_to_paste = [
            inst.track for inst in lf_to_paste.instances if inst.track is not None
        ]
        assert instance.track not in tracks_in_lf_to_paste

    def assertions_post(instances_checkpoint, lf_to_copy, lf_to_paste, *args):
        assert len(lf_to_paste.instances) == len(instances_checkpoint) + 1
        assert instance_same_pose_as_compat(lf_to_paste.instances[-1], instance)
        assert lf_to_paste.instances[-1].track == instance.track

    lf_to_paste = labels.labeled_frames[2]
    context.state["labeled_frame"] = lf_to_paste
    for inst in lf_to_paste.instances:
        inst.track = None
    paste_instance(lf_to_paste, assertions_prior, assertions_post)

    # Case 6: Instance copied, different frame selected, and frame not in Labels

    def assertions_prior(instance, lf_to_copy, *args):
        assert context.state["clipboard_instance"] == instance
        assert context.state["labeled_frame"] != lf_to_copy
        assert context.state["labeled_frame"] not in labels.labeled_frames

    def assertions_post(instances_checkpoint, lf_to_copy, lf_to_paste, *args):
        from sleap.sleap_io_adaptors.instance_utils import instance_same_pose_as_compat

        assert len(lf_to_paste.instances) == len(instances_checkpoint) + 1
        assert instance_same_pose_as_compat(lf_to_paste.instances[-1], instance)
        assert lf_to_paste.instances[-1].track == instance.track
        assert lf_to_paste in labels.labeled_frames

    from sleap.sleap_io_adaptors.lf_labels_utils import labels_get

    lf_to_paste = labels_get(labels, labels.videos[0], frame_idx=3)
    labels.labeled_frames.remove(lf_to_paste)
    lf_to_paste.instances = []
    context.state["labeled_frame"] = lf_to_paste
    paste_instance(lf_to_paste, assertions_prior, assertions_post)


def test_CopyInstanceTrack(min_tracks_2node_labels: Labels):
    """Test that copying a track from one instance to another works."""
    labels = min_tracks_2node_labels
    instance = labels.labeled_frames[0].instances[0]

    # Set-up CommandContext
    context: CommandContext = CommandContext.from_labels(labels)

    # Case 1: No instance selected
    context.copyInstanceTrack()
    assert context.state["clipboard_track"] is None

    # Case 2: Instance selected and track
    context.state["instance"] = instance
    context.copyInstanceTrack()
    assert context.state["clipboard_track"] == instance.track

    # Case 3: Instance selected and no track
    instance.track = None
    context.copyInstanceTrack()
    assert context.state["clipboard_track"] is None


def test_PasteInstanceTrack(min_tracks_2node_labels: Labels):
    """Test that pasting a track from one instance to another works."""
    labels = min_tracks_2node_labels
    instance = labels.labeled_frames[0].instances[0]

    # Set-up CommandContext
    context: CommandContext = CommandContext.from_labels(labels)

    # Case 1: No instance selected
    context.state["clipboard_track"] = instance.track

    context.pasteInstanceTrack()
    assert context.state["instance"] is None

    # Case 2: Instance selected and track
    lf_to_paste = labels.labeled_frames[1]
    instance_with_same_track = lf_to_paste.instances[0]
    instance_to_paste = lf_to_paste.instances[1]
    context.state["instance"] = instance_to_paste
    context.state["labeled_frame"] = lf_to_paste
    print(instance.track, instance_with_same_track.track, instance_to_paste.track)
    assert instance_to_paste.track != instance.track
    assert instance_with_same_track.track == instance.track

    context.pasteInstanceTrack()
    assert instance_to_paste.track == instance.track
    assert instance_with_same_track.track is None

    # Case 3: Instance selected and no track
    lf_to_paste = labels.labeled_frames[2]
    instance_to_paste = lf_to_paste.instances[0]
    instance.track = None

    context.pasteInstanceTrack()
    assert isinstance(instance_to_paste.track, Track)


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Files being using in parallel by linux CI tests via Github Actions "
    "(and linux tests give us codecov reports)",
)
@pytest.mark.parametrize("video_move_case", ["new_directory", "new_name"])
def test_LoadProjectFile(
    centered_pair_predictions_slp_path: str,
    video_move_case,
    tmpdir,
):
    """Test that changing a labels object on load flags any changes."""

    def ask_LoadProjectFile(params):
        """Implement `LoadProjectFile.ask` without GUI elements."""
        filename: Path = params["filename"]
        from sleap.sleap_io_adaptors.lf_labels_utils import make_video_callback

        gui_video_callback = make_video_callback(
            search_paths=[str(filename)], context=params
        )
        from sleap.sleap_io_adaptors.lf_labels_utils import (
            labels_load_file,
        )

        labels = labels_load_file(
            centered_pair_predictions_slp_path, video_search=gui_video_callback
        )
        return labels

    def load_and_assert_changes(new_video_path: Path):
        # Load the project
        params = {"filename": new_video_path}
        ask_LoadProjectFile(params)

        # Assert project has changes
        assert params["changed_on_load"]

    # Get labels and video path
    from sleap.sleap_io_adaptors.lf_labels_utils import labels_load_file

    labels = labels_load_file(centered_pair_predictions_slp_path)
    expected_video_path = Path(labels.video.filename)

    # Move video to new location based on case
    if video_move_case == "new_directory":  # Needs to have same name
        new_video_path = Path(tmpdir, expected_video_path.name)
    else:  # Needs to have different name
        new_video_path = expected_video_path.with_name("new_name.mp4")
    shutil.move(expected_video_path, new_video_path)  # Move video to new location

    # Shorten video path if using directory location only
    search_path = (
        new_video_path.parent if video_move_case == "new_directory" else new_video_path
    )

    # Load project and assert changes
    try:
        load_and_assert_changes(search_path)
    finally:  # Move video back to original location - for ease of re-testing
        shutil.move(new_video_path, expected_video_path)


def test_DeleteFrameLimitPredictions(
    centered_pair_predictions: Labels, centered_pair_vid: Video
):
    """Test deleting instances beyond a certain frame limit."""
    labels = centered_pair_predictions

    # Set-up command context
    context = CommandContext.from_labels(labels)
    context.state["video"] = centered_pair_vid

    # Set-up params for the command
    params = {"min_frame_idx": 900, "max_frame_idx": 1000}

    instances_to_delete = DeleteFrameLimitPredictions.get_frame_instance_list(
        context, params
    )

    assert len(instances_to_delete) == 2070


@pytest.mark.parametrize("export_extension", [".slp"])
def test_exportLabelsPackage(export_extension, centered_pair_labels: Labels, tmpdir):
    def assert_loaded_package_similar(path_to_pkg: Path, sugg=False, pred=False):
        """Assert that the loaded labels are similar to the original."""

        from sleap.sleap_io_adaptors.lf_labels_utils import labels_load_file

        # Load the labels, but first copy file to a location (which pytest can and will
        # keep in memory, but won't affect our re-use of the original file name)
        filename_for_pytest_to_hoard: Path = path_to_pkg.with_name(
            f"pytest_labels_{time.perf_counter_ns()}{export_extension}"
        )
        shutil.copyfile(path_to_pkg.as_posix(), filename_for_pytest_to_hoard.as_posix())
        labels_reload: Labels = labels_load_file(
            filename_for_pytest_to_hoard.as_posix()
        )

        assert len(labels_reload.labeled_frames) == len(centered_pair_labels)
        assert len(labels_reload.videos) == len(centered_pair_labels.videos)
        assert len(labels_reload.suggestions) == len(centered_pair_labels.suggestions)
        assert len(labels_reload.tracks) == len(centered_pair_labels.tracks)
        assert len(labels_reload.skeletons) == len(centered_pair_labels.skeletons)
        assert (
            len(
                set(labels_reload.skeleton.node_names)
                - set(centered_pair_labels.skeleton.node_names)
            )
            == 0
        )
        num_images = len(labels_reload)
        if sugg:
            num_images += len(lfs_sugg)
        if not pred:
            num_images -= len(lfs_pred)
        assert len(labels_reload.video) == num_images

    # Set-up CommandContext
    path_to_pkg = Path(tmpdir, "test_exportLabelsPackage.ext")
    path_to_pkg = path_to_pkg.with_suffix(export_extension)

    def no_gui_ask(cls, context, params):
        """No GUI version of `ExportDatasetWithImages.ask`."""
        params["filename"] = path_to_pkg.as_posix()
        params["verbose"] = False
        return True

    ExportDatasetWithImages.ask = no_gui_ask

    # Remove frames we want to use for suggestions and predictions
    lfs_sugg = [centered_pair_labels[idx] for idx in [-1, -2]]
    lfs_pred = [centered_pair_labels[idx] for idx in [-3, -4]]
    from sleap.sleap_io_adaptors.lf_labels_utils import remove_frames

    remove_frames(centered_pair_labels, lfs_sugg)

    # Add suggestions
    for lf in lfs_sugg:
        add_suggestion(centered_pair_labels, centered_pair_labels.video, lf.frame_idx)

    # Add predictions and remove user instances from those frames
    for lf in lfs_pred:
        predicted_inst = PredictedInstance.from_numpy(
            lf.instances[0].points["xy"], skeleton=lf.instances[0].skeleton, score=0.5
        )
        lf.instances = []
        labels_add_instance(centered_pair_labels, lf, predicted_inst)
        # for inst in lf.user_instances:
        # remove_instance(centered_pair_labels, inst, lf)

    # FIXME: DEEPCOPY Labels to prevent mutation during case exports
    context = CommandContext.from_labels(deepcopy(centered_pair_labels))

    # Case 1: Export user-labeled frames with image data into a single SLP file.
    context.exportUserLabelsPackage()
    assert path_to_pkg.exists()
    assert_loaded_package_similar(path_to_pkg)

    # Case 2: Export user-labeled frames and suggested frames with image data.
    context = CommandContext.from_labels(deepcopy(centered_pair_labels))
    context.exportTrainingPackage()
    assert_loaded_package_similar(path_to_pkg, sugg=True)

    # Case 3: Export all frames and suggested frames with image data.
    context = CommandContext.from_labels(deepcopy(centered_pair_labels))
    context.exportFullPackage()
    assert_loaded_package_similar(path_to_pkg, sugg=True, pred=True)


def test_newInstance(qtbot, centered_pair_predictions: Labels):
    # Get the data
    labels = centered_pair_predictions
    lf = labels[0]
    pred_inst = lf.instances[0]
    video = labels.video

    # Set-up command context
    main_window = MainWindow(labels=labels)
    context = main_window.commands
    context.state["labeled_frame"] = lf
    context.state["frame_idx"] = lf.frame_idx
    context.state["skeleton"] = labels.skeleton
    context.state["video"] = labels.videos[0]

    # Case 1: Double clicking a prediction results in no offset for new instance

    # Double click on prediction
    assert len(lf.instances) == 2
    main_window._handle_instance_double_click(instance=pred_inst)

    # Check new instance
    assert len(lf.instances) == 3
    new_inst = lf.instances[-1]
    assert new_inst.from_predicted is pred_inst
    assert np.array_equal(new_inst.numpy(), pred_inst.numpy())  # No offset

    # Case 2: Using Ctrl + I (or menu "Add Instance" button)

    # Connect the action to a slot
    add_instance_menu_action = main_window._menu_actions["add instance"]
    triggered = False

    def on_triggered():
        nonlocal triggered
        triggered = True

    add_instance_menu_action.triggered.connect(on_triggered)

    # Find which instance we are going to copy from
    (
        copy_instance,
        from_predicted,
        from_prev_frame,
    ) = AddInstance.find_instance_to_copy_from(
        context, copy_instance=None, init_method="best"
    )

    # Click on the menu action
    assert len(lf.instances) == 3
    add_instance_menu_action.trigger()
    assert triggered, "Action not triggered"

    # Check new instance
    assert len(lf.instances) == 4
    new_inst = lf.instances[-1]
    offset = 10
    np.nan_to_num(new_inst.numpy() - copy_instance.numpy(), nan=offset)
    assert np.all(
        np.nan_to_num(new_inst.numpy() - copy_instance.numpy(), nan=offset) == offset
    )

    # Case 3: Using right click and "Default" option

    # Find which instance we are going to copy from
    (
        copy_instance,
        from_predicted,
        from_prev_frame,
    ) = AddInstance.find_instance_to_copy_from(
        context, copy_instance=None, init_method="best"
    )

    video_player = main_window.player
    right_click_location_x = video.shape[2] / 2
    right_click_location_y = video.shape[1] / 2
    right_click_location = QtCore.QPointF(
        right_click_location_x, right_click_location_y
    )
    video_player.create_contextual_menu(scene_pos=right_click_location)
    default_action = video_player._menu_actions["Default"]
    default_action.trigger()

    # Check new instance
    assert len(lf.instances) == 5
    new_inst = lf.instances[-1]
    reference_node_idx = np.where(
        np.all(
            new_inst.numpy()
            == np.array([right_click_location_x, right_click_location_y]),
            axis=1,
        )
    )[0]
    offset = (
        new_inst.numpy()[reference_node_idx] - copy_instance.numpy()[reference_node_idx]
    )
    diff = np.nan_to_num(new_inst.numpy() - copy_instance.numpy(), nan=offset)
    assert np.all(diff == offset)


def test_ExportLabelsSubset(
    tmp_path, centered_pair_predictions: Labels, small_robot_mp4_vid: Video
):
    """Test that exporting a subset of labels works as expected."""
    # Get the data
    labels = centered_pair_predictions
    n_labels_original = len(labels.labeled_frames)
    video: Video = labels.videos[0]

    # Select subset of frames
    from sleap.sleap_io_adaptors.video_utils import video_get_frames

    n_frames = video_get_frames(video)
    lower_bound = int(n_frames / 4)
    upper_bound = int(n_frames / 4 + 2)

    # Alter data.
    add_suggestion(labels, video, lower_bound)  # Should be included.
    add_suggestion(labels, video, 1)  # Should be excluded since outside video clip.
    add_suggestion(labels, video, upper_bound + 1)  # Should be excluded.
    video_extra = small_robot_mp4_vid
    from sleap.sleap_io_adaptors.lf_labels_utils import labels_add_video

    labels_add_video(
        labels, video_extra
    )  # Should be excluded since outside video clip.
    add_suggestion(
        labels, video_extra, 0
    )  # Should be excluded since outside video clip
    n_suggestions_original = len(labels.suggestions)
    n_videos_original = len(labels.videos)

    # Path to save the exported labels
    name_to_export = "export_labels_subset.slp"
    path_to_export = Path(tmp_path).with_name(name_to_export)
    video_path_to_export = path_to_export.as_posix() + ".mp4"
    video_name_to_export = name_to_export + ".mp4"

    # Set-up command context
    context = CommandContext.from_labels(labels)
    context.state["labels"] = labels
    context.state["video"] = video
    context.state["frame_range"] = (lower_bound, upper_bound)
    context.state["has_frame_range"] = True

    # 1. Mimick ExportDatasetWithImages.ask() method
    def ExportFullPackage_ask(context, params):
        """No GUI version of `ExportVideoClip.ask`."""
        as_package = params["as_package"]
        params["filename"] = (
            path_to_export.with_suffix(".pkg.slp")
            if as_package
            else path_to_export.as_posix()
        )
        params["verbose"] = False
        return True

    ExportFullPackage.ask = ExportFullPackage_ask

    # 2. Mimick ExportVideoClip.ask() method
    def ExportVideoClip_ask(context, params):
        """No GUI version of `ExportVideoClip.ask`."""
        params["video_filename"] = video_path_to_export
        params["fps"] = 30
        params["open_when_done"] = False
        params["frames"] = range(lower_bound, upper_bound)
        params["scale"] = 1.0
        params["background"] = None
        params["crop"] = None
        params["gui_progress"] = False
        return True

    ExportVideoClip.ask = ExportVideoClip_ask

    # Case 1: Export labels as slp and trimmed video
    context.exportLabelsSubset(as_package=False, open_new_project=False)

    # Verify the slp file.
    assert path_to_export.exists()
    assert path_to_export.is_file()
    assert path_to_export.name == name_to_export
    labels_subset = labels_load_file(path_to_export.as_posix())
    # Should only contain video from selected clip.
    assert len(labels_subset.videos) == 1
    # Should only contain frames from selected clip.
    n_frames_expected = upper_bound - lower_bound
    assert len(labels_subset.labeled_frames) <= n_frames_expected
    # Labels are shifted since reference trimmed video.
    assert (
        max([lf.frame_idx for lf in labels_subset.labeled_frames]) < n_frames_expected
    )
    assert min([lf.frame_idx for lf in labels_subset.labeled_frames]) >= 0

    # Verify suggestions were pruned.
    video_subset = labels_subset.videos[0]
    assert len(labels_subset.suggestions) == 1
    assert labels_subset.suggestions[0].video == video_subset

    # Verify the video file.
    assert Path(video_path_to_export).exists()
    assert Path(video_path_to_export).is_file()
    assert Path(video_path_to_export).name == video_name_to_export
    assert video_subset.filename == video_path_to_export
    assert video_get_frames(video_subset) == n_frames_expected

    # Do not mutate original labels.
    assert len(labels.labeled_frames) == n_labels_original
    assert len(labels.videos) == n_videos_original
    assert len(labels.suggestions) == n_suggestions_original

    # Case 2: Export labels as pkg.slp
    context.exportLabelsSubset(as_package=True, open_new_project=False)

    # Verify the slp file.
    path_to_export = Path(path_to_export.with_suffix(".pkg.slp"))
    assert path_to_export.exists()
    assert path_to_export.is_file()
    labels_subset: Labels = labels_load_file(path_to_export.as_posix())
    # Should only contain video from selected clip.
    assert len(labels_subset.videos) == 1
    n_frames_expected = upper_bound - lower_bound
    assert len(labels_subset.labeled_frames) <= n_frames_expected
    # Labels are not shifted since reference original video.
    assert max([lf.frame_idx for lf in labels_subset.labeled_frames]) < upper_bound
    assert min([lf.frame_idx for lf in labels_subset.labeled_frames]) >= lower_bound

    # Verify suggestions were pruned.
    video_subset = labels_subset.videos[0]
    assert len(labels_subset.suggestions) == 1
    assert labels_subset.suggestions[0].video == video_subset

    # Videos in package reference pkg.slp. filename.
    assert video_subset.filename == path_to_export.as_posix()
    assert video_get_frames(video_subset) <= n_frames_expected + len(
        labels_subset.suggestions
    )

    # Do not mutate original labels.
    assert len(labels.labeled_frames) == n_labels_original
    assert len(labels.videos) == n_videos_original
    assert len(labels.suggestions) == n_suggestions_original


def test_remove_video_uses_identity_not_content_matching(centered_pair_predictions):
    """Test that remove_video only removes frames from the target video.

    This is a regression test for GitHub issue #2534. The bug was that remove_video
    used matches_content() which compares video shape (resolution + frame count),
    not object identity. This caused frames from ALL videos with the same shape
    to be deleted when removing any one of them.

    The fix uses identity comparison (is) instead of content matching.
    """
    from sleap.sleap_io_adaptors.lf_labels_utils import (
        remove_video,
        labels_copy,
        labels_add_video,
        add_suggestion,
    )
    from sleap_io import Video, LabeledFrame, Instance

    # Create a copy of the labels to avoid mutating the fixture
    labels = labels_copy(centered_pair_predictions)
    original_video = labels.videos[0]

    # Create additional videos - they don't need actual files since we're
    # testing object identity, not content matching
    # Note: The bug was that videos with the same shape would all be deleted
    # when any one was removed. With the fix, only the specific video is removed.
    video2 = Video(filename="fake_video_2.mp4")
    video3 = Video(filename="fake_video_3.mp4")

    # Add the new videos to labels
    labels_add_video(labels, video2)
    labels_add_video(labels, video3)

    # Count original frames for video1
    original_frames_video1 = len([lf for lf in labels if lf.video is original_video])

    # Create labeled frames for video2 and video3
    skeleton = labels.skeleton
    for frame_idx in [0, 5, 10]:
        # Add frame to video2
        lf2 = LabeledFrame(video=video2, frame_idx=frame_idx)
        inst2 = Instance(points=Instance.empty(skeleton).points, skeleton=skeleton)
        lf2.instances.append(inst2)
        labels.append(lf2)

        # Add frame to video3
        lf3 = LabeledFrame(video=video3, frame_idx=frame_idx)
        inst3 = Instance(points=Instance.empty(skeleton).points, skeleton=skeleton)
        lf3.instances.append(inst3)
        labels.append(lf3)

    # Add suggestions for each video
    add_suggestion(labels, original_video, 50)
    add_suggestion(labels, video2, 50)
    add_suggestion(labels, video3, 50)

    # Verify setup
    assert len(labels.videos) == 3
    frames_for_video1 = [lf for lf in labels if lf.video is original_video]
    frames_for_video2 = [lf for lf in labels if lf.video is video2]
    frames_for_video3 = [lf for lf in labels if lf.video is video3]
    assert len(frames_for_video1) == original_frames_video1
    assert len(frames_for_video2) == 3
    assert len(frames_for_video3) == 3

    suggestions_video1 = [s for s in labels.suggestions if s.video is original_video]
    suggestions_video2 = [s for s in labels.suggestions if s.video is video2]
    suggestions_video3 = [s for s in labels.suggestions if s.video is video3]
    assert len(suggestions_video1) == 1
    assert len(suggestions_video2) == 1
    assert len(suggestions_video3) == 1

    # Remove video2 - this is where the bug would occur
    # Before the fix: ALL frames with matching shape would be deleted (catastrophic)
    # After the fix: ONLY frames from video2 should be deleted
    remove_video(labels, video2)

    # Verify only video2 was removed
    assert len(labels.videos) == 2
    assert original_video in labels.videos
    assert video2 not in labels.videos
    assert video3 in labels.videos

    # Verify ONLY video2's frames were removed (this is the critical test)
    remaining_frames_video1 = [lf for lf in labels if lf.video is original_video]
    remaining_frames_video2 = [lf for lf in labels if lf.video is video2]
    remaining_frames_video3 = [lf for lf in labels if lf.video is video3]

    assert len(remaining_frames_video1) == original_frames_video1, (
        "Frames from video1 should NOT have been deleted! "
        "This indicates the bug where matches_content() was used instead of identity."
    )
    assert len(remaining_frames_video2) == 0, "All frames from video2 should be removed"
    assert len(remaining_frames_video3) == 3, (
        "Frames from video3 should NOT have been deleted! "
        "This indicates the bug where matches_content() was used instead of identity."
    )

    # Verify only video2's suggestions were removed
    remaining_suggestions_video1 = [
        s for s in labels.suggestions if s.video is original_video
    ]
    remaining_suggestions_video2 = [s for s in labels.suggestions if s.video is video2]
    remaining_suggestions_video3 = [s for s in labels.suggestions if s.video is video3]

    assert len(remaining_suggestions_video1) == 1, (
        "Suggestions for video1 should NOT have been deleted!"
    )
    assert len(remaining_suggestions_video2) == 0, (
        "Suggestions for video2 should be removed"
    )
    assert len(remaining_suggestions_video3) == 1, (
        "Suggestions for video3 should NOT have been deleted!"
    )
