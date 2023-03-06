from pathlib import PurePath, Path
import shutil
import sys
from typing import List

import pytest
from qtpy.QtWidgets import QComboBox

from sleap import Skeleton, Track
from sleap.gui.commands import (
    CommandContext,
    ImportDeepLabCutFolder,
    ExportAnalysisFile,
    ReplaceVideo,
    OpenSkeleton,
    SaveProjectAs,
    get_new_version_filename,
)
from sleap.instance import Instance, LabeledFrame
from sleap.io.convert import default_analysis_filename
from sleap.io.dataset import Labels
from sleap.io.format.adaptor import Adaptor
from sleap.io.format.ndx_pose import NDXPoseAdaptor
from sleap.io.pathutils import fix_path_separator
from sleap.io.video import Video
from sleap.util import get_package_file

# These imports cause trouble when running `pytest.main()` from within the file
# Comment out to debug tests file via VSCode's "Debug Python File"
from tests.info.test_h5 import extract_meta_hdf5
from tests.io.test_video import assert_video_params
from tests.io.test_formats import read_nix_meta


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


@pytest.mark.parametrize("out_suffix", ["h5", "nix"])
def test_ExportAnalysisFile(
    centered_pair_predictions: Labels,
    small_robot_mp4_vid: Video,
    out_suffix: str,
    tmpdir,
):
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

        # Check for labeled frames in each video
        videos = [video for video in all_videos if len(labels.get(video)) != 0]
        if len(videos) == 0:
            raise ValueError("No labeled frames in video(s). Nothing to export.")

        default_name = context.state["filename"] or "labels"
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

            if labels_path is not None:
                meta_reader = extract_meta_hdf5 if out_suffix == "h5" else read_nix_meta
                labels_key = "labels_path" if out_suffix == "h5" else "project"
                read_meta = meta_reader(output_path, dset_names_in=["labels_path"])
                assert read_meta[labels_key] == labels_path

        assert len(output_paths) == num_videos, "Wrong number of outputs written"
        assert len(set(output_paths)) == num_videos, "Some output paths overwritten"

    tmpdir = Path(tmpdir)

    labels = centered_pair_predictions.copy()
    context = CommandContext.from_labels(labels)
    context.state["filename"] = None

    # Test with all_videos False (single video)
    params = {"all_videos": False}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay == True
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add labels path and test with all_videos True (single video)
    context.state["filename"] = str(tmpdir.with_name("path.to.labels"))
    params = {"all_videos": True}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay == True
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add a video (no labels) and test with all_videos True
    labels.add_video(small_robot_mp4_vid)

    params = {"all_videos": True}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay == True
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add labels and test with all_videos False
    labeled_frame = labels.find(video=labels.videos[1], frame_idx=0, return_new=True)[0]
    instance = Instance(skeleton=labels.skeleton, frame=labeled_frame)
    labels.add_instance(frame=labeled_frame, instance=instance)
    labels.append(labeled_frame)

    params = {"all_videos": False}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay == True
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Add specific video and test with all_videos False
    context.state["videos"] = labels.videos[1]

    params = {"all_videos": False}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay == True
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=1, labels_path=context.state["filename"])

    # Test with all videos True
    params = {"all_videos": True}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay == True
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=2, labels_path=context.state["filename"])

    # Test with videos with the same filename
    (tmpdir / "session1").mkdir()
    (tmpdir / "session2").mkdir()
    shutil.copy(
        centered_pair_predictions.video.backend.filename,
        tmpdir / "session1" / "video.mp4",
    )
    shutil.copy(small_robot_mp4_vid.backend.filename, tmpdir / "session2" / "video.mp4")

    labels.videos[0].backend.filename = str(tmpdir / "session1" / "video.mp4")
    labels.videos[1].backend.filename = str(tmpdir / "session2" / "video.mp4")

    params = {"all_videos": True}
    okay = ExportAnalysisFile_ask(context=context, params=params)
    assert okay == True
    ExportAnalysisFile.do_action(context=context, params=params)
    assert_videos_written(num_videos=2, labels_path=context.state["filename"])

    # Remove all videos and test
    all_videos = list(labels.videos)
    for video in all_videos:
        labels.remove_video(labels.videos[-1])

    params = {"all_videos": True}
    with pytest.raises(ValueError):
        okay = ExportAnalysisFile_ask(context=context, params=params)


def test_ToggleGrayscale(centered_pair_predictions: Labels):
    """Test functionality for ToggleGrayscale on mp4/avi video"""
    labels = centered_pair_predictions
    video = labels.video
    grayscale = video.backend.grayscale
    filename = video.backend.filename

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
        lfs: List[LabeledFrame] = list(labels.get(videos[0]))
        lfs.sort(key=lambda lf: lf.frame_idx)
        return lfs[-1].frame_idx

    def replace_video(
        new_video: Video, videos_to_replace: List[Video], context: CommandContext
    ):
        # Video to be imported
        new_video_filename = new_video.backend.filename

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
    last_lf_frame = get_last_lf_in_video(labels, videos[0])

    # Replace the video
    new_video_filename = replace_video(small_robot_mp4_vid, videos, context)

    # Ensure video backend was replaced
    video = labels.video
    assert len(labels.videos) == 1
    assert video.backend.grayscale == True
    assert video.backend.filename == new_video_filename

    # Ensure labels were truncated (Original video was fully labeled)
    new_last_lf_frame = get_last_lf_in_video(labels, video)
    # Original video was fully labeled
    assert new_last_lf_frame == labels.video.last_frame_idx

    # Attempt to replace an mp4 with an hdf5 video
    with pytest.raises(TypeError):
        replace_video(hdf5_vid, labels.videos, context)


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
    params = {"adaptor": NDXPoseAdaptor()}
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
        # Node names match
        assert len(set(new_skeleton.nodes) - set(skeleton.nodes))
        # Edges match
        for (new_src, new_dst), (src, dst) in zip(new_skeleton.edges, skeleton.edges):
            assert new_src.name == src.name
            assert new_dst.name == dst.name

        # Symmetries match
        for (new_src, new_dst), (src, dst) in zip(
            new_skeleton.symmetries, skeleton.symmetries
        ):
            assert new_src.name == src.name
            assert new_dst.name == dst.name

    def OpenSkeleton_ask(context: CommandContext, params: dict) -> bool:
        """Implement `OpenSkeleton.ask` without GUI elements."""
        template = (
            context.app.currentText
        )  # Original function uses `QComboBox.currentText()`
        if template == "Custom":
            # Original function opens FileDialog here
            filename = params["filename_in"]
        else:
            filename = get_package_file(f"sleap/skeletons/{template}.json")
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
    context = CommandContext.from_labels(labels)
    context.app.__setattr__("currentText", "Custom")
    # Add multiple skeletons to and ensure the unused skeleton is removed
    labels.skeletons.append(stickman)

    # Run without OpenSkeleton.ask()
    params = {"filename": fly_legs_skeleton_json}
    new_skeleton = OpenSkeleton.load_skeleton(fly_legs_skeleton_json)
    new_skeleton.add_symmetry(new_skeleton.nodes[0], new_skeleton.nodes[1])
    OpenSkeleton.do_action(context, params)
    assert len(labels.skeletons) == 1

    # State is updated
    assert context.state["skeleton"] == skeleton

    # Structure is identical
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
    fly32_json = get_package_file(f"sleap/skeletons/fly32.json")
    OpenSkeleton_ask(context, params)
    assert params["filename"] == fly32_json
    fly32_skeleton = Skeleton.load_json(fly32_json)
    OpenSkeleton.do_action(context, params)
    assert_skeletons_match(labels.skeleton, fly32_skeleton)


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
    labels.remove_all_tracks()

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
    tracks.append(Track(name="unused", spawned_on=0))
    assert len(tracks) == 3

    # Set-up command context
    context: CommandContext = CommandContext.from_labels(labels)
    context.state["labels"] = labels

    # Delete unused tracks
    context.deleteMultipleTracks(delete_all=False)
    assert len(labels.tracks) == 2

    # Add back an unused track and delete all tracks
    tracks.append(Track(name="unused", spawned_on=0))
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
        assert lf_to_paste.instances[-1].points == instance.points

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
        assert lf_to_paste.instances[-1].points == instance.points
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
        assert len(lf_to_paste.instances) == len(instances_checkpoint) + 1
        assert lf_to_paste.instances[-1].points == instance.points
        assert lf_to_paste.instances[-1].track == instance.track
        assert lf_to_paste in labels.labeled_frames

    lf_to_paste = labels.get((labels.video, 3))
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
    assert instance_to_paste.track != instance.track
    assert instance_with_same_track.track == instance.track

    context.pasteInstanceTrack()
    assert instance_to_paste.track == instance.track
    assert instance_with_same_track.track != instance.track

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
        gui_video_callback = Labels.make_video_callback(
            search_paths=[str(filename)], context=params
        )
        labels = Labels.load_file(
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
    labels = Labels.load_file(centered_pair_predictions_slp_path)
    expected_video_path = Path(labels.video.backend.filename)

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
