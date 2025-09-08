"""
Standalone utility functions for working with Labels and LabeledFrame objects.
"""

import math
import copy
from typing import List, Dict, Optional, Text, Union

from pathlib import Path
import cattr
import os

from sleap import util
from sleap_io import (
    Video,
    load_file,
    Labels,
    Track,
    Instance,
    LabeledFrame,
    SuggestionFrame,
    Skeleton,
    PredictedInstance,
)
from sleap_io.model.matching import SkeletonMatcher
from sleap_io.model.instance import PointsArray
from sleap.util import weak_filename_match
from sleap.gui.dialogs.missingfiles import MissingFilesDialog

# For debugging, we can replace missing video files with a "dummy" video
USE_DUMMY_FOR_MISSING_VIDEOS = os.getenv("SLEAP_USE_DUMMY_VIDEOS", default="")


# Create a simple range object with start, end, and list properties
class SimpleRange:
    def __init__(self, ranges_list):
        self.list = ranges_list
        self.start = ranges_list[0][0] if ranges_list else None
        self.end = ranges_list[-1][1] if ranges_list else None

    def is_empty(self):
        return len(self.list) == 0


def find_path_using_paths(filename: str, search_paths: List[str]) -> str:
    """Find a file in the given search paths.

    Args:
        filename: The filename to search for.
        search_paths: List of directories to search in.

    Returns:
        The found path or the original filename if not found.
    """
    filename_path = Path(filename)

    for search_path in search_paths:
        search_path_obj = Path(search_path)
        if search_path_obj.is_dir():
            potential_path = search_path_obj / filename_path.name
            if potential_path.exists():
                return str(potential_path)

    return filename


def remove_track(labels: Labels, track: Track):
    """Remove a track from the labels dataset and update all related instances.

    This function removes the specified track from the labels dataset by:
    1. Setting the track to None for all instances that were using this track
    2. Removing the track from the labels.tracks list

    Args:
        labels: The Labels object containing the dataset to modify
        track: The Track object to remove from the dataset

    Returns:
        Labels: The modified labels object (same object, modified in-place)
    """
    for lf in labels:
        for instance in lf.instances:
            if track.matches(instance.track):
                instance.track = None

    tracks = []
    for t in labels.tracks:
        if not track.matches(t):
            tracks.append(t)

    labels.tracks = tracks
    return labels


def remove_all_tracks(labels: Labels):
    """Remove all tracks from the labels dataset and update all related instances."""
    for lf in labels:
        for instance in lf.instances:
            instance.track = None
    labels.tracks = []
    return labels


def remove_frames(labels: Labels, frames: List[LabeledFrame]):
    """Remove a list of frames from the labels dataset."""
    for lf in frames:
        for lf_idx, lab_fr in enumerate(labels):
            if (
                lab_fr.video.matches_content(lf.video)
                and lab_fr.frame_idx == lf.frame_idx
            ):
                labels_pop(labels, lf_idx)
    labels.update()


def remove_instance(labels: Labels, instance: Instance, lf: LabeledFrame):
    """Remove an instance from a labeled frame and update all related instances."""
    import numpy as np

    lf_inst_to_remove = labels.find(video=lf.video, frame_idx=lf.frame_idx)[0]
    if lf_inst_to_remove:
        # Iterate backwards to safely remove from list
        for inst_idx in range(len(lf_inst_to_remove.instances) - 1, -1, -1):
            inst = lf_inst_to_remove.instances[inst_idx]

            # Compare instances using numpy arrays with NaN handling
            points_match = np.array_equal(
                inst.numpy(), instance.numpy(), equal_nan=True
            )

            if points_match:
                # Also check track matching
                tracks_match = False
                if inst.track is not None and instance.track is not None:
                    tracks_match = inst.track.matches(instance.track)
                elif inst.track is None and instance.track is None:
                    tracks_match = True

                if tracks_match:
                    lf_inst_to_remove.instances.pop(inst_idx)
                    break  # Only remove first match


def remove_unused_tracks(labels: Labels):
    """Remove all tracks from the labels dataset that are not used by any instances."""
    if len(labels.tracks) == 0:
        return

    # Check which tracks are used by instances
    all_tracks = set([track.name for track in labels.tracks])
    used_tracks = set()
    for lf in labels:
        for inst in lf.instances:
            used_tracks.add(inst.track.name)

    # Remove set difference from tracks in Labels
    tracks_to_remove = all_tracks - used_tracks
    for track in tracks_to_remove:
        for t_idx, t in enumerate(labels.tracks):
            if t.name == track:
                labels.tracks.pop(t_idx)


def remove_video(labels: Labels, video: Video):
    """Remove a video from the labels dataset and update all related instances."""
    # Remove labeled frames for this video (iterate backwards to avoid index issues)
    for lf_idx in reversed(range(len(labels.labeled_frames))):
        lf = labels.labeled_frames[lf_idx]
        if lf.video.matches_content(video):
            labels_pop(labels, lf_idx)

    # Remove video from videos list (iterate backwards to avoid index issues)
    for vid_idx in reversed(range(len(labels.videos))):
        vid = labels.videos[vid_idx]
        if video == vid:
            labels.videos.pop(vid_idx)

        # Remove any suggestions for this video
        if hasattr(labels, "suggestions"):
            labels.suggestions = [
                s for s in labels.suggestions if not s.video.matches_content(video)
            ]


def get_track_occupancy(labels, video):
    """Get track occupancy information for a specific video.

    This function recreates the functionality of labels.get_track_occupancy(video)
    from the original SLEAP codebase.

    Args:
        labels: A Labels object containing labeled frames and tracks
        video: A Video object to get track occupancy for

    Returns:
        Dict mapping Track objects to their occupancy information (frame ranges)
    """
    track_occupancy = {}

    # Get all labeled frames for this video
    labeled_frames = labels.find(video) if hasattr(labels, "find") else []

    # Build track occupancy dictionary
    for lf in labeled_frames:
        for instance in lf.instances:
            track = instance.track.name if instance.track is not None else None
            if track not in track_occupancy:
                track_occupancy[track] = []

            # Add this frame to the track's occupancy
            track_occupancy[track].append(lf.frame_idx)

    # Convert frame lists to sorted ranges
    for track in track_occupancy:
        if track_occupancy[track]:
            # Sort frame indices
            frames = sorted(track_occupancy[track])

            # Create ranges (consecutive frames)
            ranges = []
            start = frames[0]
            prev = frames[0]

            for frame in frames[1:]:
                if frame != prev + 1:
                    # Gap found, end current range
                    ranges.append((start, prev + 1))
                    start = frame
                prev = frame

            # Add final range
            ranges.append((start, prev + 1))

            track_occupancy[track] = SimpleRange(ranges)

    return track_occupancy


def add_suggestion(labels, video, frame_idx):
    """Add a suggestion to the labels dataset."""
    labels.suggestions.append(SuggestionFrame(video=video, frame_idx=frame_idx))


def get_video_suggestions(labels, video, user_labeled: bool = True) -> List[int]:
    """Get suggested frame indices for a specific video.

    This function recreates the functionality of labels.get_video_suggestions(video)
    from the original SLEAP codebase.

    Args:
        labels: A Labels object containing labeled frames and suggestions
        video: A Video object to get suggestions for
        user_labeled: If True (the default), return frame indices for suggestions
            that already have user labels. If False, only suggestions with no user
            labeled instances will be returned.

    Returns:
        List of frame indices that are suggested for the specified video.
    """
    frame_indices = []

    # Check if labels has a suggestions attribute
    if not hasattr(labels, "suggestions"):
        return frame_indices

    # Get suggestions for this video
    for suggestion in labels.suggestions:
        if suggestion.video == video:
            fidx = suggestion.frame_idx

            # If user_labeled is False, skip suggestions that already have user labels
            if not user_labeled:
                lf = labels.find((video, fidx))
                if (
                    lf is not None
                    and hasattr(lf, "has_user_instances")
                    and lf.has_user_instances
                ):
                    continue

            frame_indices.append(fidx)

    return frame_indices


def get_unused_predictions(labeled_frame) -> List:
    """Return a list of "unused" PredictedInstance objects in frame.

    This function recreates the functionality of labeled_frame.unused_predictions
    from the original SLEAP codebase.

    This is all the PredictedInstance objects which do not have
    a corresponding Instance in the same track in frame.

    Args:
        labeled_frame: A LabeledFrame object containing instances

    Returns:
        List of unused PredictedInstance objects
    """
    unused_predictions = []

    # Check if labeled_frame has instances attribute
    if not hasattr(labeled_frame, "instances"):
        return unused_predictions

    # Get all instances from the frame
    instances = labeled_frame.instances if hasattr(labeled_frame, "instances") else []

    any_tracks = [
        inst.track
        for inst in instances
        if hasattr(inst, "track") and inst.track is not None
    ]

    if len(any_tracks):
        # Use tracks to determine which predicted instances have been used
        used_tracks = [
            inst.track
            for inst in instances
            if hasattr(inst, "track")
            and inst.track is not None
            and not hasattr(inst, "from_predicted")
        ]
        unused_predictions = [
            inst
            for inst in instances
            if hasattr(inst, "track")
            and inst.track not in used_tracks
            and hasattr(inst, "from_predicted")
        ]
    else:
        # Use from_predicted to determine which predicted instances have been used
        used_instances = [
            inst.from_predicted
            for inst in instances
            if hasattr(inst, "from_predicted") and inst.from_predicted is not None
        ]
        unused_predictions = [
            inst
            for inst in instances
            if hasattr(inst, "from_predicted") and inst not in used_instances
        ]

    return unused_predictions


def get_instances_to_show(labeled_frame) -> List:
    """Return a list of instances to show in GUI for this frame.

    This function recreates the functionality of labeled_frame.instances_to_show
    from the original SLEAP codebase.

    This list will not include any predicted instances for which
    there's a corresponding regular instance.

    Args:
        labeled_frame: A LabeledFrame object containing instances

    Returns:
        List of instances to show in GUI.
    """
    unused_predictions = get_unused_predictions(labeled_frame)

    # Check if labeled_frame has instances attribute
    if not hasattr(labeled_frame, "instances"):
        return []

    instances = labeled_frame.instances if hasattr(labeled_frame, "instances") else []

    inst_to_show = [
        inst
        for inst in instances
        if not hasattr(inst, "from_predicted") or inst in unused_predictions
    ]

    return inst_to_show


def get_labeled_frame_count(labels, video=None, filter: str = "") -> int:
    """Return count of frames matching video/filter.

    This function recreates labels.get_labeled_frame_count(video, filter)
    from the original SLEAP codebase.

    Args:
        labels: A Labels object containing labeled frames
        video: Optional Video object to filter by. If None, counts all videos
        filter: Filter string. Must be one of: "", "user", "predicted"
            - "": All labeled frames
            - "user": Only frames with user-labeled instances
            - "predicted": Only frames with predicted instances

    Returns:
        Count of frames matching the criteria

    Raises:
        ValueError: If filter is not one of the valid options
    """
    if filter not in ("", "user", "predicted"):
        raise ValueError(f"get_labeled_frame_count() invalid filter: {filter}")

    # Get all labeled frames
    if hasattr(labels, "labeled_frames"):
        all_frames = labels.labeled_frames
    elif hasattr(labels, "__iter__"):
        # If labels is iterable, use it directly
        all_frames = list(labels)
    else:
        return 0

    # Apply video filter
    if video is not None:
        frames = [lf for lf in all_frames if hasattr(lf, "video") and lf.video == video]
    else:
        frames = all_frames

    # Apply type filter
    if filter == "":
        # All labeled frames
        return len(frames)
    elif filter == "user":
        # Only frames with user instances
        return len(
            [
                lf
                for lf in frames
                if hasattr(lf, "has_user_instances") and lf.has_user_instances
            ]
        )
    elif filter == "predicted":
        # Only frames with predicted instances
        return len(
            [
                lf
                for lf in frames
                if hasattr(lf, "has_predicted_instances") and lf.has_predicted_instances
            ]
        )

    return 0


def find_first(labels, video, frame_idx=None, use_cache: bool = False):
    """Find the first occurrence of a matching labeled frame.

    This function recreates labels.find_first(video, frame_idx, use_cache)
    from the original SLEAP codebase.

    Matches on frames for the given video and/or frame index.

    Args:
        labels: A Labels object containing labeled frames
        video: A Video instance that is associated with the labeled frames
        frame_idx: An integer specifying the frame index within the video (optional)
        use_cache: Boolean that determines whether to use cache. If True, use the labels
            data cache, else loop through all labels to search.

    Returns:
        First LabeledFrame that matches the criteria or None if none were found.
    """
    if use_cache and hasattr(labels, "find"):
        # Use cache if available
        label = labels.find(video=video, frame_idx=frame_idx)
        return None if len(label) == 0 else label[0]
    else:
        # Check if video is in labels
        if hasattr(labels, "videos") and video in labels.videos:
            # Loop through all labels
            for label in labels:
                if (
                    hasattr(label, "video")
                    and label.video == video
                    and (
                        frame_idx is None
                        or (
                            hasattr(label, "frame_idx") and label.frame_idx == frame_idx
                        )
                    )
                ):
                    return label
        return None


def find_last(labels, video, frame_idx=None):
    """Find the last occurrence of a matching labeled frame.

    This function recreates the functionality of labels.find_last(video, frame_idx)
    from the original SLEAP codebase.

    Matches on frames for the given video and/or frame index.

    Args:
        labels: A Labels object containing labeled frames
        video: A Video instance that is associated with the labeled frames
        frame_idx: An integer specifying the frame index within the video (optional)

    Returns:
        Last LabeledFrame that matches the criteria or None if none were found.
    """
    # Check if video is in labels
    if hasattr(labels, "videos") and video in labels.videos:
        # Loop through all labels in reverse order
        for label in reversed(list(labels)):
            if (
                hasattr(label, "video")
                and label.video == video
                and (
                    frame_idx is None
                    or (hasattr(label, "frame_idx") and label.frame_idx == frame_idx)
                )
            ):
                return label
    return None


def load_and_match(filename: str, match_to: Labels):
    # Load the Labels file
    labels: Labels = load_file(filename)

    # if we're given a Labels object to match, use its objects when they match
    if match_to is not None:
        if len(labels.skeletons) > 1 or len(match_to.skeletons) > 1:
            # Match skeletons by name
            nodes = labels.skeleton.nodes
            for idx, sk in enumerate(labels.skeletons):
                for old_sk in match_to.skeletons:
                    if SkeletonMatcher.match(sk, old_sk):
                        # use nodes from matched skeleton
                        for node, match_node in zip(sk.nodes, old_sk.nodes):
                            node_idx = nodes.index(node)
                            nodes[node_idx] = match_node
                        # use skeleton from matched skeleton
                        labels.skeletons[idx] = old_sk
                        break
        elif len(labels.skeletons) == 1 and len(match_to.skeletons) == 1:
            # Match by node names
            old_skel = match_to.skeleton
            old_node_names = old_skel.node_names
            nodes = labels.skeleton.nodes
            for i, node in enumerate(nodes):
                if node.name in old_node_names:
                    nodes[i] = old_skel.nodes[old_node_names.index(node.name)]
            labels.skeletons[0] = old_skel

        # Match Videos
        for idx, vid in enumerate(labels.videos):
            for old_vid in match_to.videos:
                # Try to match videos using either their current or source filename
                # if available.
                old_vid_paths = [old_vid.filename]
                if getattr(old_vid.backend, "has_embedded_images", False):
                    old_vid_paths.append(old_vid.filename)

                new_vid_paths = [vid.filename]
                if getattr(vid.backend, "has_embedded_images", False):
                    new_vid_paths.append(vid.filename)

                is_match = False
                for old_vid_path in old_vid_paths:
                    for new_vid_path in new_vid_paths:
                        if old_vid_path == new_vid_path or weak_filename_match(
                            old_vid_path, new_vid_path
                        ):
                            is_match = True
                            labels.videos[idx] = old_vid
                            break
                    if is_match:
                        break
                if is_match:
                    break

    return labels


def iterate_labeled_frames(
    labels, video, from_frame_idx: int = -1, reverse: bool = False
):
    """Return an iterator over lfs in a video with start pos (opt) and order control.

    This function recreates Labels.frames() from the original SLEAP codebase.

    Args:
        labels: A Labels object containing labeled frames
        video: A Video object that is associated with the project
        from_frame_idx: The frame index from which to start (default: -1 for beginning)
        reverse: Whether to iterate over frames in reverse order (default: False)

    Yields:
        LabeledFrame objects for the specified video
    """
    # Get all labeled frames for this video
    labeled_frames = labels.find(video) if hasattr(labels, "find") else []

    if not labeled_frames:
        return

    # Extract frame indices and sort them
    frame_idxs = sorted(
        [lf.frame_idx for lf in labeled_frames if hasattr(lf, "frame_idx")]
    )

    if not frame_idxs:
        return

    # Handle the case where from_frame_idx is -1 (start from beginning)
    if from_frame_idx == -1:
        if reverse:
            frame_idxs = frame_idxs[::-1]  # Reverse the list
        # Use the frame_idxs as is for forward direction

    else:
        # Find the next frame index after/before the specified frame
        if not reverse:
            # Forward direction: find next frame after from_frame_idx
            next_frame_idx = None
            for idx in frame_idxs:
                if idx > from_frame_idx:
                    next_frame_idx = idx
                    break
            if next_frame_idx is None:
                next_frame_idx = frame_idxs[0]  # Wrap to beginning
        else:
            # Reverse direction: find previous frame before from_frame_idx
            next_frame_idx = None
            for idx in reversed(frame_idxs):
                if idx < from_frame_idx:
                    next_frame_idx = idx
                    break
            if next_frame_idx is None:
                next_frame_idx = frame_idxs[-1]  # Wrap to end

        # Find the position of the next frame in the list
        try:
            cut_list_idx = frame_idxs.index(next_frame_idx)
        except ValueError:
            # If not found, use original order
            if reverse:
                frame_idxs = frame_idxs[::-1]
            return

        # Reorder the list to start from the specified position
        if reverse:
            # For reverse, we need to handle the reordering differently
            reordered = frame_idxs[cut_list_idx:] + frame_idxs[:cut_list_idx]
            frame_idxs = reordered[::-1]  # Reverse the reordered list
        else:
            # For forward, just reorder normally
            frame_idxs = frame_idxs[cut_list_idx:] + frame_idxs[:cut_list_idx]

    # Create a mapping from frame_idx to LabeledFrame for quick lookup
    frame_map = {lf.frame_idx: lf for lf in labeled_frames if hasattr(lf, "frame_idx")}

    # Yield the frames in the order specified by frame_idxs
    for idx in frame_idxs:
        if idx in frame_map:
            yield frame_map[idx]


def get_template_instance_points(labels: Labels, skeleton: Skeleton):
    """Get template instance points for a skeleton.

    This function recreates labels.get_template_instance_points(skeleton)
    from the original SLEAP codebase.

    Args:
        labels: A Labels object containing labeled frames and instances
        skeleton: A Skeleton object to get template points for

    Returns:
        numpy array of template points for the skeleton
    """
    import itertools
    import numpy as np

    # Check if labels has labeled_frames attribute
    if not hasattr(labels, "labeled_frames"):
        print("Labels object has no labeled_frames attribute")
        return None

    # Check if labeled_frame list is empty
    if not labels.labeled_frames:
        # No labeled frames so use force-directed graph layout
        try:
            import networkx as nx
            from sleap.sleap_io_adaptors.skeleton_utils import to_graph

            # Create graph from skeleton and get spring layout
            G = to_graph(skeleton)
            node_positions = nx.spring_layout(G=G, scale=50)

            # Create template points from node positions
            template_points = np.stack(
                [
                    (
                        node_positions[node]
                        if node in node_positions
                        else np.random.randint(0, 50, size=2)
                    )
                    for node in skeleton.nodes
                ]
            )

            return template_points

        except ImportError:
            # Fallback if networkx is not available
            template_points = np.random.randint(0, 50, size=(len(skeleton.nodes), 2))
            return template_points

    # Check if there are any instances
    if not hasattr(labels, "instances") or not instances(labels):
        # No instances, use fallback
        template_points = np.random.randint(0, 50, size=(len(skeleton.nodes), 2))
        return template_points

    # Get first 1000 instances for this skeleton
    try:
        from sleap.info import align

        # Get instances for this skeleton
        skeleton_instances = []
        for instance in itertools.islice(
            instances(labels=labels, skeleton=skeleton), 1000
        ):
            if hasattr(instance, "points") and instance.points is not None:
                skeleton_instances.append(instance)

        if skeleton_instances:
            # Get template points from aligned instances
            template_points = align.get_template_points_array(skeleton_instances)
            return template_points
        else:
            # No valid instances, use fallback
            template_points = np.random.randint(0, 50, size=(len(skeleton.nodes), 2))
            return template_points

    except ImportError:
        # Fallback if sleap.info.align is not available
        template_points = np.random.randint(0, 50, size=(len(skeleton.nodes), 2))
        return template_points


def fix_paths_with_saved_prefix(
    filenames,
    missing: Optional[List[bool]] = None,
    path_prefix_conversions: Optional[List[Dict[Text, Text]]] = None,
):
    if path_prefix_conversions is None:
        path_prefix_conversions = util.get_config_yaml("path_prefixes.yaml")

    if path_prefix_conversions is None:
        return

    for i, filename in enumerate(filenames):
        if missing is not None:
            if not missing[i]:
                continue
        elif os.path.exists(filename):
            continue

        for old_prefix, new_prefix in path_prefix_conversions.items():
            if filename.startswith(old_prefix):
                try_filename = filename.replace(old_prefix, new_prefix)

                # Equivalent to fix_path_separator(try_filename)
                try_filename = try_filename.replace("\\", "/")

                if os.path.exists(try_filename):
                    filenames[i] = try_filename
                    if missing is not None:
                        missing[i]
                    continue


def make_video_callback(
    search_paths: Optional[List] = None,
    use_gui: bool = False,
    context: Optional[Dict[str, bool]] = None,
):
    """Adapter function for callback function to finding missing video.

    The callback can be used while loading a saved project and
    allows the user to find videos which have been moved (or have
    paths from a different system).

    The callback function returns True to signal "abort".
    Args:
        search_paths: If specified, this is a list of paths where we'll
            automatically try to find the missing videos.
        context: A dictionary containing a "changed_on_load" key with a boolean
            value. Used externally to determine if any filenames were updated.
    Returns:
        The callback function.

    """
    search_paths = search_paths or []
    context = context or {}

    def video_callback(
        video_list: List[Video],
        new_paths: List[str] = search_paths,
        context: Optional[Dict] = context,
    ):
        """Callback to find videos which have been moved (or moved across systems).
        Args:
            video_list: A list of serialized `Video` objects stored as nested
                dictionaries.
            new_paths: If specified, this is a list of paths where we'll
                automatically try to find the missing videos.
            context: A dictionary containing a "changed_on_load" key with a boolean
                value. Used externally to determine if any filenames were updated.
        """
        filenames = [item.filename for item in video_list]
        context = context or {"changed_on_load": False}

        # Equivalent to pathutils.list_file_missing(filenames)
        missing = [not os.path.exists(filename) for filename in filenames]

        # Try changing the prefix using saved patterns
        if sum(missing):
            fix_paths_with_saved_prefix(filenames, missing)

        # Check for file in search_path dirctories
        if sum(missing) and new_paths:
            for i, filename in enumerate(filenames):
                fixed_path = find_path_using_paths(filename, new_paths)
                if fixed_path != filename:
                    filenames[i] = fixed_path
                    missing[i] = False
                    context["changed_on_load"] = True

        if use_gui:
            # If there are still missing paths, prompt user
            if sum(missing):
                # If we are using dummy for any video not found by user
                # then don't require user to find everything.
                allow_incomplete = USE_DUMMY_FOR_MISSING_VIDEOS

                okay = MissingFilesDialog(
                    filenames, missing, allow_incomplete=allow_incomplete
                ).exec_()

                if not okay:
                    return True  # True for stop

                context["changed_on_load"] = True

        if not use_gui and sum(missing):
            # If we got the same number of paths as there are videos
            if len(filenames) == len(new_paths):
                # and the file extensions match
                exts_match = all(
                    (
                        old.split(".")[-1] == new.split(".")[-1]
                        for old, new in zip(filenames, new_paths)
                    )
                )

                if exts_match:
                    # then the search paths should be a list of all the
                    # video paths, so we can get the new path for the missing
                    # old path.
                    for i, filename in enumerate(filenames):
                        if missing[i]:
                            filenames[i] = new_paths[i]
                            missing[i] = False

                    # Solely for testing since only gui will have a `CommandContext`
                    context["changed_on_load"] = True

        # Replace the video filenames with changes by user
        for i, item in enumerate(video_list):
            item.replace_filename(filenames[i])

        if USE_DUMMY_FOR_MISSING_VIDEOS and sum(missing):
            # Replace any video still missing with "dummy" video
            for is_missing, item in zip(missing, video_list):
                from sleap.io.video import DummyVideo

                vid = DummyVideo(filename=item.filename)
                item["backend"] = cattr.unstructure(vid)

    return video_callback


def load_labels_video_search(filename, video_search):
    labels = load_file(filename)

    if isinstance(video_search, str):
        video_search = [video_search]

    if hasattr(video_search, "__iter__"):
        # If the callback is an iterable, then we'll expect it to be a
        # list of strings and build a non-gui callback with those as
        # the search paths.
        # When path is to a file, use the path of parent directory.
        search_paths = [
            os.path.dirname(path) if os.path.isfile(path) else path
            for path in video_search
        ]

        # Make the search function from list of paths
        video_search = make_video_callback(search_paths)

    if callable(video_search):
        abort = video_search(labels.videos)

        if abort:
            raise FileNotFoundError

    return labels


def find_suggestion(labels: Labels, video, frame_idx):
    """Return the suggestion for the given (video, frame_idx) or None."""
    match = [
        item
        for item in labels.suggestions
        if item.video == video and item.frame_idx == frame_idx
    ]
    if match:
        return match[0]

    return None


def get_next_suggestion(labels: Labels, video, frame_idx, seek_direction=1):
    """Return a (video, frame_idx) tuple seeking from given frame."""
    # make sure we have valid seek_direction
    if seek_direction not in (-1, 1):
        raise ValueError("Invalid seek_direction. Use -1 or 1.")

    # make sure the video belongs to the labels object
    if video not in labels.videos:
        return None

    all_suggestions = labels.suggestions

    # If we are currently on a suggestion, then follow order of list
    match = find_suggestion(labels, video, frame_idx)
    if match is not None:
        suggestion_idx = all_suggestions.index(match)
        new_idx = (suggestion_idx + seek_direction) % len(all_suggestions)
        return all_suggestions[new_idx]

    # Otherwise, find the prev/next suggestion sorted by frame order...

    # Look for next (or previous) suggestion in current video.
    if seek_direction == 1:
        frame_suggestion = min(
            (i for i in get_video_suggestions(labels, video) if i > frame_idx),
            default=None,
        )
    else:
        frame_suggestion = max(
            (i for i in get_video_suggestions(labels, video) if i < frame_idx),
            default=None,
        )
        if frame_suggestion is not None:
            return (video, frame_suggestion)

    if frame_suggestion is not None:
        return find_suggestion(labels, video, frame_suggestion)

    # If we did not find suggestion in current video,
    # then we want earliest frame in next video with suggestions

    next_video_idx = (labels.videos.index(video) + seek_direction) % len(labels.videos)
    video = labels.videos[next_video_idx]
    if seek_direction == 1:
        frame_suggestion = min(
            (i for i in get_video_suggestions(labels, video)), default=None
        )
    else:
        frame_suggestion = max(
            (i for i in get_video_suggestions(labels, video)), default=None
        )

    return find_suggestion(labels, video, frame_suggestion)


def instances(
    labels: Optional[Labels],
    video: Optional[Video] = None,
    skeleton: Optional[Skeleton] = None,
):
    for labeled_frame in labels.labeled_frames:
        if labeled_frame.video is not None or labeled_frame.video == video:
            for instance in labeled_frame.instances:
                if skeleton is None or instance.skeleton == skeleton:
                    yield instance


def merge_nodes_data(
    predicted_instance: PredictedInstance,
    points_array: PointsArray,
    base_node: str,
    merge_node: str,
):
    """Copy point data from one node to another.

    Args:
        base_node: Name of node that will be merged into.
        merge_node: Name of node that will be removed after merge.

    Notes:
        This is used when merging skeleton nodes and should not be called directly.
    """

    base_pt = points_array.__getitem__(base_node) if points_array is not None else None
    merge_pt = (
        points_array.__getitem__(merge_node) if points_array is not None else None
    )

    # check x coordinate not NaN
    if math.isnan(merge_pt["xy"][0]):
        return

    # check y coordinate not NaN
    if math.isnan(merge_pt["xy"][1]) or not base_pt["visible"]:
        base_pt["xy"][0] = merge_pt["xy"][0]
        base_pt["xy"][1] = merge_pt["xy"][1]
        base_pt["visible"] = merge_pt["visible"]
        base_pt["complete"] = merge_pt["complete"]
        # if hasattr(base_instance.from_predicted, 'score'):
        predicted_points_array = predicted_instance.points
        if hasattr(predicted_instance, "score"):
            predicted_points_array.get("base_node")["score"] = (
                predicted_points_array.get("merge_node")["score"]
            )


def merge_nodes(
    base_node: str,
    merge_node: str,
    labels: Optional[Labels],
    skeleton: Optional[Skeleton],
):
    """Merge two nodes and update data accordingly.

    Args:
        base_node: Name of skeleton node that will remain after merging.
        merge_node: Name of skeleton node that will be merged into the base node.

    Notes:
        This method can be used to merge two nodes that might have been named
        differently but that should be associated with the same node.

        This is useful, for example, when merging a different set of labels where
        a node was named differently. an

        If the `base_node` is visible and has data, it will not be updated.
        Otherwise, it will be updated with the data from the `merge_node` on the
        same instance.
    """
    # Labels <- List<LabeledFrame> <- List<Instance> hierarchy
    for inst in instances(labels, skeleton):
        # inst._merge_nodes_data(base_node, merge_node)
        points_array = inst.points
        predicted_inst = inst.from_predicted
        merge_nodes_data(predicted_inst, points_array, base_node, merge_node)

    # Remove merge node from skeleton.
    # skeleton.delete_node(merge_node)
    skeleton.remove_node(merge_node)

    # No cache -> cannot implement this
    # # Fix instances.
    # for inst in instances(labels, skeleton):
    #     inst._fix_array()


def find_track_occupancy(
    labels: Labels, video: Video, track: Union[Track, int], frame_range=None
) -> List[Instance]:
    """Get instances for a given video, track, and range of frames.

    Args:
        video: the `Video`
        track: the `Track` or int ("pseudo-track" index to instance list)
        frame_range (optional):
            If specified, only return instances on frames in range.
            If None, return all instances for given track.

    Returns:
        List of :class:`Instance` objects.
    """
    frame_range = range(*frame_range) if type(frame_range) == tuple else frame_range

    def does_track_match(inst, tr, labeled_frame):
        match = False
        if type(tr) == Track and inst.track is tr:
            match = True
        elif (
            type(tr) == int
            and labeled_frame.instances.index(inst) == tr
            and inst.track is None
        ):
            match = True
        return match

    track_frame_inst = [
        instance
        for lf in labels.find(video)
        for instance in lf.instances
        if does_track_match(instance, track, lf)
        and (frame_range is None or lf.frame_idx in frame_range)
    ]

    return track_frame_inst


def track_swap(
    labels: Labels,
    video: Video,
    new_track: Track,
    old_track: Optional[Track],
    frame_range: tuple,
):
    """Swap track assignment for instances in two tracks.

    If you need to change the track to or from None, you'll need
    to use :meth:`track_set_instance` for each specific
    instance you want to modify.

    Args:
        video: The :class:`Video` for which we want to swap tracks.
        new_track: A :class:`Track` for which we want to swap
            instances with another track.
        old_track: The other :class:`Track` for swapping.
        frame_range: Tuple of (start, end) frame indexes.
            If you want to swap tracks on a single frame, use
            (frame index, frame index + 1).
    """
    # labels._cache.track_swap(video, old_track, new_track, frame_range)

    # Get all instances in old/new tracks
    old_track_instances = find_track_occupancy(labels, video, old_track, frame_range)
    new_track_instances = find_track_occupancy(labels, video, new_track, frame_range)

    # Swap instances between old and new tracks
    for instance in old_track_instances:
        instance.track = new_track
    # old_track can be `Track` or int
    # If int, it's index in instance list which we'll use as a psudo-track,
    # but we won't set instance currently on new_track to old_track
    if type(old_track) == Track:
        # Only clear old track if it's a real track
        for instance in new_track_instances:
            instance.track = old_track


def track_set_instance(
    labels: Labels, frame: LabeledFrame, instance: Instance, new_track: Track
):
    """Set track on given instance, updating occupancy."""
    track_swap(
        labels,
        frame.video,
        new_track,
        instance.track,
        (frame.frame_idx, frame.frame_idx + 1),
    )
    instance.track = new_track


def clear_suggestion(labels: Labels):
    """Delete all suggestions."""
    labels.suggestions.clear()


# Labels API Compatibility Functions
# These functions provide backward compatibility with legacy SLEAP Labels API


def labels_get_suggestions(labels: Labels):
    """Get all suggestions from labels for backward compatibility.

    This provides backward compatibility for the missing get_suggestions() method.
    In sleap-io, suggestions are stored directly as an attribute.

    Args:
        labels: Labels object to get suggestions from

    Returns:
        List of SuggestionFrame objects
    """
    return getattr(labels, "suggestions", [])


def labels_get(labels: Labels, video_and_frame_or_video, frame_idx=None, **kwargs):
    """Get labeled frames for backward compatibility.

    This provides backward compatibility for the missing get() method.
    Handles both tuple format (video, frame_idx) and separate video, frame_idx args.

    Args:
        labels: Labels object to search
        video_and_frame_or_video: Either a (Video, frame_idx) tuple or a Video object
        frame_idx: Frame index (when first arg is Video)
        **kwargs: Additional arguments like use_cache
            (ignored for sleap-io compatibility)

    Returns:
        Single LabeledFrame if found, None otherwise (when frame_idx specified)
        List of LabeledFrame objects for video (when frame_idx not specified)
    """
    # Handle tuple format: labels.get((video, frame_idx))
    if (
        isinstance(video_and_frame_or_video, tuple)
        and len(video_and_frame_or_video) == 2
    ):
        video, frame_idx = video_and_frame_or_video
    else:
        video = video_and_frame_or_video

    # Use the existing Labels.find method from sleap-io
    if frame_idx is not None:
        matches = labels.find(video, frame_idx=frame_idx)
        return matches[0] if matches else None
    else:
        return labels.find(video)


def labels_all_instances(labels: Labels):
    """Get all instances as a list for backward compatibility.

    This provides backward compatibility for the missing all_instances attribute.
    In sleap-io, labels.instances is a generator, but legacy SLEAP expects
    a list-like object.

    Args:
        labels: Labels object to get all instances from

    Returns:
        List of all Instance objects from all labeled frames
    """
    return list(labels.instances)


def labels_clear_suggestions(labels: Labels):
    """Clear all suggestions from labels for backward compatibility.

    This provides backward compatibility for the missing clear_suggestions() method.
    In sleap-io, suggestions are stored as a list that can be cleared directly.

    Args:
        labels: Labels object to clear suggestions from
    """
    labels.suggestions.clear()


def labels_copy(labels: Labels) -> Labels:
    """Create a copy of the Labels object.

    This provides backward compatibility for the missing copy() method.
    Uses copy.deepcopy() which should be handled gracefully by sleap-io.
    """
    return copy.deepcopy(labels)


def labels_add_video(labels: Labels, video: Video):
    """Add a video to the Labels object.

    This provides backward compatibility for the missing add_video() method.
    """
    if video not in labels.videos:
        labels.videos.append(video)


def labels_pop(labels: Labels, index: int) -> LabeledFrame:
    """Remove and return a labeled frame at the given index.

    This provides backward compatibility for the missing pop() method.
    """
    if 0 <= index < len(labels.labeled_frames):
        return labels.labeled_frames.pop(index)
    else:
        raise IndexError(
            f"Index {index} out of range for "
            f"{len(labels.labeled_frames)} labeled frames"
        )


def labels_remove_frame(labels: Labels, labeled_frame: LabeledFrame):
    """Remove a single labeled frame from the Labels object.

    This provides backward compatibility for a missing remove_frame() method.
    Uses the existing remove_frames() function.
    """
    remove_frames(labels, [labeled_frame])


def labels_load_file(filename: str, **kwargs) -> Labels:
    """Load a Labels object from file.

    This provides backward compatibility for the missing static load_file() method.
    Handles video_search parameter that sleap-io doesn't support.
    """
    # Extract video_search parameter if present
    video_search = kwargs.pop("video_search", None)

    if video_search is not None:
        # Use existing video search function
        return load_labels_video_search(filename, video_search)
    else:
        # Standard load without video search
        return load_file(filename, **kwargs)


# Attribute compatibility functions
def labels_get_nodes(labels: Labels):
    """Get skeleton nodes for backward compatibility.

    Maps labels.nodes to labels.skeleton.nodes or labels.skeletons[0].nodes
    """
    if labels.skeletons:
        return labels.skeletons[0].nodes
    return []


def labels_get_labels_attr(labels: Labels):
    """Get labeled frames for backward compatibility.

    Maps labels.labels to labels.labeled_frames
    """
    return labels.labeled_frames


def labels_frames(labels: Labels, video: Video = None):
    """Get labeled frames, optionally filtered by video.

    This provides backward compatibility for the missing frames() method.
    """
    if video is None:
        return labels.labeled_frames
    else:
        return labels.find(video)


def labeled_frame_find(labeled_frame: LabeledFrame, track: Track = None):
    """Find instances in a labeled frame that match the given track.

    This provides backward compatibility for the missing LabeledFrame.find() method.
    In sleap-io, we need to manually search through instances to find matching tracks.

    Args:
        labeled_frame: LabeledFrame to search in
        track: Track to search for

    Returns:
        List of instances that match the track, or empty list if none found
    """
    if track is None:
        return list(labeled_frame.instances)

    matching_instances = []
    for instance in labeled_frame.instances:
        if instance.track == track:
            matching_instances.append(instance)

    return matching_instances


def labels_append_suggestions(labels: Labels, suggestions):
    """Append suggestions to the Labels object.

    This provides backward compatibility for the missing append_suggestions() method.
    In sleap-io, suggestions are stored as a list that can be extended directly.

    Args:
        labels: Labels object to append suggestions to
        suggestions: List of SuggestionFrame objects to append
    """
    if hasattr(suggestions, "__iter__"):
        labels.suggestions.extend(suggestions)
    else:
        labels.suggestions.append(suggestions)


def labels_add_instance(labels: Labels, frame: LabeledFrame, instance):
    """Add an instance to a labeled frame.

    This provides backward compatibility for the missing add_instance() method.
    In sleap-io, we manually add instances to the frame's instances list.

    Args:
        labels: Labels object (for consistency with legacy API, but not used)
        frame: LabeledFrame to add instance to
        instance: Instance to add to the frame
    """
    # In sleap-io, instances are stored in the LabeledFrame directly
    if hasattr(frame, "instances"):
        frame.instances.append(instance)
    else:
        # Fallback if instances is not a list
        frame.instances = list(frame.instances) + [instance]
