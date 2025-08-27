"""
Standalone utility functions for working with Labels and LabeledFrame objects.
"""

import os
from typing import List, Dict, Optional, Callable, Text
from pathlib import Path
import cattr

from sleap_io import Video
from sleap import util
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
    labeled_frames = labels.find(video) if hasattr(labels, 'find') else []
    
    # Build track occupancy dictionary
    for lf in labeled_frames:
        for instance in lf.instances:
            track = instance.track.name
            if track not in track_occupancy:
                track_occupancy[track] = []
            
            # Add this frame to the track's occupancy
            track_occupancy[track].append(lf.frame_idx)

    print(f"track_occupancy: {track_occupancy}")

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
            print(f"start: {start}, prev: {prev + 1}, ranges: {ranges}")
            print(f"len(ranges): {len(ranges)}")
            print(f"is empty: {SimpleRange(ranges).is_empty()}")
            
            track_occupancy[track] = SimpleRange(ranges)
    
    return track_occupancy


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
    if not hasattr(labels, 'suggestions'):
        return frame_indices
    
    # Get suggestions for this video
    for suggestion in labels.suggestions:
        if suggestion.video == video:
            fidx = suggestion.frame_idx
            
            # If user_labeled is False, skip suggestions that already have user labels
            if not user_labeled:
                lf = labels.get((video, fidx)) if hasattr(labels, 'get') else None
                if lf is not None and hasattr(lf, 'has_user_instances') and lf.has_user_instances:
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
    if not hasattr(labeled_frame, 'instances'):
        return unused_predictions
    
    # Get all instances from the frame
    instances = labeled_frame.instances if hasattr(labeled_frame, 'instances') else []
    
    any_tracks = [inst.track for inst in instances if hasattr(inst, 'track') and inst.track is not None]
    
    if len(any_tracks):
        # Use tracks to determine which predicted instances have been used
        used_tracks = [
            inst.track
            for inst in instances
            if hasattr(inst, 'track') and inst.track is not None and not hasattr(inst, 'from_predicted')
        ]
        unused_predictions = [
            inst
            for inst in instances
            if hasattr(inst, 'track') and inst.track not in used_tracks and hasattr(inst, 'from_predicted')
        ]
    else:
        # Use from_predicted to determine which predicted instances have been used
        used_instances = [
            inst.from_predicted
            for inst in instances
            if hasattr(inst, 'from_predicted') and inst.from_predicted is not None
        ]
        unused_predictions = [
            inst
            for inst in instances
            if hasattr(inst, 'from_predicted') and inst not in used_instances
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
    if not hasattr(labeled_frame, 'instances'):
        return []
    
    instances = labeled_frame.instances if hasattr(labeled_frame, 'instances') else []
    
    inst_to_show = [
        inst
        for inst in instances
        if not hasattr(inst, 'from_predicted') or inst in unused_predictions
    ]
    
    return inst_to_show


def get_labeled_frame_count(labels, video=None, filter: str = "") -> int:
    """Return count of frames matching video/filter.
    
    This function recreates the functionality of labels.get_labeled_frame_count(video, filter)
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
        raise ValueError(
            f"get_labeled_frame_count() invalid filter: {filter}"
        )
    
    # Get all labeled frames
    if hasattr(labels, 'labeled_frames'):
        all_frames = labels.labeled_frames
    elif hasattr(labels, '__iter__'):
        # If labels is iterable, use it directly
        all_frames = list(labels)
    else:
        return 0
    
    # Apply video filter
    if video is not None:
        frames = [lf for lf in all_frames if hasattr(lf, 'video') and lf.video == video]
    else:
        frames = all_frames
    
    # Apply type filter
    if filter == "":
        # All labeled frames
        return len(frames)
    elif filter == "user":
        # Only frames with user instances
        return len([lf for lf in frames if hasattr(lf, 'has_user_instances') and lf.has_user_instances])
    elif filter == "predicted":
        # Only frames with predicted instances
        return len([lf for lf in frames if hasattr(lf, 'has_predicted_instances') and lf.has_predicted_instances])
    
    return 0


def find_first(labels, video, frame_idx=None, use_cache: bool = False):
    """Find the first occurrence of a matching labeled frame.
    
    This function recreates the functionality of labels.find_first(video, frame_idx, use_cache)
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
    if use_cache and hasattr(labels, 'find'):
        # Use cache if available
        label = labels.find(video=video, frame_idx=frame_idx)
        return None if len(label) == 0 else label[0]
    else:
        # Check if video is in labels
        if hasattr(labels, 'videos') and video in labels.videos:
            # Loop through all labels
            for label in labels:
                if hasattr(label, 'video') and label.video == video and (
                    frame_idx is None or (hasattr(label, 'frame_idx') and label.frame_idx == frame_idx)
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
    if hasattr(labels, 'videos') and video in labels.videos:
        # Loop through all labels in reverse order
        for label in reversed(list(labels)):
            if hasattr(label, 'video') and label.video == video and (
                frame_idx is None or (hasattr(label, 'frame_idx') and label.frame_idx == frame_idx)
            ):
                return label
    return None


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
    search_paths: Optional[List[str]] = None,
    use_gui: bool = False,
    context: Optional[Dict] = None,
):
    search_paths = search_paths or []
    context = context or {}

    def video_callback(
        video_list: List[dict],
        new_paths: List[str] = search_paths,
        context: Optional[Dict] = context,
    ):
        filenames = [item["backend"]["filename"] for item in video_list]
        context = context or {"changed_on_load": False}

        # Equivalent to pathutils.list_file_missing(filenames)
        missing = [not os.path.exists(filename) for filename in filenames]

        # Try changing the prefix using saved patterns
        if sum(missing):
            fix_paths_with_saved_prefix(filenames, missing)

        # Check for file in search_path dirctories
        if sum(missing) and new_paths:
            for i, filename in enumerate(filename):
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
            item["backend"]["filename"] = filenames[i]

        if USE_DUMMY_FOR_MISSING_VIDEOS and sum(missing):
            # Replace any video still missing with "dummy" video
            for is_missing, item in zip(missing, video_list):
                from sleap.io.video import DummyVideo

                vid = DummyVideo(filename=item["backend"]["filename"])
                item["backend"] = cattr.unstructure(vid)

    return video_callback
