"""
Labels Adaptor for SLEAP-IO

Provides missing methods and functionality that exist in sleap Labels
but are not available in sleap-io Labels.
"""

import os
from typing import Callable, Dict, List, Optional

from sleap_io.model.labels import Labels as SleapIOLabels


class LabelsAdaptor:
    """Adaptor class that provides missing functionality for sleap-io Labels."""

    @staticmethod
    def make_gui_video_callback(
        search_paths: Optional[List] = None,
        context: Optional[Dict[str, bool]] = None,
    ) -> Callable:
        """Create a GUI video callback for sleap-io Labels.

        This mimics the functionality of sleap Labels.make_gui_video_callback()
        but works with sleap-io Labels.

        Args:
            search_paths: List of paths to search for missing videos
            context: Dictionary containing context information

        Returns:
            A callback function for handling missing videos
        """
        return LabelsAdaptor.make_video_callback(
            search_paths=search_paths, use_gui=True, context=context
        )

    @staticmethod
    def make_video_callback(
        search_paths: Optional[List] = None,
        use_gui: bool = False,
        context: Optional[Dict[str, bool]] = None,
    ) -> Callable:
        """Create a video callback for sleap-io Labels.

        This mimics the functionality of sleap Labels.make_video_callback()
        but works with sleap-io Labels.

        Args:
            search_paths: List of paths to search for missing videos
            use_gui: Whether to use GUI dialogs for missing files
            context: Dictionary containing context information

        Returns:
            A callback function for handling missing videos
        """
        search_paths = search_paths or []
        context = context or {}

        def video_callback(
            video_list: List[dict],
            new_paths: List[str] = search_paths,
            context: Optional[Dict[str, bool]] = context,
        ):
            """Callback to find videos which have been moved.

            Args:
                video_list: List of serialized Video objects as dictionaries
                new_paths: List of paths to search for missing videos
                context: Dictionary with context information

            Returns:
                True to signal abort, False to continue
            """
            # Extract filenames from video list
            filenames = []
            for item in video_list:
                if isinstance(item, dict) and "backend" in item:
                    if "filename" in item["backend"]:
                        filenames.append(item["backend"]["filename"])
                    else:
                        filenames.append("")
                else:
                    filenames.append("")

            context = context or {"changed_on_load": False}

            # Check for missing files
            missing = []
            for filename in filenames:
                if filename and not os.path.exists(filename):
                    missing.append(True)
                else:
                    missing.append(False)

            # Try to find files in search paths
            if any(missing) and new_paths:
                for i, filename in enumerate(filenames):
                    if missing[i] and filename:
                        for search_path in new_paths:
                            if os.path.isdir(search_path):
                                potential_path = os.path.join(
                                    search_path, os.path.basename(filename)
                                )
                                if os.path.exists(potential_path):
                                    filenames[i] = potential_path
                                    missing[i] = False
                                    context["changed_on_load"] = True
                                    break

            # If using GUI and still have missing files, show dialog
            if use_gui and any(missing):
                try:
                    from sleap.gui.dialogs.missingfiles import MissingFilesDialog
                    from sleap.gui.app import get_app

                    # Get the main app instance
                    app = get_app()
                    if app and app.mainWindow:
                        # Show missing files dialog
                        dialog = MissingFilesDialog(
                            filenames, missing, allow_incomplete=True
                        )
                        if dialog.exec_() != dialog.Accepted:
                            return True  # Abort

                        context["changed_on_load"] = True
                except ImportError:
                    # If GUI components aren't available, just print warning
                    print(f"Warning: {sum(missing)} video files are missing")
                    print(
                        "Missing files:",
                        [f for f, m in zip(filenames, missing) if m],
                    )

            # Update video paths in the video list
            for i, (item, new_filename) in enumerate(zip(video_list, filenames)):
                if isinstance(item, dict) and "backend" in item:
                    item["backend"]["filename"] = new_filename

            return False  # Continue loading

        return video_callback

    @staticmethod
    def load_file(
        filename: str,
        video_search: Optional[Callable] = None,
        match_to: Optional[SleapIOLabels] = None,
    ) -> SleapIOLabels:
        """Load a file using sleap-io backend.

        This provides a compatible interface similar to sleap Labels.load_file()
        but uses sleap-io for the actual loading.

        Args:
            filename: Path to the file to load
            video_search: Optional callback for video search
            match_to: Optional Labels object to match against

        Returns:
            Loaded Labels object
        """
        from sleap_io.io.main import load_slp

        # Determine file type and load accordingly
        if filename.endswith(".slp"):
            labels = load_slp(filename, open_videos=True)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        # Apply video search callback if provided
        if video_search and callable(video_search):
            # Convert sleap-io Labels to format expected by video_search
            video_list = []
            for video in labels.videos:
                video_dict = {
                    "backend": {
                        "filename": video.filename,
                        "dataset": getattr(video, "dataset", ""),
                        "source": getattr(video, "source", ""),
                    }
                }
                video_list.append(video_dict)

            # Call the video search callback
            abort = video_search(video_list)
            if abort:
                raise ValueError("Video search callback requested abort")

            # Update video paths based on callback results
            for i, video in enumerate(labels.videos):
                if i < len(video_list):
                    new_filename = video_list[i]["backend"]["filename"]
                    if new_filename != video.filename:
                        # Create new video with updated path
                        from sleap_io.model.video import Video

                        new_video = Video.from_filename(new_filename)
                        # Replace video in labels
                        labels.videos[i] = new_video
                        # Update video references in labeled frames
                        for lf in labels.labeled_frames:
                            if lf.video == video:
                                lf.video = new_video

        return labels

    @staticmethod
    def save_file(labels: SleapIOLabels, filename: str, **kwargs) -> None:
        """Save Labels to file using sleap-io backend.

        This provides a compatible interface similar to sleap Labels.save_file()
        but uses sleap-io for the actual saving.

        Args:
            labels: The Labels object to save
            filename: Path where to save the file
            **kwargs: Additional arguments for saving
        """
        from sleap_io.io.main import save_slp

        # Determine file type and save accordingly
        if filename.endswith(".slp"):
            save_slp(labels, filename, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    @staticmethod
    def from_sleap_labels(sleap_labels) -> SleapIOLabels:
        """Convert sleap Labels to sleap-io Labels.

        Args:
            sleap_labels: A sleap Labels object

        Returns:
            Equivalent sleap-io Labels object
        """
        # This would need to be implemented based on the specific
        # conversion needs. For now, we'll raise NotImplementedError
        raise NotImplementedError(
            "Conversion from sleap Labels to sleap-io Labels not yet implemented"
        )

    @staticmethod
    def to_sleap_labels(sleap_io_labels: SleapIOLabels):
        """Convert sleap-io Labels to sleap Labels.

        Args:
            sleap_io_labels: A sleap-io Labels object

        Returns:
            Equivalent sleap Labels object
        """
        # This would need to be implemented based on the specific
        # conversion needs. For now, we'll raise NotImplementedError
        raise NotImplementedError(
            "Conversion from sleap-io Labels to sleap Labels not yet implemented"
        )

    # Add missing methods that exist in sleap-io but need to be accessible on
    # Labels class

    @staticmethod
    def load_nwb(filename: str):
        """Load an NWB dataset using sleap-io backend.

        Args:
            filename: Path to NWB file

        Returns:
            Labels object
        """
        from sleap_io.io.main import load_nwb

        return load_nwb(filename)

    @staticmethod
    def load_dlc(filename: str):
        """Load a DeepLabCut dataset using sleap-io backend.

        Args:
            filename: Path to DLC file

        Returns:
            Labels object
        """
        from sleap_io.io.main import load_dlc

        return load_dlc(filename)

    @staticmethod
    def load_coco(filename: str):
        """Load a COCO dataset using sleap-io backend.

        Args:
            filename: Path to COCO file

        Returns:
            Labels object
        """
        from sleap_io.io.main import load_coco

        return load_coco(filename)

    @staticmethod
    def load_ultralytics(filename: str):
        """Load an Ultralytics dataset using sleap-io backend.

        Args:
            filename: Path to Ultralytics file

        Returns:
            Labels object
        """
        from sleap_io.io.main import load_ultralytics

        return load_ultralytics(filename)

    @staticmethod
    def load_jabs(filename: str, skeleton=None):
        """Load a JABS dataset using sleap-io backend.

        Args:
            filename: Path to JABS file
            skeleton: Optional skeleton object

        Returns:
            Labels object
        """
        from sleap_io.io.main import load_jabs

        return load_jabs(filename, skeleton)

    @staticmethod
    def load_labelstudio(filename: str):
        """Load a Label Studio dataset using sleap-io backend.

        Args:
            filename: Path to Label Studio file

        Returns:
            Labels object
        """
        from sleap_io.io.main import load_labelstudio

        return load_labelstudio(filename)

    # Add methods that don't exist in sleap-io but are called in commands.py
    # These will raise NotImplementedError for now since they're not needed

    @staticmethod
    def load_alphatracker(filename: str, full_video: Optional[str] = None):
        """Load an AlphaTracker dataset.

        Args:
            filename: Path to AlphaTracker file
            full_video: Optional video path

        Returns:
            Labels object
        """
        raise NotImplementedError(
            "AlphaTracker loading not yet implemented in sleap-io. "
            "This method is not needed for current functionality."
        )

    @staticmethod
    def from_deepposekit(
        filename: str,
        video_path: Optional[str] = None,
        skeleton_path: Optional[str] = None,
    ):
        """Load a DeepPoseKit dataset.

        Args:
            filename: Path to DeepPoseKit file
            video_path: Optional video path
            skeleton_path: Optional skeleton path

        Returns:
            Labels object
        """
        raise NotImplementedError(
            "DeepPoseKit loading not yet implemented in sleap-io. "
            "This method is not needed for current functionality."
        )
