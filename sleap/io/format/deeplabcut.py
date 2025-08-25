"""
Adaptor for reading DeepLabCut datasets.

This can either read a CSV file with labeled frames for a single video,
or a YAML file which potentially contains multiple videos.

The adaptor was created by manually inspecting DeepLabCut files and there's no
guarantee that it will perfectly import all data (especially metadata).

If the adaptor can find full video files for the annotated frames, then the
full videos will be used in the resulting SLEAP dataset. Otherwise, we'll
create a video object which wraps the individual frame images.
"""

import os
import re
import yaml

import numpy as np
import pandas as pd

from typing import List, Optional, Dict, Tuple
from pathlib import Path

from sleap import Labels, Video
from sleap_io import Skeleton, Node
from sleap.instance import Instance, LabeledFrame, Point, Track
from sleap.util import find_files_by_suffix

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class LabelsDeepLabCutCsvAdaptor(Adaptor):
    """
    Reads DeepLabCut csv file with labeled frames for single video.
    """

    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "csv"

    @property
    def all_exts(self):
        return ["csv"]

    @property
    def name(self):
        return "DeepLabCut Dataset CSV"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        # TODO: add checks for valid deeplabcut csv
        return True

    def can_write_filename(self, filename: str):
        return False

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return False

    @classmethod
    def read(
        cls,
        file: FileHandle,
        full_video: Optional[Video] = None,
        *args,
        **kwargs,
    ) -> Labels:
        return Labels(
            labeled_frames=cls.read_frames(
                file=file, full_video=full_video, *args, **kwargs
            )
        )

    @classmethod
    def make_video_for_image_list(
        cls, image_dir, filenames
    ) -> Tuple[List[Video], List[int], List[int]]:
        """Creates a Video object from frame images.

        Args:
            image_dir: Directory where images are stored.
            filenames: List of image filenames.

        Returns:
            Tuple containing:
                - List of Video objects created from the images.
                - List of video indices for each image.
                - List of frame indices for each image.
        """

        # the image filenames in the csv may not match where the user has them
        # so we'll change the directory to match where the user has the csv
        def fix_img_path(img_dir, img_filename):
            img_filename = (Path(img_dir) / Path(img_filename).name).as_posix()
            img_filename = img_filename.replace("\\", "/")
            return img_filename

        def get_shape(filename):
            import cv2

            img = cv2.imread(filename)
            return img.shape[:2]

        # Fix image paths to match the CSV directory.
        filenames = list(map(lambda f: fix_img_path(image_dir, f), filenames))

        try:
            # Group by shape.
            shapes = list(map(get_shape, filenames))
            imgs_by_shape = {}
            for filename, shape in zip(filenames, shapes):
                if shape not in imgs_by_shape:
                    imgs_by_shape[shape] = []
                imgs_by_shape[shape].append(filename)

            # Create videos for each shape group.
            videos = []
            inds_by_img = {}
            for video_ind, (shape, img_fns) in enumerate(imgs_by_shape.items()):
                videos.append(
                    Video.from_image_filenames(img_fns, height=shape[0], width=shape[1])
                )
                for fidx, img_fn in enumerate(img_fns):
                    inds_by_img[img_fn] = (video_ind, fidx)

            # Return videos and indices in the input ordering.
            video_inds = []
            frame_inds = []
            for filename in filenames:
                video_ind, frame_ind = inds_by_img[filename]
                video_inds.append(video_ind)
                frame_inds.append(frame_ind)
        except Exception:
            # If we couldn't group by shape, create a single video for all images.
            videos = [Video.from_image_filenames(filenames)]
            video_inds = [0] * len(filenames)
            frame_inds = list(range(len(filenames)))

        return videos, video_inds, frame_inds

    @classmethod
    def read_frames(
        cls,
        file: FileHandle,
        skeleton: Optional[Skeleton] = None,
        full_video: Optional[Video] = None,
        *args,
        **kwargs,
    ) -> List[LabeledFrame]:
        filename = file.filename

        # Read CSV file.
        data = pd.read_csv(filename, header=[1, 2])

        # Check if this is in the new multi-animal format.
        is_multianimal = data.columns[0][0] == "individuals"
        is_new_format = data.columns[1][1].startswith("Unnamed")

        if is_multianimal:
            # Reload with additional header rows if using new format.
            data = pd.read_csv(filename, header=[1, 2, 3])

            # Pull out animal and node names from the columns.
            start_col = 3 if is_new_format else 1
            tracks: Dict[str, Optional[Track]] = {}
            node_names = []
            for animal_name, node_name, _ in data.columns[start_col:][::2]:
                # Keep the starting frame index for each individual/track
                if animal_name not in tracks.keys():
                    tracks[animal_name] = None
                if node_name not in node_names:
                    node_names.append(node_name)

        else:
            # Create the skeleton from the list of nodes in the csv file.
            # Note that DeepLabCut doesn't have edges, so these will need to be
            # added by user later.
            start_col = 3 if is_new_format else 1
            node_names = [n[0] for n in list(data)[start_col::2]]

        if skeleton is None:
            skeleton = Skeleton()
            skeleton.add_nodes(node_names)

        # Get list of all images filenames.
        if is_new_format:
            # New format has folder name and filename in separate columns.
            img_files = [f"{a}/{b}" for a, b in zip(data.iloc[:, 0], data.iloc[:, 2])]
        else:
            # Old format has filenames in a single column.
            img_files = data.iloc[:, 0]

        if not full_video:
            # Create the Video objects grouped by shape
            img_dir = os.path.dirname(filename)
            videos, video_inds, frame_inds = cls.make_video_for_image_list(
                img_dir, img_files
            )

        lfs = []
        for i in range(len(data)):
            # Figure out the video and frame index to use.
            if full_video:
                # Use the input provided one.
                video = full_video

                # Extract "0123" from "path/img0123.png" as original frame index.
                frame_idx_match = re.search("(?<=img)(\\d+)(?=\\.png)", img_files[i])

                if frame_idx_match is not None:
                    frame_idx = int(frame_idx_match.group(0))
                else:
                    raise ValueError(
                        f"Unable to determine frame index for image {img_files[i]}"
                    )
            else:
                # Get from pregrouped list.
                video = videos[video_inds[i]]
                frame_idx = frame_inds[i]

            instances = []
            if is_multianimal:
                for animal_name in tracks.keys():
                    any_not_missing = False
                    # Get points for each node.
                    instance_points = dict()
                    for node in node_names:
                        # node is a string (node name), not a Node object
                        if (animal_name, node) in data.columns:
                            x, y = (
                                data[(animal_name, node, "x")][i],
                                data[(animal_name, node, "y")][i],
                            )
                        else:
                            x, y = np.nan, np.nan
                        instance_points[node] = Point(x, y)
                        if ~(np.isnan(x) and np.isnan(y)):
                            any_not_missing = True

                    if any_not_missing:
                        # Create track
                        if tracks[animal_name] is None:
                            tracks[animal_name] = Track(spawned_on=i, name=animal_name)
                        # Create instance with points.
                        instances.append(
                            Instance(
                                skeleton=skeleton,
                                points=instance_points,
                                track=tracks[animal_name],
                            )
                        )
            else:
                # Get points for each node.
                any_not_missing = False
                instance_points = dict()
                for node in node_names:
                    # node is a string (node name), not a Node object
                    x, y = data[(node, "x")][i], data[(node, "y")][i]
                    instance_points[node] = Point(x, y)
                    if ~(np.isnan(x) and np.isnan(y)):
                        any_not_missing = True

                if any_not_missing:
                    # Create instance with points assuming there's a single instance per
                    # frame.
                    instances.append(
                        Instance(skeleton=skeleton, points=instance_points)
                    )

            if len(instances) > 0:
                # Create LabeledFrame and add it to list.
                lfs.append(
                    LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
                )

        return lfs


class LabelsDeepLabCutYamlAdaptor(Adaptor):
    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "yaml"

    @property
    def all_exts(self):
        return ["yaml", "yml"]

    @property
    def name(self):
        return "DeepLabCut Dataset YAML"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        if "video_sets" not in file.text:
            return False
        return True

    def can_write_filename(self, filename: str):
        return False

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return False

    @classmethod
    def read(
        cls,
        file: FileHandle,
        *args,
        **kwargs,
    ) -> Labels:
        filename = file.filename

        # Load data from the YAML file
        project_data = yaml.load(file.text, Loader=yaml.SafeLoader)

        # Create skeleton which we'll use for each video
        node_names = []
        if project_data.get("multianimalbodyparts", False):
            node_names.extend(project_data["multianimalbodyparts"])
            if "uniquebodyparts" in project_data:
                node_names.extend(project_data["uniquebodyparts"])
        else:
            node_names.extend(project_data["bodyparts"])

        skeleton = Skeleton()
        skeleton.add_nodes(node_names)

        # Get subdirectories of videos and labeled data
        root_dir = os.path.dirname(filename)
        videos_dir = os.path.join(root_dir, "videos")
        labeled_data_dir = os.path.join(root_dir, "labeled-data")

        with os.scandir(labeled_data_dir) as file_iterator:
            data_subdirs = [file.path for file in file_iterator if file.is_dir()]

        labeled_frames = []

        # Each subdirectory of labeled data corresponds to a video.
        # We'll go through each and import the labeled frames.

        for data_subdir in data_subdirs:
            csv_files = find_files_by_suffix(
                data_subdir, prefix="CollectedData", suffix=".csv"
            )

            if csv_files:
                csv_path = csv_files[0]

                # Try to find a full video corresponding to this subdir.
                # If subdirectory is foo, we look for foo.mp4 in videos dir.

                shortname = os.path.split(data_subdir)[-1]
                video_path = None
                if os.path.exists(videos_dir):
                    with os.scandir(videos_dir) as file_iterator:
                        for file in file_iterator:
                            if not file.is_file():
                                continue
                            if os.path.splitext(file.name)[0] != shortname:
                                continue
                            video_path = os.path.join(videos_dir, file.name)
                            break

                if video_path is not None and os.path.exists(video_path):
                    video = Video.from_filename(video_path)
                else:
                    # When no video is found, the individual frame images
                    # stored in the labeled data subdir will be used.
                    if video_path is None:
                        video_path = os.path.join(videos_dir, f"{shortname}.mp4")
                    print(
                        f"Unable to find {video_path} so using individual frame images."
                    )
                    video = None

                # Import the labeled fraems
                labeled_frames.extend(
                    LabelsDeepLabCutCsvAdaptor.read_frames(
                        file=FileHandle(csv_path), skeleton=skeleton, full_video=video
                    )
                )

            else:
                print(f"No csv data file found in {data_subdir}")

        return Labels(labeled_frames=labeled_frames)
