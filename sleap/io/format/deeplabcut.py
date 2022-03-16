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

from typing import List, Optional

from sleap import Labels, Video, Skeleton
from sleap.instance import Instance, LabeledFrame, Point
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
    def make_video_for_image_list(cls, image_dir, filenames) -> Video:
        """Creates a Video object from frame images."""

        # the image filenames in the csv may not match where the user has them
        # so we'll change the directory to match where the user has the csv
        def fix_img_path(img_dir, img_filename):
            img_filename = img_filename.replace("\\", "/")
            img_filename = os.path.basename(img_filename)
            img_filename = os.path.join(img_dir, img_filename)
            return img_filename

        filenames = list(map(lambda f: fix_img_path(image_dir, f), filenames))

        return Video.from_image_filenames(filenames)

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
            animal_names = []
            node_names = []
            for animal_name, node_name, _ in data.columns[start_col:][::2]:
                if animal_name not in animal_names:
                    animal_names.append(animal_name)
                if node_name not in node_names:
                    node_names.append(node_name)

        else:
            # Create the skeleton from the list of nodes in the csv file.
            # Note that DeepLabCut doesn't have edges, so these will need to be
            # added by user later.
            node_names = [n[0] for n in list(data)[1::2]]

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

        if full_video:
            video = full_video
            index_frames_by_original_index = True
        else:
            # Create the Video object
            img_dir = os.path.dirname(filename)
            video = cls.make_video_for_image_list(img_dir, img_files)

            # The frames in the video we created will be indexed from 0 to N
            # rather than having their index from the original source video.
            index_frames_by_original_index = False

        lfs = []
        for i in range(len(data)):

            # Figure out frame index to use.
            if index_frames_by_original_index:
                # Extract "0123" from "path/img0123.png" as original frame index.
                frame_idx_match = re.search("(?<=img)(\\d+)(?=\\.png)", img_files[i])

                if frame_idx_match is not None:
                    frame_idx = int(frame_idx_match.group(0))
                else:
                    raise ValueError(
                        f"Unable to determine frame index for image {img_files[i]}"
                    )
            else:
                frame_idx = i

            instances = []
            if is_multianimal:
                for animal_name in animal_names:
                    any_not_missing = False
                    # Get points for each node.
                    instance_points = dict()
                    for node in node_names:
                        x, y = (
                            data[(animal_name, node, "x")][i],
                            data[(animal_name, node, "y")][i],
                        )
                        instance_points[node] = Point(x, y)
                        if ~(np.isnan(x) and np.isnan(y)):
                            any_not_missing = True

                    if any_not_missing:
                        # Create instance with points.
                        instances.append(
                            Instance(skeleton=skeleton, points=instance_points)
                        )
            else:
                # Get points for each node.
                instance_points = dict()
                for node in node_names:
                    x, y = data[(node, "x")][i], data[(node, "y")][i]
                    instance_points[node] = Point(x, y)

                # Create instance with points assuming there's a single instance per
                # frame.
                instances.append(Instance(skeleton=skeleton, points=instance_points))

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
        skeleton = Skeleton()
        skeleton.add_nodes(project_data["bodyparts"])

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
                video_path = os.path.join(videos_dir, f"{shortname}.mp4")

                if os.path.exists(video_path):
                    video = Video.from_filename(video_path)
                else:
                    # When no video is found, the individual frame images
                    # stored in the labeled data subdir will be used.
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
