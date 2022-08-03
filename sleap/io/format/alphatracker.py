"""
Adaptor for reading AlphaTracker datasets.

This can read a JSON file with labeled frames for a single video,
or multiple videos.

The adaptor was created by manually inspecting AlphaTracker files and there's no
guarantee that it will perfectly import all data (especially metadata).

If the adaptor can find full video files for the annotated frames, then the
full videos will be used in the resulting SLEAP dataset. Otherwise, we'll
create a video object which wraps the individual frame images.
"""

import os
import json
import copy

from typing import List, Optional

from sleap.instance import Point, Instance
from sleap.io.dataset import Labels, Skeleton, LabeledFrame
from sleap.io.video import Video
from sleap.io.format.adaptor import Adaptor, SleapObjectType
from sleap.io.format.filehandle import FileHandle


class AlphaTrackerAdaptor(Adaptor):
    """
    Reads AlphaTracker JSON file with annotations for both single and multiple animals.
    """

    @property
    def handles(self) -> SleapObjectType:
        """
        Returns the type of object that can be read/written.

        The Dispatch class calls this method on all registered adaptors to
        determine which to use for reading/writing.
        """
        return SleapObjectType.labels

    @property
    def default_ext(self) -> str:
        """The default file extension, e.g., 'json' (without '.')."""
        return "json"

    @property
    def all_exts(self) -> List[str]:
        """List of all file extensions supported by adaptor."""
        return ["json"]

    @property
    def name(self) -> str:
        """Human-reading name of the file format"""
        return "AlphaTracker Dataset JSON"

    def can_read_file(self, file: FileHandle) -> bool:
        """Returns whether this adaptor can read this file.

        Checks the format of the file at three different levels:
        - First, the upper-level format of file.json must be a list of dictionaries.
        - Second, the dictionaries that represent frames must contain the specific keys
            that the adaptor reads.
        - Third, the "annotations" key in must contain a list of dictionaries.
        - Fourth, the dictionaries used to define bounding boxes for instances must
            contain the specific key that the adaptor reads.
        - Fifth, the dictionaries used to define the points of instance node must
            contain the specific keys that the adaptor reads.
        """

        def expect_keys(expected_keys: List[str], data_dict: dict) -> bool:
            """Returns whether the expected keys are in the data dictionary."""
            for key in expected_keys:
                if key not in data_dict:
                    # Data is not in expected format.
                    return False
            return True

        if not self.does_match_ext(file.filename):
            return False

        # Check for valid AlphaTracker JSON.

        data: List[dict] = file.json

        if type(data) != list:
            # Data is not in the expected format.
            return False

        if len(data) == 0:
            # Data is empty! Just create a new SLEAP Project.
            return False

        # Check that frame dictionaries are in expected format.
        frame_data: dict = data[0]
        if type(frame_data) != dict:
            # Data is not in the expected format.
            return False

        exp_keys = self.get_alpha_tracker_frame_dict().keys()
        if not expect_keys(exp_keys, frame_data):
            # Data is not in expected format.
            return False

        # Check that annotations are in expected format.
        ann: List = frame_data["annotations"]
        if type(ann) != list:
            # Data is not in the expected format.
            return False

        if len(ann) < 2:
            # Do not expect empty annotations or annotations with only bounding box.
            return False

        # Check that instance annootations are in expected format.
        ann_data_inst: dict = ann[0]
        if type(ann_data_inst) != dict:
            # Data not in expected format.
            return False

        exp_keys = self.get_alpha_tracker_instance_dict().keys()
        if not expect_keys(exp_keys, ann_data_inst):
            # Data is not in expected format.
            return False

        # Check that point annotations are in expected format.
        ann_data_point: dict = ann[1]
        if type(ann_data_point) != dict:
            # Data not in expected format.
            return False

        exp_keys = self.get_alpha_tracker_point_dict().keys()
        if not expect_keys(exp_keys, ann_data_point):
            # Data is not in expected format.
            return False

        return True

    def can_write_filename(self, filename: str) -> bool:
        """Returns whether this adaptor can write format of this filename."""
        return False

    def does_read(self) -> bool:
        """Returns whether this adaptor supports reading."""
        return True

    def does_write(self) -> bool:
        """Returns whether this adaptor supports writing."""
        return False

    def read(
        self,
        file: FileHandle,
        skeleton: Optional[Skeleton] = None,
        full_video: Optional[Video] = None,
    ) -> Labels:
        """Reads the file and returns the appropriate deserialized object.

        Args:
            file: The file to read.
            skeleton: The skeleton to use for :class:`Instance` objects. If no skeleton
                is provided, the skeleton will created with numerical node names.
            full_video: The video to use for the :class:`Labels` object.


        Returns:
            A :class:`Labels` object containing all the AlphaTracker annotations.
        """
        # TODO: Does not support multiple videos.

        def parse_instances(
            __frame_ann: List[dict], skeleton: Skeleton = skeleton
        ) -> List[Instance]:
            """
            Parse out the instances and corresponding points for each frame annotations.

            Args:
                __frame_ann: The AlphaTracker annotations for a single frame.
                skeleton: The skeleton to use for the :class:`Instance`s within
                    the frame.

            Returns:
                A list of :class:`Instance`s in the current frame.
            """
            __instances: List[Instance] = []  # Complete list of Instance objects.
            __instance_points: List[dict] = []  # Each dictionary is a unique instance.
            __instance_num: int = -1  # Index of instance in __instance_points.
            __node_num: int = 0  # Acts as node name. Warning: assumes nodes are ordered

            for __ann in __frame_ann:

                if __ann["class"] == "Face":
                    # Append a dictionary for the new instance.
                    __instance_num += 1
                    __node_num = 0
                    __instance_points.append({"instance_points": dict()})

                elif __ann["class"] == "point":
                    # Add the location of nodes to the current instance dictionary.
                    __node_num += 1
                    __instance_points[__instance_num]["instance_points"][
                        str(__node_num)
                    ] = Point(__ann["x"], __ann["y"])

                    if not skeleton.has_node(str(__node_num)):
                        # Add nodes to skeleton
                        skeleton.add_node(str(__node_num))

            for __inst in __instance_points:
                __instances.append(
                    Instance(skeleton=skeleton, points=__inst["instance_points"])
                )

            return __instances

        filename = file.filename

        # Extract animals and nodes
        data = file.json

        # Get image list of all image files
        img_files = []
        for frame in data:
            img_files.append(frame["filename"])

        if full_video:
            video = full_video
            index_frames_by_origin_index = True
        else:
            # Create the Video Object
            img_dir = os.path.dirname(filename)
            video = self.make_video_for_image_list(img_dir, img_files)

            # The frames in the video we created will be indexed from 0 to N
            # rather than having their index from the original source video.
            index_frames_by_origin_index = False

        lfs = []
        if skeleton is None:
            skeleton = Skeleton()

        # Loop through each frame to extract instances and frame index
        for i in range(len(data)):
            frame_annotations = data[i]["annotations"]

            # Get frames index
            if index_frames_by_origin_index:
                # Extract frame indices from image filenames.
                try:
                    # FIXME: Will return incorrect index if there are multiple videos.
                    frame_idx = int(img_files[i].split("_")[-2])
                    assert type(frame_idx) == int
                except:
                    raise ValueError(
                        f"Unable to determine frame index for image {img_files[i]}"
                    )
            else:
                frame_idx = i

            # Parse out all instances in frame
            instances = parse_instances(frame_annotations, skeleton)

            if len(instances) > 0:
                # Create instance with points assuming there's a single instance per
                # frame.
                lfs.append(
                    LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
                )

        return Labels(labeled_frames=lfs)

    def make_video_for_image_list(self, image_dir: str, filenames: List[str]) -> Video:
        """Creates a Video object from frame images.

        Args:
            image_dir: The directory that contains the AlphaTracker file.
            filenames: A list of filenames for the frame images.

        Returns:
            A :class:`Video` object.
        """

        # The image filenames in the JSON may not match where the user has them
        # so we'll change the directory to match where the user has the JSON
        def fix_img_path(img_dir, img_filename):
            img_filename = img_filename.replace("\\", "/")
            img_filename = os.path.basename(img_filename)
            img_filename = os.path.join(img_dir, img_filename)
            return img_filename

        filenames = list(map(lambda f: fix_img_path(image_dir, f), filenames))

        return Video.from_image_filenames(filenames)

    def get_alpha_tracker_frame_dict(annotations: List[dict] = [], filename: str = ""):
        """Returns a deep copy of the dictionary used for frames.

        Args:
            annotations: List of AlphaTracker annotations for current frame.
            filename: The filename of the image to use for the current frame.

        Returns:
            A dictionary containing the annotations in the frame, the AlphaTracker
            class of the dictionary ("image"), and the filename.
        """
        return copy.deepcopy(
            {"annotations": annotations, "class": "image", "filename": filename}
        )

    def get_alpha_tracker_instance_dict(
        height: int = 200,
        width: int = 200,
        x: float = 200.0,
        y: float = 200.0,
    ) -> dict:
        """Returns a deep copy of the dictionary used for instances.

        Args:
            height: The height of the bounding box.
            width: The width of the bounding box.
            x: The x-coor of lower left hand corner of the bounding box.
            y: The y-coor of lower left hand corner of the bounding box.

        Returns:
            A dictionary containing the AlphaTracker class of the dictionary
            ("Face") along with the height, width, and x,y-coordinates of the
            bounding box around the instance.
        """
        return copy.deepcopy(
            {
                "class": "Face",
                "height": height,
                "width": width,
                "x": x,
                "y": y,
            }
        )

    def get_alpha_tracker_point_dict(x: float = 200.0, y: float = 200.0) -> dict:
        """Returns a deep copy of the dictionary used for nodes.

        Args:
            x: The x-coordinate of the node location.
            y: The y-coordinate of the node location.

        Returns:
            A dictionary containing the AlphaTracker class of the dictionary
            ("point") and the x,y-coordinates of the node location.
        """
        return copy.deepcopy({"class": "point", "x": x, "y": y})

    # Methods with default implementation

    def write(self, filename: str, source_object: Labels) -> List[dict]:
        """Writes the object to an AlphaTracker JSON file.

        Args:
            filename: The name of the file being written to.
            souce_object: The :class:`Labels` object which contains the relevant
                information to write to an AlphaTracker JSON file.

        Raises:
            NotImplementedError: The code for the write functionality is not complete,
                see TODOs.

        Returns:
            A list of annotated frames in the same format seen in the AlphaTracker JSON.
        """

        # Code below writes to AlphaTracker format, but bounding box and image file is
        # not implemented.
        raise NotImplementedError

        def parse_data(
            __lfs: List[LabeledFrame] = source_object.labeled_frames,
        ) -> List[dict]:
            """Parses the data in the list of :class:`LabeledFrame`s to the format used
                in AlphaTracker.

            Args:
                __lfs: The list of :class:`LabeledFrame`s that contains all the
                    information to be converted to an AlphaTracker format.

            Returns:
                __data: A list of frames annotations in the AlphaTracker format.
            """

            __data: List[dict] = []

            for __i, __lf in enumerate(__lfs):
                # TODO: Extract image from video using frame index, then store the
                # images in a folder.

                # Case 1: Imported from AlphaTracker and create video from images
                # -> All images should already be present in the image directory

                # Case 2: Imported from AlphaTracker and provide full-video
                # -> Some images are present in image directory, but new images must be
                # created for new labeled frames

                # Case 3: Created in SLEAP and exporting to AlphaTracker
                # -> No images exist for any labeled frames. New images must be created
                # in an image directory for all labeled frames.

                # TODO: Pass path to frame image from AT file to  get_frame_dict.
                __frame_dict = self.get_alpha_tracker_frame_dict()
                for __instance in __lf.instances:
                    # Extract the animal -> draw bounding box
                    # TODO: AlphaTracker bounding box requires a crop of the instance.
                    __frame_dict["annotations"].append(
                        self.get_alpha_tracker_instance_dict()
                    )
                    for __node in __instance.points:
                        # Extract the nodes for each animal
                        __frame_dict["annotations"].append(
                            self.get_alpha_tracker_point_dict(__node[0], __node[1])
                        )
                __data.append(__frame_dict)
            return __data

        data = parse_data(source_object.labeled_frames)

        with open(filename, "w") as json_file:
            json_file.write(json.dumps(data, sort_keys=True, indent=4))

        print(f"Data written to {filename}.")

        return data

    def does_match_ext(self, filename: str) -> bool:
        """Returns whether this adaptor can write format of this filename."""

        # We don't match the ext against the result of os.path.splitext because
        # we want to match extensions like ".json.zip".

        return filename.endswith(tuple(self.all_exts))

    @property
    def formatted_ext_options(self):
        """String for Qt file dialog extension options."""
        return f"{self.name} ({' '.join(self.all_exts)})"
