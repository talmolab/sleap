"""
Adaptor for reading COCO keypoint detection datasets.

See http://cocodataset.org/#format-data for details about this format.
"""

import os

import numpy as np

from sleap import Labels, Video, Skeleton
from sleap.gui.dialogs.missingfiles import MissingFilesDialog
from sleap.instance import Instance, LabeledFrame, Point, Track

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class LabelsCocoAdaptor(Adaptor):
    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "json"

    @property
    def all_exts(self):
        return ["json"]

    @property
    def name(self):
        return "COCO Dataset JSON"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        if not file.is_json:
            return False
        if "annotations" not in file.json:
            return False
        if "categories" not in file.json:
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
        img_dir: str,
        use_missing_gui: bool = False,
        *args,
        **kwargs,
    ) -> Labels:

        dicts = file.json

        # Make skeletons from "categories"
        skeleton_map = dict()
        for category in dicts["categories"]:
            skeleton = Skeleton(name=category["name"])
            skeleton_id = category["id"]
            node_names = category["keypoints"]
            skeleton.add_nodes(node_names)

            try:
                for src_idx, dst_idx in category["skeleton"]:
                    skeleton.add_edge(node_names[src_idx], node_names[dst_idx])
            except IndexError as e:
                # According to the COCO data format specifications[^1], the edges
                # are supposed to be 1-indexed. But in some of their own
                # dataset the edges are 1-indexed! So we'll try.
                # [1]: http://cocodataset.org/#format-data

                # Clear any edges we already created using 0-indexing
                skeleton.clear_edges()

                # Add edges
                for src_idx, dst_idx in category["skeleton"]:
                    skeleton.add_edge(node_names[src_idx - 1], node_names[dst_idx - 1])

            skeleton_map[skeleton_id] = skeleton

        # Make videos from "images"

        # Remove images that aren't referenced in the annotations
        img_refs = [annotation["image_id"] for annotation in dicts["annotations"]]
        dicts["images"] = list(filter(lambda im: im["id"] in img_refs, dicts["images"]))

        # Key in JSON file should be "file_name", but sometimes it's "filename",
        # so we have to check both.
        img_filename_key = "file_name"
        if img_filename_key not in dicts["images"][0].keys():
            img_filename_key = "filename"

        # First add the img_dir to each image filename
        img_paths = [
            os.path.join(img_dir, image[img_filename_key]) for image in dicts["images"]
        ]

        # See if there are any missing files
        img_missing = [not os.path.exists(path) for path in img_paths]

        if sum(img_missing):
            if use_missing_gui:
                okay = MissingFilesDialog(img_paths, img_missing).exec_()

                if not okay:
                    return None
            else:
                raise FileNotFoundError(
                    f"Images for COCO dataset could not be found in {img_dir}."
                )

        # Update the image paths (with img_dir or user selected path)
        for image, path in zip(dicts["images"], img_paths):
            image[img_filename_key] = path

        # Create the video objects for the image files
        image_video_map = dict()

        vid_id_video_map = dict()
        for image in dicts["images"]:
            image_id = image["id"]
            image_filename = image[img_filename_key]

            # Sometimes images have a vid_id which links multiple images
            # together as one video. If so, we'll use that as the video key.
            # But if there isn't a vid_id, we'll treat each images as a
            # distinct video and use the image id as the video id.
            vid_id = image.get("vid_id", image_id)

            if vid_id not in vid_id_video_map:
                kwargs = dict(filenames=[image_filename])
                for key in ("width", "height"):
                    if key in image:
                        kwargs[key] = image[key]

                video = Video.from_image_filenames(**kwargs)
                vid_id_video_map[vid_id] = video
                frame_idx = 0
            else:
                video = vid_id_video_map[vid_id]
                frame_idx = video.num_frames
                video.backend.filenames.append(image_filename)

            image_video_map[image_id] = (video, frame_idx)

        # Make instances from "annotations"
        lf_map = dict()
        track_map = dict()
        for annotation in dicts["annotations"]:
            skeleton = skeleton_map[annotation["category_id"]]
            image_id = annotation["image_id"]
            video, frame_idx = image_video_map[image_id]
            keypoints = np.array(annotation["keypoints"], dtype="int").reshape(-1, 3)

            track = None
            if "track_id" in annotation:
                track_id = annotation["track_id"]
                if track_id not in track_map:
                    track_map[track_id] = Track(frame_idx, str(track_id))
                track = track_map[track_id]

            points = dict()
            any_visible = False
            for i in range(len(keypoints)):
                node = skeleton.nodes[i]
                x, y, flag = keypoints[i]

                if flag == 0:
                    # node not labeled for this instance
                    continue

                is_visible = flag == 2
                any_visible = any_visible or is_visible
                points[node] = Point(x, y, is_visible)

            if points:
                # If none of the points had 2 has the "visible" flag, we'll
                # assume this incorrect and just mark all as visible.
                if not any_visible:
                    for point in points.values():
                        point.visible = True

                inst = Instance(skeleton=skeleton, points=points, track=track)

                if image_id not in lf_map:
                    lf_map[image_id] = LabeledFrame(video, frame_idx)

                lf_map[image_id].insert(0, inst)

        return Labels(labeled_frames=list(lf_map.values()))
