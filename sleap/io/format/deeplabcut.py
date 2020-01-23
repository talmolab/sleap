import os

import pandas as pd

from sleap import Labels, Video, Skeleton
from sleap.instance import Instance, LabeledFrame, Point

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class LabelsDeepLabCutAdaptor(Adaptor):
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
    def read(cls, file: FileHandle, *args, **kwargs,) -> Labels:
        filename = file.filename

        # At the moment we don't need anything from the config file,
        # but the code to read it is here in case we do in the future.

        # # Try to find the config file by walking up file path starting at csv file looking for config.csv
        # last_dir = None
        # file_dir = os.path.dirname(filename)
        # config_filename = ""

        # while file_dir != last_dir:
        #     last_dir = file_dir
        #     file_dir = os.path.dirname(file_dir)
        #     config_filename = os.path.join(file_dir, 'config.yaml')
        #     if os.path.exists(config_filename):
        #         break

        # # If we couldn't find a config file, give up
        # if not os.path.exists(config_filename): return

        # with open(config_filename, 'r') as f:
        #     config = yaml.load(f, Loader=yaml.SafeLoader)

        # x1 = config['x1']
        # y1 = config['y1']
        # x2 = config['x2']
        # y2 = config['y2']

        data = pd.read_csv(filename, header=[1, 2])

        # Create the skeleton from the list of nodes in the csv file
        # Note that DeepLabCut doesn't have edges, so these will have to be added by user later
        node_names = [n[0] for n in list(data)[1::2]]

        skeleton = Skeleton()
        skeleton.add_nodes(node_names)

        # Create an imagestore `Video` object from frame images.
        # This may not be ideal for large projects, since we're reading in
        # each image and then writing it out in a new directory.

        img_files = data.ix[:, 0]  # get list of all images

        # the image filenames in the csv may not match where the user has them
        # so we'll change the directory to match where the user has the csv
        def fix_img_path(img_dir, img_filename):
            img_filename = os.path.basename(img_filename)
            img_filename = os.path.join(img_dir, img_filename)
            return img_filename

        img_dir = os.path.dirname(filename)
        img_files = list(map(lambda f: fix_img_path(img_dir, f), img_files))

        # we'll put the new imgstore in the same directory as the current csv
        imgstore_name = os.path.join(os.path.dirname(filename), "sleap_video")

        # create the imgstore (or open if it already exists)
        if os.path.exists(imgstore_name):
            video = Video.from_filename(imgstore_name)
        else:
            video = Video.imgstore_from_filenames(img_files, imgstore_name)

        labels = []

        for i in range(len(data)):
            # get points for each node
            instance_points = dict()
            for node in node_names:
                x, y = data[(node, "x")][i], data[(node, "y")][i]
                instance_points[node] = Point(x, y)
            # create instance with points (we can assume there's only one instance per frame)
            instance = Instance(skeleton=skeleton, points=instance_points)
            # create labeledframe and add it to list
            label = LabeledFrame(video=video, frame_idx=i, instances=[instance])
            labels.append(label)

        return Labels(labeled_frames=labels)
