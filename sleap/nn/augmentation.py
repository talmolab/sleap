import numpy as np
import tensorflow as tf
import attr
import cattr
import keras
import imgaug

from typing import Union, List, Tuple, Callable


@attr.s(auto_attribs=True)
class Augmenter(keras.utils.Sequence):
    """
    The Augmentor class is a keras.utils.Sequence that generates batches of
    augmented training data for both confidence maps and part affinity fields (PAFs).
    It utilizes imgaug transformations of keypoints (the skeleton nodes of an instance)
    and generates the confidence maps and PAFs on the fly from the augmented keypoints.
    """

    _X: Union[None, np.ndarray] = None
    Y: Union[None, np.ndarray] = None
    Points: Union[None, List[np.ndarray]] = None
    datagen: Union[None, Callable] = None
    output_names: Union[None, List[str]] = None
    batch_size: int = 32
    shuffle: bool = True
    rotation: Tuple[float, float] = (-180.0, 180.0)
    scale: Union[float, None] = None

    def __attrs_post_init__(self):

        # Setup affine augmentation
        # TODO: translation?
        self.aug_stack = []
        if self.rotation is not None:
            self.rotation = self.rotation if isinstance(self.rotation, tuple) else (-self.rotation, self.rotation)
            if self.scale is not None and self.scale[0] != self.scale[1]:
                self.scale = (min(self.scale), max(self.scale))
                self.aug_stack.append(imgaug.augmenters.Affine(rotate=self.rotation, scale=self.scale))
            else:
                self.aug_stack.append(imgaug.augmenters.Affine(rotate=self.rotation))

        # TODO: Flips?
        # imgaug.augmenters.Fliplr(0.5)

        # Create augmenter
        self.aug = imgaug.augmenters.Sequential(self.aug_stack)

    def shuffle(self, batches_only=False):
        """ Index-based shuffling """

        if batches_only:
            # Shuffle batch order
            np.random.shuffle(self.batches)
        else:
            # Re-batch after shuffling
            all_idx = np.arange(self.num_samples)
            np.random.shuffle(all_idx)
            self.batches = np.array_split(all_idx, np.ceil(self.num_samples / self.batch_size))
        
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, batch_idx):
        aug_det = self.aug.to_deterministic()
        idx = self.batches[batch_idx]

        X = self.X[idx].copy()
        X = aug_det.augment_images(X)

        if self.Y is not None:
            Y = self.Y[idx].copy()
            Y = aug_det.augment_images(Y)

        elif self.Points is not None:
            # There's no easy way to apply the keypoint augmentation when we have
            # multiple KeypointsOnImage for each image. So instead, we'll take an list
            # with multiple point_arrays, combine these into a single KeypointsOnImage,
            # and then split this back into multiple point_arrays.
            # Note that we need to keep track of the size of each point_array so we
            # can split the KeypointsOnImage properly.

            # Combine each list of point arrays (per frame) to single KeypointsOnImage
            # Points: frames -> instances -> point_array
            frames_in_batch = [self.Points[i] for i in idx]
            points_per_instance_per_frame = [[pa.shape[0] for pa in frame] for frame in frames_in_batch]

            koi_in_frame = [imgaug.augmentables.kps.KeypointsOnImage.from_xy_array(np.concatenate(frame), shape=X[i].shape)
                            for i, frame in enumerate(frames_in_batch)]

            # Augment KeypointsOnImage
            aug_per_frame = aug_det.augment_keypoints(koi_in_frame)

            # Split single KeypointsOnImage back into list of point arrays (per frame).
            # i.e., frames -> instances -> point_array

            # First we convert KeypointsOnImage back to combined point_array
            aug_points_in_frames = [koi.to_xy_array() for koi in aug_per_frame]
            # then we split into list of point_arrays.
            split_points = []
            for i, frame in enumerate(aug_points_in_frames):
                frame_point_arrays = []
                offset = 0
                for point_count in points_per_instance_per_frame[i]:
                    inst_points = frame[offset:offset+point_count]
                    frame_point_arrays.append(inst_points)
                    offset += point_count
                split_points.append(frame_point_arrays)

            if self.datagen is not None:
                Y = self.datagen(split_points)
            else:
                Y = split_points

        else:
            # We shouldn't get here
            pass

        if self.output_names is not None:
            Y = {output_name: Y for output_name in self.output_names}

        return (X, Y)

    @staticmethod
    def make_cattr(X=None, Y=None, Points=None):
        aug_cattr = cattr.Converter()

        # When we serialize the augmentor, exlude data fields, we just want to
        # parameters.
        aug_cattr.register_unstructure_hook(
            Augmenter,
            lambda x:
            attr.asdict(x,
                        filter=attr.filters.exclude(
                            attr.fields(Augmenter).X,
                            attr.fields(Augmenter).Y,
                            attr.fields(Augmenter).Points)))

        # We the user needs to unstructure, what images, outputs, and Points should
        # they use. We didn't serialize these, just the parameters.
        if X is not None:
            aug_cattr.register_structure_hook(Augmenter,
                                              lambda x: Augmenter(X=X,Y=Y,Points=Points, **x))

        return aug_cattr


def demo_augmentation():
    from sleap.io.dataset import Labels
    from sleap.nn.datagen import generate_images, generate_points
    from sleap.nn.datagen import generate_confmaps_from_points, generate_pafs_from_points

    data_path = "tests/data/json_format_v2/centered_pair_predictions.json"
    # data_path = "tests/data/json_format_v2/minimal_instance.json"

    labels = Labels.load_json(data_path)

    labels.labeled_frames = labels.labeled_frames[123:323:10]

    # Generate raw training data
    skeleton = labels.skeletons[0]
    imgs = generate_images(labels)
    points = generate_points(labels)
    shape = (imgs.shape[1], imgs.shape[2])

    def datagen_from_points(points):
        return generate_pafs_from_points(points, skeleton, shape)
        # return generate_confmaps_from_points(points, skeleton, shape)

    # Augment
    aug = Augmenter(X=imgs, Points=points, datagen=datagen_from_points, scale=(.5, 2))
    imgs, pafs = aug[0]

    from sleap.io.video import Video
    from sleap.gui.confmapsplot import demo_confmaps
    from sleap.gui.quiverplot import demo_pafs
    from PySide2.QtWidgets import QApplication

    # Visualize augmented training data
    vid = Video.from_numpy(imgs*255)
    app = QApplication([])
    # demo_confmaps(confmaps, vid)
    demo_pafs(pafs, vid)
    app.exec_()

def demo_bad_augmentation():
    from sleap.io.dataset import Labels
    from sleap.nn.datagen import generate_images, generate_confidence_maps

    data_path = "tests/data/json_format_v2/centered_pair_predictions.json"
    # data_path = "tests/data/json_format_v2/minimal_instance.json"

    labels = Labels.load_json(data_path)

    labels.labeled_frames = labels.labeled_frames[123:323:10]

    # Generate raw training data
    skeleton = labels.skeletons[0]
    imgs = generate_images(labels)
    confmaps = generate_confidence_maps(labels)

    # Augment
    aug = Augmenter(X=imgs, Y=confmaps, scale=(.5, 2))
    imgs, confmaps = aug[0]

    from sleap.io.video import Video
    from sleap.gui.confmapsplot import demo_confmaps
    from sleap.gui.quiverplot import demo_pafs
    from PySide2.QtWidgets import QApplication

    # Visualize augmented training data
    vid = Video.from_numpy(imgs*255)
    app = QApplication([])
    demo_confmaps(confmaps, vid)
    app.exec_()

if __name__ == "__main__":
    demo_augmentation()

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(x[0].squeeze(), cmap="gray")

    # plt.subplot(1,2,2)
    # plt.imshow(y[0].max(axis=-1).squeeze(), cmap="gray")

    # plt.show()
