"""
Module for generating lists of frames using frame features, pca, kmeans, etc.
"""


import attr
import cattr
import itertools
import logging
import numpy as np
import random
from time import time
from typing import Dict, List, Optional, Tuple

import cv2

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from skimage import draw
from skimage.feature import hog
from skimage.util.shape import view_as_windows

from sleap.io.video import Video

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class BriskVec:

    brisk_threshold: int
    vocab_size: int
    debug: bool = False

    def __attrs_post_init__(self):
        self._brisk = cv2.BRISK_create(thresh=self.brisk_threshold)

    def get_vecs(self, imgs):
        all_descs = []
        row_img = []

        # Create matrix with multiple brisk descriptors for each image.
        for i, img in enumerate(imgs):
            kps, descs = self._brisk.detectAndCompute(img, None)

            # Brisk descriptor is 512 bits, but opencv returns this as 16 uint8's,
            # so we'll convert it to discrete numbers.
            descs = np.unpackbits(descs, axis=1)

            # Make list with all brisk descriptors (or all images) and map which
            # tells us which descriptor goes with which image
            row_img.extend([i] * len(descs))
            all_descs.append(descs)

        # Convert to single matrix of descriptors
        all_descs = np.concatenate(all_descs)

        # Convert to single matrix of row (individual descriptor) -> image index
        row_img = np.array(row_img)

        # Create a bag of features for each image by clustering the brisk image
        # descriptors (these clusters will be the "words" in a bag of words for
        # each image), then generate vocab-length vector for each image which
        # represents whether the "word" (i.e., brisk feature in some cluster)
        # is present in the image.

        kmeans = KMeans(n_clusters=self.vocab_size).fit(all_descs)
        return self.clusters_to_vecs(kmeans.labels_, row_img, len(imgs))
        # img_bags = np.zeros((len(imgs), self.vocab_size), dtype="bool")
        #
        # for i in range(len(imgs)):
        #     img_words = kmeans.labels_[row_img == i]
        #     img_bags[(i,), img_words] = 1
        #
        # return img_bags

    def clusters_to_vecs(self, cluster_labels, ownership, img_count):

        # Make helper function that builds bag of features vector for a single
        # image by looking up all the descriptors for an image and counting
        # how many there are for each cluster (vocab word).
        def img_bof_vec(img_idx):
            return np.bincount(
                cluster_labels[ownership == img_idx], minlength=self.vocab_size
            )

        # Now make the matrix with a bag of features vector for each image
        return np.stack([img_bof_vec(i) for i in range(img_count)])


@attr.s(auto_attribs=True)
class HogVec:

    brisk_threshold: int
    vocab_size: int
    debug: bool = False

    def __attrs_post_init__(self):
        self._brisk = cv2.BRISK_create(thresh=self.brisk_threshold)
        self.points_list = []
        self.cmap = [
            [31, 120, 180],
            [51, 160, 44],
            [227, 26, 28],
            [255, 127, 0],
            [106, 61, 154],
            [177, 89, 40],
            [166, 206, 227],
            [178, 223, 138],
            [251, 154, 153],
            [253, 191, 111],
            [202, 178, 214],
            [255, 255, 153],
        ]

    def get_vecs(self, imgs):
        # Get matrix of hog descriptors for all images, and array which says
        # which image is the source for each row.
        descs, ownership = self.get_hogs(imgs)

        # Cluster the descriptors into a vocabulary for bag of features
        kmeans = KMeans(n_clusters=self.vocab_size).fit(descs)

        if self.debug:
            if imgs.shape[-1] == 1:
                new_shape = (imgs.shape[0], imgs.shape[1], imgs.shape[2], 3)

                self.vis = np.empty(new_shape, dtype=imgs.dtype)
                self.vis[..., 0] = imgs[..., 0]
                self.vis[..., 1] = imgs[..., 0]
                self.vis[..., 2] = imgs[..., 0]
            else:
                self.vis = np.copy(imgs)

            for i, img in enumerate(self.vis):
                img_desc_clusters = kmeans.labels_[ownership == i]
                img_points = self.points_list[i]
                for point, cluster in zip(img_points, img_desc_clusters):
                    color = self.cmap[cluster % len(self.cmap)]
                    cv2.circle(img, tuple(point), 3, color, lineType=cv2.LINE_AA)

        return self.clusters_to_vecs(kmeans.labels_, ownership, len(imgs))

    def clusters_to_vecs(self, cluster_labels, ownership, img_count):

        # Make helper function that builds bag of features vector for a single
        # image by looking up all the descriptors for an image and counting
        # how many there are for each cluster (vocab word).
        def img_bof_vec(img_idx):
            return np.bincount(
                cluster_labels[ownership == img_idx], minlength=self.vocab_size
            )

        # Now make the matrix with a bag of features vector for each image
        return np.stack([img_bof_vec(i) for i in range(img_count)])

    def get_hogs(self, imgs):
        """Returns descriptors and corresponding image for all images."""
        per_image_hog_descriptors = [self.get_image_hog(img) for img in imgs]
        descs = np.concatenate(
            [image_descs for image_descs in per_image_hog_descriptors]
        )
        ownership = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [i] * len(image_descs)
                        for i, image_descs in enumerate(per_image_hog_descriptors)
                    ]
                )
            )
        )
        return descs, ownership

    def get_image_hog(self, img):
        """Returns hog descriptor for all brisk keypoints on single image."""
        points = self.get_brisk_keypoints_as_points(img)
        center_points = points + np.array([8, 8])

        crops = self.get_image_crops(img, center_points)
        multichannel = img.ndim > 2

        img_descs = np.stack(
            [
                hog(
                    crop,
                    orientations=8,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1),
                    visualize=False,
                    multichannel=multichannel,
                )
                for crop in crops
            ]
        )
        return img_descs

    def get_image_crops(self, img, points):
        """Returns stack of windows around keypoints on single image."""
        W = view_as_windows(img, (16, 16, img.shape[-1]))[..., 0, :, :, :]

        max_y = W.shape[1] - 1
        max_x = W.shape[0] - 1

        xs = points[:, 0]
        ys = points[:, 1]

        # Shift crops for keypoints that are too close to edges
        # TODO: is this how we should handle this case?
        xs[xs > max_x] = max_x
        ys[ys > max_y] = max_y

        return W[xs, ys]

    def get_brisk_keypoints_as_points(self, img):
        """Returns matrix of brisk keypoints for single image."""
        kps = self._brisk.detect(img)
        points = self.keypoints_to_points_matrix(kps)
        return points

    def keypoints_to_points_matrix(self, kps):
        points = np.round(np.array([kps[idx].pt for idx in range(0, len(kps))])).astype(
            np.int
        )
        self.points_list.append(points)
        return points


@attr.s(auto_attribs=True, frozen=True)
class FrameItem(object):
    """Just a simple wrapper for (video, frame_idx), plus method to get image."""

    video: Video
    frame_idx: int

    def get_raw_image(self, scale: float = 1.0):
        if scale == 1.0:
            return self.video[self.frame_idx]
        else:
            img = self.video[self.frame_idx]
            _, h, w, c = img.shape
            h_, w_ = int(h // (1 / scale)), int(w // (1 / scale))
            # note that cv2 expects (width, height) instead of (rows, columns)
            img = cv2.resize(np.squeeze(img), (w_, h_))[None, ...]
            if c == 1:
                img = img[..., None]
            return img


@attr.s(auto_attribs=True)
class FrameGroupSet(object):
    """
    Class for a set of groups of FrameItem objects.

    Each item can have at most one group; each group is represented as an int.

    Attributes:
        method: Label for the method used to generate group set.
        item_group: Dictionary which maps each item to its group.
        group_data: Dictionary of any extra data for each group;
            keys are group ids, values are dictionaries of data.
        groupset_data: Dictionary for any data about the entire set of groups.
    """

    method: str
    item_group: Dict[FrameItem, int] = attr.ib(default=attr.Factory(dict))
    group_data: Dict[int, dict] = attr.ib(default=attr.Factory(dict))
    groupset_data: Dict = attr.ib(default=attr.Factory(dict))

    def append_to_group(self, group: int, item: FrameItem):
        """Adds item to group."""
        self.item_group[item] = group
        if group not in self.group_data:
            self.group_data[group] = dict()

    def extend_group_items(self, group: int, item_list: List[FrameItem]):
        """Adds all items in list to group."""
        for item in item_list:
            self.append_to_group(group, item)

    def get_item_group(self, item: FrameItem):
        """Returns group that contain item."""
        return self.item_group.get(item, None)

    @property
    def groups(self):
        """Iterate over groups, yielding group and list of items."""
        for group in self.group_data.keys():
            item_list = [
                frame_item
                for (frame_item, frame_group) in self.item_group.items()
                if frame_group == group
            ]
            yield group, item_list

    @property
    def all_items(self):
        """Gets list of all items."""
        return list(itertools.chain(self.item_group.keys()))

    def sample(self, per_group: int, unique_samples: bool = True):
        """
        Returns new FrameGroupSet with groups sampled from current groups.

        Note that the order of items in the new groups will not match order of
        items in the groups from which samples are drawn.

        Args:
            per_group: The number of samples to take from each group.
            unique_samples: Whether to ensure that there are no shared items
                in the resulting groups.

        Returns:
            New FrameGroupSet.
        """
        new_groupset = FrameGroupSet(method="sample_groups")
        new_groupset.groupset_data["per_group"] = per_group

        selected_set = set()
        for group, group_item_list in self.groups:

            if unique_samples:
                # Remove items that were already sampled from other groups
                group_item_list = list(set(group_item_list) - selected_set)

            # Sample items from this group
            samples_from_group = np.random.choice(
                group_item_list, min(len(group_item_list), per_group), False
            )

            # Keep track of the items we sampled so far from any group
            selected_set = selected_set.union(set(samples_from_group))

            # Add this sampled group to the new set of groups

            # samples_from_group.sort()
            new_groupset.extend_group_items(group, list(samples_from_group))

        return new_groupset


@attr.s(auto_attribs=True)
class ItemStack(object):
    """
    Container for items, each item can "own" one or more rows of data.

    Attributes:
        items: The list of items
        data: An ndarray with rows of data corresponding to items.
        ownership: List which specifies which rows of data correspond to which
            items.
        meta: List which stores metadata about each operation on stack.
        group_sets: List of GroupSets of items.

    """

    items: List = attr.ib(default=attr.Factory(list))
    data: Optional[np.ndarray] = attr.ib(default=None, repr=False)
    ownership: Optional[List[tuple]] = None
    meta: List = attr.ib(default=attr.Factory(list))
    group_sets: List[FrameGroupSet] = attr.ib(default=attr.Factory(list))

    @property
    def current_groupset(self):
        """Gets current (most recent) group set."""
        if not self.group_sets:
            return None
        return self.group_sets[-1]

    def get_item_data_idxs(self, item):
        """Returns indexes of rows in data which belong to item."""
        item_idx = self.items.index(item)
        if self.ownership:
            owns = self.ownership[item_idx]
        else:
            owns = tuple([item_idx])
        return owns

    def get_item_data(self, item):
        """Returns rows of data which belong to item."""
        owns = self.get_item_data_idxs(item)
        return self.data[owns, ...]

    def get_item_by_data_row(self, row_idx):
        if self.ownership:
            for item_idx, owns in enumerate(self.ownership):
                if row_idx in owns:
                    return self.items[item_idx]

        elif len(self.items) > row_idx:
            return self.items[row_idx]

        raise IndexError(f"No ownership for row {row_idx}.")

    def extend_ownership(self, ownership, row_count):
        """Extends an ownership list with number of rows owned by next item."""
        start_i = 0
        if len(ownership):
            # Start at 1 + (last row index of last item so far)
            start_i = 1 + ownership[-1][-1]

        item_owns = list(range(start_i, start_i + row_count))
        ownership.append(item_owns)

    def get_raw_images(self, scale=0.5):
        """Sets data to raw image for each FrameItem."""
        self.meta.append(dict(action="raw_images"))

        data_shape = [1, 1, 1]
        mixed_shapes = False
        imgs = []
        for frame in self.items:
            # Add to list of raw images
            img = frame.get_raw_image(scale=scale)
            imgs.append(img)

            # Keep track of shape large enough to hold any of the images
            img_shape = img.shape
            data_shape = [max(data_shape[i], img_shape[i + 1]) for i in (0, 1, 2)]

            if data_shape != img_shape:
                mixed_shapes = True

        if mixed_shapes:
            # Make array large enough to hold any image and pad smaller images
            self.data = np.zeros((len(self.items), *data_shape), dtype="uint8")
            for i, img in enumerate(imgs):
                _, rows, columns, channels = img.shape
                self.data[i, :rows, :columns, :channels] = img
        else:
            self.data = np.concatenate(imgs)

    def flatten(self):
        """Flattens each row of data to 1-d array."""
        meta = dict(action="flatten", shape=self.data.shape[1:])
        self.meta.append(meta)

        row_count = self.data.shape[0]
        row_size = np.product(meta["shape"])
        self.data = np.reshape(self.data, (row_count, row_size))

    def brisk_bag_of_features(self, brisk_threshold=40, vocab_size=20):
        """Transform data using bag of features based on brisk features."""
        brisk = BriskVec(brisk_threshold=brisk_threshold, vocab_size=vocab_size)
        self.data = brisk.get_vecs(self.data)

    def hog_bag_of_features(self, brisk_threshold=40, vocab_size=20):
        """Transforms data into bag of features vector of hog descriptors."""
        hog = HogVec(brisk_threshold=brisk_threshold, vocab_size=vocab_size)
        self.data = hog.get_vecs(self.data)

    def pca(self, n_components: int):
        """Transforms data by applying PCA."""
        pca = PCA(n_components=n_components)
        # PCA applies row by row, so we can modify data in place
        self.data = pca.fit_transform(self.data)
        self.meta.append(
            dict(
                action="pca",
                n_components=n_components,
                # components=pca.components_.tolist(),
            )
        )

    def kmeans(self, n_clusters: int):
        """Adds GroupSet using k-means clustering on data."""
        # print(f"kmeans on {len(self.data)} rows of data")
        kmeans = KMeans(n_clusters=n_clusters).fit(self.data)

        cluster_groupset = FrameGroupSet(method="kmeans")
        cluster_groupset.groupset_data = dict(centers=kmeans.cluster_centers_.tolist())

        # Make list of the items in each cluster
        item_labels = kmeans.labels_
        for cluster_idx in range(n_clusters):
            (cluster_item_idxs,) = np.where(item_labels == cluster_idx)
            for data_row_idx in cluster_item_idxs:
                item = self.get_item_by_data_row(data_row_idx)
                cluster_groupset.append_to_group(cluster_idx, item)

        self.group_sets.append(cluster_groupset)
        self.meta.append(dict(action="kmeans", n_clusters=n_clusters))

    def make_sample_group(
        self, videos: List[Video], samples_per_video: int, sample_method: str = "stride"
    ):
        """Adds GroupSet by sampling frames from each video."""
        groupset = FrameGroupSet(method="stride")
        groupset.groupset_data = dict(samples_per_video=samples_per_video)

        for i, video in enumerate(videos):

            if samples_per_video >= video.num_frames:
                idxs = list(range(video.num_frames))
            elif sample_method == "stride":
                idxs = list(range(0, video.frames, video.frames // samples_per_video))
                idxs = idxs[:samples_per_video]
            elif sample_method == "random":
                idxs = random.sample(range(video.frames), samples_per_video)
            else:
                raise ValueError(f"Invalid sampling method: {sample_method}")

            group_id = i
            for frame_idx in idxs:
                groupset.append_to_group(group_id, FrameItem(video, frame_idx))

        self.group_sets.append(groupset)
        self.meta.append(dict(action="sample", method="sample_method"))

    def get_all_items_from_group(self):
        """Sets items for Stack to all items from current GroupSet."""
        if self.current_groupset:
            self.items = self.current_groupset.all_items
            self.data = None  # clear data when setting items

    def sample_groups(self, samples_per_group: int):
        """Adds GroupSet by sampling items from current GroupSet."""
        if self.current_groupset:
            new_groupset = self.current_groupset.sample(
                per_group=samples_per_group, unique_samples=True
            )
            self.group_sets.append(new_groupset)

    def to_suggestion_tuples(
        self, videos, group_offset: int = 0, video_offset: int = 0
    ) -> List[Tuple[int, int, int]]:
        tuples = []
        for frame in self.items:
            group = self.current_groupset.get_item_group(frame)
            if group is not None:
                group += group_offset
            video_idx = videos.index(frame.video) + video_offset
            tuples.append((video_idx, frame.frame_idx, group))
        return tuples

    def to_suggestion_frames(self, group_offset: int = 0) -> List["SuggestionFrame"]:
        from sleap.gui.suggestions import SuggestionFrame

        suggestions = []
        for frame in self.items:
            group = self.current_groupset.get_item_group(frame)
            if group is not None:
                group += group_offset
            suggestions.append(SuggestionFrame(frame.video, frame.frame_idx, group))
        return suggestions


@attr.s(auto_attribs=True, slots=True)
class FeatureSuggestionPipeline(object):
    per_video: int
    sample_method: str
    scale: float
    feature_type: str
    n_components: int
    n_clusters: int
    per_cluster: int
    brisk_threshold: int = 40
    vocab_size: int = 20
    frame_data: Optional[ItemStack] = None

    def run_disk_stage(self, videos):
        self.frame_data = ItemStack()

        # Make the list of frames, sampling from each video
        self.frame_data.make_sample_group(
            videos, samples_per_video=self.per_video, sample_method=self.sample_method
        )
        self.frame_data.get_all_items_from_group()

        # Load the frame images
        self.frame_data.get_raw_images(scale=self.scale)

    def run_processing_state(self):
        if self.frame_data is None:
            raise ValueError(
                "Processing state called before disk stage (frame_data is None)"
            )

        # Generate feature data for each frame
        if self.feature_type == "brisk":
            # Get bag of features vector for each image from brisk descriptors
            # for brisk keypoints on each image.
            self.frame_data.brisk_bag_of_features(
                brisk_threshold=self.brisk_threshold, vocab_size=self.vocab_size
            )
        elif self.feature_type == "hog":
            # Get bag of features vector for each image from hog descriptors
            # at brisk keypoints.
            self.frame_data.hog_bag_of_features(
                brisk_threshold=self.brisk_threshold, vocab_size=self.vocab_size
            )
        else:
            # Flatten the raw image matrix for each image
            self.frame_data.flatten()

        # Transform data using PCA
        self.frame_data.pca(n_components=self.n_components)

        # Generate groups of frames using k-means
        self.frame_data.kmeans(n_clusters=self.n_clusters)

        # Limit the number of items in each group
        self.frame_data.sample_groups(samples_per_group=self.per_cluster)

        # Finally, make the list of items across all the groups
        self.frame_data.get_all_items_from_group()

        return self.frame_data

    def run(self, videos):
        # Only run disk stage is we're running from scratch; otherwise, we
        # assume that the disk stage was already run.
        if self.frame_data is None:
            self.run_disk_stage(videos)
        self.run_processing_state()
        return self.frame_data

    def reset(self):
        self.frame_data = None

    def get_suggestion_frames(self, videos, group_offset=0):
        return self.run(videos).to_suggestion_frames(group_offset)

    def get_suggestion_tuples(self, videos, group_offset=0, video_offset=0):
        return self.run(videos).to_suggestion_tuples(videos, group_offset, video_offset)


@attr.s(auto_attribs=True, slots=True)
class ParallelFeaturePipeline(object):
    """
    Enables easy per-video pipeline parallelization for feature suggestions.

    Create a `FeatureSuggestionPipeline` with the desired parameters, and
    then call `ParallelFeaturePipeline.run()` with the pipeline and the list
    of videos to process in parallel. This will take care of serializing the
    videos, running the pipelines in a process pool, and then deserializing
    the results back into a single list of `SuggestionFrame` objects.
    """

    pipeline: FeatureSuggestionPipeline
    videos_as_dicts: List[Dict]

    def get(self, video_idx):
        """Apply pipeline to single video by idx. Can be called in process."""
        video_dict = self.videos_as_dicts[video_idx]
        video = cattr.structure(video_dict, Video)
        group_offset = video_idx * self.pipeline.n_clusters

        # t0 = time()
        # logger.info(f"starting {video_idx}")

        result = self.pipeline.get_suggestion_tuples(
            videos=[video], group_offset=group_offset, video_offset=video_idx
        )
        self.pipeline.reset()

        # logger.info(f"done with {video_idx} in {time() - t0} s for {len(result)} suggestions")
        return result

    @classmethod
    def make(cls, pipeline, videos):
        """Make class object from pipeline and list of videos."""
        videos_as_dicts = cattr.unstructure(videos)
        return cls(pipeline, videos_as_dicts)

    @classmethod
    def tuples_to_suggestions(cls, tuples, videos):
        """Converts serialized data from processes back into SuggestionFrames."""
        from sleap.gui.suggestions import SuggestionFrame

        suggestions = []
        for (video_idx, frame_idx, group) in tuples:
            video = videos[video_idx]
            suggestions.append(SuggestionFrame(video, frame_idx, group))
        return suggestions

    @classmethod
    def run(cls, pipeline, videos, parallel=True):
        """Runs pipeline on all videos in parallel and returns suggestions."""
        from multiprocessing import Pool, Lock

        pp = cls.make(pipeline, videos)
        video_idxs = list(range(len(videos)))

        if parallel:

            pool = Pool()

            per_video_tuples = pool.map(pp.get, video_idxs)

        else:
            per_video_tuples = map(pp.get, video_idxs)

        tuples = list(itertools.chain.from_iterable(per_video_tuples))

        return pp.tuples_to_suggestions(tuples, videos)


def demo_pipeline():
    from sleap import Video

    vids = [
        Video.from_filename("tests/data/videos/centered_pair_small.mp4"),
        Video.from_filename("tests/data/videos/small_robot.mp4"),
    ]

    pipeline = FeatureSuggestionPipeline(
        per_video=10,
        scale=0.25,
        sample_method="random",
        feature_type="hog",
        brisk_threshold=120,
        n_components=5,
        n_clusters=5,
        per_cluster=5,
    )

    suggestions = ParallelFeaturePipeline.run(pipeline, vids, parallel=False)

    print(suggestions)


if __name__ == "__main__":
    demo_pipeline()
