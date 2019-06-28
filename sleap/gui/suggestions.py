import numpy as np
import itertools

from skimage.transform import rescale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import cv2

from sleap.io.video import Video

class VideoFrameSuggestions:

    rescale=True
    rescale_below=512

    @classmethod
    def suggest(cls, video:Video, params:dict) -> list:
        """
        This is the main entry point.

        Args:
            video: a `Video` object for which we're generating suggestions
            params: a dict with all params to control how we generate suggestions
                * minimally this will have a `method` corresponding to a method in class

        Returns:
            list of frame suggestions
        """

        # map from method param value to corresponding class method
        method_functions = dict(
                                strides=cls.strides,
                                random=cls.random,
                                pca=cls.pca_cluster,
                                hog=cls.hog,
                                brisk=cls.brisk,
                                )

        method = params["method"]
        if method_functions.get(method, None) is not None:
            return method_functions[method](video=video, **params)
        else:
            print(f"No {method} method found for generating suggestions.")

    # Functions corresponding to "method" param

    @classmethod
    def strides(cls, video, per_video=20, **kwargs):
        suggestions = list(range(0, video.frames, video.frames//per_video))
        suggestions = suggestions[:per_video]
        return suggestions

    @classmethod
    def random(cls, video, per_video=20, **kwargs):
        import random

        suggestions = random.sample(range(video.frames), per_video)
        return suggestions

    @classmethod
    def pca_cluster(cls, video, initial_samples, **kwargs):

        sample_step = video.frames//initial_samples
        feature_stack, frame_idx_map = cls.frame_feature_stack(video, sample_step)

        result = cls.feature_stack_to_suggestions(
                    feature_stack, frame_idx_map, **kwargs)

        return result

    @classmethod
    def brisk(cls, video, sample_step, **kwargs):

        feature_stack, frame_idx_map = cls.brisk_feature_stack(video, sample_step)

        result = cls.feature_stack_to_suggestions(
                    feature_stack, frame_idx_map, **kwargs)

        return result

    @classmethod
    def hog(
            cls, video,
            clusters=5,
            per_cluster=5,
            sample_step=5,
            pca_components=50,
            interleave=True,
            **kwargs):

        feature_stack, frame_idx_map = cls.hog_feature_stack(video, sample_step)

        result = cls.feature_stack_to_suggestions(
                    feature_stack, frame_idx_map,
                    clusters=clusters, per_cluster=per_cluster,
                    pca_components=pca_components,
                    interleave=interleave,
                    **kwargs)

        return result

    # Functions for building "feature stack", the (samples * features) matrix
    # These are specific to the suggestion method

    @classmethod
    def frame_feature_stack(cls, video:Video, sample_step:int = 5) -> tuple:
        sample_count = video.frames//sample_step

        factor = cls.get_scale_factor(video)

        frame_idx_map = [None] * sample_count
        flat_stack = np.zeros((sample_count, video.height//factor*video.width//factor*video.channels))

        for i in range(sample_count):
            frame_idx = i * sample_step

            img = video[frame_idx].squeeze()
            multichannel = (video.channels > 1)
            img = rescale(img, scale=.5, anti_aliasing=True, multichannel=multichannel)

            flat_stack[i] = img.flatten()
            frame_idx_map[i] = frame_idx
        return (flat_stack, frame_idx_map)

    @classmethod
    def brisk_feature_stack(cls, video:Video, sample_step:int = 5) -> tuple:
        brisk = cv2.BRISK_create()

        factor = cls.get_scale_factor(video)

        feature_stack = None
        frame_idx_map = []
        for frame_idx in range(0, video.frames, sample_step):
            img = video[frame_idx][0]
            img = cls.resize(img, factor)
            kps, descs = brisk.detectAndCompute(img, None)

            # img2 = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)
            if descs is not None:
                if feature_stack is None:
                    feature_stack = descs
                else:
                    feature_stack = np.concatenate((feature_stack, descs))
                frame_idx_map.extend([frame_idx] * descs.shape[0])

        return (feature_stack, frame_idx_map)

    @classmethod
    def hog_feature_stack(cls, video:Video, sample_step:int = 5) -> tuple:
        sample_count = video.frames//sample_step

        hog = cv2.HOGDescriptor()

        factor = cls.get_scale_factor(video)
        first_hog = hog.compute(cls.resize(video[0][0], factor))
        hog_size = first_hog.shape[0]

        frame_idx_map = [None] * sample_count
        flat_stack = np.zeros((sample_count, hog_size))

        for i in range(sample_count):
            frame_idx = i * sample_step
            img = video[frame_idx][0]
            img = cls.resize(img, factor)
            flat_stack[i] = hog.compute(img).transpose()[0]
            frame_idx_map[i] = frame_idx

        return (flat_stack, frame_idx_map)

    # Functions for making suggestions based on "feature stack"
    # These are common for all suggestion methods

    @staticmethod
    def to_frame_idx_list(selected_list:list, frame_idx_map:dict) -> list:
        """Convert list of row indexes to list of frame indexes."""
        return list(map(lambda x: frame_idx_map[x], selected_list))

    @classmethod
    def feature_stack_to_suggestions(
            cls,
            feature_stack, frame_idx_map,
            return_clusters=False,
            **kwargs):
        """
        Turns a feature stack matrix into a list of suggested frames.

        Args:
            feature_stack: (n * features) matrix
            frame_idx_map: n-length vector which gives frame_idx for each row in feature_stack
            return_clusters (optional): return the intermediate result for debugging
                i.e., a list that gives the list of suggested frames for each cluster
        """

        selected_by_cluster = cls.feature_stack_to_clusters(
                feature_stack=feature_stack,
                frame_idx_map=frame_idx_map,
                **kwargs)

        if return_clusters: return selected_by_cluster

        selected_list = cls.clusters_to_list(
                selected_by_cluster=selected_by_cluster,
                **kwargs)

        return selected_list

    @classmethod
    def feature_stack_to_clusters(
            cls,
            feature_stack,
            frame_idx_map,
            clusters=5,
            per_cluster=5,
            pca_components=50,
            **kwargs):
        """
        Turns feature stack matrix into list (per cluster) of list of frame indexes.

        Args:
            feature_stack: (n * features) matrix
            clusters: number of clusters
            per_clusters: how many suggestions to take from each cluster (at most)
            pca_components: for reducing feature space before clustering

        Returns:
            list of lists
            for each cluster, a list of frame indexes
        """

        stack_height = feature_stack.shape[0]

        # limit number of components by number of samples
        pca_components = min(pca_components, stack_height)

        # get features for each frame
        pca = PCA(n_components=pca_components)
        flat_small = pca.fit_transform(feature_stack)

        # cluster based on features
        kmeans = KMeans(n_clusters=clusters)
        row_labels = kmeans.fit_predict(flat_small)

        # take samples from each cluster
        selected_by_cluster = []
        selected_set = set()
        for i in range(clusters):
            cluster_items,  = np.where(row_labels==i)

            # convert from row indexes to frame indexes
            cluster_items = cls.to_frame_idx_list(cluster_items, frame_idx_map)

            # remove items from cluster_items if they've already been sampled for another cluster
            # TODO: should this be controlled by a param?
            cluster_items = list(set(cluster_items) - selected_set)

            # pick [per_cluster] items from this cluster
            samples_from_bin = np.random.choice(cluster_items, min(len(cluster_items), per_cluster), False)
            samples_from_bin.sort()
            selected_by_cluster.append(samples_from_bin)
            selected_set = selected_set.union( set(samples_from_bin) )

        return selected_by_cluster

    @classmethod
    def clusters_to_list(cls, selected_by_cluster, interleave:bool = True, **kwargs) -> list:
        """
        Turns list (per cluster) of lists of frame index into single list of frame indexes.

        Args:
            selected_by_cluster: the list of lists of row indexes
            frame_idx_map: map from row index to frame index
            interleave: whether we want to interleave suggestions from clusters

        Returns:
            list of frame index
        """

        if interleave:
            # cycle clusters
            all_selected = itertools.chain.from_iterable(itertools.zip_longest(*selected_by_cluster))
            # remove Nones and convert back to list
            all_selected = list(filter(lambda x:x is not None, all_selected))
        else:
            all_selected = list(itertools.chain.from_iterable(selected_by_cluster))
            all_selected.sort()

        all_selected = [int(x) for x in all_selected]

        return all_selected

    # Utility functions

    @classmethod
    def get_scale_factor(cls, video) -> int:
        factor = 1
        if cls.rescale:
            largest_dim = max(video.height, video.width)
            factor = 1
            while largest_dim/factor > cls.rescale_below:
                factor += 1
        return factor

    @classmethod
    def resize(cls, img, factor) -> np.ndarray:
        h, w, _ = img.shape
        if factor != 1:
            return cv2.resize(img, (h//factor, w//factor))
        else:
            return img

if __name__ == "__main__":
    # load some images
    filename = "tests/data/videos/centered_pair_small.mp4"
    filename = "files/190605_1509_frame23000_24000.sf.mp4"
    video = Video.from_filename(filename)

    debug=False

    x = VideoFrameSuggestions.hog(video=video, sample_step=20,
                clusters=5, per_cluster=5,
                return_clusters=debug)
    print(x)

    if debug:

        import matplotlib.pyplot as plt

        rows = len(x)
        cols = max((len(r) for r in x))

        fig, ax = plt.subplots(rows, cols)
        for r in range(rows):
            for c in range(cols):
                if c < len(x[r]):
                    frame_idx = x[r][c]
                    frame = video[frame_idx]
                    ax[r, c].imshow(frame[0].squeeze(), cmap="gray")
                    ax[r, c].set_title(f"frame {frame_idx}")
                ax[r, c].axis("off")
        plt.show()

    # brisk = cv2.BRISK_create()

    # feature_stack = None
    # frame_idx_map = []
    # for frame_idx in range(1, video.frames, 200):
    #     img = video[frame_idx][0]

    #     kps, descs = brisk.detectAndCompute(img, None)

    #     # img2 = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

    #     if feature_stack is None:
    #         feature_stack = descs
    #     else:
    #         feature_stack = np.concatenate((feature_stack, descs))
    #     frame_idx_map.extend([frame_idx] * descs.shape[0])

    # print(f"final result: {feature_stack.shape}")
    # print(frame_idx_map)

# for a,b in zip(kps,descs):
#   print(f"{a}: {b}")
# print(len(kps))
# print(len(descs))

# foo = cv2.FeatureDetector_create("SIFT")
# surf = cv2.xfeatures2d.SURF_create(400)
# kp, des = surf.detectAndCompute(video[13], None)
# print(len(kp))
# print(des.shape)

# print(VideoFrameSuggestions.suggest(video, dict(method="pca")))