"""This module contains mock inference classes that simulate real processing.

These classes all mimic the signatures of their corresponding inference classes, but
will produce the ground truth values by passing it to their set_gt() method before
calling the prediction method.

These are useful for testing, benchmarking and evaluation.
"""

import attr
import numpy as np
import tensorflow as tf

from sleap.nn import data
from sleap.nn import paf_grouping
from sleap.nn import peak_finding
from sleap.nn import utils


@attr.s
class MockCentroidPeakFinder:
    ctr_gt = attr.ib(default=None)

    def set_gt(self, ctr_gt):
        self.ctr_gt = ctr_gt

    def predict(self, imgs):
        centroids = utils.points_list_to_subs(self.ctr_gt).astype("float32")
        centroid_vals = np.ones((len(centroids),), dtype="float32")
        return centroids, centroid_vals


@attr.s
class MockConfmapPeakFinder:
    pts_gt = attr.ib(default=None)

    def set_gt(self, pts_gt):
        self.pts_gt = pts_gt

    def predict_rps(self, rps):
        peaks = []
        patch_inds = []

        for patch_ind in range(len(rps.patches)):

            sample_ind = rps.sample_inds[patch_ind]
            bbox_patch = rps.bboxes[patch_ind]  # (y1, x1, y2, x2)

            pts_patch = self.pts_gt[sample_ind].numpy()

            peaks_patch = utils.points_list_to_subs(
                [pts_patch], initial_sample_ind=sample_ind
            )

            if len(peaks_patch) > 0:

                in_patch = (
                    (peaks_patch[:, 1] > bbox_patch[0])
                    & (peaks_patch[:, 2] > bbox_patch[1])
                    & (peaks_patch[:, 1] < bbox_patch[2])
                    & (peaks_patch[:, 2] < bbox_patch[3])
                )

                if in_patch.any():
                    peaks_patch = peaks_patch[in_patch, :]

                    peaks.append(peaks_patch)
                    patch_inds.append(np.full((len(peaks_patch),), patch_ind))

        peaks = np.concatenate(peaks, axis=0).astype("float32")
        peak_vals = np.ones((len(peaks),), dtype="float32")
        patch_inds = np.concatenate(patch_inds, axis=0).astype("int32")

        region_peaks = peak_finding.RegionPeakSet(
            peaks=peaks, peak_vals=peak_vals, patch_inds=patch_inds
        )

        return region_peaks


@attr.s
class MockPAFGrouper(paf_grouping.PAFGrouper):
    inference_model = attr.ib(default=None)
    _skeleton = attr.ib(default=None)

    output_scale = attr.ib(default=1.0)
    distance_threshold = attr.ib(default=5.0)

    pafs_gt = attr.ib(default=None)

    def __attrs_post_init__(self):
        self._paf_scale = self.output_scale
        self._edge_types = [
            paf_grouping.EdgeType(src_node_ind, dst_node_ind)
            for src_node_ind, dst_node_ind in self.skeleton.edge_inds
        ]

    def set_gt(self, pts_gt, rps):
        pafs = []
        for patch_ind in range(len(rps.patches)):

            sample_ind = rps.sample_inds[patch_ind]
            bbox_patch = rps.bboxes[patch_ind]  # (y1, x1, y2, x2)

            pts_patch = pts_gt[sample_ind].numpy()
            pts_patch_bbox = pts_patch - np.array([[[bbox_patch[1], bbox_patch[0]]]])
            paf = data.make_pafs(
                points=pts_patch_bbox,
                edges=np.array(self.skeleton.edge_inds),
                image_height=rps.patches.shape[1],
                image_width=rps.patches.shape[2],
                output_scale=self.output_scale,
                distance_threshold=self.distance_threshold,
            )
            pafs.append(paf)
        self.pafs_gt = tf.stack(pafs, axis=0)

    def preproc(self, imgs=None, *args, **kwargs):
        return imgs

    def batched_inference(self, *args, **kwargs):
        return self.pafs_gt
