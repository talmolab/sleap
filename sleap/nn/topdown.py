import tensorflow as tf
import numpy as np
import attr
from collections import defaultdict

from sleap.nn import model
from sleap.nn import peak_finding
from sleap.nn import utils

from sleap.skeleton import Skeleton
from sleap.instance import PredictedPoint, PredictedInstance


def get_bbox_offsets(bboxes):
    """Returns the top-left xy coordinates of bboxes."""
    return tf.reverse(bboxes[:, :2], axis=[-1])


def make_predicted_instance(points_array, peak_vals, skeleton):
    if tf.is_tensor(points_array):
        points_array = points_array.numpy()
    if tf.is_tensor(peak_vals):
        peak_vals = peak_vals.numpy()

    predicted_points = dict()
    for point, peak_val, node_name in zip(points_array, peak_vals, skeleton.node_names):
        if np.isnan(point).any():
            continue

        predicted_points[node_name] = PredictedPoint(
            x=point[0], y=point[1], score=peak_val
        )

    instance_score = np.nansum([pt.score for pt in predicted_points.values()])
    instance_score /= len(skeleton.node_names)

    predicted_instance = PredictedInstance(
        skeleton=skeleton, points=predicted_points, score=instance_score
    )

    return predicted_instance


def make_sample_grouped_predicted_instances(
    sample_peak_pts, sample_peak_vals, sample_inds, skeleton
):
    sample_grouped_instances = defaultdict(list)
    for sample_ind, peak_pts, peak_vals in zip(
        sample_inds, sample_peak_pts, sample_peak_vals
    ):
        for inst_peak_pts, inst_peak_vals in zip(peak_pts, peak_vals):
            if not tf.reduce_any(tf.math.is_finite(inst_peak_pts)).numpy():
                continue
            sample_grouped_instances[sample_ind].append(
                make_predicted_instance(inst_peak_pts, inst_peak_vals, skeleton)
            )

    return sample_grouped_instances


@attr.s(auto_attribs=True, eq=False)
class TopDownPeakFinder:
    inference_model: model.InferenceModel
    batch_size: int = 8
    smooth_confmaps: bool = False
    smoothing_kernel_size: int = 5
    smoothing_sigma: float = 3.0
    peak_thresh: float = 0.3
    refine_peaks: bool = True

    def preproc(self, imgs):
        # Scale to model input size.
        imgs = utils.resize_imgs(
            imgs,
            self.inference_model.input_scale,
            common_divisor=2 ** self.inference_model.down_blocks,
        )

        # Convert to float32 and scale values to [0., 1.].
        imgs = utils.normalize_imgs(imgs)

        return imgs

    @tf.function
    def inference(self, imgs):
        cms = self.inference_model.keras_model(imgs)

        if self.smooth_confmaps:
            cms = peak_finding.smooth_imgs(
                cms, kernel_size=self.smoothing_kernel_size, sigma=self.smoothing_sigma
            )

        peak_subs, peak_vals = peak_finding.find_global_peaks(cms)

        if self.refine_peaks:
            peak_subs = peak_finding.refine_peaks_local_direction(cms, peak_subs)

        return tf.concat([peak_subs, tf.expand_dims(peak_vals, axis=1)], axis=1)

    def postproc(self, peak_subs, peak_vals):
        peak_subs /= tf.constant(
            [
                [
                    1,
                    self.inference_model.output_scale,
                    self.inference_model.output_scale,
                    1,
                ]
            ]
        )
        peak_pts = tf.reverse(peak_subs[:, 1:3], axis=[-1])
        peak_pts = peak_pts.numpy()
        peak_pts[peak_vals.numpy().squeeze() < self.peak_thresh, :] = np.nan

        return peak_pts

    def predict_rps(self, rps):

        imgs = self.preproc(rps.patches)

        peak_subs_and_vals, batch_inds = utils.batched_call(
            self.inference, imgs, batch_size=self.batch_size, return_batch_inds=True,
        )
        peak_subs, peak_vals = tf.split(peak_subs_and_vals, [4, 1], axis=1)
        peak_pts = self.postproc(peak_subs, peak_vals)

        # Reconstruct patch indices
        unbatched_patch_inds = tf.cast(
            peak_subs[:, 0] + (batch_inds * self.batch_size), tf.int32
        )

        # Adjust to image coordinates
        patch_offsets = get_bbox_offsets(rps.bboxes)
        patch_peak_pts = tf.RaggedTensor.from_value_rowids(
            peak_pts, unbatched_patch_inds
        )
        patch_peak_vals = tf.RaggedTensor.from_value_rowids(
            tf.squeeze(peak_vals), unbatched_patch_inds
        )
        patch_peak_pts += tf.expand_dims(patch_offsets, axis=1)

        # Reshape into (sample, instance, nodes, xy)
        sample_peak_pts = tf.RaggedTensor.from_value_rowids(
            patch_peak_pts, rps.sample_inds
        )
        sample_peak_vals = tf.RaggedTensor.from_value_rowids(
            patch_peak_vals, rps.sample_inds
        )

        return sample_peak_pts, sample_peak_vals
