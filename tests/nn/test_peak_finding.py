import numpy as np
import tensorflow as tf
from sleap.nn import peak_finding


class PeakFindingTests(tf.test.TestCase):
    def test_find_local_peaks(self):

        img_shape = [2, 8, 8, 3]
        peak_subs_gt = tf.constant(
            [[0, 2, 2, 0],
             [0, 4, 6, 0],
             [1, 2, 2, 0],
             [1, 3, 2, 1],
            ],
        )
        peak_vals_gt = tf.ones(peak_subs_gt.shape[0])

        img = tf.scatter_nd(
            peak_subs_gt,
            peak_vals_gt,
            img_shape
        )

        peak_subs, peak_vals = peak_finding.find_local_peaks(img)

        self.assertAllEqual(peak_subs, peak_subs_gt)
        self.assertAllEqual(peak_vals, peak_vals_gt)

    def test_find_global_peaks(self):

        img_shape = [2, 8, 8, 3]
        peak_subs_gt = tf.constant(
            [[0, 1, 2, 0],
             [0, 2, 3, 1],
             [0, 3, 4, 2],
             [1, 5, 6, 0],
             [1, 7, 0, 1],
             [1, 1, 2, 2],
            ],
        )
        peak_vals_gt = tf.ones(peak_subs_gt.shape[0])

        img = tf.scatter_nd(
            peak_subs_gt,
            peak_vals_gt,
            img_shape
        )

        peak_subs, peak_vals = peak_finding.find_global_peaks(img)

        self.assertAllEqual(peak_subs, peak_subs_gt)
        self.assertAllEqual(peak_vals, peak_vals_gt)
