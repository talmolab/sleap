import numpy as np
import tensorflow as tf
from sleap.nn import paf_grouping


class PafGroupingTests(tf.test.TestCase):
    def test_find_peak_pairs(self):

        peaks = np.array(
            [
                [0, 1, 2, 0],
                [0, 3, 3, 1],
                [0, 3, 5, 1],
                [0, 0, 0, 2],
                [1, 0, 0, 0],
                [1, 3, 3, 2],
            ]
        )
        edges = [paf_grouping.EdgeType(0, 1), paf_grouping.EdgeType(1, 2)]
        src_peaks, dst_peaks, edge_idxs = paf_grouping.find_peak_pairs(peaks, edges)

        # src should be: array([0, 0, 1, 2, 4])
        # dst should be: array([1, 2, 3, 3, 5])
        # edg should be: array([0, 0, 1, 1, 0]))

        # Matrix with desired src, dst, and edge idx as columns.
        groups_gt = np.array(
            [
                [0, 1, 0],  # peak 0 -> peak 1, edge type 0
                [0, 2, 0],  # peak 0 -> peak 2, edge type 0
                [1, 3, 1],  # peak 1 -> peak 3, edge type 1
                [2, 3, 1],  # peak 2 -> peak 3, edge type 1
            ]
        )

        # Make sure we got the correct indices
        self.assertAllEqual(src_peaks, groups_gt[:, 0])
        self.assertAllEqual(dst_peaks, groups_gt[:, 1])
        self.assertAllEqual(edge_idxs, groups_gt[:, 2])

    def test_make_line_segments(self):

        groups_gt = np.array(
            [
                [0, 1, 0],  # peak 0 -> peak 1, edge type 0
                [0, 2, 0],  # peak 0 -> peak 2, edge type 0
                [1, 3, 1],  # peak 1 -> peak 3, edge type 1
                [2, 3, 1],  # peak 2 -> peak 3, edge type 1
            ]
        )
        src_idxs = groups_gt[:, 0]
        dst_idxs = groups_gt[:, 1]
        edge_idxs = groups_gt[:, 2]

        peaks = np.array(
            [[0, 10, 10, 0], [0, 10, 30, 1], [0, 30, 10, 1], [0, 20, 0, 2],]
        )

        src_peaks = peaks[src_idxs]
        dst_peaks = peaks[dst_idxs]

        # Make sure we get correct number of points on line segments
        n_points = 3
        lines = paf_grouping.make_line_segments(
            src_peaks, dst_peaks, edge_idxs, n_points
        )
        self.assertLen(lines, n_points * len(groups_gt))

        # Make sure we get correct number of points on line segments
        n_points = 5
        lines = paf_grouping.make_line_segments(
            src_peaks, dst_peaks, edge_idxs, n_points
        )
        self.assertLen(lines, n_points * len(groups_gt))

        # Extract points for each line so it's easy to check them
        line_list = []
        for i in range(len(groups_gt)):
            line_list.append(lines[i * n_points : (i + 1) * n_points, 1:3])

        gt_line_0 = np.array(
            [
                [10, 10],  # line from (10, 10) to (10, 30)
                [10, 15],
                [10, 20],
                [10, 25],
                [10, 30],
            ]
        )

        gt_line_1 = np.array(
            [
                [10, 10],  # line from (10, 10) to (30, 10)
                [15, 10],
                [20, 10],
                [25, 10],
                [30, 10],
            ]
        )

        gt_line_3 = np.array(
            [
                [30, 10],  # line from (30, 10) to (20, 0)
                [28, 8],
                [25, 5],
                [22, 2],
                [20, 0],
            ]
        )

        # Make sure the points are correct
        self.assertAllEqual(line_list[0], gt_line_0)
        self.assertAllEqual(line_list[1], gt_line_1)
        self.assertAllEqual(line_list[3], gt_line_3)

    def test_gather_line_vectors(self):
        img_shape = (2, 40, 80, 4)
        img_len = np.product(img_shape)
        pafs = tf.reshape(tf.range(img_len), img_shape)

        groups_gt = np.array(
            [
                [0, 1, 0],  # peak 0 -> peak 1, edge type 0
                [0, 2, 0],  # peak 0 -> peak 2, edge type 0
                [1, 3, 1],  # peak 1 -> peak 3, edge type 1
                [2, 3, 1],  # peak 2 -> peak 3, edge type 1
            ]
        )
        src_idxs = groups_gt[:, 0]
        dst_idxs = groups_gt[:, 1]
        edge_idxs = groups_gt[:, 2]

        peaks = np.array(
            [[0, 0, 0, 0], [0, 10, 30, 1], [0, 30, 10, 1], [0, 39, 79, 2],]
        )

        src_peaks = peaks[src_idxs]
        dst_peaks = peaks[dst_idxs]

        # Make sure we get correct number of points on line segments
        n_points = 3
        lines = paf_grouping.make_line_segments(
            src_peaks, dst_peaks, edge_idxs, n_points
        )

        vectors = paf_grouping.gather_line_vectors(pafs, lines, n_points)
        self.assertLen(vectors, len(groups_gt))
        self.assertLen(vectors[0], n_points)

        # First point is (0, 0), so value should be 1 (for y) and 0 (for x)
        self.assertAllEqual(vectors[0, 0], np.array([1, 0]))

        # Last point is (39, 79) in channel 2, so values should be
        # (2 * 40 * 80 * 2) - 1 for y, (2 * 40 * 80 * 2) - 2 for x
        self.assertAllEqual(
            vectors[-1, -1], np.array([img_len / 2 - 1, img_len / 2 - 2])
        )

    def test_score_connection_candidates(self):
        groups_gt = np.array(
            [
                [0, 1, 0],  # peak 0 -> peak 1, edge type 0
                [0, 2, 0],  # peak 0 -> peak 2, edge type 0
                [1, 3, 1],  # peak 1 -> peak 3, edge type 1
                [2, 3, 1],  # peak 2 -> peak 3, edge type 1
            ]
        )
        src_idxs = groups_gt[:, 0]
        dst_idxs = groups_gt[:, 1]

        # Format is image, y (row), x (col), channel (peak type)
        peaks = np.array(
            [[0, 10, 10, 0], [0, 10, 30, 1], [0, 30, 10, 1], [0, 20, 0, 2],]
        )

        src_peaks = peaks[src_idxs]
        dst_peaks = peaks[dst_idxs]

        # Matrix with 1 for every x value and 0 for every y value
        np_zeros_ones = np.stack([np.zeros((4, 2)), np.ones((4, 2))], axis=-1)
        paf_vals = tf.constant(np_zeros_ones, dtype="float32")

        score, correct = paf_grouping.score_connection_candidates(
            src_peaks, dst_peaks, paf_vals
        )

        approx_score_gt = np.array(
            [
                1.0,  # (10, 10) -> (10, 30) lies along x (columns) so with pafs
                0.0,  # (10, 10) -> (30, 10) lies along y (rows) so orthogonal to pafs
                -0.95,  # (10, 30) -> (20, 0) is mostly against x so close to -1
                -0.71,  # (30, 10) -> (20, 0) is against both x and y, so close to -sqrt(2)
            ]
        )

        self.assertAllClose(score, approx_score_gt, atol=0.01)

        # TODO: test max_edge_length and min_edge_score

    def test_filter_connection_candidates(self):
        pass

    def test_assign_connections_to_instances(self):
        pass

    def test_create_predicted_instances(self):
        pass
