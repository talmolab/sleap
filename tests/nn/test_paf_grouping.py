import numpy as np
import tensorflow as tf

from sleap.nn import paf_grouping
from sleap import skeleton


class PafGroupingTests(tf.test.TestCase):
    def setUp(self):

        # Matrix with desired src, dst, and edge idx as columns.
        self.groups_gt = np.array(
            [
                [0, 1, 0],  # peak 0 -> peak 1, edge type 0
                [0, 2, 0],  # peak 0 -> peak 2, edge type 0
                [1, 3, 1],  # peak 1 -> peak 3, edge type 1
                [2, 3, 1],  # peak 2 -> peak 3, edge type 1
            ]
        )

        # Extract columns as separate vectors
        self.src_idxs = self.groups_gt[:, 0]
        self.dst_idxs = self.groups_gt[:, 1]
        self.edge_idxs = self.groups_gt[:, 2]

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

        # Make sure we got the correct indices
        self.assertAllEqual(src_peaks, self.src_idxs)
        self.assertAllEqual(dst_peaks, self.dst_idxs)
        self.assertAllEqual(edge_idxs, self.edge_idxs)

    def test_make_line_segments(self):
        peaks = np.array(
            [
                [0, 10, 10, 0],  # (stop black from reformatting matrix)
                [0, 10, 30, 1],
                [0, 30, 10, 1],
                [0, 20, 0, 2],
            ]
        )

        src_peaks = peaks[self.src_idxs]
        dst_peaks = peaks[self.dst_idxs]

        # Make sure we get correct number of points on line segments
        n_points = 3
        lines = paf_grouping.make_line_segments(
            src_peaks, dst_peaks, self.edge_idxs, n_points
        )
        self.assertLen(lines, n_points * len(self.groups_gt))

        # Make sure we get correct number of points on line segments
        n_points = 5
        lines = paf_grouping.make_line_segments(
            src_peaks, dst_peaks, self.edge_idxs, n_points
        )
        self.assertLen(lines, n_points * len(self.groups_gt))

        # Extract points for each line so it's easy to check them
        line_list = []
        for i in range(len(self.groups_gt)):
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

        # Make the paf values all different so they're easy to check
        pafs = tf.reshape(tf.range(img_len), img_shape)

        peaks = np.array(
            [
                [0, 0, 0, 0],  # (stop black from reformatting matrix)
                [0, 10, 30, 1],
                [0, 30, 10, 1],
                [0, 39, 79, 2],
            ]
        )

        src_peaks = peaks[self.src_idxs]
        dst_peaks = peaks[self.dst_idxs]

        # Make sure we get correct number of points on line segments
        n_points = 3
        lines = paf_grouping.make_line_segments(
            src_peaks, dst_peaks, self.edge_idxs, n_points
        )

        vectors = paf_grouping.gather_line_vectors(pafs, lines, n_points)
        self.assertLen(vectors, len(self.groups_gt))
        self.assertLen(vectors[0], n_points)

        # First point is (0, 0), so value should be 1 (for y) and 0 (for x)
        self.assertAllEqual(vectors[0, 0], np.array([1, 0]))

        # Last point is (39, 79) in channel 2, so values should be
        # (2 * 40 * 80 * 2) - 1 for y, (2 * 40 * 80 * 2) - 2 for x
        self.assertAllEqual(
            vectors[-1, -1], np.array([img_len / 2 - 1, img_len / 2 - 2])
        )

    def test_score_connection_candidates(self):

        # Make peaks so that we can check different directions
        # Format is image, y (row), x (col), channel (peak type)
        peaks = np.array(
            [
                [0, 10, 10, 0],  # (stop black from reformatting matrix)
                [0, 10, 30, 1],
                [0, 30, 10, 1],
                [0, 20, 0, 2],
            ]
        )

        src_peaks = peaks[self.src_idxs]
        dst_peaks = peaks[self.dst_idxs]

        # PAF matrix with 1 for every x value and 0 for every y value
        # This means we want edges which positive along the x axis
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
        self.assertAllEqual(correct, [1, 0, 0, 0])

        # Make sure max_edge_length penalizes longer edge.
        # Only third edge, from (10, 30) to (20, 0), should be penalized.
        penalized_score, correct = paf_grouping.score_connection_candidates(
            src_peaks, dst_peaks, paf_vals, max_edge_length=20
        )
        self.assertEqual(penalized_score[0], score[0])
        self.assertEqual(penalized_score[1], score[1])
        self.assertLess(penalized_score[2], score[2])
        self.assertEqual(penalized_score[3], score[3])

    def test_max_edge_length_score(self):
        peaks = np.array(
            [
                [0, 10, 10, 0],  # (stop black from reformatting matrix)
                [0, 20, 20, 1],
                [0, 50, 50, 1],
                [0, 20, 0, 2],
            ]
        )

        src_peaks = peaks[self.src_idxs]
        dst_peaks = peaks[self.dst_idxs]

        # Matrix with 1 for every x value and 0 for every y value
        np_zeros_ones = np.stack([np.zeros((4, 2)), np.ones((4, 2))], axis=-1)
        paf_vals = tf.constant(np_zeros_ones, dtype="float32")

        score, correct = paf_grouping.score_connection_candidates(
            src_peaks, dst_peaks, paf_vals, max_edge_length=20
        )
        # Edges 0 and 1 are along the same line, but make sure that edge 1
        # was penalized for its length
        self.assertGreater(score[0], score[1])

    def test_min_edge_length_score(self):
        peaks = np.array(
            [
                [0, 10, 10, 0],  # (stop black from reformatting matrix)
                [0, 20, 20, 1],
                [0, 10, 50, 1],
                [0, 20, 0, 2],
            ]
        )

        src_peaks = peaks[self.src_idxs]
        dst_peaks = peaks[self.dst_idxs]

        # Matrix with 1 for every x value and 0 for every y value
        np_zeros_ones = np.stack([np.zeros((4, 2)), np.ones((4, 2))], axis=-1)
        paf_vals = tf.constant(np_zeros_ones, dtype="float32")

        # If we didn't penalize length, then we'd see approximately:
        # [.707, 1, -1, -1]
        # With the penalty, the scores are instead close to:
        # [.707, .5, -1, -1.5]

        score, correct = paf_grouping.score_connection_candidates(
            src_peaks, dst_peaks, paf_vals, max_edge_length=20, min_edge_score=0.8,
        )

        # Since min_edge_score is applied before discount, only edge 1
        # (with original score of 1) should be considered correct.
        self.assertAllEqual(correct, [0, 1, 0, 0])


class PafConnectionTests(tf.test.TestCase):
    def setUp(self):
        self.peaks = np.array(
            [
                [0, 10, 10, 0],  # src
                [0, 20, 20, 1],  # better dst candidate from (10, 10)
                [0, 22, 18, 1],  # worse dst candidate from (10, 10)
                [0, 20, 0, 2],  # better dst from src of (22, 18)
                [1, 0, 0, 1],
                [1, 30, 30, 2],  # over max length from (0, 0)
            ]
        )

        # Use find_peak_pairs to get edges for these peaks.
        self.edges = [paf_grouping.EdgeType(0, 1), paf_grouping.EdgeType(1, 2)]
        (
            self.src_peak_idxs,
            self.dst_peak_idxs,
            self.edge_idxs,
        ) = paf_grouping.find_peak_pairs(self.peaks, self.edges)

        edge_count = self.src_peak_idxs.shape[0]
        self.assertEqual(edge_count, 5)

        # PAF matrix with 1 for every x value and 0 for every y value
        # This means we want edges which positive along the x axis
        np_zeros_ones = np.stack(
            [np.zeros((edge_count, 2)), np.ones((edge_count, 2))], axis=-1
        )
        paf_vals = tf.constant(np_zeros_ones, dtype="float32")

        src_peaks = self.peaks[self.src_peak_idxs]
        dst_peaks = self.peaks[self.dst_peak_idxs]

        # Use score_connection_candidates to get scores
        score, correct = paf_grouping.score_connection_candidates(
            src_peaks, dst_peaks, paf_vals, max_edge_length=20
        )

        self.score = score.numpy()  # filter_connection_candidates doesn't take tensor

    def test_filter_connection_candidates(self):
        connections = paf_grouping.filter_connection_candidates(
            src_peak_inds=self.src_peak_idxs,
            dst_peak_inds=self.dst_peak_idxs,
            connection_scores=self.score,
            edge_type_inds=self.edge_idxs,
            edge_types=self.edges,
        )

        # Make sure we got the right number of edge types
        self.assertLen(connections, 2)

        # Make sure we got the right number of candidate edges for each type
        edge_0_connections = connections[self.edges[0]]
        edge_1_connections = connections[self.edges[1]]

        self.assertLen(edge_0_connections, 1)
        self.assertLen(edge_1_connections, 2)  # for samples 0 and 1

        # Make sure we got the best candidate
        self.assertEqual(edge_0_connections[0].src_peak_ind, 0)  # from peak 0
        self.assertEqual(edge_0_connections[0].dst_peak_ind, 1)  # to peak 1 (not 2)

        self.assertEqual(edge_1_connections[0].src_peak_ind, 4)
        self.assertEqual(edge_1_connections[0].dst_peak_ind, 5)

        self.assertEqual(edge_1_connections[1].src_peak_ind, 2)
        self.assertEqual(edge_1_connections[1].dst_peak_ind, 3)

    def test_assign_connections_to_instances(self):
        connections = paf_grouping.filter_connection_candidates(
            src_peak_inds=self.src_peak_idxs,
            dst_peak_inds=self.dst_peak_idxs,
            connection_scores=self.score,
            edge_type_inds=self.edge_idxs,
            edge_types=self.edges,
        )

        assignments = paf_grouping.assign_connections_to_instances(connections)

        # Make sure each peak got an assignment
        self.assertLen(assignments, len(self.peaks))

        # Make sure we got correct number of instance groupings (i.e., 3)
        self.assertLen(set(assignments.values()), 3)

        # One instance goes with connection from peak 0 -> peak 1 (not 2)
        self.assertEqual(
            assignments[paf_grouping.PeakID(0, 0)],
            assignments[paf_grouping.PeakID(1, 1)],
        )

        # One instance goes with connection from peak 2 (not 1) -> peak 3
        self.assertEqual(
            assignments[paf_grouping.PeakID(1, 2)],
            assignments[paf_grouping.PeakID(2, 3)],
        )

        # One instance goes with connection from peak 4 -> peak 5
        self.assertEqual(
            assignments[paf_grouping.PeakID(1, 4)],
            assignments[paf_grouping.PeakID(2, 5)],
        )

    def test_create_predicted_instances(self):
        sk = skeleton.Skeleton()
        sk.add_node("a")
        sk.add_node("b")
        sk.add_node("c")
        sk.add_edge("a", "b")
        sk.add_edge("b", "c")

        connections = paf_grouping.filter_connection_candidates(
            src_peak_inds=self.src_peak_idxs,
            dst_peak_inds=self.dst_peak_idxs,
            connection_scores=self.score,
            edge_type_inds=self.edge_idxs,
            edge_types=self.edges,
        )

        assignments = paf_grouping.assign_connections_to_instances(connections)

        peak_vals = np.arange(len(self.peaks))

        instances = paf_grouping.create_predicted_instances(
            peaks=self.peaks,
            peak_vals=peak_vals,
            connections=connections,
            instance_assignments=assignments,
            skeleton=sk,
        )

        # Make sure we got the right number of instances
        self.assertLen(instances, 3)

        instances.sort(key=lambda inst: inst.score)

        # Worse instance should just have b (22, 18) -> c (20, 0)
        self.assertLen(instances[0].points, 2)
        self.assertEqual(instances[0]["b"].y, 22)
        self.assertEqual(instances[0]["c"].y, 20)

        # Next instance should just have b (0, 0) -> c (30, 30)
        self.assertLen(instances[1].points, 2)
        self.assertEqual(instances[1]["b"].y, 0)
        self.assertEqual(instances[1]["c"].y, 30)

        # Best instance should just have a (10, 10) -> b (20 ,20)
        self.assertLen(instances[2].points, 2)
        self.assertEqual(instances[2]["a"].y, 10)
        self.assertEqual(instances[2]["b"].y, 20)
