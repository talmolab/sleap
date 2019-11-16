import numpy as np
import tensorflow as tf

from sleap.nn import region_proposal


class RegionProposalTests(tf.test.TestCase):
    def test_make_centered_bboxes(self):
        centroids = np.array(
            [[10, 10], [20, 30], [0, 0],], dtype="float"
        )  # center_offset fails if dtype=int

        boxes = region_proposal.make_centered_bboxes(
            centroids, box_length=5, center_offset=True
        )

        self.assertLen(boxes, len(centroids))
        self.assertAllEqual(boxes[0], [7.5, 7.5, 12.5, 12.5])
        self.assertAllEqual(boxes[1], [17.5, 27.5, 22.5, 32.5])
        self.assertAllEqual(boxes[2], [-2.5, -2.5, 2.5, 2.5])  # is this wrong?

    def test_make_more_centered_bboxes(self):
        # Try with sample/channel numbers
        centroids = np.array(
            [[0, 10, 10, 0], [1, 20, 30, 2],], dtype="float"
        )  # center_offset fails if dtype=int

        # Try without offset
        boxes = region_proposal.make_centered_bboxes(
            centroids, box_length=10, center_offset=False
        )

        self.assertLen(boxes, len(centroids))
        self.assertAllEqual(boxes[0], [5, 5, 15, 15])
        self.assertAllEqual(boxes[1], [15, 25, 25, 35])

    def test_nms_bboxes(self):
        centroids = np.array(
            [[10, 10], [10, 12], [20, 10], [10, 30]], dtype="float"
        )  # center_offset fails if dtype=int

        boxes = region_proposal.make_centered_bboxes(centroids, box_length=10)
        box_scores = np.ones(len(boxes))

        fewer_boxes = region_proposal.nms_bboxes(boxes, box_scores, iou_threshold=0.6)

        # Make sure one box was suppressed
        self.assertLen(fewer_boxes, len(centroids) - 1)

        # Try using score to control which box we keep
        box_scores = np.array([1, 2, 1, 1])
        other_boxes = region_proposal.nms_bboxes(boxes, box_scores, iou_threshold=0.6)

        self.assertLen(other_boxes, len(centroids) - 1)

        # Make sure a different box was suppressed
        self.assertNotAllEqual(fewer_boxes, other_boxes)

        # Try with higher iou threshold
        more_boxes = region_proposal.nms_bboxes(boxes, box_scores, iou_threshold=0.8)

        # Make sure we now kept all the boxes
        self.assertLen(more_boxes, len(centroids))

    def test_generate_merged_bboxes(self):
        centroids = np.array(
            [[10, 10], [10, 12], [20, 10], [10, 30]], dtype="float"
        )  # center_offset fails if dtype=int

        boxes = region_proposal.make_centered_bboxes(centroids, box_length=10)
        box_scores = np.ones(len(boxes))

        merged_boxes, merged_scores = region_proposal.generate_merged_bboxes(
            boxes,
            box_scores,
            merged_box_length=20,
        )

        # Make sure first four boxes match
        self.assertAllEqual(boxes, merged_boxes[:4])

        # Make sure last (merged) box has score of 2
        self.assertEqual(merged_scores[-1], 2)

        # Make sure last (merged) box has correct size and offset
        self.assertAllEqual(merged_boxes[-1], [0.5, 1.5, 20.5, 21.5])

    def skip_test_overlapping_merged_bboxes(self):

        # Two pairs of overlapping boxes such that the larger boxes made from
        # merging each pair will themselves overlap.
        centroids = np.array(
            [[10, 10], [10, 12], [20, 10], [20, 12]], dtype="float"
        )  # center_offset fails if dtype=int

        boxes = region_proposal.make_centered_bboxes(centroids, box_length=16)
        box_scores = np.ones(len(boxes))

        # Merge and NMS
        merged_boxes, merged_scores = region_proposal.generate_merged_bboxes(
            boxes, box_scores, merged_box_length=32,
        )
        # self.assertLen(merged_boxes, 2)
        merged_boxes = region_proposal.nms_bboxes(merged_boxes, merged_scores)
        self.assertLen(merged_boxes, 2)

        # Reset the scores
        merged_scores = np.ones(len(merged_boxes))

        merged_boxes = region_proposal.nms_bboxes(merged_boxes, merged_scores)
        self.assertLen(merged_boxes, 1)

        #
        # merged_boxes, merged_scores = \
        #     region_proposal.generate_merged_bboxes(
        #         merged_boxes, merged_scores,
        #         merged_box_length=40,
        #     )

        # merged_boxes = region_proposal.nms_bboxes(merged_boxes, merged_scores, iou_threshold=.1)
        # self.assertLen(merged_boxes, 1)

    def test_normalize_bboxes(self):
        centroids = np.array(
            [[25, 15], [45, 15],], dtype="float"
        )  # center_offset fails if dtype=int

        boxes = region_proposal.make_centered_bboxes(
            centroids, box_length=10, center_offset=False
        )
        print(boxes)
        normed_boxes = region_proposal.normalize_bboxes(
            boxes, img_height=200, img_width=100
        )

        normed_gt = np.array([[0.1, 0.1, 0.15, 0.2], [0.2, 0.1, 0.25, 0.2],])

        self.assertAllClose(normed_boxes, normed_gt, atol=0.01)
