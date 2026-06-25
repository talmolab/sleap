"""Tests for sleap.qc.features.duplicate_split.

Covers the split-duplicate detector (one animal split across two instances on
largely disjoint node sets) and the ``duplicate_score`` signal combiner.

The module functions are pure (numpy arrays + a learned-stats dict in, dict /
float out), so most tests build poses directly as numpy arrays. A couple of
tests build real ``sleap_io`` instances to exercise the same arrays the
integration layer would extract.
"""

import numpy as np
import pytest

from sleap.qc.features.duplicate_split import (
    compute_split_duplicate,
    duplicate_score,
)


# A vertical 6-node chain standing in for a head->tail skeleton, edge ~20 px:
# 0 head, 1 neck, 2 thorax, 3 abd1, 4 abd2, 5 tail.
FULL_POSE = np.array(
    [[100, 100], [100, 120], [100, 140], [100, 160], [100, 180], [100, 200]],
    dtype=float,
)

# Learned mean edge lengths keyed by sorted (src, dst) tuples, matching
# DatasetStats.edge_means. Median is 20.0 -> the scale.
EDGE_MEANS = {(0, 1): 20.0, (1, 2): 20.0, (2, 3): 20.0, (3, 4): 20.0, (4, 5): 20.0}

# A score this high is unambiguously a split for these tests; well above any
# sensible operating threshold.
HIGH = 0.5
# A score this low means "not a split".
LOW = 0.1


def front_half(pose=FULL_POSE):
    """Pose with only the front (head-side) nodes visible."""
    p = pose.copy()
    p[3:] = np.nan
    return p


def back_half(pose=FULL_POSE):
    """Pose with only the back (tail-side) nodes visible."""
    p = pose.copy()
    p[:3] = np.nan
    return p


def rotate(pose, degrees, about=(100.0, 150.0)):
    """Rotate a pose about a pivot (preserving NaNs)."""
    th = np.deg2rad(degrees)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    about = np.asarray(about, dtype=float)
    return (pose - about) @ rot.T + about


class TestComputeSplitDuplicatePositive:
    """The detector should fire on one animal split across two instances."""

    def test_complementary_head_tail_split_is_high(self):
        """A = head/front, B = tail/back, disjoint nodes -> high score."""
        result = compute_split_duplicate(front_half(), back_half(), EDGE_MEANS)

        assert result["split_duplicate_score"] > HIGH
        assert "split" in result["reason"].lower()

    def test_split_with_one_shared_node_is_high(self):
        """Halves that meet at a shared node are the clearest split."""
        a = FULL_POSE.copy()
        a[4:] = np.nan  # nodes 0,1,2,3
        b = FULL_POSE.copy()
        b[:3] = np.nan  # nodes 3,4,5 (node 3 shared, coincident)

        result = compute_split_duplicate(a, b, EDGE_MEANS)

        assert result["split_duplicate_score"] > HIGH

    def test_score_is_in_unit_interval(self):
        """Score is always a float in [0, 1]."""
        for a, b in [
            (front_half(), back_half()),
            (FULL_POSE, FULL_POSE.copy()),
            (front_half(), front_half()),
        ]:
            score = compute_split_duplicate(a, b, EDGE_MEANS)["split_duplicate_score"]
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_uses_bbox_diagonal_fallback_without_edge_means(self):
        """Without edge_means, the bbox-diagonal scale still yields high."""
        result = compute_split_duplicate(front_half(), back_half(), None)

        assert result["split_duplicate_score"] > HIGH

    def test_empty_edge_means_falls_back(self):
        """An empty edge_means dict behaves like None (fallback scale)."""
        result = compute_split_duplicate(front_half(), back_half(), {})

        assert result["split_duplicate_score"] > HIGH


class TestSplitInvariance:
    """Split detection should be translation/rotation/scale invariant."""

    def test_translation_invariant(self):
        offset = np.array([500.0, -300.0])
        base = compute_split_duplicate(front_half(), back_half(), EDGE_MEANS)
        moved = compute_split_duplicate(
            front_half(FULL_POSE + offset),
            back_half(FULL_POSE + offset),
            EDGE_MEANS,
        )
        assert moved["split_duplicate_score"] == pytest.approx(
            base["split_duplicate_score"], abs=1e-6
        )

    def test_rotation_invariant(self):
        base = compute_split_duplicate(front_half(), back_half(), EDGE_MEANS)
        rotated_pose = rotate(FULL_POSE, 37.0)
        rot = compute_split_duplicate(
            front_half(rotated_pose), back_half(rotated_pose), EDGE_MEANS
        )
        assert rot["split_duplicate_score"] == pytest.approx(
            base["split_duplicate_score"], abs=1e-6
        )

    def test_scale_invariant_with_fallback(self):
        """3x larger pose with the bbox fallback gives the same score."""
        base = compute_split_duplicate(front_half(), back_half(), None)
        big = compute_split_duplicate(
            front_half(FULL_POSE * 3.0), back_half(FULL_POSE * 3.0), None
        )
        assert big["split_duplicate_score"] == pytest.approx(
            base["split_duplicate_score"], abs=1e-6
        )

    def test_scale_invariant_with_scaled_edge_means(self):
        """3x pose with 3x edge stats also matches the 1x score."""
        base = compute_split_duplicate(front_half(), back_half(), EDGE_MEANS)
        big_edges = {k: v * 3.0 for k, v in EDGE_MEANS.items()}
        big = compute_split_duplicate(
            front_half(FULL_POSE * 3.0), back_half(FULL_POSE * 3.0), big_edges
        )
        assert big["split_duplicate_score"] == pytest.approx(
            base["split_duplicate_score"], abs=1e-6
        )


class TestComputeSplitDuplicateNegative:
    """The detector must NOT fire on non-splits."""

    def test_identical_copies_are_not_split(self):
        """Identical copies share all nodes (high co-visibility) -> not split.

        This case is meant to be caught by the IoU / node-overlap signals, not
        the split signal.
        """
        result = compute_split_duplicate(FULL_POSE, FULL_POSE.copy(), EDGE_MEANS)

        assert result["split_duplicate_score"] < LOW

    def test_two_distinct_animals_side_by_side_full_labels(self):
        """Two fully-labeled distinct animals share node sets -> not split."""
        b = FULL_POSE.copy()
        b[:, 0] += 60  # shift right by ~3 edges

        result = compute_split_duplicate(FULL_POSE, b, EDGE_MEANS)

        assert result["split_duplicate_score"] < LOW

    def test_two_distinct_animals_disjoint_nodes_with_gap(self):
        """Disjoint node sets but a clear spatial gap -> distinct, not split.

        This is the hardest negative: low co-visibility (so the disjointness
        precondition passes) but the two point clouds are far apart, which the
        proximity / coherence checks reject.
        """
        a = front_half()  # left animal, front nodes
        b = FULL_POSE.copy()
        b[:, 0] += 80
        b[:3] = np.nan  # right animal, back nodes

        result = compute_split_duplicate(a, b, EDGE_MEANS)

        assert result["split_duplicate_score"] < LOW

    def test_huddling_overlapping_full_animals_low(self):
        """Two close, fully-labeled (huddling) animals -> not a split.

        High co-visibility is the precondition violation here.
        """
        b = FULL_POSE.copy()
        b[:, 0] += 8  # very close, still both fully labeled

        result = compute_split_duplicate(FULL_POSE, b, EDGE_MEANS)

        assert result["split_duplicate_score"] < LOW


class TestComputeSplitDuplicateDegenerate:
    """Degenerate inputs must never raise or leak NaN."""

    def test_one_instance_too_few_visible(self):
        a = FULL_POSE.copy()
        a[1:] = np.nan  # only one visible node
        result = compute_split_duplicate(a, back_half(), EDGE_MEANS)

        assert result["split_duplicate_score"] == 0.0
        assert "too few" in result["reason"].lower()

    def test_each_single_node_too_few_visible(self):
        a = np.full((6, 2), np.nan)
        a[0] = [0.0, 0.0]
        b = np.full((6, 2), np.nan)
        b[5] = [1.0, 1.0]

        result = compute_split_duplicate(a, b, EDGE_MEANS)

        assert result["split_duplicate_score"] == 0.0

    def test_all_invisible(self):
        nan_pose = np.full((6, 2), np.nan)
        result = compute_split_duplicate(nan_pose, nan_pose.copy(), EDGE_MEANS)

        assert result["split_duplicate_score"] == 0.0

    def test_score_never_nan(self):
        """No combination should produce a NaN score."""
        cases = [
            (front_half(), back_half()),
            (FULL_POSE, FULL_POSE.copy()),
            (np.full((6, 2), np.nan), back_half()),
            (front_half(), np.full((6, 2), np.nan)),
        ]
        for a, b in cases:
            score = compute_split_duplicate(a, b, EDGE_MEANS)["split_duplicate_score"]
            assert np.isfinite(score)

    def test_coincident_points_no_internal_spacing(self):
        """Instances whose visible nodes are all coincident don't crash.

        Each instance has 2+ visible nodes but zero internal spacing; the
        coherence factor must degrade gracefully (governed by proximity).
        """
        a = np.full((6, 2), np.nan)
        a[0:2] = [10.0, 10.0]  # two coincident visible nodes
        b = np.full((6, 2), np.nan)
        b[3:5] = [10.0, 10.0]  # disjoint node indices, same location

        result = compute_split_duplicate(a, b, EDGE_MEANS)

        assert 0.0 <= result["split_duplicate_score"] <= 1.0
        assert np.isfinite(result["split_duplicate_score"])

    def test_accepts_list_inputs(self):
        """Plain nested lists are coerced to arrays without error."""
        a = front_half().tolist()
        b = back_half().tolist()
        result = compute_split_duplicate(a, b, EDGE_MEANS)

        assert result["split_duplicate_score"] > HIGH


class TestComputeSplitDuplicateWithSleapIO:
    """Exercise the same arrays the integration layer extracts from instances."""

    def test_split_from_real_instances(self):
        """One long-bodied animal split front/back across two instances.

        Uses a 6-node linear (head->tail) skeleton, the shape where the split
        failure mode actually occurs: instance A labels the front half,
        instance B the disjoint back half, and together they form one animal.
        """
        from sleap_io import Skeleton
        from sleap_io.model.instance import Instance

        sk = Skeleton(nodes=["n0", "n1", "n2", "n3", "n4", "n5"], name="worm")
        sk.add_edges(
            [
                ("n0", "n1"),
                ("n1", "n2"),
                ("n2", "n3"),
                ("n3", "n4"),
                ("n4", "n5"),
            ]
        )

        inst_a = Instance.from_numpy(front_half(), skeleton=sk)
        inst_b = Instance.from_numpy(back_half(), skeleton=sk)

        result = compute_split_duplicate(
            inst_a.numpy(invisible_as_nan=True),
            inst_b.numpy(invisible_as_nan=True),
            EDGE_MEANS,
        )
        assert result["split_duplicate_score"] > HIGH

    def test_distinct_fly_instances_low(self, skeleton):
        """Two separate, fully-labeled flies -> not a split."""
        from sleap_io.model.instance import Instance

        fly_a = np.array(
            [[100, 100], [100, 115], [100, 130], [85, 110], [115, 110]], dtype=float
        )
        fly_b = fly_a.copy()
        fly_b[:, 0] += 60  # well separated

        inst_a = Instance.from_numpy(fly_a, skeleton=skeleton)
        inst_b = Instance.from_numpy(fly_b, skeleton=skeleton)

        edge_means = {(0, 1): 15.0, (1, 2): 15.0, (1, 3): 15.0, (1, 4): 15.0}

        result = compute_split_duplicate(
            inst_a.numpy(invisible_as_nan=True),
            inst_b.numpy(invisible_as_nan=True),
            edge_means,
        )
        assert result["split_duplicate_score"] < LOW


class TestDuplicateScore:
    """Tests for the duplicate_score signal combiner."""

    def test_identical_via_iou(self):
        """A strong IoU alone flags a duplicate."""
        assert duplicate_score(1.0, 0.0, 0.0) == 1.0

    def test_partial_via_node_overlap(self):
        """A strong node overlap alone flags a duplicate."""
        assert duplicate_score(0.0, 0.95, 0.0) == pytest.approx(0.95)

    def test_split_via_split_signal(self):
        """A strong split signal alone flags a duplicate."""
        assert duplicate_score(0.0, 0.0, 0.85) == pytest.approx(0.85)

    def test_distinct_instances_low(self):
        """Weak signals across the board -> low combined score."""
        assert duplicate_score(0.05, 0.1, 0.0) == pytest.approx(0.1)
        assert duplicate_score(0.0, 0.0, 0.0) == 0.0

    def test_saturating_max(self):
        """Combined score is the max of the (clamped) signals."""
        assert duplicate_score(0.3, 0.6, 0.4) == pytest.approx(0.6)

    def test_output_in_unit_interval(self):
        """Out-of-range inputs are clamped into [0, 1]."""
        assert duplicate_score(2.0, 5.0, 3.0) == 1.0
        assert duplicate_score(-1.0, -2.0, -0.5) == 0.0

    def test_nan_and_inf_treated_as_zero(self):
        """Non-finite signals never inflate or corrupt the result.

        Both NaN and +/-inf are treated as 0 so a garbage/missing signal cannot
        spuriously flag a duplicate (an inf must not saturate the score).
        """
        assert duplicate_score(np.nan, np.nan, np.nan) == 0.0
        # inf is dropped to 0, so the real 0.7 split signal wins.
        assert duplicate_score(np.inf, 0.3, 0.7) == pytest.approx(0.7)
        # All non-finite -> 0, never 1.
        assert duplicate_score(np.nan, 0.0, np.inf) == 0.0

    def test_returns_float(self):
        result = duplicate_score(0.4, 0.2, 0.1)
        assert isinstance(result, float)
