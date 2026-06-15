"""Tests for sleap.qc.features.ordering (wrong keypoint order along a chain)."""

import numpy as np
import pytest

from sleap.qc.features.ordering import (
    DEFAULT_MAX_TURN_ANGLE,
    compute_chain_ordering,
    resolve_chains,
)


# A canonical 5-node tail chain (indices used directly as the chain).
TAIL_CHAIN = [0, 1, 2, 3, 4]

# A correctly-ordered tail with a gentle 2D arc. Monotone in x with a small
# bend in y -- a realistic, correctly-labeled (slightly curved) tail.
CORRECT_CURVED_TAIL = np.array(
    [[0, 0], [1, 0.5], [2, 0.8], [3, 0.9], [4, 0.8]], dtype=float
)


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate points about the origin by ``degrees`` (NaN-safe)."""
    theta = np.deg2rad(degrees)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return points @ R.T


class TestComputeChainOrderingStraight:
    """A correctly-ordered chain should not be flagged."""

    def test_straight_chain_zero_inversions(self):
        """A perfectly straight chain -> ~0 turning angle, 0 inversions."""
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
        result = compute_chain_ordering(points, [TAIL_CHAIN])

        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0
        assert result["max_turn_angle"] < 1e-6

    def test_gently_curved_chain_not_flagged(self):
        """A gently curved (correct) tail has small turns, no inversions."""
        result = compute_chain_ordering(CORRECT_CURVED_TAIL, [TAIL_CHAIN])

        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0
        # Largest turn is well under the 60-degree default threshold.
        assert result["max_turn_angle"] < DEFAULT_MAX_TURN_ANGLE


class TestComputeChainOrderingCurled:
    """Robustness: a correctly-ordered but strongly curled chain is NOT flagged."""

    def test_half_circle_curl_not_flagged(self):
        """A half-circle (correct order) has gentle per-step turns."""
        t = np.linspace(0, np.pi, 6)
        points = np.stack([np.cos(t), np.sin(t)], axis=1) * 5.0
        chain = list(range(6))

        result = compute_chain_ordering(points, [chain])

        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0
        assert result["max_turn_angle"] < DEFAULT_MAX_TURN_ANGLE

    def test_strong_curl_three_quarter_circle_not_flagged(self):
        """A 3/4-circle curl with enough nodes is still correctly ordered."""
        t = np.linspace(0, 1.5 * np.pi, 8)
        points = np.stack([np.cos(t), np.sin(t)], axis=1) * 5.0
        chain = list(range(8))

        result = compute_chain_ordering(points, [chain])

        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0
        assert result["max_turn_angle"] < DEFAULT_MAX_TURN_ANGLE


class TestComputeChainOrderingAdjacentSwap:
    """An adjacent swap shows up as large turning angles + inversions."""

    def test_adjacent_swap_high_turn_and_inversions(self):
        """Swapping t1<->t2 (idx 2 & 3 positions) -> sharp reversals."""
        # Correct straight tail is x = 0,1,2,3,4. Swap the positions held by
        # node indices 2 and 3 (an adjacent label swap).
        points = np.array([[0, 0], [1, 0], [3, 0], [2, 0], [4, 0]], dtype=float)

        result = compute_chain_ordering(points, [TAIL_CHAIN])

        # Two consecutive interior nodes turn sharply -> high inversion rate.
        assert result["order_inversion_rate"] > 0.5
        # A full direction reversal approaches pi radians.
        assert result["max_turn_angle"] > np.deg2rad(150.0)
        assert result["worst_chain"] == tuple(TAIL_CHAIN)
        assert result["worst_node"] in (2, 3)

    def test_adjacent_swap_two_consecutive_large_turns(self):
        """An adjacent swap produces (at least) two large turning angles."""
        points = np.array([[0, 0], [1, 0], [3, 0], [2, 0], [4, 0]], dtype=float)

        result = compute_chain_ordering(points, [TAIL_CHAIN])

        # 3 interior nodes; an adjacent swap flags 2 of them -> rate ~= 2/3.
        assert result["order_inversion_rate"] == pytest.approx(2.0 / 3.0)


class TestComputeChainOrderingNonAdjacentReversal:
    """A non-adjacent swap makes segments geometrically cross."""

    def test_t1_t3_reversal_has_intersection(self):
        """Swapping t1<->t3 (idx 1 & 3) makes a self-crossing chain."""
        points = CORRECT_CURVED_TAIL.copy()
        points[[1, 3]] = points[[3, 1]]

        result = compute_chain_ordering(points, [TAIL_CHAIN])

        assert result["chain_intersection_count"] > 0
        # It also produces sharp turns.
        assert result["order_inversion_rate"] > 0.0
        assert result["max_turn_angle"] > DEFAULT_MAX_TURN_ANGLE

    def test_correct_chain_has_no_intersection(self):
        """A correctly ordered curved chain has zero self-crossings."""
        result = compute_chain_ordering(CORRECT_CURVED_TAIL, [TAIL_CHAIN])
        assert result["chain_intersection_count"] == 0


class TestComputeChainOrderingInvariance:
    """Turning angle is invariant to translation, rotation, and scale."""

    def test_translation_invariant(self):
        points = CORRECT_CURVED_TAIL.copy()
        shifted = points + np.array([100.0, -50.0])

        base = compute_chain_ordering(points, [TAIL_CHAIN])
        moved = compute_chain_ordering(shifted, [TAIL_CHAIN])

        assert moved["max_turn_angle"] == pytest.approx(base["max_turn_angle"])
        assert moved["order_inversion_rate"] == base["order_inversion_rate"]

    def test_rotation_invariant(self):
        points = CORRECT_CURVED_TAIL.copy()
        rotated = _rotate(points, 73.0)

        base = compute_chain_ordering(points, [TAIL_CHAIN])
        rot = compute_chain_ordering(rotated, [TAIL_CHAIN])

        assert rot["max_turn_angle"] == pytest.approx(base["max_turn_angle"])
        assert rot["order_inversion_rate"] == base["order_inversion_rate"]

    def test_scale_invariant(self):
        points = CORRECT_CURVED_TAIL.copy()
        scaled = points * 17.0

        base = compute_chain_ordering(points, [TAIL_CHAIN])
        sc = compute_chain_ordering(scaled, [TAIL_CHAIN])

        assert sc["max_turn_angle"] == pytest.approx(base["max_turn_angle"])
        assert sc["order_inversion_rate"] == base["order_inversion_rate"]

    def test_swap_detected_under_rotation(self):
        """An adjacent swap is still flagged after an arbitrary rotation."""
        swapped = np.array([[0, 0], [1, 0], [3, 0], [2, 0], [4, 0]], dtype=float)
        rotated = _rotate(swapped, 41.0)

        result = compute_chain_ordering(rotated, [TAIL_CHAIN])
        assert result["order_inversion_rate"] > 0.5
        assert result["max_turn_angle"] > np.deg2rad(150.0)


class TestComputeChainOrderingThreshold:
    """The per-chain turning-angle threshold is configurable."""

    def test_lower_threshold_flags_gentle_curve(self):
        """A very low threshold flags even a gentle (correct) curve."""
        # CORRECT_CURVED_TAIL has a max turn of ~0.2 rad; threshold below it.
        result = compute_chain_ordering(
            CORRECT_CURVED_TAIL, [TAIL_CHAIN], max_turn_angle=0.1
        )
        assert result["order_inversion_rate"] > 0.0

    def test_higher_threshold_tolerates_more(self):
        """A high threshold tolerates a turn that the default would flag."""
        # A ~90-degree turn at the middle node.
        points = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], dtype=float)

        default_res = compute_chain_ordering(points, [TAIL_CHAIN])
        loose_res = compute_chain_ordering(
            points, [TAIL_CHAIN], max_turn_angle=np.deg2rad(170.0)
        )

        assert default_res["order_inversion_rate"] > 0.0
        assert loose_res["order_inversion_rate"] == 0.0


class TestComputeChainOrderingDegenerate:
    """Degenerate inputs must never leak NaN or crash."""

    def test_no_chains_returns_default(self):
        points = CORRECT_CURVED_TAIL.copy()
        result = compute_chain_ordering(points, [])

        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0
        assert result["max_turn_angle"] == 0.0
        assert result["worst_chain"] == ()
        assert result["worst_node"] == -1

    def test_chain_too_short(self):
        """A chain with fewer than 3 nodes has no interior turning angle."""
        points = np.array([[0, 0], [1, 0]], dtype=float)
        result = compute_chain_ordering(points, [[0, 1]])

        assert result["order_inversion_rate"] == 0.0
        assert result["max_turn_angle"] == 0.0
        assert result["worst_node"] == -1

    def test_all_invisible_chain(self):
        """A fully occluded chain yields the default (no evaluable turns)."""
        points = np.full((5, 2), np.nan)
        result = compute_chain_ordering(points, [TAIL_CHAIN])

        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0
        assert result["max_turn_angle"] == 0.0
        assert result["worst_node"] == -1
        assert not np.isnan(result["max_turn_angle"])

    def test_occluded_interior_node_skipped(self):
        """An invisible interior node is skipped, not treated as a reversal."""
        # Straight chain but node idx 2 (interior) is invisible.
        points = np.array(
            [[0, 0], [1, 0], [np.nan, np.nan], [3, 0], [4, 0]], dtype=float
        )
        result = compute_chain_ordering(points, [TAIL_CHAIN])

        # The two interior turns touching the NaN node are skipped; the
        # remaining geometry is straight, so nothing is flagged and no NaN
        # leaks through.
        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0
        assert not np.isnan(result["max_turn_angle"])

    def test_partial_visibility_still_evaluates_visible_turns(self):
        """With one endpoint occluded, the visible interior turn is evaluated."""
        # Endpoint idx 0 invisible; interior nodes 1,2,3 form a sharp turn.
        points = np.array(
            [[np.nan, np.nan], [0, 0], [1, 0], [1, 1], [1, 2]], dtype=float
        )
        result = compute_chain_ordering(points, [TAIL_CHAIN])

        # Node 2 turn (idx1->idx2->idx3): straight->up is a 90-degree turn,
        # which exceeds the 60-degree default and is flagged.
        assert result["order_inversion_rate"] > 0.0
        assert not np.isnan(result["max_turn_angle"])

    def test_coincident_nodes_no_crash(self):
        """Coincident (zero-length segment) nodes produce no NaN / crash."""
        points = np.array([[0, 0], [0, 0], [0, 0], [3, 0], [4, 0]], dtype=float)
        result = compute_chain_ordering(points, [TAIL_CHAIN])

        assert not np.isnan(result["max_turn_angle"])
        assert result["order_inversion_rate"] >= 0.0


class TestComputeChainOrderingMultipleChains:
    """Aggregation is the max over chains; worst chain/node are recorded."""

    def test_aggregates_max_over_chains(self):
        """The worst chain drives the reported metrics."""
        # Two chains: a clean one (nodes 0-2) and a swapped one (nodes 3-7).
        points = np.array(
            [
                [0, 0],  # 0  clean chain
                [1, 0],  # 1
                [2, 0],  # 2
                [0, 5],  # 3  swapped chain start
                [1, 5],  # 4
                [3, 5],  # 5  <- swapped position
                [2, 5],  # 6  <- swapped position
                [4, 5],  # 7
            ],
            dtype=float,
        )
        clean_chain = [0, 1, 2]
        swapped_chain = [3, 4, 5, 6, 7]

        result = compute_chain_ordering(points, [clean_chain, swapped_chain])

        assert result["order_inversion_rate"] > 0.0
        # Worst node belongs to the swapped chain.
        assert result["worst_chain"] == tuple(swapped_chain)
        assert result["worst_node"] in swapped_chain


class TestResolveChains:
    """resolve_chains maps names to indices, with auto-chain fallback."""

    NODE_NAMES = ["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"]

    def test_resolves_user_defined_names(self):
        chains = resolve_chains(
            self.NODE_NAMES,
            [["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"]],
            auto_chains=None,
        )
        assert chains == [[0, 1, 2, 3, 4]]

    def test_user_defined_takes_priority_over_auto(self):
        """A user-declared order is ground truth; auto chains are ignored."""
        chains = resolve_chains(
            self.NODE_NAMES,
            [["Tail_2", "Tail_1", "Tail_0"]],  # user's intended order
            auto_chains=[[0, 1, 2, 3, 4]],
        )
        assert chains == [[3, 2, 1]]

    def test_falls_back_to_auto_chains(self):
        chains = resolve_chains(
            self.NODE_NAMES,
            ordered_chains_by_name=None,
            auto_chains=[[0, 1, 2, 3, 4]],
        )
        assert chains == [[0, 1, 2, 3, 4]]

    def test_empty_when_nothing_provided(self):
        chains = resolve_chains(self.NODE_NAMES, None, None)
        assert chains == []

    def test_unknown_names_skipped(self):
        """Names not in the skeleton are dropped from the resolved chain."""
        chains = resolve_chains(
            self.NODE_NAMES,
            [["TTI", "NotANode", "Tail_1", "Tail_2", "TailTip"]],
            auto_chains=None,
        )
        assert chains == [[0, 2, 3, 4]]

    def test_too_short_user_chain_falls_back_to_auto(self):
        """A user chain that resolves to <3 nodes falls back to auto chains."""
        chains = resolve_chains(
            self.NODE_NAMES,
            [["TTI", "Tail_0"]],  # only 2 nodes
            auto_chains=[[0, 1, 2, 3, 4]],
        )
        assert chains == [[0, 1, 2, 3, 4]]

    def test_multiple_user_chains(self):
        chains = resolve_chains(
            self.NODE_NAMES,
            [["TTI", "Tail_0", "Tail_1"], ["Tail_1", "Tail_2", "TailTip"]],
            auto_chains=None,
        )
        assert chains == [[0, 1, 2], [2, 3, 4]]

    def test_auto_chains_below_min_length_dropped(self):
        """Auto chains shorter than 3 nodes are dropped in fallback."""
        chains = resolve_chains(
            self.NODE_NAMES,
            None,
            auto_chains=[[0, 1], [0, 1, 2, 3, 4]],
        )
        assert chains == [[0, 1, 2, 3, 4]]


class TestEndToEndWithSleapIO:
    """Build a real sleap-io skeleton/instance and run the detector end-to-end."""

    def _tail_skeleton(self):
        import sleap_io as sio

        return sio.Skeleton(
            nodes=["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"],
            edges=[
                ("TTI", "Tail_0"),
                ("Tail_0", "Tail_1"),
                ("Tail_1", "Tail_2"),
                ("Tail_2", "TailTip"),
            ],
        )

    def test_correct_instance_not_flagged(self):
        import sleap_io as sio

        skel = self._tail_skeleton()
        inst = sio.Instance.from_numpy(CORRECT_CURVED_TAIL, skeleton=skel)

        names = [n.name for n in skel.nodes]
        chains = resolve_chains(
            names, [["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"]], None
        )
        result = compute_chain_ordering(inst.numpy(), chains)

        assert result["order_inversion_rate"] == 0.0
        assert result["chain_intersection_count"] == 0

    def test_swapped_instance_flagged(self):
        import sleap_io as sio

        skel = self._tail_skeleton()
        # Swap Tail_1 <-> Tail_2 positions (adjacent swap).
        coords = np.array([[0, 0], [1, 0.5], [3, 0.9], [2, 0.8], [4, 0.8]], dtype=float)
        inst = sio.Instance.from_numpy(coords, skeleton=skel)

        names = [n.name for n in skel.nodes]
        chains = resolve_chains(
            names, [["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"]], None
        )
        result = compute_chain_ordering(inst.numpy(), chains)

        assert result["order_inversion_rate"] > 0.0
        assert result["max_turn_angle"] > DEFAULT_MAX_TURN_ANGLE

    def test_occluded_instance_no_nan(self):
        import sleap_io as sio

        skel = self._tail_skeleton()
        coords = CORRECT_CURVED_TAIL.copy()
        coords[2] = np.nan  # occlude an interior tail node
        inst = sio.Instance.from_numpy(coords, skeleton=skel)

        names = [n.name for n in skel.nodes]
        chains = resolve_chains(
            names, [["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"]], None
        )
        result = compute_chain_ordering(inst.numpy(), chains)

        assert not np.isnan(result["max_turn_angle"])
        assert result["order_inversion_rate"] == 0.0
