"""Tests for sleap.qc.features.chirality (left/right mirror flip detection)."""

import numpy as np
import pytest
import sleap_io as sio

from sleap.qc.features.chirality import (
    compute_chirality,
    fit_chirality,
    infer_symmetry_pairs_by_name,
)


# ---------------------------------------------------------------------------
# Fixtures: a quadruped-like layout.
#
# Node layout (index : name):
#   0 : spine_front  (axis anchor, "head" end)
#   1 : spine_back   (axis anchor, "tail" end)
#   2 : ear_L        3 : ear_R
#   4 : shoulder_L   5 : shoulder_R
#   6 : hip_L        7 : hip_R
#
# The body axis runs from spine_front (top) to spine_back (bottom) along -y.
# Left nodes have x < 0, right nodes have x > 0 in the canonical pose.
# ---------------------------------------------------------------------------

CANONICAL = np.array(
    [
        [0.0, 12.0],  # 0 spine_front
        [0.0, 0.0],  # 1 spine_back
        [-3.0, 10.0],  # 2 ear_L
        [3.0, 10.0],  # 3 ear_R
        [-4.0, 7.0],  # 4 shoulder_L
        [4.0, 7.0],  # 5 shoulder_R
        [-4.0, 2.0],  # 6 hip_L
        [4.0, 2.0],  # 7 hip_R
    ],
    dtype=float,
)

SYM_PAIRS = [(2, 3), (4, 5), (6, 7)]
AXIS = (0, 1)


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate points about the origin by ``degrees``."""
    t = np.deg2rad(degrees)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return points @ rot.T


def _jitter(points: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Add small Gaussian jitter (deterministic via seed)."""
    rng = np.random.default_rng(seed)
    return points + rng.normal(scale=scale, size=points.shape)


def _mirror_about_axis(points: np.ndarray) -> np.ndarray:
    """Whole-instance mirror flip about the body axis (the line x = 0).

    Keeps node identities/labels fixed (so the left ear point now sits on the
    right side), which is exactly the mislabel a chirality detector must catch.
    """
    flipped = points.copy()
    flipped[:, 0] *= -1
    return flipped


@pytest.fixture
def clean_model():
    """A chirality model fit on jittered copies of the canonical pose."""
    instances = [_jitter(CANONICAL, scale=0.3, seed=i) for i in range(20)]
    return fit_chirality(instances, SYM_PAIRS, AXIS)


# ---------------------------------------------------------------------------
# fit_chirality
# ---------------------------------------------------------------------------


class TestFitChirality:
    def test_learns_canonical_side_for_all_pairs(self, clean_model):
        """All co-visible pairs should get a learned canonical side."""
        assert set(clean_model["canonical_side"].keys()) == set(SYM_PAIRS)
        for side in clean_model["canonical_side"].values():
            assert side in (-1.0, 1.0)

    def test_records_support_counts(self, clean_model):
        """Support count equals the number of contributing instances."""
        assert all(c == 20 for c in clean_model["pair_support"].values())
        assert clean_model["n_instances"] == 20

    def test_canonical_side_is_consistent_left_member(self, clean_model):
        """Left members sit on a single consistent side of the downward axis.

        With the axis pointing from spine_front (top) to spine_back (bottom),
        a left node (x < 0) lies on a fixed side; all left members must share
        that canonical sign.
        """
        sides = set(clean_model["canonical_side"].values())
        assert len(sides) == 1

    def test_no_pairs_yields_empty_model(self):
        """No symmetry pairs -> empty canonical map, but valid model dict."""
        model = fit_chirality([CANONICAL], [], AXIS)
        assert model["canonical_side"] == {}
        assert model["pair_support"] == {}

    def test_pairs_never_covisible_are_omitted(self):
        """A pair that is never co-visible gets no canonical side."""
        pts = CANONICAL.copy()
        pts[6] = np.nan  # hip_L invisible in every instance
        model = fit_chirality([pts] * 5, SYM_PAIRS, AXIS)
        assert (6, 7) not in model["canonical_side"]
        assert (2, 3) in model["canonical_side"]


# ---------------------------------------------------------------------------
# compute_chirality: positive / negative / intermediate cases
# ---------------------------------------------------------------------------


class TestComputeChirality:
    def test_clean_pose_scores_zero(self, clean_model):
        """A clean canonical pose -> wrong_fraction ~ 0."""
        result = compute_chirality(CANONICAL, SYM_PAIRS, AXIS, clean_model)
        assert result["chirality_wrong_fraction"] == pytest.approx(0.0)
        assert result["n_pairs"] == 3

    def test_jittered_clean_pose_scores_zero(self, clean_model):
        """Small noise must not trip the detector."""
        noisy = _jitter(CANONICAL, scale=0.4, seed=999)
        result = compute_chirality(noisy, SYM_PAIRS, AXIS, clean_model)
        assert result["chirality_wrong_fraction"] == pytest.approx(0.0)

    def test_whole_mirror_flip_scores_one(self, clean_model):
        """A whole-instance mirror flip -> wrong_fraction ~ 1.0."""
        flipped = _mirror_about_axis(CANONICAL)
        result = compute_chirality(flipped, SYM_PAIRS, AXIS, clean_model)
        assert result["chirality_wrong_fraction"] == pytest.approx(1.0)
        assert result["n_pairs"] == 3

    def test_subset_swap_is_intermediate(self, clean_model):
        """Swapping one of three pairs -> intermediate (0 < frac < 1)."""
        swapped = CANONICAL.copy()
        swapped[[2, 3]] = swapped[[3, 2]]  # swap ear_L/ear_R coordinates
        result = compute_chirality(swapped, SYM_PAIRS, AXIS, clean_model)
        assert 0.0 < result["chirality_wrong_fraction"] < 1.0
        assert result["chirality_wrong_fraction"] == pytest.approx(1.0 / 3.0)
        assert result["n_pairs"] == 3

    def test_two_of_three_swapped(self, clean_model):
        """Swapping two of three pairs -> 2/3."""
        swapped = CANONICAL.copy()
        swapped[[2, 3]] = swapped[[3, 2]]
        swapped[[4, 5]] = swapped[[5, 4]]
        result = compute_chirality(swapped, SYM_PAIRS, AXIS, clean_model)
        assert result["chirality_wrong_fraction"] == pytest.approx(2.0 / 3.0)


class TestInvariance:
    """Chirality must be invariant to translation, rotation, and scale,
    but must flip under reflection."""

    def test_translation_invariant(self, clean_model):
        moved = CANONICAL + np.array([123.0, -45.0])
        result = compute_chirality(moved, SYM_PAIRS, AXIS, clean_model)
        assert result["chirality_wrong_fraction"] == pytest.approx(0.0)

    def test_rotation_invariant(self, clean_model):
        for deg in (15.0, 90.0, 137.0, -60.0, 180.0):
            rotated = _rotate(CANONICAL, deg)
            result = compute_chirality(rotated, SYM_PAIRS, AXIS, clean_model)
            assert result["chirality_wrong_fraction"] == pytest.approx(0.0), deg

    def test_scale_invariant(self, clean_model):
        scaled = CANONICAL * 3.7
        result = compute_chirality(scaled, SYM_PAIRS, AXIS, clean_model)
        assert result["chirality_wrong_fraction"] == pytest.approx(0.0)

    def test_combined_similarity_transform_clean_and_flip(self, clean_model):
        """Rotation+scale+translation: clean stays 0, mirror stays 1."""

        def transform(p):
            return _rotate(p, 53.0) * 2.1 + np.array([10.0, 200.0])

        clean = transform(CANONICAL)
        flipped = transform(_mirror_about_axis(CANONICAL))
        assert compute_chirality(clean, SYM_PAIRS, AXIS, clean_model)[
            "chirality_wrong_fraction"
        ] == pytest.approx(0.0)
        assert compute_chirality(flipped, SYM_PAIRS, AXIS, clean_model)[
            "chirality_wrong_fraction"
        ] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Degenerate cases
# ---------------------------------------------------------------------------


class TestDegenerate:
    def test_fewer_than_min_pairs_returns_zero(self, clean_model):
        """Only one co-visible pair -> below min_pairs -> 0.0."""
        pts = CANONICAL.copy()
        pts[4:] = np.nan  # only the ear pair (2, 3) remains co-visible
        result = compute_chirality(pts, SYM_PAIRS, AXIS, clean_model)
        assert result["n_pairs"] == 1
        assert result["chirality_wrong_fraction"] == 0.0

    def test_even_a_flip_with_too_few_pairs_returns_zero(self, clean_model):
        """A real flip but with <2 visible pairs is not scored (safety)."""
        flipped = _mirror_about_axis(CANONICAL)
        flipped[4:] = np.nan
        result = compute_chirality(flipped, SYM_PAIRS, AXIS, clean_model)
        assert result["n_pairs"] == 1
        assert result["chirality_wrong_fraction"] == 0.0

    def test_pca_axis_when_no_anchors_given(self):
        """With axis_node_indices=None, the axis is taken from PCA of the
        visible non-symmetric points (here the two spine nodes)."""
        model = fit_chirality(
            [_jitter(CANONICAL, 0.2, i) for i in range(10)], SYM_PAIRS, None
        )
        # PCA axis from the two visible spine nodes; clean -> 0, flip -> 1.
        clean = compute_chirality(CANONICAL, SYM_PAIRS, None, model)
        flipped = compute_chirality(
            _mirror_about_axis(CANONICAL), SYM_PAIRS, None, model
        )
        assert clean["chirality_wrong_fraction"] == pytest.approx(0.0)
        assert flipped["chirality_wrong_fraction"] == pytest.approx(1.0)

    def test_no_usable_axis_returns_zero(self, clean_model):
        """No anchors and no non-symmetric points -> no axis -> 0.0."""
        pts = CANONICAL.copy()
        pts[0] = np.nan  # spine_front
        pts[1] = np.nan  # spine_back (only non-symmetric nodes)
        # axis_node_indices given but NaN -> PCA fallback, which has 0 non-sym
        # visible points -> no axis.
        result = compute_chirality(pts, SYM_PAIRS, AXIS, clean_model)
        assert result["n_pairs"] == 0
        assert result["chirality_wrong_fraction"] == 0.0

    def test_all_nan_instance_returns_zero(self, clean_model):
        pts = np.full_like(CANONICAL, np.nan)
        result = compute_chirality(pts, SYM_PAIRS, AXIS, clean_model)
        assert result == {"chirality_wrong_fraction": 0.0, "n_pairs": 0}

    def test_pair_without_learned_side_is_skipped(self, clean_model):
        """A symmetry pair absent from the model is not scored."""
        extra_pairs = SYM_PAIRS + [(0, 1)]  # (0, 1) has no canonical side
        result = compute_chirality(CANONICAL, extra_pairs, AXIS, clean_model)
        # Only the three learned pairs contribute.
        assert result["n_pairs"] == 3

    def test_no_symmetry_no_pairs(self, clean_model):
        """No symmetry pairs at all -> trivially 0.0 with n_pairs 0."""
        result = compute_chirality(CANONICAL, [], AXIS, clean_model)
        assert result == {"chirality_wrong_fraction": 0.0, "n_pairs": 0}

    def test_never_emits_nan(self, clean_model):
        """Output must never contain NaN even for ragged inputs."""
        for pts in (
            CANONICAL,
            _mirror_about_axis(CANONICAL),
            _jitter(CANONICAL, 5.0, 1),
            np.full_like(CANONICAL, np.nan),
        ):
            res = compute_chirality(pts, SYM_PAIRS, AXIS, clean_model)
            assert np.isfinite(res["chirality_wrong_fraction"])
            assert 0.0 <= res["chirality_wrong_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# infer_symmetry_pairs_by_name
# ---------------------------------------------------------------------------


class TestInferSymmetryPairsByName:
    def test_underscore_l_r_suffix(self):
        names = ["spine_front", "spine_back", "Ear_L", "Ear_R"]
        assert infer_symmetry_pairs_by_name(names) == [(2, 3)]

    def test_left_right_suffix(self):
        names = ["Shoulder_left", "Shoulder_right"]
        assert infer_symmetry_pairs_by_name(names) == [(0, 1)]

    def test_haunch_left_right(self):
        names = ["nose", "Haunch_left", "Haunch_right", "tail"]
        assert infer_symmetry_pairs_by_name(names) == [(1, 2)]

    def test_camelcase_left_right(self):
        names = ["EarLeft", "EarRight", "Snout"]
        assert infer_symmetry_pairs_by_name(names) == [(0, 1)]

    def test_prefix_l_r(self):
        names = ["L_Eye", "R_Eye", "Nose"]
        assert infer_symmetry_pairs_by_name(names) == [(0, 1)]

    def test_prefix_left_right_word(self):
        names = ["left_paw", "right_paw"]
        assert infer_symmetry_pairs_by_name(names) == [(0, 1)]

    def test_dash_and_dot_separators(self):
        assert infer_symmetry_pairs_by_name(["Ear-L", "Ear-R"]) == [(0, 1)]
        assert infer_symmetry_pairs_by_name(["Ear.L", "Ear.R"]) == [(0, 1)]

    def test_multiple_pairs_ordered_by_left_index(self):
        names = [
            "spine_front",
            "spine_back",
            "Ear_L",
            "Ear_R",
            "Shoulder_left",
            "Shoulder_right",
            "Haunch_left",
            "Haunch_right",
        ]
        assert infer_symmetry_pairs_by_name(names) == [(2, 3), (4, 5), (6, 7)]

    def test_lone_left_token_no_partner(self):
        """A single-letter L with no matching R must not form a pair."""
        assert infer_symmetry_pairs_by_name(["paw_L", "head", "tail"]) == []

    def test_midline_names_not_paired(self):
        """Names without L/R tokens are never paired."""
        assert infer_symmetry_pairs_by_name(["spine", "nose", "tail_base"]) == []

    def test_each_node_in_at_most_one_pair(self):
        """Duplicate stems do not cause a node to be reused."""
        names = ["Ear_L", "Ear_R", "Ear_L", "Ear_R"]
        pairs = infer_symmetry_pairs_by_name(names)
        used = [i for pair in pairs for i in pair]
        assert len(used) == len(set(used))

    def test_distinct_stems_do_not_cross_pair(self):
        """Ear_L should not pair with Eye_R."""
        names = ["Ear_L", "Eye_R"]
        assert infer_symmetry_pairs_by_name(names) == []

    def test_empty_node_names(self):
        assert infer_symmetry_pairs_by_name([]) == []


# ---------------------------------------------------------------------------
# Integration with sleap_io: the path the integration layer actually uses.
# ---------------------------------------------------------------------------


class TestWithSleapIO:
    """Build skeletons/instances via sleap_io to mirror real usage."""

    @pytest.fixture
    def skeleton(self):
        skel = sio.Skeleton(
            [
                "spine_front",
                "spine_back",
                "ear_L",
                "ear_R",
                "shoulder_L",
                "shoulder_R",
                "hip_L",
                "hip_R",
            ]
        )
        skel.add_symmetry("ear_L", "ear_R")
        skel.add_symmetry("shoulder_L", "shoulder_R")
        skel.add_symmetry("hip_L", "hip_R")
        return skel

    def test_symmetry_inds_drive_detection(self, skeleton):
        """Use sleap_io's symmetry_inds as the symmetry_pairs argument."""
        sym_pairs = [tuple(p) for p in skeleton.symmetry_inds]
        assert len(sym_pairs) == 3

        instances = [
            sio.Instance.from_numpy(
                _jitter(CANONICAL, 0.25, i), skeleton=skeleton
            ).numpy()
            for i in range(15)
        ]
        model = fit_chirality(instances, sym_pairs, AXIS)

        clean = sio.Instance.from_numpy(CANONICAL, skeleton=skeleton).numpy()
        flipped = sio.Instance.from_numpy(
            _mirror_about_axis(CANONICAL), skeleton=skeleton
        ).numpy()

        assert compute_chirality(clean, sym_pairs, AXIS, model)[
            "chirality_wrong_fraction"
        ] == pytest.approx(0.0)
        assert compute_chirality(flipped, sym_pairs, AXIS, model)[
            "chirality_wrong_fraction"
        ] == pytest.approx(1.0)

    def test_invisible_node_via_instance(self, skeleton):
        """An invisible node (NaN through .numpy()) is handled cleanly."""
        sym_pairs = [tuple(p) for p in skeleton.symmetry_inds]
        instances = [
            sio.Instance.from_numpy(
                _jitter(CANONICAL, 0.25, i), skeleton=skeleton
            ).numpy()
            for i in range(15)
        ]
        model = fit_chirality(instances, sym_pairs, AXIS)

        partial = CANONICAL.copy()
        partial[2] = np.nan  # ear_L invisible -> ear pair not scorable
        inst = sio.Instance.from_numpy(partial, skeleton=skeleton).numpy()
        result = compute_chirality(inst, sym_pairs, AXIS, model)
        assert result["n_pairs"] == 2  # shoulder + hip pairs only
        assert result["chirality_wrong_fraction"] == pytest.approx(0.0)

    def test_skeleton_without_symmetries_uses_name_inference(self):
        """CVAT-style skeleton with no symmetries -> infer pairs from names."""
        skel = sio.Skeleton(
            ["spine_front", "spine_back", "Ear_L", "Ear_R", "Haunch_left",
             "Haunch_right"]
        )
        assert skel.symmetries == []  # none defined

        sym_pairs = infer_symmetry_pairs_by_name([n.name for n in skel.nodes])
        assert sym_pairs == [(2, 3), (4, 5)]

        layout = np.array(
            [
                [0.0, 10.0],
                [0.0, 0.0],
                [-3.0, 8.0],
                [3.0, 8.0],
                [-3.0, 2.0],
                [3.0, 2.0],
            ],
            dtype=float,
        )
        instances = [_jitter(layout, 0.2, i) for i in range(10)]
        model = fit_chirality(instances, sym_pairs, AXIS)

        assert compute_chirality(layout, sym_pairs, AXIS, model)[
            "chirality_wrong_fraction"
        ] == pytest.approx(0.0)
        flipped = layout.copy()
        flipped[:, 0] *= -1
        assert compute_chirality(flipped, sym_pairs, AXIS, model)[
            "chirality_wrong_fraction"
        ] == pytest.approx(1.0)
