"""Integration tests for the B1 detector wiring in LabelQCDetector.

These cover the five detector modules (chirality, pose-split, chain-ordering,
missing-node, split-duplicate) once they are wired into the detector scaffold:

- the fixed-width feature vector (width invariance under config toggles),
- the positional coupling between ``_extract_features`` and
  ``V3_FEATURE_NAMES`` (contributions must map to the right names),
- end-to-end flagging via the forced hard rules (flip, chain order),
- the GMM-feature path for the chimera detector,
- the missing-node channel surfacing through ``QCResults.get_flagged``, and
- the complementary-split signal in frame-level duplicate detection.

Note on GMM scores: for the small synthetic datasets used here the GMM's
percentile-normalized score is degenerate (most instances saturate at ~1.0),
exactly as in the existing ``test_detector`` suite. These tests therefore assert
on the *forced* hard-rule issues, on the raw feature values, and on the channel
merge -- none of which depend on the GMM cleanly separating instances by score.
"""

from __future__ import annotations

import numpy as np
import pytest
import sleap_io as sio
from sleap_io import LabeledFrame
from sleap_io.model.instance import Instance

from sleap.qc import LabelQCDetector, QCConfig, QCResults
from sleap.qc.detector import V3_FEATURE_NAMES
from sleap.qc.features.baseline import BASELINE_FEATURE_NAMES
from sleap.qc.features.chirality import (
    compute_chirality,
    infer_symmetry_pairs_by_name,
)
from sleap.qc.features.pose_split import compute_pose_split
from sleap.qc.features.ordering import compute_chain_ordering
from sleap.qc.frame_level import detect_duplicates
from sleap.qc.results import CHANNEL_ISSUE_LABELS, FrameKey, InstanceKey


# ---------------------------------------------------------------------------
# Skeleton / pose fixtures
# ---------------------------------------------------------------------------
#
# Quadruped midline-chain skeleton. The longest path (the SkeletonAnalyzer
# "spine") runs nose -> ... -> tailtip, so the chirality axis nodes
# (spine[0], spine[-1]) are the true body midline endpoints, and there are two
# symmetric pairs (ear_L/R, hip_L/R) -- enough for compute_chirality's
# min_pairs=2. Symmetry is left UNdefined on the skeleton so the detector must
# infer it from the _L/_R node names.
QUAD_NAMES = [
    "nose",
    "head",
    "spine1",
    "spine2",
    "tailbase",
    "tailtip",
    "ear_L",
    "ear_R",
    "hip_L",
    "hip_R",
]

QUAD_CANONICAL = np.array(
    [
        [0.0, 20.0],  # nose
        [0.0, 16.0],  # head
        [0.0, 11.0],  # spine1
        [0.0, 6.0],  # spine2
        [0.0, 2.0],  # tailbase
        [0.0, -2.0],  # tailtip
        [-3.0, 16.0],  # ear_L
        [3.0, 16.0],  # ear_R
        [-3.0, 2.0],  # hip_L
        [3.0, 2.0],  # hip_R
    ],
    dtype=float,
)


def _quad_skeleton() -> sio.Skeleton:
    # sio.Skeleton mutates the names list in place (strings -> Node objects), so
    # pass a copy to keep the shared QUAD_NAMES constant pristine for tests that
    # use it directly (e.g. name-based symmetry inference).
    skel = sio.Skeleton(list(QUAD_NAMES))
    skel.add_edges(
        [
            ("nose", "head"),
            ("head", "spine1"),
            ("spine1", "spine2"),
            ("spine2", "tailbase"),
            ("tailbase", "tailtip"),
            ("head", "ear_L"),
            ("head", "ear_R"),
            ("tailbase", "hip_L"),
            ("tailbase", "hip_R"),
        ]
    )
    return skel


def _line_skeleton(n: int = 6) -> sio.Skeleton:
    """A simple ``n``-node line graph 0-1-...-(n-1) (tail-like chain)."""
    names = [f"n{i}" for i in range(n)]  # fresh list each call (safe to mutate)
    skel = sio.Skeleton(names)
    skel.add_edges([(f"n{i}", f"n{i + 1}") for i in range(n - 1)])
    return skel


def _line_base(n: int = 6, spacing: float = 10.0) -> np.ndarray:
    return np.array([[i * spacing, 0.0] for i in range(n)], dtype=float)


def _mirror_x(points: np.ndarray) -> np.ndarray:
    """Whole-instance left/right mirror flip about the x = 0 midline."""
    out = points.copy()
    out[:, 0] *= -1.0
    return out


def _labels_from_poses(skeleton: sio.Skeleton, poses: list[np.ndarray]) -> sio.Labels:
    """Build a single-video Labels with one instance per frame."""
    video = sio.Video.from_filename("test_video.mp4")
    labels = sio.Labels()
    for frame_idx, pts in enumerate(poses):
        inst = Instance.from_numpy(pts, skeleton=skeleton)
        labels.append(LabeledFrame(video=video, frame_idx=frame_idx, instances=[inst]))
    return labels


# ---------------------------------------------------------------------------
# Width invariance
# ---------------------------------------------------------------------------


class TestFeatureWidthInvariance:
    """The fitted feature vector must always be the same fixed width."""

    EXPECTED_WIDTH = 22  # 12 baseline + 10 v3 (incl. the 4 B1 features).

    def test_feature_names_width(self):
        """feature_names == 12 baseline + 10 v3 == 22."""
        assert len(BASELINE_FEATURE_NAMES) == 12
        assert len(V3_FEATURE_NAMES) == 10
        total = len(BASELINE_FEATURE_NAMES) + len(V3_FEATURE_NAMES)
        assert total == self.EXPECTED_WIDTH

    def test_v3_feature_order_is_pinned(self):
        """The B1 names append after hull_compactness in the documented order."""
        assert V3_FEATURE_NAMES == [
            "max_curvature",
            "curvature_std",
            "visibility_pattern_score",
            "nn_distance",
            "hull_area_zscore",
            "hull_compactness",
            "chirality_wrong_fraction",
            "pose_split_score",
            "order_inversion_rate",
            "chain_intersection_count",
        ]

    @pytest.mark.parametrize(
        "config",
        [
            QCConfig(),  # defaults: reliable ON, experimental OFF
            QCConfig(use_chirality=True),
            QCConfig(use_chirality=False),
            QCConfig(use_split_detection=False),
            QCConfig(use_chain_ordering=True),
            QCConfig(use_missing_node_check=True),
            QCConfig(
                use_chirality=True,
                use_split_detection=True,
                use_chain_ordering=True,
                use_missing_node_check=True,
            ),
        ],
    )
    def test_extracted_vector_matches_feature_names(self, config):
        """fit() + a single _extract_features always yield width-22 vectors.

        Holds for the default config and with every experimental flag toggled.
        """
        skel = _quad_skeleton()
        rng = np.random.default_rng(0)
        poses = [
            QUAD_CANONICAL + rng.normal(0, 0.3, size=QUAD_CANONICAL.shape)
            for _ in range(60)
        ]
        detector = LabelQCDetector(config)
        detector.fit(_labels_from_poses(skel, poses))

        assert len(detector.feature_names) == self.EXPECTED_WIDTH
        feats = detector._extract_features(QUAD_CANONICAL)
        assert feats.shape == (self.EXPECTED_WIDTH,)

    @pytest.mark.parametrize(
        "config",
        [
            QCConfig(),
            QCConfig(use_chirality=True),
            QCConfig(use_chain_ordering=True),
            QCConfig(use_missing_node_check=True),
        ],
    )
    def test_fit_and_score_round_trip(self, config):
        """fit() then score() works under default and experimental configs."""
        skel = _quad_skeleton()
        rng = np.random.default_rng(1)
        poses = [
            QUAD_CANONICAL + rng.normal(0, 0.3, size=QUAD_CANONICAL.shape)
            for _ in range(60)
        ]
        labels = _labels_from_poses(skel, poses)
        detector = LabelQCDetector(config)
        detector.fit(labels)
        results = detector.score(labels)

        assert len(results.instance_scores) == 60
        # Every contribution dict is exactly the fixed feature width and holds
        # only finite floats (the forced marker is popped before storage).
        for contributions in results.feature_contributions.values():
            assert len(contributions) == self.EXPECTED_WIDTH
            assert "_forced_top_issue" not in contributions
            assert all(np.isfinite(v) for v in contributions.values())


# ---------------------------------------------------------------------------
# Positional contribution mapping (order coupling guard)
# ---------------------------------------------------------------------------


class TestPositionalContributionMapping:
    """Each new contribution must equal the independently-computed module value."""

    def test_contributions_match_module_outputs(self):
        """For a known instance, the B1 contributions equal the module values.

        This guards the order coupling between the append order in
        ``_extract_features`` and the name order in ``V3_FEATURE_NAMES``: if the
        two ever drift, the contribution under a name would no longer equal the
        value that name's module produces.
        """
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(2)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        labels = _labels_from_poses(skel, poses)

        # Turn on chain ordering so order_inversion_rate / chain_intersection
        # are exercised on the genuine module path (not the 0.0 default).
        config = QCConfig(use_chain_ordering=True)
        detector = LabelQCDetector(config)
        detector.fit(labels)

        # A swapped-tail instance has a non-trivial value for several channels.
        probe = base.copy()
        probe[[2, 3]] = probe[[3, 2]]

        features = detector._extract_features(probe)
        score, contributions = detector._score_instance(features)

        # (d) chimera / pose-split: log1p(raw split_score).
        raw_ps = compute_pose_split(
            probe,
            detector._adjacency,
            detector.baseline_extractor.stats.edge_means,
            detector.baseline_extractor.stats.edge_stds,
        )["split_score"]
        assert contributions["pose_split_score"] == pytest.approx(
            float(np.log1p(max(raw_ps, 0.0)))
        )

        # (b) chain ordering: order_inversion_rate + chain_intersection_count.
        ord_result = compute_chain_ordering(
            probe,
            detector._ordering_chains,
            max_turn_angle=np.deg2rad(config.chain_turn_angle_deg),
        )
        assert contributions["order_inversion_rate"] == pytest.approx(
            ord_result["order_inversion_rate"]
        )
        assert contributions["chain_intersection_count"] == pytest.approx(
            float(ord_result["chain_intersection_count"])
        )

        # The line skeleton has no symmetry, so chirality is the fixed default.
        assert contributions["chirality_wrong_fraction"] == 0.0

    def test_chirality_contribution_matches_module(self):
        """On a symmetric skeleton the chirality contribution matches the module."""
        skel = _quad_skeleton()
        rng = np.random.default_rng(3)
        poses = [
            QUAD_CANONICAL + rng.normal(0, 0.3, size=QUAD_CANONICAL.shape)
            for _ in range(60)
        ]
        detector = LabelQCDetector(QCConfig())  # auto chirality -> ON (has symmetry)
        detector.fit(_labels_from_poses(skel, poses))
        assert detector._chirality_model is not None

        flipped = _mirror_x(QUAD_CANONICAL)
        features = detector._extract_features(flipped)
        _, contributions = detector._score_instance(features)

        expected = compute_chirality(
            flipped,
            detector._symmetry_pairs,
            detector._axis_nodes,
            detector._chirality_model,
        )["chirality_wrong_fraction"]
        assert contributions["chirality_wrong_fraction"] == pytest.approx(expected)
        # A clean whole-instance mirror flip should disagree on every pair.
        assert expected == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Flip end-to-end (chirality, default config, name-inferred symmetry)
# ---------------------------------------------------------------------------


class TestFlipEndToEnd:
    """A whole-instance mirror flip is flagged 'Whole-instance L/R flip'."""

    def _flip_dataset(self):
        skel = _quad_skeleton()
        rng = np.random.default_rng(4)
        poses = [
            QUAD_CANONICAL + rng.normal(0, 0.3, size=QUAD_CANONICAL.shape)
            for _ in range(60)
        ]
        flip_frame = len(poses)
        poses.append(_mirror_x(QUAD_CANONICAL))
        return _labels_from_poses(skel, poses), flip_frame

    def test_symmetry_inferred_by_name(self):
        """The detector infers ear/hip L-R pairs purely from node names."""
        assert infer_symmetry_pairs_by_name(QUAD_NAMES) == [(6, 7), (8, 9)]

    def test_flip_flagged_with_lr_issue(self):
        """The flipped instance is flagged with the L/R-flip top issue."""
        labels, flip_frame = self._flip_dataset()
        detector = LabelQCDetector(QCConfig())  # default config
        detector.fit(labels)

        # Symmetry inferred from names; axis is the spine midline endpoints.
        assert detector._chirality_model is not None
        assert detector._symmetry_pairs == [(6, 7), (8, 9)]

        results = detector.score(labels)
        flip_key = InstanceKey(0, flip_frame, 0)

        assert results.forced_issues.get(flip_key) == "Whole-instance L/R flip"
        assert results.feature_contributions[flip_key][
            "chirality_wrong_fraction"
        ] == pytest.approx(1.0)

        flagged = results.get_flagged(threshold=0.7)
        flip_flags = [f for f in flagged if f.instance_key == flip_key]
        assert len(flip_flags) == 1
        assert flip_flags[0].top_issue == "Whole-instance L/R flip"
        assert flip_flags[0].score >= 0.9

        # Normal instances are never given the L/R-flip label.
        for f in flagged:
            if f.instance_key != flip_key:
                assert f.top_issue != "Whole-instance L/R flip"

    def test_flip_not_flagged_when_chirality_disabled(self):
        """With use_chirality=False there is no chirality model or forced flip."""
        labels, flip_frame = self._flip_dataset()
        detector = LabelQCDetector(QCConfig(use_chirality=False))
        detector.fit(labels)

        assert detector._chirality_model is None
        results = detector.score(labels)
        flip_key = InstanceKey(0, flip_frame, 0)
        assert flip_key not in results.forced_issues
        # The chirality feature falls back to its fixed 0.0 default.
        contrib = results.feature_contributions[flip_key]
        assert contrib["chirality_wrong_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Chimera (pose-split) feature
# ---------------------------------------------------------------------------


class TestChimeraFeature:
    """A split pose yields a high pose_split_score feature."""

    def test_split_pose_high_feature(self):
        """The chimera's pose_split_score far exceeds the normal-pose mean."""
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(5)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        # Chimera: nodes 0-2 near origin, nodes 3-5 displaced far -> one long
        # bridging edge (n2-n3) joining two balanced, well-separated clusters.
        chimera = base.copy()
        chimera[3:, 0] += 200.0
        labels = _labels_from_poses(skel, poses)

        detector = LabelQCDetector(QCConfig())  # split detection ON by default
        detector.fit(labels)

        chimera_feat = detector._extract_features(chimera)
        _, chim_contrib = detector._score_instance(chimera_feat)

        normal_vals = [
            detector._score_instance(detector._extract_features(p))[1][
                "pose_split_score"
            ]
            for p in poses
        ]
        normal_mean = float(np.mean(normal_vals))

        assert chim_contrib["pose_split_score"] > 1.0
        assert chim_contrib["pose_split_score"] > 5 * (normal_mean + 1e-6)

    def test_split_detection_off_zeroes_feature(self):
        """With use_split_detection=False the feature is the fixed 0.0 default."""
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(6)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        chimera = base.copy()
        chimera[3:, 0] += 200.0

        detector = LabelQCDetector(QCConfig(use_split_detection=False))
        detector.fit(_labels_from_poses(skel, poses))
        _, contrib = detector._score_instance(detector._extract_features(chimera))
        assert contrib["pose_split_score"] == 0.0


# ---------------------------------------------------------------------------
# Chain ordering (experimental flag ON)
# ---------------------------------------------------------------------------


class TestChainOrdering:
    """A swapped-tail instance is flagged 'Wrong keypoint order along chain'."""

    def _swap_dataset(self, config):
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(7)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        swap_frame = len(poses)
        swapped = base.copy()
        swapped[[2, 3]] = swapped[[3, 2]]  # adjacent label swap
        poses.append(swapped)
        return skel, _labels_from_poses(skel, poses), swap_frame

    def test_swapped_tail_flagged(self):
        """With chain ordering on, the swap is force-flagged."""
        config = QCConfig(use_chain_ordering=True)
        _, labels, swap_frame = self._swap_dataset(config)
        detector = LabelQCDetector(config)
        detector.fit(labels)
        assert detector._ordering_chains == [[0, 1, 2, 3, 4, 5]]

        results = detector.score(labels)
        swap_key = InstanceKey(0, swap_frame, 0)
        assert results.forced_issues.get(swap_key) == "Wrong keypoint order along chain"

        flagged = results.get_flagged(threshold=0.7)
        swap_flags = [f for f in flagged if f.instance_key == swap_key]
        assert len(swap_flags) == 1
        assert swap_flags[0].top_issue == "Wrong keypoint order along chain"

    def test_chain_ordering_off_by_default(self):
        """Default config (chain ordering OFF) does not force the order issue."""
        _, labels, swap_frame = self._swap_dataset(QCConfig())
        detector = LabelQCDetector(QCConfig())  # default: use_chain_ordering=False
        detector.fit(labels)
        results = detector.score(labels)
        swap_key = InstanceKey(0, swap_frame, 0)
        assert swap_key not in results.forced_issues
        # Feature falls back to fixed defaults.
        contrib = results.feature_contributions[swap_key]
        assert contrib["order_inversion_rate"] == 0.0
        assert contrib["chain_intersection_count"] == 0.0

    def test_user_defined_ordered_chains_by_name_honored(self):
        """User-defined ordered_chains (by NAME) resolve to node indices."""
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(8)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]

        # A user-declared partial chain by name (subset, reordered names).
        config = QCConfig(
            use_chain_ordering=True,
            ordered_chains=[["n1", "n2", "n3", "n4"]],
        )
        detector = LabelQCDetector(config)
        detector.fit(_labels_from_poses(skel, poses))

        # The user-defined chain (by name) is used verbatim, overriding the
        # auto-detected full-line chain.
        assert detector._ordering_chains == [[1, 2, 3, 4]]


# ---------------------------------------------------------------------------
# Missing-node channel (experimental flag ON)
# ---------------------------------------------------------------------------


class TestMissingNodeChannel:
    """An outlier missing-node instance surfaces with 'Missing labelable node'."""

    def test_detector_populates_missing_node_channel(self):
        """The detector records a missing_node channel score for the outlier.

        All training instances are fully visible, so the co-visibility column
        for any node is ~1.0; an instance that drops node n3 then exceeds the
        probability threshold and is recorded in channel_scores.
        """
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(11)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        outlier = base + rng.normal(0, 0.4, size=base.shape)
        outlier[3] = np.nan  # drop a node the peers always keep
        out_frame = len(poses)
        poses.append(outlier)

        detector = LabelQCDetector(QCConfig(use_missing_node_check=True))
        detector.fit(_labels_from_poses(skel, poses))
        results = detector.score(_labels_from_poses(skel, poses))

        out_key = InstanceKey(0, out_frame, 0)
        assert "missing_node" in results.channel_scores
        chan = results.channel_scores["missing_node"]
        assert out_key in chan
        assert chan[out_key] >= detector.config.missing_node_prob_threshold

        # Fully-visible instances are never recorded on the channel.
        for frame_idx in range(60):
            assert InstanceKey(0, frame_idx, 0) not in chan

    def test_missing_node_check_off_by_default(self):
        """Default config does not populate the missing_node channel."""
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(12)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        outlier = base + rng.normal(0, 0.4, size=base.shape)
        outlier[3] = np.nan
        poses.append(outlier)

        detector = LabelQCDetector(QCConfig())  # missing-node check OFF
        detector.fit(_labels_from_poses(skel, poses))
        results = detector.score(_labels_from_poses(skel, poses))
        assert "missing_node" not in results.channel_scores

    def test_get_flagged_surfaces_missing_node_label(self):
        """A channel-dominant key surfaces via get_flagged with the channel label.

        Mirrors the documented merge: the final score is max(gmm, channel) and,
        when the channel wins (chan > gmm), its CHANNEL_ISSUE_LABELS label is the
        top issue. A channel-only key (no GMM score, so no feature
        contributions) must also be handled safely.
        """
        results = QCResults(feature_names=["max_edge_zscore"])
        gmm_key = InstanceKey(0, 0, 0)  # GMM-only
        chan_only_key = InstanceKey(0, 1, 0)  # channel-only, no GMM score
        both_key = InstanceKey(0, 2, 0)  # both, channel dominant

        results.instance_scores = {gmm_key: 0.9, both_key: 0.4}
        results.feature_contributions = {
            gmm_key: {"max_edge_zscore": 5.0},
            both_key: {"max_edge_zscore": 1.0},
        }
        results.channel_scores = {"missing_node": {chan_only_key: 0.85, both_key: 0.95}}

        flagged = results.get_flagged(threshold=0.7)
        by_key = {f.instance_key: f for f in flagged}

        # Channel-only key: flagged on the channel, with the channel label.
        assert chan_only_key in by_key
        assert by_key[chan_only_key].score == pytest.approx(0.85)
        assert by_key[chan_only_key].top_issue == CHANNEL_ISSUE_LABELS["missing_node"]
        # Absent feature contributions are tolerated (empty dict).
        assert by_key[chan_only_key].feature_contributions == {}

        # Both present, channel wins -> channel label and the higher score.
        assert by_key[both_key].score == pytest.approx(0.95)
        assert by_key[both_key].top_issue == CHANNEL_ISSUE_LABELS["missing_node"]

        # GMM-only key keeps its inferred (feature-based) issue.
        assert by_key[gmm_key].top_issue != CHANNEL_ISSUE_LABELS["missing_node"]


# ---------------------------------------------------------------------------
# Duplicate (complementary split) at the frame level
# ---------------------------------------------------------------------------


class TestDuplicateSplit:
    """A complementary split pair raises the frame-level duplicate score."""

    def test_complementary_split_flagged_by_score(self):
        """Two disjoint contiguous halves flag via the split_duplicate reason."""
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(13)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        detector = LabelQCDetector(QCConfig())  # use_duplicate_score ON by default
        detector.fit(_labels_from_poses(skel, poses))
        edge_means = detector.baseline_extractor.stats.edge_means

        # Instance A labels the front half, B labels the back half; together
        # they form one coherent animal (no shared nodes, abutting at the body).
        inst_a = base.copy()
        inst_a[3:] = np.nan
        inst_b = base.copy()
        inst_b[:3] = np.nan

        # IoU/node-overlap thresholds set high so only the split signal can fire.
        dups = detect_duplicates(
            [inst_a, inst_b],
            iou_threshold=0.99,
            node_overlap_ratio=0.99,
            edge_means=edge_means,
            duplicate_score_threshold=0.5,
        )
        assert len(dups) == 1
        assert dups[0]["reason"] == "split_duplicate"
        assert dups[0]["duplicate_score"] >= 0.5
        assert dups[0]["split_duplicate_score"] >= 0.5

    def test_detector_records_duplicate_scores_in_frameqc(self):
        """_check_frame stores a duplicate_score parallel to pairs/reasons."""
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(14)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        detector = LabelQCDetector(QCConfig())
        detector.fit(_labels_from_poses(skel, poses))

        inst_a = base.copy()
        inst_a[3:] = np.nan
        inst_b = base.copy()
        inst_b[:3] = np.nan
        video = sio.Video.from_filename("dup.mp4")
        dup_labels = sio.Labels()
        dup_labels.append(
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    Instance.from_numpy(inst_a, skeleton=skel),
                    Instance.from_numpy(inst_b, skeleton=skel),
                ],
            )
        )
        results = detector.score(dup_labels)
        frame_qc = results.frame_results[FrameKey(0, 0)]

        assert len(frame_qc.duplicate_pairs) == 1
        # The new parallel list stays aligned with pairs/reasons.
        assert len(frame_qc.duplicate_scores) == len(frame_qc.duplicate_pairs)
        assert len(frame_qc.duplicate_reasons) == len(frame_qc.duplicate_pairs)
        assert frame_qc.duplicate_reasons[0] == "split_duplicate"
        assert frame_qc.duplicate_scores[0] >= 0.5

    def test_duplicate_score_off_keeps_legacy_behavior(self):
        """With use_duplicate_score off, the split pair is NOT flagged.

        The legacy IoU + node-overlap path does not catch a complementary
        split (disjoint nodes, near-zero IoU and node overlap), so no duplicate
        is recorded and duplicate_scores stays empty.
        """
        skel = _line_skeleton(6)
        base = _line_base(6)
        rng = np.random.default_rng(15)
        poses = [base + rng.normal(0, 0.4, size=base.shape) for _ in range(60)]
        detector = LabelQCDetector(QCConfig(use_duplicate_score=False))
        detector.fit(_labels_from_poses(skel, poses))

        inst_a = base.copy()
        inst_a[3:] = np.nan
        inst_b = base.copy()
        inst_b[:3] = np.nan
        video = sio.Video.from_filename("dup.mp4")
        dup_labels = sio.Labels()
        dup_labels.append(
            LabeledFrame(
                video=video,
                frame_idx=0,
                instances=[
                    Instance.from_numpy(inst_a, skeleton=skel),
                    Instance.from_numpy(inst_b, skeleton=skel),
                ],
            )
        )
        results = detector.score(dup_labels)
        frame_qc = results.frame_results[FrameKey(0, 0)]
        assert frame_qc.duplicate_pairs == []
        assert frame_qc.duplicate_scores == []
