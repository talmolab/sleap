"""Split/duplicate instance features for frame-level QC.

This module provides pure helpers used by ``sleap.qc.frame_level`` to
strengthen duplicate-instance detection. The existing IoU and co-visible
node-overlap signals catch *near-identical overlapping copies* (one animal
labeled twice on the same nodes), but they miss the complementary
**split** case: a single animal split across two instances that each label a
*largely disjoint* set of nodes (e.g. instance A labels head/front, instance
B labels tail/back). Together the two "halves" form one coherent animal.

The functions here are deliberately scale-normalized (by the median edge
length learned from the dataset, or the bounding-box diagonal of the larger
instance as a fallback) so a single threshold works across recordings, and
they NaN-guard every coordinate so absent/invisible nodes never leak into a
score.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# --- Tunable constants -------------------------------------------------------
# These are interpreted in units of the pose scale (typical edge length), so
# they are resolution-independent.

# Two instances are considered to label "largely disjoint" node sets once the
# disjointness ratio (see ``_disjointness``) climbs from this floor toward
# ``_DISJOINT_FULL``. A disjointness of 0.5 means the two visible-node sets are
# identical (no split); 1.0 means perfectly complementary.
_DISJOINT_MIN = 0.55
_DISJOINT_FULL = 0.85

# Nearest visible node of A to nearest visible node of B, in scale units. Split
# halves of one animal meet at the body, so the gap is tiny; two separate
# animals leave a clear gap. The proximity score decays linearly to 0 at this
# many scale units.
_GAP_TOL = 2.5

# Inter-instance gap measured relative to the instances' OWN internal node
# spacing (coherence / "nested rather than side-by-side"). When one animal is
# split, the two halves abut like a normal skeleton edge, so the join is about
# one internal spacing; two animals side by side are several spacings apart.
# Scores 1.0 at <= _COHERENCE_OK internal spacings, decaying to 0 at
# _COHERENCE_MAX. This is scale-free and does not need edge_means.
_COHERENCE_OK = 1.5
_COHERENCE_MAX = 3.0

# Minimum number of visible nodes required (per instance and in the union) to
# attempt a split judgement at all.
_MIN_VISIBLE = 2


def _visible_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of nodes with finite (visible) coordinates."""
    return ~np.isnan(points).any(axis=1)


def _bbox_diagonal(points: np.ndarray) -> float:
    """Bounding-box diagonal of the visible nodes (0.0 if < 2 visible)."""
    visible = points[_visible_mask(points)]
    if len(visible) < 2:
        return 0.0
    span = visible.max(axis=0) - visible.min(axis=0)
    return float(np.linalg.norm(span))


def _median_edge_length(edge_means: Optional[dict]) -> float:
    """Median of learned edge lengths, or 0.0 if unavailable.

    Args:
        edge_means: Mapping of edge ``(src, dst)`` tuples to mean edge length,
            as produced by :class:`~sleap.qc.features.baseline.DatasetStats`.
            May be ``None`` or empty.

    Returns:
        Median of the positive edge lengths, else ``0.0``.
    """
    if not edge_means:
        return 0.0
    values = np.array([v for v in edge_means.values() if np.isfinite(v) and v > 0])
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _resolve_scale(
    points_a: np.ndarray,
    points_b: np.ndarray,
    edge_means: Optional[dict],
) -> float:
    """Pick a positive length scale to normalize distances.

    Prefers the dataset-wide median edge length (translation/rotation/scale
    invariant and stable across instances); falls back to the bounding-box
    diagonal of the *larger* instance when no edge stats are available.

    Returns:
        A strictly positive scale (defaults to ``1.0`` if nothing is usable).
    """
    scale = _median_edge_length(edge_means)
    if scale > 1e-8:
        return scale

    # Fallback: bbox diagonal of the larger (more spatially extended) instance.
    diag = max(_bbox_diagonal(points_a), _bbox_diagonal(points_b))
    return diag if diag > 1e-8 else 1.0


def _disjointness(visible_a: np.ndarray, visible_b: np.ndarray) -> float:
    """Fraction of the visible-node union that is NOT shared.

    Defined as ``|A ∪ B| / (|A| + |B|)``:

    - ``0.5`` when the two instances label exactly the same nodes (full
      overlap; the identical/duplicate case).
    - ``1.0`` when the two instances label completely disjoint node sets (the
      complementary split case).

    Returns ``0.0`` when neither instance has visible nodes.
    """
    n_a = int(visible_a.sum())
    n_b = int(visible_b.sum())
    if n_a == 0 or n_b == 0:
        return 0.0
    n_union = int((visible_a | visible_b).sum())
    return n_union / (n_a + n_b)


def _nearest_node_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    visible_a: np.ndarray,
    visible_b: np.ndarray,
) -> float:
    """Minimum distance between any visible node of A and any of B.

    This is the inter-instance proximity used to tell a contiguous split
    (halves touching at the body) from two separated animals. Returns
    ``inf`` if either instance has no visible nodes.
    """
    pts_a = points_a[visible_a]
    pts_b = points_b[visible_b]
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")
    # Pairwise distances between the two (small) visible point sets.
    diffs = pts_a[:, None, :] - pts_b[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return float(dists.min())


def _internal_spacing(points: np.ndarray, visible: np.ndarray) -> float:
    """Typical spacing between adjacent visible nodes of one instance.

    Estimated as the median nearest-neighbor distance among the instance's own
    visible nodes. Used as a scale-free reference: a split animal's two halves
    abut at roughly one such spacing, while two side-by-side animals are
    several spacings apart.

    Returns ``nan`` if fewer than two nodes are visible.
    """
    pts = points[visible]
    if len(pts) < 2:
        return float("nan")
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    return float(np.median(dists.min(axis=1)))


def _linear_score(value: float, lo: float, hi: float) -> float:
    """Linearly ramp ``value`` from 0 at ``lo`` to 1 at ``hi`` (clamped).

    Works for both increasing (``lo < hi``) and decreasing (``lo > hi``)
    ramps.
    """
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    frac = (value - lo) / (hi - lo)
    return float(np.clip(frac, 0.0, 1.0))


def compute_split_duplicate(
    points_a: np.ndarray,
    points_b: np.ndarray,
    edge_means: Optional[dict] = None,
) -> dict:
    """Score whether two instances are one animal split across both.

    Detects the *complementary split* failure mode that bbox-IoU and
    co-visible node overlap miss: the two instances are visible on **largely
    disjoint** node sets (few co-visible nodes), yet together their union pose
    forms a single coherent animal -- the two "halves" are spatially
    contiguous (small inter-instance nearest-node distance relative to scale)
    and the union's extent matches a single animal rather than two side by
    side.

    The score is the product of three graded, scale-normalized signals:

    1. **Disjointness** -- the two visible-node sets must be largely
       complementary (this is the precondition that distinguishes a split from
       an identical overlapping copy, which shares nodes).
    2. **Proximity** -- the nearest node of A must be close to the nearest node
       of B (the halves meet at the body). This is the key signal that keeps
       the detector from firing on two genuinely distinct animals labeled on
       disjoint nodes, which leave a clear gap between them.
    3. **Coherence** -- the inter-instance gap must be small relative to the
       instances' own internal node spacing, i.e. the two halves join like a
       normal skeleton edge ("nested") rather than sitting several body-widths
       apart as two animals side by side would.

    Distances are normalized by the median edge length from ``edge_means`` (or
    the larger instance's bbox diagonal as a fallback), so the result is
    translation-, rotation-, and scale-invariant and thresholdable with a
    single cutoff.

    Args:
        points_a: ``(n_nodes, 2)`` coordinates of instance A (NaN = invisible).
        points_b: ``(n_nodes, 2)`` coordinates of instance B (NaN = invisible).
        edge_means: Optional mapping of edge ``(src, dst)`` -> mean edge length
            (from learned dataset stats) used to set the length scale. If
            ``None``/empty, the bbox diagonal of the larger instance is used.

    Returns:
        Dictionary with:

        - ``split_duplicate_score``: float in ``[0, 1]``. High = likely one
          animal split across the two instances.
        - ``reason``: short human-readable explanation of the score.
    """
    points_a = np.asarray(points_a, dtype=float)
    points_b = np.asarray(points_b, dtype=float)

    visible_a = _visible_mask(points_a)
    visible_b = _visible_mask(points_b)
    n_a = int(visible_a.sum())
    n_b = int(visible_b.sum())

    # Degenerate: need enough visible nodes in each instance to judge a split.
    if n_a < _MIN_VISIBLE or n_b < _MIN_VISIBLE:
        return {
            "split_duplicate_score": 0.0,
            "reason": "too few visible nodes",
        }

    union_visible = visible_a | visible_b
    if int(union_visible.sum()) < _MIN_VISIBLE:
        return {
            "split_duplicate_score": 0.0,
            "reason": "too few visible nodes",
        }

    scale = _resolve_scale(points_a, points_b, edge_means)

    # (1) Disjointness precondition: a split labels complementary node sets.
    # High co-visibility (shared nodes) means this is the identical/overlap
    # case, not a split -> disjoint score collapses to 0.
    disjointness = _disjointness(visible_a, visible_b)
    s_disjoint = _linear_score(disjointness, _DISJOINT_MIN, _DISJOINT_FULL)
    if s_disjoint <= 0.0:
        return {
            "split_duplicate_score": 0.0,
            "reason": (
                f"node sets overlap (disjointness={disjointness:.2f}); not a split"
            ),
        }

    # (2) Proximity: the two halves of one animal touch at the body. Two
    # distinct animals labeled on disjoint nodes leave a clear gap.
    gap = _nearest_node_distance(points_a, points_b, visible_a, visible_b)
    gap_norm = gap / scale
    s_gap = _linear_score(gap_norm, _GAP_TOL, 0.0)
    if s_gap <= 0.0:
        return {
            "split_duplicate_score": 0.0,
            "reason": (
                f"instances too far apart (gap={gap_norm:.2f} scale units); "
                "likely distinct animals"
            ),
        }

    # (3) Coherence ("nested rather than side-by-side"): the two halves of one
    # animal join like a normal skeleton edge, so the inter-instance gap is
    # about one of the instances' own internal node spacings. Two animals side
    # by side sit several internal spacings apart. This is scale-free and so
    # complements the absolute-scale proximity check above.
    spacing_a = _internal_spacing(points_a, visible_a)
    spacing_b = _internal_spacing(points_b, visible_b)
    spacings = [s for s in (spacing_a, spacing_b) if np.isfinite(s) and s > 1e-8]
    if spacings:
        rel_gap = gap / float(np.mean(spacings))
        s_coherent = _linear_score(rel_gap, _COHERENCE_MAX, _COHERENCE_OK)
    else:
        # Each instance is a single point (no internal spacing to compare to);
        # the absolute proximity check already governs, so don't penalize.
        rel_gap = float("nan")
        s_coherent = 1.0

    score = float(s_disjoint * s_gap * s_coherent)

    reason = (
        f"complementary split: disjointness={disjointness:.2f}, "
        f"gap={gap_norm:.2f} scale units, relative gap={rel_gap:.2f} spacings"
    )
    if s_coherent <= 0.0:
        reason = (
            f"instances form separate clusters (relative gap={rel_gap:.2f} "
            "spacings); likely two animals side by side"
        )

    return {
        "split_duplicate_score": score,
        "reason": reason,
    }


def duplicate_score(
    iou: float,
    node_overlap_ratio: float,
    split_duplicate_score: float,
) -> float:
    """Combine duplicate signals into one graded confidence in ``[0, 1]``.

    Merges the three complementary duplicate signals so a caller can apply a
    single threshold and get a graded confidence instead of three booleans:

    - ``iou`` -- bbox intersection-over-union (overlapping copies).
    - ``node_overlap_ratio`` -- fraction of co-visible nodes that coincide
      (partial overlapping copies, even at low IoU).
    - ``split_duplicate_score`` -- the complementary split signal from
      :func:`compute_split_duplicate` (one animal split on disjoint nodes).

    A saturating ``max`` is used: any one signal firing strongly is enough to
    flag a duplicate, and the score never exceeds 1. Inputs are individually
    clamped to ``[0, 1]`` and NaN inputs are treated as ``0`` so a missing
    signal cannot inflate or corrupt the result.

    Args:
        iou: Bounding-box IoU between the two instances.
        node_overlap_ratio: Co-visible node overlap ratio.
        split_duplicate_score: Split-duplicate score.

    Returns:
        Combined duplicate confidence in ``[0, 1]``.
    """

    def _clean(x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return 0.0
        return float(np.clip(x, 0.0, 1.0))

    return max(
        _clean(iou),
        _clean(node_overlap_ratio),
        _clean(split_duplicate_score),
    )
