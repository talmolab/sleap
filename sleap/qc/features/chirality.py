"""Chirality (left/right mirror flip) features.

A whole-instance left/right mirror flip is invariant to every distance- and
unsigned-angle-based feature (edge lengths, joint angles, pairwise distances,
convex hull, ...), so it is invisible to those detectors. Detecting it requires
a *signed* statistic that encodes which side of the body axis each landmark
falls on, i.e. the chirality of the pose.

This module learns, per symmetric landmark pair, the canonical (majority) side
of the body axis on which the "left" member of the pair sits, then scores a new
instance by the fraction of co-visible symmetric pairs whose observed side
disagrees with that learned canonical side:

- a clean pose scores ``~0``,
- a whole-instance mirror flip scores ``~1`` (every pair disagrees),
- a partial / subset swap scores an intermediate value.

The signed side of a point ``p`` relative to the body axis is the sign of the
2D cross product of the axis vector with ``(p - axis_origin)``. This is
invariant to translation, uniform scaling, and rotation of the whole instance
(it only flips under reflection), which is exactly the property needed to
isolate mirror flips from ordinary pose variation.

**Local (spine-relative) axis.** A single *straight* body axis (e.g. the chord
from nose to tail-base) misjudges the side of a pair whenever the animal curls:
a curled-but-correctly-labeled instance then trips the detector. To stay robust
to body curvature, each symmetric pair is measured against the *local* tangent
of the body midline near that pair. The midline is supplied as an **ordered**
list of non-symmetric node indices (nose -> tail), forming a polyline; for each
pair the pair midpoint is projected onto the nearest midline segment and that
segment's tangent is used as the local axis in the cross product. With a
single-segment (two-node) midline this reduces *exactly* to the straight-axis
sign, so the two-node ``axis_node_indices`` form remains a special case.

The midline polyline is resolved per instance in this order:

1. the visible nodes of ``midline_node_indices`` (the ordered midline), if at
   least two are visible;
2. otherwise the two ``axis_node_indices`` anchors, if both are visible
   (a degenerate single-segment midline);
3. otherwise the first principal component of the visible non-symmetric points
   (PCA fallback), as a single-segment midline through their centroid.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


# Recognized left/right suffix and prefix tokens for name-based inference.
# Each entry maps a "left" token to its "right" counterpart. Matching is
# case-insensitive on the token, but the surrounding stem must match exactly so
# that ``Ear_L``/``Ear_R`` pair up while ``Ear_L``/``Eye_R`` do not.
_LR_SUFFIX_TOKENS: list[tuple[str, str]] = [
    ("left", "right"),
    ("l", "r"),
]

# Separators allowed between a stem and an L/R token, e.g. "Ear_L", "Ear-L",
# "Ear.L", "EarL", "Ear L".
_LR_SEPARATORS = r"[ _\-.]?"


def _normalize_axis(
    axis_vec: np.ndarray, min_norm: float = 1e-6
) -> Optional[np.ndarray]:
    """Return a unit axis vector, or ``None`` if it is degenerate.

    Args:
        axis_vec: A length-2 vector describing the body axis direction.
        min_norm: Minimum norm below which the axis is considered degenerate.

    Returns:
        The unit-normalized axis vector, or ``None`` if its norm is below
        ``min_norm`` or it contains NaN.
    """
    axis_vec = np.asarray(axis_vec, dtype=float)
    if axis_vec.shape != (2,) or np.isnan(axis_vec).any():
        return None
    norm = float(np.linalg.norm(axis_vec))
    if norm < min_norm:
        return None
    return axis_vec / norm


def _pca_axis(
    points: np.ndarray,
    exclude_indices: set[int],
    min_points: int = 2,
    min_norm: float = 1e-6,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Estimate a body axis from the visible non-symmetric points via PCA.

    Uses the first principal component (direction of maximum variance) of the
    visible points that are *not* part of any symmetric pair as the body-axis
    direction, anchored at their centroid.

    Args:
        points: ``(n_nodes, 2)`` array of coordinates (NaN for invisible).
        exclude_indices: Node indices to exclude (members of symmetric pairs).
        min_points: Minimum number of distinct visible points required.
        min_norm: Minimum spread below which the axis is considered degenerate.

    Returns:
        Tuple of ``(axis_origin, unit_axis_vec)``, or ``None`` if a meaningful
        axis cannot be estimated.
    """
    n_nodes = points.shape[0]
    keep = [
        i
        for i in range(n_nodes)
        if i not in exclude_indices and not np.isnan(points[i]).any()
    ]
    if len(keep) < min_points:
        return None

    pts = points[keep]
    origin = pts.mean(axis=0)
    centered = pts - origin

    # Spread guard: all points effectively coincident -> no direction.
    if float(np.max(np.abs(centered))) < min_norm:
        return None

    # First principal component via SVD (robust for 2 points too).
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    axis_vec = _normalize_axis(vh[0], min_norm=min_norm)
    if axis_vec is None:
        return None
    return origin, axis_vec


def _midline_polyline(
    points: np.ndarray,
    midline_node_indices: Optional[list[int]],
    min_nodes: int = 2,
) -> Optional[np.ndarray]:
    """Ordered polyline of the visible midline nodes.

    Args:
        points: ``(n_nodes, 2)`` array of coordinates (NaN for invisible).
        midline_node_indices: Ordered (nose -> tail) non-symmetric node indices
            forming the body midline.
        min_nodes: Minimum number of visible midline nodes required to form a
            usable polyline.

    Returns:
        An ``(m, 2)`` array of the visible midline points in order (``m >=
        min_nodes``), or ``None`` if too few midline nodes are visible.
    """
    if not midline_node_indices:
        return None
    n_nodes = points.shape[0]
    poly = [
        points[i]
        for i in midline_node_indices
        if 0 <= i < n_nodes and not np.isnan(points[i]).any()
    ]
    if len(poly) < min_nodes:
        return None
    return np.asarray(poly, dtype=float)


def _project_point_to_polyline(
    point: np.ndarray, polyline: np.ndarray, min_norm: float = 1e-6
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Project ``point`` onto the nearest segment of ``polyline``.

    Args:
        point: Length-2 query point (the symmetric-pair midpoint).
        polyline: ``(m, 2)`` ordered midline points (``m >= 2``).
        min_norm: Minimum segment length to consider non-degenerate.

    Returns:
        Tuple of ``(foot, unit_tangent)`` for the nearest segment, where
        ``foot`` is the (clamped) foot of the perpendicular and ``unit_tangent``
        is the segment direction oriented nose -> tail, or ``None`` if every
        segment is degenerate.
    """
    best_d2 = np.inf
    best_foot: Optional[np.ndarray] = None
    best_tan: Optional[np.ndarray] = None
    for k in range(len(polyline) - 1):
        a = polyline[k]
        b = polyline[k + 1]
        ab = b - a
        seg_len2 = float(ab @ ab)
        if seg_len2 < min_norm * min_norm:
            continue
        # Clamp the projection parameter to the segment so the foot stays on it.
        t = float((point - a) @ ab) / seg_len2
        t = max(0.0, min(1.0, t))
        foot = a + t * ab
        d2 = float((point - foot) @ (point - foot))
        if d2 < best_d2:
            best_d2 = d2
            best_foot = foot
            best_tan = ab / np.sqrt(seg_len2)
    if best_tan is None:
        return None
    return best_foot, best_tan


def _signed_side_local(
    left_point: np.ndarray,
    right_point: np.ndarray,
    polyline: np.ndarray,
) -> Optional[float]:
    """Signed side of the left member relative to the local midline tangent.

    Projects the pair *midpoint* onto the nearest midline segment and uses that
    segment's tangent as the local body axis, returning
    ``sign(cross(tangent, left_point - foot))``. With a single-segment polyline
    this is identical to the straight-axis ``sign(cross(axis, left - origin))``.

    Args:
        left_point: Length-2 coordinate of the pair's left member.
        right_point: Length-2 coordinate of the pair's right member.
        polyline: ``(m, 2)`` ordered midline points (``m >= 2``).

    Returns:
        ``+1.0`` (left of the local tangent), ``-1.0`` (right), ``0.0`` (on the
        tangent), or ``None`` if either member is invisible or no usable segment
        exists.
    """
    left_point = np.asarray(left_point, dtype=float)
    right_point = np.asarray(right_point, dtype=float)
    if np.isnan(left_point).any() or np.isnan(right_point).any():
        return None
    midpoint = 0.5 * (left_point + right_point)
    projected = _project_point_to_polyline(midpoint, polyline)
    if projected is None:
        return None
    foot, tangent = projected
    rel = left_point - foot
    cross = float(tangent[0] * rel[1] - tangent[1] * rel[0])
    return float(np.sign(cross))


def _resolve_midline(
    points: np.ndarray,
    midline_node_indices: Optional[list[int]],
    axis_node_indices: Optional[tuple[int, int]],
    exclude_indices: set[int],
) -> Optional[np.ndarray]:
    """Resolve the midline polyline for a single instance.

    Resolution order: the ordered ``midline_node_indices`` (if >= 2 visible),
    then the two ``axis_node_indices`` anchors (a single-segment midline), then
    a PCA single-segment midline over the visible non-symmetric points.

    Args:
        points: ``(n_nodes, 2)`` array of coordinates (NaN for invisible).
        midline_node_indices: Ordered (nose -> tail) non-symmetric midline node
            indices.
        axis_node_indices: Optional ``(i, j)`` anchor node indices, used as a
            two-node midline fallback.
        exclude_indices: Symmetric-pair node indices to exclude from PCA.

    Returns:
        An ``(m, 2)`` ordered midline polyline (``m >= 2``), or ``None`` if no
        usable midline could be derived.
    """
    # 1. Full ordered midline polyline.
    poly = _midline_polyline(points, midline_node_indices)
    if poly is not None:
        return poly

    n_nodes = points.shape[0]

    # 2. Two explicit anchor nodes as a single-segment midline.
    if axis_node_indices is not None:
        i, j = axis_node_indices
        if (
            0 <= i < n_nodes
            and 0 <= j < n_nodes
            and i != j
            and not np.isnan(points[i]).any()
            and not np.isnan(points[j]).any()
        ):
            seg = np.stack([points[i], points[j]]).astype(float)
            if float(np.linalg.norm(seg[1] - seg[0])) >= 1e-6:
                return seg

    # 3. PCA single-segment midline through the non-symmetric centroid.
    pca = _pca_axis(points, exclude_indices=exclude_indices)
    if pca is None:
        return None
    origin, axis_vec = pca
    return np.stack([origin, origin + axis_vec]).astype(float)


def fit_chirality(
    instances: list[np.ndarray],
    symmetry_pairs: list[tuple[int, int]],
    midline_node_indices: Optional[list[int]] = None,
    axis_node_indices: Optional[tuple[int, int]] = None,
) -> dict:
    """Learn the canonical signed side per symmetric pair from training poses.

    For each symmetric pair ``(left, right)`` and each training instance where
    the pair is co-visible and the body midline is resolvable, the signed side
    of the *left* member relative to the **local** midline tangent is computed.
    The canonical side is the majority sign across instances (sign of the mean
    of the per-instance signs).

    Args:
        instances: List of ``(n_nodes, 2)`` pose arrays. NaN marks invisible
            nodes. Should be clean / canonical labels (e.g. user-labeled).
        symmetry_pairs: List of ``(left_idx, right_idx)`` symmetric node pairs.
        midline_node_indices: Ordered (nose -> tail) list of non-symmetric
            midline node indices defining the body midline polyline. When at
            least two of them are visible for an instance, the local tangent of
            the nearest midline segment is used as that pair's axis. If ``None``
            or too few are visible, the resolution falls back to
            ``axis_node_indices`` then to a PCA axis.
        axis_node_indices: Optional ``(i, j)`` two-node anchor used as a
            single-segment midline fallback when ``midline_node_indices`` is
            unavailable for an instance. Passing only this (with
            ``midline_node_indices=None``) reproduces the original straight-axis
            behavior exactly.

    Returns:
        A model dict with:

        - ``"canonical_side"``: ``dict[tuple[int, int], float]`` mapping each
          symmetric pair to its learned canonical side (``+1.0`` or ``-1.0``).
          Pairs that were never observed with a resolvable midline are omitted.
        - ``"pair_support"``: ``dict[tuple[int, int], int]`` mapping each pair to
          the number of training instances that contributed to its estimate.
        - ``"symmetry_pairs"``: the (normalized) list of pairs used.
        - ``"midline_node_indices"``: the ordered midline indices supplied.
        - ``"axis_node_indices"``: the anchor indices supplied at fit time.
        - ``"n_instances"``: number of training instances seen.
    """
    pairs = [tuple(p) for p in symmetry_pairs]
    midline = list(midline_node_indices) if midline_node_indices else None
    exclude_indices = {idx for pair in pairs for idx in pair}

    # Accumulate signed sides of the left member per pair across instances.
    side_sums: dict[tuple[int, int], float] = {p: 0.0 for p in pairs}
    side_counts: dict[tuple[int, int], int] = {p: 0 for p in pairs}

    for points in instances:
        points = np.asarray(points, dtype=float)
        polyline = _resolve_midline(points, midline, axis_node_indices, exclude_indices)
        if polyline is None:
            continue

        for left_idx, right_idx in pairs:
            # Require BOTH members visible so the side reflects a genuine,
            # co-visible pair rather than a lone node.
            if (
                left_idx >= points.shape[0]
                or right_idx >= points.shape[0]
                or np.isnan(points[left_idx]).any()
                or np.isnan(points[right_idx]).any()
            ):
                continue

            side = _signed_side_local(points[left_idx], points[right_idx], polyline)
            if side is None or side == 0.0:
                # On the axis: ambiguous, contributes no chirality information.
                continue

            side_sums[(left_idx, right_idx)] += side
            side_counts[(left_idx, right_idx)] += 1

    canonical_side: dict[tuple[int, int], float] = {}
    pair_support: dict[tuple[int, int], int] = {}
    for pair in pairs:
        count = side_counts[pair]
        if count == 0:
            continue
        mean_side = side_sums[pair] / count
        # Sign of the mean = majority side. Break exact ties toward +1.
        canonical_side[pair] = 1.0 if mean_side >= 0.0 else -1.0
        pair_support[pair] = count

    return {
        "canonical_side": canonical_side,
        "pair_support": pair_support,
        "symmetry_pairs": pairs,
        "midline_node_indices": midline,
        "axis_node_indices": axis_node_indices,
        "n_instances": len(instances),
    }


def compute_chirality(
    points: np.ndarray,
    symmetry_pairs: list[tuple[int, int]],
    midline_node_indices: Optional[list[int]],
    model: dict,
    axis_node_indices: Optional[tuple[int, int]] = None,
    min_pairs: int = 2,
) -> dict[str, float]:
    """Score a single instance for a left/right mirror flip.

    For each co-visible symmetric pair with a learned canonical side, the signed
    side of the *left* member relative to the **local** midline tangent is
    compared to the learned canonical side. The returned
    ``chirality_wrong_fraction`` is the fraction of such pairs whose observed
    side disagrees with the canonical one.

    Args:
        points: ``(n_nodes, 2)`` array of coordinates (NaN for invisible).
        symmetry_pairs: List of ``(left_idx, right_idx)`` symmetric node pairs.
        midline_node_indices: Ordered (nose -> tail) non-symmetric midline node
            indices for the body midline polyline. If ``None`` or too few are
            visible, the resolution falls back to ``axis_node_indices`` then to
            a PCA axis. Must match what was passed to :func:`fit_chirality`.
        model: Model dict returned by :func:`fit_chirality`.
        axis_node_indices: Optional ``(i, j)`` two-node anchor used as a
            single-segment midline fallback (see :func:`fit_chirality`).
        min_pairs: Minimum number of scorable co-visible pairs required for a
            meaningful score. Below this, ``chirality_wrong_fraction`` is 0.0.

    Returns:
        Dictionary with:

        - ``"chirality_wrong_fraction"``: float in ``[0, 1]`` (0 = consistent
          with the canonical chirality, 1 = fully flipped).
        - ``"n_pairs"``: number of co-visible pairs that were actually scored.
    """
    points = np.asarray(points, dtype=float)
    canonical_side: dict[tuple[int, int], float] = model.get("canonical_side", {})

    pairs = [tuple(p) for p in symmetry_pairs]
    midline = list(midline_node_indices) if midline_node_indices else None
    exclude_indices = {idx for pair in pairs for idx in pair}

    polyline = _resolve_midline(points, midline, axis_node_indices, exclude_indices)
    if polyline is None:
        return {"chirality_wrong_fraction": 0.0, "n_pairs": 0}

    n_pairs = 0
    n_wrong = 0
    for left_idx, right_idx in pairs:
        canonical = canonical_side.get((left_idx, right_idx))
        if canonical is None:
            # No learned canonical side for this pair -> cannot judge.
            continue
        if (
            left_idx >= points.shape[0]
            or right_idx >= points.shape[0]
            or np.isnan(points[left_idx]).any()
            or np.isnan(points[right_idx]).any()
        ):
            continue

        side = _signed_side_local(points[left_idx], points[right_idx], polyline)
        if side is None or side == 0.0:
            # On the axis: ambiguous, do not count for or against a flip.
            continue

        n_pairs += 1
        if side != canonical:
            n_wrong += 1

    if n_pairs < min_pairs:
        return {"chirality_wrong_fraction": 0.0, "n_pairs": n_pairs}

    return {
        "chirality_wrong_fraction": float(n_wrong) / float(n_pairs),
        "n_pairs": n_pairs,
    }


def order_midline_by_pca(
    instances: list[np.ndarray],
    midline_node_indices: list[int],
    min_points: int = 2,
) -> list[int]:
    """Order midline node indices nose -> tail by their mean PCA projection.

    The body midline polyline needs its nodes in anatomical order, but a
    skeleton's graph topology does not always provide it (a star-topology
    skeleton, for example, attaches several midline nodes to a single hub with
    no path between them). This orders the supplied midline nodes by the average
    of their projections onto the first principal component of the non-symmetric
    points, which recovers the nose -> tail ordering robustly across topologies.

    The PCA axis has an arbitrary sign; the returned order is therefore unique
    only up to reversal. Reversing the midline does not change the local-tangent
    *line* (only its orientation), and the signed-side cross product flips sign
    consistently for every pair, so the learned canonical sides absorb the
    choice. The orientation is fixed deterministically (first node gets the
    smaller mean projection) purely for reproducibility.

    Args:
        instances: List of ``(n_nodes, 2)`` pose arrays used to estimate the
            axis (e.g. the training instances).
        midline_node_indices: Unordered midline (non-symmetric) node indices.
        min_points: Minimum non-symmetric points needed in an instance for it to
            contribute to the projection estimate.

    Returns:
        The midline node indices ordered by mean PCA projection. If no instance
        yields a usable axis, the input order is returned unchanged.
    """
    midline = [int(i) for i in midline_node_indices]
    if len(midline) < 2:
        return midline

    exclude_indices: set[int] = set()  # PCA over all non-NaN points of the body
    proj_sums = {i: 0.0 for i in midline}
    proj_counts = {i: 0 for i in midline}

    for points in instances:
        points = np.asarray(points, dtype=float)
        pca = _pca_axis(points, exclude_indices=exclude_indices, min_points=min_points)
        if pca is None:
            continue
        origin, axis_vec = pca
        for i in midline:
            if 0 <= i < points.shape[0] and not np.isnan(points[i]).any():
                proj_sums[i] += float((points[i] - origin) @ axis_vec)
                proj_counts[i] += 1

    # Nodes that were never visible keep a neutral 0.0 projection and sort
    # stably among themselves (Python's sort is stable on ties).
    if all(c == 0 for c in proj_counts.values()):
        return midline

    def _mean_proj(i: int) -> float:
        return proj_sums[i] / proj_counts[i] if proj_counts[i] else 0.0

    return sorted(midline, key=_mean_proj)


def _split_lr_token(name: str) -> Optional[tuple[str, str, bool]]:
    """Split a node name into ``(stem, canonical_side, is_left)`` if it carries
    a recognized left/right token.

    Recognizes both suffix forms (``Ear_L``, ``ear_left``, ``EarLeft``) and
    prefix forms (``L_Ear``, ``left_ear``). The stem is the remaining text with
    the separator stripped; ``canonical_side`` is ``"left"`` or ``"right"``.

    Args:
        name: The node name.

    Returns:
        ``(stem, side, is_left)`` where ``side`` is ``"left"`` or ``"right"`` and
        ``is_left`` is ``True`` for left tokens, or ``None`` if no L/R token is
        found.
    """
    for left_tok, right_tok in _LR_SUFFIX_TOKENS:
        for tok, is_left, side in (
            (left_tok, True, "left"),
            (right_tok, False, "right"),
        ):
            # Suffix form: <stem><sep><tok>, e.g. "Ear_L", "EarLeft".
            suffix_re = re.compile(
                rf"^(?P<stem>.+?){_LR_SEPARATORS}{tok}$", re.IGNORECASE
            )
            m = suffix_re.match(name)
            if m is not None:
                stem = m.group("stem")
                if stem:
                    return f"S:{stem.lower()}", side, is_left

            # Prefix form: <tok><sep><stem>, e.g. "L_Ear", "left_ear".
            prefix_re = re.compile(
                rf"^{tok}{_LR_SEPARATORS}(?P<stem>.+)$", re.IGNORECASE
            )
            m = prefix_re.match(name)
            if m is not None:
                stem = m.group("stem")
                if stem:
                    return f"P:{stem.lower()}", side, is_left

    return None


def infer_symmetry_pairs_by_name(
    node_names: list[str],
) -> list[tuple[int, int]]:
    """Infer left/right symmetric pairs from node-name suffixes/prefixes.

    Used when a skeleton has no symmetries defined (e.g. CVAT-style imports), so
    that mirror-flip detection still works. Pairs nodes whose names share a stem
    but differ by a left/right token, e.g. ``Ear_L``/``Ear_R``,
    ``Shoulder_left``/``Shoulder_right``, ``Haunch_left``/``Haunch_right``,
    ``L_Eye``/``R_Eye``.

    The single-letter ``_L``/``_R`` form is only honored when a matching stem
    exists on the other side, which avoids spuriously treating e.g. a lone
    ``tail`` ending in no token as symmetric.

    Args:
        node_names: Ordered list of node names (index = node index).

    Returns:
        List of ``(left_idx, right_idx)`` index pairs, ordered by left index.
        Each node appears in at most one pair.
    """
    # Map (orientation:stem) -> {"left": idx, "right": idx}.
    groups: dict[str, dict[str, int]] = {}

    for idx, name in enumerate(node_names):
        parsed = _split_lr_token(name)
        if parsed is None:
            continue
        key, side, _is_left = parsed
        bucket = groups.setdefault(key, {})
        # First occurrence wins for a given side (deterministic, stable order).
        bucket.setdefault(side, idx)

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for bucket in groups.values():
        if "left" in bucket and "right" in bucket:
            left_idx = bucket["left"]
            right_idx = bucket["right"]
            if left_idx in used or right_idx in used or left_idx == right_idx:
                continue
            pairs.append((left_idx, right_idx))
            used.add(left_idx)
            used.add(right_idx)

    pairs.sort(key=lambda p: p[0])
    return pairs
