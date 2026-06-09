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

The body axis is derived from two midline / non-symmetric anchor nodes (e.g.
spine endpoints) passed as ``axis_node_indices=(i, j)``; if those anchors are
missing for a given instance it falls back to the first principal component of
the visible non-symmetric points.
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


def _resolve_axis(
    points: np.ndarray,
    axis_node_indices: Optional[tuple[int, int]],
    exclude_indices: set[int],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Resolve the body axis for a single instance.

    Prefers the two explicit anchor nodes; falls back to PCA of the visible
    non-symmetric points if the anchors are unavailable or degenerate.

    Args:
        points: ``(n_nodes, 2)`` array of coordinates (NaN for invisible).
        axis_node_indices: Optional ``(i, j)`` anchor node indices defining the
            axis as ``points[j] - points[i]``.
        exclude_indices: Symmetric-pair node indices to exclude from PCA.

    Returns:
        Tuple of ``(axis_origin, unit_axis_vec)``, or ``None`` if no usable axis
        could be derived.
    """
    n_nodes = points.shape[0]

    if axis_node_indices is not None:
        i, j = axis_node_indices
        if (
            0 <= i < n_nodes
            and 0 <= j < n_nodes
            and i != j
            and not np.isnan(points[i]).any()
            and not np.isnan(points[j]).any()
        ):
            origin = points[i].astype(float)
            axis_vec = _normalize_axis(points[j] - points[i])
            if axis_vec is not None:
                return origin, axis_vec

    # Fall back to PCA of visible non-symmetric points.
    return _pca_axis(points, exclude_indices=exclude_indices)


def _signed_side(
    point: np.ndarray, origin: np.ndarray, axis_vec: np.ndarray
) -> Optional[float]:
    """Signed side of ``point`` relative to the oriented body axis.

    Computes ``sign(cross(axis_vec, point - origin))``, i.e. which side of the
    directed axis the point lies on (+1 = left of the axis direction, -1 =
    right, 0 = on the axis).

    Args:
        point: Length-2 coordinate of the node.
        origin: Length-2 axis origin.
        axis_vec: Length-2 (unit) axis direction.

    Returns:
        ``+1.0``, ``-1.0``, ``0.0``, or ``None`` if ``point`` is invisible.
    """
    point = np.asarray(point, dtype=float)
    if np.isnan(point).any():
        return None
    rel = point - origin
    cross = float(axis_vec[0] * rel[1] - axis_vec[1] * rel[0])
    return float(np.sign(cross))


def fit_chirality(
    instances: list[np.ndarray],
    symmetry_pairs: list[tuple[int, int]],
    axis_node_indices: Optional[tuple[int, int]] = None,
) -> dict:
    """Learn the canonical signed side per symmetric pair from training poses.

    For each symmetric pair ``(left, right)`` and each training instance where
    the pair is co-visible and the body axis is resolvable, the signed side of
    the *left* member relative to the axis is computed. The canonical side is
    the majority sign across instances (sign of the mean of the per-instance
    signs).

    Args:
        instances: List of ``(n_nodes, 2)`` pose arrays. NaN marks invisible
            nodes. Should be clean / canonical labels (e.g. user-labeled).
        symmetry_pairs: List of ``(left_idx, right_idx)`` symmetric node pairs.
        axis_node_indices: Optional ``(i, j)`` anchor node indices defining the
            body axis. If ``None`` (or unavailable for an instance), a PCA axis
            over the visible non-symmetric points is used.

    Returns:
        A model dict with:

        - ``"canonical_side"``: ``dict[tuple[int, int], float]`` mapping each
          symmetric pair to its learned canonical side (``+1.0`` or ``-1.0``).
          Pairs that were never observed with a resolvable axis are omitted.
        - ``"pair_support"``: ``dict[tuple[int, int], int]`` mapping each pair to
          the number of training instances that contributed to its estimate.
        - ``"symmetry_pairs"``: the (normalized) list of pairs used.
        - ``"axis_node_indices"``: the anchor indices supplied at fit time.
        - ``"n_instances"``: number of training instances seen.
    """
    pairs = [tuple(p) for p in symmetry_pairs]
    exclude_indices = {idx for pair in pairs for idx in pair}

    # Accumulate signed sides of the left member per pair across instances.
    side_sums: dict[tuple[int, int], float] = {p: 0.0 for p in pairs}
    side_counts: dict[tuple[int, int], int] = {p: 0 for p in pairs}

    for points in instances:
        points = np.asarray(points, dtype=float)
        axis = _resolve_axis(points, axis_node_indices, exclude_indices)
        if axis is None:
            continue
        origin, axis_vec = axis

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

            side = _signed_side(points[left_idx], origin, axis_vec)
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
        "axis_node_indices": axis_node_indices,
        "n_instances": len(instances),
    }


def compute_chirality(
    points: np.ndarray,
    symmetry_pairs: list[tuple[int, int]],
    axis_node_indices: Optional[tuple[int, int]],
    model: dict,
    min_pairs: int = 2,
) -> dict[str, float]:
    """Score a single instance for a left/right mirror flip.

    For each co-visible symmetric pair with a learned canonical side, the signed
    side of the *left* member relative to the body axis is compared to the
    learned canonical side. The returned ``chirality_wrong_fraction`` is the
    fraction of such pairs whose observed side disagrees with the canonical one.

    Args:
        points: ``(n_nodes, 2)`` array of coordinates (NaN for invisible).
        symmetry_pairs: List of ``(left_idx, right_idx)`` symmetric node pairs.
        axis_node_indices: Optional ``(i, j)`` anchor node indices for the body
            axis. If ``None`` (or unavailable), a PCA axis is used.
        model: Model dict returned by :func:`fit_chirality`.
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
    exclude_indices = {idx for pair in pairs for idx in pair}

    axis = _resolve_axis(points, axis_node_indices, exclude_indices)
    if axis is None:
        return {"chirality_wrong_fraction": 0.0, "n_pairs": 0}
    origin, axis_vec = axis

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

        side = _signed_side(points[left_idx], origin, axis_vec)
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
