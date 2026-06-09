"""Keypoint ordering features: turning angles along chains.

Detects nodes that are labeled out of order along an ordered chain
(e.g. a tail ``TTI -> Tail_0 -> Tail_1 -> Tail_2 -> TailTip``).

The core idea (maintainer's design): treat consecutive chain segments as
vectors ``v_i = node_i - node_{i-1}`` and look at the *turning angle* between
consecutive segments at each interior node (the angle between ``v_i`` and
``v_{i+1}``). A correctly ordered chain -- even one that is naturally curled --
has gentle turning angles. A mislabel (e.g. an adjacent swap) produces a sharp
direction reversal, which shows up as one or two large turning angles. An
out-of-order pair that is not adjacent (e.g. ``Tail_1`` swapped with
``Tail_3``) additionally makes two non-adjacent segments geometrically cross,
which is a strong, unambiguous signal.

The turning angle is invariant to translation, rotation, and uniform scale, so
these features do not need any learned per-dataset statistics.
"""

from __future__ import annotations

import numpy as np

# Default per-chain threshold (radians) above which a turning angle at an
# interior node is considered a sharp reversal / likely mislabel. 60 degrees is
# permissive enough that a smoothly curled chain is not flagged while still
# catching adjacent swaps (which approach 180 degrees).
DEFAULT_MAX_TURN_ANGLE = np.deg2rad(60.0)

# Numerical tolerance for zero-length segments.
_EPS = 1e-8


def _turning_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Turning (deflection) angle between two consecutive segment vectors.

    The turning angle is ``pi - interior_angle``: it is 0 when the two segments
    point in the same direction (the chain continues straight) and approaches
    ``pi`` when the chain reverses direction (a sharp hairpin).

    Args:
        v1: First segment vector ``node_i - node_{i-1}``.
        v2: Second segment vector ``node_{i+1} - node_i``.

    Returns:
        Turning angle in radians in ``[0, pi]``, or NaN if either segment has
        (near) zero length.
    """
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < _EPS or norm2 < _EPS:
        return np.nan

    # Angle between the two segment directions. cos = v1.v2 / (|v1||v2|).
    cos_angle = float(np.dot(v1, v2) / (norm1 * norm2))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.arccos(cos_angle))


def _segments_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> bool:
    """Whether segment ``p1->p2`` properly intersects segment ``p3->p4``.

    Uses the orientation (signed-area) test. Collinear/touching configurations
    are treated as non-intersecting to avoid false positives from shared
    endpoints or numerically borderline cases.

    Args:
        p1: First endpoint of segment A (2,).
        p2: Second endpoint of segment A (2,).
        p3: First endpoint of segment B (2,).
        p4: Second endpoint of segment B (2,).

    Returns:
        True if the two open segments cross.
    """

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        # Signed area * 2 of triangle (a, b, c). Sign gives turn direction.
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)

    # Strict straddle on both segments => proper crossing. Using strict
    # inequalities avoids flagging collinear or endpoint-touching segments
    # (which arise naturally for a straight, correctly ordered chain).
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        # Guard against the degenerate case where an orientation is exactly 0
        # (collinear point): that is a touch, not a proper crossing.
        if min(abs(d1), abs(d2), abs(d3), abs(d4)) > _EPS:
            return True
    return False


def resolve_chains(
    skeleton_node_names: list[str],
    ordered_chains_by_name: list[list[str]] | None,
    auto_chains: list[list[int]] | None,
) -> list[list[int]]:
    """Resolve ordered chains to node-index sequences.

    A user-declared ordering is treated as ground truth: when
    ``ordered_chains_by_name`` is provided, those chains (given as lists of node
    *names*) are resolved to node indices and used directly. This makes the
    turning-angle rule deterministic, since the intended order is known.

    When no user-defined chains are given, falls back to ``auto_chains`` (chain
    index sequences inferred from the skeleton graph, e.g. from
    ``SkeletonAnalyzer.get_curvature_chains()``).

    Args:
        skeleton_node_names: Ordered list of node names; index ``i`` is the node
            index of name ``skeleton_node_names[i]``.
        ordered_chains_by_name: Optional list of user-defined chains, each a
            list of node names in the intended order (e.g.
            ``[["TTI", "Tail_0", "Tail_1", "Tail_2", "TailTip"]]``). Names not
            present in the skeleton are skipped; chains with fewer than 3
            resolvable nodes are dropped.
        auto_chains: Optional fallback list of chains as node-index sequences.

    Returns:
        List of chains as node-index sequences. Empty if nothing resolves.
    """
    name_to_idx = {name: i for i, name in enumerate(skeleton_node_names)}

    if ordered_chains_by_name:
        resolved: list[list[int]] = []
        for chain_names in ordered_chains_by_name:
            idx_chain = [
                name_to_idx[name] for name in chain_names if name in name_to_idx
            ]
            # A chain needs at least 3 nodes to have an interior turning angle.
            if len(idx_chain) >= 3:
                resolved.append(idx_chain)
        if resolved:
            return resolved
        # If user-defined chains were given but none resolved (e.g. all names
        # missing or too short), fall through to auto chains below.

    if auto_chains:
        return [list(chain) for chain in auto_chains if len(chain) >= 3]

    return []


def compute_chain_ordering(
    points: np.ndarray,
    chains: list[list[int]],
    max_turn_angle: float = DEFAULT_MAX_TURN_ANGLE,
) -> dict:
    """Detect wrong keypoint order along ordered chains via turning angles.

    For each chain, the turning angle is computed at every interior node from
    the two adjacent segment vectors. The ``order_inversion_rate`` is the
    fraction of interior nodes whose turning angle exceeds ``max_turn_angle``
    (a sharp reversal). The ``chain_intersection_count`` is the number of
    non-adjacent segment pairs that geometrically cross -- a strong signal of a
    non-local swap (e.g. ``Tail_1`` <-> ``Tail_3``). Results are aggregated as
    the maximum over all chains, and the worst chain/node are recorded for use
    in an explanation.

    All metrics are invariant to translation, rotation, and uniform scale, and
    NaN (invisible/missing) nodes are skipped without leaking into results.

    Args:
        points: ``(N_nodes, 2)`` array of node coordinates; NaN marks
            invisible/missing nodes.
        chains: List of chains, each an ordered list of node indices. Use
            :func:`resolve_chains` to obtain these from user-defined names or
            auto-detected chains.
        max_turn_angle: Per-interior-node turning-angle threshold in radians
            above which a node is counted as an inversion. Default ~60 degrees.

    Returns:
        Dictionary with:
        - ``order_inversion_rate``: float in ``[0, 1]``, max over chains of the
          fraction of interior nodes whose turning angle exceeds the threshold.
        - ``chain_intersection_count``: int, max over chains of the number of
          crossing non-adjacent segment pairs.
        - ``max_turn_angle``: float (radians), largest turning angle seen at any
          interior node across all chains.
        - ``worst_chain``: tuple of node indices for the chain with the largest
          turning angle (empty tuple if none evaluable).
        - ``worst_node``: int, the interior node index (into ``points``) with
          the largest turning angle, or ``-1`` if none evaluable.
    """
    default = {
        "order_inversion_rate": 0.0,
        "chain_intersection_count": 0,
        "max_turn_angle": 0.0,
        "worst_chain": (),
        "worst_node": -1,
    }

    if not chains:
        return default

    points = np.asarray(points, dtype=float)

    best_inversion_rate = 0.0
    best_intersection_count = 0
    overall_max_angle = 0.0
    worst_chain: tuple[int, ...] = ()
    worst_node = -1

    for chain in chains:
        if len(chain) < 3:
            continue

        # --- Turning angles at interior nodes ---
        n_interior = 0
        n_inversions = 0
        for i in range(1, len(chain) - 1):
            prev_idx, curr_idx, next_idx = chain[i - 1], chain[i], chain[i + 1]
            p_prev = points[prev_idx]
            p_curr = points[curr_idx]
            p_next = points[next_idx]

            # Skip interior node if any of the three involved nodes is missing.
            if (
                np.isnan(p_prev).any()
                or np.isnan(p_curr).any()
                or np.isnan(p_next).any()
            ):
                continue

            v1 = p_curr - p_prev  # incoming segment direction
            v2 = p_next - p_curr  # outgoing segment direction
            theta = _turning_angle(v1, v2)
            if np.isnan(theta):
                # Degenerate (coincident nodes); not an evaluable turn.
                continue

            n_interior += 1
            if theta > max_turn_angle:
                n_inversions += 1

            if theta > overall_max_angle:
                overall_max_angle = theta
                worst_chain = tuple(chain)
                worst_node = int(curr_idx)

        inversion_rate = (n_inversions / n_interior) if n_interior > 0 else 0.0
        if inversion_rate > best_inversion_rate:
            best_inversion_rate = inversion_rate

        # --- Non-adjacent segment crossings ---
        # Build the list of visible segments (consecutive node pairs in chain).
        segments: list[tuple[int, np.ndarray, np.ndarray]] = []
        for i in range(len(chain) - 1):
            a_idx, b_idx = chain[i], chain[i + 1]
            pa, pb = points[a_idx], points[b_idx]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            segments.append((i, pa, pb))

        intersection_count = 0
        for si in range(len(segments)):
            i_pos, pa, pb = segments[si]
            for sj in range(si + 1, len(segments)):
                j_pos, pc, pd = segments[sj]
                # Only non-adjacent segments: adjacent segments share an
                # endpoint by construction and would always "touch".
                if abs(i_pos - j_pos) <= 1:
                    continue
                if _segments_intersect(pa, pb, pc, pd):
                    intersection_count += 1

        if intersection_count > best_intersection_count:
            best_intersection_count = intersection_count

    return {
        "order_inversion_rate": float(best_inversion_rate),
        "chain_intersection_count": int(best_intersection_count),
        "max_turn_angle": float(overall_max_angle),
        "worst_chain": worst_chain,
        "worst_node": worst_node,
    }
