"""Visibility pattern features."""

from __future__ import annotations

from typing import Optional

import numpy as np


class VisibilityModel:
    """Learn and score visibility patterns.

    Learns which nodes tend to be visible together, then flags instances
    where the visibility pattern is unusual (e.g., hip visible but knee invisible
    when they're usually both visible or both invisible).

    Attributes:
        n_nodes: Number of nodes in skeleton.
        co_visibility_matrix: P(node_j visible | node_i visible).
        visibility_rates: Per-node visibility rates.
        n_instances: Number of instances used for fitting.
    """

    def __init__(self):
        """Initialize the visibility model."""
        self.n_nodes: int = 0
        self.co_visibility_matrix: Optional[np.ndarray] = None
        self.visibility_rates: Optional[np.ndarray] = None
        self.n_instances: int = 0

    def fit(self, visibility_masks: np.ndarray) -> "VisibilityModel":
        """Learn co-visibility patterns from data.

        Args:
            visibility_masks: (N_instances, N_nodes) boolean array.
                True = visible, False = invisible.

        Returns:
            Self for chaining.
        """
        visibility_masks = np.asarray(visibility_masks, dtype=bool)
        self.n_instances, self.n_nodes = visibility_masks.shape

        # Per-node visibility rate
        self.visibility_rates = visibility_masks.mean(axis=0)

        # Co-visibility matrix: P(node_j visible | node_i visible)
        self.co_visibility_matrix = np.zeros((self.n_nodes, self.n_nodes))

        for i in range(self.n_nodes):
            mask_i = visibility_masks[:, i]
            n_visible_i = mask_i.sum()

            if n_visible_i > 0:
                for j in range(self.n_nodes):
                    self.co_visibility_matrix[i, j] = (
                        visibility_masks[mask_i, j].sum() / n_visible_i
                    )

        return self

    def score(self, visibility_mask: np.ndarray) -> dict[str, float]:
        """Score how unusual a visibility pattern is.

        Args:
            visibility_mask: (N_nodes,) boolean array.

        Returns:
            Dictionary with:
            - pattern_score: overall unusualness (0 = normal, 1 = very unusual)
            - n_violations: count of strong violations
        """
        if self.co_visibility_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        visibility_mask = np.asarray(visibility_mask, dtype=bool)
        violations = []

        for i in range(self.n_nodes):
            if not visibility_mask[i]:
                continue

            for j in range(self.n_nodes):
                if i == j:
                    continue

                expected_prob = self.co_visibility_matrix[i, j]

                # Check for violations
                if not visibility_mask[j] and expected_prob > 0.9:
                    # Node j invisible when it should be visible
                    violations.append((i, j, expected_prob))
                elif visibility_mask[j] and expected_prob < 0.1:
                    # Node j visible when it's rarely visible with i
                    violations.append((i, j, expected_prob))

        n_violations = len(violations)
        pattern_score = min(1.0, n_violations / max(1, self.n_nodes))

        return {
            "pattern_score": pattern_score,
            "n_violations": n_violations,
        }

    def get_expected_visibility(self, partial_mask: np.ndarray) -> np.ndarray:
        """Given some visible nodes, predict expected visibility of others.

        Args:
            partial_mask: (N_nodes,) boolean array with some nodes marked visible.

        Returns:
            (N_nodes,) array of expected visibility probabilities.
        """
        if self.co_visibility_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        partial_mask = np.asarray(partial_mask, dtype=bool)
        visible_indices = np.where(partial_mask)[0]

        if len(visible_indices) == 0:
            return self.visibility_rates.copy()

        # Average co-visibility from all visible nodes
        expected = np.zeros(self.n_nodes)
        for i in visible_indices:
            expected += self.co_visibility_matrix[i]
        expected /= len(visible_indices)

        return expected


def compute_isolated_invisible(
    visibility_mask: np.ndarray,
    edges: list[tuple[int, int]],
) -> dict[str, float]:
    """Detect invisible nodes with all visible neighbors.

    Args:
        visibility_mask: (N_nodes,) boolean array.
        edges: List of (src, dst) node index pairs.

    Returns:
        Dictionary with:
        - has_isolated_invisible: bool
        - isolated_invisible_nodes: list of node indices
        - n_isolated_invisible: count
    """
    visibility_mask = np.asarray(visibility_mask, dtype=bool)

    # Build adjacency
    n_nodes = len(visibility_mask)
    neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    for src, dst in edges:
        neighbors[src].append(dst)
        neighbors[dst].append(src)

    isolated = []
    for node in range(n_nodes):
        if visibility_mask[node]:
            continue  # Node is visible

        node_neighbors = neighbors[node]
        if len(node_neighbors) == 0:
            continue

        all_neighbors_visible = all(visibility_mask[n] for n in node_neighbors)
        if all_neighbors_visible:
            isolated.append(node)

    return {
        "has_isolated_invisible": len(isolated) > 0,
        "isolated_invisible_nodes": isolated,
        "n_isolated_invisible": len(isolated),
    }
