"""Skeleton graph analysis utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    import sleap_io as sio


class SkeletonAnalyzer:
    """Analyze skeleton topology to determine feature applicability.

    This class extracts structural properties from a skeleton graph that
    determine which QC features are applicable (e.g., curvature requires
    chains of 5+ nodes, symmetry requires defined pairs).

    Attributes:
        n_nodes: Number of nodes in the skeleton.
        n_edges: Number of edges.
        edges: List of edge tuples (src, dst).
        node_names: List of node names.
        symmetry_pairs: List of symmetric node pairs as (left_idx, right_idx).
        spine: Longest path through the skeleton (main chain).
        all_chains: All simple chains of length >= 3.
        endpoints: Node indices with degree 1.
        branch_points: Node indices with degree > 2.
        max_chain_length: Length of the longest chain.
        n_triplets: Number of joint triplets (for angle features).
    """

    def __init__(self, skeleton: "sio.Skeleton"):
        """Initialize from a sleap-io Skeleton.

        Args:
            skeleton: The skeleton to analyze.
        """
        self.n_nodes = len(skeleton.nodes)
        self.node_names = [n.name for n in skeleton.nodes]

        # Build name -> index mapping
        name_to_idx = {n.name: i for i, n in enumerate(skeleton.nodes)}

        # Extract edges as index pairs
        self.edges: list[tuple[int, int]] = []
        for edge in skeleton.edges:
            src_idx = name_to_idx[edge.source.name]
            dst_idx = name_to_idx[edge.destination.name]
            self.edges.append((src_idx, dst_idx))
        self.n_edges = len(self.edges)

        # Extract symmetry pairs
        self.symmetry_pairs: list[tuple[int, int]] = []
        if skeleton.symmetries:
            for sym in skeleton.symmetries:
                # sym.nodes is a set, convert to list for indexing
                sym_nodes = list(sym.nodes)
                if len(sym_nodes) == 2:
                    left_idx = name_to_idx[sym_nodes[0].name]
                    right_idx = name_to_idx[sym_nodes[1].name]
                    self.symmetry_pairs.append((left_idx, right_idx))

        # Build graph and analyze
        self._graph = self._build_graph()
        self._analyze_structure()

    def _build_graph(self) -> nx.Graph:
        """Build a networkx graph from skeleton edges."""
        G = nx.Graph()
        for i in range(self.n_nodes):
            G.add_node(i, name=self.node_names[i])
        for src, dst in self.edges:
            G.add_edge(src, dst)
        return G

    def _analyze_structure(self) -> None:
        """Analyze skeleton structure."""
        G = self._graph

        # Find endpoints and branch points
        self.endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        self.branch_points = [n for n in G.nodes() if G.degree(n) > 2]

        # Find longest path (spine)
        self.spine = self._find_longest_path()
        self.max_chain_length = len(self.spine)

        # Find all chains
        self.all_chains = self._find_all_chains(min_length=3)

        # Count triplets (for angle features)
        self.n_triplets = self._count_triplets()

    def _find_longest_path(self) -> list[int]:
        """Find the longest simple path in the graph."""
        G = self._graph
        if len(G.nodes()) == 0:
            return []

        endpoints = self.endpoints if self.endpoints else list(G.nodes())[:1]
        longest_path: list[int] = []

        for start in endpoints:
            distances = nx.single_source_shortest_path_length(G, start)
            farthest = max(distances, key=distances.get)
            path = nx.shortest_path(G, start, farthest)
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path

    def _find_all_chains(self, min_length: int = 3) -> list[list[int]]:
        """Find all simple chains in the graph."""
        G = self._graph
        terminators = set(self.endpoints) | set(self.branch_points)
        chains: list[list[int]] = []
        visited_edges: set[tuple[int, int]] = set()

        for start in terminators:
            for neighbor in G.neighbors(start):
                edge = tuple(sorted([start, neighbor]))
                if edge in visited_edges:
                    continue

                # Follow the chain
                chain = [start, neighbor]
                visited_edges.add(edge)

                current = neighbor
                prev = start

                while current not in terminators:
                    neighbors = list(G.neighbors(current))
                    next_nodes = [n for n in neighbors if n != prev]
                    if not next_nodes:
                        break

                    next_node = next_nodes[0]
                    edge = tuple(sorted([current, next_node]))
                    visited_edges.add(edge)
                    chain.append(next_node)
                    prev = current
                    current = next_node

                if len(chain) >= min_length:
                    chains.append(chain)

        return chains

    def _count_triplets(self) -> int:
        """Count number of joint triplets (nodes with 2+ neighbors)."""
        count = 0
        G = self._graph
        for node in G.nodes():
            degree = G.degree(node)
            if degree >= 2:
                # Number of angle pairs at this node
                count += degree * (degree - 1) // 2
        return count

    def get_curvature_chains(self, min_length: int = 3) -> list[list[int]]:
        """Get chains suitable for curvature computation.

        Returns:
            List of chains sorted by length (longest first).
        """
        chains = []
        if len(self.spine) >= min_length:
            chains.append(self.spine)

        spine_set = set(self.spine)
        for chain in self.all_chains:
            if set(chain).issubset(spine_set):
                continue
            if len(chain) >= min_length:
                chains.append(chain)

        chains.sort(key=len, reverse=True)
        return chains

    @property
    def has_symmetry(self) -> bool:
        """Whether skeleton has symmetry pairs defined."""
        return len(self.symmetry_pairs) >= 1

    def get_adjacency(self) -> dict[int, list[int]]:
        """Get adjacency list representation."""
        adjacency: dict[int, list[int]] = {i: [] for i in range(self.n_nodes)}
        for src, dst in self.edges:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
        return adjacency
