from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

Assessor = tuple[int, int]  # (low, high) with low < high


def cross_assessors(dim: int) -> list[Assessor]:
    """
    Generalized "cross" assessors for 2^n-dimensional Cayley-Dickson algebras.

    For dim=16 this matches the de Marrais cross-assessor family: low in 1..7 and high in 8..15.
    For dim=32 it becomes low in 1..15 and high in 16..31, etc.
    """
    if dim <= 2 or dim & (dim - 1) != 0:
        raise ValueError("Expected dim to be a power of two >= 4.")
    half = dim // 2
    return [(i, j) for i in range(1, half) for j in range(half, dim)]


@lru_cache(maxsize=None)
def _cd_basis_mul_sign(dim: int, p: int, q: int) -> int:
    """
    Return the sign in the Cayley-Dickson basis product:

        e_p * e_q = sign(p,q) * e_{p xor q}

    This is computed by recursion on the doubling rule used by `cd_multiply_jit`,
    but without allocating dense vectors. The indices p,q are in [0, dim).
    """
    if dim == 1:
        return 1
    half = dim // 2

    if p < half and q < half:
        return _cd_basis_mul_sign(half, p, q)

    if p < half and q >= half:
        # (a,0) * (0,d) = (0, d a)
        return _cd_basis_mul_sign(half, q - half, p)

    if p >= half and q < half:
        # (0,b) * (c,0) = (0, b conj(c))
        s = _cd_basis_mul_sign(half, p - half, q)
        return s if q == 0 else -s

    # (0,b) * (0,d) = (-conj(d) b, 0)
    qh = q - half
    ph = p - half
    if qh == 0:
        # conj(1)=1 so L = -(1*b) = -b
        return -1
    # conj(e_qh) = -e_qh, so L = -((-e_qh)*b) = (e_qh*b)
    return _cd_basis_mul_sign(half, qh, ph)


def diagonal_zero_products(
    dim: int, a: Assessor, b: Assessor, *, atol: float = 1e-12
) -> list[tuple[float, float]]:
    """
    Return all sign choices (s,t) with s,t in {+1,-1} such that:

        (e_i + s e_j) * (e_k + t e_l) = 0

    where a=(i,j), b=(k,l).
    """
    (i, j) = a
    (k, ell) = b
    del atol  # exact integer-coefficient check, no float tolerances needed

    idx_ik = i ^ k
    idx_il = i ^ ell
    idx_jk = j ^ k
    idx_jl = j ^ ell

    s_ik = _cd_basis_mul_sign(dim, i, k)
    s_il = _cd_basis_mul_sign(dim, i, ell)
    s_jk = _cd_basis_mul_sign(dim, j, k)
    s_jl = _cd_basis_mul_sign(dim, j, ell)

    solutions: list[tuple[float, float]] = []
    for s in (1, -1):
        for t in (1, -1):
            coeffs: dict[int, int] = {}
            coeffs[idx_ik] = coeffs.get(idx_ik, 0) + s_ik
            coeffs[idx_il] = coeffs.get(idx_il, 0) + t * s_il
            coeffs[idx_jk] = coeffs.get(idx_jk, 0) + s * s_jk
            coeffs[idx_jl] = coeffs.get(idx_jl, 0) + (s * t) * s_jl

            if all(v == 0 for v in coeffs.values()):
                solutions.append((float(s), float(t)))

    return solutions


def xor_bucket(a: Assessor) -> int:
    i, j = a
    return i ^ j


@dataclass(frozen=True)
class MotifComponent:
    dim: int
    nodes: frozenset[Assessor]
    edges: frozenset[tuple[Assessor, Assessor]]  # undirected, stored as (a,b) with a < b

    def degree_sequence(self) -> list[int]:
        deg = {n: 0 for n in self.nodes}
        for a, b in self.edges:
            deg[a] += 1
            deg[b] += 1
        return sorted(deg.values())

    def is_octahedron_graph(self) -> bool:
        # Octahedron graph == K_{2,2,2}: 6 vertices, 12 edges, regular degree 4.
        return len(self.nodes) == 6 and len(self.edges) == 12 and self.degree_sequence() == [4] * 6

    def k2_multipartite_part_count(self) -> int:
        """
        Detect a complete multipartite graph with all parts of size 2.

        Equivalently, the complement graph is a perfect matching: each vertex has exactly one
        non-neighbor, and non-neighbor relation is symmetric with no fixed points.

        Returns:
            int: number of 2-vertex parts, or 0 if not of this form.
        """
        nodes = sorted(self.nodes)
        if len(nodes) < 4 or len(nodes) % 2 != 0:
            return 0

        edge_set = set(self.edges)

        def adjacent(a: Assessor, b: Assessor) -> bool:
            if a == b:
                return False
            x, y = (a, b) if a < b else (b, a)
            return (x, y) in edge_set

        opposite: dict[Assessor, Assessor] = {}
        for a in nodes:
            non_neighbors = [b for b in nodes if b != a and not adjacent(a, b)]
            if len(non_neighbors) != 1:
                return 0
            opposite[a] = non_neighbors[0]

        # Must be an involution with no fixed points.
        for a, b in opposite.items():
            if b == a:
                return 0
            if opposite.get(b) != a:
                return 0

        # Sanity: any non-opposite pair must be an edge.
        for a, b in itertools.combinations(nodes, 2):
            if opposite[a] == b:
                if adjacent(a, b):
                    return 0
            else:
                if not adjacent(a, b):
                    return 0

        return len(nodes) // 2

    def is_cuboctahedron_graph(self) -> bool:
        # Cuboctahedron: 12 vertices, 24 edges, regular degree 4.
        return (
            len(self.nodes) == 12
            and len(self.edges) == 24
            and self.degree_sequence() == [4] * 12
        )


def motif_components_for_cross_assessors(
    dim: int, *, atol: float = 1e-12, max_nodes: int | None = None, seed: int = 0
) -> list[MotifComponent]:
    """
    Build the diagonal-zero-product graph over cross assessors and return its connected components.

    Notes:
    - For dim=16, this recovers the de Marrais 7 box-kites as octahedron components on 42 nodes.
    - For larger dims, full enumeration may be expensive; use max_nodes for deterministic sampling.
    """
    nodes = cross_assessors(dim)
    if max_nodes is not None and max_nodes < len(nodes):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(nodes), size=max_nodes, replace=False)
        nodes = [nodes[i] for i in sorted(idx.tolist())]

    # XOR-bucket pruning (necessary condition for a 2-blade * 2-blade diagonal cancellation).
    buckets: dict[int, list[Assessor]] = defaultdict(list)
    for a in nodes:
        buckets[xor_bucket(a)].append(a)

    adj: dict[Assessor, set[Assessor]] = {a: set() for a in nodes}
    edges: dict[tuple[Assessor, Assessor], list[tuple[float, float]]] = {}

    for bucket_nodes in buckets.values():
        for a, b in itertools.combinations(sorted(bucket_nodes), 2):
            sols = diagonal_zero_products(dim, a, b, atol=atol)
            if sols:
                adj[a].add(b)
                adj[b].add(a)
                edges[(a, b)] = sols

    # Keep only nodes that participate in at least one edge.
    active_nodes = {n for n, neigh in adj.items() if neigh}
    if not active_nodes:
        return []

    # Connected components over active nodes.
    seen: set[Assessor] = set()
    comps: list[MotifComponent] = []
    for start in sorted(active_nodes):
        if start in seen:
            continue
        stack = [start]
        comp_nodes: set[Assessor] = set()
        while stack:
            x = stack.pop()
            if x in comp_nodes:
                continue
            comp_nodes.add(x)
            for y in adj[x]:
                if y in active_nodes and y not in comp_nodes:
                    stack.append(y)
        seen |= comp_nodes

        comp_edges = {
            (a, b)
            for (a, b) in edges.keys()
            if a in comp_nodes and b in comp_nodes
        }
        comps.append(
            MotifComponent(dim=dim, nodes=frozenset(comp_nodes), edges=frozenset(comp_edges))
        )

    return sorted(comps, key=lambda c: (len(c.nodes), sorted(c.nodes)))
