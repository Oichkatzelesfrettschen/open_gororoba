from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from gemini_physics.optimized_algebra import cd_multiply_jit

Assessor = tuple[int, int]  # (low, high), both in 1..15, low < high


@dataclass(frozen=True)
class BoxKite:
    """
    A de Marrais "box-kite": an octahedral 6-vertex structure of assessors.

    In this repo, we model an edge between two assessors if there exists a choice of diagonals
    (e_i +/- e_j) and (e_k +/- e_l) whose product is exactly zero under the 16D Cayley-Dickson
    multiplication convention implemented by `cd_multiply_jit`.
    """

    assessors: frozenset[Assessor]
    edges: frozenset[tuple[Assessor, Assessor]]  # undirected, ordered pairs with a < b


def _diag_vector(i: int, j: int, sign: float) -> np.ndarray:
    v = np.zeros(16)
    v[i] = 1.0
    v[j] = sign
    return v


def _has_zero_division(a: Assessor, b: Assessor, *, atol: float = 1e-12) -> bool:
    (i, j) = a
    (k, ell) = b
    for s in (1.0, -1.0):
        for t in (1.0, -1.0):
            prod = cd_multiply_jit(_diag_vector(i, j, s), _diag_vector(k, ell, t), 16)
            if np.allclose(prod, 0.0, atol=atol, rtol=0.0):
                return True
    return False


def diagonal_zero_products(
    a: Assessor, b: Assessor, *, atol: float = 1e-12
) -> list[tuple[float, float]]:
    """
    Return all sign choices (s,t) with s,t in {+1,-1} such that:

        (e_i + s e_j) * (e_k + t e_l) = 0

    where a=(i,j), b=(k,l).
    """
    (i, j) = a
    (k, ell) = b
    solutions: list[tuple[float, float]] = []
    for s in (1.0, -1.0):
        for t in (1.0, -1.0):
            prod = cd_multiply_jit(_diag_vector(i, j, s), _diag_vector(k, ell, t), 16)
            if np.allclose(prod, 0.0, atol=atol, rtol=0.0):
                solutions.append((s, t))
    return solutions


def edge_sign_type(a: Assessor, b: Assessor, *, atol: float = 1e-12) -> str:
    """
    Classify the diagonal-zero-product relationship between two assessors.

    de Marrais distinguishes "trefoil" vs "triple-zigzag" lanyards based on whether linked beads
    share the same internal sign or not. For an assessor pair edge, the diagonal zero-products are
    always of one of these two forms:

      - same-sign: (s,t) in {(+,+),(-,-)}  -> "+"
      - opposite-sign: (s,t) in {(+,-),(-,+)} -> "-"

    Returns "+" or "-".
    """
    sols = diagonal_zero_products(a, b, atol=atol)
    if not sols:
        raise ValueError(f"No diagonal zero-products exist for assessor pair {a} -- {b}")
    if (1.0, 1.0) in sols or (-1.0, -1.0) in sols:
        return "+"
    return "-"


def candidate_cross_assessors() -> list[Assessor]:
    """
    Return all "cross" imaginary pairs (low in 1..7, high in 8..15).

    These are the natural planes spanned by one octonion-like unit and one pure sedenion-like unit.
    """
    return [(i, j) for i in range(1, 8) for j in range(8, 16)]


def primitive_assessors(*, atol: float = 1e-12) -> list[Assessor]:
    """
    Compute the 42 assessors described by de Marrais by filtering cross-assessors to those that
    participate in at least one diagonal-zero product with another cross-assessor.
    """
    cross = candidate_cross_assessors()
    primitive: list[Assessor] = []
    for a in cross:
        ok = False
        for b in cross:
            if a == b:
                continue
            if _has_zero_division(a, b, atol=atol):
                ok = True
                break
        if ok:
            primitive.append(a)
    return sorted(primitive)


def primitive_unit_zero_divisors_for_assessor(a: Assessor) -> list[np.ndarray]:
    """
    Each assessor corresponds to a 2-plane spanned by (e_i, e_j); its two diagonals have four unit
    points:

        +/-(e_i + e_j) / sqrt2,  +/-(e_i - e_j) / sqrt2
    """
    (i, j) = a
    scale = 1.0 / np.sqrt(2.0)
    v1 = scale * _diag_vector(i, j, 1.0)
    v2 = scale * _diag_vector(i, j, -1.0)
    return [v1, -v1, v2, -v2]


def strut_signature(box_kite: BoxKite) -> int:
    """
    Compute de Marrais' "strut signature" for a box-kite.

    In the paper's strut table, each box-kite Roman numeral corresponds to the unique octonion
    index o in {1..7} *missing* from the low indices of the box-kite's six cross-assessors.
    """
    lows = {low for (low, _high) in box_kite.assessors}
    missing = set(range(1, 8)) - lows
    if len(missing) != 1:
        raise ValueError(f"Expected exactly one missing low index in {sorted(box_kite.assessors)}")
    return next(iter(missing))


def canonical_strut_table(box_kite: BoxKite, *, atol: float = 1e-12) -> dict[str, Assessor]:
    """
    Produce a deterministic A..F labeling for a box-kite compatible with de Marrais' "Strut Table"
    schematic:

      - (A, B, C) form a triple-zigzag face (all '-' edges) -- the "sail"
      - (D, E, F) form the opposite triple-zigzag face -- the "vent"
      - strut pairs are (A,F), (B,E), (C,D)

    There are two zigzag faces in each octahedral box-kite; we pick the lexicographically smaller
    one as (A,B,C) to make this labeling deterministic.
    """
    nodes = sorted(box_kite.assessors)
    edge_set = set(box_kite.edges)

    def adjacent(x: Assessor, y: Assessor) -> bool:
        return (x, y) in edge_set or (y, x) in edge_set

    # Opposite relation in an octahedron: each vertex has exactly one non-neighbor.
    opposite: dict[Assessor, Assessor] = {}
    for a in nodes:
        non_neighbors = [b for b in nodes if b != a and not adjacent(a, b)]
        if len(non_neighbors) != 1:
            raise ValueError(f"Expected unique opposite for {a}, got {non_neighbors}")
        opposite[a] = non_neighbors[0]

    triangles: list[tuple[Assessor, Assessor, Assessor]] = []
    for a, b, c in itertools.combinations(nodes, 3):
        if adjacent(a, b) and adjacent(b, c) and adjacent(a, c):
            triangles.append((a, b, c))

    zigzags = []
    for a, b, c in triangles:
        signs = [
            edge_sign_type(a, b, atol=atol),
            edge_sign_type(b, c, atol=atol),
            edge_sign_type(a, c, atol=atol),
        ]
        if signs.count("-") == 3:
            zigzags.append(tuple(sorted((a, b, c))))

    if len(zigzags) != 2:
        raise ValueError(f"Expected exactly 2 zigzag faces, got {len(zigzags)}")

    abc = sorted(zigzags)[0]
    A, B, C = abc
    F = opposite[A]
    E = opposite[B]
    D = opposite[C]

    # Sanity: the opposite face must also be a zigzag.
    def is_zigzag(tri: tuple[Assessor, Assessor, Assessor]) -> bool:
        a, b, c = tri
        s = [
            edge_sign_type(a, b, atol=atol),
            edge_sign_type(b, c, atol=atol),
            edge_sign_type(a, c, atol=atol),
        ]
        return s.count("-") == 3

    if not is_zigzag(tuple(sorted((D, E, F)))):
        raise ValueError("Derived opposite face is not a zigzag; labeling invariant failed")

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


def production_rule_1(a: Assessor, b: Assessor) -> Assessor:
    """
    de Marrais Production Rule #1 ("Three-Ring Circuits"), encoded for cross-assessors.

    Given two co-assessors (A,B) and (C,D), construct a third assessor (E,F) using XOR:
      E = A xor C = B xor D
      F = A xor D = B xor C

    In this repo, cross-assessors are represented as (low, high) with low in 1..7 and high in 8..15.
    The rule is applied by treating:
      a = (A,B),  b = (C,D)
    and returning (E,F) as an ordered pair (min, max).

    Note: This is a purely combinatorial construction; whether (E,F) is the "right" third assessor
    for a given lanyard orientation depends on sign/mode conventions. We validate its behavior
    via unit tests.
    """
    (A, B) = a
    (C, D) = b
    E = A ^ C
    F = A ^ D
    # Consistency checks from the paper's equalities.
    if E != (B ^ D):
        raise ValueError(f"PR#1 invariant failed: {a=}, {b=}, expected A^C == B^D")
    if F != (B ^ C):
        raise ValueError(f"PR#1 invariant failed: {a=}, {b=}, expected A^D == B^C")
    if E == F:
        raise ValueError(f"PR#1 degenerate: produced equal indices {E} from {a=} and {b=}")
    return (E, F) if E < F else (F, E)


def production_rule_2(
    a: Assessor, b: Assessor, *, atol: float = 1e-12
) -> tuple[Assessor, Assessor]:
    """
    de Marrais Production Rule #2 ("Skew-Symmetric Twisting"), encoded for cross-assessors.

    The paper describes a construction that, given co-assessors (A,B) and (C,D), creates precisely
    two new assessors that are co-assessors with each other, but not co-assessors with either
    propagator.

    In index form, the two possible "pair swaps" reduce to:
      - (A,D) with (B,C)
      - (B,D) with (A,C)

    Here we operationalize PR#2 by constructing both candidate swap pairs and selecting the unique
    one which satisfies the defining property in this repo:
      - the outputs are co-assessors with each other (have diagonal zero-products), and
      - neither output is a co-assessor with either propagator.
    """

    (A, B) = a
    (C, D) = b

    cand1 = (tuple(sorted((A, D))), tuple(sorted((B, C))))
    cand2 = (tuple(sorted((B, D))), tuple(sorted((A, C))))

    def valid_pair(p: Assessor, q: Assessor) -> bool:
        if p == q:
            return False
        if not diagonal_zero_products(p, q, atol=atol):
            return False
        if diagonal_zero_products(p, a, atol=atol) or diagonal_zero_products(p, b, atol=atol):
            return False
        if diagonal_zero_products(q, a, atol=atol) or diagonal_zero_products(q, b, atol=atol):
            return False
        return True

    val1 = valid_pair(*cand1)
    val2 = valid_pair(*cand2)
    if val1 == val2:
        raise ValueError(
            f"PR#2 expected exactly one valid swap for {a=} and {b=}, got {val1=} {val2=}"
        )

    return cand1 if val1 else cand2


O_TRIPS: tuple[tuple[int, int, int], ...] = (
    # The 7 oriented-free octonion triplets (Fano-plane lines),
    # as used by de Marrais' GoTo listings.
    (1, 2, 3),
    (1, 4, 5),
    (1, 6, 7),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 5, 6),
)


def automorpheme_assessors(o_trip: tuple[int, int, int]) -> frozenset[Assessor]:
    """
    Build the 12 assessors belonging to one of de Marrais' "automorphemes" (GoTo listings).

    de Marrais describes that for each octonion O-trip there are 12 Assessors: each pairing of
    the three O-trip letters with the four "O-copy Sedenions".

    In index form, the "Behind the 8-Ball Theorem" implies that the excluded sedenion indices are:
      {8, 8+o1, 8+o2, 8+o3}
    leaving exactly four high indices in 8..15 to pair with each low octonion index in the O-trip.
    """
    if o_trip not in O_TRIPS:
        raise ValueError(f"Unknown O-trip: {o_trip}")

    excluded_highs = {8, *(8 ^ o for o in o_trip)}
    allowed_highs = [h for h in range(8, 16) if h not in excluded_highs]
    if len(allowed_highs) != 4:
        raise ValueError(f"Expected 4 allowed highs for {o_trip=}, got {allowed_highs}")

    return frozenset((o, h) for o in o_trip for h in allowed_highs)


def automorphemes() -> list[frozenset[Assessor]]:
    """
    Return the 7 automorpheme assessor sets (one per O-trip).
    """
    return [automorpheme_assessors(t) for t in O_TRIPS]


def automorphemes_containing_assessor(a: Assessor) -> list[tuple[int, int, int]]:
    """
    Return all O-trips whose automorphemes contain the given assessor.
    """
    low, high = a
    if not (1 <= low <= 7 and 8 <= high <= 15):
        raise ValueError(f"Expected a cross-assessor (low 1..7, high 8..15), got {a}")
    return [t for t in O_TRIPS if a in automorpheme_assessors(t)]


def production_rule_3(o_trip: tuple[int, int, int], a: Assessor) -> tuple[int, int, int]:
    """
    de Marrais Production Rule #3 (automorpheme uniqueness), encoded for cross-assessors.

    Interpreting the paper's statement:
      "The unique automorpheme which shares the o and S of another's Assessor will also contain
       that Assessor, and be the only distinct automorpheme to do so in the whole Sedenion space."

    In this repo's representation, an assessor is exactly the index pair (o, S). For a fixed cross
    assessor (o, S) with S in 9..15 (i.e., excluding 8 and 8+o), the Fano-plane incidence rules
    imply:
      - There are 3 O-trips containing o.
      - Exactly one of them also contains s = (S xor 8), and therefore excludes S (8-ball theorem).
      - The other two O-trips contain o and allow S, so (o, S) appears in exactly 2 automorphemes.

    Given one such automorpheme (identified by its O-trip), this function returns the unique other
    O-trip whose automorpheme also contains the assessor.
    """
    if o_trip not in O_TRIPS:
        raise ValueError(f"Unknown O-trip: {o_trip}")
    if a not in automorpheme_assessors(o_trip):
        raise ValueError(f"Assessor {a} is not contained in automorpheme for {o_trip=}")

    candidates = automorphemes_containing_assessor(a)
    if len(candidates) != 2:
        raise ValueError(f"Expected exactly 2 automorphemes for {a=}, got {candidates}")
    other = candidates[0] if candidates[1] == o_trip else candidates[1]
    if other == o_trip:
        raise ValueError(f"PR#3 degenerate: failed to find distinct other automorpheme for {a=}")
    return other


def box_kites(*, atol: float = 1e-12) -> list[BoxKite]:
    """
    Compute the 7 box-kites as connected components of the assessor adjacency graph.
    """
    assessors = primitive_assessors(atol=atol)
    adj: dict[Assessor, set[Assessor]] = {a: set() for a in assessors}
    for a, b in itertools.combinations(assessors, 2):
        if _has_zero_division(a, b, atol=atol):
            adj[a].add(b)
            adj[b].add(a)

    seen: set[Assessor] = set()
    components: list[set[Assessor]] = []
    for start in assessors:
        if start in seen:
            continue
        stack = [start]
        comp: set[Assessor] = set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            for y in adj[x]:
                if y not in comp:
                    stack.append(y)
        seen |= comp
        components.append(comp)

    results: list[BoxKite] = []
    for comp in components:
        edges: set[tuple[Assessor, Assessor]] = set()
        for a, b in itertools.combinations(sorted(comp), 2):
            if b in adj[a]:
                edges.add((a, b))
        results.append(BoxKite(assessors=frozenset(comp), edges=frozenset(edges)))

    return sorted(results, key=lambda bk: sorted(bk.assessors))
