import itertools

import numpy as np

from gemini_physics.de_marrais_boxkites import (
    box_kites,
    candidate_cross_assessors,
    canonical_strut_table,
    diagonal_zero_products,
    edge_sign_type,
    primitive_assessors,
    primitive_unit_zero_divisors_for_assessor,
    production_rule_1,
    production_rule_2,
    strut_signature,
)


def test_primitive_assessors_count_is_42() -> None:
    ass = primitive_assessors()
    assert len(ass) == 42

    # All are cross pairs (1..7 with 8..15).
    for low, high in ass:
        assert 1 <= low <= 7
        assert 8 <= high <= 15

    # Exactly 14 cross-pairs are excluded: (i,8) and (i,i+8) for i=1..7.
    cross = set(candidate_cross_assessors())
    excluded = cross - set(ass)
    expected_excluded = {(i, 8) for i in range(1, 8)} | {(i, i + 8) for i in range(1, 8)}
    assert excluded == expected_excluded


def test_primitive_unit_zero_divisors_count_is_168() -> None:
    # de Marrais' abstract: 168 primitive unit zero-divisors arranged as quartets
    # along 42 assessors.
    # Here: each assessor yields 4 unit points on its two diagonals.
    ass = primitive_assessors()
    zs = [z for a in ass for z in primitive_unit_zero_divisors_for_assessor(a)]
    assert len(zs) == 168
    for z in zs:
        assert np.isclose(float(np.dot(z, z)), 1.0, atol=1e-12, rtol=0.0)


def test_box_kites_partition_assessors_into_7_octahedra() -> None:
    bks = box_kites()
    assert len(bks) == 7
    assert sorted(len(bk.assessors) for bk in bks) == [6] * 7

    # Disjoint and cover all assessors
    all_nodes = set().union(*(bk.assessors for bk in bks))
    assert all_nodes == set(primitive_assessors())
    for i, j in itertools.combinations(range(len(bks)), 2):
        assert bks[i].assessors.isdisjoint(bks[j].assessors)

    # Each box-kite is an octahedron graph: 6 vertices, 12 edges, degree 4 at every vertex.
    for bk in bks:
        assert len(bk.edges) == 12
        deg = {a: 0 for a in bk.assessors}
        for a, b in bk.edges:
            assert a in bk.assessors and b in bk.assessors
            deg[a] += 1
            deg[b] += 1
        assert sorted(deg.values()) == [4] * 6


def test_box_kite_edges_have_sign_solutions() -> None:
    # For any box-kite edge (a,b) there exists at least one diagonal sign pairing
    # that multiplies to 0.
    for bk in box_kites():
        for a, b in bk.edges:
            sols = diagonal_zero_products(a, b)
            assert sols, f"Expected at least one diagonal zero-product for edge {a}--{b}"


def test_box_kite_strut_signatures_are_a_partition_of_1_to_7() -> None:
    # de Marrais: each box-kite has a "strut signature" in {1..7} (Roman numeral labels),
    # which is the unique octonion index missing from the low indices of its six cross-assessors.
    sigs = [strut_signature(bk) for bk in box_kites()]
    assert sorted(sigs) == list(range(1, 8))


def test_each_box_kite_has_6_trefoil_faces_and_2_zigzag_faces() -> None:
    # de Marrais: each box-kite has 4 "sails" (triangular faces) and 4 "vents"; among each set,
    # 3 are trefoils and 1 is a triple-zigzag, so overall: 6 trefoils + 2 zigzags.
    for bk in box_kites():
        nodes = sorted(bk.assessors)
        edge_set = set(bk.edges)

        def adjacent(x, y, *, _edge_set=edge_set) -> bool:
            return (x, y) in _edge_set or (y, x) in _edge_set

        triangles = [
            (a, b, c)
            for a, b, c in itertools.combinations(nodes, 3)
            if adjacent(a, b) and adjacent(b, c) and adjacent(a, c)
        ]
        assert len(triangles) == 8

        trefoil = 0
        zigzag = 0
        other = 0
        for a, b, c in triangles:
            signs = [edge_sign_type(a, b), edge_sign_type(b, c), edge_sign_type(a, c)]
            if signs.count("-") == 3:
                zigzag += 1
            elif signs.count("+") == 2 and signs.count("-") == 1:
                trefoil += 1
            else:
                other += 1

        assert (trefoil, zigzag, other) == (6, 2, 0)


def test_strut_pairs_are_the_unique_non_edges() -> None:
    # In an octahedron graph, each vertex has a unique opposite vertex (the only non-neighbor),
    # and these 3 opposite pairs are the "struts" in de Marrais' diagrams.
    for bk in box_kites():
        nodes = sorted(bk.assessors)
        edge_set = set(bk.edges)

        def adjacent(x, y, *, _edge_set=edge_set) -> bool:
            return (x, y) in _edge_set or (y, x) in _edge_set

        opposites = {}
        for a in nodes:
            non_neighbors = [b for b in nodes if b != a and not adjacent(a, b)]
            assert len(non_neighbors) == 1
            opposites[a] = non_neighbors[0]

        struts = {tuple(sorted((a, opposites[a]))) for a in nodes}
        assert len(struts) == 3

        # Struts are not diagonal-zero-product edges.
        for a, b in struts:
            assert diagonal_zero_products(a, b) == []


def test_canonical_strut_table_labels_two_zigzag_faces() -> None:
    for bk in box_kites():
        tab = canonical_strut_table(bk)
        A, B, C = tab["A"], tab["B"], tab["C"]
        D, E, F = tab["D"], tab["E"], tab["F"]

        def is_zigzag(a, b, c) -> bool:
            signs = [edge_sign_type(a, b), edge_sign_type(b, c), edge_sign_type(a, c)]
            return signs.count("-") == 3

        assert is_zigzag(A, B, C)
        assert is_zigzag(D, E, F)


def test_production_rule_1_reconstructs_a_trefoil_third_vertex() -> None:
    # For each edge (a,b) inside a box-kite, Production Rule #1 should yield a third assessor c that
    # closes a valid co-assessor trio. In de Marrais' terminology, the induced 3-edge sign pattern
    # is either a trefoil (two '+' edges and one '-' edge) or a triple-zigzag (three '-' edges).
    for bk in box_kites():
        edge_set = set(bk.edges)

        def adjacent(x, y, *, _edge_set=edge_set) -> bool:
            return (x, y) in _edge_set or (y, x) in _edge_set

        for a, b in bk.edges:
            c = production_rule_1(a, b)
            assert c in bk.assessors
            assert adjacent(a, c)
            assert adjacent(b, c)

            signs = [edge_sign_type(a, b), edge_sign_type(a, c), edge_sign_type(b, c)]
            assert (signs.count("+"), signs.count("-")) in {(2, 1), (0, 3)}


def test_production_rule_2_creates_two_new_co_assessors() -> None:
    # For each co-assessor edge (a,b), PR#2 should produce two new assessors (p,q) that:
    # - are co-assessors with each other (have diagonal zero-products)
    # - are not co-assessors with either propagator (a or b)
    primitive = set(primitive_assessors())
    for bk in box_kites():
        for a, b in bk.edges:
            p, q = production_rule_2(a, b)
            assert p != q
            assert p in primitive and q in primitive

            assert diagonal_zero_products(p, q), (
                f"Expected PR#2 outputs to be co-assessors: {p=} {q=}"
            )

            assert diagonal_zero_products(p, a) == []
            assert diagonal_zero_products(p, b) == []
            assert diagonal_zero_products(q, a) == []
            assert diagonal_zero_products(q, b) == []
