from __future__ import annotations

from itertools import permutations

from gemini_physics.m3_cd_transfer import OctonionTable, classify_m3, compute_m3_octonion_basis


def _perm_parity_sign(original: tuple[int, int, int], permuted: tuple[int, int, int]) -> int:
    """
    Return +1 for even permutations, -1 for odd permutations, relative to `original`.

    For n=3, we can compute parity by inversion counting on positions.
    """

    pos = {v: i for i, v in enumerate(original)}
    p = [pos[v] for v in permuted]
    inv = 0
    for a in range(3):
        for b in range(a + 1, 3):
            if p[a] > p[b]:
                inv += 1
    return -1 if (inv % 2) else 1


def test_m3_distinct_triples_split_42_scalar_168_vector() -> None:
    oct_table = OctonionTable()

    counts: dict[str, int] = {"zero": 0, "scalar": 0, "vector": 0, "mixed": 0}
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                if len({i, j, k}) < 3:
                    continue
                res = compute_m3_octonion_basis(i, j, k, oct_table=oct_table)
                cls = classify_m3(res)
                counts[cls.kind] += 1

    assert counts["zero"] == 0
    assert counts["mixed"] == 0
    assert counts["scalar"] == 42
    assert counts["vector"] == 168
    assert sum(counts.values()) == 7 * 6 * 5


def test_m3_scalar_outputs_are_exactly_fano_lines() -> None:
    oct_table = OctonionTable()
    lines = oct_table.fano_lines()

    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                if len({i, j, k}) < 3:
                    continue
                res = compute_m3_octonion_basis(i, j, k, oct_table=oct_table)
                cls = classify_m3(res)

                is_fano_line = frozenset((i, j, k)) in lines
                if is_fano_line:
                    assert cls.kind == "scalar"
                    assert cls.index == 0
                    assert abs(cls.value) == 2
                else:
                    assert cls.kind == "vector"
                    assert 1 <= cls.index <= 7
                    assert abs(cls.value) == 2


def test_m3_scalar_sign_flips_with_permutation_parity() -> None:
    oct_table = OctonionTable()

    for triple in oct_table.oriented_triples():
        base_res = compute_m3_octonion_basis(*triple, oct_table=oct_table)
        base = classify_m3(base_res)
        assert base.kind == "scalar"
        assert abs(base.value) == 2

        for perm in permutations(triple):
            res = compute_m3_octonion_basis(*perm, oct_table=oct_table)
            cls = classify_m3(res)
            assert cls.kind == "scalar"
            assert abs(cls.value) == 2

            expected = base.value * _perm_parity_sign(triple, perm)
            assert cls.value == expected


def test_m3_repeated_indices_are_zero() -> None:
    oct_table = OctonionTable()

    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                if len({i, j, k}) == 3:
                    continue
                res = compute_m3_octonion_basis(i, j, k, oct_table=oct_table)
                cls = classify_m3(res)
                assert cls.kind == "zero"
