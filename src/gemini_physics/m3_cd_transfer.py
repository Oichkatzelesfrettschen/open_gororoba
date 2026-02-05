from __future__ import annotations

from dataclasses import dataclass

_O_DIM = 8
_S_DIM = 16


def _zeros(n: int) -> list[int]:
    return [0] * n


def _vec_add(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b, strict=True)]


def _vec_sub(a: list[int], b: list[int]) -> list[int]:
    return [x - y for x, y in zip(a, b, strict=True)]


class OctonionTable:
    """
    Integer-only octonion multiplication table on the standard basis {e0=1,e1,...,e7}.

    The oriented triples below encode the Fano-plane structure (sign via orientation).
    """

    def __init__(self) -> None:
        self._table: dict[tuple[int, int], tuple[int, int]] = {}

        for i in range(_O_DIM):
            self._table[(0, i)] = (1, i)
            self._table[(i, 0)] = (1, i)

        for i in range(1, _O_DIM):
            self._table[(i, i)] = (-1, 0)

        # Standard Fano-plane oriented lines.
        oriented = [
            (1, 2, 3),
            (1, 4, 5),
            (1, 7, 6),
            (2, 4, 6),
            (2, 5, 7),
            (3, 4, 7),
            (3, 6, 5),
        ]

        for i, j, k in oriented:
            self._table[(i, j)] = (1, k)
            self._table[(j, i)] = (-1, k)

            self._table[(j, k)] = (1, i)
            self._table[(k, j)] = (-1, i)

            self._table[(k, i)] = (1, j)
            self._table[(i, k)] = (-1, j)

        self._oriented_triples = oriented

    def mul_basis(self, i: int, j: int) -> tuple[int, int]:
        return self._table[(i, j)]

    def fano_lines(self) -> set[frozenset[int]]:
        return {frozenset((i, j, k)) for i, j, k in self._oriented_triples}

    def oriented_triples(self) -> list[tuple[int, int, int]]:
        return list(self._oriented_triples)


def _o_conj_vec(v: list[int]) -> list[int]:
    return [v[0]] + [-x for x in v[1:]]


def _o_mul_vec(oct_table: OctonionTable, a: list[int], b: list[int]) -> list[int]:
    """
    Bilinear octonion product for sparse integer vectors on the basis.

    This is intentionally minimal: it is used only in the CD doubling for
    basis-embedded inputs, where vectors remain very sparse.
    """

    out = _zeros(_O_DIM)
    for ia, va in enumerate(a):
        if va == 0:
            continue
        for ib, vb in enumerate(b):
            if vb == 0:
                continue
            s, k = oct_table.mul_basis(ia, ib)
            out[k] += va * vb * s
    return out


def _s_mul(oct_table: OctonionTable, x: list[int], y: list[int]) -> list[int]:
    # Cayley-Dickson multiply on S = O \oplus O.
    a, b = x[:_O_DIM], x[_O_DIM:]
    c, d = y[:_O_DIM], y[_O_DIM:]

    ac = _o_mul_vec(oct_table, a, c)
    db = _o_mul_vec(oct_table, _o_conj_vec(d), b)
    re = _vec_sub(ac, db)

    da = _o_mul_vec(oct_table, d, a)
    bc = _o_mul_vec(oct_table, b, _o_conj_vec(c))
    im = _vec_add(da, bc)

    return re + im


def _p_map_int(s: list[int]) -> list[int]:
    a, b = s[:_O_DIM], s[_O_DIM:]
    summed = _vec_add(a, b)
    if any(v % 2 != 0 for v in summed):
        raise ValueError("p-map requires even coordinates for integer division by 2")
    return [v // 2 for v in summed]


def _h_map_int(s: list[int]) -> list[int]:
    a, b = s[:_O_DIM], s[_O_DIM:]
    diff = _vec_sub(a, b)
    if any(v % 2 != 0 for v in diff):
        raise ValueError("h-map requires even coordinates for integer division by 2")
    half = [v // 2 for v in diff]
    return half + [-v for v in half]


def compute_m3_octonion_basis(
    i: int, j: int, k: int, *, oct_table: OctonionTable | None = None
) -> list[int]:
    """
    Compute the specific trilinear operation used by `src/m3_table_cd.py`,
    but with integer-only arithmetic for exactness.

    Inputs i,j,k are basis indices in 0..7; typical usage is 1..7.
    Output is an 8-vector of integers in the octonion basis.
    """

    if oct_table is None:
        oct_table = OctonionTable()

    def to_s_vec(idx: int) -> list[int]:
        v = _zeros(_S_DIM)
        v[idx] = 1
        v[idx + _O_DIM] = 1
        return v

    x = to_s_vec(i)
    y = to_s_vec(j)
    z = to_s_vec(k)

    xy = _s_mul(oct_table, x, y)
    term1_s = _s_mul(oct_table, _h_map_int(xy), z)
    term1 = _p_map_int(term1_s)

    yz = _s_mul(oct_table, y, z)
    term2_s = _s_mul(oct_table, x, _h_map_int(yz))
    term2 = _p_map_int(term2_s)

    return _vec_sub(term1, term2)


@dataclass(frozen=True, slots=True)
class M3Classification:
    kind: str  # "zero" | "scalar" | "vector" | "mixed"
    index: int  # basis index for scalar/vector; 0 for scalar; -1 otherwise
    value: int  # coefficient at index (exact integer)


def classify_m3(o_vec: list[int]) -> M3Classification:
    if len(o_vec) != _O_DIM:
        raise ValueError("expected an 8-vector in the octonion basis")

    if all(v == 0 for v in o_vec):
        return M3Classification(kind="zero", index=-1, value=0)

    if o_vec[0] != 0 and all(v == 0 for v in o_vec[1:]):
        return M3Classification(kind="scalar", index=0, value=o_vec[0])

    if o_vec[0] == 0:
        nz = [(idx, v) for idx, v in enumerate(o_vec[1:], start=1) if v != 0]
        if len(nz) == 1:
            idx, v = nz[0]
            return M3Classification(kind="vector", index=idx, value=v)

    return M3Classification(kind="mixed", index=-1, value=0)
