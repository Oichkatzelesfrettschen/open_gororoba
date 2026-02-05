from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class NilpotentJordanType:
    """
    Jordan type (partition) for a nilpotent endomorphism: block sizes in descending order.
    """

    blocks: tuple[int, ...]


def _as_sympy_matrix(a: list[list[int]]) -> sp.Matrix:
    return sp.Matrix(a)


def nilpotency_index(a: list[list[int]]) -> int | None:
    """
    Return the smallest k >= 1 such that A^k = 0, or None if not nilpotent (within n steps).
    """

    m = _as_sympy_matrix(a)
    n = m.rows
    if n != m.cols:
        raise ValueError("matrix must be square")

    p = m
    for k in range(1, n + 1):
        if p.is_zero_matrix:
            return k
        p = p * m
    return None


def jordan_type_nilpotent(a: list[list[int]]) -> NilpotentJordanType:
    """
    Compute the Jordan block sizes for a nilpotent matrix using kernel dimensions of powers.

    For nilpotent A (size n), let d_k = dim ker(A^k) for k=0..n with d_0 = 0.
    Then c_k = d_k - d_{k-1} is the number of Jordan blocks of size >= k.
    Blocks of exact size k: c_k - c_{k+1}.
    """

    m = _as_sympy_matrix(a)
    n = m.rows
    if n != m.cols:
        raise ValueError("matrix must be square")

    if nilpotency_index(a) is None:
        raise ValueError("matrix is not nilpotent (within n steps)")

    d: list[int] = [0]
    p = m
    for _k in range(1, n + 1):
        d.append(n - int(p.rank()))
        p = p * m

    # c_k = number of blocks with size >= k
    c = [0] * (n + 2)
    for k in range(1, n + 1):
        c[k] = d[k] - d[k - 1]

    blocks: list[int] = []
    for k in range(1, n + 1):
        exact = c[k] - c[k + 1]
        if exact < 0:
            raise AssertionError("invalid Jordan reconstruction (negative multiplicity)")
        blocks.extend([k] * exact)

    blocks.sort(reverse=True)
    if sum(blocks) != n:
        raise AssertionError("Jordan blocks do not sum to matrix size")

    return NilpotentJordanType(blocks=tuple(blocks))
