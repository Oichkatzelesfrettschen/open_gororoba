from __future__ import annotations

import numpy as np

from gemini_physics.cd_xor_heuristics import (
    xor_balanced_four_tuple,
    xor_bucket_necessary_for_two_blade_vs_balanced_four_blade,
    xor_bucket_necessary_for_two_blade_zero_product,
    xor_pairing_buckets_for_balanced_four_tuple,
)
from gemini_physics.optimized_algebra import cd_multiply_jit


def _basis(dim: int, idx: int, scale: float = 1.0) -> np.ndarray:
    v = np.zeros(dim)
    v[idx] = scale
    return v


def test_xor_bucket_is_necessary_for_observed_2blade_zero_products_in_16d() -> None:
    """
    Empirical but deterministic check: for a moderate search window in 16D, every
    observed 2-blade (ei+/-ej) * 2-blade (ek+/-el) zero-product satisfies XOR-bucket
    coincidence: (i xor j) == (k xor l).
    """

    dim = 16
    found = 0

    # Search all unordered index pairs in 1..15 and both signs on the second blade.
    for i in range(1, 16):
        for j in range(i + 1, 16):
            u = _basis(dim, i) + _basis(dim, j)
            for k in range(1, 16):
                for ell in range(k + 1, 16):
                    for sign in (-1.0, 1.0):
                        v = _basis(dim, k) + sign * _basis(dim, ell)
                        prod = cd_multiply_jit(u, v, dim)
                        if np.allclose(prod, 0.0, atol=1e-12, rtol=0.0):
                            found += 1
                            assert xor_bucket_necessary_for_two_blade_zero_product(i, j, k, ell)

    assert found > 0, "Expected to find at least one 2-blade zero-product in 16D."


def test_xor_pairing_buckets_for_balanced_four_tuple() -> None:
    # Pick a simple balanced tuple: 1^2^4^7 == 0.
    a, b, c, d = 1, 2, 4, 7
    assert xor_balanced_four_tuple(a, b, c, d)

    buckets = xor_pairing_buckets_for_balanced_four_tuple(a, b, c, d)
    assert buckets == {a ^ b, a ^ c, a ^ d}

    # Each perfect matching has a shared XOR:
    assert (a ^ b) == (c ^ d)
    assert (a ^ c) == (b ^ d)
    assert (a ^ d) == (b ^ c)


def test_xor_bucket_necessary_for_two_blade_vs_balanced_four_blade() -> None:
    a, b, c, d = 1, 2, 4, 7  # balanced

    # Any 2-blade whose XOR matches one of the three pairing buckets passes the filter.
    assert xor_bucket_necessary_for_two_blade_vs_balanced_four_blade(1, 2, a, b, c, d)
    assert xor_bucket_necessary_for_two_blade_vs_balanced_four_blade(1, 4, a, b, c, d)
    assert xor_bucket_necessary_for_two_blade_vs_balanced_four_blade(1, 7, a, b, c, d)

    # A non-matching bucket is rejected.
    assert not xor_bucket_necessary_for_two_blade_vs_balanced_four_blade(1, 3, a, b, c, d)
