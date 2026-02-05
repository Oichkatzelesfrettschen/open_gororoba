import numpy as np

from gemini_physics.optimized_algebra import cd_multiply_jit


def _norm_sq(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def _associator(a: np.ndarray, b: np.ndarray, c: np.ndarray, dim: int) -> np.ndarray:
    return cd_multiply_jit(cd_multiply_jit(a, b, dim), c, dim) - cd_multiply_jit(
        a, cd_multiply_jit(b, c, dim), dim
    )


def test_associativity_holds_in_1_2_4() -> None:
    rng = np.random.default_rng(0)
    for dim in (1, 2, 4):
        for _ in range(200):
            a = rng.normal(size=dim)
            b = rng.normal(size=dim)
            c = rng.normal(size=dim)
            assoc = _associator(a, b, c, dim)
            assert np.allclose(assoc, 0.0, atol=1e-10, rtol=1e-10)


def test_nonassociativity_exists_in_8() -> None:
    rng = np.random.default_rng(1)
    dim = 8
    found = False
    for _ in range(500):
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        c = rng.normal(size=dim)
        assoc = _associator(a, b, c, dim)
        if _norm_sq(assoc) > 1e-8:
            found = True
            break
    assert found, "Expected to find a non-associative triple in 8D."


def test_norm_composition_holds_in_8() -> None:
    rng = np.random.default_rng(2)
    dim = 8
    for _ in range(200):
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        ab = cd_multiply_jit(a, b, dim)
        left = _norm_sq(ab)
        right = _norm_sq(a) * _norm_sq(b)
        assert np.isclose(left, right, rtol=1e-8, atol=1e-8)


def test_norm_composition_fails_in_16() -> None:
    rng = np.random.default_rng(3)
    dim = 16
    found = False
    for _ in range(1000):
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        ab = cd_multiply_jit(a, b, dim)
        left = _norm_sq(ab)
        right = _norm_sq(a) * _norm_sq(b)
        if abs(left - right) / max(1.0, abs(right)) > 1e-3:
            found = True
            break
    assert found, "Expected norm-composition to fail for some 16D pairs."


def test_sedenion_has_explicit_zero_divisors() -> None:
    """
    In the 16D Cayley-Dickson algebra (sedenions) there exist non-zero a,b with a*b = 0.

    This test uses basis conventions consistent with the canonical Cayley-Dickson table where
    e1*e2 = e3 and e1*e4 = e5 (matching standard presentations, e.g. de Marrais' sedenion tables).
    """
    dim = 16

    # One explicit identity (verified against the repo's multiplication convention):
    # (e1 + e10) * (e4 - e15) = 0
    a = np.zeros(dim)
    b = np.zeros(dim)
    a[1] = 1.0
    a[10] = 1.0
    b[4] = 1.0
    b[15] = -1.0

    prod = cd_multiply_jit(a, b, dim)
    assert np.allclose(prod, 0.0, atol=1e-12, rtol=0.0)
