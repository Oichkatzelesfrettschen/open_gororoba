import numpy as np

from gemini_physics.optimized_algebra import cd_multiply_jit
from gemini_physics.sedenion_annihilator import (
    annihilator_info,
    find_left_annihilator_vector,
    is_reggiani_zd,
    is_zero_divisor,
)


def test_known_zero_divisor_has_nontrivial_annihilator() -> None:
    # One explicit de Marrais-style identity under this repo's Cayley-Dickson convention:
    # (e1 + e10) * (e4 - e15) = 0
    a = np.zeros(16)
    b = np.zeros(16)
    a[1] = 1.0
    a[10] = 1.0
    b[4] = 1.0
    b[15] = -1.0

    prod = cd_multiply_jit(a, b, 16)
    assert np.allclose(prod, 0.0, atol=1e-12, rtol=0.0)

    info = annihilator_info(a)
    assert info.left_nullity >= 1
    assert is_reggiani_zd(a)

    found = find_left_annihilator_vector(a)
    assert found is not None
    assert np.allclose(cd_multiply_jit(a, found, 16), 0.0, atol=1e-10, rtol=0.0)


def test_basis_unit_is_not_zero_divisor() -> None:
    # Imaginary basis units square to -1 and should be invertible, hence no annihilator.
    e1 = np.zeros(16)
    e1[1] = 1.0
    info = annihilator_info(e1)
    assert info.left_nullity == 0
    assert info.right_nullity == 0
    assert not is_reggiani_zd(e1)


def test_unit_normalized_zero_divisor_is_not_in_reggiani_zd_s() -> None:
    a = np.zeros(16)
    a[1] = 1.0
    a[10] = 1.0
    a = a / np.sqrt(2.0)  # now ||a|| = 1
    assert np.isclose(float(np.dot(a, a)), 1.0, atol=1e-12, rtol=0.0)
    assert is_zero_divisor(a)
    assert not is_reggiani_zd(a)
