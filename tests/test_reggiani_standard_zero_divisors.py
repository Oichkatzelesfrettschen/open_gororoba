from collections import Counter

import numpy as np

from gemini_physics.reggiani_replication import (
    assert_standard_zero_divisor_annihilators,
    standard_zero_divisors,
)
from gemini_physics.sedenion_annihilator import annihilator_info, is_reggiani_zd


def test_reggiani_standard_zero_divisors_count_is_84() -> None:
    zds = standard_zero_divisors()
    assert len(zds) == 84

    # All have squared norm 2 (so ||v|| = sqrt2) and satisfy Reggiani's ZD(S) normalization.
    for zd in zds:
        assert np.isclose(float(np.dot(zd.vector, zd.vector)), 2.0, atol=1e-12, rtol=0.0)
        assert is_reggiani_zd(zd.vector)


def test_standard_zero_divisors_have_annihilator_nullity_4_on_both_sides() -> None:
    pairs = []
    for zd in standard_zero_divisors():
        info = annihilator_info(zd.vector)
        pairs.append((info.left_nullity, info.right_nullity))
    assert Counter(pairs) == {(4, 4): 84}


def test_standard_zero_divisor_nullspace_basis_vectors_annihilate() -> None:
    for zd in standard_zero_divisors():
        assert_standard_zero_divisor_annihilators(zd)

