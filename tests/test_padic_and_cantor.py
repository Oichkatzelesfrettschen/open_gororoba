from __future__ import annotations

from fractions import Fraction

from gemini_physics.padic import (
    abs_p,
    cantor_function_on_cantor,
    is_dyadic,
    ternary_digits_power3,
    vp,
)


def test_vp_and_abs_p_basic() -> None:
    q = Fraction(12, 5)  # 12 = 2^2 * 3
    assert vp(q, 2) == 2
    assert abs_p(q, 2) == 2 ** (-2)

    assert vp(q, 3) == 1
    assert abs_p(q, 3) == 3 ** (-1)

    assert vp(q, 5) == -1
    assert abs_p(q, 5) == 5 ** (1)


def test_is_dyadic() -> None:
    assert is_dyadic(Fraction(3, 8))
    assert is_dyadic(Fraction(-1, 2))
    assert not is_dyadic(Fraction(1, 3))


def test_cantor_digits_and_cantor_function() -> None:
    # 2/3 has ternary expansion 0.2 and lies in Cantor set.
    q = Fraction(2, 3)
    digs = ternary_digits_power3(q, digits=8)
    assert digs.digits[0] == 2
    assert digs.is_cantor()

    # Cantor function maps 0.2_3 -> 0.1_2 = 1/2.
    y = cantor_function_on_cantor(q, digits=16)
    assert y == Fraction(1, 2)
