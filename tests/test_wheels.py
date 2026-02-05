from __future__ import annotations

from fractions import Fraction

from gemini_physics.wheels import WheelQ, assert_carlstrom_wheel_axioms


def test_wheelq_satisfies_carlstrom_axioms_on_small_set() -> None:
    xs = [
        WheelQ.zero(),
        WheelQ.one(),
        WheelQ.of(2, 1),
        WheelQ.of(-1, 1),
        WheelQ.of(1, 2),
        WheelQ.inf(),
        WheelQ.nan(),
    ]

    assert_carlstrom_wheel_axioms(xs)


def test_wheelq_zero_over_zero_is_additive_absorber() -> None:
    zero = WheelQ.zero()
    nan = zero.div(zero)
    assert nan == WheelQ.nan()

    for q in [Fraction(0), Fraction(1), Fraction(-2, 3)]:
        x = WheelQ.of(q, 1)
        assert x.add(nan) == nan
