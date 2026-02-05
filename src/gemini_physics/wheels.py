from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class WheelQ:
    """
    A tiny, concrete wheel model over rational numbers.

    This is a minimal "fractions with total reciprocal" model intended to validate Carlstrom's
    wheel axioms (Definition 1.1) with a small, deterministic test set.

    Representation:
    - Finite rationals are normalized to (value, 1).
    - "Infinity" is normalized to (1, 0).
    - "0/0" is normalized to (0, 0) and is absorbing for addition.
    """

    num: Fraction
    den: Fraction

    @staticmethod
    def of(num: Fraction | int, den: Fraction | int) -> WheelQ:
        n = Fraction(num)
        d = Fraction(den)

        if d == 0:
            if n == 0:
                return WheelQ(Fraction(0), Fraction(0))
            return WheelQ(Fraction(1), Fraction(0))

        return WheelQ(n / d, Fraction(1))

    @staticmethod
    def zero() -> WheelQ:
        return WheelQ(Fraction(0), Fraction(1))

    @staticmethod
    def one() -> WheelQ:
        return WheelQ(Fraction(1), Fraction(1))

    @staticmethod
    def inf() -> WheelQ:
        return WheelQ(Fraction(1), Fraction(0))

    @staticmethod
    def nan() -> WheelQ:
        return WheelQ(Fraction(0), Fraction(0))

    def is_nan(self) -> bool:
        return self.num == 0 and self.den == 0

    def is_inf(self) -> bool:
        return self.num == 1 and self.den == 0

    def add(self, other: WheelQ) -> WheelQ:
        # (a/b) + (c/d) = (ad + bc) / (bd)
        return WheelQ.of(self.num * other.den + self.den * other.num, self.den * other.den)

    def mul(self, other: WheelQ) -> WheelQ:
        # (a/b) * (c/d) = (ac) / (bd)
        return WheelQ.of(self.num * other.num, self.den * other.den)

    def inv(self) -> WheelQ:
        # /(a/b) = b/a (totalized: /(0/0) = 0/0, /(a/0) = 0/a -> 0)
        return WheelQ.of(self.den, self.num)

    def div(self, other: WheelQ) -> WheelQ:
        # x/y := x * /y
        return self.mul(other.inv())


def assert_carlstrom_wheel_axioms(xs: list[WheelQ]) -> None:
    """
    Assert Carlstrom's Definition 1.1 axioms on a finite set xs.

    This is intended for unit tests (it raises AssertionError on failure).
    """

    zero = WheelQ.zero()
    one = WheelQ.one()

    # 1) <H, 0, +> is a commutative monoid
    for x in xs:
        assert x.add(zero) == x
        assert zero.add(x) == x
    for x in xs:
        for y in xs:
            assert x.add(y) == y.add(x)
            for z in xs:
                assert x.add(y.add(z)) == x.add(y).add(z)

    # 2) <H, 1, *, /> is a commutative monoid with involution /
    for x in xs:
        assert x.mul(one) == x
        assert one.mul(x) == x
        assert x.inv().inv() == x
    for x in xs:
        for y in xs:
            assert x.mul(y) == y.mul(x)
            assert x.mul(y).inv() == y.inv().mul(x.inv())
            for z in xs:
                assert x.mul(y.mul(z)) == x.mul(y).mul(z)

    # 3) (x + y)*z + 0*z = x*z + y*z
    for x in xs:
        for y in xs:
            for z in xs:
                left = x.add(y).mul(z).add(zero.mul(z))
                right = x.mul(z).add(y.mul(z))
                assert left == right

    # 4) x/y + z + 0*y = (x + y*z)/y
    for x in xs:
        for y in xs:
            for z in xs:
                left = x.div(y).add(z).add(zero.mul(y))
                right = x.add(y.mul(z)).div(y)
                assert left == right

    # 5) 0*0 = 0
    assert zero.mul(zero) == zero

    # 6) (x + 0*y)*z = x*z + 0*y
    for x in xs:
        for y in xs:
            for z in xs:
                left = x.add(zero.mul(y)).mul(z)
                right = x.mul(z).add(zero.mul(y))
                assert left == right

    # 7) /(x + 0*y) = /x + 0*y
    for x in xs:
        for y in xs:
            left = x.add(zero.mul(y)).inv()
            right = x.inv().add(zero.mul(y))
            assert left == right

    # 8) x + 0/0 = 0/0
    zero_over_zero = zero.div(zero)
    for x in xs:
        assert x.add(zero_over_zero) == zero_over_zero
