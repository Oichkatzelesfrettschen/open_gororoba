from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


def vp_int(n: int, p: int) -> int:
    """
    p-adic valuation v_p(n) for a nonzero integer n.

    Returns the largest k >= 0 such that p^k divides n.
    """

    if p <= 1:
        raise ValueError("p must be an integer >= 2")
    if n == 0:
        raise ValueError("v_p(0) is undefined/infinite; handle as a special case")

    n_abs = abs(n)
    k = 0
    while n_abs % p == 0:
        n_abs //= p
        k += 1
    return k


def vp(q: Fraction, p: int) -> int:
    """
    p-adic valuation v_p(q) for a nonzero rational q = a/b in lowest terms:
        v_p(q) = v_p(a) - v_p(b)
    """

    if p <= 1:
        raise ValueError("p must be an integer >= 2")
    if q == 0:
        raise ValueError("v_p(0) is undefined/infinite; handle as a special case")

    return vp_int(q.numerator, p) - vp_int(q.denominator, p)


def abs_p(q: Fraction, p: int) -> float:
    """
    p-adic absolute value |q|_p.

    By definition:
      |0|_p = 0
      |q|_p = p^{-v_p(q)} for q != 0
    """

    if q == 0:
        return 0.0
    return float(p ** (-vp(q, p)))


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def is_dyadic(q: Fraction) -> bool:
    """
    Dyadic rational: denominator is a power of 2 in lowest terms.
    """

    return is_power_of_two(q.denominator)


@dataclass(frozen=True, slots=True)
class CantorDigits:
    """
    Exact finite ternary expansion digits for rationals with denominator a power of 3.

    digits are in {0,1,2}. For points in the Cantor set under this representation,
    digits avoid 1.
    """

    digits: tuple[int, ...]

    def is_cantor(self) -> bool:
        return all(d in (0, 2) for d in self.digits)


def ternary_digits_power3(q: Fraction, digits: int) -> CantorDigits:
    """
    Compute the first `digits` base-3 digits of q in [0,1], assuming the denominator is 3^k.

    This is exact for such rationals: no floating point is used.
    """

    if not (0 <= q <= 1):
        raise ValueError("q must be in [0,1]")

    out: list[int] = []
    x = q
    for _ in range(digits):
        x *= 3
        d = int(x)  # exact because x is Fraction
        if d not in (0, 1, 2, 3):
            raise AssertionError("unexpected digit")
        if d == 3:
            d = 2
        out.append(d)
        x -= d
    return CantorDigits(digits=tuple(out))


def cantor_function_on_cantor(q: Fraction, digits: int = 40) -> Fraction:
    """
    Cantor (Devil's staircase) function restricted to points in the Cantor set.

    For a ternary expansion using only digits {0,2}, map:
      0 -> 0 (binary digit)
      2 -> 1 (binary digit)
    """

    digs = ternary_digits_power3(q, digits)
    if not digs.is_cantor():
        raise ValueError("q is not represented as a Cantor-set point in the first digits")

    num = 0
    den = 1
    for d in digs.digits:
        num *= 2
        den *= 2
        if d == 2:
            num += 1
    return Fraction(num, den)

