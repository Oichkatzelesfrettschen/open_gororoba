from __future__ import annotations

import math


def order_psl2_q(q: int) -> int:
    """
    Return the order of PSL(2,q) for an integer q > 1.

    For q a prime power, PSL(2,q) is a finite group with a standard closed-form order:

        |PSL(2,q)| = q (q^2 - 1) / gcd(2, q - 1).

    This function uses that formula directly; it does not validate that q is a prime power.
    """
    if q <= 1:
        raise ValueError("q must be > 1")
    g = math.gcd(2, q - 1)
    return (q * (q * q - 1)) // g
