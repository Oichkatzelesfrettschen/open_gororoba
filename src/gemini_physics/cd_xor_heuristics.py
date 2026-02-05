from __future__ import annotations


def xor_key(i: int, j: int) -> int:
    """
    Return the XOR-bucket key commonly used in Cayley-Dickson basis indexing.

    For many standard Cayley-Dickson basis conventions, products of basis units satisfy
    e_i * e_j = +/- e_{i xor j}; the sign depends on the twist convention, but the index
    target is XOR.
    """

    return i ^ j


def xor_bucket_necessary_for_two_blade_zero_product(
    i: int, j: int, k: int, ell: int
) -> bool:
    """
    Necessary (not sufficient) combinatorial condition for a 2-blade vs 2-blade product
    to have any chance of canceling to 0:

        (i xor j) == (k xor l)

    This encodes the "XOR-bucket" heuristic used throughout the convos.
    """

    return xor_key(i, j) == xor_key(k, ell)


def xor_balanced_four_tuple(a: int, b: int, c: int, d: int) -> bool:
    """
    XOR-balanced 4-tuple condition:

        a xor b xor c xor d == 0

    This is a common necessary constraint for 4-term cancellation patterns in XOR-indexed algebras.
    """

    return (a ^ b ^ c ^ d) == 0


def xor_pairing_buckets_for_balanced_four_tuple(a: int, b: int, c: int, d: int) -> set[int]:
    """
    For a XOR-balanced 4-tuple, there are 3 distinct perfect matchings into two pairs:

      (a,b)-(c,d), (a,c)-(b,d), (a,d)-(b,c)

    Each matching corresponds to a shared XOR bucket value:
      a^b == c^d
      a^c == b^d
      a^d == b^c

    This helper returns the 3 bucket values (as a set).
    """

    if not xor_balanced_four_tuple(a, b, c, d):
        raise ValueError("expected a XOR-balanced 4-tuple (a^b^c^d == 0)")

    return {a ^ b, a ^ c, a ^ d}


def xor_bucket_necessary_for_two_blade_vs_balanced_four_blade(
    i: int, j: int, a: int, b: int, c: int, d: int
) -> bool:
    """
    Necessary (not sufficient) XOR constraint for a 2-blade (i,j) to be compatible with a
    XOR-balanced 4-blade (a,b,c,d):

    The 2-blade bucket (i^j) must match one of the 3 pairing buckets induced by the 4-tuple.

    This is intentionally a cheap filter, not a guarantee of a zero product.
    """

    if not xor_balanced_four_tuple(a, b, c, d):
        return False
    return (i ^ j) in xor_pairing_buckets_for_balanced_four_tuple(a, b, c, d)
