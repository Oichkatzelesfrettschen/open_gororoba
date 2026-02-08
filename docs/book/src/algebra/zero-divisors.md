# Zero-Divisor Census

At dimension 16 (sedenions) and beyond, Cayley-Dickson algebras contain
zero divisors: nonzero elements whose product is zero.  The zero-divisor
census enumerates and classifies these elements.

## Key results

The sedenion algebra has **42 primitive cross-assessor pairs** that produce
zero when multiplied.  These are pairs of 2-blade elements (sums of two
basis vectors with specific signs) whose product vanishes under the
Cayley-Dickson multiplication rule.

The census is computed using integer-exact arithmetic to avoid
floating-point error entirely.  The implementation lives in
`crates/algebra_core/src/boxkites.rs`.

## The XOR filter

A necessary (but not sufficient) condition for a zero-product pair is
the XOR filter: basis indices must satisfy GF(2)-linear conditions.

- 315 pairs pass the XOR filter
- 168 pairs are actual zero products
- Ratio: 0.533 (XOR is necessary, NOT sufficient)
- Zero false negatives confirmed (C-445)

See also: [Box-Kite Structure](./box-kites.md) for how zero divisors
organize into combinatorial structures.

## Claims

C-050 through C-060: Zero-divisor graph invariants (E-005).
C-445 through C-448: XOR balanced search extension (I-012).
