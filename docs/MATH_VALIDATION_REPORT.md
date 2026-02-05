# Mathematical Validation Report

**Date:** 2026-01-27

This report summarizes which *mathematical* statements in this repository are:
- supported by reproducible computations/tests in-repo, and
- consistent with well-known published results (with citations).

It also highlights gaps where the repo's narrative exceeds what has been validated.

## 1) Scope

This document covers the "hard math" layer:
- Cayley-Dickson construction properties (associativity, norm composition, zero divisors)
- a minimal group-theory fact used in docs (`|PSL(2,7)| = 168`)
- dimensional-geometry formulas implemented via analytic continuation (Gamma function)

It does **not** claim physical validation (astrophysics/cosmology/metamaterials), which requires
separate empirical and modeling work.

## 2) Cayley-Dickson algebra: what is validated here

Key references:
- Baez (2002), *The Octonions*: https://arxiv.org/abs/math/0105155
- de Marrais (c. 2002), *The 42 Assessors and the Box-Kites they fly*: https://arxiv.org/abs/math/0011260

### 2.1 Non-associativity appears at 8D (octonions)

**Published result:** octonions are non-associative (but alternative). See Baez (2002).  
**Repo validation:** `tests/test_cayley_dickson_properties.py` checks:
- associativity holds for dimensions 1,2,4 (reals/complex/quaternions)
- a counterexample is found in 8D via seeded random search

### 2.2 Norm composition holds through 8D and fails at 16D

**Published result:** Hurwitz-style classification implies multiplicative norms only for
dimensions 1,2,4,8. See Baez (2002).  
**Repo validation:** `tests/test_cayley_dickson_properties.py` checks:
- `||ab||^2 = ||a||^2 ||b||^2` for many random pairs in 8D
- the identity fails for some pairs in 16D (sedenions)

### 2.3 Explicit sedenion zero divisors exist

**Published result:** sedenions have zero divisors; de Marrais develops explicit "assessor/box-kite"
structure for primitive zero divisors (de Marrais, c. 2002; see `docs/BIBLIOGRAPHY.md`).  
**Repo validation:** `tests/test_cayley_dickson_properties.py` includes an explicit identity under
the repo's basis convention:

`(e1 + e10) * (e4 - e15) = 0`

This is computed with `gemini_physics.optimized_algebra.cd_multiply_jit` using the same
canonical Cayley-Dickson multiplication convention (e.g. `e1*e2=e3`).

### 2.4 de Marrais "42 assessors / 7 box-kites" structure (replicated)

**Published result (de Marrais, abstract):** 168 primitive unit zero divisors are arranged as quartets
along 42 assessors; assessors organize into seven "box-kites" (octahedral vertex structures). See
arXiv:math/0011260.

**Repo validation:**
- `src/gemini_physics/de_marrais_boxkites.py` computes primitive assessors and box-kites from the
  repo's sedenion multiplication convention.
- `tests/test_de_marrais_boxkites.py` verifies:
  - 42 primitive assessors (subset of the 56 cross-pairs),
  - `42 * 4 = 168` unit points on assessor diagonals,
  - 7 connected components, each the octahedron graph (6 vertices, 12 edges, degree 4),
  - every edge corresponds to at least one diagonal sign pairing with product exactly 0.

## 3) Group-theory fact used in the narrative

### 3.1 Order of PSL(2,7)

**Published/standard result:** `|PSL(2,q)| = q(q^2-1)/gcd(2,q-1)` for q a prime power; for `q=7`
this gives 168.

**Repo validation:** `tests/test_group_theory.py` checks `order_psl2_q(7) == 168` via
`gemini_physics.group_theory.order_psl2_q`.

**Important caveat:** This does **not** validate the *mapping* from `PSL(2,7)` to box-kite geometry;
it only validates the group order arithmetic.

## 4) Dimensional continuation (analytic continuation)

**Published result:** sphere/ball volumes and related formulas extend to non-integer dimension via
Gamma functions, with poles at non-positive even integers, etc. (Standard special-function facts.)

**Repo validation:** `tests/test_dimensional_geometry.py` verifies integer-dimension identities
and consistency checks for `src/gemini_physics/dimensional_geometry.py`.

## 5) What remains unvalidated (math-facing)

High-impact items still requiring explicit replication from primary sources:
- De Marrais "42 assessors / 7 box-kites / PSL(2,7) action" reproduction as code + tests
- Reggiani (2024) definitions of `ZD(S)`/geometry reproduced precisely and aligned to repo terms

See `docs/NEXT_ACTIONS.md`.
