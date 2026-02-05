# Terminology Glossary (Standardized)

This repo mixes:
- standard mathematics/physics terms, and
- nonstandard, author-coined terms from `convos/` and related notes.

This glossary defines the *repo meaning* of key terms and (when applicable) provides the closest
standard term(s) so that docs, code, and tests use consistent language.

## Conventions

- "CD" = Cayley-Dickson construction.
- Dimension is usually `D = 2^n` with basis elements `e_0, e_1, ..., e_{D-1}`.
- `e_0` is the scalar unit.
- When we say "exact" we mean "integer/rational exact" in the implementation sense (no floating
  point), not a formal proof.

## Cayley-Dickson algebra (computational)

- **CD multiplication convention**: the specific basis ordering and sign rule implemented in
  `gemini_physics.optimized_algebra.cd_multiply_jit`. Other references may use different sign
  conventions; results are only comparable after aligning conventions.
- **2-blade**: in this repo, a sparse vector of the form `(e_i +/- e_j)` with `i != j`, typically
  with coefficients in `{+1,-1}`.
- **4-blade**: a sparse vector supported on 4 basis elements (usually with coefficients in
  `{+1,-1}`).
- **Zero divisor (ZD)**: a nonzero element `u` such that there exists nonzero `v` with `u*v = 0`
  (left zero divisor) or `v*u = 0` (right zero divisor).
- **Left annihilator**: `Ann_L(u) = { v : v*u = 0 }`.
- **Right annihilator**: `Ann_R(u) = { v : u*v = 0 }`.

## de Marrais (nonstandard terminology; used as labels)

These terms originate in R. de Marrais' "box-kite" narrative. They are used here as *labels* for
specific combinatorial objects that we can export and test.

- **Assessor**: in this repo, a primitive cross-assessor is an ordered pair `(low, high)` of basis
  indices used to define a 2-blade diagonal. The generator is
  `gemini_physics.de_marrais_boxkites.primitive_assessors`.
- **Box-kite**: a 6-vertex combinatorial structure among assessors/diagonals derived from de
  Marrais' construction, exported by `src/export_de_marrais_boxkites.py`.
- **Strut table**: the de Marrais "strut" bookkeeping exported as CSV by
  `src/export_de_marrais_boxkites.py`.

## Reggiani (standard paper terms; aligned in-repo)

- **Z(S)**: in Reggiani (2024), a set of normalized pairs `(u,v)` with `u*v = 0`. The exact
  normalization details are paper-specific.
- **ZD(S)**: in Reggiani (2024), a normalized subset of sedenions used for geometric analysis.
  In this repo, `is_reggiani_zd` implements our *computational proxy* for ZD(S):
  squared Euclidean norm 2 and nontrivial annihilator detected numerically.
- **Standard zero divisors (84)**: in this repo, the 84 diagonals `(e_low +/- e_high)` over the 42
  primitive assessors, implemented in `gemini_physics.reggiani_replication.standard_zero_divisors`.

## A-infinity / homological algebra (standard terms; carefully scoped)

- **A-infinity algebra**: standard term (Stasheff identities). The repo does NOT claim an
  A-infinity structure is established unless we also provide a precise dg source + Stasheff-identity
  checks.
- **m3**: in this repo, a specific trilinear operation computed in
  `gemini_physics.m3_cd_transfer`. It matches a common HPL-style expression but should be treated as
  "a trilinear operator with measured properties" unless and until A-infinity identities are
  verified.

## Fractional operators (standard terms; domain-sensitive)

- **Riesz fractional Laplacian**: Fourier multiplier `|xi|^alpha` on `R^d` (or on a torus with
  periodic boundary conditions).
- **Spectral fractional Laplacian**: defined via eigenfunctions of `-Delta` on a bounded domain.
- **Caffarelli-Silvestre extension**: realizes the fractional Laplacian as a Dirichlet-to-Neumann
  map for an extension PDE in one higher dimension.

The repo must explicitly state which operator and which domain/boundary conditions are used in each
script/test.

## p-adics / Cantor sets (standard terms)

- **p-adic valuation v_p(n)**: exponent of `p` in `n` (for integers).
- **p-adic norm |x|_p**: `p^{-v_p(x)}` (with standard extensions to rationals).
- **Dyadic rational**: rational with denominator a power of 2.
- **Cantor set point (ternary)**: number in `[0,1]` with ternary expansion using only digits 0 and
  2; used in `gemini_physics.padic` for simple Cantor function experiments.
