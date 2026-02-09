# Glossary

Terminology used throughout the open_gororoba codebase, organized by domain.

---

## Cayley-Dickson Algebra Dimensions

| Dim | Name | Symbol | Properties lost at this level |
|-----|------|--------|-------------------------------|
| 1 | Reals | R | (none -- ordered, commutative, associative, normed, division) |
| 2 | Complex numbers | C | Ordering |
| 4 | Quaternions | H | Commutativity |
| 8 | Octonions | O | Associativity (retains alternativity) |
| 16 | Sedenions | S | Norm composition (zero divisors appear) |
| 32 | Pathions | P | Power-associativity degrades further |
| 64 | Chingons | -- | -- |
| 128 | Routons | -- | -- |
| 256 | (unnamed) | -- | -- |

Names from dimensions 32+ are due to Robert de Marrais (arXiv:math/0207003).
"Voudouion" appears in some informal sources for dim=256 but has no
first-party citation.

---

## Zero-Divisor Geometry (dim >= 16)

**Assessor:** A pair of basis indices (i, j) with i < j that forms a 2-blade
element `e_i + e_j` or `e_i - e_j`.  In the sedenion algebra, each assessor
has an S (sail) index and P (prow) index.

**Cross-assessor pair:** Two assessors whose product under CD multiplication
is zero.  The 42 sedenion cross-assessor pairs are the primitive zero
divisors at dim=16.

**ZD graph (complement graph):** Graph where nodes are cross-assessor pairs
and edges connect pairs whose product is zero.

**XOR filter:** A GF(2)-linear necessary condition for zero products.
Basis index pairs must satisfy specific XOR relationships to be candidates.
Necessary but not sufficient (ratio = 0.533 at dim=16).

---

## Box-Kite Anatomy (de Marrais, dim=16)

**Box-kite:** One of 7 connected components of the sedenion ZD graph,
each containing 6 assessors arranged as 3 strut pairs.  The internal
graph of each box-kite is K(2,2,2) -- an octahedron.

**Strut:** A pair of assessors within a box-kite that are diametrically
opposite.  Each box-kite has exactly 3 struts.

**Strut constant:** The integer S that parameterizes a box-kite.
For sedenions, S ranges over {1, 2, ..., 7}.

**Sail:** A triangular face of the box-kite octahedron.  Each box-kite
has 8 sails (4 "zigzag" + 4 "trefoil" by de Marrais's classification).

**Zigzag:** One of 4 sails in a box-kite that forms a zigzag pattern
through the strut structure.

**Trefoil:** One of 4 sails in a box-kite that forms a trefoil (clover)
pattern.  Three distinct trefoil types: ADE, BDF, CEG.

**DMZ (De-Militarized Zone):** The set of edges in a box-kite where
sign concordance holds.  12 DMZ edges per box-kite.

---

## Emanation Architecture (de Marrais, dim >= 32)

**Emanation Table (ET):** A multiplication-like table with
(g-2) x (g-2) entries (g = dim/2), parameterized by strut constant S,
encoding how box-kites interconnect across the next doubling.
For pathions: 14 x 14 tables.

**Trip:** An ordered triple of assessor indices related by the Fano
plane incidence structure.  Types: S-trip (uses S indices), L-trip
(uses L indices), U-trip (mixed).

**Trip Sync:** The universal property that oriented trip structure is
consistent across all box-kites within a given dimension.

**Sail Loop:** The pattern of Fano-plane incidences among the sails of
a box-kite.  Verified to match PSL(2,7) structure.

**Skybox:** The recursion structure of emanation tables across successive
doublings (Theorem 11 in de Marrais).

---

## Lattice Codebook (dim >= 256)

**Lambda_n:** The subset of the E8 lattice used as a codebook at
dimension n.  Points are 8-dimensional vectors in {-1, 0, 1}^8 with
even coordinate sum and even weight.

**Encoding dictionary (Phi_n):** A bijection from CD basis elements
to lattice vectors: `Phi_n: {e_0, ..., e_{n-1}} -> Lambda_n`.

**Prefix-cut:** A generative operation on Lambda_n that produces
Lambda_{n/2} by restricting to vectors whose first coordinate satisfies
a predicate.  Enables recursive trie construction.

**Filtration:** The nested chain
`Lambda_256 subset Lambda_512 subset Lambda_1024 subset Lambda_2048`
verified computationally (C-442).

**Scalar shadow:** The action of CD sign structure on lattice vectors:
each CD basis element contributes a factor in {-1, 0, 1} to the
lattice embedding.

---

## Motif Types (dim >= 32)

**Heptacross:** The complete 7-partite graph K_{2,2,2,2,2,2,2}, appearing
as a motif type in the dim=32 ZD complement graph.

**Mixed-degree component:** A ZD graph component with degree sequence
[4^12, 12^2], the other motif type at dim=32.

**Motif class:** An isomorphism class of connected components in the ZD
complement graph.  At dim=32: 2 classes (8 heptacross + 7 mixed-degree).
At dim=64: 4 classes.  At dim=128: 8 classes.  Doubles each doubling.

**The Pathion Cubic Anomaly (I-012):** The dim=32 motif partition
requires a degree-3 GF(2) polynomial for separation in PG(3,2).
Degrees 1 and 2 are insufficient.

---

## Ultrametric Analysis

**Ultrametric inequality:** d(x,z) <= max(d(x,y), d(y,z)).  Stronger
than the triangle inequality.

**Fraction test:** The proportion of distance triples (x,y,z) satisfying
the ultrametric inequality.  A value of 1.0 means perfectly ultrametric.

**Defect test:** The average amount by which the ultrametric inequality
is violated.  Zero for perfectly ultrametric data.

**Cophenetic correlation:** Correlation between original distances and
dendrogram-derived distances.  Measures how well a hierarchical
clustering preserves pairwise distances.

**BH-FDR:** Benjamini-Hochberg False Discovery Rate correction.
Applied to permutation p-values across multiple tests at alpha=0.05.

---

## GR and Astrophysics

**Boyer-Lindquist coordinates:** (t, r, theta, phi) -- the standard
coordinates for Kerr black holes.

**ISCO:** Innermost Stable Circular Orbit.  At 6M for Schwarzschild,
depends on spin for Kerr.

**Novikov-Thorne disk:** Standard thin accretion disk model with
known temperature, flux, and spectrum profiles.

**Spectral bands:** EHT (230 GHz), ALMA (345 GHz), V-band (550 nm),
Chandra (1 keV).

---

## Dynkin Diagram Conventions

Two node numbering conventions for E8 coexist in this codebase:

- **Convention A** (`kac_moody.rs`): branch at node 4 (0-indexed),
  optimized for E9/E10 affine extension.
- **Convention B** (`e8_lattice.rs`): Bourbaki numbering, branch at
  node 2 (0-indexed).

See [MATH_CONVENTIONS.md, Convention 10](../../MATH_CONVENTIONS.md) for
diagrams and the mapping between conventions.
