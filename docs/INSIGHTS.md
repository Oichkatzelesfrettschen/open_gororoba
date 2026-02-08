# Insights

This document records discoveries and interpretations from the open_gororoba
computational census, organized by I-ID. Each entry separates verified
algebraic facts from speculative physical interpretation per the verification
ladder in CLAUDE.md.

---

## I-012: The Pathion Cubic Anomaly

**Date:** 2026-02-07
**Status:** Verified algebraic phenomenon. Physical interpretation: speculative.
**Claims:** C-443 (verified), C-444 (verified), C-445 (refuted), C-448 (verified)

### Finding

At dimension 32 (pathions), the zero-divisor motif graph decomposes into 15
connected components mapping bijectively to PG(3,2). These components split
into two topological classes:

- **8 heptacross** (K_{2,2,2,2,2,2,2}): 14 nodes, 84 edges, degree-12 regular
- **7 mixed-degree**: 14 nodes, 36 edges, degree sequence [4^12, 12^2]

The minimum GF(2) polynomial degree that separates this 8/7 partition is **3
(cubic)**. This was established by exhaustive Boolean function search:

| Degree | Monomial count | Search space | Result |
|--------|---------------|--------------|--------|
| 1 (linear) | 5 | 2^5 = 32 | No separator |
| 2 (quadratic) | 11 | 2^11 = 2048 | No separator |
| 3 (cubic) | 15 | 2^15 = 32768 | **Separator found** |

The separator is a cubic GF(2) polynomial on the 4-bit PG(3,2) labels.

### Algebraic significance (verified)

- The 8/7 split is NOT a hyperplane (linear subspace) of PG(3,2). This refutes
  the naive AG(3,2) / hyperplane-at-infinity interpretation (C-445 refuted).
- The split IS a cubic hypersurface. Point 8 (binary 1000) has bit 3 set but
  belongs to the heptacross class (along with points 1-7), breaking any
  linear or quadratic classifier.
- At dim=16 (sedenions), the 7 box-kite components are structurally uniform
  (all octahedral K_{2,2,2}), so no class separation question arises.
  The cubic obstruction is specific to the first post-sedenion doubling.

### Open questions

1. What is the minimum separating degree at dim=64 (4 motif classes, PG(4,2))?
   Does the degree grow with the doubling level?
2. Is the cubic polynomial unique (up to GF(2) equivalence), or are there
   multiple independent cubics that separate the classes?
3. Does the cubic structure persist at dim=128 and dim=256 for the analogous
   binary partition (most-edges class vs rest)?

### Physical interpretation (speculative)

The zero-divisor geometry could be mapped to a holographic coding basis where
the PG(n-2,2) labels encode vacuum sector indices. Under this (unverified)
mapping, the cubic obstruction would mean that post-sedenion vacuum sectors
cannot be distinguished by linear probes -- a non-linear measurement is
required. However, NO direct connection to Hamiltonian dynamics, GR, or any
physical observable has been established. This interpretation remains at the
"speculative" tier of the verification ladder.

### Verification

- Test: `test_determine_exact_degree_dim32` in `projective_geometry.rs`
- Ancillary: `test_boolean_predicate_dim32_motif_classes` (degree 3 or 4)
- Data: `motif_components_for_cross_assessors(32)` in `boxkites.rs`

---

## I-011: GPU Ultrametric Sweep (9 catalogs)

**Date:** 2026-02-06
**Status:** Verified (supersedes I-008)
**Claims:** C-071, C-436..C-440, C-442

10M triples x 1000 permutations x RTX 4070 Ti via cudarc 0.19.1.
82/472 tests significant at BH-FDR < 0.05 across 7/9 catalogs.

Key result: the dominant ultrametric signal is galactic kinematics (Hipparcos
48/114, Gaia 12/114), NOT radio-transient ISM as previously concluded (I-008).
The old conclusion was an artifact of 5K subsampling that destroyed spatial
hierarchy in large catalogs.

See `data/csv/c071g_exploration_gpu_10M_1000perm.csv` for full results.

---

## I-006: Motif Census Scaling Laws (dim=16..256)

**Date:** 2026-02-06
**Status:** Verified (exact enumeration)
**Claims:** C-100..C-110, C-443

Scaling laws verified across 5 doublings (dim=16, 32, 64, 128, 256):

- n_components = dim/2 - 1
- nodes_per_component = dim/2 - 2
- n_motif_classes = dim/16 (doubles each time)
- n_K2_components = 3 + log2(dim) (+1 per doubling)
- NO octahedra beyond dim=16

All computed exactly (no sampling). dim=256 completes in ~2s release mode.
