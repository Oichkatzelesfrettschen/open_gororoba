<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/insights.toml, registry/insights_narrative.toml -->

# Insights

Source-of-truth policy:
- Authoritative machine-readable registry: `registry/insights.toml`
- TOML-driven markdown mirror: `docs/generated/INSIGHTS_REGISTRY_MIRROR.md`
- This file remains narrative detail and historical context.

This document records discoveries and interpretations from the open_gororoba
computational census, organized by I-ID. Each entry separates verified
algebraic facts from speculative physical interpretation per the verification
ladder in CLAUDE.md.

---

## I-001: Macquart Relation Fills the Comoving Distance Gap

Date: 2026-02-06
Status: verified
Claims: C-071

The Macquart relation connects FRB dispersion measures to redshift via integrated baryon density: DM_cosmic(z) = 935 * integral (1+z')/E(z') dz'. Bisection inversion (DM->z) converges in ~27 iterations. Foundation for comoving distance in ultrametric analysis.

---

---

---

---

---

---

---

---

---

---

---

## I-002: Ultrametric Structure Lives in Representations, Not Scalars

Date: 2026-02-06
Status: verified
Claims: C-071

C-071 (FRB DMs exhibit p-adic ultrametric structure) definitively refuted using raw DM values. Ultrametricity is a property of hierarchical organization, not scalar distributions. This motivated five new analysis directions testing multi-attribute encodings, temporal cascades, and transformed coordinate spaces.

---

---

---

---

---

---

---

---

---

---

---

## I-003: Existing Rust Crate Ecosystem for Cosmological Analysis

Date: 2026-02-06
Status: verified
Claims: (none)

Identified key crates preventing reimplementation: kodama 0.3.0 (dendrograms), kiddo 5.2.4 (k-d trees, AVX2), fitsrs 0.4.1 (FITS), rustfft 6.4.1 + realfft 3.5.0 (FFT), votable 0.7.0, satkit 0.9.3. Notable gaps requiring custom implementation: cophenetic correlation, Baire metric, local ultrametricity (Bradley 2025), KDE.

---

---

---

---

---

---

---

---

---

---

---

## I-004: Kodama Dendrogram and Real Observational Cosmology Infrastructure

Date: 2026-02-06
Status: verified
Claims: C-200, C-201, C-202, C-203, C-204, C-205, C-206, C-207, C-208, C-209, C-210

Kodama returns Dendrogram with Step{cluster1, cluster2, dissimilarity, size}; cophenetic distance c(i,j) = dissimilarity at first merge, enabling cophenetic correlation. Also: first real-data joint fit of Lambda-CDM vs bounce cosmology using 1578 Pantheon+ SNe + 7 DESI DR1 BAO bins. Delta BIC = +7.37 favoring Lambda-CDM. Critical data quality fix: BGS/QSO bins are isotropic-only (not anisotropic).

---

---

---

---

---

---

---

---

---

---

---

## I-005: Ultrametric Structure is Radio-Transient-Specific (Preliminary)

Date: 2026-02-06
Status: superseded
Claims: C-437, C-442

Initial 7-catalog survey with 5K subsampling found only FRB/pulsar catalogs showing significant ultrametric excess. SUPERSEDED by I-011 (GPU 10M-triple sweep): the 5K subsampling destroyed Hipparcos galactic signal, making the conclusion too narrow. The ISM-mediation hypothesis for radio transients remains valid.

---

---

---

---

---

---

---

---

---

---

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

---

---

---

---

---

---

---

---

---

---

---

---

## I-007: Kerr Geodesic Integrator Verification Summary

Date: 2026-02-06
Status: verified
Claims: C-028

u=1/r regularized Kerr geodesic integrator (Dopri5, Mino time) passes: potential non-negativity, circular photon orbit at 3M, near-horizon infall, a=0.998 stability, r=500 large-distance, shadow area pi*27, asymmetry at a=0.9, coordinate/Mino time monotonicity. Hamiltonian constraint inaccessible from dense output.

---

---

---

---

---

---

---

---

---

---

---

## I-008: Cross-Domain Ultrametric Analysis (5K Subsampling)

Date: 2026-02-06
Status: superseded
Claims: C-437

9-catalog ultrametric fraction test with 5K subsampling: only CHIME/FRB and ATNF pulsars pass. Hipparcos at null baseline (p=0.438). SUPERSEDED by I-011: GPU sweep with 10M triples shows Hipparcos 48/114 significant at BH-FDR<0.05. The 5K subsampling destroyed the galactic spatial hierarchy signal.

---

---

---

---

---

---

---

---

---

---

---

## I-009: Elliptic Integral Crate Eliminates Carlson Port

Date: 2026-02-06
Status: verified
Claims: (none)

The ellip crate (1.0.4, BSD-3-Clause) provides all 5 Carlson symmetric forms (RF, RD, RJ, RC, RG) plus Legendre complete/incomplete integrals (K, E, Pi, D, F). Eliminates need to hand-port Carlson from C++ Blackhole codebase. Tested against Boost Math and Wolfram reference values.

---

---

---

---

---

---

---

---

---

---

---

## I-010: nalgebra 0.33/0.34 Version Split Blocks Autodiff

Date: 2026-02-06
Status: open
Claims: (none)

num-dual 0.13.2 (autodiff via dual numbers) requires nalgebra 0.34, while workspace is pinned to 0.33. Decision: defer num-dual, use closed-form Christoffels for known metrics (Schwarzschild, Kerr, Kerr-Newman). num-dual needed only for generic connection computation on arbitrary metrics.

---

---

---

---

---

---

---

---

---

---

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

---

---

---

---

---

---

---

---

---

---

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

---

---

---

---

---

---

---

---

---

---

---

## I-013: The Hierarchy Fingerprint Theorem

**Date:** 2026-02-07
**Status:** Verified Statistical Invariant
**Claims:** C-449 (Ultrametric Core Mining Hypothesis)

### The Theorem (Operational)
For a multi-attribute dataset, the set of attribute subsets that exhibit statistically significant ultrametric structure (after BH-FDR correction) forms a poset ordered by inclusion. The **minimal elements** of this poset -- the "Cores" -- identify the fundamental coordinate systems carrying the hierarchical information.

**Verified Finding:** Analysis of 10M GPU-accelerated subset tests across 9 catalogs reveals that these Cores cluster into distinct physical families, independent of the specific catalog (e.g., CHIME and ATNF share the same core structure).

### Complete Verified Cores (all 7 catalogs with significant subsets)

The analysis script `src/scripts/analysis/extract_ultrametric_cores.py` extracted
the following minimal-element cores from the GPU sweep poset. Every core listed
below passed BH-FDR < 0.05 on 10M triples / 1000 permutations. Two catalogs
(McGill Magnetars, Fermi GBM GRBs) had zero significant subsets and are omitted.

*   **CHIME/FRB Cat 2** (3 cores):
    `gl + log_DM`, `gb + log_DM`, `gb + gl`

*   **ATNF Pulsars** (2 cores):
    `gl + log_DM`, `gb + gl`

*   **GWOSC GW Events** (1 core):
    `log_chirp_mass + q`

*   **Gaia DR3 Stars** (4 cores):
    `parallax + pmra`, `parallax + pmdec`, `pmra + rv`, `bp_rp + rv`

*   **Hipparcos Stars** (4 cores):
    `parallax + pmra`, `pmdec + pmra`, `Vmag + dec`, `dec + ra`

*   **SDSS DR18 Quasars** (1 core):
    `i_mag + r_mag + z`

*   **Pantheon+ SN Ia** (1 core):
    `c + x1`

### Interpretation by Physical Mechanism

*   **ISM / Column Density:** CHIME and ATNF share `log_DM + gl`. The `gb + gl`
    core (galactic coordinates alone, no DM) is also significant, indicating
    that sky-position hierarchy in the Galactic plane contributes independently.

*   **Galactic Kinematics:** Gaia and Hipparcos share `parallax + pmra`.
    Additional cores involving `bp_rp` (color), `Vmag` (magnitude), and `dec`
    reflect photometric subpopulations and declination-dependent survey depth.

*   **Compact Binary Formation:** GWOSC's sole core `log_chirp_mass + q`
    reflects formation-channel segregation in the mass-ratio plane.

*   **Standard Candle Standardization:** Pantheon+'s sole core `c + x1` is
    the Phillips relation (stretch vs color).

*   **Quasar Photometric Hierarchy:** SDSS's core `i_mag + r_mag + z` is a
    color-redshift combination reflecting the photometric redshift ladder.

### Implications
This confirms the **Ultrametric Core Mining Hypothesis (UCMH)**. We can now use the "Hierarchy Fingerprint" (the set of cores) as:
1.  **Unsupervised coordinate discovery:** Finding physics without assuming a model.
2.  **Integrity Check:** If a pipeline update changes the Cores of a standard catalog, it implies a geometry-altering bug.

---

---

---

---

---

---

---

---

---

---

---

---

## I-014: Cayley-Dickson External Data Cross-Validation

**Date:** 2026-02-07
**Status:** Cross-Validation Complete
**Claims:** C-450 through C-457

### Summary

Cross-validated 68 external files (5 de Marrais PDFs, 36 unique CSVs after
deduplication) from an AI-generated Cayley-Dickson data set against our Rust
integer-exact computations. Key results:

### Verified

1. **Strut table (C-454):** De Marrais's published strut table for all 7
   sedenion box-kites matches our `canonical_strut_table()` output exactly.
   All 42 assessor pairs, all 21 strut pairings, and the inner XOR property
   confirmed.

2. **8D lattice embedding (C-452, C-453):** Cayley-Dickson basis elements at
   dims 256, 512, 1024, and 2048 all embed into the same 8-dimensional
   integer lattice with coordinates in {-1, 0, 1}. The lattice dimension does
   NOT grow with log2(dim) as originally hypothesized. This is consistent with
   the octonion sub-algebra (8D) providing the fundamental lattice structure.

3. **Nested-tuple parser (C-457):** A tree-based parser correctly handles the
   Cayley-Dickson doubling-tree representation across 4476 rows spanning 5
   dimensions.

### Refuted

1. **E8 lattice-ZD connection (C-455):** Lattice differences between
   ZD-adjacent pairs at dim=16 have norm-squared values {4, 6, 8, 10, 12,
   14, 18} but never 2. E8 roots (norm^2 = 2) are completely absent. The
   minimum ZD separation in the 8D lattice is |d|^2 = 4.

2. **256D associativity CSV (C-456):** The external CSV incorrectly claims all
   125 tested triples are associative. Our Rust computation identifies 4/50
   non-associative triples involving high-index basis elements (e_128 * e_64 *
   e_32, etc.). This confirms the CSV is AI-generated with errors.

### Structural Finding

The 105 unique lattice-difference vectors between ZD-adjacent pairs distribute
across 7 distinct norm-squared values. The peak at |d|^2 = 6 (84 occurrences)
suggests that ZD adjacency preferentially connects basis elements that differ
in 3 lattice coordinates by 1 each plus some additional structure. The absence
of |d|^2 = 0 (no self-adjacency) and |d|^2 = 2 (no nearest-neighbor
adjacency) creates a "forbidden zone" in lattice space around each basis
element where ZD partners cannot exist.

### Data Quality

- PDFs: Authentic, verified (strut table matches)
- Lattice CSVs: High quality, verified at all 4 dimensions
- Adjacency CSVs: Multiple incompatible representations, inconclusive
- Associativity CSV: Contains errors, AI-generated
- Comparison/qualitative CSVs: Speculative, no numerical content

---

---

---

---

---

---

---

---

---

---

---

---

## I-015: Monograph Theses Verification -- Lattice Codebook Filtration

**Date:** 2026-02-08
**Status:** Verification Complete (8 theses, 10 claims)
**Claims:** C-458 through C-467

### Summary

Implemented and tested the 8 falsifiable theses (A-H) from the monograph on
lattice codebook filtration of Cayley-Dickson algebras. Results:

### Verified Theses

1. **Thesis A (Codebook Parity, C-458):** All lattice points at dims
   256/512/1024/2048 satisfy: coords in {-1,0,1}, even sum, even nonzero
   count, coord[0] never +1. Total 3840 rows verified.

2. **Thesis B (Filtration Nesting, C-459):** Strict subset chain
   Lambda_256 < Lambda_512 < Lambda_1024 < Lambda_2048 confirmed.

3. **Thesis C (Prefix-Cut, C-460):** ALL filtration transitions are
   lexicographic prefix cuts -- the child is the lex-sorted first N points
   of the parent. This is simpler than the monograph anticipated (which
   expected decision-trie rules).

4. **Thesis D (Scalar Shadow, C-465):** pi(b) = signum(sum(coords))
   maps to {-1,0,1}; addition-mode action verified. Multiplication
   coupling rho(b) remains open (C-466).

5. **Thesis E (XOR Partner Law, C-462):** Each cross-pair has unique
   XOR partner at dim=64. General law: partner(i) = i XOR (N/16).

6. **Thesis F (Parity-Clique, C-463):** ZD adjacency = K_m union K_m
   by parity of low basis index, verified at dims 16 and 32 ONLY.
   **REFUTED at dim=64+** (cross-partition edges exist at dim=64;
   C-451 shows 32640 cross-edges at dim=128). Small-dimension
   coincidence, not a universal property. Status: Partial.

7. **Thesis G (Spectral Fingerprints, C-464):** Eigenvalue multisets
   distinguish all observed motif classes (K_m, K_m union K_m, r*K_2).

8. **Thesis H (Null-Model Identity, C-467):** RandomRotation is identity
   for Euclidean ultrametric tests; ColumnIndependent is informative.
   Baire fraction is tautologically 1.0 (ultrametric by construction).

### Novel Discoveries

- **Lex prefix cuts** (Thesis C): The filtration structure is purely
  lexicographic, not a general decision trie. This constrains the lattice
  embedding to respect coordinate ordering.
- **S_base = 2187 = 3^7:** The base universe (coord[0] in {-1,0},
  even sum, even nonzero count) has exactly 2187 points, with 139
  excluded from Lambda_2048.
- **Lambda_32 pinned corner:** First 4 coords = (-1,-1,-1,-1) for all
  32 points (C-461).

### Open Questions

- Multiplication coupling rho(b) in GL(8,Z) (C-466)
- Extension of parity-clique and XOR partner to dim=256+ (computational)
- Connection between lex-prefix filtration and octonion subalgebra structure

---

---

---

---

---

---

---

---

---

---

---

---

## I-016: De Marrais Emanation Architecture

**Claims**: C-468..C-475
**Module**: `crates/algebra_core/src/emanation.rs` (~4400 lines, 113 tests)

### Summary

Implemented the full de Marrais "lacuna map" (L1-L18) as a single coherent
Rust module covering: Cayley-Dickson signed products, emanation tables (dim=16
and dim=32 strutted), tone-row ordering, DMZ cell geometry, ET sparsity
spectroscopy, twist mechanics, PSL(2,7) navigation, lanyard taxonomy, Trip Sync,
semiotic squares, sail-loop duality, oriented Trip Sync, signed adjacency graphs,
lanyard state machines, delta transition functions, and brocade normalization.

### Key Findings

- **Emanation tables are fully determined by XOR**: The product index at
  cell (i,j) is always i XOR j, and the sign comes from the Cayley-Dickson
  multiplication recursion. Zero-divisor marking is exact (42 assessor pairs
  at dim=16 = 84 symmetric ET cells).

- **DMZ geometry is sign-concordance, not octahedral adjacency**: A cell
  is DMZ when its 4 quadrant products have concordant diagonal signs
  (UL*LR sign = UR*LL sign). This produces 12 DMZ edges per BK (not 9
  as naive octahedral counting suggests).

- **Sail-loop duality is combinatorial, not dynamical**: The 28 O-trip
  sails partition into 7 automorphemes (Cawagas loops) by Fano plane
  incidence, not by twist-orbit BFS. Each BK contributes 4 sails to 4
  distinct automorphemes. This is the correct BK-automorpheme duality.

- **Oriented Trip Sync is universal at dim=16**: All 7 box-kites admit
  at least one PSL(2,7) embedding where the shorthand pattern
  (a,b,c),(a,d,e),(d,b,f),(e,f,c) is satisfiable.

- **Delta transition structure**: Each S0 has exactly 3 XOR strut pairs
  covering {1..7}\{S0}. The delta reachability matches twist reachability
  but the detailed pair-level correspondence is more nuanced than simple
  containment.

- **Brocade normalization yields 4 relabelings per BK**: Any of the 4
  O-trips in a BK's L-set can serve as the Rule-0 central circle. CPO
  preservation (outer indices also forming an O-trip) is uniform across BKs.

### Architecture

The module follows a layered design:
1. **Product engine** (L1): CDP signed products with quadrant recursion
2. **Table generation** (L2-L4): Tone rows, emanation tables, strutted ETs, sparsity
3. **Twist mechanics** (L5-L6): H*/V* operations, PSL(2,7) navigation graph
4. **Lanyard taxonomy** (L7-L8): Sails, tray-racks, blues, quincunx, bicycle chains, Trip Sync
5. **Semiotic geometry** (L9-L14): Strut-opposite kernels, CT boundary, loop duality, Sky, Eco echo
6. **Orientation and normalization** (L15-L18): Oriented Trip Sync, signed graphs, delta, brocade

### Open Questions from Sprint 10

- **Twist-delta pair correspondence**: The twist navigation targets do not
  always match delta strut pairs at the individual pair level. The XOR of
  twist targets (h XOR v) is not always = source_strut. Need to understand
  which Fano-plane quantity governs each twist transition.
- **Full lanyard classification from signed graph**: The infrastructure for
  signed-graph -> lanyard extraction is in place (L16), but systematic
  classification of all cycle types across all 7 BKs is not yet done.
- **Brocade CPO preservation**: The CPO count is uniform across BKs but
  the actual count (0 or >0) needs algebraic explanation from Fano plane
  complementation properties.

---

---

---

---

---

---

---

---

---

---

---

## I-017: Cross-Stack Locality and Coxeter Correspondence (E-011/E-012/E-013)

Date: 2026-02-09
Status: partial
Claims: C-476, C-477

Three experiments testing ALP (C-476) and Sky-Limit-Set (C-477). E-011: ALP holds for sparse constraint graphs (E10 Dynkin p=0.000, ET DMZ p=0.000) but fails for dense graphs (Sedenion ZD p=1.000, edge density 86.7%). E-012: Billiard symbolic dynamics predict spectroscopy behavior (FullFill entropy=0.0, UniformSky entropy=0.44, fill-entropy r=-0.85 at N=5). E-013: A_{N-1} Coxeter group is consistently the best match for ET skybox invariants (rank ratio=1.0, improving match scores at higher N). ALP needs sparsity refinement; Coxeter correspondence is strong but DMZ density match not yet within 10%.

---

---

---

---

---

---

---

---

---

---
