# Insights Registry Mirror

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: TOML registry files under registry/ -->
<!-- Generated at: 2026-02-09T08:38:02Z -->

Authoritative source: `registry/insights.toml`.

Total insights: 16

## I-001: Macquart Relation Fills the Comoving Distance Gap

- Date: 2026-02-06
- Status: verified
- Sprint: 6
- Claims: C-071

The Macquart relation connects FRB dispersion measures to redshift via integrated baryon density: DM_cosmic(z) = 935 * integral (1+z')/E(z') dz'. Bisection inversion (DM->z) converges in ~27 iterations. Foundation for comoving distance in ultrametric analysis.

## I-002: Ultrametric Structure Lives in Representations, Not Scalars

- Date: 2026-02-06
- Status: verified
- Sprint: 6
- Claims: C-071

C-071 (FRB DMs exhibit p-adic ultrametric structure) definitively refuted using raw DM values. Ultrametricity is a property of hierarchical organization, not scalar distributions. This motivated five new analysis directions testing multi-attribute encodings, temporal cascades, and transformed coordinate spaces.

## I-003: Existing Rust Crate Ecosystem for Cosmological Analysis

- Date: 2026-02-06
- Status: verified
- Sprint: 6
- Claims: (none)

Identified key crates preventing reimplementation: kodama 0.3.0 (dendrograms), kiddo 5.2.4 (k-d trees, AVX2), fitsrs 0.4.1 (FITS), rustfft 6.4.1 + realfft 3.5.0 (FFT), votable 0.7.0, satkit 0.9.3. Notable gaps requiring custom implementation: cophenetic correlation, Baire metric, local ultrametricity (Bradley 2025), KDE.

## I-004: Kodama Dendrogram and Real Observational Cosmology Infrastructure

- Date: 2026-02-06
- Status: verified
- Sprint: 6
- Claims: C-200, C-201, C-202, C-203, C-204, C-205, C-206, C-207, C-208, C-209, C-210

Kodama returns Dendrogram with Step{cluster1, cluster2, dissimilarity, size}; cophenetic distance c(i,j) = dissimilarity at first merge, enabling cophenetic correlation. Also: first real-data joint fit of Lambda-CDM vs bounce cosmology using 1578 Pantheon+ SNe + 7 DESI DR1 BAO bins. Delta BIC = +7.37 favoring Lambda-CDM. Critical data quality fix: BGS/QSO bins are isotropic-only (not anisotropic).

## I-005: Ultrametric Structure is Radio-Transient-Specific (Preliminary)

- Date: 2026-02-06
- Status: superseded
- Sprint: 6
- Claims: C-437, C-442

Initial 7-catalog survey with 5K subsampling found only FRB/pulsar catalogs showing significant ultrametric excess. SUPERSEDED by I-011 (GPU 10M-triple sweep): the 5K subsampling destroyed Hipparcos galactic signal, making the conclusion too narrow. The ISM-mediation hypothesis for radio transients remains valid.

## I-006: Motif Census Scaling Laws (dim=16..256)

- Date: 2026-02-06
- Status: verified
- Sprint: 6
- Claims: C-126, C-127, C-128, C-129, C-130

Exact scaling laws for Cayley-Dickson zero-divisor box-kite structure across 5 doublings: n_components = dim/2 - 1, nodes_per_component = dim/2 - 2, n_motif_classes = dim/16, n_K2_components = 3 + log2(dim), K2 part count = dim/4 - 1. NO octahedra beyond dim=16.

## I-007: Kerr Geodesic Integrator Verification Summary

- Date: 2026-02-06
- Status: verified
- Sprint: 6
- Claims: C-028

u=1/r regularized Kerr geodesic integrator (Dopri5, Mino time) passes: potential non-negativity, circular photon orbit at 3M, near-horizon infall, a=0.998 stability, r=500 large-distance, shadow area pi*27, asymmetry at a=0.9, coordinate/Mino time monotonicity. Hamiltonian constraint inaccessible from dense output.

## I-008: Cross-Domain Ultrametric Analysis (5K Subsampling)

- Date: 2026-02-06
- Status: superseded
- Sprint: 7
- Claims: C-437

9-catalog ultrametric fraction test with 5K subsampling: only CHIME/FRB and ATNF pulsars pass. Hipparcos at null baseline (p=0.438). SUPERSEDED by I-011: GPU sweep with 10M triples shows Hipparcos 48/114 significant at BH-FDR<0.05. The 5K subsampling destroyed the galactic spatial hierarchy signal.

## I-009: Elliptic Integral Crate Eliminates Carlson Port

- Date: 2026-02-06
- Status: verified
- Sprint: 6
- Claims: (none)

The ellip crate (1.0.4, BSD-3-Clause) provides all 5 Carlson symmetric forms (RF, RD, RJ, RC, RG) plus Legendre complete/incomplete integrals (K, E, Pi, D, F). Eliminates need to hand-port Carlson from C++ Blackhole codebase. Tested against Boost Math and Wolfram reference values.

## I-010: nalgebra 0.33/0.34 Version Split Blocks Autodiff

- Date: 2026-02-06
- Status: open
- Sprint: 6
- Claims: (none)

num-dual 0.13.2 (autodiff via dual numbers) requires nalgebra 0.34, while workspace is pinned to 0.33. Decision: defer num-dual, use closed-form Christoffels for known metrics (Schwarzschild, Kerr, Kerr-Newman). num-dual needed only for generic connection computation on arbitrary metrics.

## I-011: GPU Ultrametric Sweep (9 catalogs)

- Date: 2026-02-07
- Status: verified
- Sprint: 7
- Claims: C-436, C-437, C-438, C-439, C-440

10M triples x 1000 permutations x RTX 4070 Ti via cudarc 0.19. 82/472 tests significant at BH-FDR<0.05 across 7/9 catalogs. Hipparcos: 48/114 (galactic hierarchy), CHIME/FRB: 8/8 (ALL subsets), GWOSC GW: 4/66 (chirp_mass+q). Supersedes I-008 (5K subsampling bias).

## I-012: The Pathion Cubic Anomaly

- Date: 2026-02-07
- Status: verified
- Sprint: 8
- Claims: C-445, C-446, C-447, C-448

dim=32 (pathion) 8/7 motif split requires degree-3 GF(2) polynomial, NOT degree 4. Degrees 1,2 fail. This proves the polynomial controlling motif-class partition is genuinely cubic.

## I-013: The Hierarchy Fingerprint Theorem

- Date: 2026-02-07
- Status: verified
- Sprint: 7
- Claims: C-441, C-442, C-443, C-444, C-449

Ultrametric fraction test is a genuine hierarchy fingerprint: catalogs with known hierarchical structure (Hipparcos proper motions, CHIME DM) show strong signal, while isotropic catalogs (Fermi GBM GRBs) show no signal. The test discriminates physical hierarchy from noise.

## I-014: Cayley-Dickson External Data Cross-Validation

- Date: 2026-02-07
- Status: cross-validation-complete
- Sprint: 8
- Claims: C-450, C-451, C-452, C-453, C-454, C-455, C-456, C-457

Cross-validated 68 external files against Rust integer-exact computations. Strut table: VERIFIED (C-454). E8 connection: REFUTED (C-455). 8D lattice embedding: VERIFIED at 256/512/1024/2048 (C-452, C-453).

## I-015: Monograph Theses Verification -- Lattice Codebook Filtration

- Date: 2026-02-08
- Status: verified
- Sprint: 9
- Claims: C-458, C-459, C-460, C-461, C-462, C-463, C-464, C-465, C-466, C-467

8 monograph theses (A-H) verified: parity constraints, nesting, prefix-cut transitions, scalar shadow, XOR partner, parity-clique, spectral fingerprints, null-model. S_base = 2187 = 3^7.

## I-016: De Marrais Emanation Architecture

- Date: 2026-02-08
- Status: verified
- Sprint: 10
- Claims: C-468, C-469, C-470, C-471, C-472, C-473, C-474, C-475

Implemented L1-L18 emanation layers from de Marrais construction. DMZ = sign-concordance (12 edges/BK), sail-loop = Fano incidence. Oriented Trip Sync universal across all 7 BKs. 113 tests, 4400 lines.
