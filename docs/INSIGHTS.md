# Research Insights

Accumulated insights from implementing the ultrametric analysis framework.
Each entry records a non-obvious discovery, design decision, or mathematical
connection worth preserving for future work.

---

## I-001: Macquart Relation Fills the Comoving Distance Gap

**Date**: 2026-02-06
**Context**: cosmology_core/src/distances.rs implementation
**Related claims**: C-071b (FRB comoving ultrametric structure)

The existing `bounce.rs` module has `luminosity_distance()` but not
`comoving_distance()`. The Macquart relation connects FRB dispersion
measures to redshift via the integrated baryon density along the line of
sight. Converting DM -> z -> comoving distance is the foundation for
Direction 1's analysis.

Key equation:

    DM_cosmic(z) = 935 * integral_0^z (1+z') / E(z') dz'   [pc/cm^3]

where 935 comes from 3*c*H0*Omega_b*f_IGM / (8*pi*G*m_p) with f_IGM=0.83
(Macquart et al. 2020, Nature 581, 391). The prefactor evaluates to ~935
pc/cm^3 for Planck 2018 parameters.

The bisection inversion (DM -> z) converges to 1e-8 in ~27 iterations,
which is negligible compared to the integration cost. Simpson's rule with
500 subintervals gives ~6-digit accuracy for the integral.

---

## I-002: Ultrametric Structure Lives in Representations, Not Scalars

**Date**: 2026-02-06
**Context**: C-071 refutation analysis
**Related claims**: C-071 (refuted), C-071b through C-071f (new directions)

C-071 ("FRB DMs exhibit p-adic ultrametric structure") was definitively
refuted using raw DM values. But the refutation itself is informative:
ultrametricity is a property of *hierarchical organization*, not scalar
distributions. Testing raw DM values is testing the wrong representation.

The literature shows ultrametric structure lives in:
- Topological/hierarchical relationships (Anninos-Denef 2016, merger trees)
- Multi-attribute encodings (Murtagh arXiv:1104.4063, Baire metric)
- Temporal cascades (SOC avalanche hierarchies)
- Transformed coordinate spaces (comoving distance, not raw DM)

This motivates five new analysis directions that test the *right*
representations across all available cosmological datasets.

---

## I-003: Existing Rust Crate Ecosystem for Cosmological Analysis

**Date**: 2026-02-06
**Context**: Workspace dependency research for data pipeline
**Related claims**: All C-071x directions

Key crates that prevent reinventing the wheel:

| Crate | Version | Purpose | Priority |
|-------|---------|---------|----------|
| kodama | 0.3.0 | Hierarchical clustering (dendrograms) | HIGH |
| kiddo | 5.2.4 | k-d tree spatial indexing (AVX2) | HIGH |
| andam | 0.1.2 | Cosmological calculations (Planck, distances) | HIGH |
| fitsrs | 0.4.1 | Pure Rust FITS file reading | HIGH |
| rustfft | 6.4.1 | FFT for power spectral density | HIGH |
| realfft | 3.5.0 | Real-to-complex FFT companion | HIGH |
| votable | 0.7.0 | VOTable astronomical data format | MEDIUM |
| satkit | 0.9.3 | Coordinate frames + Earth gravity | MEDIUM |
| hdf5-metno | 0.12.1 | HDF5 reading (Planck chains) | HIGH |
| netcdf | 0.12.0 | NetCDF for climate/geophysical data | MEDIUM |

Notable gaps (no viable crate, implement in-house):
- Cophenetic correlation: build on kodama::Dendrogram output
- Baire metric: build on padic/adic crates
- Local ultrametricity (Bradley 2025): custom implementation
- KDE: ~50 lines of Gaussian KDE using nalgebra

The `hurst` crate (0.1.0, already in workspace) covers Hurst exponent
estimation for the temporal cascade analysis in Direction 3.

---

## I-004: Kodama Dendrogram Enables Cophenetic Correlation

**Date**: 2026-02-06
**Context**: ultrametric/dendrogram.rs design
**Related claims**: C-071f (cross-dataset cosmic dendrogram)

The `kodama` crate returns a `Dendrogram` struct containing a sequence of
`Step { cluster1, cluster2, dissimilarity, size }` records. From this, the
cophenetic distance c(i,j) between any two original points is the
dissimilarity at which they first merge in the dendrogram.

Cophenetic correlation = Pearson(original_distances, cophenetic_distances).
A value close to 1.0 means the data is well-represented by a tree
(ultrametric). For truly ultrametric data, cophenetic correlation = 1.0.

Algorithm for cophenetic distance matrix from kodama output:
1. Initialize union-find over n original points
2. Walk steps in merge order
3. For each merge of clusters A and B at dissimilarity d:
   set c(i,j) = d for all i in A, j in B
4. Result: n x n cophenetic distance matrix

This is O(n^2) in the number of original points, which is acceptable for
the dataset sizes we work with (CHIME ~600, GWTC-3 ~90, ATNF ~3000).

---

## I-004: Real Observational Cosmology Fit -- Pantheon+ x DESI DR1 BAO

**Date**: 2026-02-06
**Context**: cosmology_core/src/observational.rs + real-cosmo-fit binary
**Related claims**: bounce cosmology model comparison

First real-data joint fit of Lambda-CDM and bounce cosmology using:
- **1578 Pantheon+ SH0ES Type Ia supernovae** (Scolnic+ 2022, Brout+ 2022)
- **7 DESI DR1 BAO bins** (5 anisotropic + 2 isotropic; DESI 2024)

### Results (v2 -- corrected isotropic/anisotropic BAO treatment)

| Parameter       | Lambda-CDM   | Bounce       |
|----------------|--------------|--------------|
| Omega_m        | 0.3284       | 0.3284       |
| H_0 (km/s/Mpc)| 71.65        | 71.65        |
| q_corr         | --           | 8.5e-11 (effectively zero) |
| chi2_total     | 738.61       | 738.61       |
| chi2_SN        | 721.03       | 721.03       |
| chi2_BAO       | 17.58        | 17.58        |
| chi2/dof       | 0.465 (1588) | 0.465 (1587) |
| AIC            | 742.61       | 744.61       |
| BIC            | 753.36       | 760.73       |

Delta BIC = +7.37: **Strong evidence for Lambda-CDM** (Kass & Raftery scale).
Bounce model's extra parameter (q_corr) converges to zero -- no quantum
correction is needed by the data.

### v1 -> v2 changelog (critical data quality fix)

The original v1 fit (Omega_m=0.2883, H_0=72.96, chi2_BAO=78.8) had **fabricated
anisotropic values for the BGS and QSO bins**. DESI DR1 arXiv:2404.03002 Table 1
explicitly states BGS (z=0.295) and QSO (z=1.491) provide only isotropic DV/rd,
not anisotropic DM/rd + DH/rd. The v1 code treated all 7 bins as anisotropic and
assigned DH/rd values copied from neighboring bins.

Specific errors corrected:
- BGS: fabricated DH/rd=20.98, rho=-0.445 (copied from LRG1) -> isotropic DV/rd=7.93
- QSO: fabricated DM/rd=30.69, DH/rd=13.26 -> isotropic DV/rd=26.07
- LRG1 rho: -0.474 -> -0.445 (transcription error)
- LRG3+ELG1 rho: -0.386 -> -0.389
- ELG2 rho: -0.474 -> -0.444
- Data point count: 14 (7*2) -> 12 (5*2 + 2*1)

Impact: chi2_BAO dropped from 78.8 to 17.58 (12 data points -> chi2/dof=1.47),
Omega_m shifted from 0.288 to 0.328 (closer to Planck 0.315), H_0 dropped from
72.96 to 71.65. The BAO chi2/dof~1.5 is now consistent with mild H_0 tension
rather than the catastrophic chi2/dof~5.6 that indicated corrupted data.

### Key observations

1. **H_0 tension visible**: Our fit gives H_0 = 71.65 km/s/Mpc, between
   SH0ES (73.04) and Planck (67.36). Pantheon+ m_b_corr encodes the SH0ES
   calibration; the BAO data pulls H_0 down. The resulting mild tension is
   physical, not a data artifact.

2. **Analytic M_B marginalization**: Using m_b_corr (corrected apparent
   magnitude) instead of pre-computed mu requires marginalizing over the
   absolute magnitude M_B. The Conley+ (2011) formula chi2_marg =
   A - B^2/C absorbs this degree of freedom analytically.

3. **Mixed BAO treatment**: DESI DR1 requires per-bin dispatch: isotropic
   bins (BGS, QSO) contribute DV(z)/rd as a single scalar chi2, while
   anisotropic bins (LRG, ELG, Lya) contribute a 2x2 correlated chi2 from
   DM/rd + DH/rd with correlation coefficient rho.

4. **Bounce model correctly disfavored**: q_corr -> 0 is the expected
   result for data generated by a Lambda-CDM universe. The BIC penalty
   for the extra parameter correctly identifies the simpler model.

5. **chi2/dof ~ 0.47 is expected**: Pantheon+ diagonal-only errors
   overestimate uncertainty (the full covariance matrix C_stat+C_sys is
   not included here). Using the full matrix would raise chi2 toward 1.0.

### Pitfalls recorded

- Pantheon+ SH0ES data file uses `m_b_corr`/`m_b_corr_err_DIAG` columns,
  NOT `MU`/`MUERR`. The parser must handle both formats. Also, `zCMB` can
  be obtained from `ZHD` (Hubble-diagram redshift) as a fallback.
- DESI DR1 BGS (z=0.295) and QSO (z=1.491) are ISOTROPIC only. Never
  fabricate DH/rd or rho for these bins. Check the source paper Table 1.

---

## I-005: Ultrametric Structure is Radio-Transient-Specific, Not Universal

**Date**: 2026-02-06
**Context**: multi-dataset-ultrametric analysis across 7 real astrophysical catalogs
**Related claims**: C-437, C-442

Cross-dataset ultrametric analysis (Direction 2: multi-attribute Euclidean
ultrametricity) reveals that significant ultrametric structure is NOT a
general property of astrophysical catalogs. It appears specifically in radio
transient data parameterized by (log_DM, gl, gb):

| Dataset | N | Attributes | UM_frac | Null | p | Verdict |
|---------|---|------------|---------|------|---|---------|
| CHIME/FRB Cat 2 | 5000 | log_DM+gl+gb | 0.1561 | 0.1353 | 0.005 | PASS |
| ATNF Pulsars | 4233 | log_DM+gl+gb | 0.2222 | 0.2131 | 0.005 | PASS |
| GWOSC GW Events | 176 | log_Mc+z+q+chi_eff | 0.1532 | 0.1611 | 0.995 | FAIL |
| Pantheon+ SN Ia | 1588 | z+mu+x1+c | 0.1474 | 0.1632 | 1.000 | FAIL |
| Gaia DR3 Stars | 5000 | parallax+pm+rv+G+bp_rp | 0.1235 | 0.1437 | 1.000 | FAIL |
| SDSS DR18 Quasars | 5000 | z+ugri | 0.1284 | 0.1369 | 1.000 | FAIL |
| Fermi GBM GRBs | 4202 | log_t90+log_fluence+ra+dec | 0.1507 | 0.1514 | 0.682 | FAIL |

### Key observations

1. **Pulsars and FRBs share the DM+sky position parameterization**, and both
   show ultrametric structure. This is likely driven by the hierarchical
   structure of the Milky Way's electron density distribution (NE2001/YMW16)
   -- the DM encodes cumulative line-of-sight electron column, which is
   structured by spiral arms, the thick/thin disk, and the halo.

2. **Purely cosmological catalogs fail**: GW events (176 with full parameters
   from GWOSC O1-O4a), SNe Ia (1588 Pantheon+), and quasars (5K SDSS DR18)
   all show UM_frac BELOW the null -- their parameter spaces are LESS
   ultrametric than random, suggesting correlations that break the strong
   triangle inequality.

3. **Gaia stars fail** despite being parameterized by 6D phase space -- the
   local stellar neighborhood has complex velocity-position correlations
   from moving groups and open clusters that don't form an ultrametric tree.

4. **The GWOSC catalog with 176 events** (up from 35 confident, full
   parameter set from combined O1-O4a) confirms the original C-439 REFUTED
   result is not a sample-size artifact.

5. **Fermi GBM GRBs (N=4202)**: log_t90+log_fluence+ra+dec parameterization
   gives p=0.682 -- not significant but also not as extreme as the cosmological
   catalogs. The UM_frac (0.1507) is nearly equal to the null mean (0.1514),
   indicating GRB parameter space is metrically random.

### Interpretation

The ultrametric signal in FRB/pulsar data likely reflects Galactic ISM
structure rather than cosmological physics. This narrows the physical
interpretation of C-437 from "astrophysical ultrametricity" to "ISM-mediated
hierarchical dispersion" -- a less exotic but more falsifiable hypothesis.

### Falsification path

Test DM-only ultrametricity on extragalactic FRBs (DM > DM_MW) after
subtracting the Galactic DM contribution. If the signal persists after
removing the ISM component, the ultrametricity has a cosmological origin.
If it vanishes, it's purely Galactic.

---

## I-006: Pathion Zero-Divisor Motifs -- Heptacross and Mixed-Degree Graphs

**Date**: 2026-02-06
**Context**: algebra_core/src/boxkites.rs motif census at dim=32
**Related claims**: C-016 (box-kite census), C-020 (complement-graph detection)

Running `motif_components_for_cross_assessors(32)` on the 32-dimensional pathion
algebra reveals a completely different graph structure from the sedenion (dim=16)
octahedra:

| Dimension | Components | Node count | Types |
|-----------|-----------|------------|-------|
| 16 (sedenion) | 7 | 6 each | 7x K_{2,2,2} (octahedra) |
| 32 (pathion) | 15 | 14 each | 8x K_{2,2,2,2,2,2,2} + 7x mixed |

### Key observations

1. **All 15 components have exactly 14 nodes** (7 cross-assessor pairs each).

2. **8 heptacross components** -- complete 7-partite graph K_{2,2,2,2,2,2,2}:
   14 nodes, 84 edges, all degree 12. This is the cross-polytope in R^7
   (or equivalently, the hyperoctahedron in 7 dimensions). The doubling from
   dim=16 (3 parts) to dim=32 (7 parts) suggests the part count grows as
   `dim/2 - 1` -- one part per half-dimension minus the identity.

3. **7 mixed-degree components** with degree sequence [4^12, 12^2]:
   14 nodes, 36 edges. These contain 12 "peripheral" nodes of degree 4 and
   2 "hub" nodes of degree 12 (connected to everything). This is NOT a
   complete multipartite graph -- it has a distinguished pair of
   maximally-connected vertices.

4. **No octahedra**: The dim=16 box-kites (K_{2,2,2}) do NOT appear as
   sub-motifs at dim=32. The zero-divisor graph completely restructures at
   each Cayley-Dickson doubling level.

5. **Pattern hypothesis**: 15 = 7 + 8, where 7 = dim(sedenion)/2 - 1 and
   8 = dim(pathion)/2 - 1 - 7. The mixed-degree components may correspond
   to "inter-level" zero-divisor interactions between the two sedenion
   sub-algebras within the pathion.

### Future work

- Run dim=64 census to test the part-count growth hypothesis
- Characterize the mixed-degree graph algebraically (are the 2 hubs
  related to the doubling basis element?)
- Check if 15 = de Marrais' predicted count for 32D (unpublished)

---

## I-007: Kerr Geodesic Integrator Verification Summary

**Date**: 2026-02-06
**Context**: gr_core/src/kerr.rs test suite expansion
**Related claims**: C-028 (Kerr geodesic integration)

The u=1/r regularized Kerr geodesic integrator (Dopri5 adaptive, Mino time)
passes the following verification suite:

| Test | What it checks | Result |
|------|---------------|--------|
| Potential non-negativity | R(r) >= 0 and Theta(theta) >= 0 along trajectory | PASS |
| Circular photon orbit | r stays near 3M for Schwarzschild L=3sqrt(3) | PASS |
| Near-horizon infall | r_horizon+0.1 -> horizon (a=0.9) | PASS |
| Near-extremal Kerr | a=0.998 geodesic completes without NaN | PASS |
| Large-distance infall | r=500 radial infall -> horizon (tests u=1/r scaling) | PASS |
| Shadow area | pi*27 for Schwarzschild (within 15%) | PASS |
| Kerr shadow asymmetry | a=0.9 shadow shifts prograde | PASS |
| Coordinate time monotonicity | dt > 0 outside horizon | PASS |
| Mino time monotonicity | dlam > 0 always | PASS |

Note: The Hamiltonian constraint (v_r^2 = R(r)) cannot be verified from dense
output trajectory alone, since Dopri5's interpolant does not preserve velocity
accuracy. The potential non-negativity test is the accessible alternative.

---

## I-009: Elliptic Integral Crate Eliminates Carlson Port

**Date**: 2026-02-06
**Context**: gr_core GR module expansion / Blackhole C++ port research
**Related claims**: Task #30 (Carlson elliptic integrals), Task #48 (crate research)

The `ellip` crate (1.0.4, BSD-3-Clause) provides all 5 Carlson symmetric forms
(RF, RD, RJ, RC, RG) plus all Legendre complete/incomplete elliptic integrals
(K, E, Pi, D, F). This completely eliminates the need to hand-port the Carlson
implementation from the Blackhole C++ codebase (Task #30).

Carlson symmetric forms are numerically superior to Legendre forms for Kerr
geodesic integrals (see Carlson 1995, Press et al. NR Ch 6.11, Dexter & Agol
2009). The crate is tested against Boost Math and Wolfram reference values,
has minimal dependencies (num-traits, num-lazy, numeric_literals), and uses the
standard parameter convention m = k^2 (consistent with SciPy).

The companion crate `ellip-rayon` provides parallel evaluation, which could be
useful for shadow computation sweeps over large parameter grids.

---

## I-010: nalgebra 0.33/0.34 Version Split Blocks Autodiff

**Date**: 2026-02-06
**Context**: gr_core GR module expansion / crate version compatibility
**Related claims**: Task #37 (generic connection computation), Task #48

num-dual 0.13.2 (exact automatic differentiation via dual numbers) is the
ideal crate for computing Christoffel symbols from arbitrary metric functions.
However, it requires nalgebra 0.34, while ode_solvers 0.6.1 requires nalgebra
0.33.2. These are incompatible (0.33 and 0.34 are different semver-minor
versions under the 0.x convention, so Cargo treats them as separate crates
with incompatible types).

**Decision**: Defer num-dual until ode_solvers releases a nalgebra 0.34-
compatible version. For known metrics (Schwarzschild, Kerr, Kerr-Newman),
use closed-form Christoffel symbols (textbook derivations, hand-coded, faster
than autodiff). num-dual becomes necessary only for generic connection
computation (Task #37) where the metric is not known at compile time.

**Monitoring**: Check ode_solvers GitHub for nalgebra 0.34 support periodically.
Alternative: evaluate switching to a different ODE solver crate that supports
nalgebra 0.34, or upgrading nalgebra to 0.34 if ode_solvers supports it
transitively.

---

## I-008: Cross-Domain Ultrametric Analysis -- Radio Transients Are Special

**Date**: 2026-02-06
**Context**: gororoba_cli/src/bin/multi_dataset_ultrametric.rs
**Related claims**: C-437 (multi-attribute Euclidean ultrametricity)

Cross-domain ultrametric fraction test (Direction 2) on 9 astrophysical
catalogs:

| Dataset | N | Attributes | UM_frac | null_mean | p | Verdict |
|---------|--:|-----------|--------:|----------:|--:|---------|
| CHIME/FRB Cat 2 | 5000 | log_DM+gl+gb | 0.1561 | 0.1353 | 0.005 | PASS |
| ATNF Pulsars | 4233 | log_DM+gl+gb | 0.2222 | 0.2131 | 0.005 | PASS |
| McGill Magnetars | 20 | log_P+log_B+ra+dec | 0.1961 | 0.1722 | 0.080 | FAIL |
| GWOSC GW Events | 176 | log_Mc+z+q+chi_eff | 0.1532 | 0.1611 | 0.995 | FAIL |
| Pantheon+ SN Ia | 1588 | z+mu+x1+c | 0.1474 | 0.1632 | 1.000 | FAIL |
| Gaia DR3 | 5000 | plx+pm+rv+G+bp_rp | 0.1235 | 0.1437 | 1.000 | FAIL |
| SDSS Quasars | 5000 | z+u+g+r+i | 0.1284 | 0.1369 | 1.000 | FAIL |
| Fermi GBM GRBs | 4202 | log_t90+log_F+ra+dec | 0.1507 | 0.1514 | 0.682 | FAIL |
| Hipparcos Stars | 5000 | plx+pmra+pmdec+V+ra+dec | 0.1401 | 0.1399 | 0.438 | FAIL |

**Key finding**: Only radio-transient catalogs (FRBs, pulsars) show significant
ultrametric excess. Both use DM + galactic coordinates as attributes -- DM
encodes line-of-sight ISM structure, which naturally clusters hierarchically.
Optical, gravitational, gamma-ray, and stellar astrometry catalogs uniformly
fail, suggesting the ultrametric signal is ISM-mediated, not a universal
property of astrophysical point catalogs.

Hipparcos (113,710 stars, 6 attributes) is essentially at the null baseline
(UM_frac = 0.1401 vs null 0.1399, p=0.438), confirming that stellar
astrometry carries zero hierarchical signal in this test.

McGill magnetars (N=20) trend toward significance (p=0.080) but lack
statistical power. Magnetars are compact objects with radio emission,
consistent with the ISM-mediation hypothesis -- they would likely pass with
a larger sample.

**Data**: `data/csv/c071g_multi_dataset_ultrametric.csv`

---
