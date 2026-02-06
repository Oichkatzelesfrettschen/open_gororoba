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
