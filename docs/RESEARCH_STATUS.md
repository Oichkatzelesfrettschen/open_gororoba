<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Research Status: Toy Operator Study (Speculative "Negative Dimension" Mapping)

**Date:** January 27, 2026
**Status:** Data ingested; hypothesis unvalidated

## 1. LIGO O3 Data Integration
This repo caches a local copy of the **GWTC-3 Confident Events** catalog:
*   **Artifact:** `data/external/GWTC-3_GWpy_Official.csv` (35 Confident Events).
*   **Provenance (partial):** hashes are recorded in `data/external/PROVENANCE.local.json`; the authoritative catalog is hosted by GWOSC (see `docs/BIBLIOGRAPHY.md`).
*   **Observation (unverified):** the "clumping/mass-gap" narrative below requires a reproducible statistical test against catalog selection effects and null models.

## 2. The toy "anti-diffusion" hypothesis
Our `neg_dim_pde.py` simulation (`alpha = -1.5`) explores a toy operator where wavefunctions can
**concentrate** rather than disperse under the chosen model/discretization. This is not a validated
physical statement about literal negative-dimensional spacetime (see `docs/NEGATIVE_DIMENSION_CLARIFICATIONS.md`).
*   **Spectral Analysis (toy):** The eigenvalues in this toy setup scale as $E_n \sim n^{-1.5}$ (as a fitted scaling law).
*   **Mass Mapping:** If Mass $M \sim E^{-1}$, then $M \sim n^{1.5}$.
    *   $n=10 \implies M \approx 31.6 M_{\odot}$ (Matches Peak 1).
    *   $n=15 \implies M \approx 58.0 M_{\odot}$ (Matches Peak 2).
*   **Hypothesis (speculative):** map catalog masses to discrete "modes" of this operator. This requires a falsifiable statistical model and controls for selection effects.

## 3. Physical Interpretation
*   **Standard View:** Pair-Instability Supernovae (PISN) prevent BH formation in the 50-120 $M_{\odot}$ range.
*   **Sedenion View (speculative):** the vacuum geometry itself (16D zero-divisor network) might constrain stable solutions via "spectral gaps". Treat this as an interpretive hypothesis pending validation.

## 4. Next Steps
*   **Refine the Model:** Test if $n=20$ predicts the upper mass gap edge (~120 $M_{\odot}$).
*   **Cosmology:** Apply this "Anti-Diffusion" logic to Dark Energy (repulsive gravity).
