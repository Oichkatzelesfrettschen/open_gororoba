---
description: Source ledger for the exceptional-cosmology claim cluster C-035 through C-041
last_verified: 2026-06-25
status: active
---

# Exceptional Cosmology Sources

This ledger anchors the exceptional-cosmology claim cluster C-035 through C-041.
The cluster tests whether exceptional Lie algebra structure explains selected
cosmological or string-theory numbers. The result is deliberately mixed:
C-035 is an exact algebraic identity, C-039 is a verified toy qualitative
comparison, and C-036, C-037, C-038, C-040, and C-041 are refuted strong
predictions.

## Registry Surfaces

- `registry/claims.toml` records C-035 through C-041.
- `docs/EXCEPTIONAL_COSMOLOGY.md` summarizes the programme and the status of
  each claim.
- `docs/C037_NUMERICAL_COINCIDENCE_AUDIT.md` records the Barbero-Immirzi,
  F4 Casimir, and Gauss-Bonnet coincidence audit.
- `docs/C041_F4_STRING_DIMENSION_COINCIDENCE_AUDIT.md` records the F4 26D
  representation and bosonic-string critical-dimension audit.

## Formal Surfaces

- `proofs/verified/C035_CasimirQuarter.v`
- `proofs/verified/C036_TrialityClusteringBound.v`
- `proofs/verified/C037_EpsilonNotGamma.v`
- `proofs/verified/C038_WDarkEnergy.v`
- `proofs/verified/C039_SpectralDimensionRunning.v`
- `proofs/verified/C040_PrimordialTiltRefuted.v`

## Empirical And Code Surfaces

- `crates/cosmology_core/src/spectral.rs`
- `data/csv/c039_spectral_dimension_bigraph_curve.csv`
- `data/csv/c039_spectral_dimension_bigraph_summary.csv`
- `data/csv/c040_primordial_tilt_deff_curve.csv`
- `data/csv/c040_primordial_tilt_summary.csv`

## Source Boundary

The source role of exceptional algebras here is structural, not causal. G2, F4,
E6, E7, and E8 appear in well-established mathematical settings, but a
structural appearance does not by itself determine cosmological observables.
The registry therefore treats numerical near-equalities as conjectures that
must survive independent mechanism and observation gates.
