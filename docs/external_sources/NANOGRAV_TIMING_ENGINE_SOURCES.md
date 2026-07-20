# NANOGrav Timing Engine Sources
# Snapshot date: 2026-03-19

This dossier records the first-party and locally cached sources used for the first
independent TOA-driven timing-engine lane in this repository.

## Local ephemeris assets already present

1. Local DE440 BSP
   - `data/external/de440.bsp`

2. Local DE440 short BSP
   - `data/external/de440s.bsp`

3. Local Earth orientation binary PCK
   - `data/external/nanograv_timing_engine/earth_latest_high_prec.bpc`

4. Local generic planetary constants text PCK
   - `data/external/nanograv_timing_engine/pck00011.tpc`

5. Local DE440 GM text kernel
   - `data/external/nanograv_timing_engine/gm_de440.tpc`

## Cached upstream timing-model implementation surfaces

These files were cached under `data/external/nanograv_timing_engine/` to keep the new
Rust implementation reproducible and inspectable offline.

1. PINT observatory definitions
   - `data/external/nanograv_timing_engine/observatories.json`

2. PINT astrometry model source
   - `data/external/nanograv_timing_engine/astrometry.py`

3. PINT spindown model source
   - `data/external/nanograv_timing_engine/spindown.py`

4. PINT solar-system Shapiro source
   - `data/external/nanograv_timing_engine/solar_system_shapiro.py`

5. PINT ELL1 wrapper and family selection logic
   - `data/external/nanograv_timing_engine/binary_ell1.py`

6. PINT ELL1 forward model
   - `data/external/nanograv_timing_engine/ELL1_model.py`

7. PINT ELL1H forward model
   - `data/external/nanograv_timing_engine/ELL1H_model.py`

8. PINT BT model source
   - `data/external/nanograv_timing_engine/binary_bt.py`

9. PINT DD model source
   - `data/external/nanograv_timing_engine/binary_dd.py`

10. PINT DDK model source
    - `data/external/nanograv_timing_engine/binary_ddk.py`

11. PINT orbit parameterization source
    - `data/external/nanograv_timing_engine/binary_orbits.py`

12. PINT stand-alone BT model
    - `data/external/nanograv_timing_engine/BT_model.py`

13. PINT stand-alone DD model
    - `data/external/nanograv_timing_engine/DD_model.py`

14. PINT stand-alone DDK model
    - `data/external/nanograv_timing_engine/DDK_model.py`

## Repo-local implementation surfaces

1. Typed `.par` layer
   - `crates/gororoba_cli_data/src/nanograv_timing_model.rs`

2. Independent timing engine
   - `crates/gororoba_cli_data/src/nanograv_timing_engine.rs`

3. Independent Phase 1 driver
   - `crates/gororoba_cli_data/src/bin/nanograv_timing_phase1_independent.rs`

4. Independent Phase 1 artifacts
   - `reports/nanograv_phase1_independent_refit.toml`
   - `reports/nanograv_phase1_independent_refit_reference_2026_03_19.toml`
   - `reports/nanograv_phase1_independent_comparison.toml`
  - `reports/nanograv_phase1_independent_pairwise.toml`
  - `reports/nanograv_phase1_pairwise_vs_release_audit.toml`
  - `reports/nanograv_phase1_independent_avt_filter.toml`
  - `reports/nanograv_phase1_wideband_hardening.toml`
  - `reports/nanograv_phase1_next20.toml`
   - `docs/research/nanograv_phase1_next20_2026_03_19.md`
   - `data/csv/nanograv_phase1_independent_residuals.csv`
   - `data/csv/nanograv_phase1_independent_residuals_reference_2026_03_19.csv`
   - `data/csv/nanograv_phase1_independent_pairwise.csv`
   - `data/csv/nanograv_phase1_independent_avt_whitening_sweep.csv`
   - `data/csv/nanograv_phase1_independent_frustrations_512d.csv`

## Scope notes

- This lane uses `hifitime` for UTC/TT/TDB conversion and local DE440 geometry plus cached
  Earth-orientation kernels via `anise`.
- The current BT/DD/DDK support is family-specific rather than a shared Keplerian branch:
  BT follows the cached Blandford-Teukolsky structure, DD follows the Damour-Deruelle
  inverse/Shapiro/aberration decomposition, and DDK adds Kopeikin proper-motion/parallax
  corrections on top of DD.
- The current ELL1 lane uses the cached PINT small-eccentricity formulas, including the
  `FB0..FBn` orbit parameterization required by `J2214+3000`.
- The current GLS path uses a structured low-rank-plus-diagonal covariance rather than the older
  dense pairwise matrix: calibrated white-noise floors, Fourier basis terms, localized kernel
  basis terms for long-timescale phase/DM processes, calibrated paired phase-plus-DM white blocks,
  and ECORR group columns are assembled and solved with a Woodbury-style inverse application.
- The current Phase 1 artifact reports three acceptance views per pulsar:
  raw residual RMS, weighted RMS, and a synthesis track that combines raw/weighted/DM
  improvement fractions when those metrics disagree.
- The current Phase 1 lane is no longer limited to the dominant subgroup only; all Phase 1
  frontend/backend groups are included, with `JUMP@...` and `DMJUMP@...` parameters exposed in
  the fit output.
- The current tactical split is explicit in the comparison artifact:
  `J1903+0327` and `J2214+3000` are `GLS-first`; `J0709+0458`, `J1312+0051`, `J1713+0747`, and
  `J2317+1439` are `WLS-first`.
- The current Phase 1 wideband `.par` models for these six systems expose `EFAC`, `EQUAD`,
  `DMEFAC`, and `DMEQUAD`, but not active `ECORR` terms, so `ecorr_basis_count = 0` is currently
  expected rather than a selector-matching failure.
- The pairwise regenerated-residual surface is now emitted from the independent residual lane
  itself; it is suitable for first-pass downstream comparison work, but not yet a publication-grade
  HD inference product.
- The comparison artifact now carries both `recommended_solver` and stricter
  `operational_solver` fields, plus sigma-ranked family crosschecks, so downstream products can
  refuse numerically or physically implausible fits without silently falling back.
- The wideband hardening artifact now records per-pulsar paired-white calibration terms
  (`phase_dm_white_rho`, clip, effective coupling, and block fraction) plus readiness/blocker
  status. In the current snapshot all six Phase 1 systems now clear the old
  `development_only` bucket and sit in `advisory` readiness instead. That is
  a real hardening improvement, even though publication-grade closure is still
  incomplete because the GLS-first systems do not yet beat WLS cleanly.
- The AVT lane can now ingest independent residual CSVs directly. The current Phase 1 rerun still
  shows the same structural limitation as the larger release-wide AVT audit: the static per-pulsar
  field changes raw means slightly but leaves centered intra-pulsar scatter invariant.
- This dossier supports the repo's first independent timing-engine slice; it does not by
  itself justify any detection claim.
