# Claims: stellar-cartography

Source: docs/claims/CLAIMS_DOMAIN_MAP.csv + docs/CLAIMS_EVIDENCE_MATRIX.md

Count: 6

- Hypothesis C-025 (**Refuted** (Phase 5), 2026-01-31): GWTC-3 black hole sky positions cluster around projected sedenion zero-divisor coordinates (CMB-aligned box-kite projection).
  - Where stated: `docs/stellar_cartography/theory/HYPOTHESIS_DEF.md`, `crates/algebra_core/src/boxkites.rs`, `crates/stats_core/src/claims_gates.rs`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `tests/test_tscp_alignment_offline.py`, `docs/external_sources/TSCP_SKY_ALIGNMENT_SOURCES.md`
- Hypothesis C-058 (**Verified** (data), 2026-01-31): Planck 2018 parameter summary + CMB spectra integrated.
  - Where stated: `src/scripts/data/fetch_planck_2018_chains.py`, `src/scripts/data/fetch_planck_2018_spectra.py`, `tests/test_planck_2018_chains.py`, `tests/test_planck_2018_spectra.py`
- Hypothesis C-060 (**Verified** (data), 2026-01-31): GWTC-3 sky localizations (64 events) integrated.
  - Where stated: `src/scripts/data/fetch_gwtc3_confident.py`, `tests/test_gwtc3_sky_localization_areas.py`, `data/external/GWTC-3_confident.json`
- Hypothesis C-406 (**Partially verified** (method), 2026-02-02): TSCP/box-kite sky mapping must be invariant under the relevant algebraic symmetries (or else explicitly enumerate tested embeddings) to avoid "picked the embedding that worked".
  - Where stated: `docs/convos/CONVOS_CLAIMS_INBOX.md`, `docs/external_sources/TSCP_METHOD_SOURCES.md`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `crates/algebra_core/src/boxkites.rs`, `tests/test_tscp_alignment_offline.py`, `tests/test_tscp_embedding_sweep.py`
- Hypothesis C-407 (**Partially verified** (method), 2026-02-02): Reported sky-alignment p-values must include explicit trial-factor accounting (look-elsewhere) across tuned degrees of freedom (e.g., box-kite choice, smoothing scale, catalog cuts).
  - Where stated: `docs/convos/CONVOS_CLAIMS_INBOX.md`, `docs/external_sources/TSCP_METHOD_SOURCES.md`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `reports/tscp_trial_factor_ledger.md`, `tests/test_tscp_alignment_offline.py`, `tests/test_tscp_embedding_sweep.py`, `src/verification/verify_tscp_prereg_trial_factors.py`
- Hypothesis C-408 (**Partially verified** (method), 2026-02-02): Every thesis-level hypothesis must have symmetric falsification boundaries: a disconfirmation threshold (N_min, alpha/effect-size) under which the hypothesis is rejected, not merely "needs more data".
  - Where stated: `docs/convos/CONVOS_CLAIMS_INBOX.md`, `docs/external_sources/TSCP_METHOD_SOURCES.md`, `docs/preregistered/`, `src/verification/verify_preregistered_falsification_boundaries.py`
