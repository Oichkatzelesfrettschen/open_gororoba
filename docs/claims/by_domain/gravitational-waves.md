# Claims: gravitational-waves

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/claims_domains.toml -->

Count: 7

- Hypothesis C-006 (Verified (snapshot reproducibility), 2026-01-28): GWTC-3 "confident events" data integrated into `data/external/GWTC-3_confident.csv` and matches the GWOSC eventapi jsonfull endpoint snapshot.
  - Where stated: `docs/archive/RESEARCH_STATUS.md`, `docs/BIBLIOGRAPHY.md`

- Hypothesis C-007 (Speculative (/ suggestive but fragile (Phase 2D Bayesian mixture)), 2026-02-02): GWTC-3 BH mass distribution is suggestive of multimodality (Phase 2D mixture modeling); any link to \"negative dimension eigenmodes\" remains a speculative interpretation.
  - Where stated: `docs/archive/RESEARCH_STATUS.md`, `docs/external_sources/GWTC3_MASS_CLUMPING_PLAN.md`, `docs/external_sources/GWTC3_DECISION_RULE_SUMMARY.md`, `docs/external_sources/GWTC3_POPULATION_SOURCES.md`, `docs/external_sources/GWTC3_SELECTION_FUNCTION_SPEC.md`, `crates/stats_core/src/claims_gates.rs`, `src/scripts/analysis/gwtc3_mass_clumping_null_models.py`, `src/scripts/analysis/gwtc3_mass_clumping_bootstrap.py`, `src/scripts/analysis/gwtc3_mass_clumping_decision_rule_summary.py`, `src/scripts/analysis/gwtc3_selection_weight_sweep.py`, `src/scripts/data/convert_gwtc3_injections_hdf.py`, `src/scripts/analysis/gwtc3_selection_function_from_injections.py`, `src/scripts/analysis/gwtc3_modality_preregistered.py`, `src/scripts/analysis/gwtc3_bayesian_mixture.py`, `data/external/gwtc3_injection_summary.csv`, `data/external/gwtc3_injection_summary_7890398.csv`, `data/csv/gwtc3_selection_function_binned.csv`, `data/csv/gwtc3_selection_function_binned_7890398.csv`, `data/csv/gwtc3_selection_function_binned_combined.csv`, `data/csv/gwtc3_selection_function_binned_combined_altbins.csv`, `data/csv/gwtc3_mass_clumping_metrics.csv`, `data/csv/gwtc3_mass_clumping_null_models.csv`, `data/csv/gwtc3_mass_clumping_bootstrap_counts.csv`, `data/csv/gwtc3_mass_clumping_bootstrap_summary.csv`, `data/csv/gwtc3_mass_clumping_decision_rule_summary.csv`, `data/csv/gwtc3_selection_bias_control_metrics.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o123.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o3a_altbins.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o123_altbins.csv`, `data/csv/gwtc3_selection_weight_sweep.csv`, `data/csv/gwtc3_selection_weight_sweep_o123.csv`, `data/csv/gwtc3_selection_weight_sweep_o3a_altbins.csv`, `data/csv/gwtc3_selection_weight_sweep_o123_altbins.csv`, `data/csv/gwtc3_bayesian_mixture_results.csv`, `docs/preregistered/GWTC3_MODALITY_TEST.md`, `docs/preregistered/GWTC3_BAYESIAN_MIXTURE.md`

- Hypothesis C-025 (Refuted (Phase 5), 2026-01-31): GWTC-3 black hole sky positions cluster around projected sedenion zero-divisor coordinates (CMB-aligned box-kite projection).
  - Where stated: `docs/stellar_cartography/theory/HYPOTHESIS_DEF.md`, `crates/algebra_core/src/boxkites.rs`, `crates/stats_core/src/claims_gates.rs`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `tests/test_tscp_alignment_offline.py`, `docs/external_sources/TSCP_SKY_ALIGNMENT_SOURCES.md`

- Hypothesis C-026 (Speculative (mechanism missing; baseline metrics only), 2026-02-03): Speculative program: if a statistically significant "lower mass gap" (~2.5-5 M_sun) is established, it may correspond to an "algebraic tension" region between stable zero-divisor nodes in the sedenion box-kite structure.
  - Where stated: `docs/stellar_cartography/theory/HYPOTHESIS_DEF.md`, `docs/C026_MASS_GAP_MECHANISM.md`, `docs/external_sources/C026_MASS_GAP_SOURCES.md`, `src/scripts/analysis/gwtc3_lower_mass_gap_metrics.py`, `data/csv/gwtc3_lower_mass_gap_metrics.csv`

- Hypothesis C-060 (Verified (data), 2026-01-31): GWTC-3 sky localizations (64 events) integrated.
  - Where stated: `src/scripts/data/fetch_gwtc3_confident.py`, `tests/test_gwtc3_sky_localization_areas.py`, `data/external/GWTC-3_confident.json`

- Hypothesis C-061 (Verified (data), 2026-01-31): O4 GW events (10 confirmed) integrated.
  - Where stated: `src/scripts/data/fetch_o4_events.py`, `tests/test_gwtc_o4_events.py`

- Hypothesis C-439 (Refuted (Rust pipeline, N=35 events), 2026-02-06): GWTC-3 mass-redshift clustering exhibits ultrametric structure (C-071 follow-up, Direction 4).
  - Where stated: `crates/gororoba_cli/src/bin/gw_merger_tree.rs`, `data/csv/c071e_gw_merger_ultrametric.csv`
