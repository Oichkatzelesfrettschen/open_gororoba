# Claims -> Tasks Tracker (Executable Backlog)

Cross-references: [CLAIMS_EVIDENCE_MATRIX](CLAIMS_EVIDENCE_MATRIX.md) |
[TERMINOLOGY_GLOSSARY](TERMINOLOGY_GLOSSARY.md) |
[BIBLIOGRAPHY](BIBLIOGRAPHY.md)

This file converts `docs/CLAIMS_EVIDENCE_MATRIX.md` items into concrete, executable tasks.

Rules:
- Each claim should have (at minimum) a citation task and a validation task.
- Prefer unit tests under `tests/` for small invariants; use `src/verification/` for artifact/schema checks.
- Avoid network in tests. Network fetch scripts belong in `bin/` or `src/` and must write provenance.
- Date header is ISO: `Date: YYYY-MM-DD`.
- Task status tokens are canonical: `TODO`, `IN PROGRESS`, `PARTIAL`, `DONE`, `REFUTED`, `DEFERRED`, `BLOCKED`.

Date: 2026-02-06

Current status snapshot policy:
- Treat any test-count or resolved-count line in this file as date-scoped.
- Use the newest date-stamped snapshot as canonical for progress reporting.
- Historical snapshots are retained for provenance and should not be merged
  into one cumulative metric without re-running full gates.

## Phase 7 Sprint 6.1: Rust Infrastructure Buildouts (2026-02-06)

| Module | Work Done | Claims Affected |
|--------|-----------|-----------------|
| `stats_core::lib.rs` | Added `AssociatorGrowthFitResult`, `fit_associator_growth_law()`, `c074_decision_rule()` with bootstrap CI for C-074 verification | C-074 |
| `algebra_core::grassmannian.rs` | New module: Grassmannian geodesic distance on Gr(k,n), principal angles, pairwise distances | C-005 |

**Test counts:** 761 Rust tests + 137 Python tests = 898 total
**Claims status:** 413/435 resolved (95.0%), 23 intermediate remaining

---

## Phase 7 R6 Triage Summary (2026-02-06)

| Claim | Change | Rationale |
|-------|--------|-----------|
| C-409 | Modeled -> Verified | Engineering artifact: I-beam spaceplate design |
| C-422 | Modeled -> Verified | Simulation artifact; k^{-3} = Kraichnan 1967 enstrophy |
| C-423 | Modeled -> Verified | Engineering artifact: GPU simulator |
| C-424 | Modeled -> Verified | Engineering artifact: warp gate simulation |
| C-428 | Split closure | Kerr=Verified (per C-429,C-432); NegDim=Not supported |
| C-430 | Modeled -> Closed/Refuted | Physics refuted by C-012 (Delta-BIC=+11.6) |
| C-435 | Modeled -> Verified | Engineering artifact: Power Pipeline model |

**Previous resolution:** 420/435 (96.6%)

---

## Active tasks (start here)

| Claim ID | Task | Output artifact(s) | Status |
|---:|---|---|---|
| C-006 | GWOSC snapshot fetcher + provenance (offline-testable). | `bin/fetch_gwtc3_confident.py`, `data/external/GWTC-3_confident.*` | DONE |
| C-006 | Offline test: JSON->CSV exact match + hash checks. | `tests/test_gwosc_eventapi_snapshot.py` | DONE |
| C-025 | TSCP code migration into main package + CLI entrypoints + offline-by-default loader. | `crates/stats_core/src/claims_gates.rs`, `bin/predict_stars.py`, `bin/predict_stars`, `bin/fetch_gwtc3_skylocalizations_zenodo.py` | DONE |
| C-025 | Generate Phase 5 artifacts once Zenodo sky maps are cached (no-network in tests; record provenance; keep artifacts small). | `data/csv/tscp/alignment_scores.csv`, `data/csv/tscp/monte_carlo_results.csv`, `data/external/gwtc3/IGWN-GWTC3p0-v2-PESkyLocalizations.PROVENANCE.json` | TODO |
| C-026 | Lower mass-gap baseline metrics + mechanism plan (no algebra->mass mapping yet). | `docs/external_sources/C026_MASS_GAP_SOURCES.md`, `docs/C026_MASS_GAP_MECHANISM.md`, `src/scripts/analysis/gwtc3_lower_mass_gap_metrics.py`, `data/csv/gwtc3_lower_mass_gap_metrics.csv` | PARTIAL |
| C-027 | Define horizon density proxy + decision rule; generate deterministic mass-scaling artifacts; add invariant tests (toy model only). | `docs/C027_DEFF_HORIZON_TEST.md`, `src/scripts/analysis/deff_horizon_mass_scaling.py`, `data/csv/deff_horizon_mass_scaling.csv`, `data/csv/deff_horizon_mass_scaling_summary.csv`, `tests/test_c027_deff_horizon_mass_scaling.py` | DONE |
| C-047 | Correct legacy E9/E10/E11 statement; cache primary open-access Kac-Moody anchors. | `docs/C047_E_SERIES_KAC_MOODY_AUDIT.md`, `docs/external_sources/C047_E_SERIES_KAC_MOODY_SOURCES.md`, `data/external/papers/arxiv_hep-th_0212256_damour_henneaux_nicolai_2002_e10_small_tension.pdf`, `data/external/papers/arxiv_hep-th_0104081_west_2001_e11_and_m_theory.pdf` | DONE |
| C-047 | Offline check: E8/E9/E10/E11 Cartan signature sanity. | `tests/test_c047_e_series_cartan_signature.py` | DONE |
| C-048 | Add motivic tower anchor sources (analogy only) and cache at least one open-access reference. | `docs/external_sources/C048_MOTIVIC_TOWER_SOURCES.md`, `data/external/papers/arxiv_0901.1632_dugger_isaksen_2009_motivic_adams_spectral_sequence.pdf` | DONE |
| C-050 | Make optimization isomorphism explicit as a toy LP equivalence; add deterministic artifact + test. | `docs/C050_SPACEPLATE_FLOW_ISOMORPHISM.md`, `src/scripts/analysis/c050_spaceplate_flow_isomorphism_toy.py`, `data/csv/c050_spaceplate_flow_isomorphism_toy.csv`, `tests/test_c050_spaceplate_flow_isomorphism_toy.py` | DONE |
| C-007 | Falsifiable clumping metric + null-model comparison + bootstrap stability + decision rule. | `docs/external_sources/GWTC3_MASS_CLUMPING_PLAN.md`, `docs/external_sources/GWTC3_DECISION_RULE_SUMMARY.md`, `crates/stats_core/src/claims_gates.rs`, `src/scripts/analysis/gwtc3_mass_clumping_null_models.py`, `src/scripts/analysis/gwtc3_mass_clumping_bootstrap.py`, `src/scripts/analysis/gwtc3_mass_clumping_decision_rule_summary.py`, `data/csv/gwtc3_mass_clumping_metrics.csv`, `data/csv/gwtc3_mass_clumping_null_models.csv`, `data/csv/gwtc3_mass_clumping_bootstrap_counts.csv`, `data/csv/gwtc3_mass_clumping_bootstrap_summary.csv`, `data/csv/gwtc3_mass_clumping_decision_rule_summary.csv` | PARTIAL |
| C-007 | Add selection-bias control (toy proxy + decision rule; still not injection-based). | `docs/external_sources/GWTC3_POPULATION_SOURCES.md`, `crates/stats_core/src/claims_gates.rs`, `src/scripts/analysis/gwtc3_mass_clumping_selection_control.py`, `src/scripts/analysis/gwtc3_selection_weight_sweep.py`, `data/csv/gwtc3_selection_bias_control_metrics.csv`, `data/csv/gwtc3_selection_weight_sweep.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o123.csv`, `data/csv/gwtc3_selection_weight_sweep_o123.csv`, `data/csv/gwtc3_selection_weight_sweep_o3a_altbins.csv`, `data/csv/gwtc3_selection_weight_sweep_o123_altbins.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o3a_altbins.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o123_altbins.csv` | PARTIAL |
| C-007 | Injection-based selection function (offline cached) + refit decision rule. | `docs/external_sources/GWTC3_POPULATION_SOURCES.md`, `docs/external_sources/GWTC3_SELECTION_FUNCTION_SPEC.md`, `src/scripts/data/convert_gwtc3_injections_hdf.py`, `src/scripts/analysis/gwtc3_selection_function_from_injections.py`, `data/external/gwtc3_injection_summary.csv`, `data/csv/gwtc3_selection_function_binned.csv` | DONE |
| C-008 | CLOSED/TOY: alpha=-1.5 is a parameter choice yielding d_s=2 under Convention B; this is a dimensional-analysis coincidence, not a derivation. No physical mechanism connects fractional Laplacian alpha to d_s. | `docs/NEGATIVE_DIMENSION_CLARIFICATIONS.md`, `docs/FRACTIONAL_OPERATOR_POLICY.md`, `docs/theory/PARISI_SOURLAS_ALPHA_DERIVATION.md` | DONE |
| C-009 | Add deterministic tensor-network measurement pipeline + scaling + decision artifacts. | `docs/external_sources/TENSOR_NETWORK_SOURCES.md`, `crates/quantum_core/src/tensor_networks.rs`, `src/scripts/measure/measure_tensor_network_entropy.py`, `src/scripts/measure/measure_tensor_network_entropy_scaling.py`, `src/scripts/measure/measure_tensor_network_entropy_decision.py`, `data/csv/tensor_network_entropy_metrics.csv`, `data/csv/tensor_network_entropy_scaling.csv`, `data/csv/tensor_network_entropy_decision.csv`, `tests/test_tensor_network_entropy_tiny.py`, `tests/test_tensor_network_entropy_scaling.py`, `tests/test_tensor_network_entropy_decision.py` | PARTIAL |
| C-010 | Materials "perfect absorption" claim: keep speculative; ground in absorber literature + baseline metrics. | `docs/external_sources/METAMATERIAL_ABSORBER_SOURCES.md`, `docs/C010_ABSORBER_TCMT_MAPPING.md`, `docs/C010_ABSORBER_SALISBURY.md`, `docs/C010_ABSORBER_RLC.md`, `docs/C010_ABSORBER_CPA.md`, `src/scripts/analysis/materials_absorber_tcmt.py`, `src/scripts/analysis/materials_absorber_salisbury.py`, `src/scripts/analysis/materials_absorber_rlc.py`, `src/scripts/analysis/materials_absorber_cpa_twoport.py`, `src/scripts/analysis/materials_absorber_cpa_input_sweep.py`, `data/csv/c010_truncated_icosahedron_graph.csv`, `data/csv/c010_tcmt_truncated_icosahedron.csv`, `data/csv/c010_tcmt_truncated_icosahedron_minima.csv`, `data/csv/c010_salisbury_screen.csv`, `data/csv/c010_salisbury_screen_minima.csv`, `data/csv/c010_rlc_surface_impedance.csv`, `data/csv/c010_rlc_surface_impedance_minima.csv`, `data/csv/c010_rlc_fit_summary.csv`, `data/csv/c010_cpa_twoport_scan.csv`, `data/csv/c010_cpa_twoport_minima.csv`, `data/csv/c010_cpa_input_sweep.csv`, `data/csv/c010_cpa_input_sweep_minima.csv`, `data/csv/materials_baseline_metrics.csv`, `data/csv/materials_embedding_benchmarks.csv` | PARTIAL |
| C-011 | Gravastar equivalence: keep speculative; add primary sources; require derivation+numeric regression. | `docs/external_sources/GRAVASTAR_SOURCES.md`, `docs/SEDENION_GRAVASTAR_EQUIVALENCE.md`, `src/gravastar_tov.py`, `src/scripts/analysis/gravastar_eos_sweep.py`, `src/scripts/analysis/gravastar_thin_shell.py`, `src/scripts/analysis/gravastar_thin_shell_stability.py`, `data/csv/gravastar_eos_pressure_gradient_sweep.csv`, `data/csv/gravastar_thin_shell_matching.csv`, `data/csv/gravastar_thin_shell_stability.csv`, `tests/test_tov_uniform_density.py`, `tests/test_tov_gravastar_core.py`, `tests/test_tov_gravastar_sweep.py`, `tests/test_gravastar_thin_shell.py`, `tests/test_gravastar_thin_shell_stability.py` | PARTIAL |
| C-012 | Dark energy via "negative dimension diffusion": REFUTED BY DATA (Phase 2.2 complete). Model with eta=0 cannot vary w from -1, making it equivalent to Lambda-CDM but penalized. Extension to free eta required for genuine test (Task #94). | `docs/external_sources/NEGATIVE_DIMENSION_SOURCES.md`, `docs/PHYSICAL_INTERPRETATION.md`, `docs/NEGATIVE_DIMENSION_DARK_ENERGY_MODEL.md`, `docs/NEG_DIM_MULTIPROBE_EXPERIMENT.md`, `crates/spectral_core/src/neg_dim.rs`, `crates/spectral_core/src/neg_dim.rs`, `src/scripts/analysis/neg_dim_model_comparison.py`, `data/csv/neg_dim_model_comparison_results.csv` | REFUTED |
| C-004 | PSL(2,7) action: construct explicit 168-element action and show it permutes the 7 box-kites as subgraphs. | `docs/external_sources/PSL_2_7_SOURCES.md`, `crates/algebra_core/src/group_theory.rs`, `crates/algebra_core/src/boxkites.rs`, `tests/test_psl_2_7_action.py`, `tests/test_boxkite_symmetry_action.py` | DONE |
| C-005 | Reggiani manifold identifications: extract paper-asserted statements + keep unreplicated unless a geometric check is added. | `docs/external_sources/REGGIANI_MANIFOLD_CLAIMS.md` | PARTIAL |
| C-019 | Wheels->Cayley-Dickson mapping: exploratory totalized-division interpretation created (Phase 3.1.1). | `docs/external_sources/WHEELS_CAYLEY_DICKSON_SOURCES.md`, `crates/algebra_core/src/wheels.rs` | PARTIAL |
| C-007 | Pre-registered Hartigan dip test for GWTC-3 mass modality (Phase 2.1) - TEST NOT EXECUTED: insufficient sample size (N=34 < 50 pre-registered minimum). Validation correctly prevented underpowered test. Mass clumping claims downgraded to exploratory status only. | `src/scripts/analysis/gwtc3_modality_preregistered.py`, `docs/preregistered/GWTC3_MODALITY_TEST.md` | DONE |
| C-012 | Lambda-CDM baseline comparison for dark energy model (Phase 2.2) - MODEL REFUTED: Negative-dimension model (eta=0) is mathematically equivalent to Lambda-CDM (w=-1 always) but penalized for extra parameter. Delta-AIC=+11.6, Delta-BIC=+11.6 vs constant-w (best model) -> DECISIVELY REJECTED per pre-registered threshold (>=10). Constant-w finds w=-1.086 (phantom-like), improving chi2 by 11.6 over Lambda-CDM. Critical discovery: SDSS DR12 BAO observables use convention D_M*(rd,fid/rd) [Mpc], H*(rd/rd,fid) [km/s/Mpc], NOT dimensionless ratios (fixed in src/scripts/data/fix_bao_observables.py). Data: Pantheon+ (1701 SNe) + Moresco H(z) (33 points) + DR12 BAO (3 bins) = 1740 total points. Completion date: 2026-01-29. | `src/scripts/analysis/neg_dim_model_comparison.py`, `src/scripts/data/fix_bao_observables.py`, `docs/preregistered/NEG_DIM_MODEL_COMPARISON.md`, `docs/NEGATIVE_DIMENSION_DARK_ENERGY_MODEL.md`, `docs/external_sources/ADDITIONAL_COSMOLOGICAL_DATASETS.md`, `data/csv/neg_dim_model_comparison_results.csv` | DONE |
| C-009 | Multi-system tensor entropy study with pre-registered protocol (Phase 2.3). | `docs/preregistered/TENSOR_ENTROPY_SCALING.md` | IN PROGRESS |
| C-010 | Materials baseline v2 with k-fold cross-validation (Phase 3.2). | `data/csv/materials_baseline_metrics.csv` | DONE |
| C-024 | C++ acceleration kernels scaffold: CMake + Conan + Catch2 + pybind11 (Phase 4). | `cpp/`, `cpp/tests/test_cd_algebra.cpp`, `cpp/benchmarks/bench_cd_multiply.cpp` | DONE |
| C-018 | Wheel axioms validated on a concrete model. | `tests/test_wheels.py` | DONE |
| C-020 | Legacy 16D ZD adjacency matrix: refuted as noise/hallucination. | `docs/LEGACY_ARTIFACT_AUDIT.md` | DONE |
| C-021 | 1024D basis-to-lattice mapping: refuted as inconsistent. | `docs/LEGACY_ARTIFACT_AUDIT.md` | DONE |
| C-022 | CLOSED/ANALOGY: Ordinal birthday mapping only; maps CD doubling level n to surreal birthday n and tests property-loss cascade. Does not implement full Conway/Gonshor surreal arithmetic. | `src/scripts/analysis/surreal_cd_ordinal_construction.py`, `data/csv/surreal_cd_ordinal_mapping.csv`, `tests/test_surreal_cd_ordinal.py` | DONE |
| C-023 | CLOSED/TOY: Basis-vector holonomy model; compares associator norms across CD levels and finds statistical signal (p~9.4e-8) between ZD-adjacent vs generic triples. "Holonomy" label is metaphorical, not geometric. | `crates/algebra_core/src/grassmannian.rs`, `data/csv/cd_fiber_holonomy_comparison.csv`, `tests/test_cd_holonomy.py` | DONE |

## Backfill (triage) tasks (auto-generated)

These are placeholders to keep `docs/CLAIMS_TASKS.md` in sync with the matrix.
Each row should eventually be replaced with a claim-specific citation task and
claim-specific validation task (test or verifier).

| Claim ID | Task | Output artifact(s) | Status |
|---:|---|---|---|
| C-053 | Add dedicated sources index + deterministic toy artifact + offline unit test (diagonal-only degeneracy made explicit). | `docs/external_sources/C053_PATHION_METAMATERIAL_MAPPING_SOURCES.md`, `src/scripts/analysis/c053_pathion_metamaterial_mapping.py`, `data/csv/c053_pathion_tmm_summary.csv`, `tests/test_c053_pathion_metamaterial_mapping.py` | DONE |
| C-068 | Add deterministic spectrum-degeneracy reproduction (84 diagonal ZDs) + unit test; mark PDG-spectrum-match claim refuted for the standard set. | `docs/external_sources/C068_PDG_SPECTRUM_MATCH_SOURCES.md`, `src/scripts/analysis/c068_zd_interaction_spectrum_degeneracy.py`, `data/csv/c068_zd_interaction_eigen_summary.csv`, `tests/test_c068_zd_interaction_spectrum_degeneracy.py` | REFUTED |
| C-070 | Audit legacy Spearman-based "shape match" and encode decision rule as offline artifacts + test. | `docs/external_sources/C070_NANOGRAV_SPECTRUM_MATCH_SOURCES.md`, `src/scripts/analysis/c070_nanograv_shape_match_audit.py`, `data/csv/c070_nanograv_shape_match_summary.csv`, `tests/test_c070_nanograv_shape_match_audit.py` | DONE |
| C-074 | Anchor associator growth fit (v3 exp2) with sources index + offline fit-sanity test; leave uncertainty estimation as follow-up. | `docs/external_sources/C074_ASSOCIATOR_GROWTH_LAW_SOURCES.md`, `data/csv/cd_algebraic_experiments_v3.json`, `tests/test_claim_c074_associator_growth.py` | PARTIAL |
| C-075 | Reproduce legacy v3 exp3 pathion interaction spectrum (distinct eigenvalues + log-span) with deterministic artifacts + unit test. | `src/scripts/analysis/c075_pathion_zd_interaction_spectrum.py`, `data/csv/c075_pathion_interaction_summary.csv`, `tests/test_c075_pathion_zd_interaction_spectrum.py` | DONE |
| C-077 | REFUTED: Associator mixing matrix is near-democratic (Frobenius distance 0.611 from PMNS; mean diagonal 0.314 vs PMNS 0.54). No symmetry-breaking mechanism to generate PMNS structure. | `src/scripts/analysis/c077_associator_mixing_pmns_audit.py`, `data/csv/c077_associator_mixing_summary.csv`, `tests/test_c077_associator_mixing_pmns_audit.py` | REFUTED |
| C-078 | REFUTED: 32D/64D diagonal-form ZDs yield identical spectrum diversity to 16D (66 distinct eigenvalues, 2.83-decade span). General-form ZD search required. | `src/scripts/analysis/c078_higher_dim_zd_coverage_audit.py`, `data/csv/c078_higher_dim_zd_coverage_summary.csv`, `tests/test_c078_higher_dim_zd_coverage_audit.py` | REFUTED |
| C-082 | Extract cached v4 expE associator-saturation fit into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c082_associator_saturation_extended_audit.py`, `data/csv/c082_associator_saturation_summary.csv`, `data/csv/c082_associator_saturation_by_dim.csv`, `tests/test_c082_associator_saturation_extended_audit.py` | DONE |
| C-087 | Extract v5 cross-term/independence metrics into CSV artifacts and add a unit test that enforces vanishing correlation at high dimension. | `src/scripts/analysis/c087_associator_independence_audit.py`, `data/csv/c087_associator_independence_summary.csv`, `tests/test_c087_associator_independence_audit.py` | DONE |
| C-094 | Extract cached v6 mixed-dimension fit degradation and record it as a guardrail against non-2^n extrapolation artifacts. | `src/scripts/analysis/c094_non_power_two_dims_fit_audit.py`, `data/csv/c094_non_power_two_dims_summary.csv`, `tests/test_c094_non_power_two_dims_fit_audit.py` | DONE |
| C-129 | Extract cached v13 associator-norm distribution stats into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c129_associator_distribution_concentration_audit.py`, `data/csv/c129_assoc_norm_dist_summary.csv`, `data/csv/c129_assoc_norm_dist_by_dim.csv`, `tests/test_c129_associator_distribution_concentration_audit.py` | DONE |
| C-087 | Associator norm independence: E[||A||^2] -> 2 as d -> inf, confirmed by Monte Carlo (cross-term correlation decays monotonically). Phase 6 B3. | `src/scripts/analysis/c087_associator_independence_audit.py`, `data/csv/c087_associator_independence_summary.csv`, `tests/test_c087_associator_independence_audit.py` (4 tests) | DONE |
| C-090 | SO(7) rotation drift: ZD condition broken by all non-trivial SO(7) rotations, drift grows with angle. Phase 6 B3. | `src/scripts/analysis/c090_so7_rotation_drift_audit.py`, `data/csv/c090_so7_rotation_drift_summary.csv`, `tests/test_c090_so7_rotation_drift.py` (4 tests) | DONE |
| C-092 | Extract cached v6 orbit-classification summary and record that a "continuous orbit" claim is not supported under the current diagnostic. | `src/scripts/analysis/c092_so7_orbit_structure_audit.py`, `data/csv/c092_so7_orbit_structure_summary.csv`, `tests/test_c092_so7_orbit_structure_audit.py` | DONE |
| C-096 | Extract cached v8 associator-tensor symmetry diagnostics (dims 4/8/16) into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c096_associator_tensor_transitions_audit.py`, `data/csv/c096_associator_tensor_summary.csv`, `tests/test_c096_associator_tensor_transitions_audit.py` | DONE |
| C-097 | Compute diagonal ZD interaction graph invariants (weights + components at threshold > 1) with deterministic artifacts and a unit test. | `src/scripts/analysis/c097_zd_interaction_graph_audit.py`, `data/csv/c097_zd_interaction_graph_summary.csv`, `tests/test_c097_zd_interaction_graph_audit.py` | DONE |
| C-099 | Extract cached v8 non-diagonal ZD geometry summary (PCA dims, kernel, sparsity, angles) into a deterministic CSV artifact + offline unit test. | `src/scripts/analysis/c099_nondiag_zd_geometry_audit.py`, `data/csv/c099_nondiag_zd_geometry_summary.csv`, `tests/test_c099_nondiag_zd_geometry_audit.py` | DONE |
| C-100 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-102 | Reproduce legacy expZ alternativity-ratio convergence into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c102_alt_ratio_convergence_audit.py`, `data/csv/c102_alt_ratio_summary.csv`, `data/csv/c102_alt_ratio_by_dim.csv`, `tests/test_c102_alt_ratio_convergence_audit.py` | DONE |
| C-103 | Reproduce expAA percolation-style ZD connectivity transition into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c103_zd_topology_percolation_audit.py`, `data/csv/c103_zd_topology_by_eps.csv`, `data/csv/c103_zd_topology_summary.csv`, `tests/test_c103_zd_topology_percolation_audit.py` | DONE |
| C-108 | Extract cached v10 alternativity-ratio fit into deterministic CSV artifacts and add an offline unit test. | `src/scripts/analysis/c108_alt_ratio_convergence_audit.py`, `data/csv/c108_alt_ratio_summary.csv`, `data/csv/c108_alt_ratio_by_dim.csv`, `tests/test_c108_alt_ratio_convergence_audit.py` | DONE |
| C-109 | Extract cached v10 probing + reproduce lifted-diagonal kernel doubling into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c109_zd_construction_audit.py`, `data/csv/c109_zd_construction_by_dim.csv`, `data/csv/c109_zd_construction_summary.csv`, `tests/test_c109_zd_construction_audit.py` | DONE |
| C-113 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-115 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-120 | Extract cached v12 ZD kernel scaling into deterministic CSV artifacts and add an offline unit test. | `src/scripts/analysis/c120_zd_kernel_scaling_audit.py`, `data/csv/c120_zd_kernel_scaling_summary.csv`, `data/csv/c120_zd_kernel_scaling_by_dim.csv`, `tests/test_c120_zd_kernel_scaling_audit.py` | DONE |
| C-123 | Extract cached v12 associator Lie bracket metrics into deterministic CSV artifacts and add an offline unit test. | `src/scripts/analysis/c123_associator_lie_bracket_audit.py`, `data/csv/c123_assoc_lie_bracket_summary.csv`, `data/csv/c123_assoc_lie_bracket_by_dim.csv`, `tests/test_c123_associator_lie_bracket_audit.py` | DONE |
| C-126 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-128 | Extract cached v13 conjugate-inverse errors into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c128_conjugate_inverse_audit.py`, `data/csv/c128_conjugate_inverse_summary.csv`, `data/csv/c128_conjugate_inverse_by_dim.csv`, `tests/test_c128_conjugate_inverse_audit.py` | DONE |
| C-130 | Extract cached v14 associator norm sqrt(2) metrics into deterministic CSV artifacts and add an offline unit test. | `src/scripts/analysis/c130_associator_norm_sqrt2_audit.py`, `data/csv/c130_associator_norm_sqrt2_summary.csv`, `data/csv/c130_associator_norm_sqrt2_by_dim.csv`, `tests/test_c130_associator_norm_sqrt2_audit.py` | DONE |
| C-131 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-132 | Extract cached v14 commutator-norm convergence into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c132_commutator_norm_convergence_audit.py`, `data/csv/c132_commutator_norm_summary.csv`, `data/csv/c132_commutator_norm_by_dim.csv`, `tests/test_c132_commutator_norm_convergence_audit.py` | DONE |
| C-135 | Extract cached v14 power-norm scaling into deterministic CSV artifacts + offline unit test. | `src/scripts/analysis/c135_power_norm_scaling_audit.py`, `data/csv/c135_power_norm_summary.csv`, `data/csv/c135_power_norm_by_dim_power.csv`, `tests/test_c135_power_norm_scaling_audit.py` | DONE |
| C-136 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-139 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-141 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-143 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-149 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-150 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-163 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-164 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-165 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-169 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-170 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-171 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-173 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-174 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-176 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-179 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-180 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-183 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-185 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-186 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-187 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-191 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-195 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-197 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-201 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-206 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-207 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-212 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-217 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-218 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-219 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-220 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-221 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-223 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-228 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-231 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-234 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-235 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-239 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-240 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-241 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-243 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-244 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-247 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-248 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-251 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-253 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-256 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-257 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-258 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-259 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-264 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-268 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-269 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-271 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-274 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-278 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-280 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-281 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-283 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-284 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-285 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-286 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-287 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-288 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-289 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-290 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-291 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-300 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-304 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-306 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-309 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-314 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-315 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-316 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-317 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-318 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-321 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-324 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-326 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-329 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-330 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-333 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-334 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-335 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-338 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-339 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-340 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-341 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-342 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-343 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-346 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-349 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-354 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-355 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-358 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-362 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-363 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-366 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-374 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-375 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-378 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-379 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-380 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-381 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-385 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-386 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-387 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-390 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-391 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-394 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-396 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-399 | Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification. | TBD | TODO |
| C-401 | Triage: primary sources cached; still needs a concrete, offline validation check (or keep as blueprint-only). | `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/White_2021_Casimir_Warp.pdf` | PARTIAL |
| C-403 | Source-index + define offline check for spectral-triple-strength "geometry from spectrum" program claims. | `docs/external_sources/EMERGENCE_LAYERS_SOURCES.md`, `src/spectral/demo_pairs.py`, `tests/test_isospectral_nonisomorphic_pair.py`, `src/spectral_triple_toy.py`, `tests/test_spectral_triple_toy.py` | DONE |
| C-404 | Source-index + define offline check for modular-data/entanglement-wedge program claims. | `docs/external_sources/EMERGENCE_LAYERS_SOURCES.md`, `src/scripts/data/fetch_emergence_layers_sources.py`, `data/external/papers/arxiv_hep-th0603001_ryu_takayanagi_2006_rt.pdf`, `data/external/papers/arxiv_0705.0016_hubeny_rangamani_takayanagi_2007_hrt.pdf`, `data/external/papers/arxiv_1512.06431_jafferis_lewkowycz_maldacena_suh_2016_jlms.pdf`, `data/external/papers/arxiv_1609.00026_freedman_headrick_2016_bit_threads.pdf`, `src/holography/maxflow.py`, `tests/test_holography_bit_threads.py` | DONE |
| C-405 | Source-index + define offline check for open-systems/QEC observer program claims. | `docs/external_sources/EMERGENCE_LAYERS_SOURCES.md`, `src/quantum/open_systems/lindblad.py`, `tests/test_open_systems_lindblad.py`, `src/quantum/open_systems/redundancy.py`, `tests/test_open_systems_redundancy.py` | DONE |
| C-406 | Enumerate embedding choices (symmetry proxy) and quantify invariance/trial factors for TSCP alignment. | `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `tests/test_tscp_embedding_sweep.py`, `docs/external_sources/TSCP_METHOD_SOURCES.md` | PARTIAL |
| C-407 | Maintain a look-elsewhere parameter ledger + global correction bounds (Bonferroni/Holm) for TSCP-style searches. | `reports/tscp_trial_factor_ledger.md`, `tests/test_tscp_embedding_sweep.py`, `docs/external_sources/TSCP_METHOD_SOURCES.md` | PARTIAL |
| C-408 | Add symmetric falsification thresholds (N_min + alpha/effect-size) for selected claims; enforce via tests/verifiers. | `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `docs/external_sources/TSCP_METHOD_SOURCES.md` | PARTIAL |
| C-410 | Photon-graviton mixing scope: Schwinger B_cr=4.41e9 T verified; mixing amplitude negligible for lab fields; C-402 NOT overturned. Phase 6 task B6. | `src/scripts/analysis/c410_photon_graviton_scope.py`, `tests/test_c410_photon_graviton_scope.py` (4 tests), `docs/BIBLIOGRAPHY.md` | DONE |
| C-411 | SFWM thin-layer scaling: direct SFWM dominates 5.8x (phase-matching); coherence lengths match paper (33.3/3.1/3.4 um). Phase 6 task B6. | `src/scripts/analysis/c411_sfwm_thin_layer_check.py`, `tests/test_c411_sfwm_thin_layer.py` (4 tests), `docs/BIBLIOGRAPHY.md` | DONE |
| C-417 | Turn "Holographic Entropy Trap" into a falsifiable metric + null; keep speculative until tied to optical-capture baselines and uncertainty. | `src/scripts/analysis/sedenion_warp_synthesis.py`, `data/artifacts/images/sedenion_capture_scaling.png` | TODO |
| C-432 | Kerr geodesic solver + Bardeen analytic shadow boundary validated: Schwarzschild a=0 shadow radius sqrt(27) within 0.1%, photon orbit radii exact, impact parameters xi^2+eta=27, high-spin D-shape asymmetry, null geodesic escape/capture. Phase 6 task A9. | `src/gemini_physics/gr/kerr_geodesic.py`, `tests/test_kerr_shadow.py` (8 tests) | DONE |
| C-429 | Kerr shadow asymmetry (a=0.99) validated: D-shape center offset > 0.1 from Schwarzschild symmetric limit. Phase 6 task A9. | `src/gemini_physics/gr/kerr_geodesic.py`, `tests/test_kerr_shadow.py::test_high_spin_shadow_asymmetric` | DONE |
| C-425 | Octonionic (8D) field Hamiltonian formulation: Fano-plane multiplication, Stormer-Verlet symplectic integrator, 7 Noether charges, free-field dispersion omega^2=k^2+m^2. Restricts to octonionic subalgebra to bypass C-030 non-associativity obstruction. Phase 6 task A10. | `crates/algebra_core/src/octonion_field.rs`, `tests/test_octonion_field.py` (7 tests) | DONE |
| C-428 | Kerr geodesic infrastructure from A9 provides Boyer-Lindquist integrator + Mino-time second-order Hamiltonian form. NegDimCosmology coupling still needs validation. | `src/gemini_physics/gr/kerr_geodesic.py`, `tests/test_kerr_shadow.py` | PARTIAL |
| C-426 | Add a preregistered fit protocol (null + trial-factor ledger) for mapping ZD eigenvalues to particle masses; keep as toy until it passes robust controls. | `crates/algebra_core/src/hypercomplex.rs`, `src/scripts/analysis/pathion_particle_fit.py`, `tests/test_pathion_zd_diagonalization.py` | TODO |
| C-427 | Add unit tests for algebraic-media tensor construction invariants (symmetry/normalization bounds) and define a minimal physical decision rule for the mapping. | `src/gemini_physics/metamaterial.py`, `src/scripts/analysis/unified_spacetime_synthesis.py` | TODO |

## Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)

| Claim ID | Task | Output artifact(s) | Status |
|---:|---|---|---|
| C-028 | Aut(S) = G2 x S3 verification (Phase 6A). | `crates/algebra_core/src/hypercomplex.rs`, `tests/test_sedenion_automorphism.py` | DONE |
| C-029 | Three-generation literature review (Phase 6B). | `crates/algebra_core/src/clifford.rs`, `tests/test_sedenion_generations.py` | DONE |
| C-030 | Decision rule + offline bypass checks (associator + alternativity) for sedenion-valued actions. | `docs/C030_SEDENION_LAGRANGIAN_BYPASS.md`, `src/scripts/analysis/c030_sedenion_lagrangian_bypass_checks.py`, `data/csv/c030_sedenion_lagrangian_bypass_checks.csv`, `tests/test_c030_sedenion_lagrangian_bypass_checks.py` | DONE |
| C-031 | Hurwitz/norm-composition transition (1,2,4,8 vs 16) + zero-divisor example artifact + test. | `docs/external_sources/C031_HURWITZ_QUANTIZATION_SOURCES.md`, `src/scripts/analysis/c031_hurwitz_norm_composition_checks.py`, `data/csv/c031_hurwitz_norm_composition_checks.csv`, `tests/test_c031_hurwitz_norm_composition_checks.py`, `docs/C030_SEDENION_LAGRANGIAN_BYPASS.md` | DONE |
| C-032 | Tang (2025) non-associative QED: minimal offline reproduction (Table 2 extraction + subalgebra associator stats). | `docs/external_sources/C032_TANG_2025_SEDENIONIC_QED_SOURCES.md`, `data/external/traces/tang_2025_preprints_org_landing.txt`, `data/external/papers/preprints202511.0427_v1_tang_2025_sedenionic_qed.txt`, `src/scripts/analysis/c032_tang_2025_min_reproduction.py`, `data/csv/c032_tang_2025_table2_lepton_masses.csv`, `data/csv/c032_tang_2025_associator_subalgebra_summary.csv`, `data/csv/c032_tang_2025_associator_basis_triples.csv`, `tests/test_c032_tang_2025_min_reproduction.py` | DONE |
| C-032 | Tang (2025) non-associative QED: mechanize the associator->mass mapping (BLOCKED until the source provides a complete, convention-fixed mapping and scale choice). | `docs/external_sources/C032_TANG_2025_SEDENIONIC_QED_SOURCES.md`, `data/external/traces/tang_2025_preprints_org_landing.txt` | BLOCKED |
| C-033 | SU(5) generator basis verification complete; source does not specify a unique coefficient mapping from sedenion basis to a normalized SU(5) basis (claim demoted accordingly). | `data/external/papers/arxiv_2308.14768_tang_tang_2023_sedenion_su5_generations.pdf`, `docs/external_sources/SEDENION_FIELD_THEORY_SOURCES.md`, `docs/C033_SU5_MAPPING_CLOSURE.md`, `crates/algebra_core/src/group_theory.rs`, `src/scripts/analysis/c033_su5_generator_summary.py`, `data/csv/c033_su5_generator_summary.csv`, `tests/test_su5_generators.py` | DONE |
| C-034 | Chanyal (2014) sedenion gravi-electromagnetism: minimal structural reproduction (two 8D sectors via CD doubling). | `docs/external_sources/C034_CHANYAL_2014_GRAVI_ELECTROMAGNETISM_SOURCES.md`, `data/external/traces/chanyal_2014_springer_landing.txt`, `data/external/traces/chanyal_2014_springer_abstract.txt`, `docs/C034_CHANYAL_2014_REPRODUCTION.md`, `src/scripts/analysis/c034_chanyal_2014_structural_reproduction.py`, `data/csv/c034_sedenion_doubling_identity_check.csv`, `tests/test_c034_sedenion_doubling_identity.py` | DONE |
| C-034 | Chanyal (2014) sedenion gravi-electromagnetism: equation-level reproduction checks (BLOCKED until a legal full-text source is cached). | `docs/external_sources/C034_CHANYAL_2014_GRAVI_ELECTROMAGNETISM_SOURCES.md`, `data/external/traces/chanyal_2014_springer_landing.txt`, `data/external/traces/chanyal_2014_springer_abstract.txt` | BLOCKED |
| C-035 | F4 Casimir epsilon = 1/4 (Phase 7A). | `crates/quantum_core/src/casimir.rs`, `tests/test_f4_casimir.py` | DONE |
| C-036 | Bigraph cosmogenesis simulation (Phase 7B). | `crates/cosmology_core/src/spectral.rs`, `tests/test_bigraph_cosmogenesis.py` | DONE |
| C-037 | Demoted coincidence claim to Not supported; audit note defines mechanism gap and falsification requirements. | `docs/C037_NUMERICAL_COINCIDENCE_AUDIT.md`, `docs/EXCEPTIONAL_COSMOLOGY.md` | DONE |
| C-038 | w0=-5/6 observational test (Phase 7C). DISFAVORED. | `crates/cosmology_core/src/bounce.rs`, `tests/test_exceptional_w0.py` | DONE |
| C-039 | Spectral dimension running on bigraph (Phase 7D). Qualitative consistency with CDT. | `crates/cosmology_core/src/spectral.rs`, `src/scripts/analysis/c039_spectral_dimension_bigraph_sweep.py`, `data/csv/c039_spectral_dimension_bigraph_curve.csv`, `data/csv/c039_spectral_dimension_bigraph_summary.csv`, `tests/test_spectral_dimension.py`, `tests/test_c039_spectral_dimension_bigraph_artifacts.py` | DONE |
| C-040 | Primordial tilt n_s comparison (Phase 7E). Post-hoc; D_eff=2.8-3.0 inconsistent with Planck via Calcagni formula. | `crates/cosmology_core/src/spectral.rs`, `tests/test_primordial_tilt.py` | DONE |
| C-041 | Demote dimensional coincidence claim; record mechanism gap and decision rule. | `docs/C041_F4_STRING_DIMENSION_COINCIDENCE_AUDIT.md`, `docs/EXCEPTIONAL_COSMOLOGY.md` | DONE |
| C-042 | Kozyrev p-adic wavelets: implement eigenbasis for Vladimirov operator + offline tests. | `docs/theory/PADIC_ANALYSIS_FOUNDATIONS.md`, `crates/algebra_core/src/padic.rs`, `tests/test_padic_wavelets.py` | DONE |
| C-043 | Compact object integration pipeline + unified catalog artifact + offline schema test. | `docs/external_sources/C043_COMPACT_OBJECT_CATALOG_SOURCES.md`, `src/scripts/data/fetch_compact_objects.py`, `data/csv/compact_objects_catalog.csv`, `data/csv/compact_objects_catalog.PROVENANCE.json`, `tests/test_c043_compact_objects_catalog_artifact.py` | DONE |
| C-044 | Legacy zero-divisor adjacency matrices refuted; keep reproduction script as guard. | `data/csv/legacy/`, `src/scripts/reproduction/reproduce_zd_adjacency.py`, `docs/LEGACY_ARTIFACT_AUDIT.md` | REFUTED |
| C-045 | Strang splitting convergence demo + validation smoke check. | `examples/strang_splitting_demo.py`, `src/scripts/validation/validate_strang_splitting_convergence.py`, `tests/test_strang_splitting_demo.py` | DONE |

## Phase 6 Batch Triage (2026-02-04)

### Upgraded by Phase 6 work (Sprints 1-4)

| Claim ID | Phase 6 Task | New Status | Justification |
|---:|---|---|---|
| C-432 | A9: Kerr shadow solver | DONE | 8 passing tests: Schwarzschild shadow, photon orbits, impact parameters, D-shape asymmetry, geodesic escape |
| C-429 | A9: Kerr shadow solver | DONE | `test_high_spin_shadow_asymmetric` validates D-shape center offset |
| C-425 | A10: Octonion field Hamiltonian | DONE | 7 passing tests: energy conservation, dispersion, Noether charges, algebraic properties |
| C-428 | A9: Kerr shadow solver | PARTIAL | Boyer-Lindquist integrator validated; NegDim coupling still needs work |
| C-008 | A6: Neg-dim PDE convergence | PARTIAL | Eigenvalue convergence documented; physical interpretation remains speculative |
| C-011 | A5: Gravastar TOV sweeps | PARTIAL | Mass sweep (55 configs) complete; all dM/d(rho_c) < 0 (unstable branch); stiff-shell EoS inherent; algebra-to-gravastar link still speculative |

### Confirmed Refuted (matrix status: "Not supported (rejected)")

These claims were tested in experiments v1-v6 and conclusively rejected.
The CLAIMS_TASKS.md status should reflect this closure.

| Claim ID | Summary | Refutation basis |
|---:|---|---|
| C-069 | Octonionic subalgebra angles reproduce PMNS mixing | Boolean overlaps (0/90 deg only); v3 null test 3.3% baseline |
| C-072 | CMS resonance mass ratios as ZD eigenvalue ratios | Only 3 distinct eigenvalues in 16D; insufficient diversity |
| C-073 | Left-mult operator spectrum matches PDG masses | Diagonal-form ZDs: degenerate {1.0, sqrt(2)} spectrum |
| C-079 | E8 root eigenvalues reproduce particle masses | All roots equal-norm; adjacency gives only 3 distinct values |
| C-080 | FRB DM distribution has p-adic ultrametric structure | Ultrametric fraction 19.8% matches null 20.2%; no signal |
| C-081 | Multi-parameter Givens rotation finds exact PMNS angles | Optimizer trivially fits; see v5 triviality proof |
| C-083 | General-form ZDs have richer left-mult spectrum | CSV ZDs still diagonal-form; same {1.0, sqrt(2)} spectrum |
| C-084 | Yukawa-like symmetry breaking produces PMNS mixing | S3 symmetry too rigid for small perturbations |
| C-085 | CMS resonance ratios match E8+pathion eigenvalue ratios | Null probability too high; matches accidental |

### Remaining unresolved claims by triage category

**Speculative (resolved by Phase 6 B2/B5):**
C-008 (Speculative -- keep with falsification boundary),
C-009 (Not supported -- rescue: critical-point circuit),
C-010 (Speculative -- obstructed by spectral incompatibility),
C-011 (Speculative -- obstructed by non-associativity),
C-012 (REFUTED by observational data),
C-033 (Not supported -- needs explicit mapping from author),
C-070 (Not supported -- needs non-monotone stat),
C-071 (REFUTED -- Rust pipeline with CHIME Cat 1+2, N=600+5045),
C-417 (Speculative -- narrative only; no quantitative support),
C-422 (Speculative -- physical basis undermined by C-012 refutation),
C-424 (Modeled -- engineering concept, no physics claim),
C-426 (Refuted -- dependent on C-091 mass-from-eigenvalue hypothesis),
C-427 (Speculative -- no measured material comparison)

**Partially verified (needs completion):**
C-005 (Reggiani Grassmannian -- deferred: requires G2 isometry proof),
C-406 (TSCP embedding sweep -- deferred: needs algebra bridge),
C-407 (Bonferroni correction -- deferred: blocked by C-406),
C-408 (Falsification boundaries -- deferred: meta-task)
C-087 -> DONE (Phase 6 B3), C-090 -> DONE (Phase 6 B3)

**Modeled (resolved by Phase 6 B4):**
Engineering: C-409, C-418, C-419, C-420, C-421, C-435 (kept as Modeled with scope notes)
Simulation: C-412 (Visualization), C-422 (obstructed by C-012), C-423, C-424 (engineering concept),
  C-428 (Kerr validated, NegDim not), C-430 (physics basis refuted by C-012), C-431 (Visualization)
Math: C-022, C-023 (kept as Modeled with scope notes), C-425 (DONE)

**Literature (resolved by Phase 6 B6):**
C-410 (DONE), C-411 (DONE)

### Conversation mining (Phase 6 C1, 2026-02-04)

5 new concept entries formalized in `docs/CONVOS_CONCEPTS_STATUS_INDEX.md`:

| Concept ID | Theme | Status | Connects to |
|---|---|---|---|
| CX-021 | RG flow spectral scaling / associator growth exponent | Prototype only | C-074, C-087 |
| CX-022 | Triality / Spin(8) / outer automorphisms | Implemented + tested (via C-028) | C-028, C-029 |
| CX-023 | Fractal CSS codes / Sierpinski stabilizers | Prototype only | `fracton_code.py`, `fractal_css_expanded.py` |
| CX-024 | p-adic gauge deformations | Unimplemented (scope pending) | CX-006, C-042 |
| CX-025 | Inverse CD (un-doubling) as A-infinity co-operations | Unimplemented (theory-first) | CX-001 |

### B5: Not-supported claims final disposition (Phase 6, 2026-02-04)

20 claims with status "Not supported" examined for final disposition.

**Closed as REFUTED (12 claims -- clear rejection, no viable rescue path):**

| Claim ID | Summary | Refutation basis |
|---:|---|---|
| C-019 | Wheels provide mathematically justified CD ZD interpretation | Round-trip 0/336 recoveries; connection is notational |
| C-069 | Octonionic subalgebra angles reproduce PMNS mixing | Boolean overlaps (0/90 deg only); v3 null 3.3% |
| C-072 | CMS resonance ratios as ZD eigenvalue ratios | Only 3 distinct eigenvalues in 16D |
| C-073 | Left-mult spectrum matches PDG masses | Degenerate {1.0, sqrt(2)} spectrum |
| C-079 | E8 root eigenvalues reproduce particle masses | All roots equal-norm; 3 distinct adjacency values |
| C-080 | FRB DM distribution has p-adic ultrametric | Null 20.2% vs measured 19.8%; no signal |
| C-081 | Multi-param Givens rotation finds PMNS angles | Optimizer trivially fits; v5 triviality proof |
| C-083 | General-form ZDs from CSV have richer spectrum | Still diagonal-form; same {1.0, sqrt(2)} |
| C-084 | Yukawa symmetry breaking produces PMNS mixing | S3 too rigid for small perturbations |
| C-085 | CMS+E8+pathion eigenvalue ratios match | Null probability too high; accidental |
| C-091 | Non-diagonal ZD spectrum matches 9/12 masses | Null P(>=9)=1.0; 148 evals trivially cover 12 masses |
| C-092 | SO(7) orbit structure is continuous | Clustered (n=17, gap_ratio=16.29) under v6 diagnostic |

**Closed as REFUTED (2 claims -- numerology/coincidence with existing audit notes):**

| Claim ID | Summary | Audit doc |
|---:|---|---|
| C-037 | gamma ~ epsilon ~ 4*lambda_GB ~ 1/4 | `docs/C037_NUMERICAL_COINCIDENCE_AUDIT.md` |
| C-041 | F4 26D = bosonic string D=26 | `docs/C041_F4_STRING_DIMENSION_COINCIDENCE_AUDIT.md` |

**Kept as NOT SUPPORTED (5 claims -- salvageable with methodology upgrade):**

| Claim ID | Summary | Rescue conditions |
|---:|---|---|
| C-009 | Entropy scaling S ~ log(L) + L^{0.5} | Re-test with critical-point circuit (conformal/MERA) |
| C-033 | Sedenion -> SU(5) mapping | Needs explicit coefficient mapping from source author |
| C-070 | Associator power spectrum matches NANOGrav | Needs non-monotone-invariant statistic + null |
| C-077 | Associator mixing resembles PMNS | Needs symmetry-breaking mechanism + explicit null |
| C-078 | Higher-dim ZDs improve mass coverage | Needs non-diagonal/general-form ZD ensemble |

**Resolved (previously data-insufficient, now REFUTED):**

| Claim ID | Summary | Resolution |
|---:|---|---|
| C-071 | FRB DMs exhibit p-adic ultrametric | REFUTED (2026-02-06): Pure Rust pipeline tested CHIME Cat 1 (600 events) + Cat 2 (5045 events). No ultrametric signal in any DM column (Bonferroni-corrected). See `data/csv/c071_frb_ultrametric.csv`. |

### B2: Speculative claims deep analysis (Phase 6, 2026-02-04)

8 speculative claims examined (4 from triage already handled by B5; C-012 already Refuted).

**Reclassified to REFUTED (1 claim):**

| Claim ID | Summary | Basis |
|---:|---|---|
| C-426 | Pathion ZD mass spectrum hypothesis | Dependent on C-091 mass-from-eigenvalue, which is refuted (null P(>=9)=1.0) |

**Kept as SPECULATIVE with falsification boundaries (4 claims):**

| Claim ID | Summary | Falsification boundary |
|---:|---|---|
| C-008 | neg-dim toy operator (alpha=-1.5, d_s=2) | Provide a derivation (not parameter match) connecting frac. Laplacian alpha to physical d_s, or close |
| C-010 | ZD motifs -> metamaterial absorbers | Specify a concrete ZD->network mapping; demonstrate absorption >= 0.99; otherwise close |
| C-011 | Sedenion gravastar parameterization | Implement a non-associativity bypass that yields well-posed action; otherwise close |
| C-417 | Holographic entropy trap | Demonstrate absorption enhancement correlated with ZD density in simulation; otherwise close |

**Reclassified from SPECULATIVE to SPECULATIVE (obstructed) (2 claims):**

| Claim ID | Summary | Obstruction |
|---:|---|---|
| C-422 | Neg-dim vacuum solitons | Physical basis (neg-dim dark energy) refuted by C-012 |
| C-427 | Algebraic metamaterial tensor mapping | No comparison against measured material properties |

**Kept as MODELED (engineering concept, not physics claim) (1 claim):**

C-424: Holographic warp gate simulation -- concept integration demo only.

### B4: Modeled claims upgrade (Phase 6, 2026-02-04)

16 Modeled claims examined. No new tests required -- these are scope/label corrections.

**Math toys (2 claims -- kept as Modeled with explicit scope):**

| Claim ID | Summary | Scope note |
|---:|---|---|
| C-022 | Surreal-CD ordinal mapping | Ordinal birthday mapping only; not full surreal arithmetic |
| C-023 | CD associator as discrete holonomy | Statistical signal real (p~9.4e-8); "holonomy" label is metaphorical |

**Engineering (6 claims -- kept as Modeled; scope = format templates, not experimentally validated):**

C-409 (I-beam spaceplate), C-418 (material database), C-419 (BOM generator),
C-420 (CAD/OpenSCAD), C-421 (Rogers RT5880 design), C-435 (power pipeline model).
All coherent as engineering concepts/templates but none experimentally validated.

**Simulation/Visualization (6 claims):**

| Claim ID | Summary | Disposition |
|---:|---|---|
| C-412 | 5-phase holographic animation | Visualization artifact, not physics claim |
| C-423 | Grand unified simulator v4 | Engineering integration demo |
| C-428 | Integrated warp geodesic | Kerr component VERIFIED (C-432); NegDim component unsupported (C-012) |
| C-430 | NegDim cosmology H(z), D_L(z) | Code correct; physics model refuted by C-012 (no predictive power beyond constant-w) |
| C-431 | Sedenion ZD manifold 3D projection | Visualization artifact, not physics claim |
| C-435 | Power pipeline energy bookkeeping | Conservation-based model; no quantitative validation test |

**Already handled:**
C-422 (B2: obstructed), C-424 (B2: engineering concept), C-425 (DONE: Verified by A10)

### Backlog triage items (C-100 through C-399)

82 backfill items with status TODO remain in the backfill table above.
These represent auto-generated placeholders for claims in the matrix
that need primary sources and validation checks. They should be
batch-processed in Sprint 5 by cross-referencing against existing
`tests/` and `data/csv/` artifacts.

## Phase 7 Sprint 4: R6 400-Series Triage Summary (2026-02-04)

**Rust statistical methodology implemented** (originally `gororoba_kernels/src/stats.rs`, now `crates/stats_core/`):
- Frechet distance for spectrum comparison
- Bootstrap CI with percentile method
- Haar-distributed random unitaries (QR + phase correction)
- PMNS matrix with PDG 2024 values

**400-Series Resolution Status** (C-400 to C-435):

| Category | Count | Status |
|----------|-------|--------|
| Verified | 12 | C-400, C-410, C-411, C-413-416, C-425, C-429, C-432-434 |
| Refuted | 2 | C-402 (Rodal 2025), C-426 (dependent on C-091) |
| Format/Template | 10 | C-409, C-412, C-418-421, C-423-424, C-431, C-435 |
| Modeled (physics) | 3 | C-422, C-428, C-430 |
| Speculative | 6 | C-401, C-403-405, C-417, C-427 |
| Partially verified | 3 | C-406-408 |

**Test counts**: 64 Rust tests (originally gororoba_kernels, now in domain crates) + 154 Python tests

## Phase 7 Rust Module Expansion (2026-02-04)

**New Rust modules** (originally in `gororoba_kernels/`, consolidated into domain crates 2026-02-06):

| Module | Crates Used | Physics Domain | Claims Addressed |
|--------|-------------|----------------|------------------|
| `gravastar.rs` | nalgebra | TOV equation, polytropic EoS, anisotropic pressure | C-011 |
| `zd_graphs.rs` | petgraph | Zero-divisor interaction networks, basis participation | C-078, C-004 |
| `tang_mass.rs` | rand_chacha | Associator-norm mass ratio predictions | C-033, C-032, C-076 |
| `spectral.rs` | - | Calcagni fractal cosmology, Kraichnan k^{-3} | C-008, C-012, C-422 |
| `holographic.rs` | petgraph | Bekenstein bound, RT lattice, area law | C-412, C-417, C-404 |
| `stats.rs` | statrs | Frechet distance, Haar unitaries, PMNS | C-070, C-077 |

**External Rust crates integrated**:
- `petgraph 0.7`: Graph algorithms (max-flow/min-cut, SCC, Dijkstra)
- `statrs 0.18`: Statistical distributions and functions
- `ordered-float 4.6`: Ordered floats for graph weights

**Test count progression**: 76 -> 90 -> 98 -> 441 Rust tests (workspace integration)

**Key discoveries**:
- k^{-3} spectral ansatz EXACTLY matches Kraichnan 2D enstrophy cascade (Phys. Fluids 10, 1967)
- Sedenion ZD graph has non-trivial clustering (de Marrais box-kites confirmed)
- Anisotropic pressure permits stable gravastars at gamma < 4/3 (Cattoen et al. result)

## Phase 7 Batch A Closures (2026-02-05)

**Sprint 1 R1: Close 5 claims with existing evidence**

| Claim ID | Old Status | New Status | Rationale |
|----------|------------|------------|-----------|
| C-008 | Speculative | Closed/Toy | alpha=-1.5 is a parameter choice, not derived |
| C-022 | Modeled | Closed/Analogy | ordinal/birthday mapping only |
| C-023 | Modeled | Closed/Toy | basis-vector holonomy model, not geometric |
| C-077 | Not supported | Refuted | Frobenius distance 0.611 from PMNS |
| C-078 | Not supported | Refuted | 32D/64D yield identical spectrum to 16D |

**Infrastructure fix**: Moved `gororoba_kernels` from `src/` to `crates/` and integrated into workspace.
This resolves the "package believes it's in a workspace when it's not" error.
**Note**: `gororoba_kernels` was later removed (2026-02-06) -- all functionality
consolidated into domain crates (`algebra_core`, `stats_core`, etc.) and
`gororoba_py` serves as the sole thin PyO3 bridge.

**Test counts**: 441 Rust tests (all crates) + 1 doc test (grover)

**Remaining intermediate claims**: 32 (down from 37)

## Phase 8 Rust Integration Summary (2026-02-05)

**Python-to-Rust migration complete** ("synthesisus maximulus protocol"):

**Cross-crate integration tests added** (`crates/gororoba_cli/tests/`):
- `integration_spectral.rs`: 8 tests (fractional Laplacian, neg-dim eigenvalues)
- `integration_cd_algebra.rs`: 11 tests (CD multiply, ZD search, associators)
- `integration_gravastar.rs`: 7 tests (TOV solver, polytropic stability)

**Final test counts**:
- 242 Rust unit/integration tests
- 6 Rust doc tests
- 137 Python tests
- **385 total tests**

**Claims resolution**: 421/435 (96.8%)

**Key physical insights validated**:
1. Negative alpha fractional Laplacian has inverted kinetic energy physics
2. k^{-3} spectrum matches Kraichnan 2D enstrophy cascade exactly
3. Polytropic EoS with gamma >= 4/3 enables stable gravastars
4. Sedenion ZD existence confirmed (Reggiani theorem)

**Batch A closures** (Phase 8):
- C-008: Closed/Toy (parameter choice)
- C-022: Closed/Analogy (ordinal mapping)
- C-023: Closed/Toy (metaphorical holonomy)
- C-077, C-078: Refuted (PMNS/spectrum claims)

## Status Snapshot: 2026-02-07

| Metric | Count |
|--------|-------|
| Rust unit tests | 1449 |
| Rust doc-tests | 14 |
| Rust clippy warnings | 0 |
| Claims in matrix (C-001..C-442) | 443 |
| Dataset providers | 30 |
| Datasets with backing claims | 12 |
| Infrastructure datasets (no claims) | 18 |
| Scientific pillars | 7 |
| Section H items done | 14 / 21 |
| data_core tests | 121 |
| gr_core tests | 394 |
| algebra_core tests | 264 |
| stats_core tests | 189 |
| Rust crate path resolution in matrix | 84 / 84 (100%) |
| GPU ultrametric tests significant (I-011) | 82 / 472 |

Key completions since last snapshot (2026-02-06):
- Motif census extended to dim=256 exact (16 new tests, 5 scaling laws)
- XOR-balanced search extension CX-003 (10 new tests)
- 6 source-first research audits completed (no overclaims)
- 40 dataset validation tests (Union3, Swarm, EHT, GFC, SPK, Landsat, SORCE/TSIS, Hipparcos)
- Dataset coverage report mapping pillars to claims
- Provenance hash verifier (7 tests)
- Doc-link resolver fixing 1 dead reference (6 tests)
- rust-smoke Makefile target added

## Notes

- For current verified math claims, see `docs/VERIFIED_CLAIMS_INDEX.md`.
