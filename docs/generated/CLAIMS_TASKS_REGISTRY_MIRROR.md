# Claims Tasks Registry Mirror

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: TOML registry files under registry/ -->

Authoritative source: `registry/claims_tasks.toml`.

- Updated: 2026-02-09
- Source markdown: `docs/CLAIMS_TASKS.md`
- Task count: 212
- Section count: 20
- Canonical status task count: 212
- Noncanonical status task count: 0

## Sections

- CTS-001: Phase 7 Sprint 6.1: Rust Infrastructure Buildouts (2026-02-06) (0 tasks)
- CTS-002: Phase 7 R6 Triage Summary (2026-02-06) (0 tasks)
- CTS-003: Active tasks (start here) (31 tasks)
- CTS-004: Backfill (triage) tasks (auto-generated) (161 tasks)
- CTS-005: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7) (20 tasks)
- CTS-006: Phase 6 Batch Triage (2026-02-04) (0 tasks)
- CTS-007: Upgraded by Phase 6 work (Sprints 1-4) (0 tasks)
- CTS-008: Confirmed Refuted (matrix status: "Not supported (rejected)") (0 tasks)
- CTS-009: Remaining unresolved claims by triage category (0 tasks)
- CTS-010: Conversation mining (Phase 6 C1, 2026-02-04) (0 tasks)
- CTS-011: B5: Not-supported claims final disposition (Phase 6, 2026-02-04) (0 tasks)
- CTS-012: B2: Speculative claims deep analysis (Phase 6, 2026-02-04) (0 tasks)
- CTS-013: B4: Modeled claims upgrade (Phase 6, 2026-02-04) (0 tasks)
- CTS-014: Backlog triage items (C-100 through C-399) (0 tasks)
- CTS-015: Phase 7 Sprint 4: R6 400-Series Triage Summary (2026-02-04) (0 tasks)
- CTS-016: Phase 7 Rust Module Expansion (2026-02-04) (0 tasks)
- CTS-017: Phase 7 Batch A Closures (2026-02-05) (0 tasks)
- CTS-018: Phase 8 Rust Integration Summary (2026-02-05) (0 tasks)
- CTS-019: Status Snapshot: 2026-02-07 (Sprint 8) (0 tasks)
- CTS-020: Notes (0 tasks)

## Tasks

### CTASK-0001 (C-006, DONE)

- Section: Active tasks (start here)
- Source line: 56
- Status raw: DONE
- Canonical: `True`

GWOSC snapshot fetcher + provenance (offline-testable).

Output artifacts:
- `bin/fetch_gwtc3_confident.py`
- `data/external/GWTC-3_confident.*`

### CTASK-0002 (C-006, DONE)

- Section: Active tasks (start here)
- Source line: 57
- Status raw: DONE
- Canonical: `True`

Offline test: JSON->CSV exact match + hash checks.

Output artifacts:
- `tests/test_gwosc_eventapi_snapshot.py`

### CTASK-0003 (C-025, DONE)

- Section: Active tasks (start here)
- Source line: 58
- Status raw: DONE
- Canonical: `True`

TSCP code migration into main package + CLI entrypoints + offline-by-default loader.

Output artifacts:
- `crates/stats_core/src/claims_gates.rs`
- `bin/predict_stars.py`
- `bin/predict_stars`
- `bin/fetch_gwtc3_skylocalizations_zenodo.py`

### CTASK-0004 (C-025, TODO)

- Section: Active tasks (start here)
- Source line: 59
- Status raw: TODO
- Canonical: `True`

Generate Phase 5 artifacts once Zenodo sky maps are cached (no-network in tests; record provenance; keep artifacts small).

Output artifacts:
- `data/csv/tscp/alignment_scores.csv`
- `data/csv/tscp/monte_carlo_results.csv`
- `data/external/gwtc3/IGWN-GWTC3p0-v2-PESkyLocalizations.PROVENANCE.json`

### CTASK-0005 (C-026, PARTIAL)

- Section: Active tasks (start here)
- Source line: 60
- Status raw: PARTIAL
- Canonical: `True`

Lower mass-gap baseline metrics + mechanism plan (no algebra->mass mapping yet).

Output artifacts:
- `docs/external_sources/C026_MASS_GAP_SOURCES.md`
- `docs/C026_MASS_GAP_MECHANISM.md`
- `src/scripts/analysis/gwtc3_lower_mass_gap_metrics.py`
- `data/csv/gwtc3_lower_mass_gap_metrics.csv`

### CTASK-0006 (C-027, DONE)

- Section: Active tasks (start here)
- Source line: 61
- Status raw: DONE
- Canonical: `True`

Define horizon density proxy + decision rule; generate deterministic mass-scaling artifacts; add invariant tests (toy model only).

Output artifacts:
- `docs/C027_DEFF_HORIZON_TEST.md`
- `src/scripts/analysis/deff_horizon_mass_scaling.py`
- `data/csv/deff_horizon_mass_scaling.csv`
- `data/csv/deff_horizon_mass_scaling_summary.csv`
- `tests/test_c027_deff_horizon_mass_scaling.py`

### CTASK-0007 (C-047, DONE)

- Section: Active tasks (start here)
- Source line: 62
- Status raw: DONE
- Canonical: `True`

Correct legacy E9/E10/E11 statement; cache primary open-access Kac-Moody anchors.

Output artifacts:
- `docs/C047_E_SERIES_KAC_MOODY_AUDIT.md`
- `docs/external_sources/C047_E_SERIES_KAC_MOODY_SOURCES.md`
- `data/external/papers/arxiv_hep-th_0212256_damour_henneaux_nicolai_2002_e10_small_tension.pdf`
- `data/external/papers/arxiv_hep-th_0104081_west_2001_e11_and_m_theory.pdf`

### CTASK-0008 (C-047, DONE)

- Section: Active tasks (start here)
- Source line: 63
- Status raw: DONE
- Canonical: `True`

Offline check: E8/E9/E10/E11 Cartan signature sanity.

Output artifacts:
- `tests/test_c047_e_series_cartan_signature.py`

### CTASK-0009 (C-048, DONE)

- Section: Active tasks (start here)
- Source line: 64
- Status raw: DONE
- Canonical: `True`

Add motivic tower anchor sources (analogy only) and cache at least one open-access reference.

Output artifacts:
- `docs/external_sources/C048_MOTIVIC_TOWER_SOURCES.md`
- `data/external/papers/arxiv_0901.1632_dugger_isaksen_2009_motivic_adams_spectral_sequence.pdf`

### CTASK-0010 (C-050, DONE)

- Section: Active tasks (start here)
- Source line: 65
- Status raw: DONE
- Canonical: `True`

Make optimization isomorphism explicit as a toy LP equivalence; add deterministic artifact + test.

Output artifacts:
- `docs/C050_SPACEPLATE_FLOW_ISOMORPHISM.md`
- `src/scripts/analysis/c050_spaceplate_flow_isomorphism_toy.py`
- `data/csv/c050_spaceplate_flow_isomorphism_toy.csv`
- `tests/test_c050_spaceplate_flow_isomorphism_toy.py`

### CTASK-0011 (C-007, PARTIAL)

- Section: Active tasks (start here)
- Source line: 66
- Status raw: PARTIAL
- Canonical: `True`

Falsifiable clumping metric + null-model comparison + bootstrap stability + decision rule.

Output artifacts:
- `docs/external_sources/GWTC3_MASS_CLUMPING_PLAN.md`
- `docs/external_sources/GWTC3_DECISION_RULE_SUMMARY.md`
- `crates/stats_core/src/claims_gates.rs`
- `src/scripts/analysis/gwtc3_mass_clumping_null_models.py`
- `src/scripts/analysis/gwtc3_mass_clumping_bootstrap.py`
- `src/scripts/analysis/gwtc3_mass_clumping_decision_rule_summary.py`
- `data/csv/gwtc3_mass_clumping_metrics.csv`
- `data/csv/gwtc3_mass_clumping_null_models.csv`
- `data/csv/gwtc3_mass_clumping_bootstrap_counts.csv`
- `data/csv/gwtc3_mass_clumping_bootstrap_summary.csv`
- `data/csv/gwtc3_mass_clumping_decision_rule_summary.csv`

### CTASK-0012 (C-007, PARTIAL)

- Section: Active tasks (start here)
- Source line: 67
- Status raw: PARTIAL
- Canonical: `True`

Add selection-bias control (toy proxy + decision rule; still not injection-based).

Output artifacts:
- `docs/external_sources/GWTC3_POPULATION_SOURCES.md`
- `crates/stats_core/src/claims_gates.rs`
- `src/scripts/analysis/gwtc3_mass_clumping_selection_control.py`
- `src/scripts/analysis/gwtc3_selection_weight_sweep.py`
- `data/csv/gwtc3_selection_bias_control_metrics.csv`
- `data/csv/gwtc3_selection_weight_sweep.csv`
- `data/csv/gwtc3_selection_bias_control_metrics_o123.csv`
- `data/csv/gwtc3_selection_weight_sweep_o123.csv`
- `data/csv/gwtc3_selection_weight_sweep_o3a_altbins.csv`
- `data/csv/gwtc3_selection_weight_sweep_o123_altbins.csv`
- `data/csv/gwtc3_selection_bias_control_metrics_o3a_altbins.csv`
- `data/csv/gwtc3_selection_bias_control_metrics_o123_altbins.csv`

### CTASK-0013 (C-007, DONE)

- Section: Active tasks (start here)
- Source line: 68
- Status raw: DONE
- Canonical: `True`

Injection-based selection function (offline cached) + refit decision rule.

Output artifacts:
- `docs/external_sources/GWTC3_POPULATION_SOURCES.md`
- `docs/external_sources/GWTC3_SELECTION_FUNCTION_SPEC.md`
- `src/scripts/data/convert_gwtc3_injections_hdf.py`
- `src/scripts/analysis/gwtc3_selection_function_from_injections.py`
- `data/external/gwtc3_injection_summary.csv`
- `data/csv/gwtc3_selection_function_binned.csv`

### CTASK-0014 (C-008, DONE)

- Section: Active tasks (start here)
- Source line: 69
- Status raw: DONE
- Canonical: `True`

CLOSED/TOY: alpha=-1.5 is a parameter choice yielding d_s=2 under Convention B; this is a dimensional-analysis coincidence, not a derivation. No physical mechanism connects fractional Laplacian alpha to d_s.

Output artifacts:
- `docs/NEGATIVE_DIMENSION_CLARIFICATIONS.md`
- `docs/FRACTIONAL_OPERATOR_POLICY.md`
- `docs/theory/PARISI_SOURLAS_ALPHA_DERIVATION.md`

### CTASK-0015 (C-009, PARTIAL)

- Section: Active tasks (start here)
- Source line: 70
- Status raw: PARTIAL
- Canonical: `True`

Add deterministic tensor-network measurement pipeline + scaling + decision artifacts.

Output artifacts:
- `docs/external_sources/TENSOR_NETWORK_SOURCES.md`
- `crates/quantum_core/src/tensor_networks.rs`
- `src/scripts/measure/measure_tensor_network_entropy.py`
- `src/scripts/measure/measure_tensor_network_entropy_scaling.py`
- `src/scripts/measure/measure_tensor_network_entropy_decision.py`
- `data/csv/tensor_network_entropy_metrics.csv`
- `data/csv/tensor_network_entropy_scaling.csv`
- `data/csv/tensor_network_entropy_decision.csv`
- `tests/test_tensor_network_entropy_tiny.py`
- `tests/test_tensor_network_entropy_scaling.py`
- `tests/test_tensor_network_entropy_decision.py`

### CTASK-0016 (C-010, PARTIAL)

- Section: Active tasks (start here)
- Source line: 71
- Status raw: PARTIAL
- Canonical: `True`

Materials "perfect absorption" claim: keep speculative; ground in absorber literature + baseline metrics.

Output artifacts:
- `docs/external_sources/METAMATERIAL_ABSORBER_SOURCES.md`
- `docs/C010_ABSORBER_TCMT_MAPPING.md`
- `docs/C010_ABSORBER_SALISBURY.md`
- `docs/C010_ABSORBER_RLC.md`
- `docs/C010_ABSORBER_CPA.md`
- `src/scripts/analysis/materials_absorber_tcmt.py`
- `src/scripts/analysis/materials_absorber_salisbury.py`
- `src/scripts/analysis/materials_absorber_rlc.py`
- `src/scripts/analysis/materials_absorber_cpa_twoport.py`
- `src/scripts/analysis/materials_absorber_cpa_input_sweep.py`
- `data/csv/c010_truncated_icosahedron_graph.csv`
- `data/csv/c010_tcmt_truncated_icosahedron.csv`
- `data/csv/c010_tcmt_truncated_icosahedron_minima.csv`
- `data/csv/c010_salisbury_screen.csv`
- `data/csv/c010_salisbury_screen_minima.csv`
- `data/csv/c010_rlc_surface_impedance.csv`
- `data/csv/c010_rlc_surface_impedance_minima.csv`
- `data/csv/c010_rlc_fit_summary.csv`
- `data/csv/c010_cpa_twoport_scan.csv`
- `data/csv/c010_cpa_twoport_minima.csv`
- `data/csv/c010_cpa_input_sweep.csv`
- `data/csv/c010_cpa_input_sweep_minima.csv`
- `data/csv/materials_baseline_metrics.csv`
- `data/csv/materials_embedding_benchmarks.csv`

### CTASK-0017 (C-011, PARTIAL)

- Section: Active tasks (start here)
- Source line: 72
- Status raw: PARTIAL
- Canonical: `True`

Gravastar equivalence: keep speculative; add primary sources; require derivation+numeric regression.

Output artifacts:
- `docs/external_sources/GRAVASTAR_SOURCES.md`
- `docs/SEDENION_GRAVASTAR_EQUIVALENCE.md`
- `src/gravastar_tov.py`
- `src/scripts/analysis/gravastar_eos_sweep.py`
- `src/scripts/analysis/gravastar_thin_shell.py`
- `src/scripts/analysis/gravastar_thin_shell_stability.py`
- `data/csv/gravastar_eos_pressure_gradient_sweep.csv`
- `data/csv/gravastar_thin_shell_matching.csv`
- `data/csv/gravastar_thin_shell_stability.csv`
- `tests/test_tov_uniform_density.py`
- `tests/test_tov_gravastar_core.py`
- `tests/test_tov_gravastar_sweep.py`
- `tests/test_gravastar_thin_shell.py`
- `tests/test_gravastar_thin_shell_stability.py`

### CTASK-0018 (C-012, REFUTED)

- Section: Active tasks (start here)
- Source line: 73
- Status raw: REFUTED
- Canonical: `True`

Dark energy via "negative dimension diffusion": REFUTED BY DATA (Phase 2.2 complete). Model with eta=0 cannot vary w from -1, making it equivalent to Lambda-CDM but penalized. Extension to free eta required for genuine test (Task #94).

Output artifacts:
- `docs/external_sources/NEGATIVE_DIMENSION_SOURCES.md`
- `docs/PHYSICAL_INTERPRETATION.md`
- `docs/NEGATIVE_DIMENSION_DARK_ENERGY_MODEL.md`
- `docs/NEG_DIM_MULTIPROBE_EXPERIMENT.md`
- `crates/spectral_core/src/neg_dim.rs`
- `src/scripts/analysis/neg_dim_model_comparison.py`
- `data/csv/neg_dim_model_comparison_results.csv`

### CTASK-0019 (C-004, DONE)

- Section: Active tasks (start here)
- Source line: 74
- Status raw: DONE
- Canonical: `True`

PSL(2,7) action: construct explicit 168-element action and show it permutes the 7 box-kites as subgraphs.

Output artifacts:
- `docs/external_sources/PSL_2_7_SOURCES.md`
- `crates/algebra_core/src/group_theory.rs`
- `crates/algebra_core/src/boxkites.rs`
- `tests/test_psl_2_7_action.py`
- `tests/test_boxkite_symmetry_action.py`

### CTASK-0020 (C-005, PARTIAL)

- Section: Active tasks (start here)
- Source line: 75
- Status raw: PARTIAL
- Canonical: `True`

Reggiani manifold identifications: extract paper-asserted statements + keep unreplicated unless a geometric check is added.

Output artifacts:
- `docs/external_sources/REGGIANI_MANIFOLD_CLAIMS.md`

### CTASK-0021 (C-019, PARTIAL)

- Section: Active tasks (start here)
- Source line: 76
- Status raw: PARTIAL
- Canonical: `True`

Wheels->Cayley-Dickson mapping: exploratory totalized-division interpretation created (Phase 3.1.1).

Output artifacts:
- `docs/external_sources/WHEELS_CAYLEY_DICKSON_SOURCES.md`
- `crates/algebra_core/src/wheels.rs`

### CTASK-0022 (C-007, DONE)

- Section: Active tasks (start here)
- Source line: 77
- Status raw: DONE
- Canonical: `True`

Pre-registered Hartigan dip test for GWTC-3 mass modality (Phase 2.1) - TEST NOT EXECUTED: insufficient sample size (N=34 < 50 pre-registered minimum). Validation correctly prevented underpowered test. Mass clumping claims downgraded to exploratory status only.

Output artifacts:
- `src/scripts/analysis/gwtc3_modality_preregistered.py`
- `docs/preregistered/GWTC3_MODALITY_TEST.md`

### CTASK-0023 (C-012, DONE)

- Section: Active tasks (start here)
- Source line: 78
- Status raw: DONE
- Canonical: `True`

Lambda-CDM baseline comparison for dark energy model (Phase 2.2) - MODEL REFUTED: Negative-dimension model (eta=0) is mathematically equivalent to Lambda-CDM (w=-1 always) but penalized for extra parameter. Delta-AIC=+11.6, Delta-BIC=+11.6 vs constant-w (best model) -> DECISIVELY REJECTED per pre-registered threshold (>=10). Constant-w finds w=-1.086 (phantom-like), improving chi2 by 11.6 over Lambda-CDM. Critical discovery: SDSS DR12 BAO observables use convention D_M*(rd,fid/rd) [Mpc], H*(rd/rd,fid) [km/s/Mpc], NOT dimensionless ratios (fixed in src/scripts/data/fix_bao_observables.py). Data: Pantheon+ (1701 SNe) + Moresco H(z) (33 points) + DR12 BAO (3 bins) = 1740 total points. Completion date: 2026-01-29.

Output artifacts:
- `src/scripts/analysis/neg_dim_model_comparison.py`
- `src/scripts/data/fix_bao_observables.py`
- `docs/preregistered/NEG_DIM_MODEL_COMPARISON.md`
- `docs/NEGATIVE_DIMENSION_DARK_ENERGY_MODEL.md`
- `docs/external_sources/ADDITIONAL_COSMOLOGICAL_DATASETS.md`
- `data/csv/neg_dim_model_comparison_results.csv`

### CTASK-0024 (C-009, IN_PROGRESS)

- Section: Active tasks (start here)
- Source line: 79
- Status raw: IN PROGRESS
- Canonical: `True`

Multi-system tensor entropy study with pre-registered protocol (Phase 2.3).

Output artifacts:
- `docs/preregistered/TENSOR_ENTROPY_SCALING.md`

### CTASK-0025 (C-010, DONE)

- Section: Active tasks (start here)
- Source line: 80
- Status raw: DONE
- Canonical: `True`

Materials baseline v2 with k-fold cross-validation (Phase 3.2).

Output artifacts:
- `data/csv/materials_baseline_metrics.csv`

### CTASK-0026 (C-024, DONE)

- Section: Active tasks (start here)
- Source line: 81
- Status raw: DONE
- Canonical: `True`

C++ acceleration kernels scaffold: CMake + Conan + Catch2 + pybind11 (Phase 4).

Output artifacts:
- `cpp/`
- `cpp/tests/test_cd_algebra.cpp`
- `cpp/benchmarks/bench_cd_multiply.cpp`

### CTASK-0027 (C-018, DONE)

- Section: Active tasks (start here)
- Source line: 82
- Status raw: DONE
- Canonical: `True`

Wheel axioms validated on a concrete model.

Output artifacts:
- `tests/test_wheels.py`

### CTASK-0028 (C-020, DONE)

- Section: Active tasks (start here)
- Source line: 83
- Status raw: DONE
- Canonical: `True`

Legacy 16D ZD adjacency matrix: refuted as noise/hallucination.

Output artifacts:
- `docs/LEGACY_ARTIFACT_AUDIT.md`

### CTASK-0029 (C-021, DONE)

- Section: Active tasks (start here)
- Source line: 84
- Status raw: DONE
- Canonical: `True`

1024D basis-to-lattice mapping: refuted as inconsistent.

Output artifacts:
- `docs/LEGACY_ARTIFACT_AUDIT.md`

### CTASK-0030 (C-022, DONE)

- Section: Active tasks (start here)
- Source line: 85
- Status raw: DONE
- Canonical: `True`

CLOSED/ANALOGY: Ordinal birthday mapping only; maps CD doubling level n to surreal birthday n and tests property-loss cascade. Does not implement full Conway/Gonshor surreal arithmetic.

Output artifacts:
- `src/scripts/analysis/surreal_cd_ordinal_construction.py`
- `data/csv/surreal_cd_ordinal_mapping.csv`
- `tests/test_surreal_cd_ordinal.py`

### CTASK-0031 (C-023, DONE)

- Section: Active tasks (start here)
- Source line: 86
- Status raw: DONE
- Canonical: `True`

CLOSED/TOY: Basis-vector holonomy model; compares associator norms across CD levels and finds statistical signal (p~9.4e-8) between ZD-adjacent vs generic triples. "Holonomy" label is metaphorical, not geometric.

Output artifacts:
- `crates/algebra_core/src/grassmannian.rs`
- `data/csv/cd_fiber_holonomy_comparison.csv`
- `tests/test_cd_holonomy.py`

### CTASK-0032 (C-053, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 96
- Status raw: DONE
- Canonical: `True`

Add dedicated sources index + deterministic toy artifact + offline unit test (diagonal-only degeneracy made explicit).

Output artifacts:
- `docs/external_sources/C053_PATHION_METAMATERIAL_MAPPING_SOURCES.md`
- `src/scripts/analysis/c053_pathion_metamaterial_mapping.py`
- `data/csv/c053_pathion_tmm_summary.csv`
- `tests/test_c053_pathion_metamaterial_mapping.py`

### CTASK-0033 (C-068, REFUTED)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 97
- Status raw: REFUTED
- Canonical: `True`

Add deterministic spectrum-degeneracy reproduction (84 diagonal ZDs) + unit test; mark PDG-spectrum-match claim refuted for the standard set.

Output artifacts:
- `docs/external_sources/C068_PDG_SPECTRUM_MATCH_SOURCES.md`
- `src/scripts/analysis/c068_zd_interaction_spectrum_degeneracy.py`
- `data/csv/c068_zd_interaction_eigen_summary.csv`
- `tests/test_c068_zd_interaction_spectrum_degeneracy.py`

### CTASK-0034 (C-070, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 98
- Status raw: DONE
- Canonical: `True`

Audit legacy Spearman-based "shape match" and encode decision rule as offline artifacts + test.

Output artifacts:
- `docs/external_sources/C070_NANOGRAV_SPECTRUM_MATCH_SOURCES.md`
- `src/scripts/analysis/c070_nanograv_shape_match_audit.py`
- `data/csv/c070_nanograv_shape_match_summary.csv`
- `tests/test_c070_nanograv_shape_match_audit.py`

### CTASK-0035 (C-074, PARTIAL)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 99
- Status raw: PARTIAL
- Canonical: `True`

Anchor associator growth fit (v3 exp2) with sources index + offline fit-sanity test; leave uncertainty estimation as follow-up.

Output artifacts:
- `docs/external_sources/C074_ASSOCIATOR_GROWTH_LAW_SOURCES.md`
- `data/csv/cd_algebraic_experiments_v3.json`
- `tests/test_claim_c074_associator_growth.py`

### CTASK-0036 (C-075, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 100
- Status raw: DONE
- Canonical: `True`

Reproduce legacy v3 exp3 pathion interaction spectrum (distinct eigenvalues + log-span) with deterministic artifacts + unit test.

Output artifacts:
- `src/scripts/analysis/c075_pathion_zd_interaction_spectrum.py`
- `data/csv/c075_pathion_interaction_summary.csv`
- `tests/test_c075_pathion_zd_interaction_spectrum.py`

### CTASK-0037 (C-077, REFUTED)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 101
- Status raw: REFUTED
- Canonical: `True`

REFUTED: Associator mixing matrix is near-democratic (Frobenius distance 0.611 from PMNS; mean diagonal 0.314 vs PMNS 0.54). No symmetry-breaking mechanism to generate PMNS structure.

Output artifacts:
- `src/scripts/analysis/c077_associator_mixing_pmns_audit.py`
- `data/csv/c077_associator_mixing_summary.csv`
- `tests/test_c077_associator_mixing_pmns_audit.py`

### CTASK-0038 (C-078, REFUTED)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 102
- Status raw: REFUTED
- Canonical: `True`

REFUTED: 32D/64D diagonal-form ZDs yield identical spectrum diversity to 16D (66 distinct eigenvalues, 2.83-decade span). General-form ZD search required.

Output artifacts:
- `src/scripts/analysis/c078_higher_dim_zd_coverage_audit.py`
- `data/csv/c078_higher_dim_zd_coverage_summary.csv`
- `tests/test_c078_higher_dim_zd_coverage_audit.py`

### CTASK-0039 (C-082, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 103
- Status raw: DONE
- Canonical: `True`

Extract cached v4 expE associator-saturation fit into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c082_associator_saturation_extended_audit.py`
- `data/csv/c082_associator_saturation_summary.csv`
- `data/csv/c082_associator_saturation_by_dim.csv`
- `tests/test_c082_associator_saturation_extended_audit.py`

### CTASK-0040 (C-087, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 104
- Status raw: DONE
- Canonical: `True`

Extract v5 cross-term/independence metrics into CSV artifacts and add a unit test that enforces vanishing correlation at high dimension.

Output artifacts:
- `src/scripts/analysis/c087_associator_independence_audit.py`
- `data/csv/c087_associator_independence_summary.csv`
- `tests/test_c087_associator_independence_audit.py`

### CTASK-0041 (C-094, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 105
- Status raw: DONE
- Canonical: `True`

Extract cached v6 mixed-dimension fit degradation and record it as a guardrail against non-2^n extrapolation artifacts.

Output artifacts:
- `src/scripts/analysis/c094_non_power_two_dims_fit_audit.py`
- `data/csv/c094_non_power_two_dims_summary.csv`
- `tests/test_c094_non_power_two_dims_fit_audit.py`

### CTASK-0042 (C-129, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 106
- Status raw: DONE
- Canonical: `True`

Extract cached v13 associator-norm distribution stats into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c129_associator_distribution_concentration_audit.py`
- `data/csv/c129_assoc_norm_dist_summary.csv`
- `data/csv/c129_assoc_norm_dist_by_dim.csv`
- `tests/test_c129_associator_distribution_concentration_audit.py`

### CTASK-0043 (C-087, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 107
- Status raw: DONE
- Canonical: `True`

Associator norm independence: E[||A||^2] -> 2 as d -> inf, confirmed by Monte Carlo (cross-term correlation decays monotonically). Phase 6 B3.

Output artifacts:
- `src/scripts/analysis/c087_associator_independence_audit.py`
- `data/csv/c087_associator_independence_summary.csv`
- `tests/test_c087_associator_independence_audit.py`

### CTASK-0044 (C-090, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 108
- Status raw: DONE
- Canonical: `True`

SO(7) rotation drift: ZD condition broken by all non-trivial SO(7) rotations, drift grows with angle. Phase 6 B3.

Output artifacts:
- `src/scripts/analysis/c090_so7_rotation_drift_audit.py`
- `data/csv/c090_so7_rotation_drift_summary.csv`
- `tests/test_c090_so7_rotation_drift.py`

### CTASK-0045 (C-092, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 109
- Status raw: DONE
- Canonical: `True`

Extract cached v6 orbit-classification summary and record that a "continuous orbit" claim is not supported under the current diagnostic.

Output artifacts:
- `src/scripts/analysis/c092_so7_orbit_structure_audit.py`
- `data/csv/c092_so7_orbit_structure_summary.csv`
- `tests/test_c092_so7_orbit_structure_audit.py`

### CTASK-0046 (C-096, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 110
- Status raw: DONE
- Canonical: `True`

Extract cached v8 associator-tensor symmetry diagnostics (dims 4/8/16) into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c096_associator_tensor_transitions_audit.py`
- `data/csv/c096_associator_tensor_summary.csv`
- `tests/test_c096_associator_tensor_transitions_audit.py`

### CTASK-0047 (C-097, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 111
- Status raw: DONE
- Canonical: `True`

Compute diagonal ZD interaction graph invariants (weights + components at threshold > 1) with deterministic artifacts and a unit test.

Output artifacts:
- `src/scripts/analysis/c097_zd_interaction_graph_audit.py`
- `data/csv/c097_zd_interaction_graph_summary.csv`
- `tests/test_c097_zd_interaction_graph_audit.py`

### CTASK-0048 (C-099, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 112
- Status raw: DONE
- Canonical: `True`

Extract cached v8 non-diagonal ZD geometry summary (PCA dims, kernel, sparsity, angles) into a deterministic CSV artifact + offline unit test.

Output artifacts:
- `src/scripts/analysis/c099_nondiag_zd_geometry_audit.py`
- `data/csv/c099_nondiag_zd_geometry_summary.csv`
- `tests/test_c099_nondiag_zd_geometry_audit.py`

### CTASK-0049 (C-100, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 113
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0050 (C-102, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 114
- Status raw: DONE
- Canonical: `True`

Reproduce legacy expZ alternativity-ratio convergence into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c102_alt_ratio_convergence_audit.py`
- `data/csv/c102_alt_ratio_summary.csv`
- `data/csv/c102_alt_ratio_by_dim.csv`
- `tests/test_c102_alt_ratio_convergence_audit.py`

### CTASK-0051 (C-103, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 115
- Status raw: DONE
- Canonical: `True`

Reproduce expAA percolation-style ZD connectivity transition into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c103_zd_topology_percolation_audit.py`
- `data/csv/c103_zd_topology_by_eps.csv`
- `data/csv/c103_zd_topology_summary.csv`
- `tests/test_c103_zd_topology_percolation_audit.py`

### CTASK-0052 (C-108, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 116
- Status raw: DONE
- Canonical: `True`

Extract cached v10 alternativity-ratio fit into deterministic CSV artifacts and add an offline unit test.

Output artifacts:
- `src/scripts/analysis/c108_alt_ratio_convergence_audit.py`
- `data/csv/c108_alt_ratio_summary.csv`
- `data/csv/c108_alt_ratio_by_dim.csv`
- `tests/test_c108_alt_ratio_convergence_audit.py`

### CTASK-0053 (C-109, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 117
- Status raw: DONE
- Canonical: `True`

Extract cached v10 probing + reproduce lifted-diagonal kernel doubling into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c109_zd_construction_audit.py`
- `data/csv/c109_zd_construction_by_dim.csv`
- `data/csv/c109_zd_construction_summary.csv`
- `tests/test_c109_zd_construction_audit.py`

### CTASK-0054 (C-113, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 118
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0055 (C-115, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 119
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0056 (C-120, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 120
- Status raw: DONE
- Canonical: `True`

Extract cached v12 ZD kernel scaling into deterministic CSV artifacts and add an offline unit test.

Output artifacts:
- `src/scripts/analysis/c120_zd_kernel_scaling_audit.py`
- `data/csv/c120_zd_kernel_scaling_summary.csv`
- `data/csv/c120_zd_kernel_scaling_by_dim.csv`
- `tests/test_c120_zd_kernel_scaling_audit.py`

### CTASK-0057 (C-123, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 121
- Status raw: DONE
- Canonical: `True`

Extract cached v12 associator Lie bracket metrics into deterministic CSV artifacts and add an offline unit test.

Output artifacts:
- `src/scripts/analysis/c123_associator_lie_bracket_audit.py`
- `data/csv/c123_assoc_lie_bracket_summary.csv`
- `data/csv/c123_assoc_lie_bracket_by_dim.csv`
- `tests/test_c123_associator_lie_bracket_audit.py`

### CTASK-0058 (C-126, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 122
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0059 (C-128, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 123
- Status raw: DONE
- Canonical: `True`

Extract cached v13 conjugate-inverse errors into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c128_conjugate_inverse_audit.py`
- `data/csv/c128_conjugate_inverse_summary.csv`
- `data/csv/c128_conjugate_inverse_by_dim.csv`
- `tests/test_c128_conjugate_inverse_audit.py`

### CTASK-0060 (C-130, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 124
- Status raw: DONE
- Canonical: `True`

Extract cached v14 associator norm sqrt(2) metrics into deterministic CSV artifacts and add an offline unit test.

Output artifacts:
- `src/scripts/analysis/c130_associator_norm_sqrt2_audit.py`
- `data/csv/c130_associator_norm_sqrt2_summary.csv`
- `data/csv/c130_associator_norm_sqrt2_by_dim.csv`
- `tests/test_c130_associator_norm_sqrt2_audit.py`

### CTASK-0061 (C-131, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 125
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0062 (C-132, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 126
- Status raw: DONE
- Canonical: `True`

Extract cached v14 commutator-norm convergence into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c132_commutator_norm_convergence_audit.py`
- `data/csv/c132_commutator_norm_summary.csv`
- `data/csv/c132_commutator_norm_by_dim.csv`
- `tests/test_c132_commutator_norm_convergence_audit.py`

### CTASK-0063 (C-135, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 127
- Status raw: DONE
- Canonical: `True`

Extract cached v14 power-norm scaling into deterministic CSV artifacts + offline unit test.

Output artifacts:
- `src/scripts/analysis/c135_power_norm_scaling_audit.py`
- `data/csv/c135_power_norm_summary.csv`
- `data/csv/c135_power_norm_by_dim_power.csv`
- `tests/test_c135_power_norm_scaling_audit.py`

### CTASK-0064 (C-136, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 128
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0065 (C-139, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 129
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0066 (C-141, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 130
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0067 (C-143, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 131
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0068 (C-149, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 132
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0069 (C-150, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 133
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0070 (C-163, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 134
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0071 (C-164, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 135
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0072 (C-165, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 136
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0073 (C-169, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 137
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0074 (C-170, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 138
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0075 (C-171, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 139
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0076 (C-173, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 140
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0077 (C-174, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 141
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0078 (C-176, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 142
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0079 (C-179, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 143
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0080 (C-180, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 144
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0081 (C-183, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 145
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0082 (C-185, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 146
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0083 (C-186, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 147
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0084 (C-187, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 148
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0085 (C-191, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 149
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0086 (C-195, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 150
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0087 (C-197, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 151
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0088 (C-201, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 152
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0089 (C-206, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 153
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0090 (C-207, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 154
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0091 (C-212, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 155
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0092 (C-217, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 156
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0093 (C-218, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 157
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0094 (C-219, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 158
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0095 (C-220, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 159
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0096 (C-221, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 160
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0097 (C-223, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 161
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0098 (C-228, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 162
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0099 (C-231, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 163
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0100 (C-234, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 164
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0101 (C-235, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 165
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0102 (C-239, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 166
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0103 (C-240, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 167
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0104 (C-241, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 168
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0105 (C-243, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 169
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0106 (C-244, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 170
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0107 (C-247, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 171
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0108 (C-248, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 172
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0109 (C-251, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 173
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0110 (C-253, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 174
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0111 (C-256, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 175
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0112 (C-257, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 176
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0113 (C-258, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 177
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0114 (C-259, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 178
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0115 (C-264, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 179
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0116 (C-268, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 180
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0117 (C-269, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 181
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0118 (C-271, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 182
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0119 (C-274, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 183
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0120 (C-278, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 184
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0121 (C-280, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 185
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0122 (C-281, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 186
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0123 (C-283, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 187
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0124 (C-284, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 188
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0125 (C-285, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 189
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0126 (C-286, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 190
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0127 (C-287, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 191
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0128 (C-288, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 192
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0129 (C-289, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 193
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0130 (C-290, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 194
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0131 (C-291, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 195
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0132 (C-300, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 196
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0133 (C-304, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 197
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0134 (C-306, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 198
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0135 (C-309, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 199
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0136 (C-314, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 200
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0137 (C-315, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 201
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0138 (C-316, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 202
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0139 (C-317, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 203
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0140 (C-318, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 204
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0141 (C-321, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 205
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0142 (C-324, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 206
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0143 (C-326, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 207
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0144 (C-329, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 208
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0145 (C-330, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 209
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0146 (C-333, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 210
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0147 (C-334, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 211
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0148 (C-335, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 212
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0149 (C-338, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 213
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0150 (C-339, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 214
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0151 (C-340, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 215
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0152 (C-341, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 216
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0153 (C-342, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 217
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0154 (C-343, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 218
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0155 (C-346, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 219
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0156 (C-349, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 220
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0157 (C-354, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 221
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0158 (C-355, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 222
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0159 (C-358, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 223
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0160 (C-362, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 224
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0161 (C-363, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 225
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0162 (C-366, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 226
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0163 (C-374, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 227
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0164 (C-375, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 228
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0165 (C-378, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 229
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0166 (C-379, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 230
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0167 (C-380, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 231
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0168 (C-381, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 232
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0169 (C-385, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 233
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0170 (C-386, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 234
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0171 (C-387, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 235
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0172 (C-390, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 236
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0173 (C-391, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 237
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0174 (C-394, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 238
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0175 (C-396, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 239
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0176 (C-399, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 240
- Status raw: DONE
- Canonical: `True`

Triage: add primary sources + define validation check (test or verifier) or mark as resolved with justification.

Output artifacts:
- `docs/CLAIMS_EVIDENCE_MATRIX.md`

### CTASK-0177 (C-401, PARTIAL)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 241
- Status raw: PARTIAL
- Canonical: `True`

Triage: primary sources cached; still needs a concrete, offline validation check (or keep as blueprint-only).

Output artifacts:
- `docs/external_sources/WARP_DRIVE_SOURCES.md`
- `data/external/papers/White_2021_Casimir_Warp.pdf`

### CTASK-0178 (C-403, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 242
- Status raw: DONE
- Canonical: `True`

Source-index + define offline check for spectral-triple-strength "geometry from spectrum" program claims.

Output artifacts:
- `docs/external_sources/EMERGENCE_LAYERS_SOURCES.md`
- `src/spectral/demo_pairs.py`
- `tests/test_isospectral_nonisomorphic_pair.py`
- `src/spectral_triple_toy.py`
- `tests/test_spectral_triple_toy.py`

### CTASK-0179 (C-404, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 243
- Status raw: DONE
- Canonical: `True`

Source-index + define offline check for modular-data/entanglement-wedge program claims.

Output artifacts:
- `docs/external_sources/EMERGENCE_LAYERS_SOURCES.md`
- `src/scripts/data/fetch_emergence_layers_sources.py`
- `data/external/papers/arxiv_hep-th0603001_ryu_takayanagi_2006_rt.pdf`
- `data/external/papers/arxiv_0705.0016_hubeny_rangamani_takayanagi_2007_hrt.pdf`
- `data/external/papers/arxiv_1512.06431_jafferis_lewkowycz_maldacena_suh_2016_jlms.pdf`
- `data/external/papers/arxiv_1609.00026_freedman_headrick_2016_bit_threads.pdf`
- `src/holography/maxflow.py`
- `tests/test_holography_bit_threads.py`

### CTASK-0180 (C-405, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 244
- Status raw: DONE
- Canonical: `True`

Source-index + define offline check for open-systems/QEC observer program claims.

Output artifacts:
- `docs/external_sources/EMERGENCE_LAYERS_SOURCES.md`
- `src/quantum/open_systems/lindblad.py`
- `tests/test_open_systems_lindblad.py`
- `src/quantum/open_systems/redundancy.py`
- `tests/test_open_systems_redundancy.py`

### CTASK-0181 (C-406, PARTIAL)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 245
- Status raw: PARTIAL
- Canonical: `True`

Enumerate embedding choices (symmetry proxy) and quantify invariance/trial factors for TSCP alignment.

Output artifacts:
- `docs/preregistered/TSCP_SKY_ALIGNMENT.md`
- `tests/test_tscp_embedding_sweep.py`
- `docs/external_sources/TSCP_METHOD_SOURCES.md`

### CTASK-0182 (C-407, PARTIAL)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 246
- Status raw: PARTIAL
- Canonical: `True`

Maintain a look-elsewhere parameter ledger + global correction bounds (Bonferroni/Holm) for TSCP-style searches.

Output artifacts:
- `reports/tscp_trial_factor_ledger.md`
- `tests/test_tscp_embedding_sweep.py`
- `docs/external_sources/TSCP_METHOD_SOURCES.md`

### CTASK-0183 (C-408, PARTIAL)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 247
- Status raw: PARTIAL
- Canonical: `True`

Add symmetric falsification thresholds (N_min + alpha/effect-size) for selected claims; enforce via tests/verifiers.

Output artifacts:
- `docs/preregistered/TSCP_SKY_ALIGNMENT.md`
- `docs/external_sources/TSCP_METHOD_SOURCES.md`

### CTASK-0184 (C-410, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 248
- Status raw: DONE
- Canonical: `True`

Photon-graviton mixing scope: Schwinger B_cr=4.41e9 T verified; mixing amplitude negligible for lab fields; C-402 NOT overturned. Phase 6 task B6.

Output artifacts:
- `src/scripts/analysis/c410_photon_graviton_scope.py`
- `tests/test_c410_photon_graviton_scope.py`
- `docs/BIBLIOGRAPHY.md`

### CTASK-0185 (C-411, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 249
- Status raw: DONE
- Canonical: `True`

SFWM thin-layer scaling: direct SFWM dominates 5.8x (phase-matching); coherence lengths match paper (33.3/3.1/3.4 um). Phase 6 task B6.

Output artifacts:
- `src/scripts/analysis/c411_sfwm_thin_layer_check.py`
- `tests/test_c411_sfwm_thin_layer.py`
- `docs/BIBLIOGRAPHY.md`

### CTASK-0186 (C-417, TODO)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 250
- Status raw: TODO
- Canonical: `True`

Turn "Holographic Entropy Trap" into a falsifiable metric + null; keep speculative until tied to optical-capture baselines and uncertainty.

Output artifacts:
- `src/scripts/analysis/sedenion_warp_synthesis.py`
- `data/artifacts/images/sedenion_capture_scaling.png`

### CTASK-0187 (C-432, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 251
- Status raw: DONE
- Canonical: `True`

Kerr geodesic solver + Bardeen analytic shadow boundary validated: Schwarzschild a=0 shadow radius sqrt(27) within 0.1%, photon orbit radii exact, impact parameters xi^2+eta=27, high-spin D-shape asymmetry, null geodesic escape/capture. Phase 6 task A9.

Output artifacts:
- `src/gemini_physics/gr/kerr_geodesic.py`
- `tests/test_kerr_shadow.py`

### CTASK-0188 (C-429, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 252
- Status raw: DONE
- Canonical: `True`

Kerr shadow asymmetry (a=0.99) validated: D-shape center offset > 0.1 from Schwarzschild symmetric limit. Phase 6 task A9.

Output artifacts:
- `src/gemini_physics/gr/kerr_geodesic.py`
- `tests/test_kerr_shadow.py::test_high_spin_shadow_asymmetric`

### CTASK-0189 (C-425, DONE)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 253
- Status raw: DONE
- Canonical: `True`

Octonionic (8D) field Hamiltonian formulation: Fano-plane multiplication, Stormer-Verlet symplectic integrator, 7 Noether charges, free-field dispersion omega^2=k^2+m^2. Restricts to octonionic subalgebra to bypass C-030 non-associativity obstruction. Phase 6 task A10.

Output artifacts:
- `crates/algebra_core/src/octonion_field.rs`
- `tests/test_octonion_field.py`

### CTASK-0190 (C-428, PARTIAL)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 254
- Status raw: PARTIAL
- Canonical: `True`

Kerr geodesic infrastructure from A9 provides Boyer-Lindquist integrator + Mino-time second-order Hamiltonian form. NegDimCosmology coupling still needs validation.

Output artifacts:
- `src/gemini_physics/gr/kerr_geodesic.py`
- `tests/test_kerr_shadow.py`

### CTASK-0191 (C-426, TODO)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 255
- Status raw: TODO
- Canonical: `True`

Add a preregistered fit protocol (null + trial-factor ledger) for mapping ZD eigenvalues to particle masses; keep as toy until it passes robust controls.

Output artifacts:
- `crates/algebra_core/src/hypercomplex.rs`
- `src/scripts/analysis/pathion_particle_fit.py`
- `tests/test_pathion_zd_diagonalization.py`

### CTASK-0192 (C-427, TODO)

- Section: Backfill (triage) tasks (auto-generated)
- Source line: 256
- Status raw: TODO
- Canonical: `True`

Add unit tests for algebraic-media tensor construction invariants (symmetry/normalization bounds) and define a minimal physical decision rule for the mapping.

Output artifacts:
- `src/gemini_physics/metamaterial.py`
- `src/scripts/analysis/unified_spacetime_synthesis.py`

### CTASK-0193 (C-028, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 262
- Status raw: DONE
- Canonical: `True`

Aut(S) = G2 x S3 verification (Phase 6A).

Output artifacts:
- `crates/algebra_core/src/hypercomplex.rs`
- `tests/test_sedenion_automorphism.py`

### CTASK-0194 (C-029, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 263
- Status raw: DONE
- Canonical: `True`

Three-generation literature review (Phase 6B).

Output artifacts:
- `crates/algebra_core/src/clifford.rs`
- `tests/test_sedenion_generations.py`

### CTASK-0195 (C-030, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 264
- Status raw: DONE
- Canonical: `True`

Decision rule + offline bypass checks (associator + alternativity) for sedenion-valued actions.

Output artifacts:
- `docs/C030_SEDENION_LAGRANGIAN_BYPASS.md`
- `src/scripts/analysis/c030_sedenion_lagrangian_bypass_checks.py`
- `data/csv/c030_sedenion_lagrangian_bypass_checks.csv`
- `tests/test_c030_sedenion_lagrangian_bypass_checks.py`

### CTASK-0196 (C-031, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 265
- Status raw: DONE
- Canonical: `True`

Hurwitz/norm-composition transition (1,2,4,8 vs 16) + zero-divisor example artifact + test.

Output artifacts:
- `docs/external_sources/C031_HURWITZ_QUANTIZATION_SOURCES.md`
- `src/scripts/analysis/c031_hurwitz_norm_composition_checks.py`
- `data/csv/c031_hurwitz_norm_composition_checks.csv`
- `tests/test_c031_hurwitz_norm_composition_checks.py`
- `docs/C030_SEDENION_LAGRANGIAN_BYPASS.md`

### CTASK-0197 (C-032, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 266
- Status raw: DONE
- Canonical: `True`

Tang (2025) non-associative QED: minimal offline reproduction (Table 2 extraction + subalgebra associator stats).

Output artifacts:
- `docs/external_sources/C032_TANG_2025_SEDENIONIC_QED_SOURCES.md`
- `data/external/traces/tang_2025_preprints_org_landing.txt`
- `data/external/papers/preprints202511.0427_v1_tang_2025_sedenionic_qed.txt`
- `src/scripts/analysis/c032_tang_2025_min_reproduction.py`
- `data/csv/c032_tang_2025_table2_lepton_masses.csv`
- `data/csv/c032_tang_2025_associator_subalgebra_summary.csv`
- `data/csv/c032_tang_2025_associator_basis_triples.csv`
- `tests/test_c032_tang_2025_min_reproduction.py`

### CTASK-0198 (C-032, BLOCKED)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 267
- Status raw: BLOCKED
- Canonical: `True`

Tang (2025) non-associative QED: mechanize the associator->mass mapping (BLOCKED until the source provides a complete, convention-fixed mapping and scale choice).

Output artifacts:
- `docs/external_sources/C032_TANG_2025_SEDENIONIC_QED_SOURCES.md`
- `data/external/traces/tang_2025_preprints_org_landing.txt`

### CTASK-0199 (C-033, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 268
- Status raw: DONE
- Canonical: `True`

SU(5) generator basis verification complete; source does not specify a unique coefficient mapping from sedenion basis to a normalized SU(5) basis (claim demoted accordingly).

Output artifacts:
- `data/external/papers/arxiv_2308.14768_tang_tang_2023_sedenion_su5_generations.pdf`
- `docs/external_sources/SEDENION_FIELD_THEORY_SOURCES.md`
- `docs/C033_SU5_MAPPING_CLOSURE.md`
- `crates/algebra_core/src/group_theory.rs`
- `src/scripts/analysis/c033_su5_generator_summary.py`
- `data/csv/c033_su5_generator_summary.csv`
- `tests/test_su5_generators.py`

### CTASK-0200 (C-034, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 269
- Status raw: DONE
- Canonical: `True`

Chanyal (2014) sedenion gravi-electromagnetism: minimal structural reproduction (two 8D sectors via CD doubling).

Output artifacts:
- `docs/external_sources/C034_CHANYAL_2014_GRAVI_ELECTROMAGNETISM_SOURCES.md`
- `data/external/traces/chanyal_2014_springer_landing.txt`
- `data/external/traces/chanyal_2014_springer_abstract.txt`
- `docs/C034_CHANYAL_2014_REPRODUCTION.md`
- `src/scripts/analysis/c034_chanyal_2014_structural_reproduction.py`
- `data/csv/c034_sedenion_doubling_identity_check.csv`
- `tests/test_c034_sedenion_doubling_identity.py`

### CTASK-0201 (C-034, BLOCKED)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 270
- Status raw: BLOCKED
- Canonical: `True`

Chanyal (2014) sedenion gravi-electromagnetism: equation-level reproduction checks (BLOCKED until a legal full-text source is cached).

Output artifacts:
- `docs/external_sources/C034_CHANYAL_2014_GRAVI_ELECTROMAGNETISM_SOURCES.md`
- `data/external/traces/chanyal_2014_springer_landing.txt`
- `data/external/traces/chanyal_2014_springer_abstract.txt`

### CTASK-0202 (C-035, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 271
- Status raw: DONE
- Canonical: `True`

F4 Casimir epsilon = 1/4 (Phase 7A).

Output artifacts:
- `crates/quantum_core/src/casimir.rs`
- `tests/test_f4_casimir.py`

### CTASK-0203 (C-036, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 272
- Status raw: DONE
- Canonical: `True`

Bigraph cosmogenesis simulation (Phase 7B).

Output artifacts:
- `crates/cosmology_core/src/spectral.rs`
- `tests/test_bigraph_cosmogenesis.py`

### CTASK-0204 (C-037, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 273
- Status raw: DONE
- Canonical: `True`

Demoted coincidence claim to Not supported; audit note defines mechanism gap and falsification requirements.

Output artifacts:
- `docs/C037_NUMERICAL_COINCIDENCE_AUDIT.md`
- `docs/EXCEPTIONAL_COSMOLOGY.md`

### CTASK-0205 (C-038, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 274
- Status raw: DONE
- Canonical: `True`

w0=-5/6 observational test (Phase 7C). DISFAVORED.

Output artifacts:
- `crates/cosmology_core/src/bounce.rs`
- `tests/test_exceptional_w0.py`

### CTASK-0206 (C-039, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 275
- Status raw: DONE
- Canonical: `True`

Spectral dimension running on bigraph (Phase 7D). Qualitative consistency with CDT.

Output artifacts:
- `crates/cosmology_core/src/spectral.rs`
- `src/scripts/analysis/c039_spectral_dimension_bigraph_sweep.py`
- `data/csv/c039_spectral_dimension_bigraph_curve.csv`
- `data/csv/c039_spectral_dimension_bigraph_summary.csv`
- `tests/test_spectral_dimension.py`
- `tests/test_c039_spectral_dimension_bigraph_artifacts.py`

### CTASK-0207 (C-040, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 276
- Status raw: DONE
- Canonical: `True`

Primordial tilt n_s comparison (Phase 7E). Post-hoc; D_eff=2.8-3.0 inconsistent with Planck via Calcagni formula.

Output artifacts:
- `crates/cosmology_core/src/spectral.rs`
- `tests/test_primordial_tilt.py`

### CTASK-0208 (C-041, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 277
- Status raw: DONE
- Canonical: `True`

Demote dimensional coincidence claim; record mechanism gap and decision rule.

Output artifacts:
- `docs/C041_F4_STRING_DIMENSION_COINCIDENCE_AUDIT.md`
- `docs/EXCEPTIONAL_COSMOLOGY.md`

### CTASK-0209 (C-042, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 278
- Status raw: DONE
- Canonical: `True`

Kozyrev p-adic wavelets: implement eigenbasis for Vladimirov operator + offline tests.

Output artifacts:
- `docs/theory/PADIC_ANALYSIS_FOUNDATIONS.md`
- `crates/algebra_core/src/padic.rs`
- `tests/test_padic_wavelets.py`

### CTASK-0210 (C-043, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 279
- Status raw: DONE
- Canonical: `True`

Compact object integration pipeline + unified catalog artifact + offline schema test.

Output artifacts:
- `docs/external_sources/C043_COMPACT_OBJECT_CATALOG_SOURCES.md`
- `src/scripts/data/fetch_compact_objects.py`
- `data/csv/compact_objects_catalog.csv`
- `data/csv/compact_objects_catalog.PROVENANCE.json`
- `tests/test_c043_compact_objects_catalog_artifact.py`

### CTASK-0211 (C-044, REFUTED)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 280
- Status raw: REFUTED
- Canonical: `True`

Legacy zero-divisor adjacency matrices refuted; keep reproduction script as guard.

Output artifacts:
- `data/csv/legacy/`
- `src/scripts/reproduction/reproduce_zd_adjacency.py`
- `docs/LEGACY_ARTIFACT_AUDIT.md`

### CTASK-0212 (C-045, DONE)

- Section: Sedenion Field Theory and Exceptional Cosmology (Phases 6-7)
- Source line: 281
- Status raw: DONE
- Canonical: `True`

Strang splitting convergence demo + validation smoke check.

Output artifacts:
- `examples/strang_splitting_demo.py`
- `src/scripts/validation/validate_strang_splitting_convergence.py`
- `tests/test_strang_splitting_demo.py`
