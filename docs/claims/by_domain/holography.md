# Claims: holography

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/claims_domains.toml -->

Count: 135

- Hypothesis C-001 (Verified (math), 2026-02-02): Cayley-Dickson algebras become non-associative at 8D and beyond.
  - Where stated: `src/verification/verify_algebra.py`, `tests/test_cayley_dickson_properties.py`, `docs/external_sources/CAYLEY_DICKSON_BASICS_SOURCES.md`

- Hypothesis C-002 (Verified (math), 2026-02-02): 16D sedenions have zero divisors and lose norm composition.
  - Where stated: `docs/THE_GEMINI_PROTOCOL.md`, `tests/test_cayley_dickson_properties.py`, `docs/external_sources/DE_MARRAIS_SOURCES.md`, `docs/external_sources/CAYLEY_DICKSON_BASICS_SOURCES.md`

- Hypothesis C-007 (Speculative (/ suggestive but fragile (Phase 2D Bayesian mixture)), 2026-02-02): GWTC-3 BH mass distribution is suggestive of multimodality (Phase 2D mixture modeling); any link to \"negative dimension eigenmodes\" remains a speculative interpretation.
  - Where stated: `docs/archive/RESEARCH_STATUS.md`, `docs/external_sources/GWTC3_MASS_CLUMPING_PLAN.md`, `docs/external_sources/GWTC3_DECISION_RULE_SUMMARY.md`, `docs/external_sources/GWTC3_POPULATION_SOURCES.md`, `docs/external_sources/GWTC3_SELECTION_FUNCTION_SPEC.md`, `crates/stats_core/src/claims_gates.rs`, `src/scripts/analysis/gwtc3_mass_clumping_null_models.py`, `src/scripts/analysis/gwtc3_mass_clumping_bootstrap.py`, `src/scripts/analysis/gwtc3_mass_clumping_decision_rule_summary.py`, `src/scripts/analysis/gwtc3_selection_weight_sweep.py`, `src/scripts/data/convert_gwtc3_injections_hdf.py`, `src/scripts/analysis/gwtc3_selection_function_from_injections.py`, `src/scripts/analysis/gwtc3_modality_preregistered.py`, `src/scripts/analysis/gwtc3_bayesian_mixture.py`, `data/external/gwtc3_injection_summary.csv`, `data/external/gwtc3_injection_summary_7890398.csv`, `data/csv/gwtc3_selection_function_binned.csv`, `data/csv/gwtc3_selection_function_binned_7890398.csv`, `data/csv/gwtc3_selection_function_binned_combined.csv`, `data/csv/gwtc3_selection_function_binned_combined_altbins.csv`, `data/csv/gwtc3_mass_clumping_metrics.csv`, `data/csv/gwtc3_mass_clumping_null_models.csv`, `data/csv/gwtc3_mass_clumping_bootstrap_counts.csv`, `data/csv/gwtc3_mass_clumping_bootstrap_summary.csv`, `data/csv/gwtc3_mass_clumping_decision_rule_summary.csv`, `data/csv/gwtc3_selection_bias_control_metrics.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o123.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o3a_altbins.csv`, `data/csv/gwtc3_selection_bias_control_metrics_o123_altbins.csv`, `data/csv/gwtc3_selection_weight_sweep.csv`, `data/csv/gwtc3_selection_weight_sweep_o123.csv`, `data/csv/gwtc3_selection_weight_sweep_o3a_altbins.csv`, `data/csv/gwtc3_selection_weight_sweep_o123_altbins.csv`, `data/csv/gwtc3_bayesian_mixture_results.csv`, `docs/preregistered/GWTC3_MODALITY_TEST.md`, `docs/preregistered/GWTC3_BAYESIAN_MIXTURE.md`

- Hypothesis C-010 (Speculative (spectral comparison negative; Phase 3C), 2026-02-02): Hypothesis: Cayley-Dickson (sedenion) zero-divisor motifs can be mapped to physically realizable absorber networks (e.g., TCMT/RLC/CPA), potentially approaching critical coupling; no "perfect absorption" mapping is validated yet.
  - Where stated: `docs/MATERIALS_APPLICATIONS.md`, `docs/THE_GEMINI_PROTOCOL.md`, `docs/external_sources/METAMATERIAL_ABSORBER_SOURCES.md`, `docs/C010_ABSORBER_TCMT_MAPPING.md`, `docs/C010_ABSORBER_SALISBURY.md`, `docs/C010_ABSORBER_RLC.md`, `docs/C010_ABSORBER_CPA.md`, `src/scripts/analysis/materials_absorber_tcmt.py`, `src/scripts/analysis/materials_absorber_salisbury.py`, `src/scripts/analysis/materials_absorber_rlc.py`, `src/scripts/analysis/materials_absorber_cpa_twoport.py`, `src/scripts/analysis/materials_absorber_cpa_input_sweep.py`, `data/csv/c010_truncated_icosahedron_graph.csv`, `data/csv/c010_tcmt_truncated_icosahedron.csv`, `data/csv/c010_tcmt_truncated_icosahedron_minima.csv`, `data/csv/c010_salisbury_screen.csv`, `data/csv/c010_salisbury_screen_minima.csv`, `data/csv/c010_rlc_surface_impedance.csv`, `data/csv/c010_rlc_surface_impedance_minima.csv`, `data/csv/c010_rlc_fit_summary.csv`, `data/csv/c010_cpa_twoport_scan.csv`, `data/csv/c010_cpa_twoport_minima.csv`, `data/csv/c010_cpa_input_sweep.csv`, `data/csv/c010_cpa_input_sweep_minima.csv`, `data/csv/c010_zd_tcmt_spectral_comparison.csv`, `data/csv/materials_baseline_metrics.csv`, `data/csv/materials_embedding_benchmarks.csv`

- Hypothesis C-018 (Verified (source + tests), 2026-01-28): "Wheels" (Carlstrom) are commutative monoid-based structures <H,0,1,+,*,/> with a total reciprocal operation and defining axioms (1)-(8) (Carlstrom 2001, Definition 1.1) that provide division-by-zero semantics without partiality.
  - Where stated: `docs/WHEELS_DIVISION_BY_ZERO.md`

- Hypothesis C-022 (Modeled (Toy; Phase 4A ordinal mapping), 2026-02-02): Toy model: map Cayley-Dickson doubling level n to surreal birthday n and test the CD property-loss milestones.
  - Where stated: `docs/theory/unified_tensor_wheel_cd_framework.md`, `src/scripts/analysis/surreal_cd_ordinal_construction.py`, `data/csv/surreal_cd_ordinal_mapping.csv`, `tests/test_surreal_cd_ordinal.py`

- Hypothesis C-025 (Refuted (Phase 5), 2026-01-31): GWTC-3 black hole sky positions cluster around projected sedenion zero-divisor coordinates (CMB-aligned box-kite projection).
  - Where stated: `docs/stellar_cartography/theory/HYPOTHESIS_DEF.md`, `crates/algebra_core/src/boxkites.rs`, `crates/stats_core/src/claims_gates.rs`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `tests/test_tscp_alignment_offline.py`, `docs/external_sources/TSCP_SKY_ALIGNMENT_SOURCES.md`

- Hypothesis C-026 (Speculative (mechanism missing; baseline metrics only), 2026-02-03): Speculative program: if a statistically significant "lower mass gap" (~2.5-5 M_sun) is established, it may correspond to an "algebraic tension" region between stable zero-divisor nodes in the sedenion box-kite structure.
  - Where stated: `docs/stellar_cartography/theory/HYPOTHESIS_DEF.md`, `docs/C026_MASS_GAP_MECHANISM.md`, `docs/external_sources/C026_MASS_GAP_SOURCES.md`, `src/scripts/analysis/gwtc3_lower_mass_gap_metrics.py`, `data/csv/gwtc3_lower_mass_gap_metrics.csv`

- Hypothesis C-027 (Verified (Toy Model; decision rule implemented), 2026-02-03): Toy model: define D_eff(rho) = 3 - k*log10(rho/rho_vac). With rho_h(M) = M/((4/3)*pi*r_s^3) at r_s = 2GM/c^2 (average density inside Schwarzschild radius), D_eff(rho_h(M)) crosses 0 at a critical mass M_crit (horizon "phase transition" proxy).
  - Where stated: `docs/stellar_cartography/theory/ONTOLOGICAL_AXIOMS.md`, `crates/algebra_core/src/grassmannian.rs`, `crates/stats_core/src/claims_gates.rs`, `docs/C027_DEFF_HORIZON_TEST.md`, `src/scripts/analysis/deff_horizon_mass_scaling.py`, `data/csv/deff_horizon_mass_scaling_summary.csv`, `tests/test_c027_deff_horizon_mass_scaling.py`, `docs/external_sources/C027_EFFECTIVE_DIMENSION_SOURCES.md`

- Hypothesis C-031 (Verified (math + artifact; QFT caveat), 2026-02-03): By Hurwitz theorem, only dims 1/2/4/8 admit normed division algebras (R,C,H,O). In the CD tower at dim=16 (sedenions), norm composition fails and zero divisors exist; therefore any sedenion-valued field theory must specify an associative representation (or restrict to associative subalgebras) to define a unique action and a compatible Hilbert-space structure.
  - Where stated: `docs/convos/pdf_extract_3f6ee1e837d1_sedenion_valued_field_theories_action_principles_and_challenges.md`, `docs/external_sources/C031_HURWITZ_QUANTIZATION_SOURCES.md`, `src/scripts/analysis/c031_hurwitz_norm_composition_checks.py`, `data/csv/c031_hurwitz_norm_composition_checks.csv`, `tests/test_c031_hurwitz_norm_composition_checks.py`, `docs/C030_SEDENION_LAGRANGIAN_BYPASS.md`, `docs/SEDENION_FIELD_THEORY.md`

- Hypothesis C-039 (Verified (sources + toy implementation; finite-N caveats), 2026-02-04): In CDT and asymptotic safety literature, spectral dimension D_s runs from ~4 (large scales) to ~2 (short scales); the repo implements a finite-graph toy D_s(t) computation for qualitative comparison.
  - Where stated: `docs/convos/pdf_extract_2b693d92f57d_exceptional_cosmological_framework_synthesis.md`, `docs/external_sources/EXCEPTIONAL_COSMOLOGY_SOURCES.md`, `crates/cosmology_core/src/spectral.rs`, `tests/test_spectral_dimension.py`, `src/scripts/analysis/c039_spectral_dimension_bigraph_sweep.py`, `data/csv/c039_spectral_dimension_bigraph_curve.csv`, `data/csv/c039_spectral_dimension_bigraph_summary.csv`, `tests/test_c039_spectral_dimension_bigraph_artifacts.py`

- Hypothesis C-040 (Refuted (tested mismatch vs Planck under stated mapping), 2026-02-03): Primordial tilt n_s ~ 0.965 from fractal D_eff ~ 2.8-3.0 at inflation.
  - Where stated: `docs/convos/pdf_extract_2b693d92f57d_exceptional_cosmological_framework_synthesis.md`, `docs/external_sources/EXCEPTIONAL_COSMOLOGY_SOURCES.md`, `docs/EXCEPTIONAL_COSMOLOGY.md`, `crates/cosmology_core/src/spectral.rs`, `tests/test_primordial_tilt.py`, `src/scripts/analysis/c040_primordial_tilt_deff_sweep.py`, `data/csv/c040_primordial_tilt_deff_curve.csv`, `data/csv/c040_primordial_tilt_summary.csv`, `tests/test_c040_primordial_tilt_artifacts.py`

- Hypothesis C-043 (Verified (integration pipeline; data sources may be synthetic/curated), 2026-02-04): "Compact Object" populations (Pulsars, Magnetars, FRBs) can be integrated into the unified framework for multi-messenger testing.
  - Where stated: `docs/ROADMAP_DETAILED.md`, `docs/external_sources/C043_COMPACT_OBJECT_CATALOG_SOURCES.md`, `src/scripts/data/fetch_chime_frb.py`, `src/scripts/data/fetch_atnf_pulsars_full.py`, `src/scripts/data/fetch_mcgill_magnetars.py`, `src/scripts/data/fetch_compact_objects.py`, `data/csv/compact_objects_catalog.csv`, `data/csv/compact_objects_catalog.PROVENANCE.json`, `tests/test_c043_compact_objects_catalog_artifact.py`

- Hypothesis C-047 (Refuted (Kac-Moody), 2026-02-04): E9, E10, E11 are Euclidean sphere-packing lattices.
  - Where stated: `archive/legacy_conjectures/`, `docs/C047_E_SERIES_KAC_MOODY_AUDIT.md`, `docs/external_sources/C047_E_SERIES_KAC_MOODY_SOURCES.md`, `tests/test_c047_e_series_cartan_signature.py`

- Hypothesis C-052 (Verified (Simulation), 2026-01-31): MERA (Multi-scale Entanglement Renormalization) circuit produces logarithmic entropy scaling S ~ log(L).
  - Where stated: `src/scripts/analysis/verify_phase_3_tasks.py`

- Hypothesis C-056 (Verified (data), 2026-01-31): PDG lepton/boson masses verified against experiment (electron, muon, Z, W, Higgs).
  - Where stated: `src/scripts/data/fetch_pdg_particle_data.py`, `tests/test_pdg_particle_data.py`, `tests/test_pdg_coupling_constants.py`

- Hypothesis C-068 (Refuted (degenerate spectrum), 2026-02-03): Sedenion 84-ZD interaction matrix eigenvalue spectrum matches PDG particle masses.
  - Where stated: `src/scripts/analysis/c068_zd_interaction_spectrum_degeneracy.py`, `data/csv/c068_zd_interaction_eigen_summary.csv`, `tests/test_c068_zd_interaction_spectrum_degeneracy.py`, `docs/external_sources/C068_PDG_SPECTRUM_MATCH_SOURCES.md`, `data/csv/reggiani_standard_zero_divisors.csv`

- Hypothesis C-079 (Not supported (rejected (rejected)), 2026-01-31): E8 root system eigenvalues reproduce particle masses.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v4.py`, `data/csv/cd_algebraic_experiments_v4.json`

- Hypothesis C-091 (Not supported (rejected (rejected)), 2026-01-31): Non-diagonal ZD spectrum matches 9/12 particle masses.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v6.py`, `data/csv/cd_algebraic_experiments_v6.json`, `data/csv/cd_algebraic_experiments_v7_null.json`

- Hypothesis C-093 (Verified, 2026-01-31): Algebraic Gram matrix Tr(L_i^T L_j) is proportional to identity.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v6.py`, `data/csv/cd_algebraic_experiments_v6.json`

- Hypothesis C-097 (Verified (Math/Code), 2026-02-03): Diagonal ZD interaction graph has exactly 3 distinct edge weights {0, 1, sqrt(2)} and decomposes into 14 connected components at threshold > 1.
  - Where stated: `src/scripts/analysis/c097_zd_interaction_graph_audit.py`, `data/csv/c097_zd_interaction_graph_summary.csv`, `tests/test_c097_zd_interaction_graph_audit.py`, `data/csv/reggiani_standard_zero_divisors.csv`

- Hypothesis C-098 (Verified, 2026-01-31): CD algebras lose algebraic properties at precise, dimension-specific thresholds: commutativity at dim=4, associativity at dim=8, alternativity at dim=16. Power-associativity and flexibility hold through dim=256.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v8.py`, `data/csv/cd_algebraic_experiments_v8.json`

- Hypothesis C-107 (Verified, 2026-01-31): Flexibility identity A(x,y,x)=0 holds to machine precision through dim=2048, with max deviations scaling as O(epsilon_mach * sqrt(dim)).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v10.py`, `data/csv/cd_algebraic_experiments_v10.json`

- Hypothesis C-111 (Verified, 2026-01-31): Complete 13-dimension CD property table (dim=2 through 8192): flexibility and power-associativity are exactly zero at ALL dimensions; Moufang identity breaks at dim=16 simultaneously with alternativity; commutator, associator, and Moufang norms saturate by dim~64.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v10.py`, `data/csv/cd_algebraic_experiments_v10.json`

- Hypothesis C-115 (Verified, 2026-01-31): The commutator [a,b] and associator A(a,b,c) are asymptotically orthogonal in high-dimensional CD algebras. The perpendicular component of A dominates (>99.7% at dim >= 64).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v11.py`, `data/csv/cd_algebraic_experiments_v11.json`

- Hypothesis C-125 (Verified, 2026-01-31): Artin's theorem (2-generated subalgebras are associative) holds through dim=8 and fails at dim=16. However, flexibility within 2-generated subalgebras (x(yx) = (xy)x) holds at ALL dims through 256 to machine precision.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v13.py`, `data/csv/cd_algebraic_experiments_v13.json`

- Hypothesis C-129 (Verified (cached v13 expBA), 2026-02-04): The associator norm distribution concentrates as dim grows (CV -> 0.01). The distribution has slight positive skew (+0.3) and excess kurtosis trending toward 0 (Gaussian). Mean norm stabilizes at ~1.4.
  - Where stated: `src/scripts/analysis/c129_associator_distribution_concentration_audit.py`, `data/csv/c129_assoc_norm_dist_summary.csv`, `data/csv/c129_assoc_norm_dist_by_dim.csv`, `tests/test_c129_associator_distribution_concentration_audit.py`, `data/csv/cd_algebraic_experiments_v13.json`, `docs/external_sources/OPEN_CLAIMS_SOURCES.md`

- Hypothesis C-130 (Verified (Monte Carlo; cached v14 expBB), 2026-02-03): The associator norm\|\|A(a,b,c)\|\|-> sqrt(2) because (ab)c and a(bc) become uncorrelated (cos -> 0) while both preserve unit norm (\|\|product\|\|-> 1).\|\|A\|\|^2 = 2 - 2*cos -> 2.
  - Where stated: `src/scripts/analysis/c130_associator_norm_sqrt2_audit.py`, `data/csv/c130_associator_norm_sqrt2_summary.csv`, `data/csv/c130_associator_norm_sqrt2_by_dim.csv`, `tests/test_c130_associator_norm_sqrt2_audit.py`, `data/csv/cd_algebraic_experiments_v14.json`, `docs/external_sources/OPEN_CLAIMS_SOURCES.md`

- Hypothesis C-132 (Verified (cached v14 expBD), 2026-02-04): The commutator norm\|\|[a,b]\|\|^2 -> 4.01 (\|\|[a,b]\|\|-> 2.0). Commutator is zero at dim=2 (commutative) and nonzero starting at dim=4. The limit satisfies\|\|[a,b]\|\|^2 = 2 *\|\|A\|\|^2 = 2 * 2 = 4.
  - Where stated: `src/scripts/analysis/c132_commutator_norm_convergence_audit.py`, `data/csv/c132_commutator_norm_summary.csv`, `data/csv/c132_commutator_norm_by_dim.csv`, `tests/test_c132_commutator_norm_convergence_audit.py`, `data/csv/cd_algebraic_experiments_v14.json`, `docs/external_sources/OPEN_CLAIMS_SOURCES.md`

- Hypothesis C-133 (Verified, 2026-01-31): The Moufang defect and associator are asymptotically orthogonal (cos -> 0, perp_frac -> 1.0). This parallels the commutator-associator orthogonality (C-115).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v14.py`, `data/csv/cd_algebraic_experiments_v14.json`

- Hypothesis C-137 (Verified (math), 2026-02-01): ZD products at dim=32 preserve the norm trichotomy {0, 1, sqrt(2)} from dim=16, but kernel diversity increases: kernels vary over {4, 8, 12, 16} instead of a uniform kernel=4. 14 valid ZDs yield 36 pairs: 2 zero (6%), 26 ZD (72%), 8 non-ZD (22%).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v15.py`, `data/csv/cd_algebraic_experiments_v15.json`

- Hypothesis C-138 (Verified (math), 2026-02-01): 3-generated subalgebras become non-associative at dim=8 (octonions), while 2-generated subalgebras (Artin's theorem) fail at dim=16. A(x,y,xz)/A(x,y,z) ratio ~ 1.0, indicating mixed 3-gen associators are comparable to pure 3-gen.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v15.py`, `data/csv/cd_algebraic_experiments_v15.json`

- Hypothesis C-145 (Verified (math), 2026-02-01): Four-element products have exactly 5 distinct bracketings. At dim=4 (associative), all are identical (0 distinct pairs). At dim=8+, ALL 10 pairwise distances are non-zero. Product norms all near 1.0. Pairwise distances converge to sqrt(2) at high dim (uncorrelated unit vectors).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v16.py`, `data/csv/cd_algebraic_experiments_v16.json`

- Hypothesis C-147 (Verified (math), 2026-02-01): Alternator-associator decomposition: A(a,b,c) splits into alternating part Alt = A(a,b,c) - A(b,a,c) and symmetric part Sym = A(a,b,c) + A(b,a,c). At dim=8 (alternative algebras), Sym = 0 exactly (Alt^2/A^2 = 4). At high dim, Alt^2/A^2 -> 3 and Sym^2/A^2 -> 1. Alt and Sym are nearly orthogonal at all dims. Pythagorean identity holds exactly.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v16.py`, `data/csv/cd_algebraic_experiments_v16.json`

- Hypothesis C-149 (Verified (math), 2026-02-01): Composition defect delta =\|\|xy\|\|^2 -\|\|x\|\|^2*\|\|y\|\|^2 is identically zero at dim <= 8 (Hurwitz theorem). At dim >= 16, delta is zero-mean (\|E[delta]\|< 0.02) with std decreasing monotonically from 0.18 (dim=16) to 0.06 (dim=512). Skewness and kurtosis remain small.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v17.py`, `data/csv/cd_algebraic_experiments_v17.json`

- Hypothesis C-150 (Verified (math), 2026-02-01): Quadruple associator A(a,b,c,d) has\|\|A(a,b,cd)\|\|~\|\|A(a,b,c)\|\|(ratio 0.8-1.2) at all dims. The derivation-like fraction converges to ~1.45 and the nested/simple ratio converges to ~1.41 (near sqrt(2)). Both ratios lie in (1.0, 2.0).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v17.py`, `data/csv/cd_algebraic_experiments_v17.json`

- Hypothesis C-151 (Verified (math), 2026-02-01): At dim=64, ALL 780 pairwise products of 40 diagonal-form ZDs produce non-ZD elements with norm exactly 1.0. No zero products and no sqrt(2)-norm products exist. The norm spectrum is {1.0} only. This contrasts with dim=16 and dim=32 where {0, 1, sqrt(2)} all appear.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v17.py`, `data/csv/cd_algebraic_experiments_v17.json`

- Hypothesis C-154 (Verified (math), 2026-02-01): Artin's theorem: any 2-generated subalgebra is associative at dim=4 and dim=8 (alternative algebras). Fails at dim >= 16 with max associator norm > 1.0. Mean associator norm grows with dim beyond the Hurwitz boundary.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v18.py`, `data/csv/cd_algebraic_experiments_v18.json`

- Hypothesis C-158 (Verified (math), 2026-02-01): Idempotent structure: the only idempotent in CD algebras is e_0 (the identity element). No nontrivial idempotents (e^2 = e, e != 0, e != e_0) found via cubic iteration (3x^2 - 2x^3) from 200+ random starting points at each dim. (1/2)*e_0 has residual 0.25; (e_0 + e_i)/2 has residual 0.5.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v18.py`, `data/csv/cd_algebraic_experiments_v18.json`

- Hypothesis C-168 (Verified (math), 2026-02-01): ZD product spectrum at dim=128: among the first 100 diagonal-form (e_i+e_j)/sqrt(2) candidates, ZERO are actual zero divisors. This suggests the ZD structure becomes dramatically sparser at dim=128, or requires different index pairs.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v20.py`, `data/csv/cd_algebraic_experiments_v20.json`

- Hypothesis C-170 (Verified (math), 2026-02-01): Associator norm\|\|A(a,b,c)\|\|for random unit vectors converges to sqrt(2) from below as dim increases. Deviation from sqrt(2): 0.30 (dim=8), 0.098 (dim=16), 0.031 (dim=32), 0.005 (dim=128), 0.010 (dim=1024). Standard deviation decreases from 0.33 to 0.04.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v20.py`, `data/csv/cd_algebraic_experiments_v20.json`

- Hypothesis C-175 (Verified (math), 2026-02-01): Associator subspace Assoc(A) = span{A(a,b,c)} has rank = dim - 1 (= pure imaginary part) at all tested dims (8-64). It is NOT a two-sided ideal: x*w for x in A and w in Assoc(A) has 10-29% residual outside Assoc(A). Left and right residuals are equal. Residual decreases with dimension.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v21.py`, `data/csv/cd_algebraic_experiments_v21.json`

- Hypothesis C-180 (Verified (math), 2026-02-01): Inner product preservation <ax,ay> =\|\|a\|\|^2 <x,y> holds EXACTLY at composition dimensions (dim <= 8) and FAILS at dim >= 16. Mean deviation decreases with dim: 0.106 (dim=16), 0.092 (dim=64), 0.062 (dim=128). This is equivalent to the norm composition property.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v22.py`, `data/csv/cd_algebraic_experiments_v22.json`

- Hypothesis C-186 (Verified (math), 2026-02-01): Nested associator (triassociator):\|\|A(A(a,b,c),d,e)\|\|/\|\|A(a,b,c)\|\|~ 1.2-1.5 at all tested dims (8-256). The nesting amplification is bounded (no explosion).\|\|A2\|\|approaches 2.0 at high dim (consistent with A2 being an associator of a "random" vector near sqrt(2) norm).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v23.py`, `data/csv/cd_algebraic_experiments_v23.json`

- Hypothesis C-189 (Verified (math), 2026-02-01): Associator map T_{a,b}: c -> A(a,b,c) has purely imaginary eigenvalues (skew-symmetric) at dim <= 8 (alternative algebras). At dim >= 16, real eigenvalue parts appear (mean 0.12 at dim=16, 0.22 at dim=64). Rank: 0 at dim=4, 4 at dim=8, dim-2 at dim >= 16.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v23.py`, `data/csv/cd_algebraic_experiments_v23.json`

- Hypothesis C-192 (Verified (convention), 2026-02-01): CD doubling formula: our cd_multiply_batch does NOT use the standard Cayley-Dickson doubling formula (a,b)(c,d) = (ac - d*conj(b), conj(a)*d + cb). The deviation is O(1) at all dims (mean 0.39-1.14). This is a documented convention difference, not a bug: all other algebraic properties (conjugate reversal, norm composition at dim<=8, alternativity, flexibility) are verified correct.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v24.py`, `data/csv/cd_algebraic_experiments_v24.json`

- Hypothesis C-193 (Verified (math), 2026-02-01): Conjugate reversal conj(ab) = conj(b)*conj(a) holds EXACTLY (max_diff = 0.0) at ALL CD dimensions from 2 through 512. This is an exact algebraic identity, not an approximation. The anti-homomorphism property of conjugation is universal across all CD algebras regardless of associativity.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v24.py`, `data/csv/cd_algebraic_experiments_v24.json`

- Hypothesis C-196 (Verified (math), 2026-02-01): Artin's theorem: the subalgebra generated by any two elements {a, b, ab, ba, ...} is associative at dim <= 8 (alternative algebras). All four tested associator combinations A(a,b,ab), A(a,b,ba), A(ab,a,b), A(a,ab,b) are exactly zero. Fails at dim >= 16 with growing defect.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v25.py`, `data/csv/cd_algebraic_experiments_v25.json`

- Hypothesis C-197 (Verified (math), 2026-02-01): Associator norm scaling: mean\|\|A(a,b,c)\|\|= 0 at dim<=4 (associative), ~1.088 at dim=8, and converges to sqrt(2) ~ 1.414 at high dim (1.415 at dim=512). Standard deviation decreases as O(1/sqrt(dim)): 0.32 (dim=8), 0.064 (dim=512). The sqrt(2) limit is consistent with the associator of two "random" unit vectors in high dim.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v25.py`, `data/csv/cd_algebraic_experiments_v25.json`

- Hypothesis C-199 (Verified (math), 2026-02-01): Left/right alternative laws a(ab) = (aa)b and (ba)a = b(aa) hold EXACTLY at dim <= 8. Both fail symmetrically at dim >= 16 (left mean = right mean to within 1%). Left inverse property a^{-1}(ab) = b also holds exactly at dim <= 8 and fails at dim >= 16 with the same defect magnitude as the alternative laws.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v25.py`, `data/csv/cd_algebraic_experiments_v25.json`

- Hypothesis C-206 (Verified (math), 2026-02-01): Product commutativity defect\|\|ab-ba\|\|/\|\|ab\|\|= 0 at dim=2 (commutative). At dim>=4: approaches 2.0 monotonically: 1.14 (dim=4), 1.60 (dim=8), 1.80 (dim=16), 1.95 (dim=64), 1.99 (dim=512). Std decreases as O(1/sqrt(dim)): 0.46 (dim=4), 0.003 (dim=512). Verifies C-176 from a normalized perspective.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v26.py`, `data/csv/cd_algebraic_experiments_v26.json`

- Hypothesis C-209 (Verified (math), 2026-02-01): Associator norm distribution: at dim=8, slightly platykurtic (kurt=-0.63, negatively skewed). At high dim, excess kurtosis oscillates but trends toward 0. Mean -> sqrt(2). Std decreases as O(1/sqrt(dim)). The distribution becomes approximately Gaussian at high dim via CLT.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v27.py`, `data/csv/cd_algebraic_experiments_v27.json`

- Hypothesis C-212 (Verified (math), 2026-02-01): Cascade associator norms: A1 = A(a,b,c), A2 = A(A1,d,e), A3 = A(A2,f,g). Ratios\|\|A2\|\|/\|\|A1\|\|~ 1.2-1.4 and\|\|A3\|\|/\|\|A2\|\|~ 1.2-1.4 at all dims. No explosion.\|\|A1\|\|-> sqrt(2) confirming C-197. Higher-depth associators amplify but remain bounded, approaching geometric growth with ratio ~sqrt(2).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v27.py`, `data/csv/cd_algebraic_experiments_v27.json`

- Hypothesis C-214 (Verified (math), 2026-02-01): Right multiplication operator R_a (x -> xa) is isometric (all eigenvalue magnitudes = 1.0) at dim<=8 (composition algebras), non-isometric at dim>=16. Eigenvalue spread matches L_a (C-213) exactly: 0.67 (dim=16), 1.27 (dim=32), 1.64 (dim=64), 1.72 (dim=128). Tr(R_a) = dim * Re(a) at all dims. R_a and L_a have identical spectral properties.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v28.py`, `data/csv/cd_algebraic_experiments_v28.json`

- Hypothesis C-215 (Verified (math), 2026-02-01): Third and fourth power associativity: (a^2)*a = a*(a^2) and (a^2)^2 = a*(a^3) hold EXACTLY at ALL CD dims 4-512. Max deviations < 1e-15 (machine epsilon). This provides an explicit low-order check of power-associativity (C-142), confirming the specific groupings (a^2)a and a(a^2) are indistinguishable.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v28.py`, `data/csv/cd_algebraic_experiments_v28.json`

- Hypothesis C-217 (Verified (math), 2026-02-01): Jordan product norm\|\|{a,b}\|\|for unit vectors: exactly 1.0 at dim=2 (commutative), then monotonically decreasing: 0.77 (dim=4), 0.56 (dim=8), 0.40 (dim=16), 0.28 (dim=32), 0.20 (dim=64), 0.14 (dim=128), 0.10 (dim=256), 0.065 (dim=512). Anti-symmetric part\|\|(ab-ba)/2\|\|-> 1.0. CD algebras become "maximally non-commutative" at high dim: the symmetric (Jordan) component vanishes while the antisymmetric (Lie) component dominates.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v28.py`, `data/csv/cd_algebraic_experiments_v28.json`

- Hypothesis C-218 (Verified (math), 2026-02-01): Bilinear form B(a,b) = Re(a*conj(b)): the Gram matrix G[i,j] = B(e_i,e_j) equals the identity matrix EXACTLY at ALL CD dims 4-128. B(a,a) =\|\|a\|\|^2 EXACTLY at ALL dims. The standard basis is orthonormal under this bilinear form at every level of the Cayley-Dickson tower. This is a universal inner product structure.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v28.py`, `data/csv/cd_algebraic_experiments_v28.json`

- Hypothesis C-219 (Verified (math), 2026-02-01): Trace formula: Tr(L_a) = Tr(R_a) = dim * Re(a) holds EXACTLY at ALL CD dims 4-128. The traces are exactly equal (\|Tr(L)-Tr(R)\|= 0.0) and both equal dim * a_0 with ratio exactly 1.0. This is a universal operator-trace identity connecting the real part of an element to the traces of its left and right multiplication operators.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v28.py`, `data/csv/cd_algebraic_experiments_v28.json`

- Hypothesis C-220 (Verified (math), 2026-02-01): Norm product ratio\|\|ab\|\|/(\|\|a\|\|*\|\|b\|\|) = 1.0 EXACTLY at dim<=8 (composition algebras). At dim>=16, the ratio has mean ~1.0 but nonzero spread (std ~0.08 at dim=16, decreasing to ~0.04 at dim=512). The mean stays near 1.0 but the std decreases as O(1/sqrt(dim)), showing concentration of measure.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v29.py`, `data/csv/cd_algebraic_experiments_v29.json`

- Hypothesis C-223 (Verified (math), 2026-02-01): Associator trilinear ratio\|\|A(a,b,c)\|\|/(\|\|a\|\|*\|\|b\|\|*\|\|c\|\|) = 0 at dim=4 (associative). Mean -> sqrt(2) ~ 1.414 at high dim (1.415 at dim=512), confirming C-197. Max ratio bounded < 2.3 at all dims. Std decreasing as O(1/sqrt(dim)). The associator norm concentrates at sqrt(2) via CLT, same as EC (C-209).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v29.py`, `data/csv/cd_algebraic_experiments_v29.json`

- Hypothesis C-227 (Verified (math), 2026-02-01): Subalgebra gen{a,b} dimension: 4 at dim=4 (full quaternion algebra), 4 at dim=8 (Artin's theorem confirmed -- 2 elements generate a quaternion subalgebra), 6 at dim>=16 (consistently, across all trials). Artin's theorem sharp: any 2 elements of O generate at most a 4-dim associative subalgebra. Beyond O, gen{a,b} = 6 is a new universal constant.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v30.py`, `data/csv/cd_algebraic_experiments_v30.json`

- Hypothesis C-230 (Verified (math), 2026-02-01): Re(ab) = Re(ba) holds EXACTLY at ALL CD dims 4-512. The real part of the product is symmetric even though the full product is not commutative. This is equivalent to the symmetry of the bilinear form B(a,b) = <a,b> and follows from the CD conjugation structure. Max deviation = 0.0 at all dims tested.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v30.py`, `data/csv/cd_algebraic_experiments_v30.json`

- Hypothesis C-232 (Verified (math), 2026-02-01): Real part formula: Re(ab) = a_0*b_0 - sum_{k>=1} a_k*b_k (Lorentzian inner product) holds EXACTLY at ALL CD dims 2-512. Also: Re(a*conj(b)) = <a,b> (Euclidean dot product) EXACT at ALL dims. The product's real part encodes a (1,n-1)-signature metric, while using conjugation recovers the Euclidean metric. Both identities hold for arbitrary (non-unit) vectors.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v31.py`, `data/csv/cd_algebraic_experiments_v31.json`

- Hypothesis C-233 (Verified (math), 2026-02-01): Left alternative law x(xy) = (xx)y holds EXACTLY at dim<=8, fails at dim>=16. Mean failure grows from 0.51 (dim=16) to 0.96 (dim=256), saturating near 1.0. This verifies the alternative property (C-199) from the specific left-alternative perspective and shows the failure magnitude approaches a finite limit.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v31.py`, `data/csv/cd_algebraic_experiments_v31.json`

- Hypothesis C-234 (Verified (math), 2026-02-01): Jordan norm scaling:\|\|{a,b}\|\|~ C/sqrt(dim) with C ~ 1.57. Power-law fit gives slope = -0.502 (expect -0.5). sqrt(dim)*\|\|{a,b}\|\|= 1.571 +/- 0.043, nearly constant across dims 4-512. This gives a precise quantitative law for how the symmetric (Jordan) part of the product vanishes at high dimension.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v31.py`, `data/csv/cd_algebraic_experiments_v31.json`

- Hypothesis C-235 (Verified (math), 2026-02-01): Operator determinant: det(L_a) = +/-1 for unit a at dim<=8 (L_a is orthogonal). At dim>=16,\|det(L_a)\|collapses rapidly: 0.40 (dim=16), 0.002 (dim=32), ~0 (dim=64). At dim>=64, L_a is effectively singular for generic unit a. This means left multiplication by a unit element loses information (non-injective) beyond composition dimensions.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v31.py`, `data/csv/cd_algebraic_experiments_v31.json`

- Hypothesis C-237 (Verified (math), 2026-02-01): Symmetrized associator: A(a,b,c)+A(b,a,c) = 0 and A(a,b,c)+A(a,c,b) = 0 at dim<=8 (alternating property). Both fail simultaneously at dim>=16. These are the two independent adjacent transpositions generating S_3; their vanishing at dim<=8 is equivalent to the alternating property. Failure magnitudes comparable (~1.5) at dim>=16.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v31.py`, `data/csv/cd_algebraic_experiments_v31.json`

- Hypothesis C-238 (Verified (math), 2026-02-01): Right alternative law y(xx) = (yx)x holds EXACTLY at dim<=8, fails at dim>=16. Mean failure grows from 0.51 (dim=16) to 0.96 (dim=256), saturating near 1.0. Mirrors the left alternative (C-233), confirming the full alternative property at composition dimensions.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v32.py`, `data/csv/cd_algebraic_experiments_v32.json`

- Hypothesis C-240 (Verified (math), 2026-02-01): Adjoint map T_a(x) = a*x*conj(a) is an isometry (\|\|T_a(x)\|\|=\|\|x\|\|) EXACTLY at dim<=8. At dim>=16, T_a distorts norms: mean ratio 1.13 (dim=16) growing to 1.38 (dim=128). The adjoint/inner automorphism preserves the norm only in composition algebras.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v32.py`, `data/csv/cd_algebraic_experiments_v32.json`

- Hypothesis C-243 (Verified (math), 2026-02-01): Quadratic form N(a) = a*conj(a): (1) N(a) is purely real at ALL CD dims (imaginary parts = 0). (2) Re(N(a)) =\|\|a\|\|^2 EXACTLY at ALL dims. (3) N(ab) = N(a)*N(b) EXACTLY at dim<=8 (composition property). (4) N(ab) != N(a)*N(b) at dim>=16, with deviation growing rapidly (158 at dim=16, 16000 at dim=256). This is the fundamental norm form whose multiplicativity characterizes composition algebras.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v32.py`, `data/csv/cd_algebraic_experiments_v32.json`

- Hypothesis C-244 (Verified (math), 2026-02-01): Inverse element: a^{-1} = conj(a)/\|\|a\|\|^2 satisfies a*a^{-1} = a^{-1}*a = e_0 EXACTLY at ALL CD dims 2-256. Both left and right inverses work universally. This follows from N(a) = a*conj(a) =\|\|a\|\|^2*e_0 (C-243). Every nonzero CD element is invertible.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v33.py`, `data/csv/cd_algebraic_experiments_v33.json`

- Hypothesis C-245 (Verified (math), 2026-02-01): Artin's theorem: the subalgebra generated by any 2 elements is associative at dim<=8 (alternative algebras). Verified via A(a, b, ab) = A(a, b, ba) = A(a, ab, b) = 0 at dim<=8. Fails at dim>=16 with max associator > 1.0.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v33.py`, `data/csv/cd_algebraic_experiments_v33.json`

- Hypothesis C-251 (Verified (math), 2026-02-01): Eigenvalue spectrum of L_a: all eigenvalues lie on the unit circle (\|lambda\|=1) at dim<=8 (L_a orthogonal). At dim>=16, eigenvalues spread: spectral radius grows from 1.27 (dim=16) to 1.77 (dim=64); minimum\|lambda\|shrinks from 0.60 to 0.13. L_a becomes progressively more ill-conditioned.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v34.py`, `data/csv/cd_algebraic_experiments_v34.json`

- Hypothesis C-253 (Verified (math), 2026-02-01): Associator norm scaling:\|\|A(a,b,c)\|\|for unit vectors converges to sqrt(2) ~ 1.4142 as dim -> infinity. Zero at dim=4 (associative), then 1.09 (dim=8), 1.31 (dim=16), ..., 1.415 (dim=512). Log-log slope ~ 0.05 (nearly flat). The associator has a well-defined infinite-dimensional limit.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v34.py`, `data/csv/cd_algebraic_experiments_v34.json`

- Hypothesis C-256 (Verified (math), 2026-02-01): R_a and L_a eigenvalue spectra match (sorted\|eigenvalues\|identical) at ALL CD dims 4-64. This verifies C-214 (spectral mirroring) with explicit eigenvalue comparison, not just spectral radius. The left-right spectral equivalence is universal.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v35.py`, `data/csv/cd_algebraic_experiments_v35.json`

- Hypothesis C-257 (Verified (math), 2026-02-01): Associator norm concentration: CV(\|\|A\|\|) = std/mean decreases monotonically from 0.30 (dim=8) to 0.05 (dim=512). CV scaling slope ~ -0.45 (near -0.5), consistent with concentration of measure ~ 1/sqrt(dim). The associator norm becomes sharply peaked around sqrt(2) at high dim.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v35.py`, `data/csv/cd_algebraic_experiments_v35.json`

- Hypothesis C-258 (Verified (math), 2026-02-01): Product norm ratio:\|\|ab\|\|/(\|\|a\|\|*\|\|b\|\|) = 1.0 EXACTLY at dim<=8 (composition property). At dim>=16, the ratio has mean ~ 1.0 but std > 0 (spread). The mean stays near 1.0 at all dims (unbiased), but the variance decreases with dim. Norm multiplicativity fails symmetrically.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v35.py`, `data/csv/cd_algebraic_experiments_v35.json`

- Hypothesis C-264 (Verified (math), 2026-02-01): Commutator-to-associator norm ratio:\|\|[a,b]\|\|/\|\|A(a,b,c)\|\|-> sqrt(2) ~ 1.414 as dim -> infinity. This follows from\|\|[a,b]\|\|-> 2.0 (C-222) and\|\|A\|\|-> sqrt(2) (C-253), giving ratio 2/sqrt(2) = sqrt(2). Convergence is monotonic from ~1.58 at dim=8 to ~1.41 at dim>=128.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v36.py`, `data/csv/cd_algebraic_experiments_v36.json`

- Hypothesis C-266 (Verified (math), 2026-02-01): Flexible nucleus = full algebra at ALL CD dims 4-128. Every element satisfies (xa)x = x(ax), confirming universal flexibility (C-215) via direct element-by-element sampling. This is a defining property of the entire CD tower.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v36.py`, `data/csv/cd_algebraic_experiments_v36.json`

- Hypothesis C-267 (Verified (math), 2026-02-01): Moufang identity a(b(ac)) = ((ab)a)c holds EXACTLY at dim<=8 (Moufang loop). Fails at dim>=16 with violations growing from ~1.4 (dim=16) to ~2.2 (dim=64). This verifies the unit elements of R,C,H,O form a Moufang loop but sedenions and beyond do not. Combined with C-250 (Bol), C-236 (right Moufang), the full Moufang quartet is now verified.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v36.py`, `data/csv/cd_algebraic_experiments_v36.json`

- Hypothesis C-270 (Verified (math), 2026-02-01): Di-associator (ax)b - a(xb) = 0 at dim<=4 (associative). Nonzero at dim>=8 with mean norm approaching sqrt(2) ~ 1.414 at high dim. The di-associator IS the associator A(a,x,b) -- this experiment verifies the associativity boundary from the bimodule perspective. The mean di-associator norm converges to the same sqrt(2) limit as the standard associator (C-253).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v37.py`, `data/csv/cd_algebraic_experiments_v37.json`

- Hypothesis C-271 (Verified (math), 2026-02-01): Artin's theorem: subalgebra generated by any two elements is associative at dim<=8 (alternative algebras). All products a,b,ab,ba,a^2,b^2 and their triple products have zero associator. Fails at dim>=16 with\|\|A\|\|~ 1.0-1.8. This is a direct computational verification of Artin's classical theorem for CD algebras.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v37.py`, `data/csv/cd_algebraic_experiments_v37.json`

- Hypothesis C-277 (Verified (math), 2026-02-01): Basis element squares: e_k^2 = -e_0 for all k>=1 at ALL CD dims 2-128. e_0^2 = +e_0 (identity). All dim-1 imaginary basis elements are square roots of -1. This is a fundamental structural property of the CD construction: each imaginary unit generates a copy of the complex numbers.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v38.py`, `data/csv/cd_algebraic_experiments_v38.json`

- Hypothesis C-279 (Verified (math), 2026-02-01): Alternative nucleus: ALL random elements satisfy both left and right alternative laws at dim<=8. NO random element satisfies them at dim>=16 (0/16 at dim=16, 0/20 at dim>=32). The alternative property is all-or-nothing: either the full algebra is alternative or essentially no elements satisfy both laws. Note: basis elements trivially satisfy alt laws because e_k^2 is scalar.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v38.py`, `data/csv/cd_algebraic_experiments_v38.json`

- Hypothesis C-282 (Verified (math), 2026-02-01): Subalgebra embedding chain: R c C c H c O c S c P verified within dim=64. Elements with support in the first d components (d=1,2,4,8,16,32) produce products with zero overflow into higher components. Every CD algebra naturally contains all smaller CD algebras as closed subalgebras.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v39.py`, `data/csv/cd_algebraic_experiments_v39.json`

- Hypothesis C-283 (Verified (math), 2026-02-01): Associator mean norm convergence: mean(\|\|A(a,b,c)\|\|) for unit vectors increases monotonically from 1.12 (dim=8) to 1.41 (dim=256), approaching sqrt(2) with normalized ratio reaching 0.998. This reconfirms C-253 with explicit monotonicity verification over 6 dimensions.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v39.py`, `data/csv/cd_algebraic_experiments_v39.json`

- Hypothesis C-284 (Verified (math), 2026-02-01): Anti-commutator norm ratio:\|\|{a,b}\|\|/(2*\|\|a\|\|*\|\|b\|\|) = 1.0 at dim=2 (commutative), decreasing to ~0.10 at dim=256. The ratio scales approximately as 1/sqrt(dim/2), consistent with random orientation in high-dimensional space. The anti-commutator measures "commutativity overlap" which vanishes as the algebra grows more non-commutative.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v39.py`, `data/csv/cd_algebraic_experiments_v39.json`

- Hypothesis C-287 (Verified (math), 2026-02-01): Imaginary product structure: for pure imaginary a,b (Re=0), Re(ab) = -<a,b> (negative inner product) holds EXACTLY at ALL CD dims 4-128. The imaginary part \|\|Im(ab)\|\|grows proportionally to dim-1. This decomposes the product into a scalar (inner product) and vector (cross product) part universally.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v40.py`, `data/csv/cd_algebraic_experiments_v40.json`

- Hypothesis C-288 (Verified (math), 2026-02-01): L_a eigenvalue conjugate pairing: all eigenvalues of the left-multiplication matrix L_a come in conjugate pairs (real matrix property). At dim<=8, \|det(L_a)\|= 1 exactly (composition algebra). At dim>=16, \|det(L_a)\|decays rapidly: ~0.40 (dim=16), ~0.002 (dim=32), ~3e-8 (dim=64). The determinant collapse quantifies the loss of norm multiplicativity.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v40.py`, `data/csv/cd_algebraic_experiments_v40.json`

- Hypothesis C-292 (Verified (math), 2026-02-01): Moufang identity a(b(ac)) = (a(ba))c holds EXACTLY at dim<=8 (Moufang loop) and FAILS at dim>=16. The failure ratio increases monotonically: 0.80 (dim=16), 1.21 (dim=32), 1.48 (dim=64), 1.58 (dim=128). The Moufang property is a composition algebra boundary phenomenon.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v41.py`, `data/csv/cd_algebraic_experiments_v41.json`

- Hypothesis C-296 (Verified (math), 2026-02-01): Left-right multiplication intertwining: xa = conj(conj(a)*conj(x)) holds at ALL CD dims 2-256 with EXACT zero error (bitwise identical). This identity expresses right multiplication as left multiplication conjugated by the involution. It is a structural consequence of the CD construction.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v41.py`, `data/csv/cd_algebraic_experiments_v41.json`

- Hypothesis C-297 (Verified (math), 2026-02-01): Product of conjugates: conj(a)*conj(b) = conj(ba) holds at ALL CD dims 2-256 with EXACT zero error (bitwise identical). This is equivalent to the anti-automorphism property (C-293) but tested in the reverse direction. Both orderings of the anti-automorphism identity are verified.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v41.py`, `data/csv/cd_algebraic_experiments_v41.json`

- Hypothesis C-298 (Verified (math), 2026-02-01): Trace of left-multiplication: Tr(L_a) = dim * Re(a) holds EXACTLY at ALL CD dims 4-64. This is a universal spectral identity connecting the trace of the multiplication operator to the real part of the element. It follows from the conjugation structure of the CD construction.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v42.py`, `data/csv/cd_algebraic_experiments_v42.json`

- Hypothesis C-299 (Verified (math), 2026-02-01): Right Bol identity (ab)(ca) = a((bc)a) holds EXACTLY at dim<=8 and FAILS at dim>=16. The failure ratio increases monotonically: 0.78 (dim=16), 1.21 (dim=32), 1.46 (dim=64), 1.59 (dim=128). The Bol identity is a composition algebra boundary property, similar to but distinct from the Moufang identity (C-292).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v42.py`, `data/csv/cd_algebraic_experiments_v42.json`

- Hypothesis C-300 (Verified (math), 2026-02-01): Commutator-anticommutator decomposition: \|\|[a,b]\|\|^2 + \|\|{a,b}\|\|^2 = 4*\|\|ab\|\|^2 holds EXACTLY at ALL CD dims 2-256. This is the Parseval-like orthogonal decomposition ab = ([a,b] + {a,b})/2, proving that commutator and anticommutator are orthogonal components of the product. Universal.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v42.py`, `data/csv/cd_algebraic_experiments_v42.json`

- Hypothesis C-303 (Verified (math), 2026-02-01): Real part symmetry: Re(ab) = Re(ba) holds at ALL CD dims 2-256 (universal). This follows from Re(ab) = <a, conj(b)> and the symmetry of the real inner product. Even though ab != ba in general, their real parts always agree.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v42.py`, `data/csv/cd_algebraic_experiments_v42.json`

- Hypothesis C-304 (Verified (math), 2026-02-01): Inverse element: a * conj(a)/\|\|a\|\|^2 = conj(a)/\|\|a\|\|^2 * a = e_0 holds EXACTLY at ALL CD dims 2-256. Every nonzero element is invertible with inverse a^{-1} = conj(a)/\|\|a\|\|^2. Both orderings produce e_0 to machine precision. This is a universal CD property.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v43.py`, `data/csv/cd_algebraic_experiments_v43.json`

- Hypothesis C-305 (Verified (math), 2026-02-01): L_a and R_a spectral equivalence: the multisets of eigenvalue magnitudes of L_a and R_a are identical at ALL CD dims 4-64. This follows from the intertwining identity R_a = conj . L_{conj(a)} . conj (C-296), since conjugation is an isometry.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v43.py`, `data/csv/cd_algebraic_experiments_v43.json`

- Hypothesis C-307 (Verified (math), 2026-02-01): Scalar triple product associativity: Re(a(bc)) = Re((ab)c) holds EXACTLY at ALL CD dims 4-256. The real part of a triple product is associative even though the full product is not. This is a universal property stronger than Re(ab)=Re(ba) (C-303).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v43.py`, `data/csv/cd_algebraic_experiments_v43.json`

- Hypothesis C-319 (Verified (math), 2026-02-01): Flexibility: (ab)a = a(ba) holds EXACTLY at ALL CD dims 2-256. This is a defining property of flexible algebras and holds universally in the CD construction. Max diff grows with dim but stays at machine precision (1e-15 at dim=2, 1.4e-12 at dim=256). Universal.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v45.py`, `data/csv/cd_algebraic_experiments_v45.json`

- Hypothesis C-324 (Verified (math), 2026-02-01): Commutator energy partition: \|\|[a,b]\|\|^2/\|\|ab\|\|^2 increases from 0.0 (dim=2, commutative) through 1.51 (dim=4), 2.67 (dim=8), 3.21 (dim=16), 3.59 (dim=32), 3.81 (dim=64), to 3.96 (dim=256), approaching 4.0. The anticommutator fraction decreases correspondingly. Sum is exactly 4.0 at all dims (Parseval, C-300). At high dim, products are almost purely antisymmetric.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v46.py`, `data/csv/cd_algebraic_experiments_v46.json`

- Hypothesis C-329 (Verified (math), 2026-02-01): L_a eigenvalue structure: at dim<=8 (composition algebras), L_a has exactly 1 distinct eigenvalue magnitude = \|\|a\|\|(i.e. L_a is a scaled orthogonal matrix). At dim>=16, the number of distinct eigenvalue magnitudes increases: 3 (dim=16), 8 (dim=32), 16 (dim=64). This is equivalent to saying L_a is conformal at dim<=8 but not beyond.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v47.py`, `data/csv/cd_algebraic_experiments_v47.json`

- Hypothesis C-330 (Verified (math), 2026-02-01): Fourth-power norm: \|\|a^2\|\|^2 = \|\|a\|\|^4 at ALL CD dims 2-256. This follows from the quadratic identity (C-321): a^2 = 2*Re(a)*a - \|\|a\|\|^2*e_0, which gives \|\|a^2\|\|^2 = 4*Re(a)^2*\|\|a\|\|^2 - 4*Re(a)^2*\|\|a\|\|^2 + \|\|a\|\|^4. Universal.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v47.py`, `data/csv/cd_algebraic_experiments_v47.json`

- Hypothesis C-332 (Verified (math), 2026-02-01): Artin's theorem: (a, b, ab) = 0 (the associator of a, b, and their product vanishes) at dim<=8 (alternative algebras). This is a consequence of Artin's theorem: in an alternative algebra, any 2-generated subalgebra is associative. Fails at dim>=16: diff=476 (dim=16), 1031 (dim=32).
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v47.py`, `data/csv/cd_algebraic_experiments_v47.json`

- Hypothesis C-342 (Verified (math), 2026-02-01): Real part of square: Re(a^2) = 2*Re(a)^2 - \|\|a\|\|^2 at ALL CD dims 2-256. This is the Re-component of the quadratic identity (C-321). Equivalently, Re(a^2) = Re(a)^2 - \|\|Im(a)\|\|^2 (difference of squared real and imaginary norms). Universal.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v49.py`, `data/csv/cd_algebraic_experiments_v49.json`

- Hypothesis C-343 (Verified (math), 2026-02-01): Determinant of L_a: \|det(L_a)\|= \|\|a\|\|^dim iff dim<=8 (composition algebras). At dim<=8, L_a is a scaled orthogonal matrix, so all eigenvalue magnitudes equal \|\|a\|\|and det = \|\|a\|\|^dim. At dim>=16, eigenvalue magnitudes spread (C-329) and the determinant deviates: relative error ~0.23-0.31 in log space.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v49.py`, `data/csv/cd_algebraic_experiments_v49.json`

- Hypothesis C-349 (Verified (math), 2026-02-01): Imaginary product norm Pythagorean: \|\|Im(ab)\|\|^2 = \|\|ab\|\|^2 - Re(ab)^2 at ALL CD dims 2-256. This is the Pythagorean decomposition (C-338) applied to the product ab. The identity is tautological (orthogonal decomposition into real and imaginary parts). Universal.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v50.py`, `data/csv/cd_algebraic_experiments_v50.json`

- Hypothesis C-356 (Verified (math), 2026-02-01): Artin's theorem (2-generated subalgebra associativity): any subalgebra generated by two elements is associative at dim<=8. Tested with products (ab)*a, a*(ba), (ab)*(ba) and their nested associativity. At dim>=16, associativity fails with diffs ~1764-38839. This is the definitive test of alternativity via Artin.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v51.py`, `data/csv/cd_algebraic_experiments_v51.json`

- Hypothesis C-361 (Verified (math), 2026-02-01): Trace form symmetry: Re(ab) = Re(ba) at ALL CD dims 2-256. The real part of a product is commutative. This follows from the bilinear structure of the CD real-part formula. Universal and exact. Reconfirms C-303 from a direct batch test.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v52.py`, `data/csv/cd_algebraic_experiments_v52.json`

- Hypothesis C-375 (Verified (math), 2026-02-01): Anti-involution properties of CD conjugation: (1) conj(conj(a)) = a (involutive), (2)\|\|conj(a)\|\|=\|\|a\|\|(norm-preserving), (3) a + conj(a) = 2*Re(a)*e_0 (real extraction), (4) a*conj(a) =\|\|a\|\|^2*e_0 (norm product). All four hold universally at all dims 2-256. All are exact to machine precision.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v54.py`, `data/csv/cd_algebraic_experiments_v54.json`

- Hypothesis C-379 (Verified (math), 2026-02-01): Frobenius norm of L_a:\|\|L_a\|\|_F^2 = dim *\|\|a\|\|^2 holds universally at all CD dimensions 2-64. This follows from bilinearity and orthonormality of the CD basis: the multiplication table permutes basis products with signs. Exact (std=0) at all dims.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v55.py`, `data/csv/cd_algebraic_experiments_v55.json`

- Hypothesis C-392 (Verified (math), 2026-02-01): Multiplication table structure: for all CD algebras dim 2-64, every basis product e_i*e_j is exactly +/- e_k for some k. The multiplication table has exactly one nonzero entry per product, and that entry is +1 or -1. This is a fundamental property of the Cayley-Dickson construction.
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v57.py`, `data/csv/cd_algebraic_experiments_v57.json`

- Hypothesis C-400 (Verified (Analog), 2026-02-02): Metamaterials can emulate Alcubierre warp drive metrics for electromagnetic waves (Analog Gravity).
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/arxiv_1009.5663_smolyaninov_2010_metamaterial_based_model_alcubierre_warp_drive.pdf`

- Hypothesis C-401 (Theoretical (Blueprint), 2026-02-02): A Casimir cavity (1um sphere in 4um cylinder) generates the negative energy density required for a nanoscale warp bubble.
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/White_2021_Casimir_Warp.pdf`

- Hypothesis C-402 (Refuted, 2026-02-02): Metamaterial Gravitational Coupling can reduce warp drive energy requirements to achievable levels.
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/Rodal_2025_Metamaterial_Gravity.pdf`

- Hypothesis C-404 (Speculative (program), 2026-02-03): Bulk locality can be organized by boundary modular data K_A=-log rho_A and entanglement wedge reconstruction; robustness is naturally described by operator-algebra QEC (code subspace).
  - Where stated: `docs/convos/CONVOS_CLAIMS_INBOX.md`, `docs/theory/EMERGENCE_LAYERS_AXIOMS.md`, `docs/external_sources/C404_HOLOGRAPHIC_MODULAR_LOCALITY_SOURCES.md`, `docs/external_sources/EMERGENCE_LAYERS_SOURCES.md`, `docs/external_sources/EMERGENCE_LAYERS_SUMMARIES.md`, `src/holography/maxflow.py`, `tests/test_holography_bit_threads.py`

- Hypothesis C-406 (Partially verified (method), 2026-02-02): TSCP/box-kite sky mapping must be invariant under the relevant algebraic symmetries (or else explicitly enumerate tested embeddings) to avoid "picked the embedding that worked".
  - Where stated: `docs/convos/CONVOS_CLAIMS_INBOX.md`, `docs/external_sources/TSCP_METHOD_SOURCES.md`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `crates/algebra_core/src/boxkites.rs`, `tests/test_tscp_alignment_offline.py`, `tests/test_tscp_embedding_sweep.py`

- Hypothesis C-407 (Partially verified (method), 2026-02-02): Reported sky-alignment p-values must include explicit trial-factor accounting (look-elsewhere) across tuned degrees of freedom (e.g., box-kite choice, smoothing scale, catalog cuts).
  - Where stated: `docs/convos/CONVOS_CLAIMS_INBOX.md`, `docs/external_sources/TSCP_METHOD_SOURCES.md`, `docs/preregistered/TSCP_SKY_ALIGNMENT.md`, `reports/tscp_trial_factor_ledger.md`, `tests/test_tscp_alignment_offline.py`, `tests/test_tscp_embedding_sweep.py`, `src/verification/verify_tscp_prereg_trial_factors.py`

- Hypothesis C-412 (Modeled (Visualization), 2026-02-02): "Holographic Entropy Trap" 5-phase mechanism (Injection -> Lensing -> Sedenion Resonance -> Parton Decay -> Extraction) visualized in Director's Cut animation.
  - Where stated: `data/artifacts/images/warp_pulse_animation_directors_cut.mp4`, `src/scripts/visualization/animate_warp_v7_directors_cut.py`

- Hypothesis C-417 (Speculative (Synthesis), 2026-02-02): Hypothesis: Ray capture efficiency in fractal metamaterials correlates with Sedenion Zero Divisor density; "Holographic Entropy Trap" maps information loss to algebraic annihilation.
  - Where stated: `docs/external_sources/OPEN_CLAIMS_SOURCES.md`, `src/scripts/analysis/sedenion_warp_synthesis.py`, `data/artifacts/images/sedenion_capture_scaling.png`

- Hypothesis C-419 (Modeled (Engineering), 2026-02-02): Digital Matter BOM generation pipeline produces fabrication-ready CSVs tracking layer stoichiometry, mass density (ng/cm2), and specific vendor MPNs.
  - Where stated: `src/scripts/engineering/generate_digital_matter_bom.py`, `data/artifacts/manufacturing/sedenion_spaceplate_bom.csv`

- Hypothesis C-420 (Modeled (Engineering), 2026-02-02): Automated CAD generation outputs OpenSCAD geometry and SVG lithography masks for metamaterial nanostructures, linking refractive index maps to pillar diameters.
  - Where stated: `src/scripts/engineering/generate_bom_cad.py`, `crates/materials_core/src/metamaterial.rs`, `data/artifacts/engineering/spaceplate_geometry.scad`

- Hypothesis C-421 (Modeled (Design), 2026-02-02): Metamaterial designs incorporate Rogers RT5880 carrier substrates and Gold/Silicon I-beam stacks to achieve impedance-matched high-index performance.
  - Where stated: `crates/materials_core/src/effective_medium.rs`, `src/scripts/engineering/generate_bom_cad.py`

- Hypothesis C-422 (Modeled (Simulation), 2026-02-02): Negative-dimension vacuum dynamics ($D \sim k^{-3}$) coupled with attractive self-interaction spontaneously generates stable solitons (gravastar candidates) from random fluctuations.
  - Where stated: `src/scripts/simulation/genesis_simulation_v2.py`, `data/artifacts/images/genesis_simulation_grand.png`

- Hypothesis C-423 (Modeled (Simulation), 2026-02-02): Grand Unified Simulator v4 integrates CUDA-based relativistic ray tracing with robust FDFD electromagnetic field solving to visualize multi-scale warp-metamaterial interactions.
  - Where stated: `src/scripts/engineering/grand_unified_simulator_v4.py`, `data/artifacts/images/OCULUS_GRAND_DASHBOARD_v4.png`

- Hypothesis C-424 (Modeled (Simulation), 2026-02-02): "Holographic Warp Gate" simulation harmonizes Alcubierre metric gradients with Kozyrev p-adic noise and Spaceplate refractive index compression ($R \approx 5$) to model structured vacuum control.
  - Where stated: `src/scripts/engineering/holographic_warp_gate.py`, `data/artifacts/images/holographic_warp_gpu.png`

- Hypothesis C-425 (Modeled (Math), 2026-02-02): Sedenion Transport Kernel generates explicit 16x16 basis multiplication matrices for modeling non-associative quantum evolution via Strang splitting.
  - Where stated: `crates/algebra_core/src/hypercomplex.rs`

- Hypothesis C-426 (Speculative (Toy Model), 2026-02-02): Pathion (32D) Zero-Divisor interaction matrix diagonalization provides a testbed for algebraic mass spectrum hypotheses (logarithmic scaling).
  - Where stated: `docs/external_sources/OPEN_CLAIMS_SOURCES.md`, `crates/algebra_core/src/hypercomplex.rs`

- Hypothesis C-428 (Modeled (Simulation), 2026-02-02): Integrated Warp Geodesic Simulation traces null rays through a Kerr metric background using 'NegDimCosmology' expansion parameters; escape/capture conditions are successfully visualized in a plot.
  - Where stated: `src/scripts/simulation/integrated_warp_geodesic.py`, `data/artifacts/images/kerr_trace_demo.png`

- Hypothesis C-429 (Modeled (Simulation), 2026-02-02): Kerr black hole shadow asymmetry due to frame dragging (a*=0.99) is successfully simulated and visualized; deviation from Schwarzschild shadow is consistent with spin effects.
  - Where stated: `src/scripts/simulation/frame_dragging_demo.py`, `data/artifacts/images/kerr_shadow_asymmetry.png`

- Hypothesis C-430 (Modeled (Analytic), 2026-02-02): Negative Dimension Cosmology (NegDim) expansion history H(z) and luminosity distance D_L(z) deviate from standard LambdaCDM as predicted by the effective equation of state w_eff.
  - Where stated: `src/scripts/analysis/cosmology_comparison.py`, `data/artifacts/images/cosmology_comparison.png`

- Hypothesis C-431 (Modeled (Visualization), 2026-02-02): The Sedenion Zero Divisor manifold, when projected into 3D perturbation space (e2, e5, e14), forms a coherent, non-trivial isosurface, visualizing the 'shadow' of the 16D algebraic singularity.
  - Where stated: `src/scripts/visualization/vis_8d_slice.py`, `data/artifacts/images/sedenion_slice_3d.png`

- Hypothesis C-432 (Partially verified (Ray-trace only; analytic overlay pending), 2026-02-02): Kerr geodesic trajectories computed via the C++ Mino-time integrator can be compared against the analytic Bardeen (1973) shadow boundary; an overlay check is pending.
  - Where stated: `docs/external_sources/OPEN_CLAIMS_SOURCES.md`, `src/scripts/simulation/integrated_warp_geodesic.py`, `data/artifacts/images/kerr_trace_demo.png`

- Hypothesis C-434 (Verified (Provenance), 2026-02-03): Material stack Silicon-Gold-Ice VIII reconciled with literature-supported optical constants (Si n=3.48, Au complex eps, Ice VIII n=1.73 @ 30 GPa) for 1550nm Analogue Optics experiments.
  - Where stated: `docs/theory/WARP_PHYSICS_RECONCILIATION.md`, `src/scripts/visualization/animate_warp_v8_rigorous.py`

- Hypothesis C-435 (Modeled (Simulation), 2026-02-03): "Power Pipeline" energy bookkeeping model (Photon intensity -> boundary absorption -> Plasmon field -> Parton jitter field) replaces stochastic state changes in Warp Ring simulations.
  - Where stated: `docs/theory/WARP_PHYSICS_RECONCILIATION.md`, `src/scripts/visualization/animate_warp_v8_rigorous.py`
