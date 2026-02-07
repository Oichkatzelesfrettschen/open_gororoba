# Gemini Physics Function Migration Ledger

> **Reconciled 2026-02-06**: All 15 Python source modules have been deleted.
> 13 modules are fully ported, 1 has a design divergence (materials.database),
> and 1 had wrong file paths in the original ledger (m3_cd_transfer).
> The `materials_jarvis` module was marked `drop` (Python-only) and its source
> remains available in git history.

## Status Key

- **DONE**: Rust implementation complete, Python source deleted, tests pass.
- **DIVERGED**: Rust has equivalent functionality with a different API design.
  Python source deleted. No functional gap, but struct names differ.
- **DROP**: Intentionally kept as Python-only (data fetch boundary).
  Python source deleted; functionality available in git history.

## Migration Summary

| Module | Status | Rust Location | Symbols | Notes |
| --- | --- | --- | --- | --- |
| `cd_motif_census` | DONE | `crates/algebra_core/src/boxkites.rs` | 6 functions/classes | cross_assessors, _cd_basis_mul_sign, diagonal_zero_products, xor_bucket, MotifComponent, motif_components_for_cross_assessors |
| `cd_xor_heuristics` | DONE | `crates/algebra_core/src/zd_graphs.rs` | 5 functions | xor_key, xor_bucket_necessary_for_two_blade_zero_product, xor_balanced_four_tuple, xor_pairing_buckets_for_balanced_four_tuple, xor_bucket_necessary_for_two_blade_vs_balanced_four_blade |
| `cosmology` | DONE | `crates/cosmology_core/src/bounce.rs` | 15 functions/classes | QuantumCosmology, hubble_E_lcdm/bounce, luminosity_distance, distance_modulus, cmb_shift_parameter, bao_sound_horizon_approx, generate_synthetic_sn/bao_data, chi2_sn/bao, fit_model, spectral_index_bounce, run_observational_fit, synthesize_cosmology_data |
| `de_marrais_boxkites` | DONE | `crates/algebra_core/src/boxkites.rs` | 17 functions/classes | BoxKite, _diag_vector, _has_zero_division, diagonal_zero_products, edge_sign_type, candidate_cross_assessors, primitive_assessors, primitive_unit_zero_divisors_for_assessor, strut_signature, canonical_strut_table, production_rule_1/2/3, automorpheme_assessors, automorphemes, automorphemes_containing_assessor, box_kites |
| `dimensional_geometry` | DONE | `crates/cosmology_core/src/dimensional_geometry.rs` | 4 functions/classes | DimensionalQuantity, unit_sphere_surface_area, ball_volume, sample_dimensional_range |
| `fluid_dynamics` | DONE | `crates/lbm_core/src/lib.rs` | 6 functions | equilibrium, macroscopic, stream, bounce_back_top_bottom, add_body_force, simulate_poiseuille |
| `fractional_laplacian` | DONE | `crates/spectral_core/src/lib.rs` | 7 functions | fractional_laplacian_periodic_1d/2d/3d, fractional_laplacian_dirichlet_1d/2d, _dirichlet_laplacian_eigs_1d, dirichlet_laplacian_1d |
| `gr.kerr_geodesic` | DONE | `crates/gr_core/src/kerr.rs` | 7 functions | kerr_metric_quantities, photon_orbit_radius, impact_parameters, shadow_boundary, geodesic_rhs, trace_null_geodesic, shadow_ray_traced |
| `m3_cd_transfer` | DONE | `crates/algebra_core/src/m3.rs` | 12 functions/classes | _zeros, _vec_add, _vec_sub, OctonionTable, _o_conj_vec, _o_mul_vec, _s_mul, _p_map_int, _h_map_int, compute_m3_octonion_basis, M3Classification, classify_m3 |
| `materials.database` | DIVERGED | `crates/materials_core/src/optical_database.rs` | 3 classes -> different API | Python had PhaseOfMatter/Material/MaterialDatabase; Rust has MaterialEntry/MaterialType with Casimir-physics focus. Equivalent coverage, different struct design. |
| `materials_jarvis` | DROP | (git history only) | 8 functions/classes | FigshareFile, list_figshare_files, select_figshare_file, download, unzip, load_json_records, jarvis_subset_to_dataframe, write_provenance. Python-only data fetch boundary. |
| `metamaterial` | DONE | `crates/materials_core/src/effective_medium.rs` | 5 functions | maxwell_garnett, bruggeman, drude_lorentz, kramers_kronig_check, tmm_reflection |
| `optics.grin_benchmarks` | DONE | `crates/optics_core/src/grin.rs` | 14 functions | luneburg_n/grad_n, trace_luneburg, luneburg_exit_angle_analytical, fisheye_n/grad_n, trace_fisheye, fisheye_antipodal_point, _parabolic_n/grad_n_params, make_parabolic_funcs, trace_parabolic, parabolic_analytical_y, measure_rk4_convergence |
| `optics.grin_solver` | DONE | `crates/optics_core/src/grin.rs` | 4 functions | rk4_step, get_gradient_central, get_gradient_complex, rk4_step_absorbing |
| `optimized_algebra` | DONE | `crates/algebra_core/src/cayley_dickson.rs` | 1 function | measure_associator_density |
| `reggiani_replication` | DONE | `crates/algebra_core/src/reggiani.rs` | 4 functions/classes | StandardZeroDivisor, standard_zero_divisors, standard_zero_divisor_partners, assert_standard_zero_divisor_annihilators |
| `sedenion_annihilator` | DONE | `crates/algebra_core/src/annihilator.rs` | 7 functions/classes | AnnihilatorInfo, left_multiplication_matrix, right_multiplication_matrix, nullspace_basis, annihilator_info, is_zero_divisor, is_reggiani_zd, find_left_annihilator_vector |

## Corrections from Original Ledger

The original ledger (generated by Codex) had these errors:

1. **m3_cd_transfer**: Listed target as `homotopy_algebra.rs` -- actual location is `m3.rs`.
   Both files exist but serve different purposes; M3 code is in m3.rs.
2. **reggiani_replication**: Listed target as `grassmannian.rs` -- actual location is `reggiani.rs`.
3. **sedenion_annihilator**: Listed target as `grassmannian.rs` -- actual location is `annihilator.rs`.
4. **dimensional_geometry**: Listed target as `TBD (new rust module)` -- created at
   `cosmology_core/src/dimensional_geometry.rs`.
5. **All entries**: Original ledger said "call through gororoba_py/CLI" -- this wrapper
   pattern was never implemented (and should not be). Rust code is called directly.
