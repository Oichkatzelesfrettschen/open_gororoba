# Gemini Physics Function Migration Ledger

This ledger is generated from live `src/gemini_physics` code and marks each top-level symbol as:
- `wrap`: keep thin Python API, compute in Rust.
- `port`: move implementation to Rust crate, keep optional Python shim.
- `drop`: keep in Python only (non-kernel boundary), do not port to Rust core.

Package split applied: `src/gemini_physics/quantum/*` -> `src/quantum_runtime/*`.

| Module | Symbol | Kind | Target Rust/Python Location | Action | Notes |
| --- | --- | --- | --- | --- | --- |
| `gemini_physics.cd_motif_census` | `cross_assessors` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_motif_census` | `_cd_basis_mul_sign` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_motif_census` | `diagonal_zero_products` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_motif_census` | `xor_bucket` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_motif_census` | `MotifComponent` | `class` | `crates/algebra_core/src/boxkites.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.cd_motif_census` | `motif_components_for_cross_assessors` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_xor_heuristics` | `xor_key` | `function` | `crates/algebra_core/src/zd_graphs.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_xor_heuristics` | `xor_bucket_necessary_for_two_blade_zero_product` | `function` | `crates/algebra_core/src/zd_graphs.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_xor_heuristics` | `xor_balanced_four_tuple` | `function` | `crates/algebra_core/src/zd_graphs.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_xor_heuristics` | `xor_pairing_buckets_for_balanced_four_tuple` | `function` | `crates/algebra_core/src/zd_graphs.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cd_xor_heuristics` | `xor_bucket_necessary_for_two_blade_vs_balanced_four_blade` | `function` | `crates/algebra_core/src/zd_graphs.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `QuantumCosmology` | `class` | `crates/cosmology_core/src/bounce.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.cosmology` | `hubble_E_lcdm` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `hubble_E_bounce` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `luminosity_distance` | `function` | `crates/cosmology_core/src/bounce.rs` | `wrap` | Thin Python wrapper over Rust kernel via gororoba_py. |
| `gemini_physics.cosmology` | `distance_modulus` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `cmb_shift_parameter` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `bao_sound_horizon_approx` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `generate_synthetic_sn_data` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `generate_synthetic_bao_data` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `chi2_sn` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `chi2_bao` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `fit_model` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `spectral_index_bounce` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `run_observational_fit` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.cosmology` | `synthesize_cosmology_data` | `function` | `crates/cosmology_core/src/bounce.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `BoxKite` | `class` | `crates/algebra_core/src/boxkites.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.de_marrais_boxkites` | `_diag_vector` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `_has_zero_division` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `diagonal_zero_products` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `edge_sign_type` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `candidate_cross_assessors` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `primitive_assessors` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `primitive_unit_zero_divisors_for_assessor` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `strut_signature` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `canonical_strut_table` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `production_rule_1` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `production_rule_2` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `automorpheme_assessors` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `automorphemes` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `automorphemes_containing_assessor` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `production_rule_3` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.de_marrais_boxkites` | `box_kites` | `function` | `crates/algebra_core/src/boxkites.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.dimensional_geometry` | `DimensionalQuantity` | `class` | `TBD (new rust module)` | `port` | No Rust target wired yet; create dedicated Rust module. |
| `gemini_physics.dimensional_geometry` | `unit_sphere_surface_area` | `function` | `TBD (new rust module)` | `port` | No Rust target wired yet; create dedicated Rust module. |
| `gemini_physics.dimensional_geometry` | `ball_volume` | `function` | `TBD (new rust module)` | `port` | No Rust target wired yet; create dedicated Rust module. |
| `gemini_physics.dimensional_geometry` | `sample_dimensional_range` | `function` | `TBD (new rust module)` | `port` | No Rust target wired yet; create dedicated Rust module. |
| `gemini_physics.fluid_dynamics` | `equilibrium` | `function` | `crates/lbm_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fluid_dynamics` | `macroscopic` | `function` | `crates/lbm_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fluid_dynamics` | `stream` | `function` | `crates/lbm_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fluid_dynamics` | `bounce_back_top_bottom` | `function` | `crates/lbm_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fluid_dynamics` | `add_body_force` | `function` | `crates/lbm_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fluid_dynamics` | `simulate_poiseuille` | `function` | `crates/lbm_core/src/lib.rs` | `wrap` | Thin Python wrapper over Rust kernel via gororoba_py. |
| `gemini_physics.fractional_laplacian` | `fractional_laplacian_periodic_1d` | `function` | `crates/spectral_core/src/lib.rs` | `wrap` | Thin Python wrapper over Rust kernel via gororoba_py. |
| `gemini_physics.fractional_laplacian` | `_dirichlet_laplacian_eigs_1d` | `function` | `crates/spectral_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fractional_laplacian` | `fractional_laplacian_dirichlet_1d` | `function` | `crates/spectral_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fractional_laplacian` | `fractional_laplacian_periodic_2d` | `function` | `crates/spectral_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fractional_laplacian` | `fractional_laplacian_periodic_3d` | `function` | `crates/spectral_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fractional_laplacian` | `fractional_laplacian_dirichlet_2d` | `function` | `crates/spectral_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.fractional_laplacian` | `dirichlet_laplacian_1d` | `function` | `crates/spectral_core/src/lib.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.gr.kerr_geodesic` | `kerr_metric_quantities` | `function` | `crates/gr_core/src/kerr.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.gr.kerr_geodesic` | `photon_orbit_radius` | `function` | `crates/gr_core/src/kerr.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.gr.kerr_geodesic` | `impact_parameters` | `function` | `crates/gr_core/src/kerr.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.gr.kerr_geodesic` | `shadow_boundary` | `function` | `crates/gr_core/src/kerr.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.gr.kerr_geodesic` | `geodesic_rhs` | `function` | `crates/gr_core/src/kerr.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.gr.kerr_geodesic` | `trace_null_geodesic` | `function` | `crates/gr_core/src/kerr.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.gr.kerr_geodesic` | `shadow_ray_traced` | `function` | `crates/gr_core/src/kerr.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `_zeros` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `_vec_add` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `_vec_sub` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `OctonionTable` | `class` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.m3_cd_transfer` | `_o_conj_vec` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `_o_mul_vec` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `_s_mul` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `_p_map_int` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `_h_map_int` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `compute_m3_octonion_basis` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.m3_cd_transfer` | `M3Classification` | `class` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.m3_cd_transfer` | `classify_m3` | `function` | `crates/algebra_core/src/homotopy_algebra.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.materials.database` | `PhaseOfMatter` | `class` | `crates/materials_core/src/optical_database.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.materials.database` | `Material` | `class` | `crates/materials_core/src/optical_database.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.materials.database` | `MaterialDatabase` | `class` | `crates/materials_core/src/optical_database.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.materials_jarvis` | `FigshareFile` | `class` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.materials_jarvis` | `list_figshare_files` | `function` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.materials_jarvis` | `select_figshare_file` | `function` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.materials_jarvis` | `download` | `function` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.materials_jarvis` | `unzip` | `function` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.materials_jarvis` | `load_json_records` | `function` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.materials_jarvis` | `jarvis_subset_to_dataframe` | `function` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.materials_jarvis` | `write_provenance` | `function` | `-` | `drop` | Keep Python-only data fetch/provenance boundary. |
| `gemini_physics.metamaterial` | `maxwell_garnett` | `function` | `crates/materials_core/src/effective_medium.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.metamaterial` | `bruggeman` | `function` | `crates/materials_core/src/effective_medium.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.metamaterial` | `drude_lorentz` | `function` | `crates/materials_core/src/effective_medium.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.metamaterial` | `kramers_kronig_check` | `function` | `crates/materials_core/src/effective_medium.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.metamaterial` | `tmm_reflection` | `function` | `crates/materials_core/src/effective_medium.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `luneburg_n` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `luneburg_grad_n` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `trace_luneburg` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `luneburg_exit_angle_analytical` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `fisheye_n` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `fisheye_grad_n` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `trace_fisheye` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `fisheye_antipodal_point` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `_parabolic_n_params` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `_parabolic_grad_n_params` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `make_parabolic_funcs` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `trace_parabolic` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `parabolic_analytical_y` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_benchmarks` | `measure_rk4_convergence` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_solver` | `rk4_step` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_solver` | `get_gradient_central` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_solver` | `get_gradient_complex` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optics.grin_solver` | `rk4_step_absorbing` | `function` | `crates/optics_core/src/grin.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.optimized_algebra` | `measure_associator_density` | `function` | `crates/algebra_core/src/cayley_dickson.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.reggiani_replication` | `StandardZeroDivisor` | `class` | `crates/algebra_core/src/grassmannian.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.reggiani_replication` | `standard_zero_divisors` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.reggiani_replication` | `standard_zero_divisor_partners` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.reggiani_replication` | `assert_standard_zero_divisor_annihilators` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.sedenion_annihilator` | `AnnihilatorInfo` | `class` | `crates/algebra_core/src/grassmannian.rs` | `port` | Recreate as Rust struct/type in target crate if still needed. |
| `gemini_physics.sedenion_annihilator` | `left_multiplication_matrix` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.sedenion_annihilator` | `right_multiplication_matrix` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.sedenion_annihilator` | `nullspace_basis` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.sedenion_annihilator` | `annihilator_info` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.sedenion_annihilator` | `is_zero_divisor` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.sedenion_annihilator` | `is_reggiani_zd` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
| `gemini_physics.sedenion_annihilator` | `find_left_annihilator_vector` | `function` | `crates/algebra_core/src/grassmannian.rs` | `port` | Implement in Rust crate and call through gororoba_py/CLI. |
