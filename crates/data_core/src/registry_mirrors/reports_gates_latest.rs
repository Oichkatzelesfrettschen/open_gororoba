//! # Gate Audit (2026-03-15T09:59:43-07:00)
//!
//! Output directory: `reports/gates/2026-03-15/095943`
//!
//! | Step | Exit Code | Log |
//! | --- | ---: | --- |
//! | `gate-ci-registry` | `2` | `reports/gates/2026-03-15/095943/gate-ci-registry.log` |
//!
//! ## gate-ci-registry
//!
//! Exit code: `2`
//!
//! ```texttext
//! ... (70 earlier line(s) omitted)
//! - registry/data/curated_csv/CU-0216_universal_algebra_s_predictive_mathematical_structures_for_experimental_verification_in_high_energy_physics_quantum_computing_and_cosmology_2.toml: file missing on disk
//! - registry/data/curated_csv/CU-0277_fractal_surreal_tensor_network_evolution_in_quantum_computing_2.toml: file missing on disk
//! - registry/data/curated_csv/CU-0386_corrected_comprehensive_numeric_benchmarks_2.toml: file missing on disk
//! - registry/data/curated_csv/CU-0420_explicit_corrected_simd_timings_simulated_2.toml: file missing on disk
//! - registry/data/curated_csv/CU-0434_explicit_numeric_type_benchmarks_2.toml: file missing on disk
//! - registry/data/curated_csv/CU-0450_exploration_of_numeric_only_constructs_2048d_2.toml: file missing on disk
//! - registry/data/curated_csv/CU-0456_extended_numeric_types_benchmark_results_2.toml: file missing on disk
//! - registry/data/curated_csv/CU-0506_numeric_only_testing_results_2048d_2.toml: file missing on disk
//! - registry/knowledge/docs/DOC-0010.toml: file missing on disk
//! - registry/knowledge/docs/DOC-0011.toml: file missing on disk
//! - registry/knowledge/docs/DOC-0016.toml: file missing on disk
//! - markdown destination missing from TOML inventory: registry/canonical/control_plane.sqlite3
//! - markdown destination missing from TOML inventory: registry/canonical/control_plane.sqlite3
//! - markdown destination missing from TOML inventory: db/schema.sql
//! - markdown destination missing from TOML inventory: registry/canonical/control_plane.sqlite3
//! - markdown destination missing from TOML inventory: registry/canonical/control_plane.sqlite3
//! - markdown destination missing from TOML inventory: registry/canonical/control_plane.sqlite3
//! - markdown destination missing from TOML inventory: registry/canonical/control_plane.sqlite3
//! make[1]: *** [Makefile:322: registry-control-plane-gate-readonly] Error 1
//! make: *** [Makefile:254: gate-ci-registry] Error 2
//! ```texttext
//! | `gate-ci-rust` | `2` | `reports/gates/2026-03-15/095943/gate-ci-rust.log` |
//!
//! ## gate-ci-rust
//!
//! Exit code: `2`
//!
//! ```texttext
//! ... (1709 earlier line(s) omitted)
//! WARN [dataset_providers]: Provider GwoscCombinedProvider in Rust fetch registry but not documented in provider manifest
//! WARN [dataset_providers]: Provider ImapHelio1hrProvider in Rust fetch registry but not documented in provider manifest
//! WARN [dataset_providers]: Provider JarvisProvider in Rust fetch registry but not documented in provider manifest
//! WARN [dataset_providers]: Provider SohoCeliasBundleProvider in Rust fetch registry but not documented in provider manifest
//! WARN [dataset_providers]: Provider ThingsPreferredCubesProvider in Rust fetch registry but not documented in provider manifest
//! WARN [dataset_providers]: Provider ThingsTablesProvider in Rust fetch registry but not documented in provider manifest
//! WARN [dataset_providers]: Provider WowPrintoutProvider in Rust fetch registry but not documented in provider manifest
//! OK: dataset_providers
//!    Compiling gororoba_cli_data v0.1.0 (/home/eirikr/Github/open_gororoba/crates/gororoba_cli_data)
//!     Finished `dev` profile [optimized + debuginfo] target(s) in 1.06s
//!      Running `.cache/gate-target/debug/test-inventory --check`
//! ERROR: unclassified pytest file `tests/test_dataset_label_alias_verifier.py` is missing from registry/test_taxonomy.toml
//! ERROR: unclassified pytest file `tests/test_execution_planning_registry_contracts.py` is missing from registry/test_taxonomy.toml
//! ERROR: unclassified pytest file `tests/test_execution_planning_verifier_script.py` is missing from registry/test_taxonomy.toml
//! ERROR: unclassified pytest file `tests/test_external_source_operational_contracts.py` is missing from registry/test_taxonomy.toml
//! ERROR: unclassified pytest file `tests/test_external_sources_normalizer.py` is missing from registry/test_taxonomy.toml
//! ERROR: unclassified pytest file `tests/test_registry_source_namespace_contracts.py` is missing from registry/test_taxonomy.toml
//! make[2]: *** [Makefile:344: test-inventory] Error 1
//! make[1]: *** [Makefile:339: integrity-rust] Error 2
//! make: *** [Makefile:260: gate-ci-rust] Error 2
//! ```texttext
//! | `nextest-list` | `0` | `reports/gates/2026-03-15/095943/nextest-list.log` |
//!
//! ## nextest-list
//!
//! Exit code: `0`
//!
//! ```texttext
//! ... (5805 earlier line(s) omitted)
//! verified_core::cross_validate cross_validate_arbitrary_rotation
//! verified_core::cross_validate cross_validate_identity
//! verified_core::cross_validate cross_validate_multiple_axes
//! verified_core::cross_validate_refuted cross_validate_calcagni_spectral_dimension
//! verified_core::cross_validate_refuted cross_validate_democratic_mixing
//! verified_core::cross_validate_refuted cross_validate_gf2_separation
//! verified_core::cross_validate_refuted cross_validate_neg_dim_degeneracy
//! verified_core::cross_validate_refuted cross_validate_parity_clique
//! verified_core::cross_validate_sprint59 cross_validate_binary_entropy_properties
//! verified_core::cross_validate_sprint59 cross_validate_complex_mul
//! verified_core::cross_validate_sprint59 cross_validate_nordtvedt_bd
//! verified_core::cross_validate_sprint59 cross_validate_ppn_gamma_bd
//! verified_core::cross_validate_sprint59 cross_validate_quat_inverse
//! verified_core::cross_validate_sprint59 cross_validate_quat_mul
//! verified_core::cross_validate_sprint59 cross_validate_quat_norm_multiplicative
//! verified_core::cross_validate_sprint59 cross_validate_tcmt_antiresonance
//! verified_core::cross_validate_sprint59 cross_validate_tcmt_unitarity
//!    Compiling algebra_analysis v0.1.0 (/home/eirikr/Github/open_gororoba/crates/algebra_analysis)
//!    Compiling gr_core v0.1.0 (/home/eirikr/Github/open_gororoba/crates/gr_core)
//!     Finished `test` profile [optimized + debuginfo] target(s) in 14.43s
//! ```texttext
//!
//! Gate audit failed in 2 step(s).
//!
//! Review the per-step logs for full output.
//!
//!
