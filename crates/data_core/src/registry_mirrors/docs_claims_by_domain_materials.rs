//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/claims_domains.toml -->
//!
//! # Claims: materials
//!
//! Count: 12
//!
//! - Hypothesis C-010 (Closed/Negative-Result (local obstruction; non-local design lane modeled), 2026-03-11): The local absorber thesis is closed negatively: the assessor graph is 7 disconnected K6 cliques, so connected local lattices fail. A literature-calibrated non-local LC/Floquet design lane now reproduces the masked topology and clears the in-repo projection gate at design stage.
//! - Where stated: `crates/algebra_analysis/src/boxkites.rs`, `crates/algebra_analysis/src/crystal_bands.rs`, `crates/algebra_analysis/src/homotopy_algebra.rs`, `crates/algebra_analysis/src/reggiani.rs`, `crates/cosmology_core/src/sersic.rs`, `crates/gororoba_cli_physics/src/bin/nonlocal_algebraic_metamaterial.rs`, `crates/materials_core/src/nonlocal_metamaterial.rs`, `crates/optics_core/src/absorber_benchmark.rs`, `crates/quantum_core/src/magnonic_crystal.rs`, `crates/quantum_core/src/tight_binding.rs`, `data/csv/c010_nonlocal_material_calibrations.csv`, `data/csv/sedenion_box_kites_clustered.csv`, `docs/C010_NONLOCAL_ALGEBRAIC_METAMATERIALS.md`, `docs/external_sources/C010_NONLOCAL_ALGEBRAIC_METAMATERIALS_SOURCES.md`, `papers/MANIFEST.toml`, `registry/data/project_csv/canonical/PC-0051_sedenion_box_kites_clustered.toml`, `src/verification/verify_c010_c011_theses.py`, `tests/test_c010_c011_verifier_script.py`, `tests/test_c010_holographic_entropy_trap.py`
//!
//! - Hypothesis C-053 (Verified (Toy model; degeneracy explicit), 2026-03-11): Toy mapping: Pathion (32D) tensor diagonal -> uniform dielectric stack (TMM retrieval), with the diagonal-only degeneracy made explicit.
//! - Where stated: `crates/gororoba_cli_physics/src/bin/c053_pathion_metamaterial_mapping.rs`, `crates/gororoba_cli_physics/tests/c053_pathion_metamaterial_mapping.rs`, `crates/materials_core/src/pathion_toy_mapping.rs`, `data/csv/c053_pathion_tmm_summary.csv`, `docs/external_sources/C053_PATHION_METAMATERIAL_MAPPING_SOURCES.md`
//!
//! - Hypothesis C-067 (Verified (data), 2026-01-31): AFLOW 1000 + NOMAD materials + absorber experimental spectra integrated.
//! - Where stated: `src/scripts/data/fetch_aflow_materials.py`, `src/scripts/data/fetch_materials_nomad_subset.py`, `tests/test_aflow_materials.py`, `tests/test_materials_baseline_models.py`, `tests/test_materials_nomad.py`
//!
//! - Hypothesis C-400 (Verified (Analog), 2026-02-02): Metamaterials can emulate Alcubierre warp drive metrics for electromagnetic waves (Analog Gravity).
//! - Where stated: `data/papers/corpus/arxiv_1009.5663_smolyaninov_2010_metamaterial_based_model_alcubierre_warp_drive.pdf`, `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`
//!
//! - Hypothesis C-401 (Theoretical (Blueprint), 2026-02-02): A Casimir cavity (1um sphere in 4um cylinder) generates the negative energy density required for a nanoscale warp bubble.
//! - Where stated: `data/papers/corpus/White_2021_Casimir_Warp.pdf`, `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`
//!
//! - Hypothesis C-402 (Refuted, 2026-02-02): Metamaterial Gravitational Coupling can reduce warp drive energy requirements to achievable levels.
//! - Where stated: `data/papers/corpus/Rodal_2025_Metamaterial_Gravity.pdf`, `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`
//!
//! - Hypothesis C-409 (Modeled (Simulation), 2026-02-02): "Interleaved I-Beam" spaceplate design targets high refractive index via capacitive loading (metal/dielectric stack).
//! - Where stated: `crates/materials_core/src/effective_medium.rs`, `crates/optics_core/src/grin.rs`
//!
//! - Hypothesis C-417 (Speculative (Synthesis), 2026-02-02): Hypothesis: Ray capture efficiency in fractal metamaterials correlates with Sedenion Zero Divisor density; "Holographic Entropy Trap" maps information loss to algebraic annihilation.
//! - Where stated: `data/artifacts/images/sedenion_capture_scaling.png`, `docs/external_sources/OPEN_CLAIMS_SOURCES.md`, `src/scripts/analysis/sedenion_warp_synthesis.py`
//!
//! - Hypothesis C-420 (Modeled (Engineering), 2026-02-02): Automated CAD generation outputs OpenSCAD geometry and SVG lithography masks for metamaterial nanostructures, linking refractive index maps to pillar diameters.
//! - Where stated: `crates/materials_core/src/metamaterial.rs`, `data/artifacts/engineering/spaceplate_geometry.scad`, `src/scripts/engineering/generate_bom_cad.py`
//!
//! > - Hypothesis C-421 (Modeled (Design), 2026-02-02): Metamaterial designs incorporate Rogers RT5880 carrier substrates and Gold/Silicon I-beam stacks to achieve impedance-matched high-index performance.
//! - Where stated: `crates/materials_core/src/effective_medium.rs`, `src/scripts/engineering/generate_bom_cad.py`
//!
//! - Hypothesis C-423 (Modeled (Simulation), 2026-02-02): Grand Unified Simulator v4 integrates CUDA-based relativistic ray tracing with robust FDFD electromagnetic field solving to visualize multi-scale warp-metamaterial interactions.
//! - Where stated: `data/artifacts/images/OCULUS_GRAND_DASHBOARD_v4.png`, `src/scripts/engineering/grand_unified_simulator_v4.py`
//!
//! - Hypothesis C-427 (Speculative (Design), 2026-02-02): Algebraic Metamaterial synthesis maps Cayley-Dickson structure constants to permittivity tensors and Clifford subspace dimensions to quasi-periodic layer stacks.
//! - Where stated: `crates/materials_core/src/effective_medium.rs`, `docs/external_sources/OPEN_CLAIMS_SOURCES.md`
//!
