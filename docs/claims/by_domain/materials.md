# Claims: materials

Source: docs/claims/CLAIMS_DOMAIN_MAP.csv + docs/CLAIMS_EVIDENCE_MATRIX.md

Count: 12

- Hypothesis C-010 (**Speculative** (spectral comparison negative; Phase 3C), 2026-02-02): Hypothesis: Cayley-Dickson (sedenion) zero-divisor motifs can be mapped to physically realizable absorber networks (e.g., TCMT/RLC/CPA), potentially approaching critical coupling; no "perfect absorption" mapping is validated yet.
  - Where stated: `docs/MATERIALS_APPLICATIONS.md`, `docs/THE_GEMINI_PROTOCOL.md`, `docs/external_sources/METAMATERIAL_ABSORBER_SOURCES.md`, `docs/C010_ABSORBER_TCMT_MAPPING.md`, `docs/C010_ABSORBER_SALISBURY.md`, `docs/C010_ABSORBER_RLC.md`, `docs/C010_ABSORBER_CPA.md`, `src/scripts/analysis/materials_absorber_tcmt.py`, `src/scripts/analysis/materials_absorber_salisbury.py`, `src/scripts/analysis/materials_absorber_rlc.py`, `src/scripts/analysis/materials_absorber_cpa_twoport.py`, `src/scripts/analysis/materials_absorber_cpa_input_sweep.py`, `data/csv/c010_truncated_icosahedron_graph.csv`, `data/csv/c010_tcmt_truncated_icosahedron.csv`, `data/csv/c010_tcmt_truncated_icosahedron_minima.csv`, `data/csv/c010_salisbury_screen.csv`, `data/csv/c010_salisbury_screen_minima.csv`, `data/csv/c010_rlc_surface_impedance.csv`, `data/csv/c010_rlc_surface_impedance_minima.csv`, `data/csv/c010_rlc_fit_summary.csv`, `data/csv/c010_cpa_twoport_scan.csv`, `data/csv/c010_cpa_twoport_minima.csv`, `data/csv/c010_cpa_input_sweep.csv`, `data/csv/c010_cpa_input_sweep_minima.csv`, `data/csv/c010_zd_tcmt_spectral_comparison.csv`, `data/csv/materials_baseline_metrics.csv`, `data/csv/materials_embedding_benchmarks.csv`
- Hypothesis C-053 (**Verified** (Toy model; degeneracy explicit), 2026-02-04): Toy mapping: Pathion (32D) tensor diagonal -> dielectric stack (TMM retrieval).
  - Where stated: `src/scripts/analysis/c053_pathion_metamaterial_mapping.py`, `data/csv/c053_pathion_tmm_summary.csv`, `tests/test_c053_pathion_metamaterial_mapping.py`, `docs/external_sources/C053_PATHION_METAMATERIAL_MAPPING_SOURCES.md`, legacy: `src/scripts/analysis/unified_spacetime_synthesis.py`
- Hypothesis C-067 (**Verified** (data), 2026-01-31): AFLOW 1000 + NOMAD materials + absorber experimental spectra integrated.
  - Where stated: `src/scripts/data/fetch_aflow_materials.py`, `src/scripts/data/fetch_materials_nomad_subset.py`, `tests/test_aflow_materials.py`, `tests/test_materials_nomad.py`, `tests/test_materials_baseline_models.py`
- Hypothesis C-400 (**Verified** (Analog), 2026-02-02): Metamaterials can emulate Alcubierre warp drive metrics for electromagnetic waves (Analog Gravity).
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/arxiv_1009.5663_smolyaninov_2010_metamaterial_based_model_alcubierre_warp_drive.pdf`
- Hypothesis C-401 (**Theoretical** (Blueprint), 2026-02-02): A Casimir cavity (1um sphere in 4um cylinder) generates the negative energy density required for a nanoscale warp bubble.
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/White_2021_Casimir_Warp.pdf`
- Hypothesis C-402 (**Refuted**, 2026-02-02): Metamaterial Gravitational Coupling can reduce warp drive energy requirements to achievable levels.
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/Rodal_2025_Metamaterial_Gravity.pdf`
- Hypothesis C-409 (**Modeled** (Simulation), 2026-02-02): "Interleaved I-Beam" spaceplate design targets high refractive index via capacitive loading (metal/dielectric stack).
  - Where stated: `src/gemini_physics/metamaterial.py`, `crates/optics_core/src/grin.rs`
- Hypothesis C-417 (**Speculative** (Synthesis), 2026-02-02): Hypothesis: Ray capture efficiency in fractal metamaterials correlates with Sedenion Zero Divisor density; "Holographic Entropy Trap" maps information loss to algebraic annihilation.
  - Where stated: `docs/external_sources/OPEN_CLAIMS_SOURCES.md`, `src/scripts/analysis/sedenion_warp_synthesis.py`, `data/artifacts/images/sedenion_capture_scaling.png`
- Hypothesis C-420 (**Modeled** (Engineering), 2026-02-02): Automated CAD generation outputs OpenSCAD geometry and SVG lithography masks for metamaterial nanostructures, linking refractive index maps to pillar diameters.
  - Where stated: `src/scripts/engineering/generate_bom_cad.py`, `crates/materials_core/src/metamaterial.rs`, `data/artifacts/engineering/spaceplate_geometry.scad`
- Hypothesis C-421 (**Modeled** (Design), 2026-02-02): Metamaterial designs incorporate Rogers RT5880 carrier substrates and Gold/Silicon I-beam stacks to achieve impedance-matched high-index performance.
  - Where stated: `src/gemini_physics/metamaterial.py`, `src/scripts/engineering/generate_bom_cad.py`
- Hypothesis C-423 (**Modeled** (Simulation), 2026-02-02): Grand Unified Simulator v4 integrates CUDA-based relativistic ray tracing with robust FDFD electromagnetic field solving to visualize multi-scale warp-metamaterial interactions.
  - Where stated: `src/scripts/engineering/grand_unified_simulator_v4.py`, `data/artifacts/images/OCULUS_GRAND_DASHBOARD_v4.png`
- Hypothesis C-427 (**Speculative** (Design), 2026-02-02): Algebraic Metamaterial synthesis maps Cayley-Dickson structure constants to permittivity tensors and Clifford subspace dimensions to quasi-periodic layer stacks.
  - Where stated: `docs/external_sources/OPEN_CLAIMS_SOURCES.md`, `src/gemini_physics/metamaterial.py`
