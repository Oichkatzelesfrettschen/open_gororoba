# Claims: datasets

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/claims_domains.toml -->

Count: 33

- Hypothesis C-005 (Partially verified (source-aligned; geometric invariants computed; Phase 3A), 2026-01-30): "The geometry of sedenion zero divisors" (Reggiani, 2024) implies specific manifold identifications (e.g., `G2`, `V_2(R^7)`).
  - Where stated: `docs/SEDENION_ATLAS.md`, `docs/external_sources/REGGIANI_MANIFOLD_CLAIMS.md`, `data/external/papers/reggiani_2024_2411.18881.pdf`, `docs/theory/REGGIANI_GEOMETRIC_VALIDATION.md`

- Hypothesis C-006 (Verified (snapshot reproducibility), 2026-01-28): GWTC-3 "confident events" data integrated into `data/external/GWTC-3_confident.csv` and matches the GWOSC eventapi jsonfull endpoint snapshot.
  - Where stated: `docs/archive/RESEARCH_STATUS.md`, `docs/BIBLIOGRAPHY.md`

- Hypothesis C-012 (Refuted (by observational data (Phase 2.2 + 2B complete)), 2026-01-30): "Dark Energy as Negative Dimension Diffusion" is a defensible physical interpretation (not just a metaphor).
  - Where stated: `docs/NAVIGATOR.md`, `docs/PHYSICAL_INTERPRETATION.md`, `docs/external_sources/NEGATIVE_DIMENSION_SOURCES.md`, `docs/external_sources/HZ_DATASETS_SOURCES.md`, `docs/external_sources/COSMOLOGY_MULTIPROBE_SOURCES.md`, `docs/external_sources/ADDITIONAL_COSMOLOGICAL_DATASETS.md`, `docs/NEGATIVE_DIMENSION_DARK_ENERGY_MODEL.md`, `docs/NEG_DIM_MULTIPROBE_EXPERIMENT.md`, `crates/spectral_core/src/neg_dim.rs`, `src/scripts/analysis/neg_dim_model_comparison.py`, `src/scripts/analysis/neg_dim_free_eta_comparison.py`, `src/scripts/data/fix_bao_observables.py`, `tests/test_neg_dim_dark_energy.py`, `tests/test_neg_dim_multidata.py`, `tests/test_neg_dim_free_eta_comparison.py`, `data/csv/neg_dim_model_comparison_results.csv`, `data/csv/neg_dim_free_eta_comparison_results.csv`, `docs/preregistered/NEG_DIM_MODEL_COMPARISON.md`, `docs/preregistered/NEG_DIM_FREE_ETA_AMENDMENT.md`

- Hypothesis C-033 (Not supported (mapping under-specified in source), 2026-02-04): Sedenion basis maps to 24 generators of SU(5) (Tang & Tang 2023).
  - Where stated: `data/external/papers/arxiv_2308.14768_tang_tang_2023_sedenion_su5_generations.pdf`, `data/external/papers/arxiv_2308.14768_tang_tang_2023_sedenion_su5_generations.txt`, `docs/convos/pdf_extract_3f6ee1e837d1_sedenion_valued_field_theories_action_principles_and_challenges.md`, `docs/external_sources/SEDENION_FIELD_THEORY_SOURCES.md`, `docs/BIBLIOGRAPHY.md`, `docs/SEDENION_FIELD_THEORY.md`, `docs/C033_SU5_MAPPING_CLOSURE.md`, `crates/algebra_core/src/group_theory.rs`, `src/scripts/analysis/c033_su5_generator_summary.py`, `data/csv/c033_su5_generator_summary.csv`, `tests/test_su5_generators.py`

- Hypothesis C-034 (Verified (metadata+abstract; structural prerequisite reproduced), 2026-02-04): Chanyal (2014): a "sedenion unified theory of gravi-electromagnetism" expresses unified potentials/fields/currents of dyons and "gravito-dyons" using two 8D sectors (two eight-potentials), yielding compact generalized Dirac-Maxwell and related equations (abstract-level in repo).
  - Where stated: `docs/convos/pdf_extract_3f6ee1e837d1_sedenion_valued_field_theories_action_principles_and_challenges.md`, `docs/external_sources/C034_CHANYAL_2014_GRAVI_ELECTROMAGNETISM_SOURCES.md`, `docs/BIBLIOGRAPHY.md`, `docs/SEDENION_FIELD_THEORY.md`, `docs/C034_CHANYAL_2014_REPRODUCTION.md`, `src/scripts/analysis/c034_chanyal_2014_structural_reproduction.py`, `data/csv/c034_sedenion_doubling_identity_check.csv`, `tests/test_c034_sedenion_doubling_identity.py`, `data/external/traces/chanyal_2014_springer_abstract.txt`

- Hypothesis C-037 (Not supported (numerology; no mechanism), 2026-02-03): Numerical correspondence gamma ~ epsilon ~ 4*lambda_GB ~ 1/4, relating Barbero-Immirzi parameter, network clustering, and Gauss-Bonnet coupling.
  - Where stated: `docs/C037_NUMERICAL_COINCIDENCE_AUDIT.md`, `docs/convos/pdf_extract_2b693d92f57d_exceptional_cosmological_framework_synthesis.md`, `docs/external_sources/EXCEPTIONAL_COSMOLOGY_SOURCES.md`, `data/external/papers/arxiv_gr-qc0407051_domagala_lewandowski_2004_bh_entropy_quantum_geometry.pdf`, `data/external/papers/arxiv_gr-qc0407052_meissner_2004_bh_entropy_lqg.pdf`, `data/external/papers/arxiv_hep-th0504052_nojiri_odintsov_sasaki_2005_gauss_bonnet_dark_energy.pdf`, `docs/EXCEPTIONAL_COSMOLOGY.md`

- Hypothesis C-041 (Not supported (dimensional coincidence; no mechanism), 2026-02-03): F4 26D representation connects to bosonic string critical dimension (D=26).
  - Where stated: `docs/convos/pdf_extract_2b693d92f57d_exceptional_cosmological_framework_synthesis.md`, `docs/external_sources/EXCEPTIONAL_COSMOLOGY_SOURCES.md`, `docs/EXCEPTIONAL_COSMOLOGY.md`, `docs/C041_F4_STRING_DIMENSION_COINCIDENCE_AUDIT.md`, `data/external/papers/arxiv_gr-qc0306060_larranaga_2003_introduction_to_bosonic_string_theory.pdf`

- Hypothesis C-043 (Verified (integration pipeline; data sources may be synthetic/curated), 2026-02-04): "Compact Object" populations (Pulsars, Magnetars, FRBs) can be integrated into the unified framework for multi-messenger testing.
  - Where stated: `docs/ROADMAP_DETAILED.md`, `docs/external_sources/C043_COMPACT_OBJECT_CATALOG_SOURCES.md`, `src/scripts/data/fetch_chime_frb.py`, `src/scripts/data/fetch_atnf_pulsars_full.py`, `src/scripts/data/fetch_mcgill_magnetars.py`, `src/scripts/data/fetch_compact_objects.py`, `data/csv/compact_objects_catalog.csv`, `data/csv/compact_objects_catalog.PROVENANCE.json`, `tests/test_c043_compact_objects_catalog_artifact.py`

- Hypothesis C-056 (Verified (data), 2026-01-31): PDG lepton/boson masses verified against experiment (electron, muon, Z, W, Higgs).
  - Where stated: `src/scripts/data/fetch_pdg_particle_data.py`, `tests/test_pdg_particle_data.py`, `tests/test_pdg_coupling_constants.py`

- Hypothesis C-057 (Verified (data), 2026-01-31): DESI Y1 BAO 7-bin measurements integrated into multi-probe pipeline.
  - Where stated: `src/scripts/data/fetch_desi_dr1_bao.py`, `tests/test_desi_dr1_bao.py`, `data/external/bao/`

- Hypothesis C-058 (Verified (data), 2026-01-31): Planck 2018 parameter summary + CMB spectra integrated.
  - Where stated: `src/scripts/data/fetch_planck_2018_chains.py`, `src/scripts/data/fetch_planck_2018_spectra.py`, `tests/test_planck_2018_chains.py`, `tests/test_planck_2018_spectra.py`

- Hypothesis C-059 (Verified (data), 2026-02-02): NANOGrav 15yr free spectrum (log10(rho(f)) KDE) integrated.
  - Where stated: `src/scripts/data/fetch_nanograv_15yr.py`, `data/csv/nanograv_15yr_free_spectrum.csv`, `tests/test_nanograv_15yr.py`

- Hypothesis C-060 (Verified (data), 2026-01-31): GWTC-3 sky localizations (64 events) integrated.
  - Where stated: `src/scripts/data/fetch_gwtc3_confident.py`, `tests/test_gwtc3_sky_localization_areas.py`, `data/external/GWTC-3_confident.json`

- Hypothesis C-061 (Verified (data), 2026-01-31): O4 GW events (10 confirmed) integrated.
  - Where stated: `src/scripts/data/fetch_o4_events.py`, `tests/test_gwtc_o4_events.py`

- Hypothesis C-062 (Verified (data), 2026-01-31): CHIME/FRB 536-event catalog integrated.
  - Where stated: `src/scripts/data/fetch_chime_frb.py`, `tests/test_chime_frb.py`

- Hypothesis C-063 (Verified (data), 2026-01-31): ATNF 3500 pulsars + McGill 28 magnetars integrated.
  - Where stated: `src/scripts/data/fetch_atnf_pulsars_full.py`, `src/scripts/data/fetch_mcgill_magnetars.py`, `tests/test_atnf_pulsars_full.py`, `tests/test_mcgill_magnetars.py`

- Hypothesis C-064 (Verified (data), 2026-01-31): Fermi GBM 3500 GRBs integrated.
  - Where stated: `src/scripts/data/fetch_fermi_grb.py`, `tests/test_fermi_grb.py`

- Hypothesis C-065 (Verified (data), 2026-01-31): CMS dimuon + diphoton spectra (J/psi, Upsilon, Z, Higgs) integrated.
  - Where stated: `src/scripts/data/fetch_cms_dimuon.py`, `src/scripts/data/fetch_cms_higgs_diphoton.py`, `tests/test_cms_dimuon.py`, `tests/test_cms_higgs_diphoton.py`

- Hypothesis C-066 (Verified (data), 2026-01-31): Neutrino oscillation params + KATRIN upper limit integrated.
  - Where stated: `src/scripts/data/fetch_neutrino_params.py`, `tests/test_neutrino_params.py`

- Hypothesis C-067 (Verified (data), 2026-01-31): AFLOW 1000 + NOMAD materials + absorber experimental spectra integrated.
  - Where stated: `src/scripts/data/fetch_aflow_materials.py`, `src/scripts/data/fetch_materials_nomad_subset.py`, `tests/test_aflow_materials.py`, `tests/test_materials_nomad.py`, `tests/test_materials_baseline_models.py`

- Hypothesis C-400 (Verified (Analog), 2026-02-02): Metamaterials can emulate Alcubierre warp drive metrics for electromagnetic waves (Analog Gravity).
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/arxiv_1009.5663_smolyaninov_2010_metamaterial_based_model_alcubierre_warp_drive.pdf`

- Hypothesis C-401 (Theoretical (Blueprint), 2026-02-02): A Casimir cavity (1um sphere in 4um cylinder) generates the negative energy density required for a nanoscale warp bubble.
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/White_2021_Casimir_Warp.pdf`

- Hypothesis C-402 (Refuted, 2026-02-02): Metamaterial Gravitational Coupling can reduce warp drive energy requirements to achievable levels.
  - Where stated: `docs/external_sources/MULTIVERSE_METAMATERIALS_REPORT.md`, `docs/external_sources/WARP_DRIVE_SOURCES.md`, `data/external/papers/Rodal_2025_Metamaterial_Gravity.pdf`

- Hypothesis C-410 (Literature (perturbative QFT), 2026-02-02): One-loop photon-graviton mixing in constant EM fields includes non-zero tadpole contributions (Ahmadiniaz et al. 2026), relevant to semiclassical EM-gravity coupling limits.
  - Where stated: `docs/BIBLIOGRAPHY.md`, `data/external/papers/arxiv_2601.23279v1_ahmadiniaz_2026_photon_graviton_mixing.pdf`

- Hypothesis C-411 (Literature (Experimental), 2026-02-02): Spontaneous four-wave mixing (SFWM) dominates in thin LN layers due to relaxed phase matching, enabling flat entangled photon sources.
  - Where stated: `docs/BIBLIOGRAPHY.md`, `data/external/papers/arxiv_2601.23137v1_son_chekhova_2026_sfwm_thin_layer.pdf`

- Hypothesis C-418 (Modeled (Database), 2026-02-02): Material Database tracks temperature-dependent phase transitions (e.g., Ice Ih/VII/X) and wide-spectrum optical dispersion (Sellmeier/Drude) for 20+ compounds.
  - Where stated: `crates/materials_core/src/optical_database.rs`, `data/external/SiO2_refractive_index.csv`, `data/external/TiO2_rutile_refractive_index.csv`

- Hypothesis C-436 (Refuted (Rust pipeline, N=4996), 2026-02-06): FRB comoving positions exhibit local ultrametric structure (C-071 follow-up, Direction 1).
  - Where stated: `crates/gororoba_cli/src/bin/dm_ultrametric.rs`, `crates/stats_core/src/ultrametric/local.rs`, `data/csv/c071b_dm_comoving_ultrametric.csv`

- Hypothesis C-437 (Verified (Rust pipeline, N=5008 FRBs + 4233 pulsars), 2026-02-06): Compact object multi-attribute parameter space (DM, gl, gb) exhibits ultrametric structure under Euclidean distances (C-071 follow-up, Direction 2).
  - Where stated: `crates/gororoba_cli/src/bin/baire_compact.rs`, `crates/stats_core/src/ultrametric/baire.rs`, `data/csv/c071c_baire_compact_ultrametric.csv`

- Hypothesis C-438 (Verified (partial: 15/40 repeaters, 37.5%), 2026-02-06): Repeating FRB temporal cascades exhibit ultrametric hierarchy (C-071 follow-up, Direction 3).
  - Where stated: `crates/gororoba_cli/src/bin/frb_cascades.rs`, `crates/stats_core/src/ultrametric/temporal.rs`, `data/csv/c071d_frb_cascades_ultrametric.csv`

- Hypothesis C-439 (Refuted (Rust pipeline, N=35 events), 2026-02-06): GWTC-3 mass-redshift clustering exhibits ultrametric structure (C-071 follow-up, Direction 4).
  - Where stated: `crates/gororoba_cli/src/bin/gw_merger_tree.rs`, `data/csv/c071e_gw_merger_ultrametric.csv`

- Hypothesis C-440 (Refuted (Rust pipeline, N=5031), 2026-02-06): Cross-dataset cosmic objects (FRBs+GW events) exhibit dendrogram ultrametricity in comoving 3D space (C-071 follow-up, Direction 5).
  - Where stated: `crates/gororoba_cli/src/bin/cosmic_dendrogram.rs`, `data/csv/c071f_cosmic_dendrogram.csv`

- Hypothesis C-441 (Verified (Real data, N=1590), 2026-02-06): Bounce cosmology is disfavored by joint Pantheon+ SN Ia + DESI DR1 BAO fit; Lambda-CDM is the simpler adequate model.
  - Where stated: `crates/cosmology_core/src/observational.rs`, `crates/gororoba_cli/src/bin/real_cosmo_fit.rs`, `docs/INSIGHTS.md`

- Hypothesis C-442 (Verified (7 catalogs, N=25K total), 2026-02-06): Multi-attribute Euclidean ultrametricity is specific to radio transient catalogs (FRB, pulsar); not a general property of astrophysical catalogs.
  - Where stated: `crates/gororoba_cli/src/bin/multi_dataset_ultrametric.rs`, `data/csv/c071g_multi_dataset_ultrametric.csv`, `docs/INSIGHTS.md`
