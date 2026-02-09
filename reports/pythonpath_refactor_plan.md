<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Pythonpath Refactor Plan (2026-02-04)

Goal: migrate away from scripts-as-top-level-modules without breaking tests.

## Current state (pytest.ini)

- pythonpath entries:
  - `src`
  - `src/scripts/analysis`
  - `src/scripts/data`
  - `src/scripts/export`
  - `src/scripts/measure`
  - `src/scripts/simulation`
  - `src/scripts/visualization`

- injected script categories: analysis, data, export, measure, simulation, visualization

## Coupling inventory

- script modules indexed: 296
- test-imported modules that resolve to scripts: 45

| module | current file | proposed import |
| --- | --- | --- |
| `c050_spaceplate_flow_isomorphism_toy` | `src/scripts/analysis/c050_spaceplate_flow_isomorphism_toy.py` | `scripts.analysis.c050_spaceplate_flow_isomorphism_toy` |
| `compact_object_census` | `src/scripts/analysis/compact_object_census.py` | `scripts.analysis.compact_object_census` |
| `extract_gwtc3_skylocalizations` | `src/scripts/data/extract_gwtc3_skylocalizations.py` | `scripts.data.extract_gwtc3_skylocalizations` |
| `fetch_aflow_materials` | `src/scripts/data/fetch_aflow_materials.py` | `scripts.data.fetch_aflow_materials` |
| `fetch_agn_reverb_masses` | `src/scripts/data/fetch_agn_reverb_masses.py` | `scripts.data.fetch_agn_reverb_masses` |
| `fetch_atnf_pulsars_full` | `src/scripts/data/fetch_atnf_pulsars_full.py` | `scripts.data.fetch_atnf_pulsars_full` |
| `fetch_chime_frb` | `src/scripts/data/fetch_chime_frb.py` | `scripts.data.fetch_chime_frb` |
| `fetch_cms_dimuon` | `src/scripts/data/fetch_cms_dimuon.py` | `scripts.data.fetch_cms_dimuon` |
| `fetch_cms_higgs_diphoton` | `src/scripts/data/fetch_cms_higgs_diphoton.py` | `scripts.data.fetch_cms_higgs_diphoton` |
| `fetch_des_y6_shear` | `src/scripts/data/fetch_des_y6_shear.py` | `scripts.data.fetch_des_y6_shear` |
| `fetch_desi_dr1_bao` | `src/scripts/data/fetch_desi_dr1_bao.py` | `scripts.data.fetch_desi_dr1_bao` |
| `fetch_fermi_grb` | `src/scripts/data/fetch_fermi_grb.py` | `scripts.data.fetch_fermi_grb` |
| `fetch_jwst_sne` | `src/scripts/data/fetch_jwst_sne.py` | `scripts.data.fetch_jwst_sne` |
| `fetch_mcgill_magnetars` | `src/scripts/data/fetch_mcgill_magnetars.py` | `scripts.data.fetch_mcgill_magnetars` |
| `fetch_nanograv_15yr` | `src/scripts/data/fetch_nanograv_15yr.py` | `scripts.data.fetch_nanograv_15yr` |
| `fetch_neutrino_params` | `src/scripts/data/fetch_neutrino_params.py` | `scripts.data.fetch_neutrino_params` |
| `fetch_o4_events` | `src/scripts/data/fetch_o4_events.py` | `scripts.data.fetch_o4_events` |
| `fetch_pdg_particle_data` | `src/scripts/data/fetch_pdg_particle_data.py` | `scripts.data.fetch_pdg_particle_data` |
| `fetch_planck_2018_chains` | `src/scripts/data/fetch_planck_2018_chains.py` | `scripts.data.fetch_planck_2018_chains` |
| `fetch_planck_2018_spectra` | `src/scripts/data/fetch_planck_2018_spectra.py` | `scripts.data.fetch_planck_2018_spectra` |
| `fix_bao_observables` | `src/scripts/data/fix_bao_observables.py` | `scripts.data.fix_bao_observables` |
| `gravastar_thin_shell` | `src/scripts/analysis/gravastar_thin_shell.py` | `scripts.analysis.gravastar_thin_shell` |
| `gravastar_thin_shell_stability` | `src/scripts/analysis/gravastar_thin_shell_stability.py` | `scripts.analysis.gravastar_thin_shell_stability` |
| `gwtc3_bayesian_mixture` | `src/scripts/analysis/gwtc3_bayesian_mixture.py` | `scripts.analysis.gwtc3_bayesian_mixture` |
| `gwtc3_mass_clumping_selection_control` | `src/scripts/analysis/gwtc3_mass_clumping_selection_control.py` | `scripts.analysis.gwtc3_mass_clumping_selection_control` |
| `gwtc3_modality_preregistered` | `src/scripts/analysis/gwtc3_modality_preregistered.py` | `scripts.analysis.gwtc3_modality_preregistered` |
| `joint_cosmology_likelihood` | `src/scripts/analysis/joint_cosmology_likelihood.py` | `scripts.analysis.joint_cosmology_likelihood` |
| `load_cosmological_datasets` | `src/scripts/data/load_cosmological_datasets.py` | `scripts.data.load_cosmological_datasets` |
| `materials_absorber_cpa_twoport` | `src/scripts/analysis/materials_absorber_cpa_twoport.py` | `scripts.analysis.materials_absorber_cpa_twoport` |
| `materials_absorber_rlc` | `src/scripts/analysis/materials_absorber_rlc.py` | `scripts.analysis.materials_absorber_rlc` |
| `materials_absorber_salisbury` | `src/scripts/analysis/materials_absorber_salisbury.py` | `scripts.analysis.materials_absorber_salisbury` |
| `materials_absorber_tcmt` | `src/scripts/analysis/materials_absorber_tcmt.py` | `scripts.analysis.materials_absorber_tcmt` |
| `materials_baseline_models` | `src/scripts/analysis/materials_baseline_models.py` | `scripts.analysis.materials_baseline_models` |
| `measure_tensor_network_entropy` | `src/scripts/measure/measure_tensor_network_entropy.py` | `scripts.measure.measure_tensor_network_entropy` |
| `measure_tensor_network_entropy_decision` | `src/scripts/measure/measure_tensor_network_entropy_decision.py` | `scripts.measure.measure_tensor_network_entropy_decision` |
| `measure_tensor_network_entropy_large` | `src/scripts/measure/measure_tensor_network_entropy_large.py` | `scripts.measure.measure_tensor_network_entropy_large` |
| `measure_tensor_network_entropy_scaling` | `src/scripts/measure/measure_tensor_network_entropy_scaling.py` | `scripts.measure.measure_tensor_network_entropy_scaling` |
| `neg_dim_free_eta_comparison` | `src/scripts/analysis/neg_dim_free_eta_comparison.py` | `scripts.analysis.neg_dim_free_eta_comparison` |
| `neg_dim_model_comparison` | `src/scripts/analysis/neg_dim_model_comparison.py` | `scripts.analysis.neg_dim_model_comparison` |
| `parisi_sourlas_spectral_dimension` | `src/scripts/analysis/parisi_sourlas_spectral_dimension.py` | `scripts.analysis.parisi_sourlas_spectral_dimension` |
| `particle_algebra_comparison` | `src/scripts/analysis/particle_algebra_comparison.py` | `scripts.analysis.particle_algebra_comparison` |
| `rl_symmetry_model` | `src/scripts/analysis/rl_symmetry_model.py` | `scripts.analysis.rl_symmetry_model` |
| `tensor_entropy_multi_system` | `src/scripts/analysis/tensor_entropy_multi_system.py` | `scripts.analysis.tensor_entropy_multi_system` |
| `tensor_entropy_multi_system_fit` | `src/scripts/analysis/tensor_entropy_multi_system_fit.py` | `scripts.analysis.tensor_entropy_multi_system_fit` |
| `vis_data_quality_dashboard` | `src/scripts/visualization/vis_data_quality_dashboard.py` | `scripts.visualization.vis_data_quality_dashboard` |

## Phase-gated migration plan

Phase A: make scripts importable as a package (no behavior change)
- Add `__init__.py` under `src/scripts/` and each script category directory.
- Keep pytest.ini pythonpath entries unchanged initially.

Phase B: update tests to use namespaced imports
- Replace imports like `from fetch_pdg_particle_data import ...` with
  `from scripts.data.fetch_pdg_particle_data import ...`.
- Update any script-to-script imports that break under module import context.

Phase C: remove pytest.ini script pythonpath injections
- Remove `src/scripts/*` entries from pytest.ini pythonpath.
- Keep only `src` (src-layout) and rely on `pip install -e .` for packages.

Acceptance criteria
- `PYTHONWARNINGS=error make check` passes with no pytest.ini script pythonpath injections.
- No tests import scripts as top-level modules.
