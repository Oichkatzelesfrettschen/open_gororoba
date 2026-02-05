# Network Gating Audit (2026-02-04)

Goal: identify network-capable codepaths and confirm opt-in gating exists.

- Files scanned: 502
- Network-indicator hits: 20
- Network-indicator hits without obvious gating: 2

## Findings (network-indicator hits)

| path | kind | contract.network | gate_token | env_gate |
| --- | --- | --- | --- | --- |
| `src/gemini_physics/cosmology/gwosc_eventapi.py` | package | n/a | true | false |
| `src/gemini_physics/digital_matter/verification/z_evolution.py` | package | n/a | false | false |
| `src/gemini_physics/materials/materials_jarvis.py` | package | n/a | true | false |
| `src/gemini_physics/numerical/fdtd_3d.py` | package | n/a | false | false |
| `src/gemini_physics/stellar_cartography/data_loader.py` | package | n/a | false | true |
| `src/gemini_physics/stellar_cartography/fetch_zenodo.py` | package | n/a | true | false |
| `src/scripts/data/fetch_aflow_materials.py` | script | missing | false | true |
| `src/scripts/data/fetch_cosmology_data.py` | script | missing | false | true |
| `src/scripts/data/fetch_emergence_layers_sources.py` | script | missing | false | true |
| `src/scripts/data/fetch_external_data.py` | script | missing | false | true |
| `src/scripts/data/fetch_gwtc3_skylocalizations_zenodo.py` | script | missing | false | true |
| `src/scripts/data/fetch_materials_nomad_subset.py` | script | missing | false | true |
| `src/scripts/data/fetch_nanograv_15yr.py` | script | missing | false | true |
| `src/scripts/data/fetch_o4_events.py` | script | missing | false | true |
| `src/scripts/data/fetch_observatory_data.py` | script | missing | false | true |
| `src/scripts/data/fetch_pdg_particle_data.py` | script | missing | false | true |
| `src/scripts/data/fetch_planck_2018_chains.py` | script | missing | false | true |
| `src/scripts/data/fetch_planck_2018_spectra.py` | script | missing | false | true |
| `src/scripts/data/fetch_pulsars_rigorous.py` | script | missing | false | true |
| `src/scripts/data/fetch_references.py` | script | missing | false | true |

## Checklist

- [ ] Every script with network clients declares `# network: gated`.
- [ ] Every network script calls `require_network()` (directly or via a gated helper).
- [ ] Every package module that can fetch remote data uses `gemini_physics.network` gating.
- [ ] Make targets that use the network are gated behind `GEMINI_ALLOW_NETWORK=1`.
