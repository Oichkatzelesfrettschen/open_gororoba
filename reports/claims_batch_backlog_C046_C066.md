<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Claims Batch Backlog (C-046..C-066) (2026-02-03)

Purpose: planning snapshot for claim-by-claim audits (not evidence).

- Matrix: `docs/CLAIMS_EVIDENCE_MATRIX.md`
- Domain map: `docs/claims/CLAIMS_DOMAIN_MAP.csv`
- Claims in range: 21
- Open claims in range: 0

## Open claims (in-range, oldest-first by last_verified)

- (none)

## Details (all claims in range)

| Claim | Domains | Status | Last verified | Claim (short) | Where stated (short) | Evidence / notes (short) |
|---|---|---|---|---|---|---|
| C-046 | legacy, algebra | Refuted (Scalar Rescaling) | 2026-01-31 | "Fractal Doping" (sum x/n^beta) stabilizes zero divisors. | archive/legacy_conjectures/ | Mathematical audit (docs/theory/OPERATOR_DEPTH_STRATIFICATION.md) shows this... |
| C-047 | holography, legacy | Refuted (Kac-Moody) | 2026-02-04 | E9, E10, E11 are Euclidean sphere-packing lattices. | archive/legacy_conjectures/, docs/C047_E_SERIES_KAC_MOODY_AUDIT.md, docs/exte... | The legacy "Euclidean lattice" framing is rejected: E9 is affine/degenerate a... |
| C-048 | meta | Verified (Analogy; scope-... | 2026-02-04 | Analogy: "depth stratification" is used as a terminology/structure analogy to... | docs/theory/OPERATOR_DEPTH_STRATIFICATION.md, docs/external_sources/C048_MOTI... | The mapping is maintained as a terminology/structure analogy (tower levels vs... |
| C-049 | meta | Established (GR) | 2026-01-31 | Lightspace and Gravitytime are distinct geometric structures (Conformal vs Sc... | docs/theory/PHASE_IV_0_2_LEDGER.md | The typed dependency graph cleanly separates the conformal causal skeleton (N... |
| C-050 | meta | Verified (Toy model; LP e... | 2026-02-04 | Toy equivalence: spaceplate delay allocation can be cast as multi-flavor flow... | docs/theory/WARP_FLOW_ALLOCATION.md, docs/C050_SPACEPLATE_FLOW_ISOMORPHISM.md... | Scope-safe form: the delay-allocation problem and a multi-flavor flow-allocat... |
| C-051 | meta | Verified (Simulation) | 2026-01-31 | A Pareto frontier exists for Spaceplates trading Compression (R) vs Bandwidth... | src/gemini_physics/engineering/warp_pareto.py | src/gemini_physics/engineering/warp_pareto.py sweep verifies the existence of... |
| C-052 | holography, tensor-networks | Verified (Simulation) | 2026-01-31 | MERA (Multi-scale Entanglement Renormalization) circuit produces logarithmic... | src/scripts/analysis/verify_phase_3_tasks.py | Simulation of constructive MERA (L=4, 8, 16) detected logarithmic growth (Coe... |
| C-053 | materials, legacy | Verified (Toy model; dege... | 2026-02-04 | Toy mapping: Pathion (32D) tensor diagonal -> dielectric stack (TMM retrieval). | src/scripts/analysis/c053_pathion_metamaterial_mapping.py, data/csv/c053_path... | src/scripts/analysis/c053_pathion_metamaterial_mapping.py writes a determinis... |
| C-054 | algebra | Verified (Math/Code) | 2026-01-31 | Carlstrom's Wheel Algebra formally models information loss in non-associative... | src/gemini_physics/cd/cd_wheel_algebra.py | Implementation of Wheel axioms over CD algebras verifies that the "Nullity" e... |
| C-055 | meta | Verified (Monte Carlo) | 2026-01-31 | Non-associativity is the generic (bulk) state of 16D/32D CD algebras (100% pr... | src/scripts/analysis/verify_zd_32d_count.py | Random sweeps ($N=100,000$) in 16D and 32D show 100% of pairs have non-zero r... |
| C-056 | holography, datasets | Verified (data) | 2026-01-31 | PDG lepton/boson masses verified against experiment (electron, muon, Z, W, Hi... | src/scripts/data/fetch_pdg_particle_data.py, tests/test_pdg_particle_data.py,... | Offline tests check cached PDG values against NIST/PDG 2024 reference; coupli... |
| C-057 | datasets | Verified (data) | 2026-01-31 | DESI Y1 BAO 7-bin measurements integrated into multi-probe pipeline. | src/scripts/data/fetch_desi_dr1_bao.py, tests/test_desi_dr1_bao.py, data/exte... | Offline test checks 7 redshift bins against DESI DR1 Table 1 (arXiv:2404.0300... |
| C-058 | stellar-cartography, cosmology, datasets | Verified (data) | 2026-01-31 | Planck 2018 parameter summary + CMB spectra integrated. | src/scripts/data/fetch_planck_2018_chains.py, src/scripts/data/fetch_planck_2... | Tests validate parameter means (H0, Omega_m) within 1% of Planck 2018 Table 2... |
| C-059 | datasets | Verified (data) | 2026-02-02 | NANOGrav 15yr free spectrum (log10(rho(f)) KDE) integrated. | src/scripts/data/fetch_nanograv_15yr.py, data/csv/nanograv_15yr_free_spectrum... | Cached Zenodo KDE zip (DOI: 10.5281/zenodo.8060824) and derived a 30-bin quan... |
| C-060 | gravitational-waves, stellar-cartography, datasets | Verified (data) | 2026-01-31 | GWTC-3 sky localizations (64 events) integrated. | src/scripts/data/fetch_gwtc3_confident.py, tests/test_gwtc3_sky_localization_... | 64 confident events with sky area (deg^2), chirp mass, luminosity distance. |
| C-061 | gravitational-waves, datasets | Verified (data) | 2026-01-31 | O4 GW events (10 confirmed) integrated. | src/scripts/data/fetch_o4_events.py, tests/test_gwtc_o4_events.py | 10 O4 events from GWOSC with FAR < 1e-7, chirp mass > 0. |
| C-062 | datasets | Verified (data) | 2026-01-31 | CHIME/FRB 536-event catalog integrated. | src/scripts/data/fetch_chime_frb.py, tests/test_chime_frb.py | Catalog with DM, fluence, SNR validated against CHIME/FRB Catalog 1 (Amiri+ 2... |
| C-063 | datasets | Verified (data) | 2026-01-31 | ATNF 3500 pulsars + McGill 28 magnetars integrated. | src/scripts/data/fetch_atnf_pulsars_full.py, src/scripts/data/fetch_mcgill_ma... | Pulsars: period, DM, position columns validated. Magnetars: B-field, P, Pdot... |
| C-064 | datasets | Verified (data) | 2026-01-31 | Fermi GBM 3500 GRBs integrated. | src/scripts/data/fetch_fermi_grb.py, tests/test_fermi_grb.py | T90, fluence, trigger time columns validated against Fermi GBM catalog. |
| C-065 | datasets | Verified (data) | 2026-01-31 | CMS dimuon + diphoton spectra (J/psi, Upsilon, Z, Higgs) integrated. | src/scripts/data/fetch_cms_dimuon.py, src/scripts/data/fetch_cms_higgs_diphot... | Dimuon: 3 resonance peaks (J/psi ~3.1, Upsilon ~9.5, Z ~91 GeV). Diphoton: Hi... |
| C-066 | datasets | Verified (data) | 2026-01-31 | Neutrino oscillation params + KATRIN upper limit integrated. | src/scripts/data/fetch_neutrino_params.py, tests/test_neutrino_params.py | theta_12, theta_23, theta_13, Delta_m^2 values validated against PDG 2024. KA... |
