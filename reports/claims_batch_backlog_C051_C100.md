<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Claims Batch Backlog (C-051..C-100) (2026-02-03)

Purpose: planning snapshot for claim-by-claim audits (not evidence).

- Matrix: `docs/CLAIMS_EVIDENCE_MATRIX.md`
- Domain map: `docs/claims/CLAIMS_DOMAIN_MAP.csv`
- Claims in range: 49
- Open claims in range: 4

## Open claims (in-range, oldest-first by last_verified)

- C-053 (Modeled, last_verified=2026-02-03, domains=materials): Toy mapping: Pathion (32D) tensor diagonal -> dielectric stack (TMM retrieval).
- C-074 (Partially verified, last_verified=2026-02-03, domains=meta): CD associator growth law: <\|A(a,b,c)\|^2> = 2.00  (1 - 14.6  d^{-1.80}) for unit vectors.
- C-087 (Partially verified, last_verified=2026-02-03, domains=legacy): A_inf=2 follows from statistical independence of (ab)c and a(bc).
- C-090 (Speculative, last_verified=2026-02-03, domains=meta): ZD eigenvalue spectrum is NOT invariant under SO(7) rotation.

## Details (all claims in range)

| Claim | Domains | Status | Last verified | Claim (short) | Where stated (short) | Evidence / notes (short) |
|---|---|---|---|---|---|---|
| C-051 | meta | Verified (Simulation) | 2026-01-31 | A Pareto frontier exists for Spaceplates trading Compression (R) vs Bandwidth... | src/gemini_physics/engineering/warp_pareto.py | src/gemini_physics/engineering/warp_pareto.py sweep verifies the existence of... |
| C-052 | holography, tensor-networks | Verified (Simulation) | 2026-01-31 | MERA (Multi-scale Entanglement Renormalization) circuit produces logarithmic... | src/scripts/analysis/verify_phase_3_tasks.py | Simulation of constructive MERA (L=4, 8, 16) detected logarithmic growth (Coe... |
| C-053 | materials | Modeled (toy mapping; dia... | 2026-02-03 | Toy mapping: Pathion (32D) tensor diagonal -> dielectric stack (TMM retrieval). | src/scripts/analysis/c053_pathion_metamaterial_mapping.py, data/csv/c053_path... | src/scripts/analysis/c053_pathion_metamaterial_mapping.py writes a determinis... |
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
| C-067 | materials, datasets | Verified (data) | 2026-01-31 | AFLOW 1000 + NOMAD materials + absorber experimental spectra integrated. | src/scripts/data/fetch_aflow_materials.py, src/scripts/data/fetch_materials_n... | AFLOW: 1000 materials with band gap, formation energy, structure. NOMAD: subs... |
| C-068 | holography, legacy, algebra | Refuted (degenerate spect... | 2026-02-03 | Sedenion 84-ZD interaction matrix eigenvalue spectrum matches PDG particle ma... | src/scripts/analysis/c068_zd_interaction_spectrum_degeneracy.py, data/csv/c06... | Deterministic reproduction: build M[i,j]=\|\|v_iv_j\|\| from the 84 diagonal-... |
| C-069 | algebra | Not supported (rejected (... | 2026-01-31 | Three octonionic subalgebra principal angles reproduce PMNS neutrino mixing a... | src/scripts/analysis/cd_algebraic_experiments.py, data/csv/cd_algebraic_exper... | Key metric: 1970-01-01 (unknown). Notes: Angles are only 0 or 90 degrees (Boo... |
| C-070 | legacy | Not supported (legacy met... | 2026-02-03 | CD associator power spectrum shape matches NANOGrav GW background. | docs/external_sources/C070_NANOGRAV_SPECTRUM_MATCH_SOURCES.md, src/scripts/an... | Notes: The legacy "shape match" used rank correlation after reversing/subsamp... |
| C-071 | meta | Not supported (rejected (... | 2026-01-31 | FRB dispersion measures exhibit p-adic ultrametric structure. | src/scripts/analysis/cd_algebraic_experiments.py, data/csv/cd_algebraic_exper... | Key metric: 1970-01-01 (unknown). Notes: Only 3 FRB DMs available locally; in... |
| C-072 | meta | Not supported (rejected (... | 2026-01-31 | CMS resonance mass ratios appear as ZD eigenvalue ratios. | src/scripts/analysis/cd_algebraic_experiments.py, data/csv/cd_algebraic_exper... | Key metric: 1970-01-01 (unknown). Notes: Only 3 distinct eigenvalues; v2 left... |
| C-073 | meta | Not supported (rejected (... | 2026-01-31 | Left-multiplication operator spectrum matches PDG masses. | src/scripts/analysis/cd_algebraic_experiments_v2.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Diagonal-form ZDs produce extremely... |
| C-074 | meta | Partially verified (fit s... | 2026-02-03 | CD associator growth law: <\|A(a,b,c)\|^2> = 2.00  (1 - 14.6  d^{-1.80}) for... | src/scripts/analysis/cd_algebraic_experiments_v3.py (exp2v3_associator_growth... | Notes: The stored fit is mechanically checked by tests/test_claim_c074_associ... |
| C-075 | legacy | Verified (Math/Code; as-d... | 2026-02-03 | Pathion 32D ZD interaction matrix has 33 distinct eigenvalues spanning 3.3 de... | src/scripts/analysis/c075_pathion_zd_interaction_spectrum.py, data/csv/c075_p... | Notes: This is a narrow, mechanically checked statement about spectral divers... |
| C-076 | algebra | Verified (SUGGESTIVE ( ma... | 2026-01-31 | Three octonionic subalgebras have exact generation symmetry (identical Casimi... | src/scripts/analysis/cd_algebraic_experiments_v3.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: All three subalgebras are exactly cl... |
| C-077 | legacy | Not supported (near-democ... | 2026-02-03 | Subalgebra associator mixing matrix resembles PMNS matrix. | src/scripts/analysis/c077_associator_mixing_pmns_audit.py, data/csv/c077_asso... | Notes: Under the legacy v3 exp5 construction, the normalized 3x3 matrix is cl... |
| C-078 | meta | Not supported (no improve... | 2026-02-03 | Higher-dim ZDs (32D/64D) improve mass spectrum coverage. | src/scripts/analysis/c078_higher_dim_zd_coverage_audit.py, data/csv/c078_high... | Notes: Under the legacy v4 "higher-dim diagonal-form ZDs from CSV" experiment... |
| C-079 | holography | Not supported (rejected (... | 2026-01-31 | E8 root system eigenvalues reproduce particle masses. | src/scripts/analysis/cd_algebraic_experiments_v4.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: E8 roots are all equal-norm (Gram co... |
| C-080 | meta | Not supported (rejected (... | 2026-01-31 | FRB DM distribution has p-adic ultrametric structure. | src/scripts/analysis/cd_algebraic_experiments_v4.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Ultrametric fraction (19.8%) matches... |
| C-081 | meta | Not supported (rejected (... | 2026-01-31 | Multi-parameter Givens rotation finds exact PMNS angles. | src/scripts/analysis/cd_algebraic_experiments_v4.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Optimizer found perfect 0.000 residu... |
| C-083 | meta | Not supported (rejected (... | 2026-01-31 | General-form ZDs from CSV have richer left-mult spectrum. | src/scripts/analysis/cd_algebraic_experiments_v4.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: CSV ZDs are still diagonal-form (e_i... |
| C-084 | meta | Not supported (rejected (... | 2026-01-31 | Yukawa-like symmetry breaking produces PMNS-like mixing. | src/scripts/analysis/cd_algebraic_experiments_v4.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Perturbation barely moves the democr... |
| C-085 | meta | Not supported (rejected (... | 2026-01-31 | CMS resonance ratios match E8+pathion combined eigenvalue ratios. | src/scripts/analysis/cd_algebraic_experiments_v4.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Null probability too high. Matches a... |
| C-086 | meta | Verified | 2026-01-31 | PMNS angles from subalgebra rotation are trivially achievable. | src/scripts/analysis/cd_algebraic_experiments_v5.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: ANY 3 target angles in [5,85] degree... |
| C-087 | legacy | Partially verified (deriv... | 2026-02-03 | A_inf=2 follows from statistical independence of (ab)c and a(bc). | src/scripts/analysis/c087_associator_independence_audit.py, data/csv/c087_ass... | Notes: The v5 expJ decomposition writes E[\|\|A\|\|^2] = E[\|\|abc\|\|^2] + E... |
| C-088 | algebra | Verified | 2026-01-31 | Non-diagonal ZDs exist abundantly in 16D sedenion algebra. | src/scripts/analysis/cd_algebraic_experiments_v5.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Non-diagonal ZDs have 13-14 nonzero... |
| C-089 | meta | Verified | 2026-01-31 | Structure constants f_{ijk} have degenerate singular values. | src/scripts/analysis/cd_algebraic_experiments_v5.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: All SVs equal (2.828 at dim=8, 4.000... |
| C-090 | meta | Speculative (SUGGESTIVE) | 2026-02-03 | ZD eigenvalue spectrum is NOT invariant under SO(7) rotation. | data/csv/cd_algebraic_experiments_v5.json (results.N), src/scripts/analysis/c... | Key metric: 1970-01-01 (unknown). Notes: Spectrum depends on the specific emb... |
| C-091 | holography | Not supported (rejected (... | 2026-01-31 | Non-diagonal ZD spectrum matches 9/12 particle masses. | src/scripts/analysis/cd_algebraic_experiments_v6.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Null tests (v7): uniform random 148-... |
| C-092 | meta | Not supported (clustered... | 2026-02-03 | SO(7) orbit structure of ZD spectrum is continuous, not discrete. | src/scripts/analysis/c092_so7_orbit_structure_audit.py, data/csv/c092_so7_orb... | Notes: The cached v6 orbit-classification summary reports is_continuous=false... |
| C-093 | holography | Verified | 2026-01-31 | Algebraic Gram matrix Tr(L_i^T L_j) is proportional to identity. | src/scripts/analysis/cd_algebraic_experiments_v6.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Basis left-mult matrices are orthogo... |
| C-094 | legacy | Verified (guardrail) | 2026-02-03 | Associator saturation fit fails for non-power-of-2 dimensions. | src/scripts/analysis/c094_non_power_two_dims_fit_audit.py, data/csv/c094_non_... | Notes: The cached v6 expR mixed power-of-two dims with non-2^n truncations (d... |
| C-095 | meta | Verified | 2026-01-31 | Associator saturation A_inf=2 is confirmed at 9 power-of-2 dims (4 through 10... | src/scripts/analysis/cd_algebraic_experiments_v8.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Extends v3/v5 result to dim=1024. Al... |
| C-096 | legacy, algebra | Verified (Math/Code) | 2026-02-03 | Associator tensor A(e_i,e_j,e_k) exhibits phase transitions in algebraic iden... | src/scripts/analysis/c096_associator_tensor_transitions_audit.py, data/csv/c0... | Notes: Cached v8 expT computes basis-associator tensor diagnostics at dims 4,... |
| C-097 | holography, legacy | Verified (Math/Code) | 2026-02-03 | Diagonal ZD interaction graph has exactly 3 distinct edge weights {0, 1, sqrt... | src/scripts/analysis/c097_zd_interaction_graph_audit.py, data/csv/c097_zd_int... | Notes: Verified invariants for the 84 standard diagonal-form sedenion ZDs: th... |
| C-098 | holography | Verified | 2026-01-31 | CD algebras lose algebraic properties at precise, dimension-specific threshol... | src/scripts/analysis/cd_algebraic_experiments_v8.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: This is the complete CD property-los... |
| C-099 | engineering, algebra | Verified (Monte Carlo) | 2026-02-03 | Non-diagonal sedenion ZDs have consistent geometry: 14 nonzero components (mo... | src/scripts/analysis/c099_nondiag_zd_geometry_audit.py, data/csv/c099_nondiag... | Notes: Cached v8 expW reports n_found=494 non-diagonal ZDs with PCA effective... |
| C-100 | meta | Verified | 2026-01-31 | Flexibility identity A(x,y,x)=0 holds exactly at all tested CD dimensions (4... | src/scripts/analysis/cd_algebraic_experiments_v8.py, data/csv/cd_algebraic_ex... | Key metric: 1970-01-01 (unknown). Notes: Flexibility is the "last surviving i... |
