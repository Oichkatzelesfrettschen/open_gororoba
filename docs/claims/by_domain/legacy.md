# Claims: legacy

Source: docs/claims/CLAIMS_DOMAIN_MAP.csv + docs/CLAIMS_EVIDENCE_MATRIX.md

Count: 13

- Hypothesis C-006 (**Verified** (snapshot reproducibility), 2026-01-28): GWTC-3 "confident events" data integrated into `data/external/GWTC-3_confident.csv` and matches the GWOSC eventapi jsonfull endpoint snapshot.
  - Where stated: `docs/archive/RESEARCH_STATUS.md`, `docs/BIBLIOGRAPHY.md`
- Hypothesis C-008 (**Speculative** (toy operator coincidence; Phase 3B), 2026-02-02): Toy operator mapping: `alpha = -1.5` yields `abs(d_s)=2` under Convention B; physical interpretation remains speculative.
  - Where stated: `src/scripts/analysis/neg_dim_pde.py`, `docs/archive/RESEARCH_STATUS.md`, `docs/NEGATIVE_DIMENSION_CLARIFICATIONS.md`, `docs/external_sources/NEGATIVE_DIMENSION_SOURCES.md`, `docs/theory/PARISI_SOURLAS_ALPHA_DERIVATION.md`, `src/scripts/analysis/parisi_sourlas_spectral_dimension.py`, `tests/test_parisi_sourlas_connection.py`, `data/csv/parisi_sourlas_spectral_dimension.csv`
- Hypothesis C-009 (**Not supported** (rejected (rejected for brickwork circuit (Phase 2A fit complete))), 2026-01-30): Tensor-network experiment exhibits entropy scaling `S ~ log(L) + L^{0.5}`.
  - Where stated: `docs/archive/RESEARCH_STATUS_FINAL.md`, `docs/external_sources/TENSOR_NETWORK_SOURCES.md`, `crates/quantum_core/src/tensor_networks.rs`, `src/scripts/measure/measure_tensor_network_entropy.py`, `src/scripts/measure/measure_tensor_network_entropy_scaling.py`, `src/scripts/measure/measure_tensor_network_entropy_decision.py`, `src/scripts/analysis/tensor_entropy_multi_system_fit.py`, `data/csv/tensor_network_entropy_metrics.csv`, `data/csv/tensor_network_entropy_scaling.csv`, `data/csv/tensor_network_entropy_decision.csv`, `data/csv/tensor_entropy_scaling_fit.csv`, `tests/test_tensor_network_entropy_tiny.py`, `tests/test_tensor_network_entropy_scaling.py`, `tests/test_tensor_network_entropy_decision.py`, `tests/test_tensor_entropy_multi_system_fit.py`, `docs/preregistered/TENSOR_ENTROPY_SCALING.md`
- Hypothesis C-020 (**Refuted**, 2026-01-28): Legacy 16D "Zero-Divisor Adjacency Matrix" represents valid algebra.
  - Where stated: `data/csv/legacy/`
- Hypothesis C-021 (**Refuted**, 2026-01-28): 1024D Basis-to-Lattice mapping is a consistent function.
  - Where stated: `data/csv/legacy/`
- Hypothesis C-044 (**Refuted** (Noise), 2026-01-31): Legacy 16D/32D/64D "Zero-Divisor Adjacency Matrices" are valid basis-element maps.
  - Where stated: `data/csv/legacy/`
- Hypothesis C-046 (**Refuted** (Scalar Rescaling), 2026-01-31): "Fractal Doping" (sum x/n^beta) stabilizes zero divisors.
  - Where stated: `archive/legacy_conjectures/`
- Hypothesis C-047 (**Refuted** (Kac-Moody), 2026-02-04): E9, E10, E11 are Euclidean sphere-packing lattices.
  - Where stated: `archive/legacy_conjectures/`, `docs/C047_E_SERIES_KAC_MOODY_AUDIT.md`, `docs/external_sources/C047_E_SERIES_KAC_MOODY_SOURCES.md`, `tests/test_c047_e_series_cartan_signature.py`
- Hypothesis C-053 (**Verified** (Toy model; degeneracy explicit), 2026-02-04): Toy mapping: Pathion (32D) tensor diagonal -> dielectric stack (TMM retrieval).
  - Where stated: `src/scripts/analysis/c053_pathion_metamaterial_mapping.py`, `data/csv/c053_pathion_tmm_summary.csv`, `tests/test_c053_pathion_metamaterial_mapping.py`, `docs/external_sources/C053_PATHION_METAMATERIAL_MAPPING_SOURCES.md`, legacy: `src/scripts/analysis/unified_spacetime_synthesis.py`
- Hypothesis C-102 (**Verified** (Monte Carlo; standalone expZ), 2026-02-04): Alternativity ratio\|\|A(x,x,y)\|\|^2 /\|\|A(x,y,z)\|\|^2 converges to approximately 1/2 as dim -> infinity.
  - Where stated: `src/scripts/analysis/c102_alt_ratio_convergence_audit.py`, `data/csv/c102_alt_ratio_summary.csv`, `data/csv/c102_alt_ratio_by_dim.csv`, `tests/test_c102_alt_ratio_convergence_audit.py`, legacy: v9 expZ, `docs/external_sources/OPEN_CLAIMS_SOURCES.md`
- Hypothesis C-103 (**Verified** (algorithmic sample; standalone expAA components), 2026-02-04): ZD manifold topology shows sharp percolation transition at angular distance ~1.0-1.2 radians.
  - Where stated: `src/scripts/analysis/c103_zd_topology_percolation_audit.py`, `data/csv/c103_zd_topology_by_eps.csv`, `data/csv/c103_zd_topology_summary.csv`, `tests/test_c103_zd_topology_percolation_audit.py`, legacy: v9 expAA, `docs/external_sources/OPEN_CLAIMS_SOURCES.md`
- Hypothesis C-165 (**Verified** (math), 2026-02-01): Generic SO(dim) rotations do NOT preserve associator norms. Relative difference\|\|A_orig\|
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v19.py`, `data/csv/cd_algebraic_experiments_v19.json`; legacy: vs\|\|A_rot\|\|is 13-29% at all tested dims (8-64). CD automorphisms (G2 for dim=8) form a much smaller subgroup of SO(dim).
- Hypothesis C-399 (**Verified** (math), 2026-02-01): Idempotent structure: the ONLY idempotents in any CD algebra are 0 and e_0 (the identity). This follows from the quadratic identity x^2 = 2Re(x)x -\|\|x\|\|^2*e_0: if e^2=e with e=alpha*e_0+v, then (2alpha-1)v=0 and alpha(2alpha-1)=alpha^2+\|
  - Where stated: `src/scripts/analysis/cd_algebraic_experiments_v58.py`, `data/csv/cd_algebraic_experiments_v58.json`; legacy: v\|\|^2. If v!=0 then alpha=1/2 giving\|\|v\|\|^2=-1/4, impossible. If v=0 then alpha=0 or 1. Universal.
