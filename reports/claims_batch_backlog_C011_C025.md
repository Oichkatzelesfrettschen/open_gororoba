# Claims Batch Backlog (C-011..C-025) (2026-02-02)

Purpose: planning snapshot for claim-by-claim audits (not evidence).

- Matrix: `docs/CLAIMS_EVIDENCE_MATRIX.md`
- Domain map: `docs/claims/CLAIMS_DOMAIN_MAP.csv`
- Claims in range: 15
- Open claims in range: 3

## Open claims (in-range, oldest-first by last_verified)

- C-011 (Speculative, last_verified=2026-02-02, domains=algebra): Hypothesis: a Cayley-Dickson (sedenion) structure could parameterize a gravastar-like effective model, but...
- C-022 (Modeled, last_verified=2026-02-02, domains=holography, algebra): Toy model: map Cayley-Dickson doubling level n to surreal birthday n and test the CD property-loss milestones.
- C-023 (Modeled, last_verified=2026-02-02, domains=algebra): Toy model: interpret the CD associator as a discrete connection/holonomy signal over triples.

## Details (all claims in range)

| Claim | Domains | Status | Last verified | Claim (short) | Where stated (short) | Evidence / notes (short) |
|---|---|---|---|---|---|---|
| C-011 | algebra | Speculative (/ Obstructed... | 2026-02-02 | Hypothesis: a Cayley-Dickson (sedenion) structure could parameterize a gravas... | docs/NAVIGATOR.md, docs/SEDENION_GRAVASTAR_EQUIVALENCE.md, docs/NEGATIVE_DIME... | Non-associative obstruction documented (Phase 3D): sedenion associator norms... |
| C-012 | cosmology, datasets | Refuted (by observational... | 2026-01-30 | "Dark Energy as Negative Dimension Diffusion" is a defensible physical interp... | docs/NAVIGATOR.md, docs/PHYSICAL_INTERPRETATION.md, docs/external_sources/NEG... | Phase 2.2 (eta=0): Pre-registered model comparison with 1740 data points. Neg... |
| C-013 | algebra | Verified (math) | 2026-01-28 | de Marrais' GoTo "automorphemes" (from the 7 O-trips + the "8-ball" exclude r... | docs/DE_MARRAIS_REPLICATION.md, src/gemini_physics/cd/de_marrais_boxkites.py | tests/test_de_marrais_automorphemes.py reconstructs GoTo automorphemes and va... |
| C-014 | algebra | Verified (math) | 2026-01-28 | The diagonal-form family of 84 sedenion zero divisors (e_low +/- e_high) (fro... | docs/REGGIANI_REPLICATION.md, src/gemini_physics/cd/reggiani_replication.py | tests/test_reggiani_standard_zero_divisors.py verifies nullity (4,4) for all... |
| C-015 | algebra | Verified (math) | 2026-01-28 | For each of the 84 diagonal-form zero divisors u, there are exactly 4 other d... | docs/REGGIANI_REPLICATION.md, src/gemini_physics/cd/reggiani_replication.py | tests/test_reggiani_standard_zero_divisors.py + src/scripts/export/export_reg... |
| C-016 | algebra | Verified (math) | 2026-01-28 | The repo's m3 trilinear operation on distinct octonion basis triples produces... | docs/CONVOS_CONCEPTS_STATUS_INDEX.md, src/gemini_physics/cd/m3_cd_transfer.py | tests/test_m3_cd_transfer.py validates the 42/168 split and the Fano-line cri... |
| C-017 | meta | Verified (math) | 2026-01-28 | For diagonal 2-blades (e_i +/- e_j) in 16D CD, any observed zero product betw... | docs/CONVOS_CONCEPTS_STATUS_INDEX.md, src/gemini_physics/cd/cd_xor_heuristics.py | tests/test_cd_xor_heuristics.py checks necessity on a deterministic sweep; do... |
| C-018 | holography, algebra | Verified (source + tests) | 2026-01-28 | "Wheels" (Carlstrom) are commutative monoid-based structures <H,0,1,+,,/> wit... | docs/WHEELS_DIVISION_BY_ZERO.md, convos "wheel" mentions | docs/WHEELS_DIVISION_BY_ZERO.md is source-aligned; src/gemini_physics/algebra... |
| C-019 | algebra | Not supported (rejected (... | 2026-01-30 | Wheels (division-by-zero) provide a mathematically justified way to interpret... | convos narrative, docs/external_sources/WHEELS_CAYLEY_DICKSON_SOURCES.md, src... | Phase 2C structural test: all 84 diagonal-form ZDs tested with their 336 anni... |
| C-020 | legacy | Refuted | 2026-01-28 | Legacy 16D "Zero-Divisor Adjacency Matrix" represents valid algebra. | data/csv/legacy/ | Verified against commutator/parity matrices; found to be noise/hallucination. |
| C-021 | legacy | Refuted | 2026-01-28 | 1024D Basis-to-Lattice mapping is a consistent function. | data/csv/legacy/ | Analysis shows multiple mappings for same index; inconsistent sums. Label as... |
| C-022 | holography, algebra | Modeled (Toy; Phase 4A or... | 2026-02-02 | Toy model: map Cayley-Dickson doubling level n to surreal birthday n and test... | docs/theory/unified_tensor_wheel_cd_framework.md, src/scripts/analysis/surrea... | src/scripts/analysis/surreal_cd_ordinal_construction.py writes data/csv/surre... |
| C-023 | algebra | Modeled (Toy; Phase 4B ho... | 2026-02-02 | Toy model: interpret the CD associator as a discrete connection/holonomy sign... | docs/theory/unified_tensor_wheel_cd_framework.md, src/gemini_physics/geometry... | src/scripts/analysis/cd_fiber_holonomy_analysis.py compares basis-triple asso... |
| C-024 | cpp, engineering | Verified (scaffold; Phase 4) | 2026-01-28 | C++ acceleration kernels reproduce Python CD multiplication results exactly (... | cpp/, cpp/tests/test_cd_algebra.cpp, cpp/benchmarks/bench_cd_multiply.cpp | C++ scaffold complete (Phase 4): CMake + Conan build, Catch2 unit tests, Goog... |
| C-025 | gravitational-waves, stellar-cartography, holography | Refuted (/ Not Supported... | 2026-01-31 | GWTC-3 black hole sky positions cluster around projected sedenion zero-diviso... | docs/stellar_cartography/theory/HYPOTHESIS_DEF.md, src/gemini_physics/stellar... | Pre-registered Monte Carlo test (src/scripts/analysis/verify_tscp_alignment.p... |
