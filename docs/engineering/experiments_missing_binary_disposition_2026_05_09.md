# Experiments without registered binaries: 2026-05-09 disposition

This document records the 33 entries in `registry/experiments.toml` that have
`binary_registered = false` and an empty `binary` field, classifies them into
three cohorts, and recommends a disposition for each.

It is a read-only audit; closing each experiment's registration will require
the gororoba-db Claim/Experiment mutator extension (see
`docs/engineering/sqlite_canonical_write_plane_design.md`) and is therefore
gated on Phase B of the open_gororoba debt-resolution program.

## Source

- Pre-image: `registry/experiments.toml` (auto-generated read-only export
  from `registry/canonical/control_plane.sqlite3`).
- Identifier extraction:
  ```sh
  grep -A30 "^binary_registered = false" registry/experiments.toml | \
      grep "^id =" | sort -u
  ```
- Count cross-check: `grep -c "binary_registered = false" registry/experiments.toml` -> 33.

## Cohort 1: theoretical/algebraic explorations (11 entries, status=active)

Sprint 70-72 era. The experiment record exists; the analysis was performed
but the work product was a library function or a notebook-style invocation
of existing CLIs rather than a standalone binary.

| ID    | Title                                                                                  |
| ----- | -------------------------------------------------------------------------------------- |
| E-073 | Numerical Photon-Graviton Mixing Amplitude: Convergence and Ward Identity Verification |
| E-078 | Immirzi Bridge Imbalance-Entropy Mapping                                               |
| E-085 | Associator entropy mechanism decomposition                                             |
| E-094 | Zero-Divisor Graph Topology Census (Sedenion + Pathion)                                |
| E-099 | 512D Bell inequality (CHSH) via associator torque correlations                         |
| E-100 | 1024D DekaVoudon gauge sector analysis via sampled AVT                                 |
| E-101 | Complex-time Wick rotation + WKB tunneling + Pathion shadow boundary                   |
| E-102 | 128D Routon non-associative entropy filter on LBM grid                                 |
| E-103 | Fractal spacetime D_f=2.7 with Q-tensor metric: flyby + Pioneer test                   |
| E-104 | CMB Quadrupole-Octupole Alignment reconciliation via 1024D DekaVoudon                  |
| E-105 | Unified Cayley-Dickson simulation engine end-to-end integration test                   |

**Recommended disposition**: introduce a new `reproducibility_class` value
`"library_test"` for these entries. Add an explicit `library_path_refs` field
pointing to the test functions that exercise the experiment, and set
`binary = "(none -- library test)"` rather than empty string. This preserves
the experiment's epistemic status without scaffolding placeholder binaries.

## Cohort 2: heliosphere data-validation batch (8 entries, status=planned)

Sprint 76-78 plan. Each entry awaits a data source (AMDA, SPDF V2, OMNI2,
Helios corefit, or NASA published-value cross-checks). None has been run.

| ID    | Title                                                                           |
| ----- | ------------------------------------------------------------------------------- |
| E-128 | Bartol vs AMDA cross-validation for Voyager 2 B-field (1990-1995)               |
| E-129 | Pioneer AMDA MAG-only radial B-field profile (5-80 AU)                          |
| E-130 | PSP AMDA inner heliosphere perihelion pass validation                           |
| E-131 | Helios corefit radial gradient (0.3-1.0 AU) vs Voyager at matched distances     |
| E-132 | Ulysses AMDA vs SPDF cross-validation for polar wind data                       |
| E-133 | Juno cruise AMDA B-field validation against Connerney et al. published values   |
| E-134 | Wind AMDA vs OMNI2 L1 cross-check for 2024                                      |
| E-135 | Pioneer/Flyby governed benchmark staging and verification (active, not planned) |

**Recommended disposition**: keep status=planned for E-128..E-134 and add
a `data_dependency` status_note documenting the missing data source. The
experiment cannot register a binary until the data is ingested. E-135 is
status=active and should move to Cohort 3.

## Cohort 3: latest-sprint experiments awaiting binary registration (14 entries, status=active)

Sprint 80-84 era. Binaries may have been written but not linked to the
experiment record, or the binaries are short-lived analysis scripts that
were folded into the report-generation pipeline.

| ID    | Title                                                                                          |
| ----- | ---------------------------------------------------------------------------------------------- |
| E-135 | Pioneer/Flyby governed benchmark staging and verification                                      |
| E-140 | Euclid Q1 Zenodo catalog discovery and ingestion                                               |
| E-141 | YSU-engine GPU technique distillation into lbm_3d_cuda                                         |
| E-179 | Falsification of slope_ratio = 42^2/10000 claim (C-1329)                                       |
| E-181 | x87 FP-80 vs DD vs nalgebra eigenvalue solver benchmark                                        |
| E-208 | Associator Flux Measurement Around Zero Divisors in 16D, 32D, and 64D                          |
| E-209 | CKM selector pair scan: 420 combos, Rayon-parallel                                             |
| E-210 | PMNS neutrino mixing selector pair scan                                                        |
| E-211 | 3-blade zero-divisor friction scan (455 triples)                                               |
| E-212 | Electroweak mixing angle from associator flux ratio                                            |
| E-213 | G2 stabilizer extraction via thin-SVD with u(3) embedding verification                         |
| E-214 | Constructive SU(3) realization and standard Gell-Mann alignment from octonionic stabilizer     |
| E-215 | Physics bridge: SU(5) SU(3)-sector cross-validation and real-part projection                   |
| E-216 | PMD CPD 7.12.0 codebase duplication baseline scan                                              |
| E-217 | Materials data architecture decision: optical_database.rs + crystal_symmetry.rs migration path |

**Recommended disposition**: per-experiment audit. For each, search the
binaries lane (`registry/binaries.toml` and `crates/*/Cargo.toml [[bin]]`)
for a matching binary name. If found, register it. If not found, file an
implementation task for the missing binary OR re-classify as a Cohort 1
library_test if the analysis was performed in-library.

Specific known mappings:

- E-216 -> the existing `cpd-report` binary (binary already exists in
  `crates/gororoba_cli_data/src/bin/cpd_report.rs`); just needs the
  registry link.
- E-217 -> documentation/migration audit; may be a Cohort-1 library_test.
- E-179 -> there is a `slope-ratio-falsification` binary candidate in
  `crates/algebra_experimental/`; verify and link.

## Implementation order (after Phase B lands)

1. Cohort 3 known-mapping fixes (E-216, possibly E-179) -- one-line each.
2. Cohort 2 status_note adds for data_dependency -- bulk update.
3. Cohort 1 schema extension: introduce `reproducibility_class =
   "library_test"` and the `library_path_refs` field. Backfill all 11.
4. Cohort 3 remaining: per-experiment audit and either binary scaffold
   or library_test reclassification.

## Acceptance criteria for closure of DEBT-EXPERIMENT-1

- All 33 entries either have `binary_registered = true` and a real binary
  reference, OR `reproducibility_class = "library_test"` with library_path
  refs, OR a documented `data_dependency` status_note for entries that
  remain status=planned.
- The `binary_registered_false` count in
  `data/output/debt_baseline_2026_05_09.toml` reaches 0.
- registry/schema_signatures.toml refreshes cleanly via
  `make integrity-resolution`.

## See also

- Stage A baseline: `data/output/debt_baseline_2026_04_30.toml` (counted 33).
- Stage B baseline: `data/output/debt_baseline_2026_05_09.toml` (re-validated).
- Audit binary: `crates/gororoba_cli_data/src/bin/repo_audit.rs` (does NOT
  count this metric; it is recorded in the registry/experiments.toml schema
  itself).
- Mutator design: `docs/engineering/sqlite_canonical_write_plane_design.md`
  (Insight + Experiment mutators are step #43 in the task DAG).
