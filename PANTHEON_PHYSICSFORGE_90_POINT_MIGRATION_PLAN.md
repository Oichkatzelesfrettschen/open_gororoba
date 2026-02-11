<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Pantheon + PhysicsForge to open_gororoba: 90-Point Migration Plan

## Active skills and execution posture
- `gororoba-mcp-orchestrator`
- `gororoba-rust-workflow`
- `claims-evidence-executor`
- `mcp-bash-workflow`

## Phase 1: License and governance hardening (1-9)
1. Confirm `pantheon` canonical license file is GPL-2.0-only (`mcp-bash`).
2. Confirm `PhysicsForge` canonical license file is GPL-2.0-only (`mcp-bash`).
3. Remove dual-license fallback references from `PhysicsForge` policy docs (`mcp-bash`).
4. Verify package metadata uses `GPL-2.0-only` in both repos (`mcp-ripgrep`).
5. Record licensing alignment assumptions in a migration decision note (`filesystem` or `mcp-bash`).
6. Create a legal-risk checklist for incoming files and snippets (`claims-evidence-executor`).
7. Add a pre-porting gate: reject direct code copy without provenance row (`mcp-ripgrep` + verifier script).
8. Add license consistency check command to migration runbook (`mcp-bash`).
9. Snapshot current repo states and branches for all three repos (`mcp-git`).

## Phase 2: Discovery and architecture mapping (10-18)
10. Build function-level inventory for Pantheon DDT modules (`mcp-ripgrep`).
11. Build class/function inventory for Pantheon elevated calibration modules (`mcp-ripgrep`).
12. Build PhysicsForge pipeline inventory for extraction and catalog semantics (`mcp-ripgrep`).
13. Map each inventoried unit to target crate/module in open_gororoba (`mcp-bash` + markdown matrix).
14. Identify overlap already covered in `data_core` and `cosmology_core` (`mcp-ripgrep`).
15. Identify overlap already covered in `docpipe` and `gororoba_cli` (`mcp-ripgrep`).
16. Identify overlap already covered in `algebra_core` to avoid redundant octonion ports (`mcp-ripgrep`).
17. Define migration boundaries: port, rewrite, skip, deprecate (`claims-evidence-executor`).
18. Publish source-to-target mapping table under `docs/tickets/` (`filesystem` or `mcp-bash`).

## Phase 3: New Rust crate scaffolding for Type Ia core (19-27)
19. Create crate skeleton `crates/snia_core` with `lib.rs` and module tree (`mcp-rust` + `mcp-bash`).
20. Add `snia_core` to workspace `Cargo.toml` members (`mcp-bash`).
21. Set crate dependencies via workspace dependencies only (`gororoba-rust-workflow`).
22. Add crate-level error types and unit-safe wrappers for physics constants (`mcp-rust`).
23. Add deterministic config structs equivalent to Pantheon `SimulationConfig` (`mcp-rust`).
24. Define canonical state vectors for hydro + thermodynamics (`mcp-rust`).
25. Add serialization feature gates for experiment snapshots (`serde`) (`mcp-rust`).
26. Create minimal smoke test for crate initialization (`mcp-rust cargo-test`).
27. Register crate in internal architecture docs and claims matrix links (`claims-evidence-executor`).

## Phase 4: EOS rebuild (28-36)
28. Implement degenerate electron EOS core routines in Rust (`eos_white_dwarf` parity target) (`mcp-rust`).
29. Implement ion and radiation pressure terms and composition hooks (`mcp-rust`).
30. Implement `eos_from_rho_T` analog with robust domain checks (`mcp-rust`).
31. Implement `temperature_from_rho_e` inversion with bounded iterations (`mcp-rust`).
32. Implement `eos_from_rho_e` analog and cached thermodynamic state (`mcp-rust`).
33. Implement sound-speed and effective-gamma routines with positivity constraints (`mcp-rust`).
34. Add unit tests mirroring Pantheon EOS behavior envelopes (`mcp-rust cargo-test`).
35. Add property tests for monotonicity and physical bounds (`mcp-rust` + proptest).
36. Add benchmark harness for EOS hot paths (`mcp-rust cargo-bench`).

## Phase 5: Hydro rebuild (HLLC + MUSCL) (37-45)
37. Implement limiter suite (minmod, superbee, MC) in Rust (`mcp-rust`).
38. Implement MUSCL reconstruction with ghost-cell boundary policy (`mcp-rust`).
39. Implement primitive/conserved conversions with invariant checks (`mcp-rust`).
40. Implement HLLC flux solver and wave-speed estimates (`mcp-rust`).
41. Implement CFL timestep estimator with configurable safety factors (`mcp-rust`).
42. Implement first RK2 hydro update API and integration hooks (`mcp-rust`).
43. Build Sod shock-tube regression tests against reference outputs (`mcp-rust cargo-test`).
44. Add numeric stability tests for near-vacuum edge cases (`mcp-rust`).
45. Add throughput benchmark for hydro update kernel (`mcp-rust cargo-bench`).

## Phase 6: Reactions, coupling, and yields (46-54)
46. Implement C12+C12 rate model and screening factor in Rust (`mcp-rust`).
47. Implement burn substep and subcycling control with fail-safe floors (`mcp-rust`).
48. Implement Chapman-Jouguet velocity estimator (`mcp-rust`).
49. Implement Strang splitting orchestrator (reaction-hydro-reaction) (`mcp-rust`).
50. Implement detonation detection metrics and event capture (`mcp-rust`).
51. Implement Ni56 yield estimator and observable mapping API (`mcp-rust`).
52. Add cross-check tests against Pantheon baseline runs (`mcp-bash` harness + `mcp-rust`).
53. Add deterministic replay tests from saved checkpoints (`mcp-rust`).
54. Add calibration parameter structs for elevated studies (`mcp-rust`).

## Phase 7: Elevated workflows and CLI integration (55-63)
55. Port DDT parameter scan workflow into Rust experiment module (`mcp-rust`).
56. Port alpha-chain network abstractions needed for yield calibration (`mcp-rust`).
57. Port light-curve synthesis interfaces with minimal physically consistent model (`mcp-rust`).
58. Add `gororoba_cli` subcommand for SN Ia runs (`mcp-rust` + `mcp-bash`).
59. Register CLI binary entry and docs (`gororoba-rust-workflow`).
60. Add CSV/TOML output schema for scan outputs and summaries (`mcp-rust`).
61. Add snapshot artifact writer under reproducible paths (`mcp-bash` + Rust fs APIs).
62. Add integration tests for CLI quick/full modes (`mcp-rust cargo-test`).
63. Add manpage/help examples for new CLI commands (`clap_mangen`, `mcp-rust`).

## Phase 8: PhysicsForge pipeline semantics into docpipe/CLI (64-72)
64. Extract semantics spec for equation extraction states (text, PDF text, OCR) (`mcp-ripgrep`).
65. Define Rust-native catalog schema equivalent to `equation_catalog.csv` essentials (`mcp-rust`).
66. Implement normalize/dedupe/signature logic in `docpipe` or companion module (`mcp-rust`).
67. Implement merge pipeline for multi-source equation streams (`mcp-rust`).
68. Implement framework/domain/category classification rules from current scripts (`mcp-rust`).
69. Implement parity and gap reports in `gororoba_cli` command set (`mcp-rust`).
70. Integrate with existing `extract-papers` output flow (`docpipe` + `gororoba_cli`) (`mcp-rust`).
71. Add regression tests using fixture PDFs and fixture extracted text (`mcp-rust cargo-test`).
72. Add compatibility converter for historical PhysicsForge CSV artifacts (`mcp-rust`).

## Phase 9: Claims, provenance, and evidence gates (73-81)
73. Add claim rows for each imported physics capability (`docs/CLAIMS_EVIDENCE_MATRIX.md`) (`claims-evidence-executor`).
74. Link each claim to source path and verification test path (`claims-evidence-executor`).
75. Add task rows in `docs/CLAIMS_TASKS.md` for each migration tranche (`claims-evidence-executor`).
76. Add provenance notes for all external references and equations corpus inputs (`data_core provenance`).
77. Add verifier script for source-to-target mapping completeness (`mcp-bash`).
78. Add verifier script for license-header consistency in migrated files (`mcp-bash`).
79. Wire verifiers into existing check targets and CI path (`mcp-bash` + `mcp-git`).
80. Add SQLite memoization for migration findings and unresolved risks (`mcp-sqlite`).
81. Publish phase audit report with pass/fail status and blockers (`docs/tickets`).

## Phase 10: Quality gates, overflow loops, and iterative todo updates (82-90)
82. Run crate-local test loops per phase before workspace-wide checks (`mcp-rust cargo-test`).
83. Run `cargo clippy --workspace -j$(nproc) -- -D warnings` at each phase gate (`mcp-rust` + `mcp-bash`).
84. Run workspace build and selected integration binaries (`mcp-rust cargo-build`).
85. Add performance baseline capture for EOS/hydro/reaction hot paths (`mcp-rust cargo-bench`).
86. If any phase fails, open overflow task batch `OF-<phase>-<n>` with owner and ETA (`claims-evidence-executor`).
87. Enforce overflow rule: max 5 active overflow tasks; new tasks require closure or deferral rationale (`sqlite` tracker).
88. Update migration todo board after every command batch with status, evidence path, and next action (`mcp-sqlite` + markdown sync).
89. Weekly rescope pass: re-rank remaining tasks by risk, physics impact, and integration dependency (`gororoba-mcp-orchestrator`).
90. Final acceptance gate: all claims linked, tests green, clippy clean, docs updated, and reproducible command transcript attached (`mcp-bash` + `mcp-rust` + `mcp-git`).

## Overflow management protocol (applies to all phases)
- Trigger: any failed gate, blocked dependency, or uncertainty over physical correctness.
- Record: create overflow ID, root cause, impacted tasks, rollback/safe state, and proposed fix.
- Timebox: one overflow cycle is capped at 2 focused sessions before escalation.
- Escalation: convert persistent overflow to explicit design decision record.
- Closure: overflow closes only after verifier/test evidence is attached.

## Iterative todo update protocol
- After each 5-task block, update status as `todo`, `in_progress`, `blocked`, or `done`.
- Every status change must include one evidence pointer (test output, diff path, or report path).
- Maintain one canonical TOML todo index in `registry/pantheon_physicsforge_migration_todo.toml`.
- Keep `docs/tickets/INDEX.md` reserved for auto-generated claim ticket mirrors.

## Runbook commands (operational)
- `python3 src/verification/verify_pantheon_physicsforge_license_consistency.py`
- `python3 src/verification/verify_pantheon_physicsforge_provenance_gate.py`
- `cargo check --package snia_core --all-targets`
- `cargo test --package snia_core`
- `cargo clippy --package snia_core -- -D warnings`
- `cargo clippy --workspace -j$(nproc) -- -D warnings`
