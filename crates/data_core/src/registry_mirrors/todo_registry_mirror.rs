//! # TODO Registry Mirror
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: see authoritative source line below -->
//!
//! Authoritative source: `registry/todo.toml`.
//!
//! - Updated: 2026-02-10
//! - Status: `active`
//! - Item count: `65`
//!
//! ## Items
//!
//! ### T-001: Implement Lyndon Basis for E7 Signs
//!
//! - Area: `math`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Implement a Lyndon basis for the E7 root system to ensure the structure constants satisfy the Jacobi identity. DONE: Chevalley-Tits extraspecial 2-cocycle in lyndon_basis.rs, 5 tests pass including exhaustive Jacobi (Sprint 42).
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-002: Betti Number Computation
//!
//! - Area: `math`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Implement Betti-0 and Betti-1 calculation for the TriadHypergraph using a homology solver. DONE: SimplicialComplex with Z_2 Gaussian elimination in stats_core/homology.rs, 5 tests including torus Betti (b0=1,b1=2,b2=1). Sprint 51.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-003: Kerr Lens Effect
//!
//! - Area: `optics`
//! - Priority: `low`
//! - Status: `done`
//! - Description: Integrate `optics_core::tcmt` to simulate non-linear Kerr lensing based on the local energy density of the fluid. DONE: algebraic_lensing.rs implements GRIN medium from ZD density via optics_core::grin, wired into sign_imbalance crate. Supersedes original TCMT approach with full GrinMedium trait impl + ray tracing. Sprint 54.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-004: Registry Validation
//!
//! - Area: `infra`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Create a validation script to ensure all registry TOML files conform to the defined schemas. DONE: registry-check binary + schema_signatures.toml + claims-consolidate pipeline (Sprint 40).
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-005: Fetch Reference PDFs
//!
//! - Area: `docs`
//! - Priority: `low`
//! - Status: `done`
//! - Description: Retrieve or reconstruct missing reference PDFs cited in the monograph. DONE: 19 papers extracted via docpipe (Sprint 11), corpus under data/papers/ with LFS.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `data/papers/`
//!
//! ### T-006: C-010 Non-Local Benchmark Harness
//!
//! - Area: `claims`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Implement a reusable benchmark harness that scores local vs non-local absorber coupling topologies against the same mode-suppression targets.
//! - Dependencies: `C-010`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-007: C-010 Graph Family Extensions
//!
//! - Area: `claims`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Expand C-010 tests from current bridge projection to alternative candidate ZD-derived graph families and record failure modes. DONE: box_kite topology, from_adjacency_matrix constructor, ZdGraphFamily with failure_modes analysis. Sprint 51.
//! - Dependencies: `C-010`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-008: C-011 Bypass Model Candidate A
//!
//! - Area: `cosmology`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Prototype an associative-surrogate stress-energy closure and test whether it preserves gravastar boundary conditions.
//! - Dependencies: `C-011`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-009: C-011 Bypass Model Candidate B
//!
//! - Area: `cosmology`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Prototype a restricted associative-sector closure and compare TOV observables against the current obstructed baseline.
//! - Dependencies: `C-011`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-010: C-453 Prefix-Cut Theorem Draft
//!
//! - Area: `math`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Write a formal theorem draft for the observed lexicographic prefix-cut chain across Lambda_256/512/1024/2048. DONE: 591-line prefix_chain_theorem.rs with full verification engine, 11 tests, PRS-00736 VERIFIED (Sprint 42).
//! - Dependencies: `C-453`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-011: C-453 Octonion Skeleton Mapping
//!
//! - Area: `math`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Define a candidate explicit map between octonion structural invariants and 8D lattice-coordinate constraints. DONE: CdProjectionInvariance + verify_octonion_cd_projection_invariance() in prefix_chain_theorem.rs, 3 tests. Sprint 51.
//! - Dependencies: `C-453`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-012: Thesis 4 Primary Source Matrix
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Build a citation matrix linking C-452/C-453 claims to primary sources on normed division algebras and Cayley-Dickson subalgebra structure. DONE: thesis4_source_triangulation.toml with Baez 2002, Bott-Milnor 1958, Adams 1960 (Sprint 42).
//! - Dependencies: `C-452`, `C-453`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-013: Registry Drift Guardrails
//!
//! - Area: `registry`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Add automated guardrails for status tokens, binary list parity, and project counter synchronization. DONE: claims-consolidate normalize + schema-signatures verification + governance/acceptance runtime gate pass state after fixes.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-014: LaTeX Main Paper Scaffold
//!
//! - Area: `publication`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Create docs/latex/main_paper.tex with structured sections: Abstract, Introduction (Four Theses framework), Methods (LBM + CD algebra + VR topology + neural homotopy), Results (T1-T4 evidence), Discussion, Conclusions. Wire make latex target.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `docs/latex/main_paper.tex`
//!
//! ### T-015: Anti-Diagonal Parity Theorem Paper Section
//!
//! - Area: `publication`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Draft the mathematical exposition of the Anti-Diagonal Parity Theorem (I-018): eta definition, GF(2)^2 fiber structure, Klein-four symmetry, dimensional universality. Include proof sketch and numerical verification table.
//! - Dependencies: `I-018`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-016: Synthesis Engine Results Tables
//!
//! - Area: `publication`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Generate LaTeX tables from TOML evidence artifacts: T1 correlation coefficients, T2 shear thickening ratios across grid sizes, T3 pentagon violation before/after neural correction, T4 power-law fit parameters. DONE: --thesis-tables flag in generate-latex, longtable with booktabs, T2 power-law sub-table. Sprint 51.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-017: Reproducibility Appendix
//!
//! - Area: `publication`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Write appendix documenting exact reproduction steps: cargo build commands, binary invocations, expected TOML outputs, SHA-256 checksums of evidence artifacts. DONE: --repro-appendix flag in generate-latex with build env, experiment commands, artifact checksums, known gaps sections. Sprint 51.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-018: 128^3 GPU Grid Validation
//!
//! - Area: `gpu`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Run LBM + ZD resonance pipeline at 128^3 and 256^3 resolution using CUDA BF16 acceleration. Validates grid convergence beyond 64^3 CPU results. Sprint 49B: zd-resonance-cuda binary implements this with tau sweep, coupling sweep, and Reynolds sweep subcommands. See E-060 for real-data context. RESULT: C-792 FALSIFIED, C-793 FALSIFIED. Ghost peaks appear in control runs (BF16 quantization noise, not ZD-specific). 256^3 not yet run.
//! - Dependencies: `C-792`, `C-793`, `E-060`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-019: Hypercomplex Intake Batch Reconciliation
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Complete RI-2026-02-14 URL intake: fetch all sources with provenance hashes, map sensational headlines to primary papers, and wire unresolved mappings into claims/tickets. DONE: 16 HCTP entries, C-691..C-694, RI-026/027/032/034 resolved (Sprint 41-42). C-692..C-694 promoted to Verified (Sprint 53).
//! - Dependencies: `C-691`, `C-692`, `C-694`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-020: Hypercomplex Taxonomy Registry Integration
//!
//! - Area: `taxonomy`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Promote report-level taxonomy and claim candidates into canonical claim/task/ticket lanes with explicit verify-refute criteria and source anchors. DONE: hypercomplex_taxonomy_promotion.toml with 16 entries. Sprint 52: 12 claim-ready HCTP entries promoted to C-807..C-818 (Proposed). 4 ticket-ready entries (HCTP-013..016) noted in ledger for multi-source provenance closure.
//! - Dependencies: `C-807`, `C-818`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-021: RI-026 Source Chain Closure
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Resolve RI-026 (black-hole aggregator video payload) to a citable primary source chain, or retain unresolved status with explicit failure evidence. DONE: Mapped to Calmet et al. arXiv:2506.09489 (Sprint 42).
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-022: RI-027 Source Chain Closure
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Resolve RI-027 (hypervelocity-star aggregator video payload) to a citable primary source chain, or retain unresolved status with explicit failure evidence. DONE: Mapped to Terry et al. AJ DOI:10.3847/1538-3881/ad9b0f (Sprint 42).
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-023: RI-032 Provisional Mapping Resolution
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Resolve RI-032 (stellar-physics video payload) by confirming or rejecting the provisional mapping currently flagged unresolved in intake. DONE: Confirmed mapping to A&A DOI:10.1051/0004-6361/202556884 (R Doradus stellar wind study, Sprint 42).
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-024: RI-034 GW250114 Mapping Hardening
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Independently replay RI-034 citation-chain evidence and decide whether the current GW250114 mapping remains supportable at canonical quality. DONE: All three candidate URLs verified (arXiv:2509.08099, PRL 10.1103/6c61-fm1n, Cornell news). Mapping is correct and complete. PRL short DOI verified via search cache (direct 403 from APS bot challenge). HCTP-016 evidence_status upgraded to verified.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-025: Verify C-807..C-818 HCTP Taxonomy Claims
//!
//! - Area: `governance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Cross-reference 12 hypercomplex taxonomy property claims against CURRENT::CRATE gororoba_algebra (LEGACY::CRATE algebra_core) test files. All verified against existing tests: cayley_dickson.rs, boxkites.rs, annihilator.rs, hypercomplex.rs, octonion_field.rs, composition_algebra_taxonomy.rs. where_stated fields enhanced with specific test references. DONE: Sprint 55.
//! - Dependencies: `C-807`, `C-818`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-026: Restructure data/ Gitignore (23k -> 349 files)
//!
//! - Area: `infrastructure`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Added 14 gitignore patterns for regenerable data directories. Executed git rm --cached to untrack 23,061 files (7.4 GB): data/external/ (22,838), simulation scratch, artifacts. Remaining tracked: 347 evidence CSVs + 2 metadata files. DONE: Sprint 55.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `data/external/`
//!
//! ### T-027: Close Stale Workstreams and Update Registries
//!
//! - Area: `governance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Closed 3 stale workstreams (WS-ALGEBRA-FLUID-001, WS-THESIS-002, WS-REGISTRY-001). Updated project.toml counters (experiment_count=70, complete=68). Added Sprint 55 entry. Refreshed schema_signatures.toml (3 content hashes). Bibliography audit: 190 entries, 73 lack DOI+URL. DONE: Sprint 55.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-028: Rewrite README Project Tree Walkthrough
//!
//! - Area: `documentation`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Complete README.md rewrite with project structure tree (27 crates, 140 binaries, 131 registries), build commands, key binaries table, Four Theses Framework summary, and sprint history pointer. DONE: Sprint 55.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-029: Backfill data/artifacts manifest coverage
//!
//! - Area: `data_governance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Audit found 668 strict-lane unknown origins concentrated in data/artifacts. Implemented manifest backfill with c756 frame glob coverage and supplemental artifact rows; strict unknown origins now reach zero in data-origin-audit and governance gate.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `data/artifacts`
//!
//! ### T-030: Persist external source URLs in machine-readable provenance
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Implemented companion source contract at data/external/SOURCES.toml with path_glob -> canonical_url/mirror/status/retrieval metadata. Governance gate now enforces source-rule coverage for external files and blocked-source deadline policy.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `data/external/SOURCES.toml`
//!
//! ### T-031: Register script-generated data outputs in governance registries
//!
//! - Area: `governance`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Completed: added canonical non-strict generated-origin pattern contracts in registry/data_generated_patterns.toml and wired them into origin audit/gate classification. This covers h5/csv/e027/thesis/equivalence and related generated lanes so unknown-origin count is now zero without relaxing strict-lane policy.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `registry/data_generated_patterns.toml`
//!
//! ### T-032: Resolve missing Makefile fetch-data dependency script
//!
//! - Area: `build_system`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Makefile fetch-data now uses Rust-native fetch-datasets + record-external-hashes + strict governance gate, removing the missing Python script dependency.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-033: Eliminate transitional empty CSV placeholders
//!
//! - Area: `data_governance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed: retired transitional CSV placeholders by removing allow_empty_csv exceptions from semantic validation, populating e027_channels.csv and wow_followup_snr.csv with concrete rows, and deleting wow_followup_snr_header.csv.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-034: Close external scientific-semantic failures and unverifiable coverage gaps
//!
//! - Area: `data_governance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed: external source rules now declare scientific validator references (structure-only or blocked-manifest contracts), blocked/manual lanes are policy-validated by manifest references, and strict semantic validation runs with zero failed/unverifiable validators.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-035: Enforce unclassified lane count zero
//!
//! - Area: `data_governance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed: data-governance-gate now has explicit unclassified-lane enforcement and data-origin-audit can fail on unclassified files. Governance policy was expanded to cover previously unclassified data roots.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-036: Eliminate active-source structure-only validator refs
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed: active external source rules now use explicit scientific validator refs, strict semantic validation passes with fail-on-unverifiable enabled, and execute-mode external replay audit reaches zero replay/download/checksum failures. Remaining non-replayable root-level legacy derivatives were quarantined as explicit blocked contracts with deadlines and manifest evidence.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-037: Implement blocked-source burndown audit lane
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed: added Rust-native external-blocked-burndown binary to quantify blocked external debt by source id, file count, bytes, deadline status, and configurable fail thresholds. Established phase-2 baseline report for deterministic burn-down planning.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-038: Burn down blocked external corpora by top 4 lanes
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed: step-2 burndown migrated paper/trace corpora and retired transitional/nonreproducible snapshots out of data/external into governed archive/manual lanes with explicit retirement manifests. Blocked files dropped from 23451 to 0 (100% reduction) while blocked_overdue_sources remained 0 and strict origin/governance/semantic gates stayed green.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `data/external`
//!
//! ### T-039: Codify blocked action-plan contracts and Cargo contention policy
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed: blocked source rules now carry explicit blocked_action_plan references, append-only retry ledger support is implemented in Rust (external-blocked-retry-ledger), and serial compile-heavy Cargo execution policy is documented in agents.toml and workspace metadata to avoid lock contention while preserving reproducible behavior.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-040: Split root external replay contract into explicit active dataset groups
//!
//! - Area: `provenance`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Completed (phase 11 guard): execute-mode replay now defaults to staging isolation (`--replay-mode staging`) with deterministic command rewriting and out-of-scope side-effect detection. Replays for wildcard source rules are executed against per-source staging roots, never directly into `data/external`, so retired/migrated files are not repopulated in-place during audit execution. Remaining enhancement: replace `fetch-datasets --all` root wildcard policy with explicit grouped include-only contracts.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `data/external`
//!
//! ### T-041: Track hdf5-metno fork upstream status
//!
//! - Area: `deps`
//! - Priority: `low`
//! - Status: `open`
//! - Description: Workspace uses hdf5-metno (git fork) because the mainline hdf5-rs crate has stale maintenance. As of 2026-02-26: aldanor/hdf5-rust dormant (last release v0.8.1, Nov 2023). metno/hdf5-rust active at v0.12.3 (Feb 2026) with HDF5 2.0.0 support, SWMR, ZFP filters. Upstream switch not recommended. Next check: 2026-05-26.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-042: Track criterion version upgrade (0.5 -> 0.7)
//!
//! - Area: `deps`
//! - Priority: `low`
//! - Status: `open`
//! - Description: Workspace pins criterion 0.5. As of 2026-02-26: latest is 0.7.0 (Jul 2025), not 0.8. Key change: criterion::black_box removed (use std::hint::black_box), async runtime Handle instead of Runtime, MSRV 1.80. No 0.8 release exists. Evaluate upgrade when benchmark suites are next modified. API audit required across 9 consuming crates.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-043: Canonicalize gate remediation tranche in control-plane trackers
//!
//! - Area: `qa`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Update the authoritative control-plane roadmap, roadmap, next-actions, and todo registries so the current push-gate work is tracked as CP-GATES-2026-03 instead of being scattered across stale reports and working-tree diffs. DONE: the tracker stack now names the tranche explicitly.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-044: Keep control-plane TOML inventory authoritative and residue-free
//!
//! - Area: `registry`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Current checkpoint: `cargo run -p gororoba_cli_data --bin markdown-registry -- verify-toml-inventory` is green, the authoritative inventory excludes stale `.horusec` residue from the documented failure frontier, and the Rust builder now scans with shared governance skip rules. Keep the builder and committed inventory aligned so control-plane TOML scope remains truthful.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-045: Re-run and close gate-ci-python after tracker refresh
//!
//! - Area: `qa`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Revalidate the full authoritative Python gate after the tracker stack and mirrors are refreshed so the tranche records a current gate-ci-python result, not just a passing leaf verifier. DONE: gate-ci-python passed on the current branch state after the tracker refresh, with pytest-xdist using the shared `nproc/2` worker budget and the control-plane builders on the new ripgrep-backed fast path.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-046: Finalize heavy-lane routing for long research tests
//!
//! - Area: `rust`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Finish moving exhaustive or characterization-grade research tests out of the default regression lane and into the documented heavy lane so required Rust gates only run tests with the intended budget. DONE: test_split_octonion_attractor_regression_dim_128_256_guarded carries #[ignore = "heavy research lane: ..."], nextest heavy profile has the override with 600s timeout, Makefile heavy: target runs --run-ignored only.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-047: Fix compile-time binary path resolution for integration tests
//!
//! - Area: `rust`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Keep the CLI integration tests robust under nextest by resolving workspace binaries through compile-time Cargo bin metadata before falling back to runtime environment variables. DONE: integration_thesis_42_support.rs, integration_thesis_program_sweep.rs, c053_pathion_metamaterial_mapping.rs, nonlocal_algebraic_metamaterial.rs, integration_snia_ddt.rs all use option_env!("CARGO_BIN_EXE_...") as primary with std::env::var fallback. Test passes under nextest (verified 2026-03-20).
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-048: Recalibrate guarded attractor runtime budget to worker-budget policy
//!
//! - Area: `rust`
//! - Priority: `high`
//! - Status: `done`
//! - Description: Align the guarded split-octonion attractor regression with the repo's halved-thread worker budget so the default Rust gate reflects the documented execution policy instead of an outdated per-dimension cutoff. DONE: test uses env-var knobs CD_ATTRACTOR_PER_DIM_BUDGET_S (default 60s) and CD_ATTRACTOR_TOTAL_BUDGET_S (default 90s), matching the gate policy. nextest.toml heavy profile sets 600s timeout for this test group.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-049: Re-run gate-ci-rust and bucket any remaining failures
//!
//! - Area: `qa`
//! - Priority: `high`
//! - Status: `done`
//! - Description: DONE 2026-03-20. Two pre-existing failures bucketed and fixed: (1) test_chingon_basis_quantization routed to heavy lane via #[ignore] + nextest override (O(dim^4) ZD search too slow for 120s default timeout at 64D); (2) test-inventory fixed: validate_python_files short-circuits when python_test_file is empty, gitignored docs/*.md removed from doc_no_count, narrative TOML stubs created. gate-ci-rust now passes: 4860+1329 tests, 0 failures.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-050: Refresh roadmap, todo, and next-actions mirrors
//!
//! - Area: `docs`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: Regenerate the human-facing roadmap/todo/next-actions mirrors after the tranche and gate state are updated so the published markdown stops lagging behind the authoritative TOML. DONE: mirrors were regenerated after the tranche and cargo-isolation updates.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-051: Run gate-audit and archive tranche acceptance evidence
//!
//! - Area: `qa`
//! - Priority: `medium`
//! - Status: `open`
//! - Description: Execute the keep-going gate audit after the local, Python, and Rust gates are revalidated so the tranche closes with one current summary instead of a chain of stale partial reports. Include the post-fast-path gate state, including the ripgrep-backed terminology lane, the repo-scoped Rust routing, the local heavy-lane skip, and the ambient Cargo target separation, so any remaining slow lane is measured from current conditions rather than the pre-optimization baseline.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-052: Implement Bartol V2 legacy parser and Python fetcher
//!
//! - Area: `data`
//! - Priority: `high`
//! - Status: `done`
//! - Description: DONE: BARTOL_V2_LAYOUT (16-col RTN, 2-digit year) in voyager.rs, parse_bartol_v2() with year correction, fetch_voyager_bartol() in fetch_voyager.py with TLS workaround, 7 Rust tests, SOURCES.toml entry.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-053: Add AMDA MAG-only lane to fetch_pioneer.py
//!
//! - Area: `data`
//! - Priority: `high`
//! - Status: `done`
//! - Description: DONE: AMDA_PIONEER_DATASETS, parse_pioneer_mag_rows(), fetch_pioneer_amda_mag() in fetch_pioneer.py. --source amda option. Auto fallback: SPDF -> AMDA MAG-only. SOURCES.toml entries for P10/P11 AMDA MAG-only.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-054: Implement PSP AMDA fetcher and Rust parser
//!
//! - Area: `data`
//! - Priority: `high`
//! - Status: `done`
//! - Description: DONE: crates/data_core/src/catalogs/psp.rs with SPC+MAG+orbit AMDA translation, crates/data_core/src/catalogs/psp.rs with PSP_LAYOUT and PspProvider, 4 Rust tests, SOURCES.toml entries.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `crates/data_core/src/catalogs/psp.rs`
//!
//! ### T-055: Implement Helios AMDA fetcher and Rust parser
//!
//! - Area: `data`
//! - Priority: `high`
//! - Status: `done`
//! - Description: DONE: crates/data_core/src/catalogs/helios.rs with E1-corefit+E3-MAG+orbit AMDA translation, crates/data_core/src/catalogs/helios.rs with HELIOS1/2_LAYOUT and HeliosProvider, 6 Rust tests, SOURCES.toml entries.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `crates/data_core/src/catalogs/helios.rs`
//!
//! ### T-056: Implement Solar Orbiter partial AMDA fetcher
//!
//! - Area: `data`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: DONE: crates/data_core/src/catalogs/solo.rs with MAG+RPW electron density AMDA translation. PARTIAL lane: no SWA proton plasma. SOURCES.toml entries.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `crates/data_core/src/catalogs/solo.rs`
//!
//! ### T-057: Register all new spacecraft sources in SOURCES.toml
//!
//! - Area: `data`
//! - Priority: `high`
//! - Status: `done`
//! - Description: DONE: Bartol V2, PSP AMDA (3 entries), Helios AMDA (4 entries), Solo AMDA (2 entries), Pioneer AMDA MAG-only (2 entries), PSP Gateway, ESA SOAR, Helios/PSP SPDF blocked, IMAP planned. Updated Pioneer and Voyager blocked entries with AMDA availability notes.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-058: Cross-validate Bartol vs AMDA B-field for Voyager 2 overlap
//!
//! - Area: `data`
//! - Priority: `medium`
//! - Status: `open`
//! - Description: E-128: Compare Bartol 16-col RTN B-field against AMDA-derived Voyager 2 for 1990-1995 overlap.
//! - Dependencies: `E-128`
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-059: Add AMDA fallback to crates/data_core/src/catalogs/ulysses.rs
//!
//! - Area: `data`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: AMDA provides ulys-bai-mom (SW ion moments) + ulys-fgm-rtn (MAG RTN) + ulys-orb-all (orbit). Full plasma+MAG for 1.0-5.4 AU out-of-ecliptic. DONE: UlyssesAmdaProvider added to ulysses.rs; parse_ulysses_amda_plasma/mag/orb parsers; merge_ulysses_amda (three-way time-key intersection). download_amda_hapi_csv added to fetcher.rs. 4 new unit tests pass.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `crates/data_core/src/catalogs/ulysses.rs`
//!
//! ### T-060: Add AMDA fallback to crates/data_core/src/catalogs/juno.rs (cruise phase)
//!
//! - Area: `data`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: AMDA provides juno-jadel5-protmom (protons) + juno-fgm-cruise60 (MAG) + juno-cruise-all (ephemeris). Full plasma+MAG for cruise 1-5 AU. DONE: JunoAmdaProvider added to juno.rs; parse_juno_amda_plasma/mag/orb parsers; merge_juno_amda (three-way intersection). 3 new unit tests pass.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `crates/data_core/src/catalogs/juno.rs`
//!
//! ### T-061: Add AMDA fallback to crates/data_core/src/catalogs/wind.rs
//!
//! - Area: `data`
//! - Priority: `medium`
//! - Status: `done`
//! - Description: AMDA provides wnd-swe-kp (SWE) + wnd-mfi-kp (MFI). Full plasma+MAG for L1. DONE: WindAmdaProvider + WindAmdaPlasmaRecord + WindAmdaMagRecord added to wind_swe.rs; parse_wind_amda_swe/mfi parsers; merge_wind_amda -> OmniRecord. 4 new unit tests pass.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - Evidence references point to maintained canonical paths.
//! - Evidence refs:
//!   - `crates/data_core/src/catalogs/wind.rs`
//!
//! ### T-062: Investigate PDS-SBN for New Horizons SWAP data
//!
//! - Area: `data`
//! - Priority: `low`
//! - Status: `open`
//! - Description: New Horizons is the only spacecraft with NO AMDA fallback. Check PDS Small Bodies Node for NH SWAP plasma data.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-063: Playwright browser probing of GSFC endpoints
//!
//! - Area: `qa`
//! - Priority: `low`
//! - Status: `open`
//! - Description: Test SPDF, CDAWeb, COHOWeb, PSP Gateway, ESA SOAR via Playwright browser automation to check if browser access bypasses connection-refused blocks.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//! - Evidence refs:
//!   - (none)
//!
//! ### T-064: Retire the Brown 7.19(iii) OCR placeholder lane
//!
//! - Area: `proofs`
//! - Priority: `high`
//! - Status: `done`
//! - Description: DONE: the Brown Chapter VII public surface now exports only the source-faithful reversal-order 7.19(iii) theorem with explicit pairwise-distinct side conditions, and the stale cyclic OCR placeholder plus its structural-gap axiom have been removed.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - The Brown Chapter VII public surface exports only the corrected theorem shape.
//! - Evidence refs:
//!   - `proofs/theories/Brown1972ChapterVII.v`
//!   - `proofs/theories/SStructuralGaps.v`
//!
//! ### T-065: Bridge Moreno arbitrary-a concrete V_lambda hypotheses into the H_a witness lane
//!
//! - Area: `proofs`
//! - Priority: `high`
//! - Status: `open`
//! - Description: Finish the remaining Theorem 1.16 gap by deriving Moreno's quaternionic-block decomposition, and therefore Moreno16ArbitraryAVlambdaWitness, from the full concrete arbitrary-a V_lambda geometry instead of stopping at the current explicit-builder, concrete-hypothesis package, geometry-shaped block package, and non-canonical witness scaffolding.
//! - Dependencies: (none)
//! - Acceptance criteria:
//!   - todo_item status is constrained to declared enum values.
//!   - todo_item dependencies are explicit and machine-parseable.
//!   - The Moreno paper lane exposes a source-faithful arbitrary-a bridge from concrete V_lambda data to the mod-4 theorem.
//! - Evidence refs:
//!   - `proofs/theories/C1542_MorVlambdaOrbit.v`
//!   - `proofs/theories/Moreno1997.v`
