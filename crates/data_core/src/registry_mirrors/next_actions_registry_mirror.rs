//! # Next Actions Registry Mirror
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: see authoritative source line below -->
//!
//! Authoritative source: `registry/next_actions.toml`.
//!
//! - Updated: 2026-02-10
//! - Status: `active`
//!
//! ## Priority Queue
//!
//! ### NA-001 (high): Wire Optics into Warp Ring
//!
//! - Status: `done`
//! - Description: Connect `crates/optics_core` (GRIN raytracing, Kerr nonlinearity) to `warp_ring_integration.rs`. DONE: EngineGrinMedium in warp_ring.rs, ray trace integrated into WarpRingPipeline::execute(), ray_deviation recorded. Sprint 51.
//! - References:
//!   - `crates/optics_core`
//!
//! ### NA-002 (medium): Implement Betti Numbers
//!
//! - Status: `done`
//! - Description: Add homology computation to `stats_core/hypergraph.rs`. DONE: SimplicialComplex with Z_2 Gaussian elimination in stats_core/homology.rs, betti_k() for arbitrary dimension. Sprint 51.
//! - References:
//!
//! ### NA-003 (medium): Registry Validation Script
//!
//! - Status: `done`
//! - Description: Create a Rust test or script that deserializes all `registry/*.toml` files using `data_core::registry` structs to ensure schema compliance. DONE: registry-check binary extended with experiment-to-binary cross-reference check, unified status allowlist. Sprint 51.
//! - References:
//!
//! ### NA-004 (low): Refine Kerr Lens Model
//!
//! - Status: `done`
//! - Description: Duplicate of T-003; superseded by algebraic_lensing.rs (Sprint 54). T-003 DONE.
//! - References:
//!
//! ### NA-005 (high): C-010 Cluster Projection Gate
//!
//! - Status: `done`
//! - Description: Keep C-010 falsifiability gates strict by requiring cross-cluster-only absorber bridges, unique bridge pairs, and explicit projected-cluster connectivity checks. DONE: ProjectionGate struct with evaluate(), c010_default() thresholds (isolation>=0.95, xtalk<=-20dB, dominance>=0.02). Sprint 51.
//! - References:
//!
//! ### NA-006 (high): C-010 Non-Local Coupler Baseline
//!
//! - Status: `done`
//! - Description: Define a benchmark protocol comparing local connected absorbers against explicit non-local coupler designs on the same mode-suppression metrics. DONE: entropy-trap binary with compare/sweep/gate subcommands, ProjectionGate CI integration (exit 0/1). Sprint 51.
//! - References:
//!
//! ### NA-007 (high): C-011 Bridge Contrast Registry
//!
//! - Status: `done`
//! - Description: Track and verify gamma-invariant vacuum-to-shell contrast rho_v/rho_shell in gravastar bridge datasets as a fixed obstruction-side invariant. DONE: BridgeContrast struct, compute_bridge_contrast(), compare_contrast_ratios(), assert_contrast_gate() in bypass_models.rs. Sprint 51.
//! - References:
//!
//! ### NA-008 (high): C-011 Stress-Energy Mapping Prototype
//!
//! - Status: `done`
//! - Description: Prototype one explicit stress-energy bypass candidate (A-infinity, associative surrogate, or restricted sector) and test against TOV-consistent observables. DONE: BypassStressEnergy struct with DEC/NEC energy condition checks, bypass_stress_energy() interpolation in bypass_models.rs. Sprint 51.
//! - References:
//!
//! ### NA-009 (medium): C-011 Radial Margin Regression
//!
//! - Status: `done`
//! - Description: Promote strong-negative radial branch margin checks into recurring regression gates to detect accidental stabilizing drift in scan outputs. DONE: RadialMarginRegression struct, check_margin_drift() pass/fail gate in bypass_models.rs. Sprint 51.
//! - References:
//!
//! ### NA-010 (high): C-453 Prefix-Chain Formalization
//!
//! - Status: `done`
//! - Description: Promote lexicographic prefix-cut behavior of Lambda_256 < Lambda_512 < Lambda_1024 < Lambda_2048 from empirical test gate to explicit theorem statement. DONE: Proof skeleton PRS-00736 added (Sprint 42).
//! - References:
//!
//! ### NA-011 (high): 8D Embedding Proof Skeleton
//!
//! - Status: `done`
//! - Description: Draft a proof skeleton linking 8D codomain lock, injectivity, and filtration growth deltas to an octonion-driven structural argument. DONE: Proof skeleton PRS-00737 added (Sprint 42).
//! - References:
//!
//! ### NA-012 (high): Thesis 4 Source Triangulation
//!
//! - Status: `done`
//! - Description: Triangulate Thesis 4 against primary sources (Baez 2002, Bott-Milnor 1958, Adams 1960, and Cayley-Dickson subalgebra literature) in canonical TOML narratives. DONE: thesis4_source_triangulation.toml + BIB-0189/BIB-0190 added (Sprint 42).
//! - References:
//!
//! ### NA-013 (medium): Claim Status Vocabulary Guardrail
//!
//! - Status: `done`
//! - Description: Add a pre-commit guardrail ensuring new claim status tokens match the registry-check allowlist to prevent drift. DONE: claims-consolidate normalize stage + schema.rs TOML_CLAIM_STATUSES (16 canonical tokens). Status vocabulary unified (Sprint 40).
//! - References:
//!
//! ### NA-014 (medium): Binary Registry Drift Gate
//!
//! - Status: `done`
//! - Description: Add a targeted check that `registry/binaries.toml` stays synchronized with `crates/gororoba_cli/Cargo.toml` bin declarations. DONE: Cargo bin parity reconciled and registry-check no longer reports binary drift.
//! - References:
//!   - `crates/gororoba_cli/Cargo.toml`
//!   - `registry/binaries.toml`
//!
//! ### NA-015 (medium): Project Counter Auto-Sync
//!
//! - Status: `done`
//! - Description: Automate project metadata counters in `registry/project.toml` to avoid manual drift for claims, insights, experiments, and binaries. DONE: project-counter-sync reconciles counters and check mode confirms zero drift.
//! - References:
//!   - `registry/project.toml`
//!
//! ### NA-016 (high): Thesis 2-3-4 Narrative Pack
//!
//! - Status: `done`
//! - Description: Publish a single generated narrative pack documenting C-010, C-011, C-452, and C-453 falsifiability criteria, current obstruction state, and next verification checkpoints.
//! - References:
//!
//! ### NA-017 (high): T1 Scalar-Cassini Sweep Gate
//!
//! - Status: `done`
//! - Description: Run scalar-imbalance to TOV sweeps and record explicit Cassini-pass and Cassini-refute regimes for omega_eff.
//! - References:
//!
//! ### NA-018 (high): T2 Non-Newtonian Threshold Regression
//!
//! - Status: `done`
//! - Description: Validate associator-driven BGK thickening under threshold sweeps and record regime transitions with reproducible output artifacts. DONE: ratio=1.254 at 64^3, shear thickening confirmed (Sprint 40).
//! - References:
//!
//! ### NA-019 (high): T3 Plateau-to-Epoch Mapping
//!
//! - Status: `done`
//! - Description: Map neural homotopy loss plateaus to cosmological epochs and flag mismatch segments as direct refutation evidence. DONE: Burn neural correction reduces pentagon violation 78% (2.50->0.547). Three-way evidence in E-029 (Sprint 40).
//! - References:
//!
//! ### NA-020 (high): T4 Latency-Law Classifier Gate
//!
//! - Status: `done`
//! - Description: Run collision-storm latency classification across multiple bucket geometries and refute if inverse-square structure collapses to uniform or linear laws. DONE: 3D toroidal walk with sedenion keys, R2=0.9952, gamma=-2.41 power-law (Sprint 40).
//! - References:
//!
//! ### NA-021 (high): External Port License Gate
//!
//! - Status: `done`
//! - Description: Resolve licensing compatibility for GPL-2.0/GPL-3.0 sources and explicit license status for unlicensed repositories before direct imports.
//! - References:
//!
//! ### NA-022 (high): Documents Evidence Shortlist
//!
//! - Status: `done`
//! - Description: Shortlist and classify high-signal thesis evidence from the repo-owned imported document corpus into reproducible provenance lanes. DONE: 19 papers extracted via docpipe, 674 claims with where_stated, synthesis_final TOML artifacts (Sprint 40).
//! - References:
//!
//! ### NA-023 (high): Pure-Rust Port Boundaries
//!
//! - Status: `done`
//! - Description: Define clean-room pure-Rust module boundaries and acceptance tests for the highest-value external candidates.
//! - References:
//!
//! ### NA-024 (high): Strict Gate Revalidation Pass
//!
//! - Status: `done`
//! - Description: After roadmap normalization and status corrections, rerun strict gates and refresh synthesis report states. DONE: governance-gate and registry-acceptance-gate runtime validation pass after blocker fixes, strict typed-policy error mode passes, and RQ-024 planning canonicalization is complete.
//! - References:
//!
//! ### NA-025 (high): Studio Dynamic Registry Catalog
//!
//! - Status: `done`
//! - Description: Replace hard-coded thesis pipeline descriptors in gororoba-studio with registry-backed metadata to keep UI/API inventory synchronized.
//! - References:
//!
//! ### NA-026 (high): Studio Artifact Linking
//!
//! - Status: `done`
//! - Description: Expose links from studio run records to TOML/CSV artifact outputs so operators can jump directly from UI telemetry to evidence files. DONE: /api/artifacts/{*path} route with path traversal guard, addArtifactCell() in app.js, MIME inference. Sprint 51.
//! - References:
//!
//! ### NA-027 (medium): Studio Native Packaging Pilot
//!
//! - Status: `deferred`
//! - Description: Package gororoba-studio as a native shell pilot (desktop first, then Android/iOS webview wrappers) with reproducible build notes. Sprint 52: Deferred -- post-publication work, not blocking arXiv submission.
//! - References:
//!
//! ### NA-028 (high): Paper-Ready LaTeX Pipeline
//!
//! - Status: `done`
//! - Description: Build make latex target that compiles verified Four Theses results into a structured paper: hypotheses + tests + results format. Priority targets: Anti-Diagonal Parity Theorem (I-018), ultrametric hierarchy (I-013), synthesis engine (E-033).
//! - References:
//!
//! ### NA-029 (high): Evidence Artifact Packaging
//!
//! - Status: `done`
//! - Description: Package all Phase 1-4 evidence artifacts (E-027 through E-033) into a self-contained reproducible bundle with checksums, run commands, and expected outputs for peer review. DONE: evidence-package binary with check/bundle/verify subcommands, SHA-256 MANIFEST.toml generation. Sprint 51.
//! - References:
//!
//! ### NA-030 (medium): Claims Consolidation Maintenance
//!
//! - Status: `done`
//! - Description: Periodic re-run of claims-consolidate full after new claims are added. Verify idempotence and cross-link freshness. Target: run after each sprint that adds 5+ claims. DONE: idempotent run confirmed (Sprint 42).
//! - References:
//!
//! ### NA-031 (high): Sensational Primary Evidence Closure
//!
//! - Status: `done`
//! - Description: Close unresolved sensational mappings (RI-026, RI-027, RI-032) by obtaining direct provider citations or demoting them to non-evidence narrative references. DONE: All 3 resolved to primary sources (Sprint 42).
//! - References:
//!
//! ### NA-032 (high): Gate Remediation Tranche Canonicalization
//!
//! - Status: `done`
//! - Description: Represent the current push-gate cleanup as the active control-plane tranche instead of leaving it as scattered working-tree edits and stale gate reports. DONE: the tranche home is now CP-GATES-2026-03 across control-plane roadmap, roadmap, next-actions, and todo.
//! - References:
//!
//! ### NA-033 (high): TOML Inventory Scope Truth
//!
//! - Status: `done`
//! - Description: Keep the authoritative TOML inventory free of residue and scope drift so gate-ci-python reflects real control-plane health instead of stale cache paths. DONE: the inventory verifier is green, the builder now scans under the shared worker-budget fast path, and gate-ci-python has been revalidated successfully against the updated tracker state.
//! - References:
//!
//! ### NA-034 (high): Rust Regression Lane Stabilization
//!
//! - Status: `in_progress`
//! - Description: Finish the current Rust gate remediation as one bucketed stabilization pass: heavy-lane routing for research tests, integration-test binary discovery under nextest, the guarded attractor runtime budget, the repo-local scoped-routing policy that keeps local Rust verification off the full workspace path unless truly required, and the local-fast-path split that leaves heavy-package nextest authoritative in gate-ci-rust. Current refinement: local clippy now runs only --lib --tests and the light lane skips known heavy/GPU filters, but the workspace still expands many bin-test harnesses under nextest, so the next speed pass needs target-level nextest scope mapping instead of more generic concurrency tweaks.
//! - References:
//!
//! ### NA-035 (medium): Gate Audit and Mirror Truth Refresh
//!
//! - Status: `in_progress`
//! - Description: Refresh the generated markdown mirrors and produce a current keep-going gate audit so the repo reports today's gate frontier rather than stale March 7 summaries. Current checkpoint: mirrors have been refreshed, Python control-plane scans now use the new ripgrep-plus-bounded-worker fast path, the terminology gate now uses ripgrep candidate filtering, ambient Cargo tooling now builds under .cache/cargo-default-target, and the remaining step is the new gate-audit run after Rust revalidation.
//! - References:
//!
//! ### NA-036 (high): Final Required-Gate Acceptance
//!
//! - Status: `todo`
//! - Description: Collapse the current remediation tranche by rerunning gate-local, gate-ci-python, gate-ci-rust, and gate-audit sequentially, then mark CP-GATES-2026-03 complete only if the results and mirrors agree.
//! - References:
