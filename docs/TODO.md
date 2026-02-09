<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/todo.toml, registry/todo_narrative.toml -->

# TODO (updated 2026-02-08): executable, test-driven tracker

Primary trackers:
- `docs/ROADMAP.md` (consolidated roadmap: architecture, crates, migration history, GR port plan)
- `docs/ULTRA_ROADMAP.md` (test-driven, claim->evidence oriented)
- `registry/claims_tasks.toml` (claim-specific implementation backlog, authoritative)
- `docs/CLAIMS_TASKS.md` (generated mirror)
- `docs/NEXT_ACTIONS.md` (short priority queue)

Note:
- Python-to-Rust migration is COMPLETE (all 15 modules ported, gororoba_kernels removed).
- 1806 Rust tests (unit + integration + doc) pass, 0 clippy warnings.
- Historical planning docs (`RUST_MIGRATION_PLAN.md`, `RUST_REFACTOR_PLAN.md`, `docs/RESEARCH_ROADMAP.md`) deleted; content absorbed into `docs/ROADMAP.md`.
- Claims: C-001..C-475 (475 total). Insights: I-001..I-016.

## Current sprint: Sprint 13 -- TOML Centralization and Markdown Governance (IN PROGRESS)

- [x] Add central markdown knowledge index at `registry/knowledge_sources.toml`.
- [x] Add deterministic builder `src/scripts/analysis/build_knowledge_sources_registry.py`.
- [x] Wire `make registry` to regenerate knowledge index before `registry-check`.
- [x] Reconcile claims mirror wording (`docs/CLAIMS_EVIDENCE_MATRIX.md`) to match TOML source-of-truth policy.
- [x] Add raw-capture TOML corpus for non-generated markdown docs (`registry/knowledge/docs/*.toml`).
- [x] Define authoritative operational TOML registries (`registry/roadmap.toml`, `registry/todo.toml`, `registry/next_actions.toml`, `registry/requirements.toml`).
- [x] Add curated domain migration strategy (`registry/knowledge_migration_plan.toml`) to avoid blind bulk conversion.
- [x] Add TOML-first export path for insights mirror (`registry/insights.toml` -> `docs/generated/INSIGHTS_REGISTRY_MIRROR.md`).
- [x] Add TOML-first export path for experiments mirror (`registry/experiments.toml` -> `docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md`).
- [x] Add TOML-first markdown exporters for roadmap/todo/next-actions/requirements mirrors (`docs/generated/*_REGISTRY_MIRROR.md`).
- [x] Normalize `docs/CLAIMS_TASKS.md` into `registry/claims_tasks.toml` and generate `docs/generated/CLAIMS_TASKS_REGISTRY_MIRROR.md`.
- [x] Normalize `docs/claims/CLAIMS_DOMAIN_MAP.csv` + `docs/claims/by_domain/*.md` into `registry/claims_domains.toml` and generate `docs/generated/CLAIMS_DOMAINS_REGISTRY_MIRROR.md`.
- [x] Normalize `docs/tickets/*.md` into `registry/claim_tickets.toml` and generate `docs/generated/CLAIM_TICKETS_REGISTRY_MIRROR.md`.
- [x] Remove timestamp churn from registry generators to keep `make registry` deterministic across repeated runs.
- [x] Invert `make registry` to TOML-first flow; keep markdown->TOML ingest as explicit `make registry-ingest-legacy`.
- [x] Generate legacy claims-support mirrors (`docs/CLAIMS_TASKS.md`, `docs/claims/INDEX.md`, `docs/claims/CLAIMS_DOMAIN_MAP.csv`, `docs/claims/by_domain/*.md`) from TOML registries.
- [x] Add mirror freshness quality gate: `src/verification/verify_registry_mirror_freshness.py`.
- [ ] Decide if `docs/INSIGHTS.md` and `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md` should be fully replaced by generated mirrors or kept as narrative overlays.

## Current sprint: Sprint 10 -- De Marrais Emanation Architecture (COMPLETE)

### emanation.rs module (L1-L18, ~4400 lines, 113 tests)
- [x] L1: CDP signed-product engine with quadrant recursion
- [x] L2: Tone-row ordering and ET row/col label generation
- [x] L3: Exact DMZ cell test with 4-product X-pattern
- [x] L4: ET sparsity spectroscopy with per-strut regime detection
- [x] L5: Twist product mechanics with H*/V* operations
- [x] L6: Twisted Sisters PSL(2,7) navigation graph
- [x] L7: Lanyard taxonomy (Sails, TrayRacks, Blues, Quincunx, BicycleChain)
- [x] L8: Trip Sync and quaternion copy decomposition within Sails
- [x] L9: Semiotic Square with strut-opposite arithmetic kernel
- [x] L10: CT boundary detection (strutted ET edge-type transitions)
- [x] L11: Loop duality via sail-loop partition (automorpheme membership)
- [x] L12: Quincunx 5-assessor cross-linking construction
- [x] L13: Sky meta-fractal (strutted ET sparsity scaling across dims)
- [x] L14: Eco echo (semiotic square replication probe)
- [x] L15: Oriented Trip Sync (PSL(2,7) shorthand pattern verification)
- [x] L16: ET-lanyard dictionary (signed adjacency graph + state-machine traversal)
- [x] L17: Delta transition function (XOR strut pairs, reachability)
- [x] L18: Brocade/slipcover normalization (4 relabelings per BK)

### Claims and Insights
- [x] Claims C-468..C-475 added (8 new)
- [x] Insight I-016 written (De Marrais Emanation Architecture)

### Quality gates
- [x] `cargo clippy --workspace -j$(nproc) -- -D warnings` (0 warnings)
- [x] `cargo test --workspace -j$(nproc)` (1806 pass, 0 fail)

## Previous sprint: Sprint 9 -- Lattice Codebook Filtration (COMPLETE)

### Monograph theses (8/8 verified)
- [x] Thesis A: Codebook parity (C-458) -- all 4 dims verified
- [x] Thesis B: Filtration nesting (C-459) -- strict subset chain
- [x] Thesis C: Prefix-cut characterization (C-460) -- ALL transitions are lex prefix cuts
- [x] Thesis D: Scalar shadow action (C-465) -- addition mode verified; multiplication coupling open (C-466)
- [x] Thesis E: XOR partner law (C-462) -- partner(i) = i XOR (N/16) at dim=64
- [x] Thesis F: Parity-clique law (C-463) -- ZD adj = K_m union K_m at dims 16, 32
- [x] Thesis G: Spectral fingerprints (C-464) -- eigenvalues distinguish all motif classes
- [x] Thesis H: Null-model identity (C-467) -- rotation=identity, column-shuffle=informative

### Conversation extraction
- [x] 3 new docs in docs/external_sources/ (inverse CD, wheel taxonomy, sedenion ZD)
- [x] convos/ fully processed and removed (12 files)

### Novel discoveries
- [x] Lex prefix cuts (simpler than decision-trie rules)
- [x] S_base = 2187 = 3^7 (base universe, 139 excluded from Lambda_2048)
- [x] Lambda_32 = pinned corner (first 4 coords = -1)
- [x] dim=256 ZD adjacency documented as computationally infeasible

### Documentation
- [x] Claims C-458..C-467 added (10 new)
- [x] Insight I-015 written
- [x] CAYLEY_DICKSON_DATA_PROVENANCE.md updated (filtration diagram, adjacency predicates)
- [x] ROADMAP.md Section 7.9 added

### Quality gates
- [x] `cargo clippy --workspace -j$(nproc) -- -D warnings` (0 warnings)
- [x] `cargo test --workspace -j$(nproc)` (1693 pass, 0 fail)

## Previous sprint: Sprint 8 -- Consolidation, Audit, and Buildout (COMPLETE)

- [x] Casimir nonadditivity correction (Bimonte 2012 derivative expansion)
- [x] PEPS boundary MPS entropy (exact for small, boundary MPS for large)
- [x] CdMultTable generator (O(1) basis multiplication lookup)
- [x] Hartigan dip test (GCM/LCM algorithm + permutation p-value)
- [x] mass-clumping analysis binary (GWTC-3 + dip test, N>=50 guard)
- [x] EXPERIMENTS_PORTFOLIO_SHORTLIST.md (10 experiments with run commands)
- [x] MATH_CONVENTIONS.md (9 conventions with code references)
- [x] Primary-source citation sweep (10 claims)
- [x] Document hierarchy + binary contracts table in AGENTS.md
- [x] License: GPL-3.0 -> GPL-2.0-only
- [x] Cubic Anomaly (I-012): dim=32 8/7 split needs degree-3 GF(2) polynomial
- [x] External data cross-validation (I-014): 68 files, strut table verified, 9 CSVs
- [x] `cargo test --workspace` (1670 pass, 0 fail, 0 clippy warnings)

## Previous sprint: Sprint 6-7 (completed)
- [x] Source-first research passes (no deletions; only append/clarify sources):
  - [x] Wheels (division-by-zero) vs wheel graphs vs wheeled operads (CX-017). Already in wheels.rs + doc.
  - [x] de Marrais primary sources + terminology alignment (CX-002). Audit complete: dim=16 terms fully aligned, higher-dim generic naming is correct.
  - [x] Reggiani alignment (Z(S), ZD(S)) and repo terminology alignment. Audit complete: is_reggiani_zd() correct, 84 standard ZDs, nullity-4 verified, sign variants orthogonal (C-005 empirical correction documented).
  - [x] Fractional Laplacian sources: Riesz vs spectral vs extension (CX-004). Audit complete: periodic=Riesz, Dirichlet=spectral correctly labeled. C-S doc clarified. All 5 sources cited.
  - [x] p-adic operator sources (Vladimirov/Kozyrev) (CX-006). Audit complete: foundations correct (14 tests), Vladimirov/Kozyrev correctly deferred, sources cited in BIBLIOGRAPHY.md.
  - [x] Exceptional/Jordan/Albert references to correct overclaims (CX-007). Audit complete: no overclaims found. nilpotent_orbits.rs is generic (no exceptional claims), magic square verified, H3(O) correctly deferred.
- [x] Execute missing dataset pillars roadmap from `docs/ULTRA_ROADMAP.md` Section H (21/21 done, 1464 tests).
- [x] Implementation (keep every warning as an error):
  - [x] Wheels axioms checker + unit tests (CX-017). Already in wheels.rs (WheelQ, 8 axioms, 10 tests).
  - [x] XOR-balanced search extension + tests (CX-003). 10 new tests.
  - [x] Motif census: exact to 256D with scaling laws (CX-002). 16 new tests.
  - [x] Visualization hygiene and artifact saving policy (CX-019). plt.show() -> plt.close() in 2 files; no MathText/grid/deprecation issues found.
- [x] `cargo clippy --workspace -j$(nproc) -- -D warnings`
- [x] `cargo test --workspace -j$(nproc)` (1628 tests, 0 failures)
- [x] `make ascii-check` (5 files fixed, extended replacement table + skip lists)

## Completed (keep for provenance)
- [x] Sprint 10: De Marrais Emanation Architecture -- L1-L18 implemented, 113 emanation tests, 8 claims (C-468..C-475), I-016. 1806 tests.
- [x] Sprint 9: Lattice codebook filtration -- 8 theses verified, 10 claims (C-458..C-467), I-015. 1693 tests.
- [x] Sprint 8: Consolidation -- mult table, dip test, Casimir DE, PEPS entropy, I-012/I-014. 1670 tests.
- [x] Package `gemini_physics` and add tests (`pyproject.toml`, `src/gemini_physics/`, `tests/`).
- [x] Coq stub builds via `make coq` (see `curated/01_theory_frameworks/README_COQ.md`).
- [x] Materials ingestion and embedding benchmarks.
- [x] Python-to-Rust migration: all 15 modules ported.
- [x] Consolidate PyO3 bridges: gororoba_kernels removed, gororoba_py is sole bridge.
- [x] Real cosmology fitting: Pantheon+ + DESI DR1 BAO (Omega_m=0.3284, H0=71.65).
- [x] Ultrametric analysis directions C-436 through C-440 completed.
- [x] GPU ultrametric exploration: 82/472 sig at FDR<0.05, 9 catalogs (I-011).
- [x] Blackhole C++ port: COMPLETE (18 gr_core modules, 394 tests).
- [x] CUDA compute engine: cudarc 0.19.1, 10M triples/test on RTX 4070 Ti.
- [x] Deleted stale planning docs: RUST_MIGRATION_PLAN.md, RUST_REFACTOR_PLAN.md, RESEARCH_ROADMAP.md.
- [x] Motif census extended to dim=64/128/256 exact. 5 scaling laws verified, 16 new tests.
- [x] XOR-balanced search extension (CX-003): mixed 2-blade/4-blade graph, necessity statistics. 10 new tests.
- [x] Source-first research passes: 6/6 complete (CX-017, CX-002, Reggiani, CX-004, CX-006, CX-007). No overclaims found. C-S doc clarified in spectral_core.
- [x] Dataset pillar validation: +33 new tests in data_core (Union3, Swarm, EHT, GFC, SPK, Landsat STAC, SORCE/TSIS overlap, provider verifier). 21/21 Section H items done.
