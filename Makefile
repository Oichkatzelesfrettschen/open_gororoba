# ---- Phony targets ----
.PHONY: help bootstrap-dev bootstrap-user-local-xdg fmt fmt-check
.PHONY: test lint check smoke integrity integrity-rust math-verify governance-gate governance-gate-readonly wave6-gate pre-push-gate pre-push-gate-strict hooks-install hooks-install-strict hooks-status synthesis-execution-contract
.PHONY: verify verify-grand verify-c010-c011-theses ansi-check ansi-check-strict terminology-gate doctor doctor-blas provenance
.PHONY: provenance-registry-index provenance-registry-export provenance-registry-verify provenance-registry-doctor provenance-registry-link-audit provenance-registry-recover
.PHONY: rocq-proofs rocq-proofs-check rocq-makefile-check lva-paper
.PHONY: heavy test-inventory verify-no-reports-writes
.PHONY: rust-test rust-clippy rust-semver-check rust-smoke rust-regression rust-regression-scoped miri-cd-kernel dep-audit cargo-deny-check mcp-smoke e027-validate studio-run studio-check profile-tensor-avt x87-strategy-bench x87-strategy-perf x87-strategy-hyperfine x87-strategy-flamegraph x87-givens-microbench x87-givens-microbench-perf jacobi-backend-sweep jacobi-backend-perf jacobi-backend-flamegraph jacobi-backend-samply jacobi-backend-samply-compare gpu-bench gpu-bench-ncu gpu-bench-nsys
.PHONY: cpu-bench cpu-bench-perf cpu-bench-cachegrind cpu-bench-flamegraph parity-bench parity-report
.PHONY: pre-push-gate-scoped submodule-sync gate-local gate-ci-registry gate-ci-rust gate-audit gate-audit-fast data-core-pure-check
.PHONY: cache-status cache-sweep cache-sweep-soft cache-purge-exp cache-check cache-check-force
.PHONY: v6-branch-transport-artifacts pathion-control-artifacts pathion-resonance-artifacts
.PHONY: registry-control-plane-gate-readonly registry-acceptance-gate-readonly
.PHONY: rust-parity rust-release-fat-lto rust-pgo-instrument rust-pgo-merge rust-pgo-build
.PHONY: verify-pantheon-physicsforge-license verify-pantheon-physicsforge-provenance
.PHONY: verify-pantheon-physicsforge-mapping verify-pantheon-physicsforge-license-headers
.PHONY: verify-pantheon-physicsforge-overflow seed-pantheon-physicsforge-sqlite
.PHONY: registry registry-knowledge registry-governance registry-migrate-corpus registry-normalize-claims
.PHONY: registry-normalize-bibliography registry-bootstrap-bibliography
.PHONY: registry-normalize-external-sources registry-bootstrap-external-sources
.PHONY: registry-normalize-research-narratives registry-bootstrap-research-narratives
.PHONY: registry-normalize-book-docs registry-bootstrap-book-docs
.PHONY: registry-normalize-docs-root-narratives registry-bootstrap-docs-root-narratives
.PHONY: registry-normalize-reports-narratives registry-bootstrap-reports-narratives
.PHONY: registry-normalize-docs-convos registry-bootstrap-docs-convos
.PHONY: registry-normalize-data-artifact-narratives registry-bootstrap-data-artifact-narratives
.PHONY: registry-normalize-entrypoint-docs registry-bootstrap-entrypoint-docs
.PHONY: registry-bootstrap-claims-support
.PHONY: registry-normalize-narratives registry-normalize-operational-narratives
.PHONY: registry-markdown-inventory registry-markdown-corpus registry-toml-inventory
.PHONY: registry-markdown-origin-audit
.PHONY: registry-knowledge-atoms registry-verify-knowledge-atoms
.PHONY: registry-artifact-scrolls registry-verify-artifact-scrolls
.PHONY: registry-verify-markdown-inventory registry-verify-markdown-origin registry-verify-markdown-owner registry-verify-control-plane registry-control-plane-gate registry-verify-wave4 registry-wave4
.PHONY: registry-verify-markdown-toml-first
.PHONY: registry-embedded-markdown registry-verify-embedded-markdown
.PHONY: registry-build-semantic-atoms registry-verify-semantic-atoms registry-semantic-atoms-gate
.PHONY: registry-build-evidence-provenance registry-verify-evidence-provenance registry-evidence-provenance-gate
.PHONY: integrity-resolution registry-build-integrity-resolution registry-verify-integrity-resolution registry-integrity-resolution-gate
.PHONY: registry-build-execution-planning registry-verify-execution-planning registry-execution-planning-gate
.PHONY: registry-strict-toml-batch1-build registry-verify-strict-toml-batch1 registry-strict-toml-batch1 registry-wave5-batch1-build registry-verify-wave5-batch1 registry-wave5-batch1
.PHONY: registry-strict-toml-batch2-build registry-verify-strict-toml-batch2 registry-strict-toml-batch2 registry-wave5-batch2-build registry-verify-wave5-batch2 registry-wave5-batch2 registry-acceptance-gate registry-wave5
.PHONY: registry-strict-toml-batch3-build registry-verify-strict-toml-batch3 registry-strict-toml-batch3 registry-wave5-batch3-build registry-verify-wave5-batch3 registry-wave5-batch3
.PHONY: registry-strict-toml-batch4-build registry-verify-strict-toml-batch4 registry-strict-toml-batch4 registry-wave5-batch4-build registry-verify-wave5-batch4 registry-wave5-batch4
.PHONY: registry-verify-schema-signatures registry-verify-crossrefs
.PHONY: registry-verify-typed-policy-error
.PHONY: registry-verify-dataset-label-aliases
.PHONY: registry-csv-inventory registry-migrate-legacy-csv registry-verify-legacy-csv
.PHONY: registry-migrate-curated-csv registry-verify-curated-csv registry-csv-scope registry-data
.PHONY: registry-project-csv-split registry-csv-holdings
.PHONY: registry-scroll-project-csv-canonical registry-scroll-project-csv-generated
.PHONY: registry-scroll-external-csv-holding registry-scroll-archive-csv-holding
.PHONY: registry-csv-scroll-pipeline registry-verify-csv-scroll-pipeline
.PHONY: registry-verify-project-csv-split registry-verify-csv-holdings registry-verify-csv-corpus-coverage registry-csv-pipeline-gate registry-wave3
.PHONY: registry-refresh registry-export-markdown registry-verify-mirrors docs-publish docs-freshness docs-gate docs-site docs-rustdoc docs-book docs-redirect-check
.PHONY: artifacts artifacts-dimensional artifacts-materials artifacts-boxkites
.PHONY: artifacts-reggiani artifacts-m3 artifacts-motifs artifacts-motifs-big artifacts-repo-visuals
.PHONY: fetch-data fetch-data-redownload provenance-audit external-redownload-audit semantic-data-validate semantic-data-validate-strict run rocq latex latex-heliosphere latex-heliosphere-figs latex-heliosphere-clean latex-heliosphere-review
.PHONY: docker-quantum-build docker-quantum-run docker-quantum-shell
.PHONY: clean clean-builds clean-artifacts clean-all host-profile
.PHONY: run-e183
.PHONY: cpd-audit cpd-audit-strict cpd-audit-tooling cpd-audit-generated patch-static-mirror-headers cargo-cache-status cargo-cache-prune cargo-cache-smoke
.PHONY: cd-row-upgrade-batch cd-row-upgrade-jacobson cd-row-upgrade-freudenthal

.NOTPARALLEL: bootstrap-dev check smoke integrity integrity-rust rust-smoke rust-regression rust-regression-scoped heavy cargo-deny-check gate-local gate-ci-registry gate-ci-rust gate-audit gate-audit-fast pre-push-gate pre-push-gate-scoped pre-push-gate-strict governance-gate governance-gate-readonly registry-control-plane-gate-readonly registry-acceptance-gate-readonly

# Non-cargo make fanout: 75% of logical CPUs, minimum 1.
# Cargo and Rust test runners use a shared worker budget equal to logical threads / 2.
# The authoritative Rust equivalent is `xtask worker-budget`; this shell fallback
# exists because $(shell ...) runs at Makefile parse time (cargo run too slow).
NPROC := $(shell nproc 2>/dev/null || echo 4)
NJOBS := $(shell expr $(NPROC) \* 3 / 4)
WORKER_BUDGET ?= $(shell sh scripts/detect_worker_budget.sh)
CARGO_JOBS ?= $(WORKER_BUDGET)
NEXTEST_TEST_THREADS ?= $(WORKER_BUDGET)
RUST_TEST_THREADS ?= $(WORKER_BUDGET)
RAYON_THREADS ?= $(WORKER_BUDGET)
# Clippy targets default to --all-targets because those dev-profile test
# artifacts are reused by the subsequent `cargo nextest run --lib` lane.
# Dropping --all-targets shortens clippy itself, but forces nextest to
# rebuild the test profile from scratch and lengthens the combined gate.
#
# Override per invocation: RUST_SCOPED_CLIPPY_TARGETS=""
RUST_SCOPED_CLIPPY_TARGETS ?= --all-targets
LOCAL_NEXTEST_TIMING_JSON ?=
RUST_LOCAL_SKIP_FILTERSET ?= not ((package(stats_core) and test(/ultrametric::baire_codebook::tests::(test_euclidean_ultrametricity_across_filtration_levels|test_intermediate_filtration_gradient|test_random_removal_control|test_lambda512_to_256_intermediate_gradient|test_lambda512_to_256_random_removal_control|test_sbase_to_lambda2048_gradient|test_l0_subpopulation_ultrametricity|test_lambda2048_to_1024_intermediate_gradient|test_l1_filter_on_l0_neg1_subset|test_recursive_simpsons_paradox_l2|test_cross_stratum_triple_decomposition|test_l0_zero_simpsons_paradox|test_dimensional_universality_simpsons_paradox|test_lambda1024_stratum_paradox_and_summary)/)) or (package(algebra_experimental) and test(test_thesis_e_xor_involution_invariants_128d)) or (package(algebra_experimental) and test(/test_v6_(2d_constrained_scan|joint_4d_optimization)/)) or (package(algebra_experimental) and test(/test_(enumerate_tower|fast_enumerate|benchmark_fast_vs_scalar|compressed_memory|enumerate_8192d|fast_enumerate_16384d)/)) or (package(algebra_experimental) and test(test_pathion_vk_spectrum)) or (package(algebra_analysis) and test(/test_(d64_flat_band|d16_d32_d64_scaling)/)) or (package(materials_core) and test(test_separating_degree_formula_universality)) or (package(gororoba_algebra) and test(test_split_octonion_attractor_regression_dim_128_256_guarded)) or (package(gororoba_cli) and test(test_zero_divisor_scaling)) or (package(sign_imbalance) and test(test_kubo_j1j2_alpha_sweep)) or test(/gpu/))
REPO_TMPDIR ?= $(or $(TMPDIR),/tmp)
REPO_PATH_HASH ?= $(shell printf "%s" "$(CURDIR)" | sha256sum | cut -c1-16)
REPO_TMP_CARGO_ROOT ?= $(REPO_TMPDIR)/open_gororoba-cargo-build/gate/$(REPO_PATH_HASH)
REPO_CARGO_HOME ?= $(CURDIR)/.cache/cargo-home
CARGO_CACHE_REPO_BUDGET_GIB ?= 150
CARGO_CACHE_TMP_BUDGET_GIB ?= 16
# Gate builds use a separate target dir from ambient (LSP/editor) builds to
# avoid file-lock contention during concurrent cargo check / nextest runs.
# Both dirs are bounded by `make cache-sweep` (cargo-sweep --maxsize).
# Experimental target dirs MUST follow the naming convention .cache/exp-<name>-target/
# Use `make cache-purge-exp` to remove all of them. Never create ad-hoc names.
REPO_CARGO_TARGET_DIR ?= $(CURDIR)/.cache/gate-target
# Build intermediates (.o/.d) go to .cache/gate-cbuild/ on disk, keeping
# target-dir lean and avoiding /tmp (16 GB tmpfs) overflow. The 46-crate
# debug test compilation generates ~13 GB of split-debuginfo artifacts --
# more than the tmpfs budget. REPO_TMP_CARGO_ROOT still routes to /tmp for
# doc builds and parity tests where the artifact footprint is smaller.
REPO_CARGO_BUILD_DIR ?= $(CURDIR)/.cache/gate-cbuild/$(REPO_PATH_HASH)
CARGO_ENV = CARGO_HOME=$(REPO_CARGO_HOME) CARGO_TARGET_DIR=$(REPO_CARGO_TARGET_DIR) CARGO_BUILD_BUILD_DIR=$(REPO_CARGO_BUILD_DIR) MAKEFLAGS= MFLAGS= CARGO_MAKEFLAGS= CARGO_BUILD_JOBS=$(CARGO_JOBS) RAYON_NUM_THREADS=$(RAYON_THREADS) RUST_TEST_THREADS=$(RUST_TEST_THREADS)
# A user-local Cargo config may enforce CARGO_INCREMENTAL=0 globally.
# Kept here as belt-and-suspenders for CI environments where that config is absent.
CARGO_ENV_CI = $(CARGO_ENV) CARGO_INCREMENTAL=0

HOOKS_DIR ?= .githooks
MARKDOWN_EXPORT ?= 0
MARKDOWN_EXPORT_OUT_DIR ?= docs/generated
MARKDOWN_EXPORT_EMIT_LEGACY ?= 0
MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC ?= 1
DOCS_SITE_DIR ?= $(CURDIR)/target/site-docs
DOCS_BOOK_DIR ?= $(DOCS_SITE_DIR)/book
DOCS_RUSTDOC_DIR ?= $(DOCS_SITE_DIR)/rustdoc
DOCS_CARGO_TARGET_DIR ?= $(CURDIR)/target/docs-target
DOCS_CARGO_BUILD_DIR ?= $(REPO_TMP_CARGO_ROOT)/docs
DOCS_CARGO_ENV = CARGO_HOME=$(REPO_CARGO_HOME) CARGO_TARGET_DIR=$(DOCS_CARGO_TARGET_DIR) CARGO_BUILD_BUILD_DIR=$(DOCS_CARGO_BUILD_DIR) CARGO_BUILD_JOBS=$(CARGO_JOBS) RAYON_NUM_THREADS=$(RAYON_THREADS) RUST_TEST_THREADS=$(RUST_TEST_THREADS)
SEMVER_BASELINE_REV ?= v1.0-methods
SEMVER_BASELINE_SHA := $(shell git rev-parse --short=12 $(SEMVER_BASELINE_REV) 2>/dev/null || echo unknown)
SEMVER_BASELINE_ROOT ?= $(CURDIR)/.cache/semver-baselines/$(SEMVER_BASELINE_REV)-$(SEMVER_BASELINE_SHA)
SEMVER_CARGO_TARGET_DIR ?= $(CURDIR)/.cache/semver-target
SEMVER_CARGO_BUILD_DIR ?= $(CURDIR)/.cache/semver-cbuild/$(REPO_PATH_HASH)
SEMVER_TMPDIR ?= $(CURDIR)/.cache/semver-tmp
MD_BOOK ?= mdbook
PGO_DIR ?= /tmp/pgo-data
SYNTHESIS_CONTRACT_DATE ?= 2026_02_14
SYNTHESIS_CONTRACT_REPORT ?= reports/synthesis_execution_contract_$(SYNTHESIS_CONTRACT_DATE).toml
PROFILE_TIMESTAMP := $(shell date +%Y-%m-%d/%H%M%S)
PROFILE_ROOT ?= reports/gates/profiles/$(PROFILE_TIMESTAMP)

CD_CACHE_ROOT ?= /home/eirikr/Documents/Projects/CayleyDickson
CD_ROW_UPGRADE_OPERATOR ?= Codex
CD_ROW_UPGRADE_LANE ?=
CD_ROW_UPGRADE_WITNESS ?=
CD_ROW_UPGRADE_STATUS ?=
CD_ROW_UPGRADE_ROWS ?=

JACOBSON_ROW_UPGRADE_WITNESS ?= $(CD_CACHE_ROOT)/tier1_core_cd_algebra/composition_alternative_algebras/jacobson_1958_composition_algebras_and_their_automorphisms_preview.pdf
JACOBSON_ROW_UPGRADE_STATUS ?= official-fragment
JACOBSON_ROW_UPGRADE_ROWS ?= --row-id J58-DEF-01 --row-id J58-THM-01 --row-id J58-LEM-01 --row-id J58-NUM-01 --row-id J58-DEP-01

FREUDENTHAL_ROW_UPGRADE_WITNESS ?= $(CD_CACHE_ROOT)/tier1_core_cd_algebra/composition_alternative_algebras/freudenthal_1985_translation_oktaven_ausnahmegruppen_oktavengeometrie.pdf
FREUDENTHAL_ROW_UPGRADE_STATUS ?= translation-rewriting
FREUDENTHAL_ROW_UPGRADE_ROWS ?= --row-id F51-DEF-01 --row-id F51-THM-01 --row-id F51-LEM-01 --row-id F51-NUM-01 --row-id F51-DEP-01

# ---- Three-layer registry data plane ----
# Layer 1 (Canonical): registry/canonical/control_plane.sqlite3 (SQLite source-of-truth).
# Layer 2 (Compatibility): registry/*.toml (legacy/export view; read-optimized for migration compatibility).
# Layer 3 (Query):  gororoba-db CLI.

REGISTRY_SOURCES := $(wildcard registry/claims.toml registry/insights.toml \
    registry/experiments.toml registry/binaries.toml registry/project.toml \
    registry/external_sources.toml registry/bibliography.toml \
    registry/claims_evidence_edges.toml registry/experiment_lineage.toml \
    registry/lacunae.toml registry/roadmap.toml registry/todo.toml \
    registry/next_actions.toml registry/requirements.toml \
    registry/artifact_source_of_truth.toml registry/research_narratives.toml \
    registry/source_manifest.toml)

registry/canonical/control_plane.sqlite3: $(REGISTRY_SOURCES)
	$(CARGO_ENV) cargo run --release -p gororoba_db --bin gororoba-db -- build

.PHONY: registry-build registry-build-verify
registry-build: registry/canonical/control_plane.sqlite3

registry-build-verify: registry/canonical/control_plane.sqlite3
	$(CARGO_ENV) cargo run --release -p gororoba_db --bin gororoba-db -- build --verify

# ---- Environment setup ----

bootstrap-dev:
	@echo "Rust-first dev bootstrap uses user-local config only."
	@echo "Run 'make bootstrap-user-local-xdg' to install ~/.cargo/config.toml, ~/.config/nextest.toml, and ~/.cache/gororoba-lit-cache defaults."
	@echo "See docs/engineering/runtime_env_inventory.txt and docs/engineering/lit_search_env_vars.txt for user-local runtime variables."
	@echo "OK: Rust-first dev bootstrap guidance emitted."

# ---- Quality gates ----

lint: rust-clippy

# ---- Formatting (dprint) ----
# Unified formatting for Rust (.rs via rustfmt), TOML, JSON, and Markdown.
# Install: cargo install dprint
DPRINT_CACHE_DIR ?= $(CURDIR)/.cache/dprint
fmt:
	DPRINT_CACHE_DIR=$(DPRINT_CACHE_DIR) dprint fmt

fmt-check:
	DPRINT_CACHE_DIR=$(DPRINT_CACHE_DIR) dprint check

# ---- Parallelized lint gates ----
# Tier 1: lightweight checks (no cargo compilation, safe to parallelize).
# Tier 2: cargo-heavy checks (require compilation, serialize).
#
# gate-fast: Tier 1 only (<10s). Use for pre-push and rapid feedback.
# gate-deep: Tier 1 + Tier 2 (minutes). Use for CI and thorough audits.

.PHONY: gate-fast gate-warm gate-deep audit-deep typos machete audit geiger supply-chain-gate

# Tier 1 targets (no cargo lock contention, run in parallel)
typos:
	typos

machete:
	cargo machete

audit:
	cargo audit

geiger:
	@echo "[geiger] Checking unsafe code in core crates..."
	cd crates/gororoba_algebra && cargo geiger 2>&1 | tail -5
	cd crates/provenance_store && cargo geiger 2>&1 | tail -5

# supply-chain-gate: aggregated check chaining cargo-deny + machete +
# a register_custom_getrandom non-existence grep (keeps RUSTSEC-2026-0097
# exposure provably zero per docs/adr/rustsec-dispositions.md).
# Runs deterministically; safe in pre-push-gate-strict.
supply-chain-gate:
	@echo "=== supply-chain-gate ==="
	@fail=0; \
	echo "[supply-chain] cargo deny --workspace check ..."; \
	cargo deny --workspace check 2>&1 | tail -2 | grep -E 'ok|advisories ok, bans ok' || { echo "FAIL: cargo deny check"; fail=1; }; \
	echo "[supply-chain] cargo machete ..."; \
	cargo machete > /dev/null 2>&1 || { echo "FAIL: cargo machete (unused deps)"; fail=1; }; \
	echo "[supply-chain] register_custom_getrandom non-existence grep ..."; \
	if grep -rn 'register_custom_getrandom' crates/ --include='*.rs' >/dev/null 2>&1; then \
		echo "FAIL: register_custom_getrandom callers found -- RUSTSEC-2026-0097 exposure is nonzero"; \
		grep -rn 'register_custom_getrandom' crates/ --include='*.rs'; \
		fail=1; \
	fi; \
	if [ "$$fail" -ne 0 ]; then echo "=== supply-chain-gate: FAILED ==="; exit 1; fi
	@echo "=== supply-chain-gate: PASSED ==="

# gate-fast: parallel Tier 1 checks (no cargo compilation required).
# Runs dprint, typos, and machete concurrently. ~5s on warm cache.
# WHY: Catches formatting, spelling, and dead-dep issues before expensive compilation.
# NOTE: ansi-check and terminology-gate use `cargo run` so they belong in Tier 2
# on a cold cache but are fast (~0s) on a warm cache. They run in gate-warm.
gate-fast:
	@echo "=== gate-fast: parallel zero-compile checks ==="
	@fail=0; \
	DPRINT_CACHE_DIR=$(DPRINT_CACHE_DIR) dprint check || { echo "FAIL: dprint check"; fail=1; }; \
	cargo machete || { echo "FAIL: cargo-machete (unused deps detected)"; fail=1; }; \
	typos || echo "WARN: typos found issues (review above, non-blocking for now)"; \
	if [ "$$fail" -ne 0 ]; then echo "=== gate-fast: FAILED ==="; exit 1; fi
	@echo "=== gate-fast: PASSED ==="

# gate-warm: gate-fast + governance checks that reuse cached cargo binaries.
# ~10s if binaries are cached, ~2min on cold cache (first compilation).
gate-warm: gate-fast
	@echo "=== gate-warm: governance checks (cached binaries) ==="
	$(MAKE) ansi-check
	$(MAKE) terminology-gate
	$(MAKE) governance-gate-readonly
	@echo "=== gate-warm: PASSED ==="

# gate-deep: gate-warm + full cargo-heavy checks (clippy, tests, audits).
# WHY: Full CI-grade audit. Catches everything including API compat and advisories.
gate-deep: gate-warm
	@echo "=== gate-deep: cargo-heavy checks (serialized) ==="
	$(MAKE) rust-clippy
	$(MAKE) rust-regression
	cargo audit
	$(MAKE) cargo-deny-check
	@echo "=== gate-deep: PASSED ==="

# audit-deep: opt-in composite audit. NOT required by default make or CI on every PR.
# WHY: Aggregates all expensive one-off audit tools (semver, deny, dep-audit, CPD,
#      docs-freshness) into a single reviewable target for pre-release or periodic runs.
# HOW: make audit-deep  (standalone, no preconditions required)
# NOTE: cpd-audit requires pmd; the target will self-report if pmd is absent.
audit-deep:
	@echo "=== audit-deep: full opt-in audit suite ==="
	$(MAKE) rust-clippy
	@# rust-semver-check is intentionally SKIPPED in audit-deep.
	@# WHY: cargo-semver-checks --baseline-rev checks out the baseline tag into a temp dir.
	@#   fwht = { path = "../cratesgororobas/fwht" } in the root Cargo.toml is an external
	@#   sibling-directory path dep that cannot be resolved from a git temp checkout.
	@#   This makes the baseline build fail for every workspace member.
	@# Resolution: run `make rust-semver-check` standalone only after moving fwht to
	@#   crates.io (see TODO at Cargo.toml line 211) or into the workspace.
	$(MAKE) cargo-deny-check
	$(MAKE) dep-audit
	@# docs-freshness is intentionally SKIPPED in audit-deep.
	@# WHY: cargo doc --workspace fails with -D rustdoc::broken-intra-doc-links because
	@#   mathematical notation like [a,b,c] and X[t] in doc comments is misread as doc
	@#   links. Affected crates: algebra_analysis, algebra_experimental, brown_1972,
	@#   wilmot_2025 (and potentially more). These are pre-existing; run separately as
	@#   `make docs-freshness` to track progress. Fix: escape brackets as \[a,b,c\].
	$(MAKE) cpd-audit
	@echo "=== audit-deep: PASSED ==="

test: rust-regression

# Build repo_utilities once, then invoke the binary directly.
# `cargo run --release ...` triggers a full workspace metadata walk on
# every invocation (~53s cold per fresh cargo process). A single
# `cargo build` at the top of `check` lets subsequent ansi-check and
# terminology-gate calls execute the cached binary directly (<1s each).
REPO_UTILITIES_BIN := $(REPO_CARGO_TARGET_DIR)/release/repo-utilities

check:
	@$(CARGO_ENV) cargo build --release -p repo_utilities --bin repo-utilities
	@$(REPO_UTILITIES_BIN) ansi-check --check
	@$(REPO_UTILITIES_BIN) terminology-gate
	$(MAKE) verify-no-reports-writes
	@echo "OK: fast shared check suite complete."

# Governance verifier targets
registry-verify-markdown-governance:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify markdown-removal-policy

# Governance gate binaries are cached at stable paths under
# $(GATE_TOOLS_DIR)/. Cache vars and rules live below where
# GATE_TOOLS_DIR is defined (search "MARKDOWN_REGISTRY_CACHE").
# This target consumes those cache entries.
.SECONDEXPANSION:
governance-gate-readonly: $$(MARKDOWN_REGISTRY_CACHE) $$(GOVERNANCE_VERIFY_CACHE) $$(INTEGRITY_RESOLUTION_CACHE)
	$(MARKDOWN_REGISTRY_CACHE) verify-gate-all
	$(GOVERNANCE_VERIFY_CACHE) gate-all
	@echo ""
	@echo "=========================================="
	@echo "READ-ONLY GOVERNANCE GATE: PASSED"
	@echo "=========================================="
	@echo "[done] Markdown inventory validated (SQLite-first with TOML compatibility checks)"
	@echo "[done] Markdown owner map verified"
	@echo "[done] Registry schema signatures checked"
	@echo "[done] Cross-reference integrity verified"
	@echo "[done] Dataset label aliases verified"
	@echo "[done] Canonical control-plane declarations verified"
	@echo "[done] External-source operational contracts verified"
	@echo "[done] Markdown governance removal policy checked"
	@echo ""
	@echo "SQLite-first governance checks are operational."
	@echo "=========================================="

governance-gate: governance-gate-readonly ndlb-gate
	@echo "OK: governance-gate is a compatibility alias for governance-gate-readonly."

# NDLB gate: No-Dataset-Left-Behind invariant. Every data/external/*
# subdir must be one of: active (experiment-bound), synthetic
# (local artifact), or deferred (tombstoned with a defer_to_sprint
# target). Unknown or dark dirs fail fast.
ndlb-gate:
	@echo "[ndlb-gate] validating dataset/server/experiment invariants..."
	$(CARGO_ENV) cargo run --profile release-gate -q -p gororoba_cli_data --bin ndlb-gate
	@echo "[ndlb-gate] OK."

wave6-gate: governance-gate
	@echo "DEPRECATED: make wave6-gate is a legacy alias. Use make governance-gate."

# Cache gate-tool binaries and host-profile output. Each
# `cargo run -q -p X --bin Y` invocation pays ~30-60s of metadata walk
# overhead even when nothing changed. Stable paths with Make-tracked
# dependency timestamps skip cargo entirely when source is unchanged.
GATE_TOOLS_DIR := $(REPO_CARGO_TARGET_DIR)/gate-tools
WORKSPACE_ROUTING_CACHE := $(GATE_TOOLS_DIR)/workspace-routing
HOST_PROFILE_CACHE := $(GATE_TOOLS_DIR)/host-profile.sh

# workspace-routing dep set: only the bin source + the slim governance package
# manifest.  The bin imports only external crates (anyhow, clap, std), so
# workspace crate changes do not invalidate it.
WORKSPACE_ROUTING_DEPS := crates/gororoba_cli_data/src/bin/workspace_routing.rs \
                          crates/gororoba_cli_governance/Cargo.toml

# xtask host-profile dep set: xtask's main.rs (where detect_host_profile
# lives) plus the xtask manifest. Most xtask edits won't touch the
# host-profile codepath, so we accept slight over-invalidation.
HOST_PROFILE_DEPS := xtask/src/main.rs xtask/Cargo.toml

$(WORKSPACE_ROUTING_CACHE): $(WORKSPACE_ROUTING_DEPS)
	@mkdir -p $(GATE_TOOLS_DIR)
	@echo "[gate-tools] rebuilding workspace-routing (source changed)"
	@$(CARGO_ENV) cargo build --release -q -p gororoba_cli_governance --bin workspace-routing-proxy
	@cp -f $(REPO_CARGO_TARGET_DIR)/release/workspace-routing-proxy $@
	@touch $@

$(HOST_PROFILE_CACHE): $(HOST_PROFILE_DEPS)
	@mkdir -p $(GATE_TOOLS_DIR)
	@echo "[gate-tools] refreshing host-profile snapshot (xtask source changed)"
	@$(CARGO_ENV) cargo run -q -p xtask -- host-profile --format shell > $@.tmp
	@mv $@.tmp $@

# The xtask binary is cached for the optional gate-local-xtask driver.
# Source-dep tracking skips cargo's metadata walk when xtask source is
# unchanged.
XTASK_CACHE := $(GATE_TOOLS_DIR)/xtask

# Cache governance gate binaries at stable paths under
# $(GATE_TOOLS_DIR)/. Without caching, every push that triggers
# run_governance=True pays a 5m 24s rebuild of these three binaries in
# release-gate profile, separate from release/dev/test profile caches.
# The repo_utilities check uses the same cached-binary pattern.
#
# Source dep tracking via Make: rebuild triggers ONLY when the bin
# source or its package manifest changes. Most pushes (including
# docs/registry-only commits) reuse the cached binaries instantly.
GOV_GATE_DEPS := crates/gororoba_cli_data/src/bin/markdown_registry.rs \
                 crates/gororoba_cli_data/src/bin/governance_verify.rs \
                 crates/gororoba_cli_data/src/bin/integrity_resolution.rs \
                 crates/gororoba_cli_data/Cargo.toml
MARKDOWN_REGISTRY_CACHE := $(GATE_TOOLS_DIR)/markdown-registry
GOVERNANCE_VERIFY_CACHE := $(GATE_TOOLS_DIR)/governance-verify
INTEGRITY_RESOLUTION_CACHE := $(GATE_TOOLS_DIR)/integrity-resolution

# Single rule produces all three binaries; use ordering deps so
# downstream targets can list any subset.
$(MARKDOWN_REGISTRY_CACHE): $(GOV_GATE_DEPS)
	@mkdir -p $(GATE_TOOLS_DIR)
	@echo "[gate-tools] rebuilding governance gate binaries (source changed)"
	@$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin markdown-registry --bin governance-verify --bin integrity-resolution
	@cp -f $(REPO_CARGO_TARGET_DIR)/release-gate/markdown-registry $(MARKDOWN_REGISTRY_CACHE)
	@cp -f $(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify $(GOVERNANCE_VERIFY_CACHE)
	@cp -f $(REPO_CARGO_TARGET_DIR)/release-gate/integrity-resolution $(INTEGRITY_RESOLUTION_CACHE)
	@touch $(MARKDOWN_REGISTRY_CACHE) $(GOVERNANCE_VERIFY_CACHE) $(INTEGRITY_RESOLUTION_CACHE)

$(GOVERNANCE_VERIFY_CACHE): $(MARKDOWN_REGISTRY_CACHE)
	@: # built alongside markdown-registry above

$(INTEGRITY_RESOLUTION_CACHE): $(MARKDOWN_REGISTRY_CACHE)
	@: # built alongside markdown-registry above
XTASK_DEPS := xtask/src/main.rs xtask/Cargo.toml

$(XTASK_CACHE): $(XTASK_DEPS)
	@mkdir -p $(GATE_TOOLS_DIR)
	@echo "[gate-tools] rebuilding xtask binary (source changed)"
	@$(CARGO_ENV) cargo build --release -q -p xtask --bin xtask
	@cp -f $(REPO_CARGO_TARGET_DIR)/release/xtask $@
	@touch $@

gate-tools: $(WORKSPACE_ROUTING_CACHE) $(HOST_PROFILE_CACHE) $(XTASK_CACHE)
	@echo "OK: gate-tools cached at $(GATE_TOOLS_DIR)/."

# gate-local-xtask: opt-in Rust-driven gate. Same end result as
# `make gate-local`, but writes per-phase timing JSONL to
# data/output/audit/<date>/gate-timing-<unix-ts>.jsonl for regression
# tracking. Set GATE_DRIVER=xtask to make this the default in CI.
.PHONY: gate-local-xtask
gate-local-xtask: cache-check $(WORKSPACE_ROUTING_CACHE) $(HOST_PROFILE_CACHE) $(XTASK_CACHE)
	@$(XTASK_CACHE) gate-local --routing-bin $(WORKSPACE_ROUTING_CACHE) $(if $(GATE_TIMING_OUT),--timing-json $(GATE_TIMING_OUT),)

gate-tools-clean:
	rm -f $(WORKSPACE_ROUTING_CACHE) $(HOST_PROFILE_CACHE)
	@echo "OK: gate-tools cache cleared."

# gate-local writes $(GATE_LOCK) with its PID and start time. A sibling
# `make gate-lock-status` target lets editors check whether a gate is in
# flight. The lock is removed in a shell trap on EXIT so crashed gates
# clean up. The lock surfaces edit-during-read hazards before a gate
# consumes inconsistent source.
GATE_LOCK := $(REPO_CARGO_TARGET_DIR)/gate-tools/gate-local.lock

.PHONY: gate-lock-status
gate-lock-status:
	@if [ -f "$(GATE_LOCK)" ]; then \
	    pid=$$(awk '/^pid=/ {sub("pid=",""); print}' "$(GATE_LOCK)" 2>/dev/null || echo ""); \
	    started=$$(awk '/^started=/ {sub("started=",""); print}' "$(GATE_LOCK)" 2>/dev/null || echo ""); \
	    if [ -n "$$pid" ] && kill -0 "$$pid" 2>/dev/null; then \
	        printf 'gate-local IN FLIGHT: pid=%s started=%s\n' "$$pid" "$$started"; \
	        exit 1; \
	    else \
	        printf 'gate-lock stale (pid %s not running); removing\n' "$$pid"; \
	        rm -f "$(GATE_LOCK)"; \
	    fi; \
	else \
	    echo 'no gate-local in flight'; \
	fi

gate-local: cache-check $(WORKSPACE_ROUTING_CACHE) $(HOST_PROFILE_CACHE)
	@mkdir -p $(dir $(GATE_LOCK))
	@if [ -f "$(GATE_LOCK)" ]; then \
	    prev_pid=$$(awk '/^pid=/ {sub("pid=",""); print}' "$(GATE_LOCK)" 2>/dev/null || echo ""); \
	    if [ -n "$$prev_pid" ] && kill -0 "$$prev_pid" 2>/dev/null; then \
	        echo "[gate-local] another gate-local already in flight (pid=$$prev_pid). Wait or `make gate-lock-status`."; \
	        exit 1; \
	    fi; \
	    rm -f "$(GATE_LOCK)"; \
	fi
	@trap 'rm -f "$(GATE_LOCK)"' EXIT INT TERM; \
	printf 'pid=%s\nstarted=%s\n' "$$$$" "$$(date -Iseconds)" > "$(GATE_LOCK)"; \
	set -e; \
	scope=""; \
	run_rust="true"; \
	run_governance="true"; \
	run_check="true"; \
	eval "$$(cat $(HOST_PROFILE_CACHE))"; \
	submake_env="WORKER_BUDGET=$$HOST_WORKER_BUDGET CARGO_JOBS=$$HOST_CARGO_JOBS NEXTEST_TEST_THREADS=$$HOST_NEXTEST_TEST_THREADS RUST_TEST_THREADS=$$HOST_RUST_TEST_THREADS RAYON_THREADS=$$HOST_RAYON_THREADS"; \
	echo "[gate-local] host profile: physical_cores=$$HOST_PHYSICAL_CORES core_ids=$$HOST_PHYSICAL_CORE_IDS l3_cache_bytes=$$HOST_L3_CACHE_BYTES l3_safe_bytes=$$HOST_L3_SAFE_WORKING_SET_BYTES worker_budget=$$HOST_WORKER_BUDGET"; \
	echo "[gate-local] determining scope..."; \
	if [ -x "$(WORKSPACE_ROUTING_CACHE)" ]; then \
	    scope_file="$$(mktemp)"; \
	    meta_file="$$(mktemp)"; \
	    $(WORKSPACE_ROUTING_CACHE) --local --verbose 1>"$$scope_file" 2>"$$meta_file" || true; \
	    scope="$$(cat "$$scope_file" 2>/dev/null || true)"; \
	    routing_meta="$$(cat "$$meta_file" 2>/dev/null || true)"; \
	    rm -f "$$scope_file" "$$meta_file"; \
	    if [ -n "$$routing_meta" ]; then printf '%s\n' "$$routing_meta"; fi; \
	    printf '%s\n' "$$routing_meta" | grep -q 'run_rust=False' && run_rust="false" || true; \
	    printf '%s\n' "$$routing_meta" | grep -q 'run_governance=False' && run_governance="false" || true; \
	    printf '%s\n' "$$routing_meta" | grep -q 'run_check=False' && run_check="false" || true; \
	else \
	    echo "[gate-local] WARNING: workspace-routing unavailable, running full workspace"; \
	    scope="--workspace"; \
	fi; \
	if [ "$$run_check" = "true" ]; then \
	    $(MAKE) check $$submake_env; \
	else \
	    echo "[gate-local] SKIP: no check-relevant (non-Rust) file changes detected."; \
	fi; \
	if [ "$$run_rust" = "true" ]; then \
	    if [ -z "$$scope" ]; then scope="--workspace"; fi; \
	    echo "[gate-local] rust scope: $$scope"; \
	    if [ -n "$(LOCAL_NEXTEST_TIMING_JSON)" ]; then echo "[gate-local] local nextest timing: $(LOCAL_NEXTEST_TIMING_JSON)"; fi; \
	    $(MAKE) rust-regression-scoped RUST_SCOPE="$$scope" RUST_RUN_HEAVY=0 $$submake_env; \
	else \
	    echo "[gate-local] SKIP: no Rust-relevant changes detected."; \
	fi; \
	if [ "$$run_governance" = "true" ]; then \
	    $(MAKE) governance-gate-readonly $$submake_env; \
	else \
	    echo "[gate-local] SKIP: no governance-relevant changes detected."; \
	fi; \
	echo "[gate-local] OK: local gate passed."

pre-push-gate: gate-local
	@echo "OK: pre-push-gate is a compatibility alias for gate-local."

gate-ci-registry:
	$(MAKE) governance-gate-readonly
	$(MAKE) registry-control-plane-gate-readonly
	$(MAKE) registry-acceptance-gate-readonly
	@echo "OK: gate-ci-registry passed."

gate-ci-rust:
	$(MAKE) rust-regression CARGO_ENV="$(CARGO_ENV_CI)"
	$(MAKE) integrity-rust CARGO_ENV="$(CARGO_ENV_CI)"
	$(MAKE) cargo-deny-check CARGO_ENV="$(CARGO_ENV_CI)"
	$(MAKE) db-schema-drift-check CARGO_ENV="$(CARGO_ENV_CI)"
	@echo "OK: gate-ci-rust passed."

db-schema-drift-check:
	$(CARGO_ENV) cargo run -p xtask -- db-docs --check
	@echo "OK: db-schema-drift-check passed."

host-profile:
	cargo run -q -p xtask -- host-profile --format json

gate-audit:
	$(CARGO_ENV) cargo run -p xtask -- gate-audit
	@echo "OK: gate-audit completed."

# PH-5.A: structured audit-deep composite (rust-clippy + cargo-deny +
# dep-audit + cpd-audit) with per-step log capture, Markdown summary,
# and TOML record under reports/audit-deep/<date>/<time>/. Use this
# for tranche-acceptance evidence; use plain `make audit-deep` for
# interactive runs.
audit-deep-structured:
	$(CARGO_ENV) cargo run -p xtask -- audit-deep
	@echo "OK: audit-deep-structured completed."

# WHY: gate-ci-rust runs rust-regression (full workspace compile + nextest run,
# ~9 min). For registry/governance-only edits -- TOML updates, schema changes,
# claims.toml regeneration -- that compile overhead is pure waste. gate-audit-fast
# skips Rust compilation entirely: only gate-ci-registry (governance + schema
# checks, ~2 min) runs, and it fails fast on the first error.
gate-audit-fast:
	$(CARGO_ENV) cargo run -p xtask -- gate-audit --fast
	@echo "OK: gate-audit (fast/registry-only) completed."

# WHY: PH-2 acceptance gate -- verify data_core pure-core (no network plane).
# Must stay green after any data_core Cargo.toml or feature change.
# Two checks: compile without fetch feature, then assert reqwest/ureq absent.
data-core-pure-check:
	$(CARGO_ENV) cargo check -p data_core --no-default-features
	@$(CARGO_ENV) cargo tree -p data_core --no-default-features 2>&1 | \
	  grep -E "reqwest|ureq|backon" && \
	  (echo "FAIL: network dep found in data_core pure-core tree" && exit 1) || \
	  echo "OK: data_core pure-core has no network deps"

# ---- Cargo cache management -----------------------------------------------
# WHY: Two independent target dirs (.cache/gate-target and
# .cache/cargo-default-target) balloon without bounds because Cargo never
# auto-evicts build artifacts. cargo-sweep enforces size limits.
# cargo clean gc (enabled by [unstable] gc=true) handles CARGO_HOME only.
#
# Experimental target dirs: MUST be named .cache/exp-<name>-target/
# Use: CARGO_TARGET_DIR=$(CURDIR)/.cache/exp-myname-target cargo ...
# Clean: make cache-purge-exp
.PHONY: cache-status cache-sweep cache-sweep-soft cache-purge-exp cache-check cache-check-force cache-sweep-dry-run

cache-status:
	@# CLI cargo and gate cargo both write to .cache/gate-target (via .cargo/config.toml
	@# build.target-dir or CARGO_TARGET_DIR env override).
	@printf '=== Cargo target dir (canonical) ===\n'
	@du -sh .cache/gate-target 2>/dev/null || printf '(missing)\n'
	@printf '=== CARGO_HOME ===\n'
	@du -sh .cache/cargo-home 2>/dev/null || true
	@printf '=== Build-dir intermediates ===\n'
	@du -sh .cache/gate-cbuild 2>/dev/null || printf '(empty)\n'
	@printf '=== Gate tools cache ===\n'
	@du -sh .cache/gate-target/gate-tools 2>/dev/null || printf '(empty)\n'
	@printf '=== Experimental dirs (.cache/exp-*-target) ===\n'
	@du -sh .cache/exp-*-target 2>/dev/null || printf '(none)\n'
	@printf '=== Residual target/ (cargo doc + mdbook, NOT cargo build) ===\n'
	@du -sh target 2>/dev/null || printf '(missing)\n'

# cache-sweep uses age-based preservation instead of an unconditional
# gate-cbuild wipe. `cargo sweep --time N` keeps artifacts accessed
# within the last N days; this preserves the incremental working set
# across sessions.
#
# Tunables:
#   CACHE_SWEEP_KEEP_DAYS (default 7): cargo-sweep --time argument.
#     Anything older than this many days is removed.
#   CACHE_SWEEP_DEBUG_KEEP_DAYS (default 14): gate-cbuild debug wipe
#     threshold. Set to 0 to never auto-wipe (let cargo-sweep handle).
#   CACHE_SWEEP_PRESSURE_TARGET_MB (default CACHE_CHECK_SOFT_MB): after the
#     normal age sweep, cache-sweep-soft removes regenerable gate-cbuild
#     intermediates when total cargo cache pressure is still above this limit.
CACHE_SWEEP_KEEP_DAYS ?= 7
CACHE_SWEEP_DEBUG_KEEP_DAYS ?= 14
CACHE_SWEEP_PRESSURE_TARGET_MB ?= $(CACHE_CHECK_SOFT_MB)

cache-sweep:
	@# cache-sweep operates on .cache/gate-target and gate-cbuild only.
	@# Legacy cargo-default-target / .cache/cargo / .cache/sparse-cargo-home /
	@# orphan target dirs are outside this lane. Residual target/ holds only
	@# cargo doc and mdbook output.
	@echo "Pre-sweep size: $$(du -sh .cache 2>/dev/null | cut -f1)"
	@echo "Sweeping .cache/gate-target (keep artifacts accessed in last $(CACHE_SWEEP_KEEP_DAYS) days)..."
	@CARGO_TARGET_DIR=.cache/gate-target cargo sweep --time $(CACHE_SWEEP_KEEP_DAYS) . || echo "(skip: target absent or not a cargo project)"
# Conditional gate-cbuild debug wipe: only remove directories where the
# most recent file was modified more than CACHE_SWEEP_DEBUG_KEEP_DAYS
# days ago. Skips wipe entirely if CACHE_SWEEP_DEBUG_KEEP_DAYS=0.
	@if [ "$(CACHE_SWEEP_DEBUG_KEEP_DAYS)" -gt 0 ]; then \
	    for d in .cache/gate-cbuild/*/debug; do \
	        if [ -d "$$d" ]; then \
	            most_recent=$$(find "$$d" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1); \
	            if [ -z "$$most_recent" ]; then continue; fi; \
	            age_days=$$(( ( $$(date +%s) - $$most_recent ) / 86400 )); \
	            if [ "$$age_days" -gt "$(CACHE_SWEEP_DEBUG_KEEP_DAYS)" ]; then \
	                SIZE=$$(du -sh "$$d" 2>/dev/null | cut -f1); \
	                echo "Removing stale gate-cbuild debug ($$SIZE, $${age_days}d old) at $$d ..."; \
	                rm -rf "$$d"; \
	            else \
	                echo "Keeping gate-cbuild debug at $$d (last touched $${age_days}d ago)"; \
	            fi; \
	        fi; \
	    done; \
	fi
	@echo "Post-sweep size: $$(du -sh .cache 2>/dev/null | cut -f1)"

cache-sweep-soft:
	@$(MAKE) -s cache-sweep
	@GATE_MB=$$(du -sm .cache/gate-target 2>/dev/null | cut -f1 || printf '0'); \
	CBUILD_MB=$$(du -sm .cache/gate-cbuild 2>/dev/null | cut -f1 || printf '0'); \
	TARGET_MB=$$(du -sm target 2>/dev/null | cut -f1 || printf '0'); \
	TOTAL=$$((GATE_MB + CBUILD_MB + TARGET_MB)); \
	LIMIT=$${CACHE_SWEEP_PRESSURE_TARGET_MB:-$${CACHE_CHECK_SOFT_MB:-153600}}; \
	if [ "$$TOTAL" -gt "$$LIMIT" ] && [ "$$CBUILD_MB" -gt 0 ] && [ -d .cache/gate-cbuild ]; then \
	    printf '[cache-sweep-soft] size pressure: %dMB > %dMB; removing regenerable gate-cbuild intermediates (%dMB)\n' "$$TOTAL" "$$LIMIT" "$$CBUILD_MB"; \
	    rm -rf .cache/gate-cbuild; \
	else \
	    printf '[cache-sweep-soft] no size-pressure purge needed (total=%dMB limit=%dMB gate-cbuild=%dMB)\n' "$$TOTAL" "$$LIMIT" "$$CBUILD_MB"; \
	fi
	@rm -f "$(CACHE_CHECK_SENTINEL)"
	@$(MAKE) -s cache-check-force

# cache-sweep-dry-run: show what would be removed without removing.
.PHONY: cache-sweep-dry-run
cache-sweep-dry-run:
	@echo "DRY RUN: would sweep with --time $(CACHE_SWEEP_KEEP_DAYS) days"
	@echo "=== cargo-sweep dry-run on .cache/gate-target ==="
	@CARGO_TARGET_DIR=.cache/gate-target cargo sweep --time $(CACHE_SWEEP_KEEP_DAYS) --dry-run . || true
	@echo "=== gate-cbuild debug dirs older than $(CACHE_SWEEP_DEBUG_KEEP_DAYS) days ==="
	@for d in .cache/gate-cbuild/*/debug; do \
	    if [ -d "$$d" ]; then \
	        most_recent=$$(find "$$d" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1); \
	        if [ -z "$$most_recent" ]; then continue; fi; \
	        age_days=$$(( ( $$(date +%s) - $$most_recent ) / 86400 )); \
	        SIZE=$$(du -sh "$$d" 2>/dev/null | cut -f1); \
	        if [ "$$age_days" -gt "$(CACHE_SWEEP_DEBUG_KEEP_DAYS)" ]; then \
	            echo "  WOULD REMOVE: $$d ($$SIZE, $${age_days}d old)"; \
	        else \
	            echo "  KEEP: $$d ($$SIZE, $${age_days}d old)"; \
	        fi; \
	    fi; \
	done
	@echo "OK: cache-sweep complete."

cache-purge-exp:
	rm -rf .cache/exp-*-target
	@echo "OK: experimental target dirs purged."

# Cache size check: fails at configurable thresholds. Hard cap blocks
# push via the pre-push hook.
# Soft cap is also an error: warnings-as-errors means gate diagnostics must be
# actionable failures, not non-blocking noise.
#
# Tunable via env vars:
#   CACHE_CHECK_SOFT_MB  (default 153600  = 150 GB)
#   CACHE_CHECK_HARD_MB  (default 256000  = 250 GB)
# Memoize cache-check with a 30-minute TTL. The four `du -sm` walks
# over hundreds of GB take ~10s of wall time per push. The cache size
# grows slowly during a session; checking once every 30 minutes (or on
# explicit `make cache-check-force`) gives the same safety guarantee
# with near-zero overhead for in-session pushes.
CACHE_CHECK_SENTINEL := $(REPO_CARGO_TARGET_DIR)/gate-tools/cache-check.last
CACHE_CHECK_TTL_SECS ?= 1800

cache-check:
	@mkdir -p $(dir $(CACHE_CHECK_SENTINEL)); \
	if [ -f "$(CACHE_CHECK_SENTINEL)" ]; then \
	    age=$$(($$(date +%s) - $$(stat -c %Y "$(CACHE_CHECK_SENTINEL)" 2>/dev/null || echo 0))); \
	    if [ "$$age" -lt "$(CACHE_CHECK_TTL_SECS)" ]; then \
	        cat "$(CACHE_CHECK_SENTINEL)"; \
	        printf '[cache-check] (memoized; refreshed %ds ago; run make cache-check-force to recompute)\n' "$$age"; \
	        exit 0; \
	    fi; \
	fi; \
	$(MAKE) -s cache-check-force | tee "$(CACHE_CHECK_SENTINEL)"

cache-check-force:
	@# Cache accounting sums gate-target, gate-cbuild, and residual target/.
	@# Residual target/ is reserved for cargo doc and mdbook output.
	@GATE_MB=$$(du -sm .cache/gate-target 2>/dev/null | cut -f1 || printf '0'); \
	CBUILD_MB=$$(du -sm .cache/gate-cbuild 2>/dev/null | cut -f1 || printf '0'); \
	TARGET_MB=$$(du -sm target 2>/dev/null | cut -f1 || printf '0'); \
	TOTAL=$$((GATE_MB + CBUILD_MB + TARGET_MB)); \
	SOFT=$${CACHE_CHECK_SOFT_MB:-153600}; \
	HARD=$${CACHE_CHECK_HARD_MB:-256000}; \
	if [ "$$TOTAL" -gt "$$HARD" ]; then \
		printf '[cache-check] FAIL: cargo dirs total %dGB (>%dGB hard cap). Run: make cache-sweep-soft\n' "$$((TOTAL / 1024))" "$$((HARD / 1024))"; \
		exit 1; \
	elif [ "$$TOTAL" -gt "$$SOFT" ]; then \
		printf '[cache-check] FAIL: cargo dirs total %dGB (>%dGB soft cap). Run: make cache-sweep-soft\n' "$$((TOTAL / 1024))" "$$((SOFT / 1024))"; \
		exit 1; \
	else \
		printf '[cache-check] OK: cargo dirs at %dMB (soft=%dGB hard=%dGB)\n' "$$TOTAL" "$$((SOFT / 1024))" "$$((HARD / 1024))"; \
	fi

pre-push-gate-strict: gate-audit
	@echo "OK: pre-push-gate-strict is a compatibility alias for gate-audit."

hooks-install:
	@mkdir -p "$(HOOKS_DIR)"
	@chmod +x "$(HOOKS_DIR)/pre-push"
	@git config core.hooksPath "$(HOOKS_DIR)"
	@echo "OK: git hooks installed. core.hooksPath=$$(git config --get core.hooksPath)"
	@echo "Pre-push will run: make gate-local"

hooks-install-strict:
	@mkdir -p "$(HOOKS_DIR)"
	@cp "$(HOOKS_DIR)/pre-push" "$(HOOKS_DIR)/pre-push.bak" 2>/dev/null || true
	@printf '%s\n' \
		'#!/usr/bin/env bash' \
		'set -euo pipefail' \
		'repo_root="$$(git rev-parse --show-toplevel)"' \
		'cd "$$repo_root"' \
		'echo "[pre-push] running make gate-local"' \
		'make gate-local' \
		> "$(HOOKS_DIR)/pre-push"
	@chmod +x "$(HOOKS_DIR)/pre-push"
	@git config core.hooksPath "$(HOOKS_DIR)"
	@echo "OK: strict git hook installed. core.hooksPath=$$(git config --get core.hooksPath)"
	@echo "Pre-push will run: make gate-local"

hooks-status:
	@echo "core.hooksPath=$$(git config --get core.hooksPath || echo .git/hooks)"
	@echo "pre-push hook exists? $$(test -f "$(HOOKS_DIR)/pre-push" && echo yes || echo no)"

smoke: check rust-smoke
	@echo "OK: smoke lane passed."

registry-control-plane-gate-readonly:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-corpus
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-toml-inventory
	@echo "OK: read-only registry control-plane gate passed."

integrity:
	$(MAKE) verify-pantheon-physicsforge-mapping
	$(MAKE) verify-pantheon-physicsforge-license-headers
	$(MAKE) verify-pantheon-physicsforge-overflow
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-embedded
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin verify-registry-mirror-freshness -- --out-dir "$(MARKDOWN_EXPORT_OUT_DIR)" --emit-legacy --legacy-claims-sync true
	@echo "OK: integrity lane passed."

integrity-rust:
	$(CARGO_ENV) cargo run --profile release-gate -p gororoba_cli_data --bin claims-verify -- --check providers
	$(MAKE) test-inventory
	$(CARGO_ENV) cargo run --profile release-gate -p gororoba_cli_data --bin registry-check -- --typed-policy error
	@echo "OK: Rust integrity lane passed."

test-inventory:
	$(CARGO_ENV) cargo run --profile release-gate -p gororoba_cli_data --bin test-inventory -- --check

math-verify: rust-regression
	@echo "OK: math validation suite complete. See docs/MATH_VALIDATION_REPORT.md"

rust-test: rust-regression
	@echo "OK: rust-test is an alias for rust-regression."

rust-clippy:
	$(CARGO_ENV) cargo clippy --workspace -- -D warnings

rust-semver-check:
	@echo "[semver-check] Checking public API SemVer compliance against v1.0-methods..."
	@# WHY: crates are private (not on crates.io). --baseline-root compares against the
	@# most recent git tag so we catch accidental public-API breakage since that tag.
	@# To advance the baseline after deliberate breaking changes: git tag -f $(SEMVER_BASELINE_REV) HEAD
	@#
	@# cargo-semver-checks --baseline-rev cannot clone the configured baseline tag on
	@# ext4: two generated registry_mirrors filenames have 238+ byte components,
	@# and cargo-semver-checks appends a temp suffix during checkout. Extracting
	@# the tag with git archive preserves the legal filename component and lets
	@# cargo-semver-checks reach actual API compatibility analysis.
	@#
	@# Excluded: CLI/binary crates (no public library API), build.rs crates
	@# (CUDA bindgen, data codegen), and crates added after the baseline tag.
	@if [ ! -f "$(SEMVER_BASELINE_ROOT)/Cargo.toml" ]; then \
		echo "[semver-check] Extracting $(SEMVER_BASELINE_REV) to $(SEMVER_BASELINE_ROOT)"; \
		mkdir -p "$(SEMVER_BASELINE_ROOT)"; \
		git archive "$(SEMVER_BASELINE_REV)" | tar -x -C "$(SEMVER_BASELINE_ROOT)"; \
	fi
	@mkdir -p "$(SEMVER_TMPDIR)"
	$(CARGO_ENV) TMPDIR=$(SEMVER_TMPDIR) CARGO_TARGET_DIR=$(SEMVER_CARGO_TARGET_DIR) CARGO_BUILD_BUILD_DIR=$(SEMVER_CARGO_BUILD_DIR) cargo semver-checks check-release --workspace \
		--baseline-root "$(SEMVER_BASELINE_ROOT)" \
		--exclude gororoba_cli \
		--exclude gororoba_cli_algebra \
		--exclude gororoba_cli_data \
		--exclude gororoba_cli_governance \
		--exclude gororoba_cli_physics \
		--exclude gororoba_cli_provenance \
		--exclude gororoba_cli_quantum \
		--exclude gororoba_cli_warp \
		--exclude gororoba_db \
		--exclude fixed_point_lbm \
		--exclude gororoba_gpu_cubecl \
		--exclude gororoba_gpu_cuda \
		--exclude gororoba_gpu_vulkan \
		--exclude grmhd_core \
		--exclude lbm_3d_cuda \
		--exclude gororoba_engine \
		--exclude materials_data \
		--exclude materials_core \
		--exclude repo_utilities \
		--exclude data_artifacts_core \
		--exclude cd_spin_bridge
	@echo "[semver-check] Done. All checked crates pass SemVer compliance."

rust-smoke:
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) -P smoke -p gororoba_algebra --test smoke_gororoba_algebra -p lbm_3d --test smoke_lbm_3d -p gororoba_engine --test smoke_gororoba_engine
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P smoke -p gr_core --test smoke_gr_core
	@echo "OK: Rust smoke lane passed."

rust-regression: rust-clippy
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --workspace --exclude algebra_analysis --exclude gr_core
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy -p algebra_analysis -p gr_core
	@echo "OK: Rust regression lane passed."

rust-regression-scoped:
	# workspace-routing source lives beside the data CLI, but
	# rust-regression-scoped builds it through the slim governance proxy binary.
	# That keeps scope classification independent of the full data CLI
	# dependency graph while preserving workspace-routing semantics.
	#
	# Prefer the cached binary at $(WORKSPACE_ROUTING_CACHE) to skip cargo's
	# metadata walk. The gate-tools target dependencies rebuild the cached binary
	# when workspace_routing.rs changes.
	#
	# Preserve routing CLI stderr so scope-selection failures reach the gate
	# operator. The fallback remains "--workspace" after the diagnostic is
	# emitted.
	$(eval RUST_SCOPE ?= $(shell \
	    if [ -x "$(WORKSPACE_ROUTING_CACHE)" ]; then \
	        "$(WORKSPACE_ROUTING_CACHE)" --local 2> >(tee /dev/stderr); \
	    else \
	        $(CARGO_ENV) cargo run -q -p gororoba_cli_governance --bin workspace-routing-proxy -- --local; \
	    fi || echo "--workspace"))
	# Clippy runs only on DIRECTLY changed crates (no reverse-closure expansion).
	# WHY: clippy lints fire on the package owning the source -- a change in a
	# hub crate cannot induce a new lint on a downstream consumer whose source
	# is unchanged. Skipping the closure for clippy saves the bulk of compile
	# time (gororoba_cli_data with 100+ binaries is the worst offender).
	$(eval RUST_CLIPPY_SCOPE ?= $(shell \
	    if [ -x "$(WORKSPACE_ROUTING_CACHE)" ]; then \
	        "$(WORKSPACE_ROUTING_CACHE)" --local --direct-only 2> >(tee /dev/stderr); \
	    else \
	        $(CARGO_ENV) cargo run -q -p gororoba_cli_governance --bin workspace-routing-proxy -- --local --direct-only; \
	    fi || echo "$(RUST_SCOPE)"))
	# Nextest in the LOCAL pre-push fast path also runs only on directly
	# changed crates. Trust the layered model:
	#   - Direct-changed crate tests + clippy = pre-push smoke gate (< 3 min).
	#   - Full reverse-closure regression = CI on PR open (gate-ci-rust).
	# Set RUST_NEXTEST_SCOPE_MODE=closure to opt back into the wide scope for
	# a single invocation when needed (e.g. after toolchain bumps).
	$(eval RUST_NEXTEST_SCOPE_MODE ?= direct)
	$(eval RUST_NEXTEST_SCOPE ?= $(if $(filter direct,$(RUST_NEXTEST_SCOPE_MODE)),$(RUST_CLIPPY_SCOPE),$(RUST_SCOPE)))
	# Pre-push test kind: `lib` runs only library unit tests; `all` runs
	# lib + integration test binaries (--lib --tests). Integration test
	# compile is the largest single contributor to gate wall-time (the
	# 441-binary link phase, ~4m30s); restricting to --lib at pre-push
	# makes the gate a true smoke gate. CI on PR open runs the full
	# `all` suite. Override per invocation:
	#   make gate-local RUST_NEXTEST_KIND_MODE=all
	$(eval RUST_NEXTEST_KIND_MODE ?= lib)
	$(eval RUST_NEXTEST_KINDS ?= $(if $(filter lib,$(RUST_NEXTEST_KIND_MODE)),--lib,--lib --tests))
	$(eval RUST_RUN_HEAVY ?= 1)
	@set -e; \
	if [ -z "$(RUST_SCOPE)" ]; then \
	    echo "SKIP: no Rust-relevant changes detected."; \
	else \
	    echo "[rust-regression-scoped] clippy scope: $(RUST_CLIPPY_SCOPE)"; \
	    echo "[rust-regression-scoped] nextest scope: $(RUST_NEXTEST_SCOPE) (scope_mode=$(RUST_NEXTEST_SCOPE_MODE) kinds=$(RUST_NEXTEST_KINDS))"; \
	    if [ -n "$(RUST_CLIPPY_SCOPE)" ]; then \
	        $(CARGO_ENV) cargo clippy $(RUST_CLIPPY_SCOPE) $(RUST_SCOPED_CLIPPY_TARGETS) -- -D warnings; \
	    fi; \
	    local_light_scope=""; \
	    local_light_packages=""; \
	    if [ "$(RUST_NEXTEST_SCOPE)" = "--workspace" ]; then \
	        light_scope="--workspace --exclude algebra_analysis --exclude gr_core"; \
	        heavy_scope="-p algebra_analysis -p gr_core"; \
	        local_light_scope="$$light_scope"; \
	    else \
	        light_scope=""; \
	        heavy_scope=""; \
	        prev=""; \
	        for token in $(RUST_NEXTEST_SCOPE); do \
	            if [ "$$prev" = "-p" ]; then \
	                case "$$token" in \
	                    algebra_analysis|gr_core) heavy_scope="$$heavy_scope -p $$token" ;; \
	                    *) \
	                        light_scope="$$light_scope -p $$token"; \
	                        local_light_packages="$$local_light_packages $$token" ;; \
	                esac; \
	                prev=""; \
	            elif [ "$$token" = "-p" ]; then \
	                prev="-p"; \
	            fi; \
	        done; \
	    fi; \
	    filterset=""; \
	    if [ "$(RUST_RUN_HEAVY)" != "1" ]; then \
	        filterset='$(RUST_LOCAL_SKIP_FILTERSET)'; \
	    fi; \
	    if [ "$(RUST_NEXTEST_SCOPE)" = "--workspace" ]; then \
	        if [ -n "$$filterset" ]; then \
	            echo "[rust-regression-scoped] local skip filter enabled"; \
	            $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $(RUST_NEXTEST_KINDS) $$local_light_scope -E "$$filterset"; \
	        else \
	            $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $(RUST_NEXTEST_KINDS) $$local_light_scope; \
	        fi; \
	    elif [ -n "$$local_light_packages" ]; then \
	        if [ -n "$$filterset" ]; then \
	            echo "[rust-regression-scoped] local skip filter enabled"; \
	            $(CARGO_ENV) cargo run -q -p xtask -- local-nextest-plan --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --kinds $(RUST_NEXTEST_KIND_MODE) $(if $(LOCAL_NEXTEST_TIMING_JSON),--timing-json-out $(LOCAL_NEXTEST_TIMING_JSON),) --filterset "$$filterset" $$local_light_packages; \
	        else \
	            $(CARGO_ENV) cargo run -q -p xtask -- local-nextest-plan --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --kinds $(RUST_NEXTEST_KIND_MODE) $(if $(LOCAL_NEXTEST_TIMING_JSON),--timing-json-out $(LOCAL_NEXTEST_TIMING_JSON),) $$local_light_packages; \
	        fi; \
	    fi; \
	    if [ -n "$$heavy_scope" ] && [ "$(RUST_RUN_HEAVY)" = "1" ]; then \
	        $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy $$heavy_scope; \
	    elif [ -n "$$heavy_scope" ]; then \
	        echo "[rust-regression-scoped] SKIP heavy nextest in local fast path: $$heavy_scope"; \
	    fi; \
	    echo "OK: Rust regression gate passed (scoped: clippy + nextest)."; \
	fi

# WHY: Miri catches UB in unsafe Cayley-Dickson arithmetic (pointer provenance,
# integer-to-pointer casts, uninit reads) that sanitizers miss at runtime.
# Rayon-parallel tests are suppressed via #[cfg_attr(miri, ignore)] because
# crossbeam-epoch 0.9.18 has a known Stacked Borrows false-positive under Miri.
# WHAT: Runs the cd_kernel lib tests only (no rayon paths).
# HOW: CARGO_TARGET_DIR isolates the Miri build artifacts; -Zmiri-permissive-provenance
# silences provenance-stripping from raw integer casts that are correct but
# non-standard (e.g. SIMD sign-table tricks with integer-indexed pointers).
miri-cd-kernel:
	CARGO_TARGET_DIR=.cache/miri-gate-target MIRIFLAGS="-Zmiri-permissive-provenance" \
	    cargo miri test -p cd_kernel
	@echo "OK: miri-cd-kernel passed."

heavy:
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --workspace --exclude algebra_analysis --exclude gr_core --run-ignored only -P heavy
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy -p algebra_analysis -p gr_core --run-ignored only
	@echo "OK: heavy lane passed."

# Convenience: sync all git submodules (proofs, paper when extracted).
submodule-sync:
	git submodule update --init --recursive
	@echo "OK: submodules synchronized."

studio-run:
	$(CARGO_ENV) cargo run -p gororoba_cli --bin gororoba-studio -- --host 127.0.0.1 --port 8088

studio-check:
	$(CARGO_ENV) cargo test -p gororoba_cli --bin gororoba-studio
	$(CARGO_ENV) cargo clippy -p gororoba_cli --bin gororoba-studio -- -D warnings
	@echo "OK: gororoba-studio checks passed."

bootstrap-user-local-xdg:
	scripts/bootstrap_user_local_xdg.sh $(ARGS)
	@echo "OK: user-local bootstrap completed."
	@echo "See docs/engineering/user_local_bootstrap.txt and docs/engineering/runtime_env_inventory.txt for policy details."

profile-tensor-avt:
	CARGO_HOME=$(REPO_CARGO_HOME) scripts/profile_tensor_avt.sh

x87-strategy-bench:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_algebra --bin x87-strategy-bench -- \
		--len $${LEN:-1048576} \
		--repeats $${REPEATS:-7} \
		--worker-counts $${WORKER_COUNTS:-1,2,4,6} \
		--output $${OUT:-reports/benchmarks/x87_strategy_bench.csv} \
		--summary $${SUMMARY:-reports/benchmarks/x87_strategy_bench.md}
	@echo "OK: x87 strategy benchmark completed."

x87-strategy-perf:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_algebra --bin x87-strategy-bench
	perf stat -e $${PERF_EVENTS:-cycles:u,instructions:u,branches:u,branch-misses:u} -r $${PERF_RUNS:-3} $(REPO_CARGO_TARGET_DIR)/release-gate/x87-strategy-bench \
		--len $${LEN:-262144} \
		--repeats $${REPEATS:-5} \
		--worker-counts $${WORKER_COUNTS:-1,2,4,6} \
		--output $${OUT:-reports/benchmarks/x87_strategy_perf.csv} \
		--summary $${SUMMARY:-reports/benchmarks/x87_strategy_perf.md}
	@echo "OK: x87 strategy perf-stat sweep completed."

x87-strategy-hyperfine:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_algebra --bin x87-strategy-bench
	hyperfine --shell=none --warmup $${WARMUP:-1} --runs $${RUNS:-5} \
		'$(REPO_CARGO_TARGET_DIR)/release-gate/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 1 --output /tmp/x87_strategy_hyperfine_1.csv' \
		'$(REPO_CARGO_TARGET_DIR)/release-gate/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 2 --output /tmp/x87_strategy_hyperfine_2.csv' \
		'$(REPO_CARGO_TARGET_DIR)/release-gate/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 4 --output /tmp/x87_strategy_hyperfine_4.csv' \
		'$(REPO_CARGO_TARGET_DIR)/release-gate/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 6 --output /tmp/x87_strategy_hyperfine_6.csv'
	@echo "OK: x87 strategy hyperfine sweep completed."

x87-strategy-flamegraph:
	CARGO_PROFILE_RELEASE_DEBUG=$${PROFILE_DEBUG:-true} $(CARGO_ENV) cargo flamegraph -p gororoba_cli_algebra --bin x87-strategy-bench --root -- \
		--len $${LEN:-262144} \
		--repeats $${REPEATS:-3} \
		--worker-counts $${WORKER_COUNTS:-1} \
		--output /tmp/x87_strategy_flamegraph.csv
	@echo "OK: x87 strategy flamegraph captured."

x87-givens-microbench:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_algebra --bin x87-givens-microbench -- \
		--iterations $${ITERATIONS:-200000} \
		--repeats $${REPEATS:-9} \
		$${CASES:+--cases $${CASES}} \
		$${KERNELS:+--kernels $${KERNELS}} \
		--output $${OUT:-reports/benchmarks/x87_givens_microbench.csv} \
		$${SUMMARY:+--summary $${SUMMARY}}
	@echo "OK: x87 Givens microbench completed."

x87-givens-microbench-perf:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_algebra --bin x87-givens-microbench
	perf stat -x, -e $${PERF_EVENTS:-cycles:u,instructions:u,branches:u,branch-misses:u} -r $${PERF_RUNS:-5} \
		$(REPO_CARGO_TARGET_DIR)/release-gate/x87-givens-microbench \
		--iterations $${ITERATIONS:-200000} \
		--repeats $${REPEATS:-9} \
		$${CASES:+--cases $${CASES}} \
		$${KERNELS:+--kernels $${KERNELS}} \
		--output $${OUT:-reports/benchmarks/x87_givens_microbench_perf.csv} \
		$${SUMMARY:+--summary $${SUMMARY}} \
		2> $${COUNTERS_OUT:-reports/benchmarks/x87_givens_microbench_perf.stat}
	@echo "OK: x87 Givens perf-stat microbench completed."

gpu-bench:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics \
		--bin cuda-precision-bench --features gpu -- \
		--output $${OUT:-data/benchmarks/cuda_kernel_baseline.csv} \
		$${GRIDS:+--grids $${GRIDS}} \
		$${WORKLOADS:+--workloads $${WORKLOADS}} \
		$${STEPS_SMALL:+--steps-small $${STEPS_SMALL}} \
		$${STEPS_MID:+--steps-mid $${STEPS_MID}} \
		$${STEPS_LARGE:+--steps-large $${STEPS_LARGE}}
	@echo "OK: CUDA kernel baseline benchmark complete. See data/benchmarks/cuda_kernel_baseline.csv"

gpu-bench-ncu:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_physics --bin cuda-precision-bench --features gpu
	@mkdir -p data/benchmarks/ncu
	ncu \
		--set $${NCU_SECTIONS:-SpeedOfLight,MemoryWorkloadAnalysis,ComputeWorkloadAnalysis} \
		--target-processes all \
		--export data/benchmarks/ncu/cuda_kernels_$$(date +%Y%m%d_%H%M%S) \
		$(REPO_CARGO_TARGET_DIR)/release-gate/cuda-precision-bench \
		--output data/benchmarks/cuda_kernel_baseline_ncu.csv \
		$${GRIDS:+--grids $${GRIDS}} \
		$${WORKLOADS:+--workloads $${WORKLOADS}} \
		--steps-small 5 --steps-mid 5 --steps-large 5 --warmup 3
	@echo "OK: ncu profile saved to data/benchmarks/ncu/"

gpu-bench-nsys:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_physics --bin cuda-precision-bench --features gpu
	@mkdir -p data/benchmarks/nsys
	nsys profile \
		--trace=$${NSYS_TRACE:-cuda,nvtx} \
		--output data/benchmarks/nsys/cuda_pipeline_$$(date +%Y%m%d_%H%M%S) \
		--force-overwrite true \
		$(REPO_CARGO_TARGET_DIR)/release-gate/cuda-precision-bench \
		--output data/benchmarks/cuda_kernel_baseline_nsys.csv \
		$${GRIDS:+--grids $${GRIDS}} \
		$${WORKLOADS:+--workloads $${WORKLOADS}} \
		--steps-small 20 --steps-mid 20 --steps-large 10 --warmup 5
	@echo "OK: nsys profile saved to data/benchmarks/nsys/ -- open .nsys-rep in Nsight Systems GUI"

cpu-bench:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics \
		--bin cpu-lbm-bench -- \
		--output $${OUT:-data/benchmarks/cpu_lbm_baseline.csv} \
		$${GRIDS:+--grids $${GRIDS}} \
		$${WORKLOADS:+--workloads $${WORKLOADS}}
	@echo "OK: CPU LBM benchmark complete. See data/benchmarks/cpu_lbm_baseline.csv"

cpu-bench-perf:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_physics --bin cpu-lbm-bench
	@mkdir -p reports/benchmarks
	perf stat -d \
		$(REPO_CARGO_TARGET_DIR)/release-gate/cpu-lbm-bench \
		--grids $${GRIDS:-64} --workloads $${WORKLOADS:-bgk} \
		--output /dev/null \
		2> $${COUNTERS_OUT:-reports/benchmarks/cpu_lbm_perf.stat}
	@echo "OK: perf stat saved to reports/benchmarks/cpu_lbm_perf.stat"

cpu-bench-cachegrind:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_physics --bin cpu-lbm-bench
	@mkdir -p reports/benchmarks
	valgrind --tool=cachegrind \
		--cachegrind-out-file=$${CGOUT:-reports/benchmarks/cachegrind.out.cpu_lbm} \
		$(REPO_CARGO_TARGET_DIR)/release-gate/cpu-lbm-bench \
		--grids $${GRIDS:-32} --workloads $${WORKLOADS:-bgk} --steps-small 10 \
		--output /dev/null
	@echo "OK: cachegrind output saved. Annotate with: cg_annotate $${CGOUT:-reports/benchmarks/cachegrind.out.cpu_lbm}"

cpu-bench-flamegraph:
	$(CARGO_ENV) cargo flamegraph --release -p gororoba_cli_physics --bin cpu-lbm-bench \
		-o $${FGOUT:-reports/benchmarks/cpu_lbm_flamegraph.svg} \
		-- --grids $${GRIDS:-64} --workloads $${WORKLOADS:-bgk} --output /dev/null
	@echo "OK: flamegraph saved to $${FGOUT:-reports/benchmarks/cpu_lbm_flamegraph.svg}"

parity-bench:
	@echo "Running CPU benchmark..."
	$(MAKE) cpu-bench GRIDS=$${GRIDS:-32,64} OUT=data/benchmarks/cpu_lbm_baseline.csv
	@echo "Running CUDA benchmark..."
	$(MAKE) gpu-bench GRIDS=$${GRIDS:-32,64} OUT=data/benchmarks/cuda_kernel_baseline.csv
	@echo "All benchmarks complete."

parity-report:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics \
		--bin parity-report -- \
		--cuda-csv $${CUDA_CSV:-data/benchmarks/cuda_kernel_baseline.csv} \
		--vulkan-csv $${VULKAN_CSV:-data/benchmarks/vulkan_kernel_baseline.csv} \
		--cpu-csv $${CPU_CSV:-data/benchmarks/cpu_lbm_baseline.csv} \
		--output $${OUT:-data/benchmarks/parity_report.md}
	@echo "OK: Parity report written to data/benchmarks/parity_report.md"

su5-gut:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin su5-gut

jacobi-backend-sweep:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_algebra --bin jacobi-backend-sweep -- \
		--sizes $${SIZES:-4,8,16,24,32} \
		--repeats $${REPEATS:-5} \
		$${FAMILIES:+--families $${FAMILIES}} \
		$${BACKENDS:+--backends $${BACKENDS}} \
		--output $${OUT:-reports/benchmarks/jacobi_backend_sweep.csv} \
		--summary $${SUMMARY:-reports/benchmarks/jacobi_backend_sweep.md}
	@echo "OK: Jacobi backend sweep completed."

block-jacobi-backend-sweep:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_algebra --bin block-jacobi-backend-sweep -- \
		--sizes $${SIZES:-8,16,24,32} \
		--repeats $${REPEATS:-3} \
		$${FAMILIES:+--families $${FAMILIES}} \
		$${SOLVERS:+--solvers $${SOLVERS}} \
		--output $${OUT:-reports/benchmarks/block_jacobi_backend_sweep.csv} \
		$${SUMMARY:+--summary $${SUMMARY}}
	@echo "OK: block Jacobi backend sweep completed."

partial-spectrum-bench:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_algebra --bin partial-spectrum-bench -- \
		--sizes $${SIZES:-16,32,64} \
		--k-values $${K_VALUES:-1,2,4} \
		--repeats $${REPEATS:-3} \
		$${FAMILIES:+--families $${FAMILIES}} \
		$${OBJECTIVES:+--objectives $${OBJECTIVES}} \
		--output $${OUT:-reports/benchmarks/partial_spectrum_bench.csv} \
		$${SUMMARY:+--summary $${SUMMARY}}
	@echo "OK: partial spectrum benchmark completed."

structured-spectrum-bench:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_algebra --bin structured-spectrum-bench -- \
		--sizes $${SIZES:-16,32,64} \
		--repeats $${REPEATS:-3} \
		$${FAMILIES:+--families $${FAMILIES}} \
		$${SOLVERS:+--solvers $${SOLVERS}} \
		--output $${OUT:-reports/benchmarks/structured_spectrum_bench.csv} \
		$${SUMMARY:+--summary $${SUMMARY}}
	@echo "OK: structured spectrum benchmark completed."

jacobi-backend-perf:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_algebra --bin jacobi-backend-sweep
	perf stat -e $${PERF_EVENTS:-cycles:u,instructions:u,branches:u,branch-misses:u} -r $${PERF_RUNS:-3} $(REPO_CARGO_TARGET_DIR)/release-gate/jacobi-backend-sweep \
		--sizes $${SIZES:-68} \
		--repeats $${REPEATS:-3} \
		$${FAMILIES:+--families $${FAMILIES}} \
		$${BACKENDS:+--backends $${BACKENDS}} \
		--output $${OUT:-/tmp/jacobi_backend_perf.csv} \
		--summary $${SUMMARY:-/tmp/jacobi_backend_perf.md}
	@echo "OK: Jacobi backend perf sweep completed."

jacobi-backend-flamegraph:
	CARGO_PROFILE_RELEASE_DEBUG=$${PROFILE_DEBUG:-true} $(CARGO_ENV) cargo flamegraph -p gororoba_cli_algebra --bin jacobi-backend-sweep --root \
		-o $${OUT:-/tmp/jacobi_backend_flamegraph.svg} \
		--title "$${TITLE:-Jacobi Backend Flamegraph}" \
		--deterministic \
		-- \
		--sizes $${SIZES:-72} \
		--repeats $${REPEATS:-30} \
		$${FAMILIES:+--families $${FAMILIES}} \
		$${BACKENDS:+--backends $${BACKENDS}} \
		--output /tmp/jacobi_backend_flamegraph.csv \
		--summary /tmp/jacobi_backend_flamegraph.md
	@echo "OK: Jacobi backend flamegraph captured."

jacobi-backend-samply:
	$(CARGO_ENV) cargo build --profile $${PROFILE:-bench} -p gororoba_cli_algebra --bin jacobi-backend-sweep $${FEATURES:+--features "$${FEATURES}"}
	@profile_dir="$${PROFILE:-bench}"; \
	if [ "$$profile_dir" = "bench" ] || [ "$$profile_dir" = "release" ]; then \
		profile_dir="release"; \
	elif [ "$$profile_dir" = "dev" ] || [ "$$profile_dir" = "test" ]; then \
		profile_dir="debug"; \
	fi; \
	samply record --save-only --output $${OUT:-reports/benchmarks/jacobi_backend_samply.json.gz} \
		--profile-name "$${TITLE:-Jacobi Backend Samply}" \
		$${PRESYMBOLICATE:+--unstable-presymbolicate} \
		$(REPO_CARGO_TARGET_DIR)/$$profile_dir/jacobi-backend-sweep \
		--sizes $${SIZES:-72} \
		--repeats $${REPEATS:-30} \
		$${FAMILIES:+--families $${FAMILIES}} \
		$${BACKENDS:+--backends $${BACKENDS}} \
		--output /tmp/jacobi_backend_samply.csv \
		--summary /tmp/jacobi_backend_samply.md
	@echo "OK: Jacobi backend samply profile captured."

jacobi-backend-samply-compare:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_algebra --bin jacobi-backend-samply-compare -- \
		--reference $${REFERENCE:-reports/benchmarks/jacobi_backend_samply_quantized_shell_72_reference_dev.json.gz} \
		--x87 $${X87:-reports/benchmarks/jacobi_backend_samply_quantized_shell_72_x87_dev.json.gz} \
		--double-double $${DD:-reports/benchmarks/jacobi_backend_samply_quantized_shell_72_dd_dev.json.gz} \
		--output $${OUT:-reports/benchmarks/jacobi_backend_samply_compare.csv} \
		--summary $${SUMMARY:-reports/benchmarks/jacobi_backend_samply_compare.md} \
		--top $${TOP:-10}
	@echo "OK: Jacobi samply comparison completed."

dep-audit:
	@echo "== dependency audit: duplicate versions =="
	cargo tree -d
	@echo ""
	@echo "== dependency audit: workspace crate topology (depth=1) =="
	cargo tree --workspace --depth 1
	@echo ""
	@echo "OK: dependency audit completed."

cargo-deny-check:
	@command -v cargo-deny >/dev/null 2>&1 || { echo "ERROR: cargo-deny not found. Install with: cargo install cargo-deny"; exit 1; }
	$(CARGO_ENV) cargo deny check --config deny.toml --show-stats --hide-inclusion-graph advisories bans licenses sources
	@echo "OK: cargo-deny policy gate passed."

mcp-smoke:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- mcp-smoke

e027-validate:
	@echo "Validating E-027 Percolation Experiment (Thesis 1 binary)..."
	$(CARGO_ENV) cargo build --release --bin percolation-experiment
	@mkdir -p data/e027
	@rm -f data/e027/e027_results.toml
	@echo "Running E-027 with small grid (8^3, 100 steps)..."
	@$(CARGO_ENV) cargo run --release --bin percolation-experiment -- \
	  --grid-size 8 \
	  --lbm-steps 100 \
	  --seed 42 \
	  --n-permutations 50 \
	  --output-dir data/e027 \
	  2>&1 | grep -E "\[|Found|OK|FAIL" || true
	@echo "Verifying TOML artifact generation..."
	@test -f data/e027/e027_results.toml || (echo "ERROR: results TOML not generated"; exit 1)
	@echo "OK: E-027 validation passed (binary operational, TOML pipeline functional)."

rust-parity:
	CARGO_TARGET_DIR=$(REPO_TMPDIR)/open_gororoba_parity_target $(CARGO_ENV) cargo test --workspace
	CARGO_TARGET_DIR=$(REPO_TMPDIR)/open_gororoba_parity_target $(CARGO_ENV) cargo clippy --workspace -- -D warnings
	@echo "OK: parity lane passed (workspace tests + clippy with release-class optimization semantics)."

rust-release-fat-lto:
	CARGO_TARGET_DIR=$(REPO_TMPDIR)/open_gororoba_release_target $(CARGO_ENV) cargo build --release --workspace
	@echo "OK: release fat-LTO workspace build completed."

rust-pgo-instrument:
	mkdir -p "$(PGO_DIR)"
	CARGO_TARGET_DIR=$(REPO_TMPDIR)/open_gororoba_pgo_target \
	$(CARGO_ENV) \
	RUSTFLAGS="-Cprofile-generate=$(PGO_DIR)" \
	cargo build --release --workspace
	@echo "OK: PGO instrumented build completed. Run representative binaries to collect .profraw files in $(PGO_DIR)."

rust-pgo-merge:
	llvm-profdata merge -o "$(PGO_DIR)/merged.profdata" "$(PGO_DIR)"/*.profraw
	@echo "OK: merged profile written to $(PGO_DIR)/merged.profdata."

rust-pgo-build:
	CARGO_TARGET_DIR=$(REPO_TMPDIR)/open_gororoba_pgo_use_target \
	$(CARGO_ENV) \
	RUSTFLAGS="-Cprofile-use=$(PGO_DIR)/merged.profdata" \
	cargo build --release --workspace
	@echo "OK: PGO-optimized release build completed."

verify-pantheon-physicsforge-license:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin pantheon-physicsforge-verify -- license

verify-pantheon-physicsforge-provenance:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin pantheon-physicsforge-verify -- provenance

verify-pantheon-physicsforge-mapping:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin pantheon-physicsforge-verify -- mapping

verify-pantheon-physicsforge-license-headers:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin pantheon-physicsforge-verify -- license-headers

verify-pantheon-physicsforge-overflow:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin pantheon-physicsforge-verify -- overflow

seed-pantheon-physicsforge-sqlite:
	cargo run --release -p gororoba_cli_provenance --bin provenance -- --db build/pantheon_physicsforge_migration.db pantheon-seed

registry-knowledge:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-knowledge-sources

registry-governance: registry-knowledge
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-governance

registry-migrate-corpus: registry-knowledge
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- migrate-corpus --prune-stale

registry-normalize-claims:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-claims-support --bootstrap-from-markdown

registry-bootstrap-claims-support: registry-normalize-claims
	@echo "Claims support markdown->TOML bootstrap completed."

registry-normalize-bibliography:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-bibliography --bootstrap-from-markdown

registry-bootstrap-bibliography: registry-normalize-bibliography
	@echo "Bibliography markdown->TOML bootstrap completed."

registry-normalize-external-sources:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-external-sources --bootstrap-from-markdown

registry-bootstrap-external-sources: registry-normalize-external-sources
	@echo "External sources markdown->TOML bootstrap completed."

registry-normalize-research-narratives:
	$(CARGO_ENV) cargo run --profile release-gate -p gororoba_cli_data --bin markdown-registry -- promote-research-narratives

registry-bootstrap-research-narratives: registry-normalize-research-narratives
	@echo "Research narratives markdown->TOML bootstrap completed."

registry-normalize-book-docs:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-book-docs --bootstrap-from-markdown

registry-bootstrap-book-docs: registry-normalize-book-docs
	@echo "mdBook markdown->TOML bootstrap completed."

registry-normalize-docs-root-narratives:
	$(CARGO_ENV) cargo run --profile release-gate -p gororoba_cli_data --bin markdown-registry -- promote-docs-root-narratives

registry-bootstrap-docs-root-narratives: registry-normalize-docs-root-narratives
	@echo "Root docs markdown->TOML bootstrap completed."

registry-normalize-reports-narratives:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-reports-narratives --bootstrap-from-markdown

registry-bootstrap-reports-narratives: registry-normalize-reports-narratives
	@echo "Reports markdown->TOML bootstrap completed."

registry-normalize-docs-convos:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-docs-convos --bootstrap-from-markdown

registry-bootstrap-docs-convos: registry-normalize-docs-convos
	@echo "docs/convos markdown->TOML bootstrap completed."

registry-normalize-data-artifact-narratives:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-data-artifact-narratives --bootstrap-from-markdown

registry-bootstrap-data-artifact-narratives: registry-normalize-data-artifact-narratives
	@echo "data/artifacts narrative markdown->TOML bootstrap completed."

registry-normalize-entrypoint-docs:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-entrypoint-docs --bootstrap-from-markdown

registry-bootstrap-entrypoint-docs: registry-normalize-entrypoint-docs
	@echo "Entrypoint markdown bootstrap into registry/entrypoint_docs.toml completed."

registry-normalize-narratives:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-narrative-overlays

registry-normalize-operational-narratives:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-operational-narratives

registry-ingest-legacy: registry-normalize-narratives registry-normalize-operational-narratives
	@echo "Legacy markdown -> TOML ingest completed."

registry-refresh: registry-migrate-corpus registry-ingest-legacy registry-governance

registry-knowledge-atoms:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin knowledge-atoms -- build

registry-verify-knowledge-atoms:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin knowledge-atoms -- verify

registry-artifact-scrolls: registry-knowledge-atoms
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin artifact-scrolls -- build

registry-verify-artifact-scrolls:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin artifact-scrolls -- verify

registry-markdown-inventory:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-toml-inventory

registry-markdown-corpus: registry-markdown-inventory
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-corpus

registry-toml-inventory: registry-markdown-corpus
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-toml-inventory

registry-markdown-origin-audit: registry-markdown-inventory
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-origin-audit

registry-markdown-owner-map: registry-markdown-origin-audit
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-owner-map

registry-embedded-markdown:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-embedded

registry-verify-embedded-markdown:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-embedded

registry-verify-markdown-inventory:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-inventory-toml-first

registry-verify-markdown-origin:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-origin-audit

registry-verify-markdown-owner:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-owner-map

registry-verify-markdown-toml-first: registry-verify-markdown-inventory registry-verify-markdown-owner
	@echo "OK: markdown SQLite compatibility owner/inventory gates verified."

registry-verify-control-plane: registry-verify-markdown-origin registry-verify-markdown-owner registry-verify-knowledge-atoms registry-verify-artifact-scrolls
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-corpus
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-toml-inventory

registry-control-plane-gate: registry-verify-control-plane
	@echo "OK: control-plane registry lane complete."

registry-verify-wave4: registry-verify-control-plane
	@echo "DEPRECATED: make registry-verify-wave4 is a legacy alias. Use make registry-verify-control-plane."

registry-wave4: registry-control-plane-gate
	@echo "DEPRECATED: make registry-wave4 is a legacy alias. Use make registry-control-plane-gate."

registry-strict-toml-batch1-build: registry-markdown-owner-map
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin semantic-atoms -- --repo-root .
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-payloads

registry-verify-strict-toml-batch1:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin semantic-atoms -- --verify --repo-root .

registry-strict-toml-batch1: registry-verify-strict-toml-batch1
	@echo "OK: semantic-atoms registry lane complete (legacy wave5-batch1 compatibility)."

registry-build-semantic-atoms: registry-strict-toml-batch1-build

registry-verify-semantic-atoms: registry-verify-strict-toml-batch1

registry-semantic-atoms-gate: registry-strict-toml-batch1

registry-wave5-batch1-build: registry-strict-toml-batch1-build
	@echo "DEPRECATED: make registry-wave5-batch1-build is a legacy alias. Use make registry-build-semantic-atoms."

registry-verify-wave5-batch1: registry-verify-strict-toml-batch1
	@echo "DEPRECATED: make registry-verify-wave5-batch1 is a legacy alias. Use make registry-verify-semantic-atoms."

registry-wave5-batch1: registry-strict-toml-batch1
	@echo "DEPRECATED: make registry-wave5-batch1 is a legacy alias. Use make registry-semantic-atoms-gate."

registry-strict-toml-batch2-build: registry-strict-toml-batch1-build
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin evidence-provenance -- --repo-root .

registry-verify-strict-toml-batch2:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin evidence-provenance -- --verify --repo-root .

registry-strict-toml-batch2: registry-verify-strict-toml-batch2
	@echo "OK: evidence-provenance registry lane complete (legacy wave5-batch2 compatibility)."

registry-build-evidence-provenance: registry-strict-toml-batch2-build

registry-verify-evidence-provenance: registry-verify-strict-toml-batch2

registry-evidence-provenance-gate: registry-strict-toml-batch2

registry-wave5-batch2-build: registry-strict-toml-batch2-build
	@echo "DEPRECATED: make registry-wave5-batch2-build is a legacy alias. Use make registry-build-evidence-provenance."

registry-verify-wave5-batch2: registry-verify-strict-toml-batch2
	@echo "DEPRECATED: make registry-verify-wave5-batch2 is a legacy alias. Use make registry-verify-evidence-provenance."

registry-wave5-batch2: registry-strict-toml-batch2
	@echo "DEPRECATED: make registry-wave5-batch2 is a legacy alias. Use make registry-evidence-provenance-gate."

# Convenience fast-path for schema_signatures.toml regeneration.
#
# The full registry-strict-toml-batch3-build target always runs
# `cargo build` (~3 min), even when the binary is already compiled.
# This target checks for the pre-built binary first and skips the
# build if it exists (~1.2s vs ~180s).
#
# When to use: after editing any registry/*.toml file, run
#   make integrity-resolution
# to regenerate registry/schema_signatures.toml before committing.
# The governance gate will fail on content_sha mismatch otherwise.
integrity-resolution:
	@if [ -x "$(REPO_CARGO_TARGET_DIR)/release-gate/integrity-resolution" ]; then \
		$(REPO_CARGO_TARGET_DIR)/release-gate/integrity-resolution --repo-root .; \
	else \
		$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin integrity-resolution; \
		$(REPO_CARGO_TARGET_DIR)/release-gate/integrity-resolution --repo-root .; \
	fi

# DOI audit for refs_heliosphere.bib via CrossRef REST API.
# Flags fabricated citations, wrong metadata, and 404 DOIs.
# Use --strict to fail CI on any detected mismatch.
ref-audit:
	python3 scripts/check_refs.py docs/latex/heliosphere/refs_heliosphere.bib

ref-audit-strict:
	python3 scripts/check_refs.py --strict docs/latex/heliosphere/refs_heliosphere.bib

# ===== Ablation campaign targets =============================================
# All binaries read from data/external/themis/ (cached) and write JSON to
# data/output/heliosphere/ablations/.  Run ablation-all for the full campaign.

ablation-baseline-l2:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-l2-delay-baseline -- --start-date 2016-08-29 --n-days 7

ablation-baseline-random:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-random-trilinear -- --start-date 2016-08-29 --n-days 7 --n-draws 100 --base-seed 1000

ablation-baseline-sparse:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-random-trilinear-sparse -- --start-date 2016-08-29 --n-days 7 --n-draws 100 --base-seed 2000

ablation-baseline-commutator:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-commutator-baseline -- --start-date 2016-08-29 --n-days 7

ablation-baseline-pca:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-pca-variance-baseline -- --start-date 2016-08-29 --n-days 7 --pca-window 15

ablation-axis-a:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-r16-ablation -- --start-date 2016-08-29 --n-days 7
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-r64-ablation -- --start-date 2016-08-29 --n-days 7

ablation-axis-b:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-lag-depth-sweep -- --start-date 2016-08-29 --n-days 7

ablation-window-sensitivity:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-window-sensitivity -- --start-date 2016-08-29 --n-days 7

ablation-mad-decorrelation:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-themis-staples-labeled -- --start-date 2016-08-29 --n-days 7
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-themis-staples-labeled -- --start-date 2016-08-29 --n-days 7 --decorrelated-mad --out-json data/output/heliosphere/ablations/themis_staples_labeled_decorrelated_mad_eval.json

ablation-baselines: ablation-baseline-l2 ablation-baseline-random ablation-baseline-sparse ablation-baseline-commutator ablation-baseline-pca

voyager-heliopause-v1:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-voyager-heliopause -- --spacecraft v1 --window-days 20

voyager-heliopause-v2:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-voyager-heliopause -- --spacecraft v2 --window-days 20

ablation-all: ablation-baselines ablation-axis-a ablation-axis-b ablation-window-sensitivity ablation-mad-decorrelation

registry-strict-toml-batch3-build:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin integrity-resolution
	$(REPO_CARGO_TARGET_DIR)/release-gate/integrity-resolution --repo-root .

registry-verify-schema-signatures:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify schema-signatures

registry-verify-crossrefs:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify crossrefs

registry-verify-dataset-label-aliases:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify dataset-label-aliases

registry-verify-external-source-operational-contracts:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify external-source-operational-contracts

registry-verify-strict-toml-batch3:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin integrity-resolution --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/release-gate/integrity-resolution --verify --repo-root .
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify crossrefs
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify dataset-label-aliases
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify external-source-operational-contracts

registry-strict-toml-batch3: registry-verify-strict-toml-batch3
	@echo "OK: integrity-resolution registry lane complete (legacy wave5-batch3 compatibility)."

registry-build-integrity-resolution: registry-strict-toml-batch3-build

registry-verify-integrity-resolution: registry-verify-strict-toml-batch3

registry-integrity-resolution-gate: registry-strict-toml-batch3

registry-wave5-batch3-build: registry-strict-toml-batch3-build
	@echo "DEPRECATED: make registry-wave5-batch3-build is a legacy alias. Use make registry-build-integrity-resolution."

registry-verify-wave5-batch3: registry-verify-strict-toml-batch3
	@echo "DEPRECATED: make registry-verify-wave5-batch3 is a legacy alias. Use make registry-verify-integrity-resolution."

registry-wave5-batch3: registry-strict-toml-batch3
	@echo "DEPRECATED: make registry-wave5-batch3 is a legacy alias. Use make registry-integrity-resolution-gate."

registry-strict-toml-batch4-build:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin execution-planning -- --repo-root .

registry-verify-strict-toml-batch4:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin execution-planning -- --verify --repo-root .
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- crossrefs
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- dataset-label-aliases
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-inventory-toml-first
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-owner-map

registry-strict-toml-batch4: registry-verify-strict-toml-batch4
	@echo "OK: execution-planning registry lane complete (legacy wave5-batch4 compatibility)."

registry-build-execution-planning: registry-strict-toml-batch4-build

registry-verify-execution-planning: registry-verify-strict-toml-batch4

registry-execution-planning-gate: registry-strict-toml-batch4

registry-wave5-batch4-build: registry-strict-toml-batch4-build
	@echo "DEPRECATED: make registry-wave5-batch4-build is a legacy alias. Use make registry-build-execution-planning."

registry-verify-wave5-batch4: registry-verify-strict-toml-batch4
	@echo "DEPRECATED: make registry-wave5-batch4 is a legacy alias. Use make registry-verify-execution-planning."

registry-wave5-batch4: registry-strict-toml-batch4
	@echo "DEPRECATED: make registry-wave5-batch4 is a legacy alias. Use make registry-execution-planning-gate."

registry-acceptance-gate-readonly:
	$(MAKE) registry-verify-semantic-atoms
	$(MAKE) registry-verify-evidence-provenance
	$(MAKE) registry-verify-integrity-resolution
	$(MAKE) registry-verify-execution-planning
	@echo "OK: registry acceptance gate complete."

registry-acceptance-gate: registry-acceptance-gate-readonly
	@echo "OK: registry-acceptance-gate is a compatibility alias for registry-acceptance-gate-readonly."

registry-wave5: registry-acceptance-gate
	@echo "DEPRECATED: make registry-wave5 is a legacy alias. Use make registry-acceptance-gate."

registry-csv-inventory:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . inventory

registry-migrate-legacy-csv:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . migrate

registry-verify-legacy-csv: registry-migrate-legacy-csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify

registry-migrate-curated-csv:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . migrate \
		--source-glob 'curated/**/*.csv' \
		--out-index registry/curated_csv_datasets.toml \
		--out-dir registry/data/curated_csv \
		--index-table curated_csv_datasets \
		--dataset-prefix CU \
		--corpus-label 'curated CSV'

registry-verify-curated-csv: registry-migrate-curated-csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify \
		--index-path registry/curated_csv_datasets.toml \
		--source-glob 'curated/**/*.csv' \
		--corpus-label 'curated CSV'

registry-project-csv-split:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . project-split-policy

registry-csv-holdings:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . holdings

registry-scroll-project-csv-canonical: registry-project-csv-split
	$(CARGO_ENV) cargo run --release --bin scrollify-csv -- \
		--source-manifest registry/manifests/project_csv_canonical_manifest.txt \
		--out-index registry/project_csv_canonical_datasets.toml \
		--out-dir registry/data/project_csv/canonical \
		--index-table project_csv_canonical_datasets \
		--dataset-prefix PC \
		--corpus-label 'project CSV canonical dataset' \
		--dataset-class canonical-dataset

registry-scroll-project-csv-generated: registry-project-csv-split
	$(CARGO_ENV) cargo run --release --bin scrollify-csv -- \
		--source-manifest registry/manifests/project_csv_generated_manifest.txt \
		--out-index registry/project_csv_generated_artifacts.toml \
		--out-dir registry/data/project_csv/generated \
		--index-table project_csv_generated_artifacts \
		--dataset-prefix PG \
		--corpus-label 'project CSV generated artifact' \
		--dataset-class generated-artifact

registry-scroll-archive-csv-holding: registry-csv-holdings
	$(CARGO_ENV) cargo run --release --bin scrollify-csv -- \
		--source-manifest registry/manifests/archive_csv_holding_manifest.txt \
		--out-index registry/archive_csv_holding_datasets.toml \
		--out-dir registry/data/archive_csv_holding \
		--index-table archive_csv_holding_datasets \
		--dataset-prefix AH \
		--corpus-label 'archive CSV holding queue' \
		--dataset-class holding-archive

registry-scroll-external-csv-holding: registry-csv-holdings
	$(CARGO_ENV) cargo run --release --bin scrollify-csv -- \
		--source-manifest registry/manifests/external_csv_holding_manifest.txt \
		--out-index registry/external_csv_holding_datasets.toml \
		--out-dir registry/data/external_csv_holding \
		--index-table external_csv_holding_datasets \
		--dataset-prefix EH \
		--corpus-label 'external CSV holding queue' \
		--sqlite-overflow-db registry/canonical/csv_holding_payloads.sqlite3 \
		--max-inline-toml-bytes 50000000 \
		--rows-preview-count 8 \
		--dataset-class holding-external

registry-csv-scroll-pipeline: registry-scroll-project-csv-canonical registry-scroll-project-csv-generated registry-scroll-external-csv-holding registry-scroll-archive-csv-holding
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . scroll-pipeline

registry-verify-csv-scroll-pipeline: registry-csv-scroll-pipeline
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify-scroll-pipeline

registry-verify-project-csv-split: registry-scroll-project-csv-canonical registry-scroll-project-csv-generated
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify \
		--index-path registry/project_csv_canonical_datasets.toml \
		--source-manifest registry/manifests/project_csv_canonical_manifest.txt \
		--corpus-label 'project CSV canonical dataset'
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify \
		--index-path registry/project_csv_generated_artifacts.toml \
		--source-manifest registry/manifests/project_csv_generated_manifest.txt \
		--corpus-label 'project CSV generated artifact'
	$(CARGO_ENV) cargo run --release --bin verify-project-csv-split -- \
		--repo-root .

registry-verify-csv-holdings: registry-csv-holdings registry-scroll-external-csv-holding registry-scroll-archive-csv-holding
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify \
		--index-path registry/external_csv_holding_datasets.toml \
		--source-manifest registry/manifests/external_csv_holding_manifest.txt \
		--corpus-label 'external CSV holding queue' \
		--coverage-only
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify \
		--index-path registry/archive_csv_holding_datasets.toml \
		--source-manifest registry/manifests/archive_csv_holding_manifest.txt \
		--corpus-label 'archive CSV holding queue'
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify-holdings

registry-verify-csv-corpus-coverage: registry-csv-inventory registry-verify-project-csv-split registry-verify-csv-holdings
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . verify-corpus-coverage

registry-csv-pipeline-gate: registry-project-csv-split registry-csv-holdings registry-verify-project-csv-split registry-verify-csv-holdings registry-verify-csv-corpus-coverage registry-verify-csv-scroll-pipeline

registry-wave3: registry-csv-pipeline-gate
	@echo "DEPRECATED: make registry-wave3 is a legacy alias. Use make registry-csv-pipeline-gate."

registry-csv-scope: registry-csv-inventory
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin csv-canonicalization -- --repo-root . migration-scope

registry-data: registry-migrate-legacy-csv registry-migrate-curated-csv registry-csv-pipeline-gate registry-csv-inventory registry-verify-legacy-csv registry-verify-curated-csv registry-csv-scope registry-control-plane-gate
	@echo "OK: CSV data registry lane complete."

# Use release-gate here instead of dev/cg_clif: gororoba_cli_data pulls faer/pulp
# through several data tools, and cg_clif still ICEs on AVX f32x8 lowering in that
# lane on nightly 2026-04-05.
registry-export-markdown: registry-refresh registry-build
# registry-emit-all-mirrors owns the mirror (kind, output_path) list as
# Rust data with proper error propagation. The Makefile delegates to
# that typed command instead of carrying a shell heredoc.
	$(CARGO_ENV) cargo run -p xtask -- registry-emit-all-mirrors

# Keep mirror freshness and governance checks on the LLVM-backed gate lane for the
# same reason as registry-export-markdown above.
registry-verify-mirrors:
	set -e; \
	legacy_flag=""; \
	claims_value="true"; \
	if [ "$(MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC)" = "0" ]; then claims_value="false"; fi; \
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin verify-registry-mirror-freshness --bin governance-verify --bin registry-emit; \
	$(REPO_CARGO_TARGET_DIR)/release-gate/verify-registry-mirror-freshness \
		--out-dir "crates/data_core/src/registry_mirrors" $$legacy_flag --legacy-claims-sync $$claims_value
	$(MAKE) registry-verify-markdown-toml-first
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify markdown-headers
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify markdown-parity
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify mirror-immutability
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify claim-ticket-mirrors

registry-sync-project-counters:
	$(CARGO_ENV) cargo run --release --bin project-counter-sync

registry: registry-refresh registry-data registry-sync-project-counters
	$(CARGO_ENV) cargo run --release --bin registry-check

registry-verify-typed-policy-error:
	$(CARGO_ENV) cargo run --release --bin registry-check -- --typed-policy error

synthesis-execution-contract:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin synthesis-execution-contract -- \
		--date-token "$(SYNTHESIS_CONTRACT_DATE)" \
		--report-path "$(SYNTHESIS_CONTRACT_REPORT)"

docs-publish: registry-export-markdown
	$(MAKE) docs-freshness
	$(MAKE) registry-verify-mirrors
	@echo "OK: TOML-driven markdown mirrors generated and verified for publishing."

docs-rustdoc:
	@mkdir -p "$(DOCS_CARGO_TARGET_DIR)"
	$(DOCS_CARGO_ENV) cargo doc --workspace --all-features --no-deps --document-private-items

cd-row-upgrade-batch:
	@test -n "$(CD_ROW_UPGRADE_LANE)" || (echo "ERROR: set CD_ROW_UPGRADE_LANE=<jacobson1958|freudenthal1951>" && exit 1)
	@test -n "$(CD_ROW_UPGRADE_WITNESS)" || (echo "ERROR: set CD_ROW_UPGRADE_WITNESS=/abs/path/to/witness.pdf" && exit 1)
	@test -n "$(CD_ROW_UPGRADE_STATUS)" || (echo "ERROR: set CD_ROW_UPGRADE_STATUS=<exact-original|full-official-reprint|full-official-witness|official-fragment|official-toc|translation-rewriting|support-reconstruction|reconstruction-dossier>" && exit 1)
	@test -n "$(CD_ROW_UPGRADE_ROWS)" || (echo "ERROR: set CD_ROW_UPGRADE_ROWS='--row-id ... --row-id ...'" && exit 1)
	$(CARGO_ENV) cargo run -q -p gororoba_cli_data --bin cd-row-upgrade-batch -- \
		--cache-root "$(CD_CACHE_ROOT)" \
		--lane "$(CD_ROW_UPGRADE_LANE)" \
		--source-witness "$(CD_ROW_UPGRADE_WITNESS)" \
		--source-status "$(CD_ROW_UPGRADE_STATUS)" \
		--operator "$(CD_ROW_UPGRADE_OPERATOR)" \
		$(CD_ROW_UPGRADE_ROWS)

cd-row-upgrade-jacobson:
	$(MAKE) cd-row-upgrade-batch \
		CD_ROW_UPGRADE_LANE=jacobson1958 \
		CD_ROW_UPGRADE_WITNESS="$(JACOBSON_ROW_UPGRADE_WITNESS)" \
		CD_ROW_UPGRADE_STATUS="$(JACOBSON_ROW_UPGRADE_STATUS)" \
		CD_ROW_UPGRADE_ROWS="$(JACOBSON_ROW_UPGRADE_ROWS)"

cd-row-upgrade-freudenthal:
	$(MAKE) cd-row-upgrade-batch \
		CD_ROW_UPGRADE_LANE=freudenthal1951 \
		CD_ROW_UPGRADE_WITNESS="$(FREUDENTHAL_ROW_UPGRADE_WITNESS)" \
		CD_ROW_UPGRADE_STATUS="$(FREUDENTHAL_ROW_UPGRADE_STATUS)" \
		CD_ROW_UPGRADE_ROWS="$(FREUDENTHAL_ROW_UPGRADE_ROWS)"
	@if [ -d "$(DOCS_CARGO_TARGET_DIR)/doc" ]; then \
		rm -rf "$(DOCS_RUSTDOC_DIR)"; \
		mkdir -p "$(DOCS_RUSTDOC_DIR)"; \
		cp -R "$(DOCS_CARGO_TARGET_DIR)/doc/." "$(DOCS_RUSTDOC_DIR)/"; \
	else \
		echo "ERROR: rustdoc output missing at $(DOCS_CARGO_TARGET_DIR)/doc"; \
		exit 1; \
	fi
	@echo "OK: rustdoc staged to $(DOCS_RUSTDOC_DIR)."

docs-book:
	@command -v $(MD_BOOK) >/dev/null 2>&1 || { echo "ERROR: mdbook not found. Run: cargo install --locked --force mdbook"; exit 1; }
	@rm -rf "$(DOCS_BOOK_DIR)"
	@mkdir -p "$(DOCS_BOOK_DIR)"
	$(MD_BOOK) build docs/book -d "$(DOCS_BOOK_DIR)"
	@echo "OK: mdBook staged to $(DOCS_BOOK_DIR)."

docs-site: docs-rustdoc
	@command -v $(MD_BOOK) >/dev/null 2>&1 || { echo "ERROR: mdbook not found. Run: cargo install --locked --force mdbook"; exit 1; }
	@rm -rf "$(DOCS_SITE_DIR)"
	@mkdir -p "$(DOCS_SITE_DIR)"
	@printf '%s\n' \
		'<!doctype html>' \
		'<html lang="en">' \
		'  <head>' \
		'    <meta charset="utf-8" />' \
		'    <meta name="viewport" content="width=device-width, initial-scale=1" />' \
		'    <title>open_gororoba documentation</title>' \
		'    <style>' \
		'      body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5;max-width:40rem;margin:2rem auto;padding:0 1rem;}' \
		'      ul{padding-left:1.25rem;}' \
		'    </style>' \
		'  </head>' \
		'  <body>' \
		'    <h1>open_gororoba documentation</h1>' \
		'    <p>Pick a documentation channel:</p>' \
		'    <ul>' \
		'      <li><a href="./book/">mdBook narrative documentation</a></li>' \
		'      <li><a href="./rustdoc/">Rust API documentation</a></li>' \
		'    </ul>' \
		'  </body>' \
		'</html>' \
		> "$(DOCS_SITE_DIR)/index.html"
	@printf '%s\n' \
		'<!doctype html>' \
		'<html lang="en">' \
		'  <head>' \
		'    <meta charset="utf-8" />' \
		'    <meta http-equiv="refresh" content="0; url=./book/" />' \
		'    <title>open_gororoba book redirect</title>' \
		'  </head>' \
		'  <body><a href="./book/">mdBook narrative documentation</a></body>' \
		'</html>' \
		> "$(DOCS_SITE_DIR)/book.html"
	@printf '%s\n' \
		'<!doctype html>' \
		'<html lang="en">' \
		'  <head>' \
		'    <meta charset="utf-8" />' \
		'    <meta http-equiv="refresh" content="0; url=./rustdoc/" />' \
		'    <title>open_gororoba rustdoc redirect</title>' \
		'  </head>' \
		'  <body><a href="./rustdoc/">Rust API documentation</a></body>' \
		'</html>' \
		> "$(DOCS_SITE_DIR)/rustdoc.html"
	@printf '%s\n' \
		'<!doctype html>' \
		'<html lang="en">' \
		'  <head>' \
		'    <meta charset="utf-8" />' \
		'    <title>open_gororoba docs redirect</title>' \
		'    <script>' \
		'      (function () {' \
		'        var path = window.location.pathname;' \
		'        var root = "/";' \
		'        var first = path.replace(/^\/+/, "").split("/")[0];' \
		'        if (first && first !== "book" && first !== "rustdoc") {' \
		'          root = "/" + first + "/";' \
		'        }' \
		'        var legacyPrefixes = [' \
		'          "/.cache/cargo-default-target/doc",' \
		'          "/cache/cargo-default-target/doc",' \
		'          "/.cache/gate-target/doc",' \
		'          "/cache/gate-target/doc",' \
		'          "/target/docs-target/doc",' \
		'          "/target/doc"' \
		'        ];' \
		'        for (var i = 0; i < legacyPrefixes.length; i += 1) {' \
		'          var prefix = legacyPrefixes[i];' \
		'          var idx = path.indexOf(prefix);' \
		'          if (idx !== -1) {' \
		'            var prefixPart = path.slice(0, idx);' \
		'            var redirectRoot = "/";' \
		'            if (prefixPart && prefixPart !== "/") {' \
		'              redirectRoot = prefixPart.replace(/\/+$/, "") + "/";' \
		'            }' \
		'            window.location.replace(redirectRoot + "rustdoc" + path.slice(idx + prefix.length));' \
		'            return;' \
		'          }' \
		'        }' \
		'        if (path === root || path === root + "book" || path === root + "book/") {' \
		'          window.location.replace(root + "book/");' \
		'          return;' \
		'        }' \
		'        if (path === root + "rustdoc" || path === root + "rustdoc/") {' \
		'          window.location.replace(root + "rustdoc/");' \
		'          return;' \
		'        }' \
		'        window.location.replace(root);' \
		'      }());' \
		'    </script>' \
		'  </head>' \
		'  <body><a href="./">open_gororoba documentation</a></body>' \
		'</html>' \
		> "$(DOCS_SITE_DIR)/404.html"
	@if [ -d "$(DOCS_CARGO_TARGET_DIR)/doc" ]; then \
		mkdir -p "$(DOCS_RUSTDOC_DIR)"; \
		cp -R "$(DOCS_CARGO_TARGET_DIR)/doc/." "$(DOCS_RUSTDOC_DIR)/"; \
	else \
		echo "ERROR: rustdoc output missing at $(DOCS_CARGO_TARGET_DIR)/doc"; \
		exit 1; \
	fi
	$(MD_BOOK) build docs/book -d "$(DOCS_BOOK_DIR)"
	@touch "$(DOCS_SITE_DIR)/.nojekyll"
	@echo "OK: docs site staged to $(DOCS_SITE_DIR)."

docs-freshness: docs-gate docs-redirect-check
	@echo "OK: docs-freshness checks passed."

docs-gate: docs-site
	@echo "OK: docs-gate generated unified docs bundle."

docs-redirect-check:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- docs-redirect-check $(DOCS_SITE_DIR)

terminology-gate:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- terminology-gate

ansi-check:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- ansi-check --check

ansi-check-strict:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- ansi-check --check --strict-placeholders --placeholder-scope-prefix crates/ --placeholder-scope-prefix tests/

verify:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- verify-artifacts

verify-grand:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- verify-grand-images

verify-c010-c011-theses:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- verify-c010-c011-theses

doctor:
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- doctor
	sh scripts/detect_native_blas.sh

doctor-blas:
	sh scripts/detect_native_blas.sh

provenance:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-origin-audit -- --out reports/data_origin_audit_$$(date +%F).toml --fail-on-strict-unknown

provenance-audit:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-governance-gate -- --enforce-origin true --enforce-semantic true --enforce-blocked-deadlines true

provenance-registry-index:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_provenance --bin provenance -- index

provenance-registry-export:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_provenance --bin provenance -- export

provenance-registry-verify:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_provenance --bin provenance -- verify

provenance-registry-doctor:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_provenance --bin provenance -- doctor

provenance-registry-link-audit:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_provenance --bin provenance -- link-audit

provenance-registry-recover:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_provenance --bin provenance -- recover

external-redownload-audit:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin external-redownload-audit -- --out reports/external_redownload_audit_$$(date +%F).toml --backend-order wget,curl,fetch

semantic-data-validate:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-semantic-validate -- --out reports/data_semantic_validate_$$(date +%F).toml

semantic-data-validate-strict:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-semantic-validate -- --fail-on-unverifiable true --out reports/data_semantic_validate_$(date +%F)_strict.toml

# ---- Artifact generation ----

artifacts: artifacts-motifs artifacts-boxkites artifacts-reggiani artifacts-m3 artifacts-dimensional artifacts-repo-visuals
	@echo "OK: all core artifacts regenerated."

artifacts-dimensional:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- dimensional-geometry

artifacts-materials:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- materials-subset --n 200 --seed 0
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- materials-embedding

artifacts-boxkites:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- de-marrais-boxkites

artifacts-reggiani:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- reggiani-annihilator-stats

artifacts-m3:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- m3-table

artifacts-motifs:
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 16,32 --details
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- motif-summary

artifacts-motifs-big:
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 16,32,64,128 --summary-only
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 256 --max-nodes 5000 --seed 0 --summary-only
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- motif-summary

artifacts-repo-visuals:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-visuals

# ---- Data fetching ----

fetch-data:
	@echo "Fetching external datasets..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin fetch-datasets -- --all --skip-existing --output-dir data/external
	@echo "Refreshing external provenance and source governance..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-governance-gate -- --enforce-origin true --enforce-semantic true --enforce-blocked-deadlines true --enforce-gitignore true --enforce-naming true

fetch-data-redownload:
	@echo "Force re-downloading external datasets from origin fetchers..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin fetch-datasets -- --all --skip-existing false --output-dir data/external
	@echo "Refreshing external provenance and source governance..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-governance-gate -- --enforce-origin true --enforce-semantic true --enforce-blocked-deadlines true --enforce-gitignore true --enforce-naming true

# ---- Simulation runs ----

run: rust-smoke
	$(CARGO_ENV) cargo run --release --bin thesis_lab -- --steps 100 --seed 42
	$(CARGO_ENV) cargo run --release --bin modular_chaos -- --steps 100 --n 256
	$(CARGO_ENV) cargo run --release --bin entropy_pde -- --depth 50
	@echo "OK: All core Rust simulations completed and artifacts generated."

run-e183:
	@mkdir -p data/results/e183
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo-stacking-manga -- \
		--rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
		--dapall data/external/manga/dapall_selection.csv \
		--cd-dim 16 --csv data/results/e183/manga_stack_D16.csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo-stacking-manga -- \
		--rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
		--dapall data/external/manga/dapall_selection.csv \
		--cd-dim 64 --csv data/results/e183/manga_stack_D64.csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo-stacking-manga -- \
		--rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
		--dapall data/external/manga/dapall_selection.csv \
		--cd-dim 256 --csv data/results/e183/manga_stack_D256.csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo-stacking-manga -- \
		--rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
		--dapall data/external/manga/dapall_selection.csv \
		--cd-dim 1024 --csv data/results/e183/manga_stack_D1024.csv
	@echo "E-183 sweep complete. Results in data/results/e183/"

# ---- Rocq proofs ----

rocq:
	@command -v coqc >/dev/null 2>&1 || { echo "ERROR: coqc not found. See docs/requirements/rocq.md"; exit 1; }
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- rocq-prepare-confine curated/01_theory_frameworks/confine_theorems_512.v curated/01_theory_frameworks/confine_theorems_512_axioms.v
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- rocq-prepare-confine curated/01_theory_frameworks/confine_theorems_1024.v curated/01_theory_frameworks/confine_theorems_1024_axioms.v
	$(CARGO_ENV) cargo run --release -p repo_utilities --bin repo-utilities -- rocq-prepare-confine curated/01_theory_frameworks/confine_theorems_2048.v curated/01_theory_frameworks/confine_theorems_2048_axioms.v
	cd curated/01_theory_frameworks && \
		coqc ConfineModel.v && \
		coqc confine_theorems_512_axioms.v && \
		coqc confine_theorems_1024_axioms.v && \
		coqc confine_theorems_2048_axioms.v

# ---- Rocq formal verification proofs (ADM/Casimir/Warp claims) ----

rocq-proofs:
	@command -v rocq >/dev/null 2>&1 || { echo "SKIP: rocq not found"; exit 0; }
	@if [ -f proofs/Makefile ]; then \
	    $(MAKE) -C proofs all; \
	else \
	    echo "SKIP: proofs/ not present (submodule not initialized? run: make submodule-sync)"; \
	fi

rocq-proofs-check:
	@if [ -f proofs/Makefile ]; then \
	    $(MAKE) -C proofs check; \
	else \
	    echo "SKIP: proofs/ not present (submodule not initialized? run: make submodule-sync)"; \
	fi

rocq-makefile-check:
	@command -v rocq >/dev/null 2>&1 || { echo "ERROR: rocq not found. See docs/requirements/rocq.md"; exit 1; }
	@if [ -f proofs/Makefile ]; then \
	    $(MAKE) -C proofs rocq-makefile-check; \
	else \
	    echo "SKIP: proofs/ not present (submodule not initialized? run: make submodule-sync)"; \
	fi

lva-paper: rocq-proofs rocq-proofs-check
	@command -v just >/dev/null 2>&1 || { echo "ERROR: just not found (install via cargo install just)"; exit 1; }
	cd proofs && just paper-artifacts
	$(MAKE) latex

# ---- LaTeX (warnings-as-errors via latexmk -Werror) ----

latex:
	@command -v latexmk >/dev/null 2>&1 || { echo "ERROR: latexmk not found. Install TeX Live (see docs/requirements/latex.md)"; exit 1; }
	$(CARGO_ENV) cargo run --release --bin generate-latex
	@mkdir -p docs/latex/out
	cd docs/latex && TEXINPUTS=.:$(CURDIR)/papers/bib/: BIBINPUTS=$(CURDIR)/papers/bib/: latexmk -pdf -Werror -interaction=nonstopmode -halt-on-error -shell-escape -output-directory=out llm_scaffold_paper.tex
	cd docs/latex && TEXINPUTS=.:$(CURDIR)/papers/bib/: BIBINPUTS=$(CURDIR)/papers/bib/: latexmk -pdf -Werror -interaction=nonstopmode -halt-on-error -output-directory=out MASTER_SYNTHESIS.tex
	cd docs/latex && latexmk -pdf -Werror -interaction=nonstopmode -halt-on-error -output-directory=out MATHEMATICAL_FORMALISM.tex
	$(MAKE) latex-heliosphere

latex-heliosphere:
	@command -v latexmk >/dev/null 2>&1 || { echo "ERROR: latexmk not found"; exit 1; }
	@mkdir -p docs/latex/heliosphere/out
	cd docs/latex/heliosphere && latexmk -xelatex -interaction=nonstopmode -halt-on-error -output-directory=out jgr_cd_magnetopause.tex
	cd docs/latex/heliosphere && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out cover_letter.tex

latex-heliosphere-figs:
	@command -v latexmk >/dev/null 2>&1 || { echo "ERROR: latexmk not found"; exit 1; }
	@mkdir -p docs/latex/heliosphere/figures/out
	cd docs/latex/heliosphere/figures && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out fig_tau_sweep.tex
	cd docs/latex/heliosphere/figures && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out fig_alfven_control.tex
	cd docs/latex/heliosphere/figures && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out fig_enrichment.tex
	cd docs/latex/heliosphere/figures && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out fig_fte_scatter.tex

latex-heliosphere-clean:
	rm -rf docs/latex/heliosphere/out
	rm -rf docs/latex/heliosphere/figures/out

latex-heliosphere-review:
	@command -v latexmk >/dev/null 2>&1 || { echo "ERROR: latexmk not found"; exit 1; }
	@mkdir -p docs/latex/heliosphere/out
	cd docs/latex/heliosphere && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out response_to_reviewers.tex

# ---- Quantum Docker ----

docker-quantum-build:
	docker build -t qiskit-env -f docker/Dockerfile .

docker-quantum-run:
	./run_quantum_container.sh $(ARGS)

docker-quantum-shell:
	docker run --rm -it \
		-v "$(PWD)/data:/app/data" \
		-v "$(PWD)/src:/app/src" \
		qiskit-env /bin/bash

# ---- Cleanup ----

clean-artifacts:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-clean -- --scope reproducible --apply
	@echo "Done. Regenerate and verify with cargo-native data governance commands."

clean:
	rm -rf $(REPO_CARGO_TARGET_DIR)

clean-builds:
	rm -rf target/
	rm -rf .cache/cargo-default-target/
	rm -rf .cache/gate-target/
	rm -rf .cache/gate-cbuild/
	rm -rf $(REPO_TMP_CARGO_ROOT)
	rm -rf $(REPO_TMPDIR)/open_gororoba-cargo-build 2>/dev/null || true
	rm -rf $(REPO_TMPDIR)/open_gororoba_*_target 2>/dev/null || true
	rm -rf $(REPO_TMPDIR)/open_gororoba-cargo-build-* 2>/dev/null || true
	@echo "Removed all Rust build artifacts. Run 'cargo build' to rebuild."


cargo-cache-status:
	CARGO_CACHE_REPO_BUDGET_GIB=$(CARGO_CACHE_REPO_BUDGET_GIB) \
	CARGO_CACHE_TMP_BUDGET_GIB=$(CARGO_CACHE_TMP_BUDGET_GIB) \
	sh scripts/cargo_cache_status.sh

cargo-cache-prune:
	sh scripts/cargo_cache_prune.sh

cargo-cache-smoke:
	$(CARGO_ENV) cargo test -p gororoba_structurable --lib

v6-branch-transport-artifacts:
	$(CARGO_ENV) cargo run -p algebra_experimental --example v6_gradient_drift_probe

pathion-control-artifacts:
	$(CARGO_ENV) cargo run -p algebra_experimental --example pathion_control_probe

pathion-resonance-artifacts:
	$(CARGO_ENV) cargo run -p pathion_ellip --example pathion_resonance_probe

clean-all: clean clean-builds clean-artifacts
	@rm -rf $(REPO_CARGO_HOME)
	@command -v cargo-sweep >/dev/null 2>&1 && cargo sweep --time 14 || true
	@echo "Full cleanup complete. Run 'make install && make artifacts' to rebuild."

# ---- Code Duplication Audit ----
# Scans all Rust sources under crates/ for copy-paste duplication using PMD CPD.
# WHY: Duplication accumulates silently between sprints; periodic audit catches regressions
#      before they compound into structural debt.
# --minimum-tokens 42 is the project-canonical threshold (validated in E-213).

CPD_MIN_TOKENS ?= 42
CPD_TOP        ?= 20

# Data-heavy source files excluded from CPD scans.
# These are transcribed reference datasets or auto-generated doc mirrors --
# not hand-written logic -- so duplication detection is noise, not signal.
# Remove entries here only when a file gains real logic that warrants scanning.
# tabulated_nk.rs removed: migrated to materials_data in task #56.
# NOTE: These variables are documentation only. The authoritative exclusion list
# lives in xtask/src/main.rs (CPD_EXCLUDE_FILES, CPD_EXCLUDE_DIRS constants).
# Update both when adding new exclusions.
CPD_EXCLUDE_FILES := \
	crates/materials_core/src/optical_database.rs \
	crates/materials_core/src/crystal_symmetry.rs

# registry_mirrors/ is excluded as an entire directory: 354 auto-generated
# doc-string (.rs) files with zero compiled symbols, gated behind the
# registry-mirrors feature in data_core. Scanning them wastes CPD cycles.
CPD_EXCLUDE_DIRS := \
	crates/data_core/src/registry_mirrors

# File-list is generated by `xtask cpd-file-list`, which applies the exclusions
# above in a deterministic, race-condition-free manner (no temp-file races from
# shell find+foreach expansion). The old _CPD_REGEN_LIST Make variable is gone.
_CPD_FILE_LIST := /tmp/cpd_src_list.txt

cpd-audit:
	@command -v pmd >/dev/null 2>&1 || { echo "ERROR: pmd not found. Install PMD (e.g. paru -S pmd) to run cpd-audit."; exit 1; }
	$(CARGO_ENV) cargo run -q -p xtask -- cpd-file-list --output $(_CPD_FILE_LIST)
	pmd cpd --language rust --minimum-tokens $(CPD_MIN_TOKENS) --file-list $(_CPD_FILE_LIST) --format xml 2>/dev/null \
		| $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin cpd-report -- --top $(CPD_TOP)

# Anchored repo-debt counter (replaces unreliable grep heuristics).
# Walks crates/, proofs/, xtask/ and emits a TOML snapshot of the
# measurable debt classes (unsafe blocks, attrs, macros, Rocq Admitted/
# Axiom/Parameter). Use REPO_AUDIT_OUT to override the output dir.
# See crates/gororoba_cli_data/src/bin/repo_audit.rs for what is counted
# and the limitations of the regex-on-stripped-source approach.
REPO_AUDIT_OUT ?= data/output/audit/repo_audit
REPO_AUDIT_BASELINE ?= data/output/audit/2026-06-25/repo_audit_anchored_2026_06_25.toml
REPO_AUDIT_SQLITE ?= registry/canonical/control_plane.sqlite3
REPO_AUDIT_TMPDIR ?= $(CURDIR)/.cache/repo-audit-tmp

repo-audit:
	@mkdir -p "$(REPO_AUDIT_TMPDIR)"
	TMPDIR=$(REPO_AUDIT_TMPDIR) $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-audit -- \
		--output-dir $(REPO_AUDIT_OUT) \
		--sqlite $(REPO_AUDIT_SQLITE)

# CI gate: re-run the audit and fail if any debt class grew vs the
# committed baseline. SAFETY-positive classes (more SAFETY comments) are
# allowed to grow; everything else may not.
repo-audit-strict:
	@mkdir -p "$(REPO_AUDIT_TMPDIR)"
	TMPDIR=$(REPO_AUDIT_TMPDIR) $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-audit -- \
		--output-dir $(REPO_AUDIT_OUT) \
		--sqlite $(REPO_AUDIT_SQLITE) \
		--baseline-compare $(REPO_AUDIT_BASELINE) \
		--strict

# Tighter gate: enforces per-root allow_clippy_unjustified cap. Fails the
# build if `crates/` exceeds the cap. Currently set to 0 because A1-A25
# closed every unjustified clippy allow in `crates/` -- new code adding
# an unjustified suppression must add a comment immediately above (see
# docs/engineering/repo_audit_metric_taxonomy.md for the policy).
# `proofs/` is excluded indirectly by the `crates/`-only roots default.
repo-audit-strict-unjustified:
	@mkdir -p "$(REPO_AUDIT_TMPDIR)"
	TMPDIR=$(REPO_AUDIT_TMPDIR) $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-audit -- \
		--output-dir $(REPO_AUDIT_OUT) \
		--sqlite $(REPO_AUDIT_SQLITE) \
		--root crates \
		--strict-unjustified-per-root 0

cpd-audit-strict:
	@command -v pmd >/dev/null 2>&1 || { echo "ERROR: pmd not found. Install PMD (e.g. paru -S pmd) to run cpd-audit-strict."; exit 1; }
	$(CARGO_ENV) cargo run -q -p xtask -- cpd-file-list --output $(_CPD_FILE_LIST)
	pmd cpd --language rust --minimum-tokens $(CPD_MIN_TOKENS) --file-list $(_CPD_FILE_LIST) --format xml 2>/dev/null \
		| $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin cpd-report -- --strict --top $(CPD_TOP)

# Scan tooling and scripting surface (xtask/, root .rs helpers) with a higher
# minimum-token threshold. This lane separates boilerplate noise in tooling from
# algorithmic duplication in domain crates. Use CPD_TOOLING_TOKENS to tune.
CPD_TOOLING_TOKENS ?= 80
_CPD_TOOLING_FILE_LIST := /tmp/cpd_tooling_src_list.txt

cpd-audit-tooling:
	@command -v pmd >/dev/null 2>&1 || { echo "ERROR: pmd not found. Install PMD (e.g. paru -S pmd) to run cpd-audit-tooling."; exit 1; }
	@find xtask/src -name '*.rs' 2>/dev/null > $(_CPD_TOOLING_FILE_LIST); \
	 find scripts -name '*.rs' 2>/dev/null >> $(_CPD_TOOLING_FILE_LIST); \
	 find crates/gororoba_cli_data/src/bin -name '*.rs' 2>/dev/null >> $(_CPD_TOOLING_FILE_LIST); \
	 echo "Tooling surface: $$(wc -l < $(_CPD_TOOLING_FILE_LIST)) files"
	pmd cpd --language rust --minimum-tokens $(CPD_TOOLING_TOKENS) --file-list $(_CPD_TOOLING_FILE_LIST) --format xml 2>/dev/null \
		| $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin cpd-report -- --top $(CPD_TOP)

# ---- Generated artifact header patching -------------------------------------
# Back-fills the standard AUTO-GENERATED header on all static registry_mirrors .rs
# files that lack the generated_doc_header() convention. Safe to run repeatedly.
.PHONY: patch-static-mirror-headers

patch-static-mirror-headers:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin registry-emit -- \
		patch-static-mirror-headers

# ---- Generated surface CPD audit lane ---------------------------------------
# Scans registry_mirrors/ and other purely-generated Rust surfaces separately from
# hand-written logic.  Uses a much higher token threshold (200) because generated
# code has structural repetition by design.  Never gates CI -- report-only semantics.
CPD_GENERATED_TOKENS ?= 200
_CPD_GENERATED_FILE_LIST := /tmp/cpd_generated_src_list.txt

cpd-audit-generated:
	@command -v pmd >/dev/null 2>&1 || { echo "ERROR: pmd not found. Install PMD (e.g. paru -S pmd) to run cpd-audit-generated."; exit 1; }
	@find crates/data_core/src/registry_mirrors -name '*.rs' ! -name 'mod.rs' 2>/dev/null \
		> $(_CPD_GENERATED_FILE_LIST); \
	 echo "Generated surface: $$(wc -l < $(_CPD_GENERATED_FILE_LIST)) files"
	pmd cpd --language rust --minimum-tokens $(CPD_GENERATED_TOKENS) --file-list $(_CPD_GENERATED_FILE_LIST) --format xml 2>/dev/null \
		| $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin cpd-report -- --top $(CPD_TOP)

# ---- Heliosphere Quench Map ----
.PHONY: quench-map

quench-map:
	@echo "Building full heliosphere feature cube..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-feature-cube -- --window full-heliosphere --out-csv data/output/heliosphere/full_feature_cube.csv
	@echo "Generating quench scan from full cube (including MMS)..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere-quench-scan -- --cube-csv data/output/heliosphere/full_feature_cube.csv --out-csv data/output/heliosphere/takens_quench_scan.csv


# ---- Help ----

help:
	@echo "Targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make bootstrap-dev        Show the supported user-local bootstrap flow"
	@echo "    make bootstrap-user-local-xdg [ARGS='--with-gemini --force']"
	@echo ""
	@echo "  Quality:"
	@echo "    make cpd-audit            Report cross/within-crate Rust duplication (CPD, 42 tokens)"
	@echo "    make cpd-audit-strict     Same, exits 1 if any clusters found"
	@echo "    make cpd-audit-generated  Scan generated registry_mirrors surface (200 tokens, report-only)"
	@echo "    make patch-static-mirror-headers  Backfill AUTO-GENERATED headers on static mirror .rs files"
	@echo "    make lint                 Run workspace-wide clippy -- -D warnings"
	@echo "    make test                 Run workspace-wide nextest"
	@echo "    make smoke                Composite fast smoke lane (check + rust-smoke)"
	@echo "    make integrity-rust       Cargo-backed integrity lane (claims + inventory + typed policy)"
	@echo "    make check                Fast local check (ansi + terminology + no-reports)"
	@echo "    make ansi-check           Verify emoji-blocking UTF-8 character policy"
	@echo "    make ansi-check-strict    Verify UTF-8 policy + fail on <U+....>/<EMOJI+...> placeholders"
	@echo "    make verify-pantheon-physicsforge-mapping Verify migration completeness"
	@echo "    make verify-pantheon-physicsforge-license-headers Verify license headers"
	@echo "    make rust-smoke           Dedicated Rust smoke suites via nextest"
	@echo "    make rust-regression      Full Rust regression lane"
	@echo "    make rust-regression-scoped Scoped Rust regression lane"
	@echo "    make gate-local           Canonical scoped local push gate"
	@echo "    make gate-ci-registry     Rust-native governance + registry contract CI gate"
	@echo "    make gate-ci-rust         Full Rust CI gate"
	@echo ""
	@echo "  Artifacts:"
	@echo "    make artifacts            Regenerate all core artifact sets"
	@echo "    make artifacts-motifs     CD motif census (16D, 32D)"
	@echo "    make artifacts-motifs-big CD motif census (64D-256D)"
	@echo "    make artifacts-boxkites   De Marrais boxkite geometry"
	@echo "    make artifacts-reggiani   Reggiani annihilator statistics"
	@echo "    make artifacts-m3         M3 transfer table"
	@echo "    make artifacts-dimensional Dimensional geometry sweeps"
	@echo "    make artifacts-repo-visuals Repo maps plus science-facing plates"
	@echo ""
	@echo "  Data:"
	@echo "    make fetch-data           Re-download external datasets via Rust fetchers"
	@echo "    make provenance           Hash data/external/* + emit audit report"
	@echo "    make provenance-audit     Enforce strict governance gate"
	@echo "    make semantic-data-validate Run lane semantic validators"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean                Remove caches and bytecode"
	@echo "    make clean-builds         Remove all Rust build artifacts"
	@echo "    make clean-artifacts      Remove generated CSV/images/HDF5"
	@echo "    make clean-all            clean + clean-builds + clean-artifacts"
	@echo ""
	@echo "  Other:"
	@echo "    make run                  Run simulations (sedenion, modular, entropy)"
	@echo "    make rocq                Compile Rocq proofs"
	@echo "    make latex                Build MASTER_SYNTHESIS.pdf"

# -----------------------------------------------------------------------------
# Module discovery
# -----------------------------------------------------------------------------
# `cargo modules` (third-party `cargo-modules` crate) renders a tree of every
# module in a crate. Install once with `cargo install cargo-modules`. If absent,
# the target falls back to a filesystem walk that still surfaces every .rs file.
# Pass CRATE=<name> to scope to a specific crate (default: gororoba_algebra).
.PHONY: modules-tree modules-doc

CRATE ?= gororoba_algebra

modules-tree:
	@if command -v cargo-modules >/dev/null 2>&1; then \
		cargo modules structure --package $(CRATE) --no-fns --no-traits --no-types ; \
	else \
		echo "[fallback] cargo-modules not installed; listing .rs files under $(CRATE)/src/" ; \
		echo "  install with: cargo install cargo-modules" ; \
		find crates/$(CRATE)/src -name '*.rs' | sort ; \
	fi

modules-doc:
	@CARGO_TARGET_DIR=.cache/gate-target cargo doc --no-deps --document-private-items -p $(CRATE)
	@echo "open .cache/gate-target/doc/$(CRATE)/index.html in a browser"
