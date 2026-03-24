# ---- Phony targets ----
.PHONY: help bootstrap-dev fmt fmt-check
.PHONY: test lint check smoke integrity integrity-rust math-verify governance-gate governance-gate-readonly wave6-gate pre-push-gate pre-push-gate-strict hooks-install hooks-install-strict hooks-status synthesis-execution-contract
.PHONY: verify verify-grand verify-c010-c011-theses ansi-check ansi-check-strict terminology-gate doctor doctor-blas provenance
.PHONY: provenance-registry-index provenance-registry-export provenance-registry-verify provenance-registry-doctor provenance-registry-link-audit provenance-registry-recover
.PHONY: rocq-proofs rocq-proofs-check lva-paper
.PHONY: heavy test-inventory verify-no-reports-writes
.PHONY: rust-test rust-clippy rust-semver-check rust-smoke rust-regression rust-regression-scoped dep-audit cargo-deny-check mcp-smoke e027-validate studio-run studio-check profile-tensor-avt x87-strategy-bench x87-strategy-perf x87-strategy-hyperfine x87-strategy-flamegraph x87-givens-microbench x87-givens-microbench-perf jacobi-backend-sweep jacobi-backend-perf jacobi-backend-flamegraph jacobi-backend-samply jacobi-backend-samply-compare gpu-bench gpu-bench-ncu gpu-bench-nsys
.PHONY: cpu-bench cpu-bench-perf cpu-bench-cachegrind cpu-bench-flamegraph parity-bench parity-report
.PHONY: pre-push-gate-scoped submodule-sync gate-local gate-ci-registry gate-ci-rust gate-audit
.PHONY: cache-status cache-sweep cache-purge-exp cache-check
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
.PHONY: fetch-data fetch-data-redownload provenance-audit external-redownload-audit semantic-data-validate semantic-data-validate-strict run rocq latex
.PHONY: docker-quantum-build docker-quantum-run docker-quantum-shell
.PHONY: clean clean-builds clean-artifacts clean-all host-profile
.PHONY: run-e183
.PHONY: cpd-audit cpd-audit-strict cargo-cache-status cargo-cache-prune cargo-cache-smoke

.NOTPARALLEL: bootstrap-dev check smoke integrity integrity-rust rust-smoke rust-regression rust-regression-scoped heavy cargo-deny-check gate-local gate-ci-registry gate-ci-rust gate-audit pre-push-gate pre-push-gate-scoped pre-push-gate-strict governance-gate governance-gate-readonly registry-control-plane-gate-readonly registry-acceptance-gate-readonly

# Non-cargo make fanout: 75% of logical CPUs, minimum 1.
# Cargo and Rust test runners use a shared worker budget equal to logical threads / 2.
NPROC := $(shell nproc 2>/dev/null || echo 4)
NJOBS := $(shell expr $(NPROC) \* 3 / 4)
WORKER_BUDGET ?= $(shell sh scripts/detect_worker_budget.sh)
CARGO_JOBS ?= $(WORKER_BUDGET)
NEXTEST_TEST_THREADS ?= $(WORKER_BUDGET)
RUST_TEST_THREADS ?= $(WORKER_BUDGET)
RAYON_THREADS ?= $(WORKER_BUDGET)
RUST_SCOPED_CLIPPY_TARGETS ?= --lib --tests
LOCAL_NEXTEST_TIMING_JSON ?=
RUST_LOCAL_SKIP_FILTERSET ?= not ((package(stats_core) and test(/ultrametric::baire_codebook::tests::(test_euclidean_ultrametricity_across_filtration_levels|test_intermediate_filtration_gradient|test_random_removal_control|test_lambda512_to_256_intermediate_gradient|test_lambda512_to_256_random_removal_control|test_sbase_to_lambda2048_gradient|test_l0_subpopulation_ultrametricity|test_lambda2048_to_1024_intermediate_gradient|test_l1_filter_on_l0_neg1_subset|test_recursive_simpsons_paradox_l2|test_cross_stratum_triple_decomposition|test_l0_zero_simpsons_paradox|test_dimensional_universality_simpsons_paradox|test_lambda1024_stratum_paradox_and_summary)/)) or (package(algebra_experimental) and test(test_thesis_e_xor_involution_invariants_128d)) or (package(gororoba_algebra) and test(test_split_octonion_attractor_regression_dim_128_256_guarded)) or (package(gororoba_cli) and test(test_zero_divisor_scaling)) or (package(sign_imbalance) and test(test_kubo_j1j2_alpha_sweep)) or test(/gpu/))
REPO_TMPDIR ?= $(or $(TMPDIR),/tmp)
REPO_PATH_HASH ?= $(shell printf "%s" "$(CURDIR)" | sha256sum | cut -c1-16)
REPO_TMP_CARGO_ROOT ?= $(REPO_TMPDIR)/open_gororoba-cargo-build/gate/$(REPO_PATH_HASH)
REPO_CARGO_HOME ?= $(CURDIR)/.cache/cargo-home
CARGO_CACHE_REPO_BUDGET_GIB ?= 8
CARGO_CACHE_TMP_BUDGET_GIB ?= 16
# Gate builds use a separate target dir from ambient (LSP/editor) builds to
# avoid file-lock contention during concurrent cargo check / nextest runs.
# Both dirs are bounded by `make cache-sweep` (cargo-sweep --maxsize).
# Experimental target dirs MUST follow the naming convention .cache/exp-<name>-target/
# Use `make cache-purge-exp` to remove all of them. Never create ad-hoc names.
REPO_CARGO_TARGET_DIR ?= $(CURDIR)/.cache/gate-target
# Build intermediates (.o/.d) go to /tmp via build-dir, keeping target-dir lean.
# /tmp on this machine is the same nvme partition (not tmpfs), so benefit is
# layout isolation only. See .cargo/config.toml [build] build-dir for details.
REPO_CARGO_BUILD_DIR ?= $(REPO_TMP_CARGO_ROOT)
CARGO_ENV = CARGO_HOME=$(REPO_CARGO_HOME) CARGO_TARGET_DIR=$(REPO_CARGO_TARGET_DIR) CARGO_BUILD_BUILD_DIR=$(REPO_CARGO_BUILD_DIR) MAKEFLAGS= MFLAGS= CARGO_MAKEFLAGS= CARGO_BUILD_JOBS=$(CARGO_JOBS) RAYON_NUM_THREADS=$(RAYON_THREADS) RUST_TEST_THREADS=$(RUST_TEST_THREADS)
# [env] CARGO_INCREMENTAL=0 in .cargo/config.toml now enforces this globally.
# Kept here as belt-and-suspenders for CI environments where the config may be absent.
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
MD_BOOK ?= mdbook
PGO_DIR ?= /tmp/pgo-data
SYNTHESIS_CONTRACT_DATE ?= 2026_02_14
SYNTHESIS_CONTRACT_REPORT ?= reports/synthesis_execution_contract_$(SYNTHESIS_CONTRACT_DATE).toml
PROFILE_TIMESTAMP := $(shell date +%Y-%m-%d/%H%M%S)
PROFILE_ROOT ?= reports/gates/profiles/$(PROFILE_TIMESTAMP)

# ---- Three-layer registry build ----
# Layer 1 (Source): TOML files in registry/ (human-edited, git-tracked).
# Layer 2 (Build):  .cache/registry.sqlite3 (derived, .gitignore'd).
# Layer 3 (Query):  gororoba-db CLI.

REGISTRY_SOURCES := $(wildcard registry/claims.toml registry/insights.toml \
    registry/experiments.toml registry/binaries.toml registry/project.toml \
    registry/external_sources.toml registry/bibliography.toml \
    registry/claims_evidence_edges.toml registry/experiment_lineage.toml \
    registry/lacunae.toml registry/roadmap.toml registry/todo.toml \
    registry/next_actions.toml registry/requirements.toml \
    registry/artifact_source_of_truth.toml registry/research_narratives.toml \
    registry/source_manifest.toml)

.cache/registry.sqlite3: $(REGISTRY_SOURCES)
	$(CARGO_ENV) cargo run --release -p gororoba_db --bin gororoba-db -- build --repo-root .

.PHONY: registry-build registry-build-verify
registry-build: .cache/registry.sqlite3

registry-build-verify: .cache/registry.sqlite3
	$(CARGO_ENV) cargo run --release -p gororoba_db --bin gororoba-db -- build --verify --repo-root .

# ---- Environment setup ----

bootstrap-dev:
	@echo "OK: Rust-first dev bootstrap is current."

# ---- Quality gates ----

lint: rust-clippy

# ---- Formatting (dprint) ----
# Unified formatting for Rust (.rs via rustfmt), TOML, JSON, and Markdown.
# Install: cargo install dprint
fmt:
	dprint fmt

fmt-check:
	dprint check

# ---- Parallelized lint gates ----
# Tier 1: lightweight checks (no cargo compilation, safe to parallelize).
# Tier 2: cargo-heavy checks (require compilation, serialize).
#
# gate-fast: Tier 1 only (<10s). Use for pre-push and rapid feedback.
# gate-deep: Tier 1 + Tier 2 (minutes). Use for CI and thorough audits.

.PHONY: gate-fast gate-warm gate-deep typos machete audit geiger

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

# gate-fast: parallel Tier 1 checks (no cargo compilation required).
# Runs dprint, typos, and machete concurrently. ~5s on warm cache.
# WHY: Catches formatting, spelling, and dead-dep issues before expensive compilation.
# NOTE: ansi-check and terminology-gate use `cargo run` so they belong in Tier 2
# on a cold cache but are fast (~0s) on a warm cache. They run in gate-warm.
gate-fast:
	@echo "=== gate-fast: parallel zero-compile checks ==="
	@fail=0; \
	dprint check || { echo "FAIL: dprint check"; fail=1; }; \
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

test: rust-regression

check: ansi-check terminology-gate verify-no-reports-writes
	@echo "OK: fast shared check suite complete."

# Governance verifier targets
registry-verify-markdown-governance:
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify markdown-removal-policy

governance-gate-readonly:
	@# release-gate profile: thin LTO + 6 codegen-units = 10x faster compile
	@# than fat LTO, <5% runtime difference for TOML-parsing gate binaries.
	$(CARGO_ENV) cargo build --profile release-gate -p gororoba_cli_data --bin markdown-registry --bin governance-verify --bin integrity-resolution
	$(REPO_CARGO_TARGET_DIR)/release-gate/markdown-registry verify-gate-all
	$(REPO_CARGO_TARGET_DIR)/release-gate/governance-verify gate-all
	@echo ""
	@echo "=========================================="
	@echo "READ-ONLY GOVERNANCE GATE: PASSED"
	@echo "=========================================="
	@echo "[done] Markdown inventory validated (TOML-first)"
	@echo "[done] Markdown owner map verified"
	@echo "[done] Registry schema signatures checked"
	@echo "[done] Cross-reference integrity verified"
	@echo "[done] Dataset label aliases verified"
	@echo "[done] External-source operational contracts verified"
	@echo "[done] Markdown governance removal policy checked"
	@echo ""
	@echo "TOML-first governance checks are operational."
	@echo "=========================================="

governance-gate: governance-gate-readonly
	@echo "OK: governance-gate is a compatibility alias for governance-gate-readonly."

wave6-gate: governance-gate
	@echo "DEPRECATED: make wave6-gate is a legacy alias. Use make governance-gate."

gate-local: cache-check
	@set -e; \
	scope=""; \
	run_rust="true"; \
	run_governance="true"; \
	eval "$$(cargo run -q -p xtask -- host-profile --format shell)"; \
	submake_env="WORKER_BUDGET=$$HOST_WORKER_BUDGET CARGO_JOBS=$$HOST_CARGO_JOBS NEXTEST_TEST_THREADS=$$HOST_NEXTEST_TEST_THREADS RUST_TEST_THREADS=$$HOST_RUST_TEST_THREADS RAYON_THREADS=$$HOST_RAYON_THREADS"; \
	echo "[gate-local] host profile: physical_cores=$$HOST_PHYSICAL_CORES core_ids=$$HOST_PHYSICAL_CORE_IDS l3_cache_bytes=$$HOST_L3_CACHE_BYTES l3_safe_bytes=$$HOST_L3_SAFE_WORKING_SET_BYTES worker_budget=$$HOST_WORKER_BUDGET"; \
	echo "[gate-local] determining scope..."; \
	if command -v cargo >/dev/null 2>&1; then \
	    scope_file="$$(mktemp)"; \
	    meta_file="$$(mktemp)"; \
	    CARGO_HOME=$(REPO_CARGO_HOME) CARGO_TARGET_DIR=$(REPO_CARGO_TARGET_DIR) MAKEFLAGS= MFLAGS= CARGO_MAKEFLAGS= CARGO_BUILD_JOBS=$$HOST_CARGO_JOBS RAYON_NUM_THREADS=$$HOST_RAYON_THREADS RUST_TEST_THREADS=$$HOST_RUST_TEST_THREADS cargo run -q -p gororoba_cli_governance --bin workspace-routing -- --local --verbose 1>"$$scope_file" 2>"$$meta_file" || true; \
	    scope="$$(cat "$$scope_file" 2>/dev/null || true)"; \
	    routing_meta="$$(cat "$$meta_file" 2>/dev/null || true)"; \
	    rm -f "$$scope_file" "$$meta_file"; \
	    if [ -n "$$routing_meta" ]; then printf '%s\n' "$$routing_meta"; fi; \
	    printf '%s\n' "$$routing_meta" | grep -q 'run_rust=False' && run_rust="false" || true; \
	    printf '%s\n' "$$routing_meta" | grep -q 'run_governance=False' && run_governance="false" || true; \
	else \
	    echo "[gate-local] WARNING: workspace-routing unavailable, running full workspace"; \
	    scope="--workspace"; \
	fi; \
	$(MAKE) check $$submake_env; \
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

# ---- Cargo cache management -----------------------------------------------
# WHY: Two independent target dirs (.cache/gate-target and
# .cache/cargo-default-target) balloon without bounds because Cargo never
# auto-evicts build artifacts. cargo-sweep enforces size limits.
# cargo clean gc (enabled by [unstable] gc=true) handles CARGO_HOME only.
#
# Experimental target dirs: MUST be named .cache/exp-<name>-target/
# Use: CARGO_TARGET_DIR=$(CURDIR)/.cache/exp-myname-target cargo ...
# Clean: make cache-purge-exp
.PHONY: cache-status cache-sweep cache-purge-exp cache-check

cache-status:
	@printf '=== Cargo target dirs ===\n'
	@du -sh .cache/gate-target .cache/cargo-default-target 2>/dev/null || true
	@printf '=== CARGO_HOME ===\n'
	@du -sh .cache/cargo-home 2>/dev/null || true
	@printf '=== /tmp build-dir intermediates ===\n'
	@du -sh /tmp/open_gororoba-cargo-build 2>/dev/null || printf '(empty)\n'
	@printf '=== Experimental dirs (.cache/exp-*-target) ===\n'
	@du -sh .cache/exp-*-target 2>/dev/null || printf '(none)\n'

cache-sweep:
	@echo "Sweeping gate-target to <= 100GB..."
	cargo sweep --maxsize 100GB .cache/gate-target
	@echo "Sweeping cargo-default-target to <= 100GB..."
	cargo sweep --maxsize 100GB .cache/cargo-default-target
	@echo "OK: cache-sweep complete."

cache-purge-exp:
	rm -rf .cache/exp-*-target
	@echo "OK: experimental target dirs purged."

# Fast cache size check: warns but does not fail. Integrated into gate-local.
cache-check:
	@GATE_MB=$$(du -sm .cache/gate-target 2>/dev/null | cut -f1 || printf '0'); \
	AMBIENT_MB=$$(du -sm .cache/cargo-default-target 2>/dev/null | cut -f1 || printf '0'); \
	TOTAL=$$((GATE_MB + AMBIENT_MB)); \
	if [ "$$TOTAL" -gt 102400 ]; then \
		printf '[cache-check] WARNING: cargo target dirs total %dGB (>100GB). Run: make cache-sweep\n' "$$((TOTAL / 1024))"; \
	else \
		printf '[cache-check] OK: cargo target dirs at %dMB\n' "$$TOTAL"; \
	fi

profile-python-toml-inventory:
	@mkdir -p "$(PROFILE_ROOT)"
	@if command -v /usr/bin/time >/dev/null 2>&1; then \
		echo "[profile] timing Rust TOML inventory builder"; \
		/usr/bin/time -v -o "$(PROFILE_ROOT)/toml_inventory.time.txt" $(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-toml-inventory; \
	else \
		echo "[profile] running Rust TOML inventory builder without /usr/bin/time"; \
		$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-toml-inventory; \
	fi
	@echo "OK: Rust TOML inventory profile written under $(PROFILE_ROOT)"

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
	$(CARGO_ENV) cargo run -p gororoba_cli_data --bin claims-verify -- --check providers
	$(MAKE) test-inventory
	$(CARGO_ENV) cargo run -p gororoba_cli_data --bin registry-check -- --typed-policy error
	@echo "OK: Rust integrity lane passed."

test-inventory:
	$(CARGO_ENV) cargo run -p gororoba_cli_data --bin test-inventory -- --check

math-verify: rust-regression
	@echo "OK: math validation suite complete. See docs/MATH_VALIDATION_REPORT.md"

rust-test: rust-regression
	@echo "OK: rust-test is an alias for rust-regression."

rust-clippy:
	$(CARGO_ENV) cargo clippy --workspace -- -D warnings

rust-semver-check:
	@echo "[semver-check] Checking public API SemVer compliance..."
	$(CARGO_ENV) cargo semver-checks check-release --workspace \
		--exclude gororoba_cli \
		--exclude gororoba_cli_algebra \
		--exclude gororoba_cli_data \
		--exclude gororoba_cli_governance \
		--exclude gororoba_cli_physics \
		--exclude gororoba_cli_provenance \
		--exclude gororoba_cli_quantum \
		--exclude gororoba_cli_warp \
		--exclude gororoba_db
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
	$(eval RUST_SCOPE ?= $(shell $(CARGO_ENV) cargo run -q -p gororoba_cli_governance --bin workspace-routing -- --local 2>/dev/null || echo "--workspace"))
	$(eval RUST_RUN_HEAVY ?= 1)
	@set -e; \
	if [ -z "$(RUST_SCOPE)" ]; then \
	    echo "SKIP: no Rust-relevant changes detected."; \
	else \
	    echo "[rust-regression-scoped] scope: $(RUST_SCOPE)"; \
	    $(CARGO_ENV) cargo clippy $(RUST_SCOPE) $(RUST_SCOPED_CLIPPY_TARGETS) -- -D warnings; \
	    local_light_scope=""; \
	    local_light_packages=""; \
	    if [ "$(RUST_SCOPE)" = "--workspace" ]; then \
	        light_scope="--workspace --exclude algebra_analysis --exclude gr_core"; \
	        heavy_scope="-p algebra_analysis -p gr_core"; \
	        local_light_scope="$$light_scope"; \
	    else \
	        light_scope=""; \
	        heavy_scope=""; \
	        prev=""; \
	        for token in $(RUST_SCOPE); do \
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
	    if [ "$(RUST_SCOPE)" = "--workspace" ]; then \
	        if [ -n "$$filterset" ]; then \
	            echo "[rust-regression-scoped] local skip filter enabled"; \
	            $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $$local_light_scope -E "$$filterset"; \
	        else \
	            $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $$local_light_scope; \
	        fi; \
	    elif [ -n "$$local_light_packages" ]; then \
	        if [ -n "$$filterset" ]; then \
	            echo "[rust-regression-scoped] local skip filter enabled"; \
	            $(CARGO_ENV) cargo run -q -p xtask -- local-nextest-plan --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $(if $(LOCAL_NEXTEST_TIMING_JSON),--timing-json-out $(LOCAL_NEXTEST_TIMING_JSON),) --filterset "$$filterset" $$local_light_packages; \
	        else \
	            $(CARGO_ENV) cargo run -q -p xtask -- local-nextest-plan --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $(if $(LOCAL_NEXTEST_TIMING_JSON),--timing-json-out $(LOCAL_NEXTEST_TIMING_JSON),) $$local_light_packages; \
	        fi; \
	    fi; \
	    if [ -n "$$heavy_scope" ] && [ "$(RUST_RUN_HEAVY)" = "1" ]; then \
	        $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy $$heavy_scope; \
	    elif [ -n "$$heavy_scope" ]; then \
	        echo "[rust-regression-scoped] SKIP heavy nextest in local fast path: $$heavy_scope"; \
	    fi; \
	    echo "OK: Rust regression gate passed (scoped: clippy + nextest)."; \
	fi

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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- mcp-smoke

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
	CARGO_TARGET_DIR=/tmp/open_gororoba_parity_target $(CARGO_ENV) cargo test --workspace
	CARGO_TARGET_DIR=/tmp/open_gororoba_parity_target $(CARGO_ENV) cargo clippy --workspace -- -D warnings
	@echo "OK: parity lane passed (workspace tests + clippy with release-class optimization semantics)."

rust-release-fat-lto:
	CARGO_TARGET_DIR=/tmp/open_gororoba_release_target $(CARGO_ENV) cargo build --release --workspace
	@echo "OK: release fat-LTO workspace build completed."

rust-pgo-instrument:
	mkdir -p "$(PGO_DIR)"
	CARGO_TARGET_DIR=/tmp/open_gororoba_pgo_target \
	$(CARGO_ENV) \
	RUSTFLAGS="-Cprofile-generate=$(PGO_DIR)" \
	cargo build --release --workspace
	@echo "OK: PGO instrumented build completed. Run representative binaries to collect .profraw files in $(PGO_DIR)."

rust-pgo-merge:
	llvm-profdata merge -o "$(PGO_DIR)/merged.profdata" "$(PGO_DIR)"/*.profraw
	@echo "OK: merged profile written to $(PGO_DIR)/merged.profdata."

rust-pgo-build:
	CARGO_TARGET_DIR=/tmp/open_gororoba_pgo_use_target \
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
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- promote-research-narratives

registry-bootstrap-research-narratives: registry-normalize-research-narratives
	@echo "Research narratives markdown->TOML bootstrap completed."

registry-normalize-book-docs:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-book-docs --bootstrap-from-markdown

registry-bootstrap-book-docs: registry-normalize-book-docs
	@echo "mdBook markdown->TOML bootstrap completed."

registry-normalize-docs-root-narratives:
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- promote-docs-root-narratives

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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-inventory

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
	@echo "OK: markdown TOML-first owner/inventory gates verified."

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

registry-export-markdown: registry-refresh
	@legacy_claims_sync=1; \
	if [ "$(MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC)" = "0" ]; then legacy_claims_sync=0; fi; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- insights-mirror \
		--output "crates/data_core/src/registry_mirrors/insights_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-mirror \
		--output "crates/data_core/src/registry_mirrors/claims_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- bibliography-mirror \
		--output "crates/data_core/src/registry_mirrors/bibliography_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- experiments-mirror \
		--output "crates/data_core/src/registry_mirrors/experiments_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- theorems-mirror \
		--output "crates/data_core/src/registry_mirrors/theorems_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- roadmap-mirror \
		--output "crates/data_core/src/registry_mirrors/roadmap_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- todo-mirror \
		--output "crates/data_core/src/registry_mirrors/todo_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- next-actions-mirror \
		--output "crates/data_core/src/registry_mirrors/next_actions_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- navigator-mirror \
		--output "crates/data_core/src/registry_mirrors/navigator_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- entrypoint-docs-mirror \
		--output "crates/data_core/src/registry_mirrors/entrypoint_docs_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- requirements-mirror \
		--output "crates/data_core/src/registry_mirrors/requirements_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- knowledge-migration-plan-mirror \
		--output "crates/data_core/src/registry_mirrors/knowledge_migration_plan_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- markdown-governance-mirror \
		--output "crates/data_core/src/registry_mirrors/markdown_governance_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-tasks-mirror \
		--output "crates/data_core/src/registry_mirrors/claims_tasks_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-domains-mirror \
		--output "crates/data_core/src/registry_mirrors/claims_domains_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claim-tickets-mirror \
		--output "crates/data_core/src/registry_mirrors/claim_tickets_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- external-sources-mirror \
		--output "crates/data_core/src/registry_mirrors/external_sources_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- book-docs-mirror \
		--output "crates/data_core/src/registry_mirrors/book_docs_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- data-artifact-narratives-mirror \
		--output "crates/data_core/src/registry_mirrors/data_artifact_narratives_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- reports-narratives-mirror \
		--output "crates/data_core/src/registry_mirrors/reports_narratives_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- docs-convos-mirror \
		--output "crates/data_core/src/registry_mirrors/docs_convos_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- docs-root-narratives-mirror \
		--output "crates/data_core/src/registry_mirrors/docs_root_narratives_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- research-narratives-mirror \
		--output "crates/data_core/src/registry_mirrors/research_narratives_registry_mirror.rs"; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-inventory; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-corpus; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-origin-audit; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-owner-map; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-payloads

registry-verify-mirrors:
	legacy_flag=""; \
	claims_value="true"; \
	if [ "$(MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC)" = "0" ]; then claims_value="false"; fi; \
	cargo run -q -p gororoba_cli_data --bin verify-registry-mirror-freshness -- \
		--out-dir "$(MARKDOWN_EXPORT_OUT_DIR)" $$legacy_flag --legacy-claims-sync $$claims_value
	$(MAKE) registry-verify-markdown-toml-first
	cargo run -q -p gororoba_cli_data --bin governance-verify -- markdown-headers; \
	cargo run -q -p gororoba_cli_data --bin governance-verify -- markdown-parity; \
	cargo run -q -p gororoba_cli_data --bin governance-verify -- mirror-immutability; \
	cargo run -q -p gororoba_cli_data --bin governance-verify -- claim-ticket-mirrors;

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

docs-site: docs-rustdoc docs-book
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
	@touch "$(DOCS_SITE_DIR)/.nojekyll"
	@echo "OK: docs site staged to $(DOCS_SITE_DIR)."

docs-freshness: docs-gate docs-redirect-check
	@echo "OK: docs-freshness checks passed."

docs-gate: docs-site
	@echo "OK: docs-gate generated unified docs bundle."

docs-redirect-check:
	./scripts/docs-redirect-check.sh $(DOCS_SITE_DIR)

terminology-gate:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- terminology-gate

ansi-check:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- ansi-check --check

ansi-check-strict:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- ansi-check --check --strict-placeholders --placeholder-scope-prefix crates/ --placeholder-scope-prefix tests/

verify:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- verify-artifacts

verify-grand:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- verify-grand-images

verify-c010-c011-theses:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- verify-c010-c011-theses

verify-python-core-algorithms:
	@echo "SKIP: legacy Python core algorithms verification removed."

doctor:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- doctor
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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-semantic-validate -- --fail-on-unverifiable true --out reports/data_semantic_validate_$$(date +%F)_strict.toml

patch-pyfilesystem2:
	@echo "SKIP: patch-pyfilesystem2 removed (no Python runtime dependency)."

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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- rocq-prepare-confine curated/01_theory_frameworks/confine_theorems_512.v curated/01_theory_frameworks/confine_theorems_512_axioms.v
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- rocq-prepare-confine curated/01_theory_frameworks/confine_theorems_1024.v curated/01_theory_frameworks/confine_theorems_1024_axioms.v
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- rocq-prepare-confine curated/01_theory_frameworks/confine_theorems_2048.v curated/01_theory_frameworks/confine_theorems_2048_axioms.v
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
	rm -rf .pytest_cache .ruff_cache
	rm -rf src/*.egg-info
	rm -rf $(REPO_CARGO_TARGET_DIR)

clean-builds:
	rm -rf target/
	rm -rf .cache/cargo-default-target/
	rm -rf .cache/gate-target/
	rm -rf $(REPO_TMP_CARGO_ROOT)
	rm -rf /tmp/open_gororoba-cargo-build 2>/dev/null || true
	rm -rf /tmp/open_gororoba_*_target 2>/dev/null || true
	rm -rf /tmp/open_gororoba-cargo-build-* 2>/dev/null || true
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

# Data-heavy source files excluded from CPD scans until materials_data codegen migration
# is complete (tasks #57-#58). These files are pure const-array data, not logic.
# Remove entries here as each file is migrated to build.rs + CSV/TOML.
# tabulated_nk.rs removed: migrated to materials_data in task #56.
CPD_EXCLUDE_FILES := \
	crates/materials_core/src/optical_database.rs \
	crates/materials_core/src/crystal_symmetry.rs

# Build a file-list of all .rs sources under crates/ minus the excluded data files.
_CPD_FILE_LIST := /tmp/cpd_src_list.txt
_CPD_REGEN_LIST = find crates -name '*.rs' \
	$(foreach f,$(CPD_EXCLUDE_FILES), ! -path './$(f)') \
	> $(_CPD_FILE_LIST)

cpd-audit:
	@command -v pmd >/dev/null 2>&1 || { echo "ERROR: pmd not found. Install PMD (e.g. paru -S pmd) to run cpd-audit."; exit 1; }
	@$(_CPD_REGEN_LIST)
	pmd cpd --language rust --minimum-tokens $(CPD_MIN_TOKENS) --file-list $(_CPD_FILE_LIST) --format xml 2>/dev/null \
		| python3 scripts/cpd_report.py --top $(CPD_TOP)

cpd-audit-strict:
	@command -v pmd >/dev/null 2>&1 || { echo "ERROR: pmd not found. Install PMD (e.g. paru -S pmd) to run cpd-audit-strict."; exit 1; }
	@$(_CPD_REGEN_LIST)
	pmd cpd --language rust --minimum-tokens $(CPD_MIN_TOKENS) --file-list $(_CPD_FILE_LIST) --format xml 2>/dev/null \
		| python3 scripts/cpd_report.py --strict --top $(CPD_TOP)

# ---- Help ----

help:
	@echo "Targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make bootstrap-dev        Ensure the dev environment is current"
	@echo ""
	@echo "  Quality:"
	@echo "    make cpd-audit            Report cross/within-crate Rust duplication (CPD, 42 tokens)"
	@echo "    make cpd-audit-strict     Same, exits 1 if any clusters found"
	@echo "    make lint                 Run workspace-wide clippy -- -D warnings"
	@echo "    make test                 Run workspace-wide nextest"
	@echo "    make smoke                Composite fast smoke lane (check + rust-smoke)"
	@echo "    make integrity-rust       Cargo-backed integrity lane (claims + inventory + typed policy)"
	@echo "    make check                Fast local check (ansi + terminology + no-reports)"
	@echo "    make ansi-check           Verify ANSI-safe UTF-8 character policy"
	@echo "    make ansi-check-strict    Verify ANSI-safe UTF-8 policy + fail on <U+....> placeholders"
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
