# ---- Phony targets ----
.DEFAULT_GOAL := help
.PHONY: help bootstrap-dev bootstrap-user-local-xdg fmt fmt-check
.PHONY: test lint check check-local smoke integrity integrity-rust math-verify governance-gate governance-gate-readonly wave6-gate pre-push-gate pre-push-gate-strict synthesis-execution-contract
.PHONY: verify verify-grand verify-c010-c011-theses ansi-check ansi-check-strict terminology-gate doctor doctor-blas provenance cuda-source-ownership
.PHONY: provenance-registry-index provenance-registry-export provenance-registry-verify provenance-registry-doctor provenance-registry-link-audit provenance-registry-recover
.PHONY: rocq-proofs rocq-proofs-check rocq-project-check rocq-makefile-check lva-paper
.PHONY: heavy test-inventory
.PHONY: rust-test rust-clippy rust-semver-check rust-smoke rust-regression rust-regression-scoped miri-cd-kernel dep-audit cargo-deny-check mcp-smoke e027-validate studio-run studio-check profile-tensor-avt x87-strategy-bench x87-strategy-perf x87-strategy-hyperfine x87-strategy-flamegraph x87-givens-microbench x87-givens-microbench-perf jacobi-backend-sweep jacobi-backend-perf jacobi-backend-flamegraph jacobi-backend-samply jacobi-backend-samply-compare gpu-bench gpu-bench-ncu gpu-bench-nsys
.PHONY: cpu-bench cpu-bench-perf cpu-bench-cachegrind cpu-bench-flamegraph parity-bench parity-report
.PHONY: pre-push-gate-scoped submodule-sync validate-local validate-local-xtask validate-ci validate-ci-registry validate-ci-rust validate-repository validate-repository-fast validate-governance validation-tools registry-validation-tools validation-tools-clean validation-tools-rebuild validation-tools-check-paths validation-resource-contract validation-resource-contract-authority validation-resource-contract-retired-local validation-resource-contract-registry validation-resource-contract-workers validation-resource-contract-collectors print-validation-resource-config require-ci-validation-authority casimir-optics-discrimination-audit-check data-core-pure-check
.PHONY: gate-local gate-local-xtask gate-ci-registry gate-ci-rust gate-audit gate-audit-fast
.PHONY: cache-status cache-sweep cache-sweep-soft cache-purge-exp cache-check cache-check-force
.PHONY: v6-branch-transport-artifacts pathion-control-artifacts pathion-resonance-artifacts
.PHONY: registry-control-plane-gate-readonly registry-acceptance-gate-readonly validate-registry validate-registry-integrity validate-rust-integrity
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
.PHONY: experiment-manifest-verify
.PHONY: registry-markdown-inventory registry-markdown-corpus registry-toml-inventory
.PHONY: registry-markdown-origin-audit
.PHONY: registry-knowledge-atoms registry-verify-knowledge-atoms
.PHONY: registry-artifact-scrolls registry-verify-artifact-scrolls
.PHONY: registry-verify-markdown-inventory registry-verify-markdown-origin registry-verify-markdown-owner registry-verify-control-plane registry-control-plane-gate registry-verify-wave4 registry-wave4
.PHONY: registry-verify-markdown-toml-first
.PHONY: registry-embedded-markdown registry-verify-embedded-markdown
.PHONY: registry-build-semantic-atoms registry-verify-semantic-atoms registry-semantic-atoms-gate
.PHONY: registry-build-evidence-provenance registry-verify-evidence-provenance registry-evidence-provenance-gate
.PHONY: registry-integrity integrity-resolution registry-build-integrity-resolution registry-verify-integrity-resolution registry-integrity-resolution-gate
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
.PHONY: check-ansi check-terminology
.PHONY: validate-rust-integrity-claims validate-rust-integrity-test-inventory validate-rust-integrity-typed-policy
.PHONY: validate-registry-control-plane validate-registry-project-counter validate-registry-markdown validate-registry-governance validate-registry-semantic-atoms validate-registry-evidence-provenance validate-registry-execution-planning
.PHONY: run-e183
.PHONY: cpd-audit cpd-audit-strict cpd-audit-tooling cpd-audit-generated patch-static-mirror-headers cargo-cache-status cargo-cache-prune cargo-cache-smoke
.PHONY: cd-row-upgrade-batch cd-row-upgrade-jacobson cd-row-upgrade-freudenthal

.NOTPARALLEL: bootstrap-dev check smoke integrity integrity-rust validate-rust-integrity rust-smoke rust-regression rust-regression-scoped heavy cargo-deny-check validate-local validate-ci validate-ci-registry validate-ci-rust validate-repository validate-repository-fast validate-supply-chain validate-dataset-experiments pre-push-gate pre-push-gate-scoped pre-push-gate-strict governance-gate governance-gate-readonly registry-control-plane-gate-readonly registry-acceptance-gate-readonly validate-registry validation-tools

# GitHub Actions exports every CPU visible to the validation process before
# invoking Cargo workloads. Local repository-validation targets remain closed.
TRUSTED_GITHUB_ACTIONS := $(if $(and $(filter true,$(CI)),$(filter true,$(GITHUB_ACTIONS))),1,0)
CI_WORKER_BUDGET ?= 1
override WORKER_BUDGET := $(if $(filter 1,$(TRUSTED_GITHUB_ACTIONS)),$(CI_WORKER_BUDGET),1)
override CARGO_JOBS := $(WORKER_BUDGET)
override NEXTEST_TEST_THREADS := $(WORKER_BUDGET)
override RUST_TEST_THREADS := $(WORKER_BUDGET)
override RAYON_THREADS := $(WORKER_BUDGET)
# Parse-time refusal prevents Make from constructing or updating a validation
# prerequisite locally. Mutation generators such as registry-integrity remain
# available because they produce reviewed source artifacts rather than a gate
# verdict.
RETIRED_LOCAL_VALIDATION_GOALS := check-local validate-local validate-local-xtask \
                                  gate-local gate-local-xtask pre-push-gate \
                                  pre-push-gate-scoped rust-regression-scoped
REQUESTED_RETIRED_LOCAL_VALIDATION_GOALS := $(filter $(RETIRED_LOCAL_VALIDATION_GOALS),$(MAKECMDGOALS))
ifneq ($(strip $(REQUESTED_RETIRED_LOCAL_VALIDATION_GOALS)),)
$(error retired local validation target(s) $(REQUESTED_RETIRED_LOCAL_VALIDATION_GOALS); push the branch to trigger GitHub Actions)
endif
CI_ONLY_VALIDATION_GOALS := test lint check smoke integrity integrity-rust \
                           math-verify fmt-check governance-gate governance-gate-readonly \
	                           wave6-gate ndlb-gate pre-push-gate-strict validate-ci \
	                           validate-ci-registry validate-ci-rust validate-ci-scoped-rust \
	                           validate-ci-scoped-clippy validate-ci-scoped-light \
	                           validate-ci-scoped-heavy validate-ci-scoped-rust-lane \
                           validate-repository validate-repository-fast \
                           validate-governance validate-rust-integrity validate-registry \
                           validate-registry-integrity validation-tools \
                           registry-validation-tools validation-tools-rebuild \
                           validation-tools-check-paths validation-resource-contract \
                           validate-static validate-static-and-registry \
                           validate-comprehensive validate-supply-chain \
                           validate-dataset-experiments rust-test rust-clippy \
                           rust-smoke rust-regression ansi-check ansi-check-strict \
                           terminology-gate casimir-optics-discrimination-audit-check \
                           supply-chain-gate gate-fast gate-warm gate-deep \
                           gate-ci-registry gate-ci-rust gate-audit gate-audit-fast \
                           audit audit-comprehensive audit-comprehensive-structured \
                           audit-deep audit-deep-structured dep-audit cargo-deny-check \
                           cpd-audit cpd-audit-strict cpd-audit-tooling cpd-audit-generated \
                           repo-audit repo-audit-strict repo-audit-strict-unjustified \
                           ref-audit ref-audit-strict docs-gate docs-freshness \
                           docs-redirect-check rocq-proofs-check rocq-project-check \
                           rocq-makefile-check registry-control-plane-gate \
                           registry-control-plane-gate-readonly registry-semantic-atoms-gate \
                           registry-evidence-provenance-gate \
                           registry-execution-planning-gate \
                           registry-integrity-resolution-gate registry-acceptance-gate \
                           registry-acceptance-gate-readonly registry-csv-pipeline-gate \
                           provenance-audit provenance-registry-link-audit \
                           external-redownload-audit semantic-data-validate \
                           semantic-data-validate-strict data-core-pure-check \
                           db-schema-drift-check test-inventory mcp-smoke e027-validate \
                           studio-check cargo-cache-smoke
REQUESTED_CI_ONLY_VALIDATION_GOALS := $(filter $(CI_ONLY_VALIDATION_GOALS),$(MAKECMDGOALS))
ifneq ($(strip $(REQUESTED_CI_ONLY_VALIDATION_GOALS)),)
ifneq ($(TRUSTED_GITHUB_ACTIONS),1)
$(error repository validation target(s) $(REQUESTED_CI_ONLY_VALIDATION_GOALS) run only in GitHub Actions; push the branch to trigger CI)
endif
endif
REPO_TMPDIR ?= $(or $(TMPDIR),/tmp)
# Every tool resolves the repository through repo_root::resolve!(), which
# reads this variable first; the compile-time manifest path inside a binary
# may belong to another worktree under a shared build-dir.
export GOROROBA_REPO_ROOT := $(CURDIR)
# Cache roots, the worktree-sharing knob, the validation-tools state
# paths and the guarded removal helper live in mk/cache_roots.mk so the
# xtask layout tests can load them without the rest of this file.
# Gate builds use a separate target dir from ambient (LSP/editor) builds to
# avoid file-lock contention during concurrent cargo check / nextest runs.
# Experimental target dirs MUST follow the naming convention .cache/exp-<name>-target/
# Use `make cache-purge-exp` to remove all of them. Never create ad-hoc names.
# Build intermediates (.o/.d) go to .cache/gate-cbuild/ on disk, keeping
# target-dir lean and avoiding /tmp (16 GB tmpfs) overflow. The 46-crate
# debug test compilation generates ~13 GB of split-debuginfo artifacts --
# more than the tmpfs budget. REPO_TMP_CARGO_ROOT still routes to /tmp for
# doc builds and parity tests where the artifact footprint is smaller.
include mk/cache_roots.mk
CARGO_CACHE_REPO_BUDGET_GIB ?= 150
CARGO_CACHE_TMP_BUDGET_GIB ?= 16
CARGO_ENV = CARGO_HOME=$(REPO_CARGO_HOME) CARGO_TARGET_DIR=$(REPO_CARGO_TARGET_DIR) CARGO_BUILD_BUILD_DIR=$(REPO_CARGO_BUILD_DIR) MAKEFLAGS= MFLAGS= CARGO_MAKEFLAGS= CARGO_BUILD_JOBS=$(CARGO_JOBS) RAYON_NUM_THREADS=$(RAYON_THREADS) RUST_TEST_THREADS=$(RUST_TEST_THREADS)
# A user-local Cargo config may enforce CARGO_INCREMENTAL=0 globally.
# Kept here as belt-and-suspenders for CI environments where that config is absent.
CARGO_ENV_CI = $(CARGO_ENV) CARGO_INCREMENTAL=0

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
# Hosted documentation uses default features; SDK-equipped hosts can opt in.
DOCS_FEATURE_FLAGS ?=
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
PROFILE_ROOT ?= reports/validation/profiles/$(PROFILE_TIMESTAMP)

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
    registry/external_sources.toml data/external/SOURCES.toml \
    registry/bibliography.toml \
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

# ---- Validation and audit commands ----
# Repository validation runs only in GitHub Actions. Developers may invoke a
# focused Cargo command directly, but no Make validation target starts a local
# compiler or test closure.

require-ci-validation-authority:
	@if [ "$(TRUSTED_GITHUB_ACTIONS)" != "1" ]; then \
	    echo "ERROR: repository validation runs only in GitHub Actions." >&2; \
	    echo "Push the branch to trigger CI. No local override is supported." >&2; \
	    exit 2; \
	fi

print-validation-resource-config:
	@printf 'workers=%s trusted_github_actions=%s\n' "$(WORKER_BUDGET)" "$(TRUSTED_GITHUB_ACTIONS)"

validation-resource-contract: validation-resource-contract-authority \
                              validation-resource-contract-retired-local \
                              validation-resource-contract-registry \
                              validation-resource-contract-workers \
                              validation-resource-contract-collectors
	@echo "OK: CI-only validation authority and worker boundaries are pinned."

validation-resource-contract-authority:
	@status=0; \
	local_config="$$( $(MAKE) --no-print-directory -s print-validation-resource-config CI=false GITHUB_ACTIONS=false CI_WORKER_BUDGET=99 WORKER_BUDGET=99 CARGO_JOBS=99)" || status=1; \
	if [ "$$local_config" != 'workers=1 trusted_github_actions=0' ]; then echo "ERROR: local worker configuration is not closed." >&2; status=1; fi; \
	ci_config="$$( $(MAKE) --no-print-directory -s print-validation-resource-config CI=true GITHUB_ACTIONS=true CI_WORKER_BUDGET=4 WORKER_BUDGET=99 CARGO_JOBS=99)" || status=1; \
	if [ "$$ci_config" != 'workers=4 trusted_github_actions=1' ]; then echo "ERROR: hosted worker configuration does not preserve the supplied count." >&2; status=1; fi; \
	if $(MAKE) --no-print-directory -s require-ci-validation-authority CI=true GITHUB_ACTIONS=false >/dev/null 2>&1; then echo "ERROR: generic CI variable bypassed broad-validation authority." >&2; status=1; fi; \
	if $(MAKE) --no-print-directory -s check CI=false GITHUB_ACTIONS=true >/dev/null 2>&1; then echo "ERROR: GitHub Actions marker bypassed CI-only validation authority." >&2; status=1; fi; \
	$(MAKE) --no-print-directory -s require-ci-validation-authority CI=true GITHUB_ACTIONS=true || status=1; \
	for contract in 'rust-clippy: require-ci-validation-authority' 'rust-regression: require-ci-validation-authority rust-clippy' 'check: require-ci-validation-authority'; do \
	    if ! grep -Fq "$$contract" Makefile; then echo "ERROR: missing validation authority contract: $$contract" >&2; status=1; fi; \
	done; \
	exit "$$status"

validation-resource-contract-retired-local:
	@status=0; \
	if $(MAKE) --no-print-directory -s validate-local >/dev/null 2>&1; then echo "ERROR: retired validate-local target unexpectedly succeeded." >&2; status=1; fi; \
	if ! grep -Fq 'rust-regression-scoped' Makefile; then echo "ERROR: scoped Rust CI entrypoint is missing." >&2; status=1; fi; \
	if grep -Fq 'local-nextest-plan' xtask/src/main.rs crates/gororoba_cli_data/Cargo.toml; then echo "ERROR: retired local nextest executor remains registered." >&2; status=1; fi; \
	for retired_path in crates/gororoba_cli_data/src/bin/local_nextest_plan.rs crates/gororoba_cli/src/bin/pre_push_hook.rs .githooks/pre-push scripts/detect_worker_budget.sh scripts/detect_physical_cores.sh; do \
	    if [ -e "$$retired_path" ]; then echo "ERROR: retired local validation path remains: $$retired_path" >&2; status=1; fi; \
	done; \
	if grep -Fq 'pre-push-hook' crates/gororoba_cli/Cargo.toml; then echo "ERROR: retired local pre-push executor remains registered." >&2; status=1; fi; \
	if grep -Fq 'scripts/detect_worker_budget.sh' agents.toml; then echo "ERROR: agents.toml advertises the retired shell worker detector." >&2; status=1; fi; \
	if grep -Eq 'cmd = "make (rust-smoke|rust-regression|heavy|python-smoke|python-regression)' agents.toml; then echo "ERROR: agents.toml advertises a local repository-validation command." >&2; status=1; fi; \
	for residual in 'path = "registry/engineering_standards.toml"' 'path = "registry/agents_contract.toml"' 'This TOML remains the canonical' 'mutation surface until SQLite migration'; do \
	    if ! grep -Fq "$$residual" registry/source_manifest.toml; then echo "ERROR: unmigrated generated-policy residual is not tracked: $$residual" >&2; status=1; fi; \
	done; \
	if grep -Eq 'scripts/detect_worker_budget[.]sh|make validate-local|make hooks-install|[.]githooks/pre-push|divide by two|logical threads / 2' registry/engineering_standards.toml registry/agents_contract.toml; then echo "ERROR: canonical policy records advertise retired local validation or divided workers." >&2; status=1; fi; \
	if ! grep -Fq 'check-local validate-local validate-local-xtask' Makefile; then echo "ERROR: retired local validation targets are not explicit refusal targets." >&2; status=1; fi; \
	exit "$$status"

validation-resource-contract-registry:
	@status=0; \
	for contract in '"--profile",' 'registry-integrity: $$(REGISTRY_INTEGRITY_CACHE)' 'cargo build --keep-going --profile validation -p gororoba_cli_governance --bin registry-integrity'; do \
	    case "$$contract" in \
	        '"--profile",') contract_file=crates/gororoba_db/src/bin/gororoba_db.rs ;; \
	        *) contract_file=Makefile ;; \
	    esac; \
	    if ! grep -Fq "$$contract" "$$contract_file"; then echo "ERROR: missing registry ownership contract: $$contract" >&2; status=1; fi; \
	done; \
	if ! grep -Fq 'name = "registry-integrity"' crates/gororoba_cli_governance/Cargo.toml; then echo "ERROR: governance package does not own registry-integrity." >&2; status=1; fi; \
	if grep -Fq 'name = "registry-integrity"' crates/gororoba_cli_data/Cargo.toml; then echo "ERROR: registry-integrity remains owned by the broad data CLI package." >&2; status=1; fi; \
	if grep -Eq '\.env\("(CARGO_BUILD_JOBS|RAYON_NUM_THREADS|RUST_TEST_THREADS)", "[0-9]+"\)' crates/gororoba_db/src/bin/gororoba_db.rs; then echo "ERROR: registry regeneration hard-codes a worker limit." >&2; status=1; fi; \
	exit "$$status"

validation-resource-contract-workers:
	@status=0; \
	for contract in 'detect_worker_budget.rs' 'CI_WORKER_BUDGET'; do \
	    if ! grep -Fq "$$contract" .github/workflows/ci.yml; then echo "ERROR: main CI worker contract is missing: $$contract" >&2; status=1; fi; \
	done; \
	if ! grep -Fq 'std::thread::available_parallelism()' crates/gororoba_cli/src/bin/detect_worker_budget.rs; then echo "ERROR: Rust worker detector does not use process-visible parallelism." >&2; status=1; fi; \
	if grep -Eq 'GOROROBA_WORKER_TEST_CPUS|max[(]|min[(]|clamp|/ *2|checked_div|unwrap_or' crates/gororoba_cli/src/bin/detect_worker_budget.rs; then echo "ERROR: worker detection contains an override, divisor, clamp, or fallback." >&2; status=1; fi; \
	if ! awk 'index($$0, "detect_worker_budget.rs") { pending++; setups++ } index($$0, "for variable in CI_WORKER_BUDGET WORKER_BUDGET CARGO_JOBS CARGO_BUILD_JOBS NEXTEST_TEST_THREADS RUST_TEST_THREADS RAYON_THREADS RAYON_NUM_THREADS") { if (pending != 1) bad=1; covered++; pending=0 } END { exit !(setups > 0 && pending == 0 && covered == setups && !bad) }' .github/workflows/ci.yml; then echo "ERROR: main CI does not export every canonical worker variable after each detector invocation." >&2; status=1; fi; \
	if grep -Eq 'divide by two|logical threads / 2' agents.toml; then echo "ERROR: active agent policy retains a divided-worker rule." >&2; status=1; fi; \
	if ! grep -Fq 'std::thread::available_parallelism()' xtask/src/main.rs; then echo "ERROR: xtask host profile does not use process-visible parallelism." >&2; status=1; fi; \
	if grep -Eq '(worker_budget|cargo_jobs|rayon_threads|rust_test_threads|nextest_test_threads|pytest_workers): physical_core_count' xtask/src/main.rs; then echo "ERROR: xtask host profile substitutes physical cores for process-visible workers." >&2; status=1; fi; \
	if grep -Eq '^(NPROC|NJOBS)[[:space:]]*:=' Makefile; then echo "ERROR: Makefile retains a second CPU-count heuristic." >&2; status=1; fi; \
	if grep -Eq 'heavy-(math|research)[[:space:]]*=[[:space:]]*\{[[:space:]]*max-threads[[:space:]]*=[[:space:]]*1' .config/nextest.toml; then echo "ERROR: nextest retains a CPU or memory safety serialization group." >&2; status=1; fi; \
	if grep -Fq -- '--test-threads=1' .github/workflows/bench-cd-kernel.yml; then echo "ERROR: benchmark CI fixes the Rust test harness to one worker." >&2; status=1; fi; \
	if grep -Eq 'physical_core_ids|init_physical_rayon_pool' crates/algebra_analysis/src/test_support.rs; then echo "ERROR: algebra tests substitute or pin physical cores." >&2; status=1; fi; \
	if grep -Eq 'PHYS_CORES|PHYS_CPUS|taskset' proofs/Makefile || grep -Fq -- '-j$$(JOBS)' proofs/Makefile; then echo "ERROR: proof validation replaces or constrains the inherited Make jobserver." >&2; status=1; fi; \
	for contract in 'cargo clippy --keep-going' 'cargo nextest run --no-fail-fast'; do \
	    if ! grep -Fq "$$contract" Makefile; then echo "ERROR: Rust collector contract is missing: $$contract" >&2; status=1; fi; \
	done; \
	if ! grep -Fq 'make --jobs="$$MAKE_JOBS" --keep-going all' .github/workflows/proofs.yml; then echo "ERROR: proof workflow does not pass every detected worker to Make." >&2; status=1; fi; \
	if ! grep -Fq 'components: clippy, rustfmt' .github/workflows/proofs.yml; then echo "ERROR: proof workflow does not provision pinned Rust components before parallel rustc calls." >&2; status=1; fi; \
	for workflow in .github/workflows/ci.yml .github/workflows/proofs.yml .github/workflows/bench-cd-kernel.yml .github/workflows/unsafe-survey.yml; do \
	    if ! grep -Fq 'detect_worker_budget.rs' "$$workflow"; then echo "ERROR: hosted Rust workflow lacks process-visible worker detection: $$workflow" >&2; status=1; fi; \
	done; \
	exit "$$status"

validation-resource-contract-collectors:
	@status=0; \
	rust_shard_block="$$(sed -n '/id: rust-shards/,/name: Check repository hygiene/p' .github/workflows/ci.yml)"; \
	for contract in 'ci-rust-shard-matrix' 'target-shard-package=gororoba_cli_physics' 'target-shard-package=gororoba_cli_data' 'target-shard-package=gororoba_cli_algebra' 'fallback_matrix=' 'CI_CARGO_TARGET_ARGS: $${{ matrix.cargo_target_args }}' 'CI_CARGO_FEATURES: $${{ matrix.cargo_features }}' 'matrix: $${{ fromJSON(needs.validation-core.outputs.rust_matrix) }}' 'make --jobs="$$WORKER_BUDGET" --keep-going "validate-ci-scoped-$${{ matrix.target }}"' 'timeout-minutes: 80' 'fail-fast: false' 'Report collected validation failures' 'Report aggregate validation admission' 'needs: [validation-policy, validation-core, rust-validation, scientific-replay]' 'scientific-replay:' '--bin hydrate-scientific-payloads' '--no-fail-fast -p algebra_experimental --lib' '--no-fail-fast -p algebra_experimental --test nufit_reference_identity' '--no-fail-fast -p gororoba_cli_physics --test box_counting_amplitude_identity' 'state=blocked_input' 'state=not_selected' 'executed_pass' 'executed_fail' "needs.validation-policy.result == 'success'" "needs.validation.result == 'success'" 'make --keep-going validation-resource-contract' 'make --jobs="$$WORKER_BUDGET" --keep-going casimir-optics-discrimination-audit-check' 'make --jobs="$$WORKER_BUDGET" --keep-going docs-freshness'; do \
	    if ! grep -Fq "$$contract" .github/workflows/ci.yml; then echo "ERROR: main CI collector contract is missing: $$contract" >&2; status=1; fi; \
	done; \
	for package in gororoba_cli_physics gororoba_cli_data gororoba_cli_algebra; do \
	    occurrences="$$(printf '%s\n' "$$rust_shard_block" | grep -Fc -- "--target-shard-package=$$package")"; \
	    if [ "$$occurrences" -ne 1 ]; then echo "ERROR: Rust shard block must select $$package exactly once." >&2; status=1; fi; \
	done; \
	target_package_count="$$(printf '%s\n' "$$rust_shard_block" | grep -Fc -- '--target-shard-package=')"; \
	if [ "$$target_package_count" -ne 3 ]; then echo "ERROR: Rust shard block must select exactly three large binary packages." >&2; status=1; fi; \
	for lane in clippy light heavy; do \
	    if ! grep -Fq "validate-ci-scoped-$$lane" Makefile; then echo "ERROR: missing independently runnable Rust CI shard: $$lane" >&2; status=1; fi; \
	done; \
	if ! sed -n '/id: lint/,/name: Check repository hygiene/p' .github/workflows/ci.yml | grep -Fq 'continue-on-error: true'; then echo "ERROR: lint routing failure is not collectable." >&2; status=1; fi; \
	for condition in \
	    "if: always() && steps.route.outputs.run_rust == 'true'" \
	    "if: always() && steps.route.outputs.run_check == 'true'" \
	    "if: always() && steps.route.outputs.run_governance == 'true'" \
	    "if: always() && (steps.route.outputs.run_rust == 'true' || steps.route.outputs.run_governance == 'true')" \
	    "if: always() && steps.paths.outputs.casimir_audit == 'true'" \
	    "if: always() && steps.paths.outputs.dependencies == 'true'" \
	    "if: always() && steps.paths.outputs.warp == 'true'"; do \
	    if ! grep -Fq "$$condition" .github/workflows/ci.yml; then echo "ERROR: independent CI leaf lacks an always-based condition: $$condition" >&2; status=1; fi; \
	done; \
	if ! grep -Fq 'LINT_OUTCOME: $${{ steps.lint.outcome }}' .github/workflows/ci.yml; then echo "ERROR: final validation collector omits lint routing." >&2; status=1; fi; \
	if ! grep -Fq 'ROUTE_OUTCOME: $${{ steps.route.outcome }}' .github/workflows/ci.yml; then echo "ERROR: final validation collector omits reverse dependency routing." >&2; status=1; fi; \
	if ! grep -Fq 'PATHS_OUTCOME: $${{ steps.paths.outcome }}' .github/workflows/ci.yml; then echo "ERROR: final validation collector omits path routing." >&2; status=1; fi; \
	if ! grep -Fq 'RUST_SHARDS_OUTCOME: $${{ steps.rust-shards.outcome }}' .github/workflows/ci.yml; then echo "ERROR: final validation collector omits Rust shard routing." >&2; status=1; fi; \
	if ! grep -Fq 'no_tests_args=(--no-tests=pass)' Makefile; then echo "ERROR: binary-only CI shards reject successful zero-test compilation." >&2; status=1; fi; \
	if [ "$$(sed -n '/id: rust-shards/,/name: Check repository hygiene/p' .github/workflows/ci.yml | grep -Fc "printf 'matrix=%s\\n' \"\$$rust_matrix\" >> \"\$$GITHUB_OUTPUT\"")" -ne 1 ]; then echo "ERROR: dynamic Rust shard matrix must be emitted exactly once." >&2; status=1; fi; \
	if [ "$$(sed -n '/id: rust-shards/,/name: Check repository hygiene/p' .github/workflows/ci.yml | grep -Fc "printf 'matrix=%s\\n' \"\$$fallback_matrix\" >> \"\$$GITHUB_OUTPUT\"")" -ne 2 ]; then echo "ERROR: fallback Rust shard matrix must be emitted only by the two failure branches." >&2; status=1; fi; \
	if ! sed -n '/name: Retain successful core validation artifacts/,/key: $${{ steps.rust-cache.outputs.cache-primary-key }}/p' .github/workflows/ci.yml | grep -Fq 'if: success()'; then echo "ERROR: core cache retention is not success-only." >&2; status=1; fi; \
	if ! sed -n '/name: Retain successful Rust validation artifacts/,/key: $${{ steps.rust-cache.outputs.cache-primary-key }}/p' .github/workflows/ci.yml | grep -Fq 'if: success()'; then echo "ERROR: Rust shard cache save can run while Cargo artifacts are unstable." >&2; status=1; fi; \
	if ! grep -Fq 'Report collected proof failures' .github/workflows/proofs.yml; then echo "ERROR: proof collector contract is missing." >&2; status=1; fi; \
	if ! grep -Fq 'data/output/audit/casimir-optics-discrimination/sources/** -text' .gitattributes; then echo "ERROR: hash-bound source-byte contract is missing." >&2; status=1; fi; \
	for contract in 'id = "ci.validation.scoped"' 'id = "ci.validation.full"'; do \
	    if ! grep -Fq "$$contract" agents.toml; then echo "ERROR: agents.toml CI entrypoint is missing: $$contract" >&2; status=1; fi; \
	done; \
	exit "$$status"

casimir-optics-discrimination-audit-check: require-ci-validation-authority
	$(CARGO_ENV) cargo run --locked --profile validation -p gororoba_cli_physics \
	    --bin casimir-optics-discrimination-audit -- \
	    --output-directory data/output/audit/casimir-optics-discrimination \
	    --check \
	    --expected-output-directory reports/validation/casimir-optics-discrimination-expected

.PHONY: validate-static validate-static-and-registry validate-comprehensive
.PHONY: audit-comprehensive audit-comprehensive-structured validate-supply-chain validate-dataset-experiments gate-fast gate-warm gate-deep audit-deep audit-deep-structured typos machete audit geiger supply-chain-gate ndlb-gate

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

# validate-supply-chain: aggregated check chaining cargo-deny + machete +
# a register_custom_getrandom non-existence grep (keeps RUSTSEC-2026-0097
# exposure provably zero per docs/adr/rustsec-dispositions.md).
# Runs deterministically; safe in validate-comprehensive.
validate-supply-chain:
	@echo "=== validate-supply-chain ==="
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
	if [ "$$fail" -ne 0 ]; then echo "=== validate-supply-chain: FAILED ==="; exit 1; fi
	@echo "=== validate-supply-chain: PASSED ==="

supply-chain-gate: validate-supply-chain
	@echo "DEPRECATED: make supply-chain-gate is a compatibility alias for make validate-supply-chain."

# validate-static: lightweight hygiene and dependency checks.
# Runs the cached ASCII/terminology check and cargo-machete without compiling
# the workspace Rust test closure. Formatter convergence remains the explicit
# `fmt-check` surface until its 59-file baseline is reconciled deliberately.
validate-static:
	@echo "=== validate-static: lightweight hygiene checks ==="
	$(MAKE) check
	cargo machete
	@echo "=== validate-static: PASSED ==="

# validate-static-and-registry: lightweight checks plus registry policy checks.
# ~10s if binaries are cached, ~2min on cold cache (first compilation).
validate-static-and-registry: validate-static
	@echo "=== validate-static-and-registry: policy checks ==="
	$(MAKE) validate-governance
	@echo "=== validate-static-and-registry: PASSED ==="

# validate-comprehensive: static, registry, Rust, and dependency checks.
# WHY: Full CI-grade audit. Catches everything including API compat and advisories.
validate-comprehensive: require-ci-validation-authority validate-static-and-registry
	@echo "=== validate-comprehensive: Rust and dependency checks ==="
	# rust-regression owns the clippy prerequisite and the two nextest profiles.
	$(MAKE) rust-regression
	cargo audit
	$(MAKE) cargo-deny-check
	@echo "=== validate-comprehensive: PASSED ==="

# Compatibility aliases. New automation uses the descriptive validate-* names.
gate-fast: validate-static
	@echo "DEPRECATED: make gate-fast is a compatibility alias for make validate-static."

gate-warm: validate-static-and-registry
	@echo "DEPRECATED: make gate-warm is a compatibility alias for make validate-static-and-registry."

gate-deep: validate-comprehensive
	@echo "DEPRECATED: make gate-deep is a compatibility alias for make validate-comprehensive."

# audit-comprehensive: opt-in composite audit. It is not required by default
# make or CI on every pull request.
# WHY: Aggregates all expensive one-off audit tools (semver, deny, dep-audit, CPD,
#      docs-freshness) into a single reviewable target for pre-release or periodic runs.
# HOW: make audit-comprehensive (standalone, no preconditions required)
# NOTE: cpd-audit requires pmd; the target will self-report if pmd is absent.
audit-comprehensive: require-ci-validation-authority
	@echo "=== audit-comprehensive: full opt-in audit suite ==="
	$(MAKE) rust-clippy
	@# rust-semver-check is intentionally skipped in audit-comprehensive.
	@# WHY: cargo-semver-checks --baseline-rev checks out the baseline tag into a temp dir.
	@#   fwht = { path = "../cratesgororobas/fwht" } in the root Cargo.toml is an external
	@#   sibling-directory path dep that cannot be resolved from a git temp checkout.
	@#   This makes the baseline build fail for every workspace member.
	@# Resolution: run `make rust-semver-check` standalone only after moving fwht to
	@#   crates.io (see TODO at Cargo.toml line 211) or into the workspace.
	$(MAKE) cargo-deny-check
	$(MAKE) dep-audit
	@# docs-freshness is intentionally skipped in audit-comprehensive.
	@# WHY: cargo doc --workspace fails with -D rustdoc::broken-intra-doc-links because
	@#   mathematical notation like [a,b,c] and X[t] in doc comments is misread as doc
	@#   links. Affected crates: algebra_analysis, algebra_experimental, brown_1972,
	@#   wilmot_2025 (and potentially more). These are pre-existing; run separately as
	@#   `make docs-freshness` to track progress. Fix: escape brackets as \[a,b,c\].
	$(MAKE) cpd-audit
	@echo "=== audit-comprehensive: PASSED ==="

audit-deep: audit-comprehensive
	@echo "DEPRECATED: make audit-deep is a compatibility alias for make audit-comprehensive."

test: rust-regression

# Build repo_utilities once in the shared validation profile, then invoke the
# binary directly. A separate release-profile build adds a second target tree
# without improving the ASCII or terminology checks.
REPO_UTILITIES_BIN := $(REPO_CARGO_TARGET_DIR)/validation-tools/repo-utilities

check: require-ci-validation-authority check-ansi check-terminology cuda-source-ownership
	@echo "OK: fast shared check suite complete."

check-ansi: $(REPO_UTILITIES_BIN)
	@$(REPO_UTILITIES_BIN) ansi-check --check

check-terminology: $(REPO_UTILITIES_BIN)
	@$(REPO_UTILITIES_BIN) terminology-gate

# Compatibility names refuse execution. Their recipes have no prerequisites,
# so an invocation cannot build a validation tool before returning the error.
check-local validate-local validate-local-xtask gate-local gate-local-xtask pre-push-gate pre-push-gate-scoped rust-regression-scoped:
	@echo "ERROR: $@ is retired; repository validation runs only in GitHub Actions." >&2
	@echo "Push the branch to trigger the scoped CI workflow." >&2
	@exit 2

# Governance verifier targets
registry-verify-markdown-governance:
	$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify markdown-removal-policy

# Registry validation binaries are cached at stable paths under
# $(VALIDATION_TOOLS_DIR)/. Cache vars and rules live below where
# VALIDATION_TOOLS_DIR is defined (search "MARKDOWN_REGISTRY_CACHE").
# This target consumes those cache entries.
validate-governance: require-ci-validation-authority registry-validation-tools
	$(MARKDOWN_REGISTRY_CACHE) verify-all
	$(GOVERNANCE_VERIFY_CACHE) validate-all
# execution-planning cross-checks every experiment and lineage row against the
# declared execution targets and against the sha256 of its own run command. It
# stayed outside this chain while three binary-collapse commits regressed it,
# so it runs here rather than only under registry-verify-execution-planning.
	$(EXECUTION_PLANNING_CACHE) --verify --repo-root .
	@echo ""
	@echo "=========================================="
	@echo "REGISTRY GOVERNANCE VALIDATION: PASSED"
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

governance-gate-readonly: validate-governance
	@echo "DEPRECATED: make governance-gate-readonly is a compatibility alias for make validate-governance."

governance-gate: validate-governance validate-dataset-experiments
	@echo "DEPRECATED: make governance-gate is a compatibility alias for make validate-governance plus make validate-dataset-experiments."

# NDLB gate: No-Dataset-Left-Behind invariant. Every data/external/*
# subdir must be one of: active (experiment-bound), synthetic
# (local artifact), or deferred (tombstoned with a defer_to_sprint
# target). Unknown or dark dirs fail fast.
validate-dataset-experiments:
	@echo "[validate-dataset-experiments] validating dataset/server/experiment invariants..."
	$(CARGO_ENV) cargo run --profile validation -q -p gororoba_cli_data --bin ndlb-gate
	@echo "[validate-dataset-experiments] OK."

ndlb-gate: validate-dataset-experiments
	@echo "DEPRECATED: make ndlb-gate is a compatibility alias for make validate-dataset-experiments."

wave6-gate: governance-gate
	@echo "DEPRECATED: make wave6-gate is a legacy alias. Use make governance-gate."

# Cache validation binaries and host-profile output. Each
# `cargo run -q -p X --bin Y` invocation pays ~30-60s of metadata walk
# overhead even when nothing changed. Stable paths and one Cargo build per
# dependency tier eliminate repeated package selection, linking, and process
# startup without forcing a local source edit to build the broad registry CLI.
WORKSPACE_ROUTING_CACHE := $(VALIDATION_TOOLS_DIR)/workspace-routing
HOST_PROFILE_CACHE := $(VALIDATION_TOOLS_DIR)/host-profile.sh
REPO_UTILITIES_SOURCE_DEPS := $(shell find crates/repo_utilities -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -print) \
                              Cargo.toml Cargo.lock rust-toolchain.toml

$(REPO_UTILITIES_BIN): $(REPO_UTILITIES_SOURCE_DEPS) $(VALIDATION_SOURCE_IDENTITY_FILE)
	@mkdir -p $(VALIDATION_TOOLS_DIR)
	@echo "[validation-tools] building repo-utilities in the validation profile"
	@$(CARGO_ENV) cargo build --profile validation -p repo_utilities --bin repo-utilities
	@$(call stage_tool,$(REPO_CARGO_TARGET_DIR)/validation/repo-utilities,$@)
	@touch $@

# The core bundle depends on workspace crates through provenance and verified
# data paths. Track the full Rust and manifest source set so a copied binary
# cannot outlive a changed transitive dependency. Cargo's incremental cache
# still limits recompilation to the actual dependency closure.
CORE_VALIDATION_SOURCE_DEPS := $(shell find crates xtask -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -print) \
                               Cargo.toml Cargo.lock rust-toolchain.toml Makefile \
                               crates/lbm_3d_cuda/cuda_source_ownership.toml

# The routing proxy has a dedicated stamp because CI validation needs scope
# classification but does not need xtask. Broad and structured lanes build
# xtask through the separate core stamp.
XTASK_CACHE := $(VALIDATION_TOOLS_DIR)/xtask
ROUTING_VALIDATION_STAMP := $(VALIDATION_TOOLS_DIR)/routing-validation.stamp

$(ROUTING_VALIDATION_STAMP): $(CORE_VALIDATION_SOURCE_DEPS) $(VALIDATION_SOURCE_IDENTITY_FILE)
	@mkdir -p $(VALIDATION_TOOLS_DIR)
	@echo "[validation-tools] building the local scope router"
	@$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_governance --bin workspace-routing-proxy
	@$(call stage_tool,$(REPO_CARGO_TARGET_DIR)/validation/workspace-routing-proxy,$(WORKSPACE_ROUTING_CACHE))
	@touch $(ROUTING_VALIDATION_STAMP) $(WORKSPACE_ROUTING_CACHE)

cuda-source-ownership: $(XTASK_CACHE)
	$(XTASK_CACHE) cuda-source-ownership
CORE_VALIDATION_STAMP := $(VALIDATION_TOOLS_DIR)/core-validation.stamp

$(CORE_VALIDATION_STAMP): $(CORE_VALIDATION_SOURCE_DEPS) $(VALIDATION_SOURCE_IDENTITY_FILE)
	@mkdir -p $(VALIDATION_TOOLS_DIR)
	@echo "[validation-tools] building xtask for broad or structured validation"
	@$(CARGO_ENV) cargo build --profile validation -p xtask --bin xtask
	@$(call stage_tool,$(REPO_CARGO_TARGET_DIR)/validation/xtask,$(XTASK_CACHE))
	@touch $(CORE_VALIDATION_STAMP) $(XTASK_CACHE)

$(WORKSPACE_ROUTING_CACHE): $(ROUTING_VALIDATION_STAMP)
	@touch $@

$(XTASK_CACHE): $(CORE_VALIDATION_STAMP)
	@touch $@

$(HOST_PROFILE_CACHE): $(CORE_VALIDATION_STAMP)
	@mkdir -p $(VALIDATION_TOOLS_DIR)
	@echo "[validation-tools] refreshing host-profile snapshot (xtask source changed)"
	@$(XTASK_CACHE) host-profile --format shell > $@.tmp
	@mv $@.tmp $@

# Cache every read-only registry and Rust-integrity executable from one
# validation-profile Cargo invocation. The source set deliberately covers all
# workspace Rust and manifest files, so a transitive code change cannot leave a
# stale copied executable behind. Registry TOML edits do not rebuild tools:
# every executable reads the current working tree at runtime.
REGISTRY_VALIDATION_BINS := claims-verify registry-check test-inventory \
                            semantic-atoms evidence-provenance registry-integrity \
                            execution-planning governance-verify markdown-registry \
                            project-counter-sync provenance
REGISTRY_BUNDLED_VALIDATION_BINS := $(filter-out registry-integrity,$(REGISTRY_VALIDATION_BINS))
REGISTRY_VALIDATION_SOURCE_DEPS := $(shell find crates xtask -type f \( -name '*.rs' -o -name 'Cargo.toml' \) -print) \
                                   Cargo.toml Cargo.lock rust-toolchain.toml Makefile
REGISTRY_VALIDATION_STAMP := $(VALIDATION_TOOLS_DIR)/registry-validation.stamp
REGISTRY_INTEGRITY_STAMP := $(VALIDATION_TOOLS_DIR)/registry-integrity.stamp
REGISTRY_VALIDATION_CACHE_FILES := $(addprefix $(VALIDATION_TOOLS_DIR)/,$(REGISTRY_VALIDATION_BINS))
REGISTRY_BUNDLED_VALIDATION_CACHE_FILES := $(addprefix $(VALIDATION_TOOLS_DIR)/,$(REGISTRY_BUNDLED_VALIDATION_BINS))
MARKDOWN_REGISTRY_CACHE := $(VALIDATION_TOOLS_DIR)/markdown-registry
GOVERNANCE_VERIFY_CACHE := $(VALIDATION_TOOLS_DIR)/governance-verify
EXECUTION_PLANNING_CACHE := $(VALIDATION_TOOLS_DIR)/execution-planning
REGISTRY_INTEGRITY_CACHE := $(VALIDATION_TOOLS_DIR)/registry-integrity
PROJECT_COUNTER_CACHE := $(VALIDATION_TOOLS_DIR)/project-counter-sync
PROVENANCE_CACHE := $(VALIDATION_TOOLS_DIR)/provenance

# stage_tool copies a compiled tool into the worktree-local tools dir and drops
# its debug sections. The shared build-dir hands every worktree the rlibs another
# checkout compiled, and DWARF comp_dir strings in those rlibs name that
# checkout; workspace sources are compiled by relative path, so after the strip
# the only absolute checkout paths left in a tool are compile-time bake-ins
# such as env!("CARGO_MANIFEST_DIR"), which is what validation-tools-check-paths
# exists to catch. A missing strip leaves the copy unstripped and the scan
# reports the debug residue instead.
define stage_tool
cp -f $(1) $(2) && { command -v strip >/dev/null 2>&1 && strip --strip-debug $(2) || true; }
endef

# One rule produces the complete validation tool set. The stamp is the only
# Make dependency; the copied binaries are stable execution paths.
$(REGISTRY_VALIDATION_STAMP): $(REGISTRY_VALIDATION_SOURCE_DEPS) $(VALIDATION_SOURCE_IDENTITY_FILE)
	@mkdir -p $(VALIDATION_TOOLS_DIR)
	@echo "[validation-tools] building registry validation tools in one Cargo session"
	@$(CARGO_ENV) cargo build --keep-going --profile validation \
		-p gororoba_cli_data $(foreach binary,$(filter-out provenance,$(REGISTRY_BUNDLED_VALIDATION_BINS)),--bin $(binary)) \
		-p gororoba_cli_provenance --bin provenance
	@status=0; \
	for binary in $(REGISTRY_BUNDLED_VALIDATION_BINS); do \
		if ! ( $(call stage_tool,"$(REPO_CARGO_TARGET_DIR)/validation/$$binary","$(VALIDATION_TOOLS_DIR)/$$binary") ); then \
		    echo "ERROR: failed to stage registry validation tool: $$binary" >&2; \
		    status=1; \
		fi; \
	done; \
	exit "$$status"
	@touch $(REGISTRY_VALIDATION_STAMP) $(REGISTRY_BUNDLED_VALIDATION_CACHE_FILES)

$(REGISTRY_INTEGRITY_STAMP): $(REGISTRY_VALIDATION_SOURCE_DEPS) $(VALIDATION_SOURCE_IDENTITY_FILE)
	@mkdir -p $(VALIDATION_TOOLS_DIR)
	@echo "[validation-tools] building the registry-integrity tool from its slim governance owner"
	@$(CARGO_ENV) cargo build --keep-going --profile validation -p gororoba_cli_governance --bin registry-integrity
	@$(call stage_tool,$(REPO_CARGO_TARGET_DIR)/validation/registry-integrity,$(REGISTRY_INTEGRITY_CACHE))
	@touch $(REGISTRY_INTEGRITY_STAMP) $(REGISTRY_INTEGRITY_CACHE)

$(REGISTRY_INTEGRITY_CACHE): $(REGISTRY_INTEGRITY_STAMP)
	@touch $@

registry-validation-tools: $(REGISTRY_VALIDATION_STAMP) $(REGISTRY_INTEGRITY_CACHE)
	@echo "OK: registry validation tools cached at $(VALIDATION_TOOLS_DIR)/."
validation-tools: $(WORKSPACE_ROUTING_CACHE) $(HOST_PROFILE_CACHE) $(XTASK_CACHE) registry-validation-tools
	@echo "OK: validation-tools cached at $(VALIDATION_TOOLS_DIR)/."

# Every file under $(VALIDATION_TOOLS_DIR) that a gate lane executes or
# compares against. The lock and the cache-check sentinel stay out, so a
# rebuild operation excludes the cache-check sentinel.
VALIDATION_TOOL_PACKAGES := repo_root repo_utilities gororoba_cli_governance xtask \
                            gororoba_cli_data gororoba_cli_provenance

VALIDATION_TOOL_ARTIFACTS := $(REPO_UTILITIES_BIN) $(WORKSPACE_ROUTING_CACHE) \
                             $(HOST_PROFILE_CACHE) $(XTASK_CACHE) \
                             $(ROUTING_VALIDATION_STAMP) $(CORE_VALIDATION_STAMP) \
                             $(REGISTRY_VALIDATION_STAMP) $(REGISTRY_INTEGRITY_STAMP) \
                             $(REGISTRY_VALIDATION_CACHE_FILES)

# validation-tools-rebuild discards every staged binary, stamp and identity
# file, then rebuilds through the ordinary tool lane. Reach for it when a
# staged binary resolves a path from a checkout that no longer exists: the
# identity stamp decides whether Make re-runs the build, while the bytes come
# from the shared Cargo build-dir, so only a forced discard replaces them.
validation-tools-rebuild:
	@for f in $(VALIDATION_TOOL_ARTIFACTS) $(wildcard $(VALIDATION_TOOLS_DIR)/tool-identity.*) $(wildcard $(VALIDATION_TOOLS_DIR)/source-identity.*); do \
	    ( $(call guarded_rm,$$f) ) || exit 1; \
	done
	@echo "[validation-tools] discarded every staged tool and identity file under $(VALIDATION_TOOLS_DIR)"
# Discarding the copy is not enough. Cargo keys a workspace crate on content,
# so the shared build-dir answers the next build with the artifact another
# worktree compiled, path bake-in and all. Cleaning the tool packages and the
# repo_root crate that carries the compile-time fallback forces Cargo to
# recompile them under this checkout.
	@$(CARGO_ENV) cargo clean --profile validation $(foreach pkg,$(VALIDATION_TOOL_PACKAGES),-p $(pkg)) || true
	$(MAKE) validation-tools
	$(MAKE) validation-tools-check-paths

# validation-tools-check-paths reads each staged file and fails on an absolute
# path under $(REPO_WORKTREES_ROOT) that names neither this checkout nor a
# directory any live entry accounts for. Debuginfo and panic-location strings carry the compiling
# checkout's path, so a binary handed over from a removed worktree aborts on
# its first path resolution; the scan catches it before a lane executes it.
REPO_WORKTREES_ROOT ?= $(HOME)/worktrees
validation-tools-check-paths: $(REPO_UTILITIES_BIN)
	@$(REPO_UTILITIES_BIN) validation-tool-paths \
	    --tools-dir $(VALIDATION_TOOLS_DIR) \
	    --worktrees-root $(REPO_WORKTREES_ROOT) \
	    --current-root $(CURDIR)

validation-tools-clean:
	rm -f $(WORKSPACE_ROUTING_CACHE) $(HOST_PROFILE_CACHE)
	@echo "OK: validation-tools cache cleared."

validate-ci-registry: require-ci-validation-authority validate-registry
	@echo "OK: CI registry validation passed."

gate-ci-registry: validate-ci-registry
	@echo "DEPRECATED: make gate-ci-registry is a compatibility alias for make validate-ci-registry."

validate-ci-rust: require-ci-validation-authority
	$(MAKE) validation-tools
	$(MAKE) rust-regression CARGO_ENV="$(CARGO_ENV_CI)"
	$(MAKE) validate-rust-integrity CARGO_ENV="$(CARGO_ENV_CI)"
	$(MAKE) cargo-deny-check CARGO_ENV="$(CARGO_ENV_CI)"
	$(MAKE) db-schema-drift-check CARGO_ENV="$(CARGO_ENV_CI)"
	@echo "OK: CI Rust validation passed."

gate-ci-rust: validate-ci-rust
	@echo "DEPRECATED: make gate-ci-rust is a compatibility alias for make validate-ci-rust."

.PHONY: validate-ci-scoped-rust validate-ci-scoped-clippy validate-ci-scoped-light validate-ci-scoped-heavy validate-ci-scoped-rust-lane
validate-ci-scoped-rust: validate-ci-scoped-clippy validate-ci-scoped-light validate-ci-scoped-heavy
	@echo "OK: scoped CI Rust validation passed."

validate-ci-scoped-clippy:
	@$(MAKE) --no-print-directory CI_RUST_LANE=clippy validate-ci-scoped-rust-lane

validate-ci-scoped-light:
	@$(MAKE) --no-print-directory CI_RUST_LANE=light validate-ci-scoped-rust-lane

validate-ci-scoped-heavy:
	@$(MAKE) --no-print-directory CI_RUST_LANE=heavy validate-ci-scoped-rust-lane

validate-ci-scoped-rust-lane: SHELL := /bin/bash
validate-ci-scoped-rust-lane: export CI_RUST_SCOPE := $(CI_RUST_SCOPE)
validate-ci-scoped-rust-lane: export CI_CLIPPY_SCOPE := $(CI_CLIPPY_SCOPE)
validate-ci-scoped-rust-lane: export CI_CARGO_TARGET_ARGS := $(CI_CARGO_TARGET_ARGS)
validate-ci-scoped-rust-lane: export CI_CARGO_FEATURES := $(CI_CARGO_FEATURES)
validate-ci-scoped-rust-lane: require-ci-validation-authority
	@set -euo pipefail; \
	validate_scope() { \
	    local scope_name="$$1" scope_value="$$2"; \
	    local -a scope_tokens=(); \
	    if [[ "$$scope_value" == *$$'\n'* || "$$scope_value" == *$$'\r'* ]]; then \
	        echo "ERROR: $$scope_name requires a single-line package scope." >&2; return 1; \
	    fi; \
	    read -r -a scope_tokens <<< "$$scope_value"; \
	    if [ "$${#scope_tokens[@]}" -eq 1 ] && [ "$${scope_tokens[0]}" = --workspace ]; then return; fi; \
	    if [ "$${#scope_tokens[@]}" -eq 0 ] || (( $${#scope_tokens[@]} % 2 != 0 )); then \
	        echo "ERROR: $$scope_name requires --workspace or explicit -p package pairs." >&2; return 1; \
	    fi; \
	    for ((scope_index=0; scope_index<$${#scope_tokens[@]}; scope_index+=2)); do \
	        if [ "$${scope_tokens[scope_index]}" != -p ] || \
	           [[ ! "$${scope_tokens[scope_index+1]}" =~ ^[A-Za-z0-9_][A-Za-z0-9_-]*$$ ]]; then \
	            echo "ERROR: invalid $$scope_name package scope." >&2; return 1; \
	        fi; \
	    done; \
	}; \
	case "$$CI_RUST_LANE" in \
	    clippy) \
	        validate_scope CI_CLIPPY_SCOPE "$$CI_CLIPPY_SCOPE"; \
	        read -r -a clippy_scope <<< "$$CI_CLIPPY_SCOPE"; \
	        echo "[ci-rust-clippy] scope: $$CI_CLIPPY_SCOPE"; \
	        $(CARGO_ENV_CI) cargo clippy --keep-going --locked --profile validation --all-targets "$${clippy_scope[@]}" -- -D warnings; \
	        ;; \
	    light|heavy) \
	        validate_scope CI_RUST_SCOPE "$$CI_RUST_SCOPE"; \
	        read -r -a rust_scope <<< "$$CI_RUST_SCOPE"; \
	        ;; \
	    *) echo "ERROR: CI_RUST_LANE must be clippy, light, or heavy." >&2; exit 1 ;; \
	esac; \
	if [ "$$CI_RUST_LANE" = clippy ]; then exit 0; fi; \
	cargo_target_scope="$${CI_CARGO_TARGET_ARGS:---all-targets}"; \
	read -r -a cargo_target_args <<< "$$cargo_target_scope"; \
	cargo_feature_args=(); \
	no_tests_args=(); \
	if [ -n "$$CI_CARGO_FEATURES" ]; then \
	    if [[ ! "$$CI_CARGO_FEATURES" =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$$ ]]; then echo "ERROR: CI_CARGO_FEATURES requires a comma-separated feature list." >&2; exit 1; fi; \
	    cargo_feature_args=(--features "$$CI_CARGO_FEATURES"); \
	fi; \
	case "$${cargo_target_args[0]}" in \
	    --lib) \
	        if ! { [ "$${#cargo_target_args[@]}" -eq 2 ] && [ "$${cargo_target_args[1]}" = --tests ]; } && \
	           ! { [ "$${#cargo_target_args[@]}" -eq 4 ] && [ "$${cargo_target_args[1]}" = --bins ] && [ "$${cargo_target_args[2]}" = --tests ] && [ "$${cargo_target_args[3]}" = --examples ]; }; then \
	            echo "ERROR: non-binary target shard requires --lib --tests or --lib --bins --tests --examples." >&2; exit 1; \
	        fi \
	        ;; \
	    --bin) \
	        no_tests_args=(--no-tests=pass); \
	        if (( $${#cargo_target_args[@]} % 2 != 0 )); then echo "ERROR: binary target shard requires --bin name pairs." >&2; exit 1; fi; \
	        for ((target_index=0; target_index<$${#cargo_target_args[@]}; target_index+=2)); do \
	            if [ "$${cargo_target_args[target_index]}" != --bin ] || [[ ! "$${cargo_target_args[target_index+1]}" =~ ^[A-Za-z0-9_][A-Za-z0-9_-]*$$ ]]; then echo "ERROR: invalid binary target shard." >&2; exit 1; fi; \
	        done \
	        ;; \
	    *) echo "ERROR: CI_CARGO_TARGET_ARGS must select all targets, library and tests, or named binaries." >&2; exit 1 ;; \
	esac; \
	light_scope=(); heavy_scope=(); \
	if [ "$${rust_scope[0]}" = --workspace ]; then \
	    light_scope=(--workspace --exclude algebra_analysis --exclude gr_core); \
	    heavy_scope=(-p algebra_analysis -p gr_core); \
	else \
	    for ((scope_index=1; scope_index<$${#rust_scope[@]}; scope_index+=2)); do \
	        package_name="$${rust_scope[scope_index]}"; \
	        case "$$package_name" in \
	            algebra_analysis|gr_core) heavy_scope+=(-p "$$package_name") ;; \
	            *) light_scope+=(-p "$$package_name") ;; \
	        esac; \
	    done; \
	fi; \
	case "$$CI_RUST_LANE" in \
	    light) \
	        if [ "$${#light_scope[@]}" -eq 0 ]; then echo "[ci-rust-light] no applicable packages"; exit 0; fi; \
	        echo "[ci-rust-light] scope: $${light_scope[*]}"; \
	        $(CARGO_ENV_CI) cargo nextest run --no-fail-fast "$${no_tests_args[@]}" --locked --cargo-profile test -P ci "$${cargo_target_args[@]}" "$${cargo_feature_args[@]}" --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) "$${light_scope[@]}"; \
	        ;; \
	    heavy) \
	        if [ "$${#heavy_scope[@]}" -eq 0 ]; then echo "[ci-rust-heavy] no applicable packages"; exit 0; fi; \
	        echo "[ci-rust-heavy] scope: $${heavy_scope[*]}"; \
	        $(CARGO_ENV_CI) cargo nextest run --no-fail-fast --locked --cargo-profile test-heavy -P heavy "$${cargo_target_args[@]}" "$${cargo_feature_args[@]}" --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) "$${heavy_scope[@]}"; \
	        ;; \
	esac

db-schema-drift-check: $(XTASK_CACHE)
	$(CARGO_ENV) $(XTASK_CACHE) db-docs --check
	@echo "OK: db-schema-drift-check passed."

host-profile: $(XTASK_CACHE)
	$(XTASK_CACHE) host-profile --format json

validate-ci: require-ci-validation-authority
	$(MAKE) validation-tools
	$(CARGO_ENV) $(XTASK_CACHE) validate-ci
	@echo "OK: CI validation completed."

validate-repository: require-ci-validation-authority
	$(MAKE) validation-tools
	$(CARGO_ENV) $(XTASK_CACHE) validate-repository
	@echo "OK: repository validation completed."

gate-audit: validate-repository
	@echo "DEPRECATED: make gate-audit is a compatibility alias for make validate-repository."

# PH-5.A: structured audit-comprehensive composite (rust-clippy + cargo-deny +
# dep-audit + cpd-audit) with per-step log capture, Markdown summary,
# and TOML record under reports/audit-comprehensive/<date>/<time>/. Use this
# for tranche-acceptance evidence; use plain `make audit-comprehensive` for
# interactive runs.
audit-comprehensive-structured:
	$(CARGO_ENV) cargo run -p xtask -- audit-comprehensive
	@echo "OK: audit-comprehensive-structured completed."

audit-deep-structured: audit-comprehensive-structured
	@echo "DEPRECATED: make audit-deep-structured is a compatibility alias for make audit-comprehensive-structured."

# WHY: validate-ci-rust runs rust-regression (full workspace compile + nextest run,
# ~9 min). For registry/governance-only edits -- TOML updates, schema changes,
# claims.toml regeneration -- that compile overhead is pure waste. The fast
# repository validation path skips Rust compilation entirely: only
# validate-ci-registry (governance + schema
# checks, ~2 min) runs, and it fails fast on the first error.
validate-repository-fast: require-ci-validation-authority
	$(MAKE) validation-tools
	$(CARGO_ENV) $(XTASK_CACHE) validate-repository --fast
	@echo "OK: fast repository validation completed."

gate-audit-fast: validate-repository-fast
	@echo "DEPRECATED: make gate-audit-fast is a compatibility alias for make validate-repository-fast."

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
	@# CLI cargo and gate cargo both write to the resolved target dir (via
	@# .cargo/config.toml build.target-dir or the CARGO_TARGET_DIR override).
	@printf '=== Cache mode: %s (owner %s) ===\n' '$(REPO_CACHE_MODE)' '$(REPO_CACHE_OWNER)'
	@printf '=== Cargo target dir %s ===\n' '$(REPO_CARGO_TARGET_DIR)'
	@du -sh $(REPO_CARGO_TARGET_DIR) 2>/dev/null || printf '(missing)\n'
	@printf '=== CARGO_HOME %s ===\n' '$(REPO_CARGO_HOME)'
	@du -sh $(REPO_CARGO_HOME) 2>/dev/null || true
	@printf '=== Build-dir intermediates %s ===\n' '$(CACHE_SWEEP_CBUILD_ROOT)'
	@du -sh $(CACHE_SWEEP_CBUILD_ROOT) 2>/dev/null || printf '(empty)\n'
	@printf '=== Validation tools cache %s ===\n' '$(VALIDATION_TOOLS_DIR)'
	@du -sh $(VALIDATION_TOOLS_DIR) 2>/dev/null || printf '(empty)\n'
	@printf '=== Experimental dirs (%s/exp-*-target) ===\n' '$(REPO_LOCAL_CACHE_ROOT)'
	@du -sh $(REPO_LOCAL_CACHE_ROOT)/exp-*-target 2>/dev/null || printf '(none)\n'
	@printf '=== Residual target/ (cargo doc + mdbook, NOT cargo build) ===\n'
	@du -sh $(CURDIR)/target 2>/dev/null || printf '(missing)\n'

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
	@# cache-sweep operates on the resolved target dir and the owner's
	@# gate-cbuild tree only. Legacy cargo-default-target / .cache/cargo /
	@# .cache/sparse-cargo-home / orphan target dirs are outside this lane.
	@# Residual target/ holds only cargo doc and mdbook output.
	@echo "Pre-sweep size: $$(du -sh $(REPO_CACHE_ROOT) 2>/dev/null | cut -f1)"
	@echo "Sweeping $(CACHE_SWEEP_TARGET_DIR) (keep artifacts accessed in last $(CACHE_SWEEP_KEEP_DAYS) days)..."
	@CARGO_TARGET_DIR=$(CACHE_SWEEP_TARGET_DIR) cargo sweep --time $(CACHE_SWEEP_KEEP_DAYS) . || echo "(skip: target absent or not a cargo project)"
# Conditional gate-cbuild debug wipe: only remove directories where the
# most recent file was modified more than CACHE_SWEEP_DEBUG_KEEP_DAYS
# days ago. Skips wipe entirely if CACHE_SWEEP_DEBUG_KEEP_DAYS=0. Each
# removal passes through guarded_rm, which refuses the owner's tree from
# a worktree unless REPO_ALLOW_SHARED_CLEAN=1.
	@if [ "$(CACHE_SWEEP_DEBUG_KEEP_DAYS)" -gt 0 ]; then \
	    for d in $(CACHE_SWEEP_CBUILD_ROOT)/*/debug; do \
	        if [ -d "$$d" ]; then \
	            most_recent=$$(find "$$d" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 | cut -d. -f1); \
	            if [ -z "$$most_recent" ]; then continue; fi; \
	            age_days=$$(( ( $$(date +%s) - $$most_recent ) / 86400 )); \
	            if [ "$$age_days" -gt "$(CACHE_SWEEP_DEBUG_KEEP_DAYS)" ]; then \
	                SIZE=$$(du -sh "$$d" 2>/dev/null | cut -f1); \
	                echo "Removing stale gate-cbuild debug ($$SIZE, $${age_days}d old) at $$d ..."; \
	                ( $(call guarded_rm,$$d) ) || echo "[cache-sweep] skipped $$d"; \
	            else \
	                echo "Keeping gate-cbuild debug at $$d (last touched $${age_days}d ago)"; \
	            fi; \
	        fi; \
	    done; \
	fi
	@echo "Post-sweep size: $$(du -sh $(REPO_CACHE_ROOT) 2>/dev/null | cut -f1)"

cache-sweep-soft:
	@$(MAKE) -s cache-sweep
	@TOTAL=0; for d in $(CACHE_ACCOUNT_DIRS); do \
	    MB=$$(du -sm "$$d" 2>/dev/null | cut -f1 || printf '0'); TOTAL=$$((TOTAL + $${MB:-0})); \
	done; \
	CBUILD_MB=$$(du -sm $(CACHE_SWEEP_CBUILD_ROOT) 2>/dev/null | cut -f1 || printf '0'); \
	LIMIT=$${CACHE_SWEEP_PRESSURE_TARGET_MB:-$${CACHE_CHECK_SOFT_MB:-153600}}; \
	if [ "$$TOTAL" -gt "$$LIMIT" ] && [ "$${CBUILD_MB:-0}" -gt 0 ] && [ -d $(CACHE_SWEEP_CBUILD_ROOT) ]; then \
	    printf '[cache-sweep-soft] size pressure: %dMB > %dMB; removing regenerable gate-cbuild intermediates (%dMB)\n' "$$TOTAL" "$$LIMIT" "$$CBUILD_MB"; \
	    $(call guarded_rm,$(CACHE_SWEEP_CBUILD_ROOT)); \
	else \
	    printf '[cache-sweep-soft] no size-pressure purge needed (total=%dMB limit=%dMB gate-cbuild=%dMB)\n' "$$TOTAL" "$$LIMIT" "$${CBUILD_MB:-0}"; \
	fi
	@rm -f "$(CACHE_CHECK_SENTINEL)"
	@$(MAKE) -s cache-check-force

# cache-sweep-dry-run: show what would be removed without removing.
.PHONY: cache-sweep-dry-run
cache-sweep-dry-run:
	@echo "DRY RUN: would sweep with --time $(CACHE_SWEEP_KEEP_DAYS) days"
	@echo "=== cargo-sweep dry-run on $(CACHE_SWEEP_TARGET_DIR) ==="
	@CARGO_TARGET_DIR=$(CACHE_SWEEP_TARGET_DIR) cargo sweep --time $(CACHE_SWEEP_KEEP_DAYS) --dry-run . || true
	@echo "=== gate-cbuild debug dirs under $(CACHE_SWEEP_CBUILD_ROOT) older than $(CACHE_SWEEP_DEBUG_KEEP_DAYS) days ==="
	@for d in $(CACHE_SWEEP_CBUILD_ROOT)/*/debug; do \
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
	@for d in $(REPO_LOCAL_CACHE_ROOT)/exp-*-target; do \
	    [ -e "$$d" ] || continue; \
	    $(call guarded_rm,$$d); \
	done
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
# The sentinel path is set in mk/cache_roots.mk under the worktree-local
# target-dir, so one checkout's memoized result never answers for another.
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
	@# Cache accounting sums CACHE_ACCOUNT_DIRS from mk/cache_roots.mk: the
	@# resolved target dir, the owner's gate-target and gate-cbuild trees,
	@# and residual target/ (reserved for cargo doc and mdbook output).
	@TOTAL=0; for d in $(CACHE_ACCOUNT_DIRS); do \
	    MB=$$(du -sm "$$d" 2>/dev/null | cut -f1 || printf '0'); TOTAL=$$((TOTAL + $${MB:-0})); \
	done; \
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

pre-push-gate-strict: validate-repository
	@echo "DEPRECATED: make pre-push-gate-strict is a compatibility alias for make validate-repository."

smoke: check rust-smoke
	@echo "OK: smoke lane passed."

validate-rust-integrity: require-ci-validation-authority registry-validation-tools \
                         validate-rust-integrity-claims \
                         validate-rust-integrity-test-inventory \
                         validate-rust-integrity-typed-policy
	@echo "OK: Rust integrity validation passed."

validate-rust-integrity-claims: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/claims-verify --check providers

validate-rust-integrity-test-inventory: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/test-inventory --check

validate-rust-integrity-typed-policy: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/registry-check --typed-policy error

registry-control-plane-gate-readonly: validate-governance
	@echo "DEPRECATED: make registry-control-plane-gate-readonly is a compatibility alias for make validate-governance."

integrity:
	$(MAKE) verify-pantheon-physicsforge-mapping
	$(MAKE) verify-pantheon-physicsforge-license-headers
	$(MAKE) verify-pantheon-physicsforge-overflow
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-embedded
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin verify-registry-mirror-freshness -- --out-dir "$(MARKDOWN_EXPORT_OUT_DIR)" --emit-legacy --legacy-claims-sync true
	@echo "OK: integrity lane passed."

integrity-rust: validate-rust-integrity
	@echo "DEPRECATED: make integrity-rust is a compatibility alias for make validate-rust-integrity."

test-inventory: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/test-inventory --check

math-verify: rust-regression
	@echo "OK: math validation suite complete. See docs/MATH_VALIDATION_REPORT.md"

rust-test: rust-regression
	@echo "OK: rust-test is an alias for rust-regression."

rust-clippy: require-ci-validation-authority
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

rust-regression: require-ci-validation-authority rust-clippy
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --workspace --exclude algebra_analysis --exclude gr_core
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy -p algebra_analysis -p gr_core
	@echo "OK: Rust regression lane passed."

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
	perf stat -e $${PERF_EVENTS:-cycles:u,instructions:u,branches:u,branch-misses:u} -r $${PERF_RUNS:-3} $(REPO_CARGO_TARGET_DIR)/validation/x87-strategy-bench \
		--len $${LEN:-262144} \
		--repeats $${REPEATS:-5} \
		--worker-counts $${WORKER_COUNTS:-1,2,4,6} \
		--output $${OUT:-reports/benchmarks/x87_strategy_perf.csv} \
		--summary $${SUMMARY:-reports/benchmarks/x87_strategy_perf.md}
	@echo "OK: x87 strategy perf-stat sweep completed."

x87-strategy-hyperfine:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_algebra --bin x87-strategy-bench
	hyperfine --shell=none --warmup $${WARMUP:-1} --runs $${RUNS:-5} \
		'$(REPO_CARGO_TARGET_DIR)/validation/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 1 --output /tmp/x87_strategy_hyperfine_1.csv' \
		'$(REPO_CARGO_TARGET_DIR)/validation/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 2 --output /tmp/x87_strategy_hyperfine_2.csv' \
		'$(REPO_CARGO_TARGET_DIR)/validation/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 4 --output /tmp/x87_strategy_hyperfine_4.csv' \
		'$(REPO_CARGO_TARGET_DIR)/validation/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 6 --output /tmp/x87_strategy_hyperfine_6.csv'
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
		$(REPO_CARGO_TARGET_DIR)/validation/x87-givens-microbench \
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
		$(REPO_CARGO_TARGET_DIR)/validation/cuda-precision-bench \
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
		$(REPO_CARGO_TARGET_DIR)/validation/cuda-precision-bench \
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
		$(REPO_CARGO_TARGET_DIR)/validation/cpu-lbm-bench \
		--grids $${GRIDS:-64} --workloads $${WORKLOADS:-bgk} \
		--output /dev/null \
		2> $${COUNTERS_OUT:-reports/benchmarks/cpu_lbm_perf.stat}
	@echo "OK: perf stat saved to reports/benchmarks/cpu_lbm_perf.stat"

cpu-bench-cachegrind:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_physics --bin cpu-lbm-bench
	@mkdir -p reports/benchmarks
	valgrind --tool=cachegrind \
		--cachegrind-out-file=$${CGOUT:-reports/benchmarks/cachegrind.out.cpu_lbm} \
		$(REPO_CARGO_TARGET_DIR)/validation/cpu-lbm-bench \
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
	perf stat -e $${PERF_EVENTS:-cycles:u,instructions:u,branches:u,branch-misses:u} -r $${PERF_RUNS:-3} $(REPO_CARGO_TARGET_DIR)/validation/jacobi-backend-sweep \
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
	# cargo-deny discovers deny.toml from the workspace root. cargo-deny 0.20
	# treats --config as a value for a different command and rejects it here.
	$(CARGO_ENV) cargo deny check --show-stats --hide-inclusion-graph advisories bans licenses sources
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

EXPERIMENT_MANIFEST ?=
experiment-manifest-verify:
	@test -n "$(EXPERIMENT_MANIFEST)" || (echo "EXPERIMENT_MANIFEST=<path> is required"; exit 2)
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin experiment-manifest -- verify "$(EXPERIMENT_MANIFEST)"

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
	$(CARGO_ENV) cargo run --profile validation -p gororoba_cli_data --bin markdown-registry -- promote-research-narratives

registry-bootstrap-research-narratives: registry-normalize-research-narratives
	@echo "Research narratives markdown->TOML bootstrap completed."

registry-normalize-book-docs:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- normalize-book-docs --bootstrap-from-markdown

registry-bootstrap-book-docs: registry-normalize-book-docs
	@echo "mdBook markdown->TOML bootstrap completed."

registry-normalize-docs-root-narratives:
	$(CARGO_ENV) cargo run --profile validation -p gororoba_cli_data --bin markdown-registry -- promote-docs-root-narratives

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
# The CLI validates the retained corpus; it has no writable corpus
# projection because the registered TOML lanes remain the durable source.
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-corpus

registry-toml-inventory: registry-markdown-corpus
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- build-toml-inventory

registry-markdown-origin-audit: registry-markdown-inventory
# The owner map is registered state. `markdown-registry register` is the
# only mutation path, so a legacy "build" label verifies inventory coverage
# instead of reconstructing provenance from Markdown text.
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-inventory-toml-first

registry-markdown-owner-map: registry-markdown-origin-audit
# Preserve the legacy target name while validating the registered owner map.
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-owner-map

registry-embedded-markdown:
# Embedded Markdown has no writable projection in the current CLI.
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-embedded

registry-verify-embedded-markdown:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-embedded

registry-verify-markdown-inventory:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-inventory-toml-first

# The retired origin audit is represented by the registered inventory and
# owner-map checks. Keep the legacy target as an inventory verification alias.
registry-verify-markdown-origin: registry-verify-markdown-inventory

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

registry-verify-strict-toml-batch1: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/semantic-atoms --verify --repo-root .

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

registry-verify-strict-toml-batch2: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/evidence-provenance --verify --repo-root .

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
#   make registry-integrity
# to regenerate registry/schema_signatures.toml before committing.
# Registry validation will fail on content_sha mismatch otherwise.
registry-integrity: $(REGISTRY_INTEGRITY_CACHE)
	$(REGISTRY_INTEGRITY_CACHE) --repo-root .

validate-registry-integrity: require-ci-validation-authority $(REGISTRY_INTEGRITY_CACHE)
	$(REGISTRY_INTEGRITY_CACHE) --verify --repo-root .
	@echo "OK: registry consistency records are current."

integrity-resolution: registry-integrity
	@echo "DEPRECATED: make integrity-resolution is a compatibility alias for make registry-integrity."

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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- l2-delay-baseline --start-date 2016-08-29 --n-days 7

ablation-baseline-random:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- random-trilinear --start-date 2016-08-29 --n-days 7 --n-draws 100 --base-seed 1000

ablation-baseline-sparse:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- random-trilinear-sparse --start-date 2016-08-29 --n-days 7 --n-draws 100 --base-seed 2000

ablation-baseline-commutator:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- commutator-baseline --start-date 2016-08-29 --n-days 7

ablation-baseline-pca:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- pca-variance-baseline --start-date 2016-08-29 --n-days 7 --pca-window 15

ablation-axis-a:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- r16-ablation --start-date 2016-08-29 --n-days 7
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- r64-ablation --start-date 2016-08-29 --n-days 7

ablation-axis-b:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- lag-depth-sweep --start-date 2016-08-29 --n-days 7

ablation-window-sensitivity:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- window-sensitivity --start-date 2016-08-29 --n-days 7

ablation-mad-decorrelation:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- themis-staples-labeled --start-date 2016-08-29 --n-days 7
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- themis-staples-labeled --start-date 2016-08-29 --n-days 7 --decorrelated-mad --out-json data/output/heliosphere/ablations/themis_staples_labeled_decorrelated_mad_eval.json

ablation-baselines: ablation-baseline-l2 ablation-baseline-random ablation-baseline-sparse ablation-baseline-commutator ablation-baseline-pca

voyager-heliopause-v1:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- voyager-heliopause --spacecraft v1 --window-days 20

voyager-heliopause-v2:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- voyager-heliopause --spacecraft v2 --window-days 20

ablation-all: ablation-baselines ablation-axis-a ablation-axis-b ablation-window-sensitivity ablation-mad-decorrelation

registry-strict-toml-batch3-build:
	$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_governance --bin registry-integrity
	$(REPO_CARGO_TARGET_DIR)/validation/registry-integrity --repo-root .

registry-verify-schema-signatures:
	$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify schema-signatures

registry-verify-crossrefs:
	$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify crossrefs

registry-verify-dataset-label-aliases:
	$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify dataset-label-aliases

registry-verify-external-source-operational-contracts:
	$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_data --bin governance-verify
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify external-source-operational-contracts

registry-verify-strict-toml-batch3: registry-validation-tools
	$(REGISTRY_INTEGRITY_CACHE) --verify --repo-root .
	$(GOVERNANCE_VERIFY_CACHE) validate-all

registry-strict-toml-batch3: registry-verify-strict-toml-batch3
	@echo "OK: registry-integrity registry lane complete (legacy batch3 compatibility)."

registry-build-integrity-resolution: registry-integrity
	@echo "DEPRECATED: make registry-build-integrity-resolution is a compatibility alias for make registry-integrity."

registry-verify-integrity-resolution: validate-registry-integrity
	@echo "DEPRECATED: make registry-verify-integrity-resolution is a compatibility alias for make validate-registry-integrity."

registry-integrity-resolution-gate: validate-registry
	@echo "DEPRECATED: make registry-integrity-resolution-gate is a compatibility alias for make validate-registry."

registry-wave5-batch3-build: registry-strict-toml-batch3-build
	@echo "DEPRECATED: make registry-wave5-batch3-build is a legacy alias. Use make registry-integrity."

registry-verify-wave5-batch3: registry-verify-strict-toml-batch3
	@echo "DEPRECATED: make registry-verify-wave5-batch3 is a legacy alias. Use make validate-registry-integrity."

registry-wave5-batch3: validate-registry
	@echo "DEPRECATED: make registry-wave5-batch3 is a legacy alias. Use make validate-registry."

registry-strict-toml-batch4-build:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin execution-planning -- --repo-root .

registry-verify-strict-toml-batch4: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/execution-planning --verify --repo-root .
	$(GOVERNANCE_VERIFY_CACHE) validate-all
	$(MARKDOWN_REGISTRY_CACHE) verify-all

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

validate-registry: require-ci-validation-authority registry-validation-tools \
                   validate-registry-control-plane \
                   validate-registry-project-counter \
                   validate-registry-markdown \
                   validate-registry-governance \
                   validate-registry-semantic-atoms \
                   validate-registry-evidence-provenance \
                   validate-registry-integrity \
                   validate-registry-execution-planning
	@echo "OK: registry validation completed in one tool session."

validate-registry-control-plane: registry-validation-tools
	$(PROVENANCE_CACHE) --repo-root . verify-control-plane --verify-compat-exports

validate-registry-project-counter: registry-validation-tools
	$(PROJECT_COUNTER_CACHE) --check

validate-registry-markdown: registry-validation-tools
	$(MARKDOWN_REGISTRY_CACHE) verify-all

validate-registry-governance: registry-validation-tools
	$(GOVERNANCE_VERIFY_CACHE) validate-all

validate-registry-semantic-atoms: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/semantic-atoms --verify --repo-root .

validate-registry-evidence-provenance: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/evidence-provenance --verify --repo-root .

validate-registry-execution-planning: registry-validation-tools
	$(VALIDATION_TOOLS_DIR)/execution-planning --verify --repo-root .

registry-acceptance-gate-readonly: validate-registry
	@echo "DEPRECATED: make registry-acceptance-gate-readonly is a compatibility alias for make validate-registry."

registry-acceptance-gate: validate-registry
	@echo "DEPRECATED: make registry-acceptance-gate is a compatibility alias for make validate-registry."

registry-wave5: validate-registry
	@echo "DEPRECATED: make registry-wave5 is a legacy alias. Use make validate-registry."

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

# Use validation here instead of dev/cg_clif: gororoba_cli_data pulls faer/pulp
# through several data tools, and cg_clif still ICEs on AVX f32x8 lowering in that
# lane on nightly 2026-04-05.
# registry-build is NOT a prerequisite here. It rebuilds the canonical SQLite
# from the compatibility TOMLs, and REGISTRY_SOURCES lists none of
# claim_transitions.toml, claim_relations.toml or the claim_revisions lane, so
# the importer has nowhere to read them from and recreates those tables empty.
# The append-only triggers do not catch it because nothing issues a DELETE: the
# tables are dropped and recreated. Since export-control-plane leaves every
# listed TOML newer than the database, the file rule at
# registry/canonical/control_plane.sqlite3 fires on the very next run of this
# target and silently discards the entire adjudication history. The database is
# the canonical source and the TOMLs are its exports, so the dependency belongs
# in that direction only. Run `make registry-build` deliberately when importing
# hand-authored TOML, never as a side effect of exporting.
registry-export-markdown:
# Export the SQLite control plane before any compatibility-TOML consumer runs.
# registry-refresh derives the Markdown and TOML lanes from those exports.
# Artifact scrolls derive structured corpora from the refreshed narrative
# registry. Evidence provenance then derives proof skeletons and derivation
# records from that complete registry surface. registry-integrity signs the
# complete derived surface before the mirror emitter derives Rust mirrors.
	$(CARGO_ENV) cargo run --release -p gororoba_cli_provenance --bin provenance -- export-control-plane
	$(MAKE) registry-refresh
	$(MAKE) registry-artifact-scrolls
	$(MAKE) registry-build-evidence-provenance
	$(MAKE) registry-integrity
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
	$(CARGO_ENV) cargo build --profile validation -p gororoba_cli_data --bin verify-registry-mirror-freshness --bin governance-verify --bin registry-emit; \
	$(REPO_CARGO_TARGET_DIR)/validation/verify-registry-mirror-freshness \
		--out-dir "crates/data_core/src/registry_mirrors" $$legacy_flag --legacy-claims-sync $$claims_value
	$(MAKE) registry-verify-markdown-toml-first
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify markdown-headers
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify markdown-parity
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify mirror-immutability
	$(REPO_CARGO_TARGET_DIR)/validation/governance-verify claim-ticket-mirrors

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
	$(DOCS_CARGO_ENV) cargo doc --locked --keep-going --workspace $(DOCS_FEATURE_FLAGS) --no-deps --document-private-items

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

.PHONY: docs-book-source
docs-book-source:
	$(CARGO_ENV) cargo run --locked --profile validation -p gororoba_cli_data --bin registry-emit -- book-docs-legacy

docs-book: docs-book-source
	@command -v $(MD_BOOK) >/dev/null 2>&1 || { echo "ERROR: mdbook not found. Run: cargo install --locked --force mdbook"; exit 1; }
	@rm -rf "$(DOCS_BOOK_DIR)"
	@mkdir -p "$(DOCS_BOOK_DIR)"
	$(MD_BOOK) build docs/book -d "$(DOCS_BOOK_DIR)"
	@echo "OK: mdBook staged to $(DOCS_BOOK_DIR)."

docs-site: docs-rustdoc docs-book-source
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

docs-redirect-check: $(REPO_UTILITIES_BIN)
	$(REPO_UTILITIES_BIN) docs-redirect-check $(DOCS_SITE_DIR)

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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo -- stacking-manga \
		--rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
		--dapall data/external/manga/dapall_selection.csv \
		--cd-dim 16 --csv data/results/e183/manga_stack_D16.csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo -- stacking-manga \
		--rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
		--dapall data/external/manga/dapall_selection.csv \
		--cd-dim 64 --csv data/results/e183/manga_stack_D64.csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo -- stacking-manga \
		--rotcurves data/external/manga/rotcurves/manga_rotcurves_all.csv \
		--dapall data/external/manga/dapall_selection.csv \
		--cd-dim 256 --csv data/results/e183/manga_stack_D256.csv
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin harmonic-halo -- stacking-manga \
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

rocq-project-check:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin rocq-project-audit -- --repo-root .

rocq-makefile-check: rocq-project-check
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
	@$(call guarded_rm,$(REPO_CARGO_TARGET_DIR))

# clean-builds removes this checkout's own build state. In shared mode the
# owner's gate-cbuild tree stays unless REPO_ALLOW_SHARED_CLEAN=1, because a
# sibling worktree may be compiling into it.
clean-builds:
	@$(call guarded_rm,$(CURDIR)/target)
	@$(call guarded_rm,$(REPO_LOCAL_CACHE_ROOT)/cargo-default-target)
	@$(call guarded_rm,$(REPO_CARGO_TARGET_DIR))
	@$(call guarded_rm,$(REPO_CARGO_BUILD_DIR))
	@$(call guarded_rm,$(REPO_TMP_CARGO_ROOT))
	@$(call guarded_rm,$(REPO_TMPDIR)/open_gororoba-cargo-build)
	@for d in $(REPO_TMPDIR)/open_gororoba_*_target $(REPO_TMPDIR)/open_gororoba-cargo-build-*; do \
	    [ -e "$$d" ] || continue; \
	    $(call guarded_rm,$$d); \
	done
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
	@$(call guarded_rm,$(REPO_CARGO_HOME))
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
REPO_AUDIT_BASELINE ?= data/output/audit/2026-07-09/repo_audit_anchored_2026_07_10.toml
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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- feature-cube --window full-heliosphere --out-csv data/output/heliosphere/full_feature_cube.csv
	@echo "Generating quench scan from full cube (including MMS)..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin heliosphere -- quench-scan --cube-csv data/output/heliosphere/full_feature_cube.csv --out-csv data/output/heliosphere/takens_quench_scan.csv


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
	@echo "    make check                CI-only hygiene check"
	@echo "    make check-local          Retired compatibility target; CI-only"
	@echo "    make ansi-check           Verify emoji-blocking UTF-8 character policy"
	@echo "    make ansi-check-strict    Verify UTF-8 policy + fail on <U+....>/<EMOJI+...> placeholders"
	@echo "    make verify-pantheon-physicsforge-mapping Verify migration completeness"
	@echo "    make verify-pantheon-physicsforge-license-headers Verify license headers"
	@echo "    make rust-smoke           Dedicated Rust smoke suites via nextest"
	@echo "    make rust-regression      Full Rust regression lane"
	@echo "    make rust-regression-scoped Retired local compatibility target"
	@echo "    make validate-local       Retired; push a branch to trigger CI"
	@echo "    make validation-resource-contract  Verify CI-only validation boundaries"
	@echo "    make validate-static      Lightweight hygiene and dependency validation"
	@echo "    make validate-static-and-registry  Hygiene plus registry validation"
	@echo "    make validate-comprehensive  Full Rust and dependency validation"
	@echo "    make validate-supply-chain  Dependency and unsafe-source policy validation"
	@echo "    make validate-dataset-experiments  Dataset and experiment invariant validation"
	@echo "    make validate-registry    Registry and evidence validation"
	@echo "    make validate-ci          Single-session CI validation"
	@echo "    make audit-comprehensive  Opt-in broad audit with structured aliases"
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
