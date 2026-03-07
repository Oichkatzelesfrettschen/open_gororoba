# ---- Phony targets ----
.PHONY: help install install-analysis install-astro install-particle install-quantum
.PHONY: test lint lint-all lint-all-stats lint-all-fix-safe check smoke integrity math-verify governance-gate wave6-gate pre-push-gate pre-push-gate-strict hooks-install hooks-install-strict hooks-status synthesis-execution-contract
.PHONY: verify verify-grand verify-c010-c011-theses ascii-check ascii-check-strict terminology-gate doctor doctor-blas provenance patch-pyfilesystem2
.PHONY: rocq-proofs rocq-proofs-check lva-paper
.PHONY: python-smoke python-regression heavy test-inventory
.PHONY: rust-test rust-clippy rust-smoke rust-regression rust-regression-scoped rust-smoke-scoped dep-audit cargo-deny-check mcp-smoke e027-validate studio-run studio-check
.PHONY: pre-push-gate-scoped submodule-sync
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
.PHONY: registry-build-integrity-resolution registry-verify-integrity-resolution registry-integrity-resolution-gate
.PHONY: registry-build-execution-planning registry-verify-execution-planning registry-execution-planning-gate
.PHONY: registry-strict-toml-batch1-build registry-verify-strict-toml-batch1 registry-strict-toml-batch1 registry-wave5-batch1-build registry-verify-wave5-batch1 registry-wave5-batch1
.PHONY: registry-strict-toml-batch2-build registry-verify-strict-toml-batch2 registry-strict-toml-batch2 registry-wave5-batch2-build registry-verify-wave5-batch2 registry-wave5-batch2 registry-acceptance-gate registry-wave5
.PHONY: registry-strict-toml-batch3-build registry-verify-strict-toml-batch3 registry-strict-toml-batch3 registry-wave5-batch3-build registry-verify-wave5-batch3 registry-wave5-batch3
.PHONY: registry-strict-toml-batch4-build registry-verify-strict-toml-batch4 registry-strict-toml-batch4 registry-wave5-batch4-build registry-verify-wave5-batch4 registry-wave5-batch4
.PHONY: registry-verify-schema-signatures registry-verify-crossrefs
.PHONY: registry-verify-typed-policy-error
.PHONY: registry-csv-inventory registry-migrate-legacy-csv registry-verify-legacy-csv
.PHONY: registry-migrate-curated-csv registry-verify-curated-csv registry-csv-scope registry-data
.PHONY: registry-project-csv-split registry-csv-holdings
.PHONY: registry-scroll-project-csv-canonical registry-scroll-project-csv-generated
.PHONY: registry-scroll-external-csv-holding registry-scroll-archive-csv-holding
.PHONY: registry-csv-scroll-pipeline registry-verify-csv-scroll-pipeline
.PHONY: registry-verify-project-csv-split registry-verify-csv-holdings registry-verify-csv-corpus-coverage registry-csv-pipeline-gate registry-wave3
.PHONY: registry-ingest-legacy registry-refresh registry-export-markdown registry-verify-mirrors docs-publish
.PHONY: verify-python-core-algorithms
.PHONY: artifacts artifacts-dimensional artifacts-materials artifacts-boxkites
.PHONY: artifacts-reggiani artifacts-m3 artifacts-motifs artifacts-motifs-big
.PHONY: fetch-data fetch-data-redownload provenance-audit external-redownload-audit semantic-data-validate semantic-data-validate-strict run coq latex
.PHONY: docker-quantum-build docker-quantum-run docker-quantum-shell
.PHONY: clean clean-artifacts clean-all

# Non-cargo make fanout: 75% of logical CPUs, minimum 1.
# Cargo and Rust test runners use a shared worker budget equal to logical threads / 2.
NPROC := $(shell nproc 2>/dev/null || echo 4)
NJOBS := $(shell expr $(NPROC) \* 3 / 4)
WORKER_BUDGET ?= $(shell sh scripts/detect_worker_budget.sh)
CARGO_JOBS ?= $(WORKER_BUDGET)
NEXTEST_TEST_THREADS ?= $(WORKER_BUDGET)
RUST_TEST_THREADS ?= $(WORKER_BUDGET)
RAYON_THREADS ?= $(WORKER_BUDGET)
CARGO_ENV = MAKEFLAGS= MFLAGS= CARGO_MAKEFLAGS= CARGO_BUILD_JOBS=$(CARGO_JOBS) RAYON_NUM_THREADS=$(RAYON_THREADS) RUST_TEST_THREADS=$(RUST_TEST_THREADS)

VENV ?= venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
HOOKS_DIR ?= .githooks
MARKDOWN_EXPORT ?= 0
MARKDOWN_EXPORT_OUT_DIR ?= build/docs/generated
MARKDOWN_EXPORT_EMIT_LEGACY ?= 0
MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC ?= 1
PGO_DIR ?= /tmp/pgo-data
SYNTHESIS_CONTRACT_DATE ?= 2026_02_14
SYNTHESIS_CONTRACT_REPORT ?= reports/synthesis_execution_contract_$(SYNTHESIS_CONTRACT_DATE).toml

# ---- Environment setup ----

venv:
	python3 -m venv $(VENV)
	$(PIP) install -U pip

install: venv
	$(PIP) install -e ".[dev]"

install-analysis: install
	$(PIP) install -e ".[analysis]"

install-astro: install
	$(PIP) install -e ".[astro]"

install-particle: install
	$(PIP) install -e ".[particle]"

install-quantum: install
	$(PIP) install -e ".[quantum]"

# ---- Quality gates ----

python-smoke: install
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONWARNINGS=error $(PYTHON) -m pytest -m smoke tests/ -x -q --tb=short

python-regression: install
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONWARNINGS=error $(PYTHON) -m pytest -m regression tests/ -x -q --tb=short

test: python-regression

lint: install
	$(PYTHON) -m ruff check src/gemini_physics tests

lint-all: install
	$(PYTHON) -m ruff check src

lint-all-stats: install
	$(PYTHON) -m ruff check src --statistics --exit-zero

lint-all-fix-safe: install
	$(PYTHON) -m ruff check src --select W291,W293,I001 --fix

check: lint smoke integrity
	@echo "OK: check suite complete."

# Governance verifier targets
registry-verify-markdown-governance:
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_markdown_governance_removal_policy.py

# Governance acceptance gate: 5 TOML registry checks, run in parallel.
# Prerequisites share registry-markdown-inventory -- GNU Make deduplicates correctly.
governance-gate:
	$(MAKE) -j$(NJOBS) \
	    registry-verify-markdown-inventory \
	    registry-verify-markdown-owner \
	    registry-verify-schema-signatures \
	    registry-verify-crossrefs \
	    registry-verify-markdown-governance
	@echo ""
	@echo "=========================================="
	@echo "GOVERNANCE ACCEPTANCE GATE: PASSED"
	@echo "=========================================="
	@echo "[done] Markdown inventory validated (TOML-first)"
	@echo "[done] Markdown owner map verified"
	@echo "[done] Registry schema signatures checked"
	@echo "[done] Cross-reference integrity verified"
	@echo "[done] Markdown governance removal policy checked"
	@echo ""
	@echo "TOML-first governance checks are operational."
	@echo "=========================================="
	@echo ""
	@echo "To run the full fast validation pipeline:"
	@echo "  make check"

wave6-gate: governance-gate
	@echo "DEPRECATED: make wave6-gate is a legacy alias. Use make governance-gate."

pre-push-gate:
	$(MAKE) check
	$(MAKE) rust-regression
	$(MAKE) governance-gate
	$(MAKE) terminology-gate
	@echo "OK: pre-push gate passed (check + rust-regression + governance + terminology)."

# Strict pre-push: 3-way parallel audit, then scoped gate, then strict ASCII.
pre-push-gate-strict:
	$(MAKE) dep-audit
	$(MAKE) cargo-deny-check
	$(MAKE) mcp-smoke
	$(MAKE) pre-push-gate
	$(MAKE) ascii-check-strict
	@echo "OK: strict pre-push gate passed (dep-audit + cargo-deny + mcp-smoke + pre-push-gate + ascii-check-strict)."

hooks-install:
	@mkdir -p "$(HOOKS_DIR)"
	@chmod +x "$(HOOKS_DIR)/pre-push"
	@git config core.hooksPath "$(HOOKS_DIR)"
	@echo "OK: git hooks installed. core.hooksPath=$$(git config --get core.hooksPath)"
	@echo "Pre-push will run: make pre-push-gate"

hooks-install-strict:
	@mkdir -p "$(HOOKS_DIR)"
	@cp "$(HOOKS_DIR)/pre-push" "$(HOOKS_DIR)/pre-push.bak" 2>/dev/null || true
	@printf '%s\n' \
		'#!/usr/bin/env bash' \
		'set -euo pipefail' \
		'repo_root="$$(git rev-parse --show-toplevel)"' \
		'cd "$$repo_root"' \
		'echo "[pre-push] running ./makew pre-push-gate-strict"' \
		'./makew pre-push-gate-strict' \
		> "$(HOOKS_DIR)/pre-push"
	@chmod +x "$(HOOKS_DIR)/pre-push"
	@git config core.hooksPath "$(HOOKS_DIR)"
	@echo "OK: strict git hook installed. core.hooksPath=$$(git config --get core.hooksPath)"
	@echo "Pre-push will run: make pre-push-gate-strict"

hooks-status:
	@echo "core.hooksPath=$$(git config --get core.hooksPath || echo .git/hooks)"
	@echo "pre-push hook exists? $$(test -f "$(HOOKS_DIR)/pre-push" && echo yes || echo no)"

smoke: install
	$(MAKE) python-smoke
	$(MAKE) rust-smoke

integrity: install
	PYTHONWARNINGS=error $(PYTHON) -m compileall -q src
	$(PYTHON) -m ruff check src/gemini_physics tests
	$(PYTHON) -m ruff check src --statistics --exit-zero
	$(PYTHON) bin/ascii_check.py --check
	$(MAKE) registry-verify-markdown-owner
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_python_core_algorithms_pyo3.py
	$(CARGO_ENV) cargo run -p gororoba_cli_data --bin claims-verify -- --check providers
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_generated_artifacts.py
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_grand_images.py
	$(MAKE) verify-pantheon-physicsforge-mapping
	$(MAKE) verify-pantheon-physicsforge-license-headers
	$(MAKE) verify-pantheon-physicsforge-overflow
	$(MAKE) registry-verify-embedded-markdown
	$(MAKE) test-inventory

test-inventory:
	$(CARGO_ENV) cargo run -p gororoba_cli_data --bin test-inventory -- --check

math-verify: test lint
	@echo "OK: math validation suite complete. See docs/MATH_VALIDATION_REPORT.md"

rust-test: rust-regression
	@echo "OK: rust-test is an alias for rust-regression."

rust-clippy:
	$(CARGO_ENV) cargo clippy --workspace -- -D warnings

rust-smoke:
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) -P smoke -p algebra_core --test smoke_algebra_core -p lbm_3d --test smoke_lbm_3d -p gororoba_engine --test smoke_gororoba_engine
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P smoke -p gr_core --test smoke_gr_core
	@echo "OK: Rust smoke lane passed."

rust-regression: rust-clippy
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --workspace --exclude algebra_analysis --exclude gr_core
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy -p algebra_analysis -p gr_core
	@echo "OK: Rust regression lane passed."

# Scoped Rust regression gate: only affected crates (via ci_affected_crates.py).
# Usage: make rust-regression-scoped  (auto-detects changes vs origin/main)
#        make rust-regression-scoped RUST_SCOPE="-p algebra_core -p gr_core"
rust-regression-scoped:
	$(eval RUST_SCOPE ?= $(shell python3 scripts/ci_affected_crates.py --local 2>/dev/null || echo "--workspace"))
	@if [ -z "$(RUST_SCOPE)" ]; then \
	    echo "SKIP: no Rust-relevant changes detected."; \
	else \
	    echo "[rust-regression-scoped] scope: $(RUST_SCOPE)"; \
	    $(CARGO_ENV) cargo clippy $(RUST_SCOPE) --all-targets -- -D warnings; \
	    if [ "$(RUST_SCOPE)" = "--workspace" ]; then \
	        light_scope="--workspace --exclude algebra_analysis --exclude gr_core"; \
	        heavy_scope="-p algebra_analysis -p gr_core"; \
	    else \
	        light_scope=""; \
	        heavy_scope=""; \
	        prev=""; \
	        for token in $(RUST_SCOPE); do \
	            if [ "$$prev" = "-p" ]; then \
	                case "$$token" in \
	                    algebra_analysis|gr_core) heavy_scope="$$heavy_scope -p $$token" ;; \
	                    *) light_scope="$$light_scope -p $$token" ;; \
	                esac; \
	                prev=""; \
	            elif [ "$$token" = "-p" ]; then \
	                prev="-p"; \
	            fi; \
	        done; \
	    fi; \
	    if [ -n "$$light_scope" ]; then \
	        $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $$light_scope; \
	    fi; \
	    if [ -n "$$heavy_scope" ]; then \
	        $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy $$heavy_scope; \
	    fi; \
	    echo "OK: Rust regression gate passed (scoped: clippy + nextest)."; \
	fi

rust-smoke-scoped: rust-regression-scoped
	@echo "DEPRECATED: make rust-smoke-scoped is a legacy alias. Use make rust-regression-scoped."

# Scoped pre-push: routes Rust/governance to affected scope only.
pre-push-gate-scoped:
	$(MAKE) check
	$(MAKE) rust-regression-scoped
	$(MAKE) terminology-gate
	@echo "OK: scoped pre-push gate passed."

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
	cargo deny check --config deny.toml --show-stats --hide-inclusion-graph advisories bans licenses sources
	@echo "OK: cargo-deny policy gate passed."

mcp-smoke:
	PYTHONWARNINGS=error python3 src/verification/verify_mcp_smoke.py

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
	@python3 -c "import tomli; tomli.loads(open('data/e027/e027_results.toml').read()); print('TOML structure valid')" || (echo "ERROR: results TOML is malformed"; exit 1)
	@echo "OK: E-027 validation passed (binary operational, TOML pipeline functional)."
	@echo "NOTE: Small grid validation may refute Thesis 1 (expected with mock data). Full validation requires 32x32x32 grid."

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
	PYTHONWARNINGS=error python3 src/verification/verify_pantheon_physicsforge_license_consistency.py

verify-pantheon-physicsforge-provenance:
	PYTHONWARNINGS=error python3 src/verification/verify_pantheon_physicsforge_provenance_gate.py

verify-pantheon-physicsforge-mapping:
	PYTHONWARNINGS=error python3 src/verification/verify_pantheon_physicsforge_mapping_completeness.py

verify-pantheon-physicsforge-license-headers:
	PYTHONWARNINGS=error python3 src/verification/verify_pantheon_physicsforge_license_headers.py

verify-pantheon-physicsforge-overflow:
	PYTHONWARNINGS=error python3 src/verification/verify_pantheon_physicsforge_overflow_tracker.py

seed-pantheon-physicsforge-sqlite:
	PYTHONWARNINGS=error python3 src/scripts/analysis/seed_pantheon_physicsforge_migration_sqlite.py

registry-knowledge:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_knowledge_sources_registry.py

registry-governance: registry-knowledge
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_markdown_governance_registry.py

registry-migrate-corpus: registry-knowledge
	PYTHONWARNINGS=error python3 src/scripts/analysis/migrate_markdown_corpus_to_toml.py

registry-normalize-claims:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_claims_support_registries.py --bootstrap-from-markdown

registry-bootstrap-claims-support: registry-normalize-claims
	@echo "Claims support markdown->TOML bootstrap completed."

registry-normalize-bibliography:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_bibliography_registry.py --bootstrap-from-markdown

registry-bootstrap-bibliography: registry-normalize-bibliography
	@echo "Bibliography markdown->TOML bootstrap completed."

registry-normalize-external-sources:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_external_sources_registry.py --bootstrap-from-markdown

registry-bootstrap-external-sources: registry-normalize-external-sources
	@echo "External sources markdown->TOML bootstrap completed."

registry-normalize-research-narratives:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_research_narratives_registry.py --bootstrap-from-markdown

registry-bootstrap-research-narratives: registry-normalize-research-narratives
	@echo "Research narratives markdown->TOML bootstrap completed."

registry-normalize-book-docs:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_book_docs_registry.py --bootstrap-from-markdown

registry-bootstrap-book-docs: registry-normalize-book-docs
	@echo "mdBook markdown->TOML bootstrap completed."

registry-normalize-docs-root-narratives:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_docs_root_narratives_registry.py --bootstrap-from-markdown

registry-bootstrap-docs-root-narratives: registry-normalize-docs-root-narratives
	@echo "Root docs markdown->TOML bootstrap completed."

registry-normalize-reports-narratives:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_reports_narratives_registry.py --bootstrap-from-markdown

registry-bootstrap-reports-narratives: registry-normalize-reports-narratives
	@echo "Reports markdown->TOML bootstrap completed."

registry-normalize-docs-convos:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_docs_convos_registry.py --bootstrap-from-markdown

registry-bootstrap-docs-convos: registry-normalize-docs-convos
	@echo "docs/convos markdown->TOML bootstrap completed."

registry-normalize-data-artifact-narratives:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_data_artifact_narratives_registry.py --bootstrap-from-markdown

registry-bootstrap-data-artifact-narratives: registry-normalize-data-artifact-narratives
	@echo "data/artifacts narrative markdown->TOML bootstrap completed."

registry-normalize-entrypoint-docs:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_entrypoint_docs_registry.py --bootstrap-from-markdown

registry-bootstrap-entrypoint-docs: registry-normalize-entrypoint-docs
	@echo "Entrypoint markdown bootstrap into registry/entrypoint_docs.toml completed."

registry-normalize-narratives:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_narrative_overlays.py

registry-normalize-operational-narratives:
	PYTHONWARNINGS=error python3 src/scripts/analysis/normalize_operational_narrative_overlays.py

registry-ingest-legacy: registry-normalize-narratives registry-normalize-operational-narratives
	@echo "Legacy markdown -> TOML ingest completed."

registry-refresh: registry-migrate-corpus registry-ingest-legacy registry-governance

registry-knowledge-atoms:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_structured_knowledge_atoms.py

registry-verify-knowledge-atoms: registry-knowledge-atoms
	PYTHONWARNINGS=error python3 src/verification/verify_structured_knowledge_atoms.py

registry-artifact-scrolls: registry-knowledge-atoms
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_artifact_scrolls_registry.py

registry-verify-artifact-scrolls: registry-artifact-scrolls
	PYTHONWARNINGS=error python3 src/verification/verify_artifact_scrolls_registry.py

registry-markdown-inventory:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_markdown_inventory_registry.py

registry-markdown-corpus: registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_markdown_corpus_registry.py

registry-toml-inventory: registry-markdown-corpus
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_toml_inventory_registry.py

registry-markdown-origin-audit: registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_markdown_origin_audit.py

registry-embedded-markdown:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_embedded_markdown_structured_registry.py

registry-verify-embedded-markdown: registry-embedded-markdown
	PYTHONWARNINGS=error python3 src/verification/verify_embedded_markdown_structured_registry.py

registry-verify-markdown-inventory: registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_inventory_toml_first.py

registry-verify-markdown-origin: registry-markdown-origin-audit
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_origin_audit.py

registry-verify-markdown-owner: registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_owner_map.py

registry-verify-markdown-toml-first: registry-verify-markdown-inventory registry-verify-markdown-owner
	@echo "OK: markdown TOML-first owner/inventory gates verified."

registry-verify-control-plane: registry-markdown-corpus registry-toml-inventory registry-verify-markdown-origin registry-verify-markdown-owner registry-verify-knowledge-atoms registry-verify-artifact-scrolls
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_corpus_registry.py
	PYTHONWARNINGS=error python3 src/verification/verify_toml_inventory_registry.py

registry-control-plane-gate: registry-verify-control-plane
	@echo "OK: control-plane registry lane complete."

registry-verify-wave4: registry-verify-control-plane
	@echo "DEPRECATED: make registry-verify-wave4 is a legacy alias. Use make registry-verify-control-plane."

registry-wave4: registry-control-plane-gate
	@echo "DEPRECATED: make registry-wave4 is a legacy alias. Use make registry-control-plane-gate."

registry-strict-toml-batch1-build:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_registry_semantic_atoms.py
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_markdown_payload_registries.py

registry-verify-strict-toml-batch1: registry-strict-toml-batch1-build
	PYTHONWARNINGS=error python3 src/verification/verify_registry_semantic_atoms.py

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
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_registry_evidence_provenance.py

registry-verify-strict-toml-batch2: registry-strict-toml-batch2-build
	PYTHONWARNINGS=error python3 src/verification/verify_registry_evidence_provenance.py

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

registry-strict-toml-batch3-build:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_registry_integrity_resolution.py

registry-verify-schema-signatures: registry-strict-toml-batch3-build
	PYTHONWARNINGS=error python3 src/verification/verify_registry_schema_signatures.py

registry-verify-crossrefs:
	PYTHONWARNINGS=error python3 src/verification/verify_registry_crossrefs.py

registry-verify-strict-toml-batch3: registry-strict-toml-batch3-build
	PYTHONWARNINGS=error python3 src/verification/verify_registry_integrity_resolution.py
	PYTHONWARNINGS=error python3 src/verification/verify_registry_schema_signatures.py
	PYTHONWARNINGS=error python3 src/verification/verify_registry_crossrefs.py

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
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_registry_execution_planning.py

registry-verify-strict-toml-batch4: registry-strict-toml-batch4-build registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/verification/verify_registry_execution_planning.py
	PYTHONWARNINGS=error python3 src/verification/verify_registry_crossrefs.py
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_inventory_toml_first.py
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_owner_map.py

registry-strict-toml-batch4: registry-verify-strict-toml-batch4
	@echo "OK: execution-planning registry lane complete (legacy wave5-batch4 compatibility)."

registry-build-execution-planning: registry-strict-toml-batch4-build

registry-verify-execution-planning: registry-verify-strict-toml-batch4

registry-execution-planning-gate: registry-strict-toml-batch4

registry-wave5-batch4-build: registry-strict-toml-batch4-build
	@echo "DEPRECATED: make registry-wave5-batch4-build is a legacy alias. Use make registry-build-execution-planning."

registry-verify-wave5-batch4: registry-verify-strict-toml-batch4
	@echo "DEPRECATED: make registry-verify-wave5-batch4 is a legacy alias. Use make registry-verify-execution-planning."

registry-wave5-batch4: registry-strict-toml-batch4
	@echo "DEPRECATED: make registry-wave5-batch4 is a legacy alias. Use make registry-execution-planning-gate."

registry-acceptance-gate:
	$(MAKE) -j$(NJOBS) \
	    registry-semantic-atoms-gate \
	    registry-evidence-provenance-gate \
	    registry-integrity-resolution-gate \
	    registry-execution-planning-gate
	@echo "OK: registry acceptance gate complete."

registry-wave5: registry-acceptance-gate
	@echo "DEPRECATED: make registry-wave5 is a legacy alias. Use make registry-acceptance-gate."

registry-csv-inventory:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_csv_inventory_registry.py

registry-migrate-legacy-csv:
	PYTHONWARNINGS=error python3 src/scripts/analysis/migrate_legacy_csv_to_toml.py

registry-verify-legacy-csv: registry-migrate-legacy-csv
	PYTHONWARNINGS=error python3 src/verification/verify_legacy_csv_toml_parity.py

registry-migrate-curated-csv:
	PYTHONWARNINGS=error python3 src/scripts/analysis/migrate_legacy_csv_to_toml.py \
		--source-glob 'curated/**/*.csv' \
		--out-index registry/curated_csv_datasets.toml \
		--out-dir registry/data/curated_csv \
		--index-table curated_csv_datasets \
		--dataset-prefix CU \
		--corpus-label 'curated CSV'

registry-verify-curated-csv: registry-migrate-curated-csv
	PYTHONWARNINGS=error python3 src/verification/verify_legacy_csv_toml_parity.py \
		--index-path registry/curated_csv_datasets.toml \
		--source-glob 'curated/**/*.csv' \
		--corpus-label 'curated CSV'

registry-project-csv-split:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_project_csv_split_policy_registry.py

registry-csv-holdings:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_csv_holding_registries.py

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
		--dataset-class holding-external

registry-csv-scroll-pipeline: registry-scroll-project-csv-canonical registry-scroll-project-csv-generated registry-scroll-external-csv-holding registry-scroll-archive-csv-holding
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_csv_scroll_pipeline_registry.py

registry-verify-csv-scroll-pipeline: registry-csv-scroll-pipeline
	PYTHONWARNINGS=error python3 src/verification/verify_csv_scroll_pipeline.py

registry-verify-project-csv-split: registry-scroll-project-csv-canonical registry-scroll-project-csv-generated
	PYTHONWARNINGS=error python3 src/verification/verify_legacy_csv_toml_parity.py \
		--index-path registry/project_csv_canonical_datasets.toml \
		--source-manifest registry/manifests/project_csv_canonical_manifest.txt \
		--corpus-label 'project CSV canonical dataset'
	PYTHONWARNINGS=error python3 src/verification/verify_legacy_csv_toml_parity.py \
		--index-path registry/project_csv_generated_artifacts.toml \
		--source-manifest registry/manifests/project_csv_generated_manifest.txt \
		--corpus-label 'project CSV generated artifact'
	$(CARGO_ENV) cargo run --release --bin verify-project-csv-split -- \
		--repo-root .

registry-verify-csv-holdings: registry-csv-holdings registry-scroll-external-csv-holding registry-scroll-archive-csv-holding
	PYTHONWARNINGS=error python3 src/verification/verify_legacy_csv_toml_parity.py \
		--index-path registry/external_csv_holding_datasets.toml \
		--source-manifest registry/manifests/external_csv_holding_manifest.txt \
		--corpus-label 'external CSV holding queue' \
		--coverage-only
	PYTHONWARNINGS=error python3 src/verification/verify_legacy_csv_toml_parity.py \
		--index-path registry/archive_csv_holding_datasets.toml \
		--source-manifest registry/manifests/archive_csv_holding_manifest.txt \
		--corpus-label 'archive CSV holding queue'
	PYTHONWARNINGS=error python3 src/verification/verify_csv_holding_registries.py

registry-verify-csv-corpus-coverage: registry-csv-inventory registry-verify-project-csv-split registry-verify-csv-holdings
	PYTHONWARNINGS=error python3 src/verification/verify_csv_corpus_coverage.py

registry-csv-pipeline-gate: registry-project-csv-split registry-csv-holdings registry-verify-project-csv-split registry-verify-csv-holdings registry-verify-csv-corpus-coverage registry-verify-csv-scroll-pipeline

registry-wave3: registry-csv-pipeline-gate
	@echo "DEPRECATED: make registry-wave3 is a legacy alias. Use make registry-csv-pipeline-gate."

registry-csv-scope: registry-csv-inventory
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_csv_migration_scope_registry.py

registry-data: registry-migrate-legacy-csv registry-migrate-curated-csv registry-csv-pipeline-gate registry-csv-inventory registry-verify-legacy-csv registry-verify-curated-csv registry-csv-scope registry-control-plane-gate
	@echo "OK: CSV data registry lane complete."

registry-export-markdown: registry-refresh
	@legacy_flag="--no-emit-legacy"; \
	if [ "$(MARKDOWN_EXPORT_EMIT_LEGACY)" = "1" ]; then legacy_flag="--emit-legacy"; fi; \
	claims_flag="--legacy-claims-sync"; \
	if [ "$(MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC)" = "0" ]; then claims_flag="--no-legacy-claims-sync"; fi; \
	PYTHONWARNINGS=error python3 src/scripts/analysis/export_registry_markdown_mirrors.py \
		--out-dir "$(MARKDOWN_EXPORT_OUT_DIR)" $$legacy_flag $$claims_flag

registry-verify-mirrors: registry-export-markdown
	MARKDOWN_EXPORT_OUT_DIR="$(MARKDOWN_EXPORT_OUT_DIR)" \
	MARKDOWN_EXPORT_EMIT_LEGACY="$(MARKDOWN_EXPORT_EMIT_LEGACY)" \
	MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC="$(MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC)" \
	PYTHONWARNINGS=error python3 src/verification/verify_registry_mirror_freshness.py
	PYTHONWARNINGS=error $(MAKE) registry-verify-markdown-toml-first
	@if [ "$(MARKDOWN_EXPORT_EMIT_LEGACY)" = "1" ]; then \
		PYTHONWARNINGS=error python3 src/verification/verify_markdown_governance_headers.py; \
		PYTHONWARNINGS=error python3 src/verification/verify_markdown_governance_parity.py; \
		PYTHONWARNINGS=error python3 src/verification/verify_toml_generated_mirror_immutability.py; \
		PYTHONWARNINGS=error python3 src/verification/verify_claim_ticket_mirrors.py; \
	else \
		echo "SKIP: legacy mirror immutability checks disabled in strict markdown-free publish profile."; \
	fi

registry-sync-project-counters:
	$(CARGO_ENV) cargo run --release --bin project-counter-sync

registry: registry-refresh registry-data registry-sync-project-counters
	$(CARGO_ENV) cargo run --release --bin registry-check

registry-verify-typed-policy-error:
	$(CARGO_ENV) cargo run --release --bin registry-check -- --typed-policy error

synthesis-execution-contract:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_synthesis_gate_rollup.py \
		--date-token "$(SYNTHESIS_CONTRACT_DATE)" \
		--report-path "$(SYNTHESIS_CONTRACT_REPORT)"

docs-publish: registry-verify-mirrors
	@echo "OK: TOML-driven markdown mirrors generated and verified for publishing."

terminology-gate:
	python3 bin/terminology_gate.py

ascii-check:
	python3 bin/ascii_check.py --check

ascii-check-strict:
	python3 bin/ascii_check.py --check --strict-placeholders --placeholder-scope-prefix crates/ --placeholder-scope-prefix tests/

verify: install
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_generated_artifacts.py

verify-grand: install
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_grand_images.py

verify-c010-c011-theses:
	PYTHONWARNINGS=error python3 src/verification/verify_c010_c011_theses.py

verify-python-core-algorithms:
	PYTHONWARNINGS=error python3 src/verification/verify_python_core_algorithms_pyo3.py

doctor: install
	$(PYTHON) bin/doctor.py
	sh scripts/detect_native_blas.sh

doctor-blas:
	sh scripts/detect_native_blas.sh

provenance: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-origin-audit -- --out reports/data_origin_audit_$$(date +%F).toml --fail-on-strict-unknown

provenance-audit: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-governance-gate -- --enforce-origin true --enforce-semantic true --enforce-blocked-deadlines true

external-redownload-audit: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin external-redownload-audit -- --out reports/external_redownload_audit_$$(date +%F).toml --backend-order wget,curl,fetch

semantic-data-validate: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-semantic-validate -- --out reports/data_semantic_validate_$$(date +%F).toml

semantic-data-validate-strict: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-semantic-validate -- --fail-on-unverifiable true --out reports/data_semantic_validate_$$(date +%F)_strict.toml

patch-pyfilesystem2: install
	$(PYTHON) bin/patch_pyfilesystem_pkg_resources.py

# ---- Artifact generation ----
#
# Each artifacts-* target produces deterministic output under data/csv/
# and data/artifacts/images/.  All generated files are reproducible from
# source code + pinned dependencies and can be removed with make clean-artifacts.

artifacts: artifacts-motifs artifacts-boxkites artifacts-reggiani artifacts-m3 artifacts-dimensional
	@echo "OK: all core artifacts regenerated."

artifacts-dimensional: install
	PYTHONWARNINGS=error $(PYTHON) src/vis_dimensional_geometry.py

artifacts-materials: install
	PYTHONWARNINGS=error $(PYTHON) src/fetch_materials_jarvis_subset.py --n 200 --seed 0
	PYTHONWARNINGS=error $(PYTHON) src/materials_embedding_experiments.py

artifacts-boxkites: install
	PYTHONWARNINGS=error $(PYTHON) src/export_de_marrais_boxkites.py

artifacts-reggiani: install
	PYTHONWARNINGS=error $(PYTHON) src/export_reggiani_annihilator_stats.py

artifacts-m3: install
	PYTHONWARNINGS=error $(PYTHON) src/export_m3_table.py

artifacts-motifs:
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 16,32 --details
	PYTHONWARNINGS=error $(PYTHON) src/vis_cd_motif_summary.py

artifacts-motifs-big:
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 16,32,64,128 --summary-only
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 256 --max-nodes 5000 --seed 0 --summary-only
	PYTHONWARNINGS=error $(PYTHON) src/vis_cd_motif_summary.py

# ---- Data fetching ----
#
# External datasets are NOT committed to the repo.  These targets download
# them into the locations expected by analysis scripts.

fetch-data: install
	@echo "Fetching external datasets..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin fetch-datasets -- --all --skip-existing --output-dir data/external
	@echo "Refreshing external provenance and source governance..."
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-governance-gate -- --enforce-origin true --enforce-semantic true --enforce-blocked-deadlines true --enforce-gitignore true --enforce-naming true

fetch-data-redownload: install
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

# ---- Coq proofs ----

coq:
	@command -v coqc >/dev/null 2>&1 || { echo "ERROR: coqc not found. See docs/requirements/coq.md"; exit 1; }
	python3 bin/coq_prepare_confine.py curated/01_theory_frameworks/confine_theorems_512.v curated/01_theory_frameworks/confine_theorems_512_axioms.v
	python3 bin/coq_prepare_confine.py curated/01_theory_frameworks/confine_theorems_1024.v curated/01_theory_frameworks/confine_theorems_1024_axioms.v
	python3 bin/coq_prepare_confine.py curated/01_theory_frameworks/confine_theorems_2048.v curated/01_theory_frameworks/confine_theorems_2048_axioms.v
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

# Build proofs, generate paper artifacts, then compile LaTeX
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
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	rm -rf src/*.egg-info

clean-all: clean clean-artifacts
	@command -v cargo-sweep >/dev/null 2>&1 && cargo sweep --time 14 || true
	@rm -rf /tmp/open_gororoba_*_target 2>/dev/null || true
	@echo "Full cleanup complete. Run 'make install && make artifacts' to rebuild."

# ---- Help ----

help:
	@echo "Targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install              Create venv and install (editable, dev deps)"
	@echo "    make install-analysis     Add analysis extras (networkx, ripser, sklearn)"
	@echo "    make install-astro        Add astronomy extras (gwpy, astroquery)"
	@echo "    make install-particle     Add particle-analysis extras (uproot, awkward, vector)"
	@echo "    make install-quantum      Add quantum extras (qiskit, Docker recommended)"
	@echo ""
	@echo "  Quality:"
	@echo "    make python-smoke         Run smoke-marked pytest coverage"
	@echo "    make python-regression    Run regression-marked pytest coverage"
	@echo "    make test                 Alias for python-regression"
	@echo "    make lint                 Ruff check on src/gemini_physics + tests"
	@echo "    make smoke                Composite fast smoke lane (python-smoke + rust-smoke)"
	@echo "    make integrity            Verifier, ASCII, and inventory lane"
	@echo "    make check                lint + smoke + integrity"
	@echo "    make ascii-check          Verify ASCII-only policy"
	@echo "    make ascii-check-strict   Verify ASCII-only policy + fail on <U+....> placeholders in crates/tests"
	@echo "    make verify-pantheon-physicsforge-mapping Verify migration matrix/todo mapping completeness"
	@echo "    make verify-pantheon-physicsforge-license-headers Verify GPL-2.0-only header consistency in migrated files"
	@echo "    make verify-pantheon-physicsforge-overflow Verify overflow tracker max-5-active policy"
	@echo "    make seed-pantheon-physicsforge-sqlite Seed sqlite memoization for migration findings/risks"
	@echo "    make rust-smoke           Dedicated Rust smoke suites via nextest"
	@echo "    make rust-regression      Full Rust regression lane with heavy-crate routing"
	@echo "    make rust-regression-scoped Scoped Rust regression lane for affected crates"
	@echo "    make heavy                Ignored/GPU/research-heavy nextest lane"
	@echo "    make test-inventory       Enforce taxonomy coverage and stale-doc checks"
	@echo "    make mcp-smoke            Re-test configured MCP server parity and startup health"
	@echo "    make cargo-deny-check     Enforce deny.toml (advisories, bans, licenses, sources)"
	@echo "    make registry             Validate TOML registry consistency"
	@echo "    make registry-verify-typed-policy-error Strict registry-check typed-policy lane (--typed-policy error)"
	@echo "    make synthesis-execution-contract Run full synthesis execution contract and emit rollup TOML"
	@echo "    make registry-control-plane-gate Validate markdown/TOML control-plane + atom extraction gates"
	@echo "    make registry-csv-pipeline-gate  Validate project/external/archive CSV scroll pipeline lanes"
	@echo "    make registry-semantic-atoms-gate      Build+verify claims/equation/proof/payload TOML lanes"
	@echo "    make registry-evidence-provenance-gate Build+verify derivation/bibliography/provenance/paragraph lanes"
	@echo "    make registry-integrity-resolution-gate Build+verify contradiction/lacuna/schema/crossref lanes"
	@echo "    make registry-execution-planning-gate  Build+verify experiment/planning/requirements lanes"
	@echo "    make registry-acceptance-gate    Run full registry acceptance gate (semantic-atoms + evidence-provenance + integrity-resolution + execution-planning)"
	@echo "    make registry-verify-schema-signatures Verify critical registry schema signatures"
	@echo "    make registry-verify-crossrefs Verify dangling cross-registry references"
	@echo "    make registry-verify-knowledge-atoms Verify claim/equation/proof atom registries"
	@echo "    make registry-verify-markdown-toml-first Verify markdown owner/inventory TOML-first hard gate"
	@echo "    MARKDOWN_EXPORT=1 make docs-publish Export mirrors in strict mode (out-of-tree, no legacy writes)"
	@echo "  Deprecated legacy aliases (compatibility-only entrypoints):"
	@echo "    make registry-wave5              DEPRECATED: make registry-wave5 is a legacy alias. Use make registry-acceptance-gate."
	@echo "    make registry-wave5-batch1-build DEPRECATED: make registry-wave5-batch1-build is a legacy alias. Use make registry-build-semantic-atoms."
	@echo "    make registry-verify-wave5-batch1 DEPRECATED: make registry-verify-wave5-batch1 is a legacy alias. Use make registry-verify-semantic-atoms."
	@echo "    make registry-wave5-batch1       DEPRECATED: make registry-wave5-batch1 is a legacy alias. Use make registry-semantic-atoms-gate."
	@echo "    make registry-wave5-batch2-build DEPRECATED: make registry-wave5-batch2-build is a legacy alias. Use make registry-build-evidence-provenance."
	@echo "    make registry-verify-wave5-batch2 DEPRECATED: make registry-verify-wave5-batch2 is a legacy alias. Use make registry-verify-evidence-provenance."
	@echo "    make registry-wave5-batch2       DEPRECATED: make registry-wave5-batch2 is a legacy alias. Use make registry-evidence-provenance-gate."
	@echo "    make registry-wave5-batch3-build DEPRECATED: make registry-wave5-batch3-build is a legacy alias. Use make registry-build-integrity-resolution."
	@echo "    make registry-verify-wave5-batch3 DEPRECATED: make registry-verify-wave5-batch3 is a legacy alias. Use make registry-verify-integrity-resolution."
	@echo "    make registry-wave5-batch3       DEPRECATED: make registry-wave5-batch3 is a legacy alias. Use make registry-integrity-resolution-gate."
	@echo "    make registry-wave5-batch4-build DEPRECATED: make registry-wave5-batch4-build is a legacy alias. Use make registry-build-execution-planning."
	@echo "    make registry-verify-wave5-batch4 DEPRECATED: make registry-verify-wave5-batch4 is a legacy alias. Use make registry-verify-execution-planning."
	@echo "    make registry-wave5-batch4       DEPRECATED: make registry-wave5-batch4 is a legacy alias. Use make registry-execution-planning-gate."
	@echo "    make registry-wave4              DEPRECATED: make registry-wave4 is a legacy alias. Use make registry-control-plane-gate."
	@echo "    make registry-verify-wave4       DEPRECATED: make registry-verify-wave4 is a legacy alias. Use make registry-verify-control-plane."
	@echo "    make registry-wave3              DEPRECATED: make registry-wave3 is a legacy alias. Use make registry-csv-pipeline-gate."
	@echo "    make wave6-gate                  DEPRECATED: make wave6-gate is a legacy alias. Use make governance-gate."
	@echo ""
	@echo "  Gates (tiered):"
	@echo "    make governance-gate      5 TOML registry checks (inventory, owner, schema, crossrefs, governance)"
	@echo "    make registry-verify-typed-policy-error Supplemental strict typed-policy contract lane"
	@echo "    make synthesis-execution-contract governance-gate + registry-acceptance-gate + strict typed-policy + project-counter-sync --check"
	@echo "    make terminology-gate     enforce banned-term policy from terminology_standards.toml"
	@echo "    make pre-push-gate        check + rust-regression + governance-gate + terminology-gate"
	@echo "    make pre-push-gate-strict dep-audit + cargo-deny + mcp-smoke + pre-push-gate + ascii-check-strict"
	@echo ""
	@echo "  Artifacts:"
	@echo "    make artifacts            Regenerate all core artifact sets"
	@echo "    make artifacts-motifs     CD motif census (16D, 32D)"
	@echo "    make artifacts-motifs-big CD motif census (64D-256D)"
	@echo "    make artifacts-boxkites   De Marrais boxkite geometry"
	@echo "    make artifacts-reggiani   Reggiani annihilator statistics"
	@echo "    make artifacts-m3         M3 transfer table"
	@echo "    make artifacts-dimensional Dimensional geometry sweeps"
	@echo "    make artifacts-materials  JARVIS subset + embeddings"
	@echo ""
	@echo "  Data:"
	@echo "    make fetch-data           Re-download external datasets via Rust fetchers + strict governance checks"
	@echo "    make fetch-data-redownload Force re-download all fetch-datasets providers (skip-existing=false)"
	@echo "    make provenance           Hash data/external/* + emit data-origin audit report"
	@echo "    make provenance-audit     Enforce strict origin + semantic + blocked-deadline governance gate"
	@echo "    make external-redownload-audit  Run external source coverage/re-download audit (wget->curl->fetch)"
	@echo "    make semantic-data-validate Run lane semantic validators from registry/data_semantic_validators.toml"
	@echo "    make semantic-data-validate-strict Fail on any semantic unverifiable status"
	@echo "    cargo run -p gororoba_cli_data --bin data-governance-gate --     Run fail-closed data governance gate"
	@echo "    cargo run -p gororoba_cli_data --bin data-clean -- --scope reproducible --apply  Rust-native reproducible-data cleanup"
	@echo ""
	@echo "  Verification:"
	@echo "    make verify               Verify artifact schemas"
	@echo "    make verify-grand         Verify grand images"
	@echo "    make math-verify          Full math validation suite"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean                Remove venv, caches, bytecode"
	@echo "    make clean-artifacts      Remove generated CSV/images/HDF5 (keep source data)"
	@echo "    make clean-all            clean + clean-artifacts"
	@echo ""
	@echo "  Other:"
	@echo "    make run                  Run simulations (sedenion, modular, entropy)"
	@echo "    make coq                  Compile Coq proofs"
	@echo "    make latex                Build MASTER_SYNTHESIS.pdf"
	@echo "    make docker-quantum-build Build qiskit-env Docker image"
	@echo "    make docker-quantum-run   Run quantum script in Docker (ARGS=...)"
	@echo "    make docker-quantum-shell Open interactive shell in qiskit-env"
	@echo "    make doctor               Environment diagnostics"
	@echo "    make doctor-blas          Detect native BLAS/LAPACK candidates and Cargo feature mapping"
	@echo "    ./makew <target>          Run make with inherited jobserver env stripped (useful when shells/tools export MAKEFLAGS)"
