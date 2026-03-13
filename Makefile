# ---- Phony targets ----
.PHONY: help install install-analysis install-astro install-particle install-quantum bootstrap-dev
.PHONY: test lint lint-all lint-all-stats lint-all-fix-safe lint-advisory check smoke integrity integrity-rust math-verify governance-gate governance-gate-readonly wave6-gate pre-push-gate pre-push-gate-strict hooks-install hooks-install-strict hooks-status synthesis-execution-contract
.PHONY: verify verify-grand verify-c010-c011-theses ansi-check ansi-check-strict terminology-gate doctor doctor-blas provenance patch-pyfilesystem2
.PHONY: provenance-registry-index provenance-registry-export provenance-registry-verify provenance-registry-doctor provenance-registry-link-audit provenance-registry-recover
.PHONY: rocq-proofs rocq-proofs-check lva-paper
.PHONY: python-smoke python-regression heavy test-inventory verify-no-reports-writes
.PHONY: rust-test rust-clippy rust-smoke rust-regression rust-regression-scoped rust-smoke-scoped dep-audit cargo-deny-check mcp-smoke e027-validate studio-run studio-check profile-tensor-avt x87-strategy-bench x87-strategy-perf x87-strategy-hyperfine x87-strategy-flamegraph x87-givens-microbench x87-givens-microbench-perf jacobi-backend-sweep jacobi-backend-perf jacobi-backend-flamegraph jacobi-backend-samply jacobi-backend-samply-compare
.PHONY: pre-push-gate-scoped submodule-sync gate-local gate-ci-python gate-ci-python-compat gate-ci-rust gate-audit profile-python-toml-inventory
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
.PHONY: registry-build-integrity-resolution registry-verify-integrity-resolution registry-integrity-resolution-gate
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
.PHONY: registry-ingest-legacy registry-refresh registry-export-markdown registry-verify-mirrors docs-publish
.PHONY: verify-python-core-algorithms
.PHONY: artifacts artifacts-dimensional artifacts-materials artifacts-boxkites
.PHONY: artifacts-reggiani artifacts-m3 artifacts-motifs artifacts-motifs-big
.PHONY: fetch-data fetch-data-redownload provenance-audit external-redownload-audit semantic-data-validate semantic-data-validate-strict run rocq latex
.PHONY: docker-quantum-build docker-quantum-run docker-quantum-shell
.PHONY: clean clean-builds clean-artifacts clean-all
.PHONY: run-e183

.NOTPARALLEL: install bootstrap-dev check smoke integrity integrity-rust rust-smoke rust-regression rust-regression-scoped heavy cargo-deny-check gate-local gate-ci-python gate-ci-python-compat gate-ci-rust gate-audit pre-push-gate pre-push-gate-scoped pre-push-gate-strict governance-gate governance-gate-readonly registry-control-plane-gate-readonly registry-acceptance-gate-readonly

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
REPO_CARGO_HOME ?= $(CURDIR)/.cache/cargo-home
REPO_CARGO_TARGET_DIR ?= $(CURDIR)/.cache/gate-target
CARGO_ENV = CARGO_HOME=$(REPO_CARGO_HOME) CARGO_TARGET_DIR=$(REPO_CARGO_TARGET_DIR) MAKEFLAGS= MFLAGS= CARGO_MAKEFLAGS= CARGO_BUILD_JOBS=$(CARGO_JOBS) RAYON_NUM_THREADS=$(RAYON_THREADS) RUST_TEST_THREADS=$(RUST_TEST_THREADS)
PYTEST_WORKERS ?= $(WORKER_BUDGET)
PYTEST_XDIST_ARGS = -p xdist.plugin -n $(PYTEST_WORKERS) --dist worksteal

VENV ?= venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
DEV_STAMP := $(VENV)/.installed-dev
HOOKS_DIR ?= .githooks
MARKDOWN_EXPORT ?= 0
MARKDOWN_EXPORT_OUT_DIR ?= docs/generated
MARKDOWN_EXPORT_EMIT_LEGACY ?= 0
MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC ?= 1
PGO_DIR ?= /tmp/pgo-data
SYNTHESIS_CONTRACT_DATE ?= 2026_02_14
SYNTHESIS_CONTRACT_REPORT ?= reports/synthesis_execution_contract_$(SYNTHESIS_CONTRACT_DATE).toml
GATE_AUDIT_SCRIPT := scripts/gate_audit.py
PROFILE_TIMESTAMP := $(shell date +%Y-%m-%d/%H%M%S)
PROFILE_ROOT ?= reports/gates/profiles/$(PROFILE_TIMESTAMP)

# ---- Environment setup ----

$(VENV)/bin/python3:
	python3 -m venv $(VENV)
	$(PIP) install -U pip

venv: $(VENV)/bin/python3

$(DEV_STAMP): $(VENV)/bin/python3 pyproject.toml
	$(PIP) install -e ".[dev]"
	@touch "$(DEV_STAMP)"

install: $(DEV_STAMP)

bootstrap-dev: install
	@echo "OK: dev bootstrap is current."

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
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONWARNINGS=error $(PYTHON) -m pytest $(PYTEST_XDIST_ARGS) -m "smoke and not requires_ext" tests/ -x -q --tb=short

python-regression: install
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONWARNINGS=error $(PYTHON) -m pytest $(PYTEST_XDIST_ARGS) -m "regression and not requires_ext" tests/ -x -q --tb=short

python-requires-ext: install
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONWARNINGS=error $(PYTHON) -m pytest $(PYTEST_XDIST_ARGS) -m "requires_ext" tests/ -x -q --tb=short

test: python-regression

lint: install
	$(PYTHON) scripts/run_ruff_changed.py

lint-all: install
	$(PYTHON) -m ruff check src tests bin scripts

lint-all-stats: install
	$(PYTHON) -m ruff check src tests bin scripts --statistics --exit-zero

lint-advisory: lint-all-stats
	@echo "OK: advisory lint statistics collected."

lint-all-fix-safe: install
	$(PYTHON) -m ruff check src --select W291,W293,I001 --fix

verify-no-reports-writes: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- no-reports-writes

check: lint python-smoke ansi-check terminology-gate verify-no-reports-writes
	@echo "OK: fast shared check suite complete."

# Governance verifier targets
registry-verify-markdown-governance:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- markdown-removal-policy

governance-gate-readonly:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-inventory-toml-first
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-owner-map
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- schema-signatures
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- crossrefs
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- dataset-label-aliases
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- external-source-operational-contracts
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- markdown-removal-policy
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

gate-local:
	@set -e; \
	scope=""; \
	run_rust="true"; \
	run_governance="true"; \
	echo "[gate-local] determining scope..."; \
	if command -v cargo >/dev/null 2>&1; then \
	    scope_file="$$(mktemp)"; \
	    meta_file="$$(mktemp)"; \
	    $(CARGO_ENV) cargo run -q -p gororoba_cli_data --bin workspace-routing -- --local --verbose 1>"$$scope_file" 2>"$$meta_file" || true; \
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
	$(MAKE) check; \
	if [ "$$run_rust" = "true" ]; then \
	    if [ -z "$$scope" ]; then scope="--workspace"; fi; \
	    echo "[gate-local] rust scope: $$scope"; \
	    if [ -n "$(LOCAL_NEXTEST_TIMING_JSON)" ]; then echo "[gate-local] local nextest timing: $(LOCAL_NEXTEST_TIMING_JSON)"; fi; \
	    $(MAKE) rust-regression-scoped RUST_SCOPE="$$scope" RUST_RUN_HEAVY=0; \
	else \
	    echo "[gate-local] SKIP: no Rust-relevant changes detected."; \
	fi; \
	if [ "$$run_governance" = "true" ]; then \
	    $(MAKE) governance-gate-readonly; \
	else \
	    echo "[gate-local] SKIP: no governance-relevant changes detected."; \
	fi; \
	echo "[gate-local] OK: local gate passed."

pre-push-gate: gate-local
	@echo "OK: pre-push-gate is a compatibility alias for gate-local."

gate-ci-python: install
	$(MAKE) check
	$(MAKE) python-regression
	$(MAKE) integrity
	$(MAKE) governance-gate-readonly
	$(MAKE) registry-control-plane-gate-readonly
	$(MAKE) registry-acceptance-gate-readonly
	@echo "OK: gate-ci-python passed."

gate-ci-python-compat: check
	@echo "OK: gate-ci-python-compat passed."

gate-ci-rust:
	$(MAKE) rust-regression
	$(MAKE) integrity-rust
	$(MAKE) cargo-deny-check
	@echo "OK: gate-ci-rust passed."

gate-audit: install
	PYTHONWARNINGS=error $(PYTHON) $(GATE_AUDIT_SCRIPT)
	@echo "OK: gate-audit completed."

profile-python-toml-inventory: install
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
		'echo "[pre-push] running ./makew gate-local"' \
		'./makew gate-local' \
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

registry-control-plane-gate-readonly: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-corpus
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin markdown-registry -- verify-toml-inventory
	@echo "OK: read-only registry control-plane gate passed."

integrity: install
	PYTHONWARNINGS=error $(PYTHON) -m compileall -q src
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_python_core_algorithms_pyo3.py
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_generated_artifacts.py
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_grand_images.py
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

math-verify: test lint
	@echo "OK: math validation suite complete. See docs/MATH_VALIDATION_REPORT.md"

rust-test: rust-regression
	@echo "OK: rust-test is an alias for rust-regression."

rust-clippy:
	$(CARGO_ENV) cargo clippy --workspace -- -D warnings

rust-smoke:
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) -P smoke -p gororoba_algebra --test smoke_gororoba_algebra -p lbm_3d --test smoke_lbm_3d -p gororoba_engine --test smoke_gororoba_engine
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P smoke -p gr_core --test smoke_gr_core
	@echo "OK: Rust smoke lane passed."

rust-regression: rust-clippy
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --workspace --exclude algebra_analysis --exclude gr_core
	$(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy -p algebra_analysis -p gr_core
	@echo "OK: Rust regression lane passed."

# Scoped Rust regression gate: only affected crates (via workspace-routing).
# Usage: make rust-regression-scoped  (auto-detects changes vs origin/main)
#        make rust-regression-scoped RUST_SCOPE="-p gororoba_algebra -p gr_core"
rust-regression-scoped:
	$(eval RUST_SCOPE ?= $(shell $(CARGO_ENV) cargo run -q -p gororoba_cli_data --bin workspace-routing -- --local 2>/dev/null || echo "--workspace"))
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
	            $(CARGO_ENV) cargo run -q -p gororoba_cli_data --bin local-nextest-plan -- --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $(if $(LOCAL_NEXTEST_TIMING_JSON),--timing-json-out $(LOCAL_NEXTEST_TIMING_JSON),) --filterset "$$filterset" $$local_light_packages; \
	        else \
	            $(CARGO_ENV) cargo run -q -p gororoba_cli_data --bin local-nextest-plan -- --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) $(if $(LOCAL_NEXTEST_TIMING_JSON),--timing-json-out $(LOCAL_NEXTEST_TIMING_JSON),) $$local_light_packages; \
	        fi; \
	    fi; \
	    if [ -n "$$heavy_scope" ] && [ "$(RUST_RUN_HEAVY)" = "1" ]; then \
	        $(CARGO_ENV) cargo nextest run --build-jobs $(CARGO_JOBS) --test-threads $(NEXTEST_TEST_THREADS) --cargo-profile test-heavy -P heavy $$heavy_scope; \
	    elif [ -n "$$heavy_scope" ]; then \
	        echo "[rust-regression-scoped] SKIP heavy nextest in local fast path: $$heavy_scope"; \
	    fi; \
	    echo "OK: Rust regression gate passed (scoped: clippy + nextest)."; \
	fi

rust-smoke-scoped: rust-regression-scoped
	@echo "DEPRECATED: make rust-smoke-scoped is a legacy alias. Use make rust-regression-scoped."

# Scoped pre-push: routes Rust/governance to affected scope only.
pre-push-gate-scoped:
	$(MAKE) gate-local
	@echo "OK: scoped pre-push gate passed via gate-local."

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
	perf stat -e $${PERF_EVENTS:-cycles:u,instructions:u,branches:u,branch-misses:u} -r $${PERF_RUNS:-3} $(REPO_CARGO_TARGET_DIR)/release/x87-strategy-bench \
		--len $${LEN:-262144} \
		--repeats $${REPEATS:-5} \
		--worker-counts $${WORKER_COUNTS:-1,2,4,6} \
		--output $${OUT:-reports/benchmarks/x87_strategy_perf.csv} \
		--summary $${SUMMARY:-reports/benchmarks/x87_strategy_perf.md}
	@echo "OK: x87 strategy perf-stat sweep completed."

x87-strategy-hyperfine:
	$(CARGO_ENV) cargo build --release -p gororoba_cli_algebra --bin x87-strategy-bench
	hyperfine --shell=none --warmup $${WARMUP:-1} --runs $${RUNS:-5} \
		'$(REPO_CARGO_TARGET_DIR)/release/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 1 --output /tmp/x87_strategy_hyperfine_1.csv' \
		'$(REPO_CARGO_TARGET_DIR)/release/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 2 --output /tmp/x87_strategy_hyperfine_2.csv' \
		'$(REPO_CARGO_TARGET_DIR)/release/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 4 --output /tmp/x87_strategy_hyperfine_4.csv' \
		'$(REPO_CARGO_TARGET_DIR)/release/x87-strategy-bench --len '$${LEN:-262144}' --repeats '$${REPEATS:-3}' --worker-counts 6 --output /tmp/x87_strategy_hyperfine_6.csv'
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
		$(REPO_CARGO_TARGET_DIR)/release/x87-givens-microbench \
		--iterations $${ITERATIONS:-200000} \
		--repeats $${REPEATS:-9} \
		$${CASES:+--cases $${CASES}} \
		$${KERNELS:+--kernels $${KERNELS}} \
		--output $${OUT:-reports/benchmarks/x87_givens_microbench_perf.csv} \
		$${SUMMARY:+--summary $${SUMMARY}} \
		2> $${COUNTERS_OUT:-reports/benchmarks/x87_givens_microbench_perf.stat}
	@echo "OK: x87 Givens perf-stat microbench completed."

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
	perf stat -e $${PERF_EVENTS:-cycles:u,instructions:u,branches:u,branch-misses:u} -r $${PERF_RUNS:-3} $(REPO_CARGO_TARGET_DIR)/release/jacobi-backend-sweep \
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
	cargo run --release -p gororoba_cli_data --bin provenance -- --db build/pantheon_physicsforge_migration.db pantheon-seed

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

python-retention-sync:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin python-retention -- sync

python-retention-verify:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin python-retention -- verify

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

registry-strict-toml-batch3-build:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin integrity-resolution -- --repo-root .

registry-verify-schema-signatures:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- schema-signatures

registry-verify-crossrefs:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- crossrefs

registry-verify-dataset-label-aliases:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- dataset-label-aliases

registry-verify-external-source-operational-contracts:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- external-source-operational-contracts

registry-verify-strict-toml-batch3:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin integrity-resolution -- --verify --repo-root .
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- schema-signatures
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- crossrefs
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- dataset-label-aliases
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin governance-verify -- external-source-operational-contracts

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
	@echo "DEPRECATED: make registry-verify-wave5-batch4 is a legacy alias. Use make registry-verify-execution-planning."

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
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/INSIGHTS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/CLAIMS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- bibliography-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/BIBLIOGRAPHY_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- experiments-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/EXPERIMENTS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- theorems-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/THEOREMS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- roadmap-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/ROADMAP_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- todo-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/TODO_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- next-actions-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/NEXT_ACTIONS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- navigator-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/NAVIGATOR_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- entrypoint-docs-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/ENTRYPOINT_DOCS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- requirements-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/REQUIREMENTS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- knowledge-migration-plan-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/KNOWLEDGE_MIGRATION_PLAN_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- markdown-governance-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-tasks-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/CLAIMS_TASKS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-domains-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/CLAIMS_DOMAINS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- claim-tickets-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/CLAIM_TICKETS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- external-sources-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/EXTERNAL_SOURCES_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- book-docs-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/BOOK_DOCS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- data-artifact-narratives-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- reports-narratives-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/REPORTS_NARRATIVES_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- docs-convos-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/DOCS_CONVOS_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- docs-root-narratives-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/DOCS_ROOT_NARRATIVES_REGISTRY_MIRROR.md"; \
	cargo run -q -p gororoba_cli_data --bin registry-emit -- research-narratives-mirror \
		--output "$(MARKDOWN_EXPORT_OUT_DIR)/RESEARCH_NARRATIVES_REGISTRY_MIRROR.md"; \
	if [ "$(MARKDOWN_EXPORT_EMIT_LEGACY)" = "1" ]; then \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- insights-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- experiments-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- theorems-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- roadmap-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- todo-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- next-actions-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- bibliography-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- navigator-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- entrypoint-docs-legacy; \
		cargo run -q -p gororoba_cli_data --bin registry-emit -- requirements-legacy; \
		if [ "$$legacy_claims_sync" = "1" ]; then \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-matrix-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-tasks-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- claims-domains-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- claim-tickets-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- external-sources-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- book-docs-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- data-artifact-narratives-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- reports-narratives-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- docs-convos-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- monograph-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- docs-root-narratives-legacy; \
			cargo run -q -p gororoba_cli_data --bin registry-emit -- research-narratives-legacy; \
		fi; \
	fi; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-inventory; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-corpus; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-origin-audit; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-owner-map; \
	cargo run -q -p gororoba_cli_data --bin markdown-registry -- build-payloads

registry-verify-mirrors:
	legacy_flag=""; \
	if [ "$(MARKDOWN_EXPORT_EMIT_LEGACY)" = "1" ]; then legacy_flag="--emit-legacy"; fi; \
	claims_value="true"; \
	if [ "$(MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC)" = "0" ]; then claims_value="false"; fi; \
	cargo run -q -p gororoba_cli_data --bin verify-registry-mirror-freshness -- \
		--out-dir "$(MARKDOWN_EXPORT_OUT_DIR)" $$legacy_flag --legacy-claims-sync $$claims_value
	PYTHONWARNINGS=error $(MAKE) registry-verify-markdown-toml-first
	@if [ "$(MARKDOWN_EXPORT_EMIT_LEGACY)" = "1" ]; then \
		cargo run -q -p gororoba_cli_data --bin governance-verify -- markdown-headers; \
		cargo run -q -p gororoba_cli_data --bin governance-verify -- markdown-parity; \
		cargo run -q -p gororoba_cli_data --bin governance-verify -- mirror-immutability; \
		cargo run -q -p gororoba_cli_data --bin governance-verify -- claim-ticket-mirrors; \
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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin synthesis-execution-contract -- \
		--date-token "$(SYNTHESIS_CONTRACT_DATE)" \
		--report-path "$(SYNTHESIS_CONTRACT_REPORT)"

docs-publish: registry-export-markdown
	$(MAKE) registry-verify-mirrors
	@echo "OK: TOML-driven markdown mirrors generated and verified for publishing."

terminology-gate:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- terminology-gate

ansi-check:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- ansi-check --check

ansi-check-strict:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- ansi-check --check --strict-placeholders --placeholder-scope-prefix crates/ --placeholder-scope-prefix tests/

verify: install
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_generated_artifacts.py

verify-grand: install
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_grand_images.py

verify-c010-c011-theses:
	PYTHONWARNINGS=error python3 src/verification/verify_c010_c011_theses.py

verify-python-core-algorithms:
	PYTHONWARNINGS=error python3 src/verification/verify_python_core_algorithms_pyo3.py

doctor: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin repo-utilities -- doctor
	sh scripts/detect_native_blas.sh

doctor-blas:
	sh scripts/detect_native_blas.sh

provenance: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-origin-audit -- --out reports/data_origin_audit_$$(date +%F).toml --fail-on-strict-unknown

provenance-audit: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin data-governance-gate -- --enforce-origin true --enforce-semantic true --enforce-blocked-deadlines true

provenance-registry-index:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin provenance -- index

provenance-registry-export:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin provenance -- export

provenance-registry-verify:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin provenance -- verify

provenance-registry-doctor:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin provenance -- doctor

provenance-registry-link-audit:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin provenance -- link-audit

provenance-registry-recover:
	$(CARGO_ENV) cargo run --release -p gororoba_cli_data --bin provenance -- recover

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
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- dimensional-geometry

artifacts-materials: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- materials-subset --n 200 --seed 0
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- materials-embedding

artifacts-boxkites: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- de-marrais-boxkites

artifacts-reggiani: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- reggiani-annihilator-stats

artifacts-m3: install
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- m3-table

artifacts-motifs:
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 16,32 --details
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- motif-summary

artifacts-motifs-big:
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 16,32,64,128 --summary-only
	$(CARGO_ENV) cargo run -p gororoba_cli_algebra --bin motif-census --release -- --dims 256 --max-nodes 5000 --seed 0 --summary-only
	$(CARGO_ENV) cargo run --release -p gororoba_cli_physics --bin artifact-regen -- motif-summary

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

# E-183: MaNGA N~2500 harmonic halo stacking sweep (D=16,64,256,1024).
# WHY: Full-sample N>=2500 stack reaches alpha_zd threshold ~0.004,
#      7x improvement over SPARC N=93. Each CD dimension is an independent run.
# HOW: Run after manga-maps-extractor finishes (check: wc -l data/external/manga/rotcurves/manga_rotcurves_all.csv)
# OUTPUT: data/results/e183/manga_stack_D{16,64,256,1024}.csv
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
	rm -rf $(REPO_CARGO_TARGET_DIR)

clean-builds:
	rm -rf target/
	rm -rf .cache/cargo-default-target/
	rm -rf .cache/gate-target/
	rm -rf /tmp/open_gororoba_*_target 2>/dev/null || true
	@echo "Removed all Rust build artifacts. Run 'cargo build' to rebuild."

clean-all: clean clean-builds clean-artifacts
	@rm -rf $(REPO_CARGO_HOME)
	@command -v cargo-sweep >/dev/null 2>&1 && cargo sweep --time 14 || true
	@echo "Full cleanup complete. Run 'make install && make artifacts' to rebuild."

# ---- Help ----

help:
	@echo "Targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install              Create venv and install (editable, dev deps)"
	@echo "    make bootstrap-dev        Ensure the dev venv/install stamp is current"
	@echo "    make install-analysis     Add analysis extras (networkx, ripser, sklearn)"
	@echo "    make install-astro        Add astronomy extras (gwpy, astroquery)"
	@echo "    make install-particle     Add particle-analysis extras (uproot, awkward, vector)"
	@echo "    make install-quantum      Add quantum extras (qiskit, Docker recommended)"
	@echo ""
	@echo "  Quality:"
	@echo "    make python-smoke         Run smoke-marked pytest coverage"
	@echo "    make python-regression    Run regression-marked pytest coverage without optional-extension tests"
	@echo "    make python-requires-ext  Run opt-in pytest coverage that needs optional vendor/extensions"
	@echo "    make test                 Alias for python-regression"
	@echo "    make lint                 Changed-file Ruff ratchet on src + tests + bin + scripts"
	@echo "    make smoke                Composite fast smoke lane (check + rust-smoke)"
	@echo "    make integrity            Python-only integrity lane (artifacts, mirrors, markdown)"
	@echo "    make integrity-rust       Cargo-backed integrity lane (claims + inventory + typed policy)"
	@echo "    make check                Fast local check (lint + python-smoke + ascii + terminology + no-reports)"
	@echo "    make ansi-check           Verify ANSI-safe UTF-8 character policy"
	@echo "    make ansi-check-strict    Verify ANSI-safe UTF-8 policy + fail on <U+....> placeholders in crates/tests"
	@echo "    make verify-pantheon-physicsforge-mapping Verify migration matrix/todo mapping completeness"
	@echo "    make verify-pantheon-physicsforge-license-headers Verify GPL-2.0-only header consistency in migrated files"
	@echo "    make verify-pantheon-physicsforge-overflow Verify overflow tracker max-5-active policy"
	@echo "    make seed-pantheon-physicsforge-sqlite Seed sqlite memoization for migration findings/risks"
	@echo "    make rust-smoke           Dedicated Rust smoke suites via nextest"
	@echo "    make rust-regression      Full Rust regression lane with heavy-crate routing"
	@echo "    make rust-regression-scoped Scoped Rust regression lane for affected crates"
	@echo "    make gate-local           Canonical scoped local push gate"
	@echo "    make gate-ci-python       Full Python/read-only governance CI gate"
	@echo "    make gate-ci-python-compat Python compatibility gate for non-authoritative versions"
	@echo "    make gate-ci-rust         Full Rust CI gate"
	@echo "    make gate-audit           Keep-going dry-run audit that writes reports/gates/*"
	@echo "    make heavy                Ignored/GPU/research-heavy nextest lane"
	@echo "    make test-inventory       Enforce taxonomy coverage and stale-doc checks"
	@echo "    make mcp-smoke            Re-test configured MCP server parity and startup health"
	@echo "    make cargo-deny-check     Enforce deny.toml (advisories, bans, licenses, sources)"
	@echo "    make x87-strategy-bench   Run pinned-core x87/AVX2 worker sweep and write CSV+Markdown reports"
	@echo "    make x87-strategy-perf    Run perf stat around the pinned-core x87/AVX2 benchmark binary"
	@echo "    make x87-strategy-hyperfine Compare 1/2/4/6-worker strategy runs with hyperfine"
	@echo "    make x87-strategy-flamegraph Capture a flamegraph for a focused x87-strategy-bench run"
	@echo "    make x87-givens-microbench Measure the actual composed x87 Givens/transcendental helper costs"
	@echo "    make x87-givens-microbench-perf Run perf stat around focused x87 Givens/transcendental microbench cases/kernels"
	@echo "    make jacobi-backend-sweep Run solver-shaped x87/DD/f64 Jacobi backend sweep across matrix sizes (optional FAMILIES=a,b and BACKENDS=x,y subsets)"
	@echo "    make block-jacobi-backend-sweep Run the block-Jacobi prototype sweep with block sizes 2 and 4 against current dense backends"
	@echo "    make partial-spectrum-bench Benchmark k=1,2,4 largest/smallest-magnitude partial-spectrum lanes against full-spectrum Jacobi baselines"
	@echo "    make structured-spectrum-bench Benchmark exact structured zero-mode deflation on quantized and real obstruction-like matrices"
	@echo "    make jacobi-backend-perf   Run perf stat around a focused jacobi-backend-sweep configuration"
	@echo "    make jacobi-backend-flamegraph Capture a flamegraph for a focused jacobi-backend-sweep configuration (defaults to release debuginfo for readable stacks)"
	@echo "    make jacobi-backend-samply Capture a focused samply profile for jacobi-backend-sweep (supports FEATURES=profile-dd-hotspots for less-inlined DD attribution)"
	@echo "    make jacobi-backend-samply-compare Summarize weighted line-level hotspots across the current reference/x87/DD samply artifacts"
	@echo "    make registry             Validate TOML registry consistency"
	@echo "    make registry-verify-typed-policy-error Strict registry-check typed-policy lane (--typed-policy error)"
	@echo "    make synthesis-execution-contract Run full synthesis execution contract and emit rollup TOML"
	@echo "    make governance-gate-readonly Read-only TOML registry governance gate"
	@echo "    make registry-control-plane-gate-readonly Read-only markdown/TOML control-plane gate"
	@echo "    make registry-csv-pipeline-gate  Validate project/external/archive CSV scroll pipeline lanes"
	@echo "    make registry-semantic-atoms-gate      Rust semantic-atoms build+verify lane"
	@echo "    make registry-evidence-provenance-gate Legacy build+verify evidence-provenance lane"
	@echo "    make registry-integrity-resolution-gate Legacy build+verify integrity-resolution lane"
	@echo "    make registry-execution-planning-gate  Legacy build+verify execution-planning lane"
	@echo "    make registry-acceptance-gate-readonly Read-only semantic/evidence/integrity/execution gate"
	@echo "    make registry-acceptance-gate    Compatibility alias for registry-acceptance-gate-readonly"
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
	@echo "    make governance-gate      Compatibility alias for governance-gate-readonly"
	@echo "    make registry-control-plane-gate-readonly Read-only control-plane registry gate"
	@echo "    make registry-acceptance-gate-readonly Read-only semantic/evidence/integrity/execution gate"
	@echo "    make registry-verify-typed-policy-error Supplemental strict typed-policy contract lane"
	@echo "    make synthesis-execution-contract governance-gate + registry-acceptance-gate + strict typed-policy + project-counter-sync --check"
	@echo "    make terminology-gate     enforce banned-term policy from terminology_standards.toml"
	@echo "    make pre-push-gate        Compatibility alias for gate-local"
	@echo "    make pre-push-gate-strict Compatibility alias for gate-audit"
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
	@echo "    make clean-builds         Remove all Rust build artifacts (target/, .cache/*-target/)"
	@echo "    make clean-artifacts      Remove generated CSV/images/HDF5 (keep source data)"
	@echo "    make clean-all            clean + clean-builds + clean-artifacts"
	@echo ""
	@echo "  Other:"
	@echo "    make run                  Run simulations (sedenion, modular, entropy)"
	@echo "    make rocq                Compile Rocq proofs"
	@echo "    make latex                Build MASTER_SYNTHESIS.pdf"
	@echo "    make docker-quantum-build Build qiskit-env Docker image"
	@echo "    make docker-quantum-run   Run quantum script in Docker (ARGS=...)"
	@echo "    make docker-quantum-shell Open interactive shell in qiskit-env"
	@echo "    make doctor               Environment diagnostics"
	@echo "    make doctor-blas          Detect native BLAS/LAPACK candidates and Cargo feature mapping"
	@echo "    ./makew <target>          Run make with inherited jobserver env stripped (useful when shells/tools export MAKEFLAGS)"
