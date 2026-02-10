# ---- Phony targets ----
.PHONY: help install install-analysis install-astro install-particle install-quantum
.PHONY: test lint lint-all lint-all-stats lint-all-fix-safe check smoke math-verify wave6-gate
.PHONY: verify verify-grand verify-c010-c011-theses ascii-check doctor provenance patch-pyfilesystem2
.PHONY: rust-test rust-clippy rust-smoke
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
.PHONY: registry-verify-markdown-inventory registry-verify-markdown-origin registry-verify-markdown-owner registry-verify-wave4 registry-wave4
.PHONY: registry-verify-markdown-toml-first
.PHONY: registry-wave5-batch1-build registry-verify-wave5-batch1 registry-wave5-batch1
.PHONY: registry-wave5-batch2-build registry-verify-wave5-batch2 registry-wave5-batch2 registry-wave5
.PHONY: registry-wave5-batch3-build registry-verify-wave5-batch3 registry-wave5-batch3
.PHONY: registry-wave5-batch4-build registry-verify-wave5-batch4 registry-wave5-batch4
.PHONY: registry-verify-schema-signatures registry-verify-crossrefs
.PHONY: registry-csv-inventory registry-migrate-legacy-csv registry-verify-legacy-csv
.PHONY: registry-migrate-curated-csv registry-verify-curated-csv registry-csv-scope registry-data
.PHONY: registry-project-csv-split registry-csv-holdings
.PHONY: registry-scroll-project-csv-canonical registry-scroll-project-csv-generated
.PHONY: registry-scroll-external-csv-holding registry-scroll-archive-csv-holding
.PHONY: registry-csv-scroll-pipeline registry-verify-csv-scroll-pipeline
.PHONY: registry-verify-project-csv-split registry-verify-csv-holdings registry-verify-csv-corpus-coverage registry-wave3
.PHONY: registry-ingest-legacy registry-refresh registry-export-markdown registry-verify-mirrors docs-publish
.PHONY: verify-python-core-algorithms
.PHONY: artifacts artifacts-dimensional artifacts-materials artifacts-boxkites
.PHONY: artifacts-reggiani artifacts-m3 artifacts-motifs artifacts-motifs-big
.PHONY: fetch-data run coq latex
.PHONY: cpp-deps cpp-build cpp-test cpp-bench cpp-clean
.PHONY: docker-quantum-build docker-quantum-run docker-quantum-shell
.PHONY: clean clean-artifacts clean-all

VENV ?= venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
MARKDOWN_EXPORT ?= 0
MARKDOWN_EXPORT_OUT_DIR ?= build/docs/generated
MARKDOWN_EXPORT_EMIT_LEGACY ?= 0
MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC ?= 1

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

test: install
	PYTHONWARNINGS=error $(PYTHON) -m pytest

lint: install
	$(PYTHON) -m ruff check src/gemini_physics tests

lint-all: install
	$(PYTHON) -m ruff check src

lint-all-stats: install
	$(PYTHON) -m ruff check src --statistics --exit-zero

lint-all-fix-safe: install
	$(PYTHON) -m ruff check src --select W291,W293,I001 --fix

check: registry-verify-markdown-owner test lint smoke
	@echo "OK: check suite complete."

# Wave 6: TOML-first governance acceptance gate (W6-023)
wave6-gate: registry-verify-markdown-inventory registry-verify-markdown-owner registry-verify-schema-signatures registry-verify-crossrefs
	@echo ""
	@echo "=========================================="
	@echo "WAVE 6 ACCEPTANCE GATE: PASSED"
	@echo "=========================================="
	@echo "✓ Markdown inventory validated (TOML-first)"
	@echo "✓ Markdown owner map verified"
	@echo "✓ Registry schema signatures checked"
	@echo "✓ Cross-reference integrity verified"
	@echo ""
	@echo "Wave 6 TOML-first governance is operational."
	@echo "=========================================="
	@echo ""
	@echo "To run full validation pipeline including ASCII check:"
	@echo "  make check && make ascii-check"

smoke: install
	PYTHONWARNINGS=error $(PYTHON) -m compileall -q src
	$(PYTHON) -m ruff check src/gemini_physics tests
	$(PYTHON) -m ruff check src --statistics --exit-zero
	$(PYTHON) bin/ascii_check.py --check
	$(MAKE) registry-verify-markdown-owner
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_python_core_algorithms_pyo3.py
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_dataset_manifest_providers.py
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_generated_artifacts.py
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_grand_images.py

math-verify: test lint
	@echo "OK: math validation suite complete. See docs/MATH_VALIDATION_REPORT.md"

rust-test:
	cargo test --workspace -j$$(nproc)

rust-clippy:
	cargo clippy --workspace -j$$(nproc) -- -D warnings

rust-smoke: rust-clippy rust-test
	@echo "OK: Rust quality gate passed (clippy + tests)."

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

registry-verify-markdown-inventory: registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_inventory_toml_first.py

registry-verify-markdown-origin: registry-markdown-origin-audit
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_origin_audit.py

registry-verify-markdown-owner: registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_owner_map.py

registry-verify-markdown-toml-first: registry-verify-markdown-inventory registry-verify-markdown-owner
	@echo "OK: markdown TOML-first owner/inventory gates verified."

registry-verify-wave4: registry-markdown-corpus registry-toml-inventory registry-verify-markdown-origin registry-verify-markdown-owner registry-verify-knowledge-atoms registry-verify-artifact-scrolls
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_corpus_registry.py
	PYTHONWARNINGS=error python3 src/verification/verify_toml_inventory_registry.py

registry-wave4: registry-verify-wave4
	@echo "OK: Wave 4 control-plane registry lane complete."

registry-wave5-batch1-build:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_wave5_batch1_registries.py
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_markdown_payload_registries.py

registry-verify-wave5-batch1: registry-wave5-batch1-build
	PYTHONWARNINGS=error python3 src/verification/verify_wave5_batch1_registries.py

registry-wave5-batch1: registry-verify-wave5-batch1
	@echo "OK: Wave 5 batch 1 strict TOML registries complete."

registry-wave5-batch2-build:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_wave5_batch2_registries.py

registry-verify-wave5-batch2: registry-wave5-batch2-build
	PYTHONWARNINGS=error python3 src/verification/verify_wave5_batch2_registries.py

registry-wave5-batch2: registry-verify-wave5-batch2
	@echo "OK: Wave 5 batch 2 strict TOML registries complete."

registry-wave5-batch3-build:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_wave5_batch3_registries.py

registry-verify-schema-signatures:
	PYTHONWARNINGS=error python3 src/verification/verify_registry_schema_signatures.py

registry-verify-crossrefs:
	PYTHONWARNINGS=error python3 src/verification/verify_registry_crossrefs.py

registry-verify-wave5-batch3: registry-wave5-batch3-build
	PYTHONWARNINGS=error python3 src/verification/verify_wave5_batch3_registries.py
	PYTHONWARNINGS=error python3 src/verification/verify_registry_schema_signatures.py
	PYTHONWARNINGS=error python3 src/verification/verify_registry_crossrefs.py

registry-wave5-batch3: registry-verify-wave5-batch3
	@echo "OK: Wave 5 batch 3 strict TOML registries complete."

registry-wave5-batch4-build:
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_wave5_batch4_registries.py

registry-verify-wave5-batch4: registry-wave5-batch4-build registry-markdown-inventory
	PYTHONWARNINGS=error python3 src/verification/verify_wave5_batch4_registries.py
	PYTHONWARNINGS=error python3 src/verification/verify_registry_crossrefs.py
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_inventory_toml_first.py
	PYTHONWARNINGS=error python3 src/verification/verify_markdown_owner_map.py

registry-wave5-batch4: registry-verify-wave5-batch4
	@echo "OK: Wave 5 batch 4 strict TOML registries complete."

registry-wave5: registry-wave5-batch1 registry-wave5-batch2 registry-wave5-batch3 registry-wave5-batch4
	@echo "OK: Wave 5 acceptance gate complete."

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
	cargo run --release --bin scrollify-csv -- \
		--source-manifest registry/manifests/project_csv_canonical_manifest.txt \
		--out-index registry/project_csv_canonical_datasets.toml \
		--out-dir registry/data/project_csv/canonical \
		--index-table project_csv_canonical_datasets \
		--dataset-prefix PC \
		--corpus-label 'project CSV canonical dataset' \
		--dataset-class canonical-dataset

registry-scroll-project-csv-generated: registry-project-csv-split
	cargo run --release --bin scrollify-csv -- \
		--source-manifest registry/manifests/project_csv_generated_manifest.txt \
		--out-index registry/project_csv_generated_artifacts.toml \
		--out-dir registry/data/project_csv/generated \
		--index-table project_csv_generated_artifacts \
		--dataset-prefix PG \
		--corpus-label 'project CSV generated artifact' \
		--dataset-class generated-artifact

registry-scroll-archive-csv-holding: registry-csv-holdings
	cargo run --release --bin scrollify-csv -- \
		--source-manifest registry/manifests/archive_csv_holding_manifest.txt \
		--out-index registry/archive_csv_holding_datasets.toml \
		--out-dir registry/data/archive_csv_holding \
		--index-table archive_csv_holding_datasets \
		--dataset-prefix AH \
		--corpus-label 'archive CSV holding queue' \
		--dataset-class holding-archive

registry-scroll-external-csv-holding: registry-csv-holdings
	cargo run --release --bin scrollify-csv -- \
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
	cargo run --release --bin verify-project-csv-split -- \
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

registry-wave3: registry-project-csv-split registry-csv-holdings registry-verify-project-csv-split registry-verify-csv-holdings registry-verify-csv-corpus-coverage registry-verify-csv-scroll-pipeline

registry-csv-scope: registry-csv-inventory
	PYTHONWARNINGS=error python3 src/scripts/analysis/build_csv_migration_scope_registry.py

registry-data: registry-migrate-legacy-csv registry-migrate-curated-csv registry-wave3 registry-csv-inventory registry-verify-legacy-csv registry-verify-curated-csv registry-csv-scope registry-wave4
	@echo "OK: CSV data registry lane complete."

registry-export-markdown: registry-refresh
	@if [ "$(MARKDOWN_EXPORT)" != "1" ]; then \
		echo "SKIP: markdown export disabled (set MARKDOWN_EXPORT=1)"; \
		exit 0; \
	fi
	@legacy_flag="--no-emit-legacy"; \
	if [ "$(MARKDOWN_EXPORT_EMIT_LEGACY)" = "1" ]; then legacy_flag="--emit-legacy"; fi; \
	claims_flag="--legacy-claims-sync"; \
	if [ "$(MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC)" = "0" ]; then claims_flag="--no-legacy-claims-sync"; fi; \
	PYTHONWARNINGS=error python3 src/scripts/analysis/export_registry_markdown_mirrors.py \
		--out-dir "$(MARKDOWN_EXPORT_OUT_DIR)" $$legacy_flag $$claims_flag

registry-verify-mirrors: registry-export-markdown
	@if [ "$(MARKDOWN_EXPORT)" != "1" ]; then \
		echo "SKIP: mirror verification disabled (set MARKDOWN_EXPORT=1)"; \
		exit 0; \
	fi
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

registry: registry-refresh registry-data
	cargo run --release --bin registry-check

docs-publish: registry-verify-mirrors
	@echo "OK: TOML-driven markdown mirrors generated and verified for publishing."

ascii-check:
	python3 bin/ascii_check.py --check

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

provenance: install
	PYTHONWARNINGS=error $(PYTHON) bin/record_external_hashes.py

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
	cargo run -p gororoba_cli --bin motif-census --release -j$$(nproc) -- --dims 16,32 --details
	PYTHONWARNINGS=error $(PYTHON) src/vis_cd_motif_summary.py

artifacts-motifs-big:
	cargo run -p gororoba_cli --bin motif-census --release -j$$(nproc) -- --dims 16,32,64,128 --summary-only
	cargo run -p gororoba_cli --bin motif-census --release -j$$(nproc) -- --dims 256 --max-nodes 5000 --seed 0 --summary-only
	PYTHONWARNINGS=error $(PYTHON) src/vis_cd_motif_summary.py

# ---- Data fetching ----
#
# External datasets are NOT committed to the repo.  These targets download
# them into the locations expected by analysis scripts.

fetch-data: install
	@echo "Fetching external datasets..."
	PYTHONWARNINGS=error $(PYTHON) src/fetch_materials_jarvis_subset.py --n 200 --seed 0
	@echo "Run 'make provenance' to update hash registry after fetching."

# ---- Simulation runs ----

run: install
	PYTHONWARNINGS=error $(PYTHON) src/sedenion_field_sim.py
	PYTHONWARNINGS=error $(PYTHON) src/modular_classical_sim.py
	PYTHONWARNINGS=error $(PYTHON) src/entropy_pde_fit.py

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

# ---- LaTeX ----

latex:
	@command -v latexmk >/dev/null 2>&1 || { echo "ERROR: latexmk not found. Install TeX Live (see docs/requirements/latex.md)"; exit 1; }
	cargo run --release --bin generate-latex
	@mkdir -p docs/latex/out
	cd docs/latex && TEXINPUTS=.:$(CURDIR)/papers/bib/: BIBINPUTS=$(CURDIR)/papers/bib/: latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out MASTER_SYNTHESIS.tex
	cd docs/latex && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=out MATHEMATICAL_FORMALISM.tex

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

# ---- C++ kernels ----

cpp-deps:
	@test -f cpp/conanfile.txt -o -f cpp/conanfile.py || { echo "ERROR: missing cpp/conanfile.*"; exit 1; }
	cd cpp && conan install . --build=missing

cpp-build:
	@test -f cpp/CMakeLists.txt || { echo "ERROR: missing cpp/CMakeLists.txt"; exit 1; }
	cmake -S cpp -B cpp/build/Release -DCMAKE_BUILD_TYPE=Release
	cmake --build cpp/build/Release -j$$(nproc)

cpp-test: cpp-build
	ctest --test-dir cpp/build/Release --output-on-failure

cpp-bench: cpp-build
	@command -v ./cpp/build/Release/bench_cd_multiply >/dev/null 2>&1 || { echo "ERROR: bench_cd_multiply not built"; exit 1; }
	./cpp/build/Release/bench_cd_multiply

cpp-clean:
	rm -rf cpp/build

# ---- Cleanup ----

clean-artifacts:
	@echo "Removing generated CSV artifacts..."
	rm -f data/csv/cd_motif_*.csv
	rm -f data/csv/de_marrais_*.csv
	rm -f data/csv/reggiani_*.csv
	rm -f data/csv/m3_table.csv
	rm -f data/csv/dimensional_geometry_*.csv
	rm -f data/csv/materials_jarvis_subset.csv
	rm -f data/csv/materials_embedding_benchmarks.csv
	rm -f data/csv/modular_chaos_*.csv
	rm -f data/csv/sedenion_field_metrics_*.csv
	rm -f data/csv/spectral_flow.csv
	@echo "Removing generated images..."
	rm -f data/artifacts/images/cd_motif_summary_*.png
	rm -f data/artifacts/images/dimensional_geometry_*.png
	rm -f data/artifacts/images/materials_pca_*.png
	@echo "Removing HDF5 outputs..."
	rm -rf data/h5/
	@echo "Removing LaTeX build output..."
	rm -rf docs/latex/out/
	@echo "Done. Regenerate with: make artifacts"

clean:
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	rm -rf src/*.egg-info

clean-all: clean clean-artifacts
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
	@echo "    make test                 Run pytest (warnings-as-errors)"
	@echo "    make lint                 Ruff check on src/gemini_physics + tests"
	@echo "    make smoke                Compileall + lint stats + artifact verifiers"
	@echo "    make check                test + lint + smoke (CI entry point)"
	@echo "    make ascii-check          Verify ASCII-only policy"
	@echo "    make rust-smoke           Rust clippy + full test suite"
	@echo "    make registry             Validate TOML registry consistency"
	@echo "    make registry-wave4       Validate markdown/TOML control-plane + atom extraction gates"
	@echo "    make registry-wave3       Validate project/external/archive CSV scroll pipeline lanes"
	@echo "    make registry-wave5-batch1 Build+verify strict claims/equation/proof/payload TOML registries"
	@echo "    make registry-wave5-batch2 Build+verify strict derivation/bibliography/provenance/paragraph TOML registries"
	@echo "    make registry-wave5-batch3 Build+verify strict contradiction/signature/crossref TOML registries"
	@echo "    make registry-wave5-batch4 Build+verify strict experiment/planning/requirements TOML registries"
	@echo "    make registry-wave5       Run full Wave 5 acceptance gate (batch1 + batch2 + batch3 + batch4)"
	@echo "    make registry-verify-schema-signatures Verify critical registry schema signatures"
	@echo "    make registry-verify-crossrefs Verify dangling cross-registry references"
	@echo "    make registry-verify-knowledge-atoms Verify claim/equation/proof atom registries"
	@echo "    make registry-verify-markdown-toml-first Verify markdown owner/inventory TOML-first hard gate"
	@echo "    MARKDOWN_EXPORT=1 make docs-publish Export mirrors in strict mode (out-of-tree, no legacy writes)"
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
	@echo "    make fetch-data           Download external datasets"
	@echo "    make provenance           Hash data/external/* into PROVENANCE.local.json"
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
	@echo "    make cpp-build            Build optional C++ kernels"
	@echo "    make cpp-test             Run C++ kernel tests"
	@echo "    make cpp-bench            Run C++ kernel benchmarks"
	@echo "    make docker-quantum-build Build qiskit-env Docker image"
	@echo "    make docker-quantum-run   Run quantum script in Docker (ARGS=...)"
	@echo "    make docker-quantum-shell Open interactive shell in qiskit-env"
	@echo "    make doctor               Environment diagnostics"
