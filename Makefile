# ---- Phony targets ----
.PHONY: help install install-analysis install-astro install-particle install-quantum
.PHONY: test lint lint-all lint-all-stats lint-all-fix-safe check smoke math-verify
.PHONY: verify verify-grand ascii-check doctor provenance patch-pyfilesystem2
.PHONY: rust-test rust-clippy rust-smoke
.PHONY: artifacts artifacts-dimensional artifacts-materials artifacts-boxkites
.PHONY: artifacts-reggiani artifacts-m3 artifacts-motifs artifacts-motifs-big
.PHONY: fetch-data run coq latex
.PHONY: cpp-deps cpp-build cpp-test cpp-bench cpp-clean
.PHONY: docker-quantum-build docker-quantum-run docker-quantum-shell
.PHONY: clean clean-artifacts clean-all

VENV ?= venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

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

check: test lint smoke
	@echo "OK: check suite complete."

smoke: install
	PYTHONWARNINGS=error $(PYTHON) -m compileall -q src
	$(PYTHON) -m ruff check src/gemini_physics tests
	$(PYTHON) -m ruff check src --statistics --exit-zero
	$(PYTHON) bin/ascii_check.py --check
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

ascii-check:
	python3 bin/ascii_check.py --check

verify: install
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_generated_artifacts.py

verify-grand: install
	PYTHONWARNINGS=error $(PYTHON) src/verification/verify_grand_images.py

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
	latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=docs/latex/out docs/latex/MASTER_SYNTHESIS.tex

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
