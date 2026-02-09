<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

# Requirements for Gemini Experiments

This project follows a "Maximal Synthesis" protocol. Installs are designed to be reproducible,
offline-testable by default, and compatible with warnings-as-errors.

If you need module-specific details, see:
- `docs/requirements/algebra.md`
- `docs/requirements/analysis.md`
- `docs/requirements/astro.md`
- `docs/requirements/materials.md`
- `docs/requirements/quantum-docker.md`
- `docs/requirements/coq.md`
- `docs/requirements/latex.md`

## Python version policy

- **Recommended**: Python 3.11 or 3.12 (best wheel availability).
- **Allowed**: Python 3.13+ for the core engine, but some optional extras may be skipped by
  dependency markers or may not have wheels yet (use Docker or a Python 3.11/3.12 env).

## Quickstart (core engine)

1. Create an isolated venv + install the editable package:
   ```bash
   make install
   ```
2. Run the full gate (tests + lint + verifiers) with warnings-as-errors:
   ```bash
   PYTHONWARNINGS=error make check
   ```

The canonical Python dependency sets live in `pyproject.toml`. Prefer using the `make install-*`
targets (they match the extras).

## Optional Python extras

- **Analysis** (topology + embeddings):
  - Install: `make install-analysis`
  - Notes: scikit-learn is intentionally skipped on very new Python versions (see markers in
    `pyproject.toml`); use Python 3.11/3.12 if you need it.
- **Astronomy** (astroquery + GWpy + healpy + dataset loaders):
  - Install: `make install-astro`
- **Particle physics** (CMS/ROOT readers):
  - Install: `make install-particle`
  - Notes: `docs/requirements/particle.md`
- **Quantum** (Qiskit):
  - Preferred: Docker (`docs/requirements/quantum-docker.md`)
  - Optional local install: `make install-quantum` (only on Python versions supported by wheels)

Notes:
- Extras currently defined in `pyproject.toml`: `dev`, `analysis`, `astro`, `particle`, `quantum`.
- Some domains have requirements docs but no dedicated extra (e.g., algebra is part of core deps).

## Quantum (Dockerized)
See `docs/requirements/quantum-docker.md` and `docs/QISKIT_SETUP.md`.

## C++ Acceleration Kernels
- CMake >= 3.25
- Conan 2.x (C++ package manager)
- GCC 14 or Clang 18 (C++20 required)
- Catch2 (via Conan; unit tests)
- Google Benchmark (via Conan; micro-benchmarks)
- pybind11 (via Conan; Python bindings)

Build: `make cpp-build`, test: `make cpp-test`, bench: `make cpp-bench`.

See also: `docs/requirements/cpp.md`.

## System tools (non-Python)

Some targets rely on standard system utilities (documented here for reproducibility):

- `pdftotext` (Poppler) for `make convos-extract` and some source-caching pipelines.
  - Arch: `poppler`
  - Debian/Ubuntu: `poppler-utils`
- `wget` (preferred) and/or `curl` for opt-in fetch scripts (tests remain offline).
- `git`, `make`
- Optional: `tor` + `torsocks` for Tor-based fetches when required.
  - Arch: `sudo pacman -S tor torsocks` (or `paru -S tor torsocks`)
  - Enable: `sudo systemctl enable --now tor`
  - Note: do NOT run a second `tor` instance on `127.0.0.1:9050` if `tor.service` is already running.
    Check: `sudo ss -lntp | rg ':9050'`
  - Use: `torsocks wget -O <path> <url>`

## Dev / Quality
- `pytest>=7.4.0`    # Testing framework
- `ruff>=0.6.0`      # Fast linter
