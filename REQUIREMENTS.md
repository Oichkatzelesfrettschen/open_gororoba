<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

# Installation Requirements (Reproducible)

This repo targets a **host Python** for non-quantum work and a **Docker Python 3.11**
environment for Qiskit.

Canonical (more detailed) requirements live in `docs/REQUIREMENTS.md`. This file is a short
top-level entrypoint.

Maintenance note: when requirements guidance drifts, update `docs/REQUIREMENTS.md` first and
keep this file as a stable pointer + minimal quickstart.

## 1) Core (Python >=3.11)

```bash
make install
make check
make doctor
```

Notes:
- `make check` runs `make test`, `make lint`, and `make smoke` (warnings-as-errors).
- `make smoke` also runs `make ascii-check` (repo-authored ASCII-only gate).
- Some doc/source extraction utilities rely on system tools like `pdftotext`; see `docs/REQUIREMENTS.md`.
- Optional modules may not install cleanly on very new Python versions; use a Python 3.11/3.12 env or Docker when needed.
- In particular, the `analysis` extra conditionally installs `scikit-learn` only for Python <3.14 (see `pyproject.toml`).
- System-installed Python packages (e.g., Arch/AUR) are not visible inside this repo's default venv unless you use `--system-site-packages` or install the pip equivalents.
- Editable installs may create `src/*.egg-info/`; these are gitignored and should never be committed.

## 2) Optional Modules

- Analysis/Topology/Embeddings: see `docs/requirements/analysis.md`
- Astronomy data fetching: see `docs/requirements/astro.md`
- Algebra engine / sedenion replications: see `docs/requirements/algebra.md`
- Particle physics (PDG/CMS toy spectra): see `docs/requirements/particle.md`
- Quantum (Qiskit, Aer): see `docs/requirements/quantum-docker.md`
- C++ kernels (Conan + CMake): see `docs/requirements/cpp.md`
- Coq proofs: see `docs/requirements/coq.md`
- LaTeX / paper build: see `docs/requirements/latex.md`
- Materials datasets: see `docs/requirements/materials.md`

## 3) Artifact generation (optional)

```bash
make artifacts-dimensional
make artifacts-materials
make artifacts-motifs
make artifacts-motifs-big
make verify
```
