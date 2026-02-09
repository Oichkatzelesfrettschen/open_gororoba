# Requirements Registry Mirror

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: TOML registry files under registry/ -->
<!-- Generated at: 2026-02-09T08:38:02Z -->

Authoritative source: `registry/requirements.toml`.

- Updated: 2026-02-09
- Python recommended: `3.11-3.12`
- Python allowed: `3.13+ (with optional extras caveats)`
- Primary markdown: `docs/REQUIREMENTS.md`

## Modules

### REQ-CORE: core

- Status: `active`
- Markdown: `docs/REQUIREMENTS.md`
- Install targets:
  - `make install`
  - `PYTHONWARNINGS=error make check`

### REQ-ALGEBRA: algebra

- Status: `active`
- Markdown: `docs/requirements/algebra.md`

### REQ-ANALYSIS: analysis

- Status: `active`
- Markdown: `docs/requirements/analysis.md`
- Install targets:
  - `make install-analysis`

### REQ-ASTRO: astro

- Status: `active`
- Markdown: `docs/requirements/astro.md`
- Install targets:
  - `make install-astro`

### REQ-MATERIALS: materials

- Status: `active`
- Markdown: `docs/requirements/materials.md`

### REQ-PARTICLE: particle

- Status: `active`
- Markdown: `docs/requirements/particle.md`
- Install targets:
  - `make install-particle`

### REQ-QUANTUM: quantum_docker

- Status: `active`
- Markdown: `docs/requirements/quantum-docker.md`
- Install targets:
  - `make install-quantum`
  - `docker-quantum-build`
  - `docker-quantum-run`

### REQ-COQ: coq

- Status: `active`
- Markdown: `docs/requirements/coq.md`
- Install targets:
  - `make coq`

### REQ-LATEX: latex

- Status: `active`
- Markdown: `docs/requirements/latex.md`
- Install targets:
  - `make latex`

### REQ-CPP: cpp

- Status: `active`
- Markdown: `docs/requirements/cpp.md`
- Install targets:
  - `make cpp-build`
  - `make cpp-test`

## Coverage Gaps

### REQ-GAP-001: crate_specific_docs

- Status: `open`
- Description: Per-crate requirements markdown is missing for several Rust crates.
- Proposed resolution: Define crate-level requirements entries in this TOML and generate markdown stubs as needed.

### REQ-GAP-002: operational_sync

- Status: `open`
- Description: No generator currently syncs requirements markdown from this TOML registry.
- Proposed resolution: Add TOML -> markdown exporter and include in make registry.
