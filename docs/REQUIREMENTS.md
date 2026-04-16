<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# open_gororoba Requirements

This repository is a Rust-first scientific workspace with optional Python,
Docker, Rocq, LaTeX, C++, and web-app lanes layered on top. The canonical
structured requirements inventory lives in
`registry/canonical/control_plane.sqlite3`. The checked-in
`registry/requirements.toml` and `registry/module_requirements.toml` files are
DB-backed compatibility exports; this narrative file exists to emit the
human-facing compatibility docs from the same source-of-truth.

## Core policy

- Treat warnings as errors.
- Keep installs reproducible and offline-testable by default.
- Keep generated docs emitted from TOML sources rather than hand-edited.
- Keep external data and optional toolchains behind explicit, auditable lanes.

## Toolchain baseline

- Rust: `nightly-2026-04-05` via `rust-toolchain.toml`
- Edition: `2024`
- Python: `3.11` or `3.12` recommended; `3.13+` allowed with optional-extras caveats
- Formal proofs: Rocq `9.1.1`
- Formatting/linting: `cargo fmt --all`, `cargo clippy --workspace -- -D warnings`

## Verification entrypoints

```sh
cargo build --workspace -j"$(nproc)"
cargo test --workspace -j"$(nproc)"
cargo clippy --workspace -j"$(nproc)" -- -D warnings
make governance-gate
make rust-semver-check
make cargo-deny-check
make dep-audit
make cpd-audit CPD_TOP=20
make docs-freshness
```

Optional audit tools for the deeper lanes include PMD/CPD, cargo-deny,
cargo-semver-checks, cargo-machete, cargo-geiger, dprint, and typos. Keep CPD
strict mode (`make cpd-audit-strict`) behind the current generated-surface and
lexical-error triage so structural duplicates fail CI only after known noise is
classified.

## Module docs

For module-specific requirements, see:

- `docs/requirements/algebra.md`
- `docs/requirements/analysis.md`
- `docs/requirements/astro.md`
- `docs/requirements/cpp.md`
- `docs/requirements/heliosphere.md`
- `docs/requirements/latex.md`
- `docs/requirements/materials.md`
- `docs/requirements/particle.md`
- `docs/requirements/quantum-docker.md`
- `docs/requirements/rocq.md`
- `crates/lbm_3d_cuda/README.md`
- `apps/gororoba_studio/README.md`

## Audit Tools

Each tool listed below is available via a dedicated `make` target. Tools marked **audit-deep** are included in `make audit-deep`.

| Tool | Make Target | Install | audit-deep | Status |
| --- | --- | --- | ---: | --- |
| Clippy | `make rust-clippy` | `rustup component add clippy` | yes | active |
| cargo-deny | `make cargo-deny-check` | `cargo install cargo-deny` | yes | active |
| Dependency Audit | `make dep-audit` | (built-in) | yes | active |
| PMD CPD | `make cpd-audit` | `paru -S pmd` | yes | active |
| PMD CPD (Tooling) | `make cpd-audit-tooling` | `paru -S pmd` | no | active |
| PMD CPD (Generated) | `make cpd-audit-generated` | `paru -S pmd` | no | active |
| cargo-semver-checks | `make rust-semver-check` | `cargo install cargo-semver-checks` | no | blocked -- fwht external path dep cannot resolve from git temp checkout. |
| Docs Freshness | `make docs-freshness` | (built-in) | no | blocked -- Mathematical bracket notation [a,b,c] triggers broken-intra-doc-links. |
