<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# open_gororoba Requirements

This repository is a Rust-first scientific workspace with optional Python,
Docker, Rocq, LaTeX, C++, and web-app lanes layered on top. The canonical
requirements inventory lives in `registry/requirements.toml` and
`registry/module_requirements.toml`; this narrative file exists to emit the
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
```

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
