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

## SQLite-canonical registry tooling

Since 2026-03-23 the registry source of truth is
`registry/canonical/control_plane.sqlite3` rather than `registry/*.toml`.
Mutating claims, insights, or experiments requires the workspace-internal
`gororoba-db` CLI plus the `provenance` exporter that regenerates compat
TOMLs from canonical SQLite. Walk-through:
`docs/engineering/registry_canonical_architecture.md`.

| Tool | Source | Run | Purpose |
| --- | --- | --- | --- |
| `gororoba-db` | `crates/gororoba_db` | `cargo run --release -p gororoba_db --bin gororoba-db -- <subcommand>` | Canonical mutator for claims/insights/experiments + planning/requirements |
| `provenance` | `crates/gororoba_cli_provenance` | `cargo run --release -p gororoba_cli_provenance --bin provenance -- export-control-plane` | Re-emit `registry/*.toml` from canonical SQLite |
| `repo-audit` | `crates/gororoba_cli_data` | `cargo run --release -p gororoba_cli_data --bin repo-audit` | Anchored debt counter; supports `--sqlite` for revisions audit |
| `integrity-resolution` | `crates/gororoba_cli_data` | `cargo run --release -p gororoba_cli_data --bin integrity-resolution` | Recompute `registry/schema_signatures.toml` after legitimate Layer-2 changes |

## ONNX runtime (turboquant-onnx-eval)

The TurboQuant real-model evaluation lane uses the ort 2.0.0-rc.12 crate
configured with `load-dynamic` + `api-18`. This loads
`libonnxruntime.so` at runtime via dlopen from the path in
`ORT_DYLIB_PATH` or the system search path. Required:

```
paru -S onnxruntime-opt-cuda     # Arch; provides /usr/lib64/libonnxruntime.so
ORT_DYLIB_PATH=/usr/lib64/libonnxruntime.so \
  cargo run --release -p gororoba_cli_physics --bin turboquant-onnx-eval \
    --features onnx-eval -- --model <path-to.onnx> --bits 2,3,4
```

Verified: end-to-end on data/external/onnx_test/distilgpt2.onnx (12-layer
GPT-2 decoder export) -- detects 24 KV-cache outputs and produces
real-model RMSE / top-1 / kv-byte metrics per requested bit count.

## Cache budget policy

The local pre-push 4-gate chain enforces a 250 GB hard cap (and 150 GB
soft cap) on cargo build artifacts in `.cache/`. Run `make cache-sweep`
when the gate complains; the sweep target now also clears stale
debug-profile artifacts in `.cache/gate-cbuild/<hash>/debug/` (the
local 4-gate chain only uses `--profile release-gate`).

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
| PMD CPD | `make cpd-audit` | `GitHub release: https://github.com/pmd/pmd/releases (Java tool; download tarball + JAVA_HOME). System-package fallback: paru -S pmd on Arch.` | yes | active |
| PMD CPD (Tooling) | `make cpd-audit-tooling` | `GitHub release: https://github.com/pmd/pmd/releases (Java tool; download tarball + JAVA_HOME). System-package fallback: paru -S pmd on Arch.` | no | active |
| PMD CPD (Generated) | `make cpd-audit-generated` | `GitHub release: https://github.com/pmd/pmd/releases (Java tool; download tarball + JAVA_HOME). System-package fallback: paru -S pmd on Arch.` | no | active |
| cargo-semver-checks | `make rust-semver-check` | `cargo install cargo-semver-checks` | no | blocked -- fwht external path dep cannot resolve from git temp checkout. |
| Docs Freshness | `make docs-freshness` | (built-in) | no | blocked -- Mathematical bracket notation [a,b,c] triggers broken-intra-doc-links. |
| repo-audit | `make repo-audit` | (built-in) | no | active |
| repo-audit (strict baseline) | `make repo-audit-strict` | (built-in) | no | active |
| repo-audit (strict unjustified) | `make repo-audit-strict-unjustified` | (built-in) | no | active |
| Cache Sweep | `make cache-sweep` | `cargo install cargo-sweep` | no | active |
| audit-deep (structured) | `make audit-deep-structured` | (built-in) | no | active |
| gororoba-db | -- | (built-in) | no | active |
| provenance export-control-plane | -- | (built-in) | no | active |
| integrity-resolution | `make integrity-resolution` | (built-in) | no | active |
| turboquant-onnx-eval | -- | `Vendored option: enable `ort` crate `download-binaries` feature (auto-fetches ONNX Runtime; +50 MB CI cache). System-pkg fallback: paru -S onnxruntime-opt-cuda on Arch; Microsoft GitHub release: https://github.com/microsoft/onnxruntime/releases.` | no | active |
| Vulkan SPIR-V build | -- | `GitHub release: https://github.com/google/shaderc/releases (prebuilt glslc tarball). System-pkg fallback: paru -S shaderc on Arch.` | no | active |
