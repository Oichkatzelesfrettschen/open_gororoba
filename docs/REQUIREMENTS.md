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

- Rust: stable `1.97.0` via `rust-toolchain.toml`
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

## Technical-debt taxonomy and baseline

The repository tracks technical, structural, methodological, and
domain-specific debt through one authoritative taxonomy at
`plans/repo_debt_taxonomy_roadmap_2026_06_04.toml`. It organizes 21 debt
classes into six axes (code/structure, build/toolchain/delivery,
test/verification, data/provenance, scientific/methodology, and
documentation/organizational), reconciles every legacy `DEBT-*`
identifier into a canonical class, and binds each class to a reproducible
measurement and a falsifiable closure criterion.

The measured evidence baseline is `data/output/debt_baseline_2026_06_04.toml`.
The strict regression gate compares against the machine-readable snapshot at
`data/output/audit/2026-06-25/repo_audit_anchored_2026_06_25.toml`, regenerated
by the anchored `repo-audit` binary (comment/string-stripped regex on Rust,
line-anchored on Rocq):

```sh
make repo-audit                      # emit a fresh anchored snapshot
make repo-audit-strict               # fail if a non-safety-positive class grew
make repo-audit-strict-unjustified   # cap unjustified clippy allows per root
```

Metric definitions live in `docs/engineering/repo_audit_metric_taxonomy.md`.

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
