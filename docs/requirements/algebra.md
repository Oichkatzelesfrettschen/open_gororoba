<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Algebra Engine (Cayley-Dickson, de Marrais, Reggiani)

These components primarily live in Rust crates under `crates/gororoba_algebra/`,
`crates/cd_kernel/`, and related sibling algebra crates, and are exercised by
Rust unit and integration tests.

Install:

```ignore
make install
make rust-smoke
make rust-regression
```

Notes:

- The core algebra replication/validation code depends only on the base install extras.
- Keep `PYTHONWARNINGS=error` enabled in scripts and CI; warnings are errors here.
- The workspace pins stable Rust (`1.97.0`); root profiles use default LLVM codegen.
- Native BLAS backends for the Burn-based `neural_homotopy` path are explicit
  Cargo feature opt-ins, not defaults.
- Use `make doctor-blas` or `sh scripts/detect_native_blas.sh` to see which
  native BLAS/LAPACK candidates are present on the host and how they map to the
  repo's exposed feature surface.

Artifact generation entrypoints:

- `make artifacts-boxkites`
- `make artifacts-reggiani`
- `make artifacts-motifs`
- `make artifacts-motifs-big`

Recommended verification commands:

```ignore
cargo test -p cd_kernel --lib
cargo test -p algebra_analysis --lib x87_jacobi
cargo test -p algebra_analysis --lib reference_jacobi
cargo test -p gororoba_cli_algebra --bin x87_strategy_bench
cargo check -p gororoba_cli_algebra --bin jacobi-backend-sweep
cargo clippy -p cd_kernel -p algebra_analysis -p gororoba_cli_algebra --all-targets -- -D warnings
```
