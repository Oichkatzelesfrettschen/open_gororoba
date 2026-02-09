<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/book_docs.toml -->

# Project Layout

open\_gororoba is organized as a Rust workspace with 15 member crates under
`crates/`, a TOML registry under `registry/`, and a PDF research collection
under `papers/`.

## Directory structure

```
crates/
  algebra_core/       Cayley-Dickson, Clifford, wheels, p-adic, groups, box-kites
  cosmology_core/     FLRW, bouncing models, dark energy
  gr_core/            Schwarzschild, Kerr, Novikov-Thorne, spectral bands
  materials_core/     Periodic table, metamaterial models
  optics_core/        GRIN lenses, ray tracing
  quantum_core/       Casimir, tensor networks, Grover
  stats_core/         Ultrametric, dip test, bootstrap, GPU (cudarc)
  spectral_core/      Fractional PDE, negative dimensions
  lbm_core/           Lattice Boltzmann
  control_core/       Control theory
  data_core/          Data loading, benchmarks, HDF5 export
  docpipe/            PDF extraction (pdfium-render)
  gororoba_py/        PyO3 bindings (thin wrappers)
  gororoba_cli/       37 analysis binaries

registry/
  claims.toml         475 tracked claims (C-001..C-475)
  insights.toml       16 insights (I-001..I-016)
  experiments.toml    10 reproducible experiments (E-001..E-010)
  binaries.toml       37 CLI binaries
  project.toml        Version, counts, sprint history

papers/
  pdf/                19 research PDFs
  bib/                BibTeX database (cayley_dickson.bib)
  extracted/          Structured TOML extractions per paper
  MANIFEST.toml       Paper metadata registry

docs/
  latex/              MASTER_SYNTHESIS.tex, MATHEMATICAL_FORMALISM.tex
  book/               This mdbook
  CLAIMS_EVIDENCE_MATRIX.md   Markdown mirror of claims.toml
  INSIGHTS.md                 Markdown mirror of insights.toml
  MATH_CONVENTIONS.md         9 mathematical conventions
```

## Build commands

```sh
# Full test suite (2063 tests)
cargo test --workspace -j$(nproc)

# Clippy with warnings-as-errors
cargo clippy --workspace -j$(nproc) -- -D warnings

# Registry validation
cargo run --release --bin registry-check

# LaTeX documents (requires texlive)
make latex

# GPU tests (requires CUDA + RTX 4070 Ti or similar)
cargo test --workspace -j$(nproc) --features gpu
```

## Dependency management

All external crates are declared at the workspace level in the root `Cargo.toml`
under `[workspace.dependencies]` and referenced by sub-crates with
`workspace = true`.  This prevents version conflicts and keeps the dependency
tree consistent.
