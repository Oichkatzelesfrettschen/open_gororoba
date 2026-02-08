# open_gororoba

A research workbench exploring whether algebraic structure in Cayley-Dickson
algebras (real -> complex -> quaternion -> octonion -> sedenion -> ...) can
explain or predict phenomena in quantum gravity, particle physics, and cosmology.

## Key Numbers

- **1828** Rust tests (0 failures)
- **475** tracked claims (C-001..C-475)
- **16** insights (I-001..I-016)
- **10** reproducible experiments (E-001..E-010)
- **37** CLI binaries
- **19** extracted research papers
- **15** workspace crates

## Quick Start

```sh
# Build everything
cargo build --workspace -j$(nproc)

# Run tests
cargo test --workspace -j$(nproc)

# Run clippy (warnings-as-errors)
cargo clippy --workspace -j$(nproc) -- -D warnings

# Check registry integrity
cargo run --release --bin registry-check

# Extract a paper to TOML
cargo run --release --bin extract-papers -- --only demarrais-2000-math0011260
```

## Codebase Structure

```
crates/
  algebra_core/     Cayley-Dickson, Clifford, wheels, p-adic, groups
  cosmology_core/   FLRW, bouncing models, dark energy
  gr_core/          Schwarzschild, Kerr, Novikov-Thorne, spectral bands
  materials_core/   Periodic table, metamaterial models
  optics_core/      GRIN lenses, ray tracing
  quantum_core/     Casimir, tensor networks, Grover
  stats_core/       Ultrametric, dip test, bootstrap, GPU
  spectral_core/    Fractional PDE, negative dimensions
  lbm_core/         Lattice Boltzmann
  control_core/     Control theory
  data_core/        Data loading, benchmarks
  docpipe/          PDF extraction (pdfium-render primary)
  gororoba_py/      PyO3 bindings (thin wrappers)
  gororoba_cli/     37 analysis binaries
registry/           TOML registry (claims, insights, experiments)
papers/             PDF collection + TOML extractions
docs/               Documentation and tracking
```
