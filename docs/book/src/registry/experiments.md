# Experiments Portfolio

The experiments registry tracks 10 reproducible experiments (E-001 through E-010)
in `registry/experiments.toml`.  Each experiment specifies the binary, method,
input data, output files, run command, and associated claims.

## Experiment catalog

| ID | Title | Binary | GPU | Deterministic |
|----|-------|--------|-----|---------------|
| E-001 | Cayley-Dickson Motif Census | motif-census | No | Yes |
| E-002 | Multi-Dataset GPU Ultrametric Sweep | multi-dataset-ultrametric | Yes | No |
| E-003 | Real Cosmological Fit | real-cosmo-fit | No | Yes |
| E-004 | Kerr Shadow Boundaries | kerr-shadow | No | Yes |
| E-005 | Zero-Divisor Graph Invariants | zd-search | No | Yes |
| E-006 | Gravastar TOV Parameter Sweep | gravastar-sweep | No | Yes |
| E-007 | Tensor Network / PEPS Entropy | tensor-network | No | No |
| E-008 | GWTC-3 Mass Clumping (Dip Test) | mass-clumping | No | No |
| E-009 | Negative-Dimension Eigenvalue Convergence | neg-dim-eigen | No | Yes |
| E-010 | Materials Science Baselines | materials-baseline | No | No |

## Reproducibility

- **Deterministic** experiments produce identical output on every run.
- **Non-deterministic** experiments use a fixed seed (typically 42) for
  pseudorandom number generation, making them reproducible given the same
  hardware and Rust version.
- The GPU experiment (E-002) requires an NVIDIA GPU with CUDA support.

## Running all experiments

```sh
# Core algebraic experiments (no external data needed)
cargo run --release --bin motif-census -- --dims 16,32,64,128,256 --details
cargo run --release --bin zd-search -- --dim 16 --max-pairs 5000

# Experiments requiring external data
make fetch-data  # download datasets first
cargo run --release --bin real-cosmo-fit
cargo run --release --bin mass-clumping -- --n-permutations 10000 --seed 42
```
