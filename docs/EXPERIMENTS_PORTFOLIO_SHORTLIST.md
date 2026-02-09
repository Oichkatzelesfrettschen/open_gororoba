# Experiments Portfolio Shortlist

Source-of-truth policy:
- Authoritative machine-readable registry: `registry/experiments.toml`
- TOML-driven markdown mirror: `docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md`
- This file remains narrative detail and historical context.

Status: Sprint 8 (2026-02-07)

This document lists the primary reproducible experiments in the open_gororoba
workbench, each with a unique E-ID, method summary, input data, output
artifacts, run command, and related claims.

All experiments are implemented as Rust binaries under `crates/gororoba_cli/`.
Run `cargo run --release --bin <name> -- --help` for full options.

---

## E-001: Cayley-Dickson Motif Census

**Method:** Compute connected-component structure of the diagonal zero-product
graph for cross-assessor pairs at each Cayley-Dickson doubling level
(dim=16,32,64,128,256).  Exact enumeration via `cross_assessors()` and
`motif_components_for_cross_assessors()`.

**Input:** None (purely algebraic, generated from CD multiplication table).

**Output:** `data/csv/motif_census_dim{N}.csv`, `data/csv/motif_census_summary.csv`

**Run:**
```
cargo run --release --bin motif-census -- --dims 16,32,64,128,256 --details
```

**Related claims:** C-100..C-110 (box-kite scaling laws), CX-003 (XOR-balanced search)

**Determinism:** Fully deterministic (no RNG).

---

## E-002: Multi-Dataset GPU Ultrametric Sweep

**Method:** For each of 9 astrophysical catalogs, normalize attributes to [0,1],
compute Euclidean distance matrices, run ultrametric fraction test against
column-shuffled null.  GPU-accelerated: 10M triples x 1000 permutations per
attribute subset.  BH-FDR correction across all tests.

**Input:** `data/external/` (CHIME/FRB, ATNF, GWOSC, Pantheon+, Gaia DR3,
SDSS DR18, Fermi GBM, Hipparcos, McGill).  Downloaded by `fetch-datasets`.

**Output:** `data/csv/c071g_multi_dataset_ultrametric.csv`

**Run:**
```
cargo run --release --bin multi-dataset-ultrametric -- \
    --explore --n-triples 10000000 --n-permutations 1000
```

**Related claims:** C-071, C-436..C-440, I-011

**Determinism:** Seed-controlled (default 42).  Requires NVIDIA GPU with CUDA.

---

## E-003: Real Cosmological Fit (Pantheon+ / DESI BAO)

**Method:** Joint chi-square minimization over 1578 Pantheon+ SNe Ia and 7
DESI DR1 BAO bins.  Analytic M_B marginalization (Conley+ 2011).  Bounded
Nelder-Mead optimizer.  Lambda-CDM vs w0-CDM model comparison via delta-BIC.

**Input:** `data/external/Pantheon+SH0ES.dat`, DESI DR1 BAO (hardcoded).

**Output:** Stdout (parameters, chi2/dof, AIC, BIC, delta-BIC).

**Run:**
```
cargo run --release --bin real-cosmo-fit
```

**Related claims:** C-200..C-210

**Determinism:** Fully deterministic (no RNG in optimizer).

---

## E-004: Kerr Shadow Boundaries

**Method:** Compute Bardeen shadow boundary curve for a Kerr black hole at
given spin and inclination.  Outputs (alpha, beta) celestial coordinates.

**Input:** None (analytic calculation).

**Output:** CSV to stdout or `--output` file.

**Run:**
```
cargo run --release --bin kerr-shadow -- --spin 0.998 --n-points 1000 --inclination 17
```

**Related claims:** C-301..C-310 (Kerr GR)

**Determinism:** Fully deterministic.

---

## E-005: Zero-Divisor Graph Invariants

**Method:** Build the interaction graph of sedenion zero-divisors, compute
graph-theoretic invariants (diameter, components, clustering, degree
distribution).  Extend to dim=32 (pathions).

**Input:** None (purely algebraic).

**Output:** Graph analysis summary to stdout.

**Run:**
```
cargo run --release --bin zd-search -- --dim 16 --max-pairs 5000
```

**Related claims:** C-050..C-060 (ZD graph structure)

**Determinism:** Fully deterministic.

---

## E-006: Gravastar TOV Parameter Sweep

**Method:** Solve the Tolman-Oppenheimer-Volkoff equation for a three-layer
gravastar (de Sitter vacuum + stiff shell + Schwarzschild exterior) across
a grid of polytropic indices and target masses.

**Input:** None (parametric sweep).

**Output:** CSV to `--output` file.

**Run:**
```
cargo run --release --bin gravastar-sweep -- \
    --n-gamma 32 --n-mass 32 --output data/csv/gravastar_sweep.csv
```

**Related claims:** C-400..C-410 (gravastar stability)

**Determinism:** Fully deterministic (ODE integration).

---

## E-007: Tensor Network / PEPS Entropy

**Method:** Classical tensor network simulator for quantum circuits.  Bell/GHZ
state preparation, random circuit evolution, entanglement entropy via SVD
bipartition.  PEPS boundary MPS entropy for 2D systems.

**Input:** None (synthetic quantum circuits).

**Output:** CSV or JSON to stdout or `--output` file.

**Run:**
```
cargo run --release --bin tensor-network -- \
    scaling --n-min 2 --n-max 12 --output data/csv/entropy_scaling.csv
```

**Related claims:** C-350..C-360 (entanglement scaling)

**Determinism:** Seed-controlled (default 42).

---

## E-008: GWTC-3 Mass Clumping (Dip Test)

**Method:** Hartigan dip test (Hartigan & Hartigan, 1985) for multimodality
of the BBH primary mass distribution.  Permutation-based p-value with 10000
draws from U(0,1).  Applied to mass_1_source, mass_2_source, chirp_mass_source.

**Input:** `data/external/gwosc_all_events.csv` (219 events, O1-O4a combined).

**Output:** `data/csv/gwtc3_mass_clumping_dip.csv`

**Run:**
```
cargo run --release --bin mass-clumping -- --n-permutations 10000 --seed 42
```

**Related claims:** C-007 (mass distribution multimodality)

**Determinism:** Seed-controlled (default 42).

---

## E-009: Negative-Dimension Eigenvalue Convergence

**Method:** Compute eigenvalues of H = T(k) + V(x) with regularized fractional
kinetic operator T(k) = (|k| + epsilon)^alpha for alpha < 0.  Imaginary-time
evolution on a spectral grid.  Sweep epsilon -> 0 to study convergence.

**Input:** None (parametric computation).

**Output:** CSV to `--output` file.

**Run:**
```
cargo run --release --bin neg-dim-eigen -- sweep \
    --alpha -1.5 --eps-start 0.5 --eps-end 0.01 --eps-steps 20 \
    --output data/csv/neg_dim_convergence.csv
```

**Related claims:** C-420..C-425 (negative-dimension PDE)

**Determinism:** Fully deterministic.

---

## E-010: Materials Science Baselines (JARVIS + AFLOW)

**Method:** Load JARVIS-DFT and AFLOW datasets, featurize compounds with
Magpie-style composition descriptors, run OLS linear regression baselines
for formation energy and band gap prediction.  80/20 train/test split.

**Input:** `data/external/` (JARVIS JSON, AFLOW CSV).  Downloaded by `fetch-datasets`.

**Output:** Stdout (MAE, R^2, feature importance).

**Run:**
```
cargo run --release --bin materials-baseline -- --data-dir data/external --seed 42
```

**Related claims:** C-450..C-455 (materials baselines)

**Determinism:** Seed-controlled (default 42).

---

## Summary Table

| E-ID  | Name                        | Binary                       | Det.  | GPU |
|-------|-----------------------------|------------------------------|-------|-----|
| E-001 | Motif Census                | motif-census                 | Yes   | No  |
| E-002 | GPU Ultrametric Sweep       | multi-dataset-ultrametric    | Seed  | Yes |
| E-003 | Real Cosmo Fit              | real-cosmo-fit               | Yes   | No  |
| E-004 | Kerr Shadow                 | kerr-shadow                  | Yes   | No  |
| E-005 | ZD Graph Invariants         | zd-search                    | Yes   | No  |
| E-006 | Gravastar TOV Sweep         | gravastar-sweep              | Yes   | No  |
| E-007 | Tensor Network Entropy      | tensor-network               | Seed  | No  |
| E-008 | Mass Clumping Dip Test      | mass-clumping                | Seed  | No  |
| E-009 | Neg-Dim Eigenvalue Sweep    | neg-dim-eigen                | Yes   | No  |
| E-010 | Materials Baselines         | materials-baseline           | Seed  | No  |
