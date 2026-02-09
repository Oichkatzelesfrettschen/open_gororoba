<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/experiments.toml, registry/experiments_narrative.toml -->

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

---

## E-011: Cross-Stack Locality Comparison (Experiment A)

Method: Compare adjacency-locality metrics across three independent constraint systems:
  Stack 1 (E10 Billiard): Wall-transition sequence from HyperbolicBilliard, E10 Dynkin graph.
    Metric: r_e8 = fraction of consecutive pairs that are E8-adjacent.
    Null: ColumnIndependentNull (uniform random generator selection).
  Stack 2 (ET DMZ Walk): Navigation sequences through de Marrais Emanation Table.
    Metric: r_dmz = fraction of consecutive ET cells that share a DMZ edge.
    Null: uniform random cell selection in the ET grid.
  Stack 3 (CD ZD Catamaran/Twist): Twist-product navigation across zero-divisor graph.
    Metric: r_zd = fraction of consecutive basis-pair transitions that are ZD-adjacent.
    Null: uniform random basis-pair selection.
  All three use the common abstract model: generator-driven piecewise geodesic dynamics
  (generator set G, constraint graph Gamma subset G x G, sequence s_1..s_n, locality
  ratio r = |{i : (s_i, s_{i+1}) in Gamma}| / (n-1)).
  Prediction (ALP, C-476): all three r values significantly exceed their nulls.
  Falsifier: one or more shows r consistent with uniform/random.
Input: None (purely algebraic / simulation)
Output: data/csv/cross_stack_locality_comparison.csv
Run:
```bash
cargo run --release --bin cross-stack-locality -- --n-bounces 10000 --n-permutations 1000 --seed 42
```

---

## E-012: ET Discrete Billiard (Experiment B)

Method: Treat the Emanation Table as a discrete billiard wall set.
  Walls: ET constraints define forbidden/allowed transitions in (S, row, col) space.
  Reflections: when a trajectory hits an ET "wall" (DMZ boundary), apply the ET rule
    update (fill/hide involution or strut-constant increment).
  Symbolic dynamics: record the sequence of wall types hit (DMZ, label-line, empty).
  Phase structure: compute Lyapunov exponent (or mixing time) as a function of strut
    constant S. Compare against spectroscopy band classification (Dense/Sparse/Mixed
    from spectroscopy_bands()).
  Prediction: spectroscopy band transitions correspond to qualitative phase changes in
    the toy billiard (e.g., Dense bands -> low Lyapunov, Sparse -> high Lyapunov).
  Falsifier: no correlation between spectroscopy classification and billiard dynamics.
Input: None (ET computation from emanation.rs)
Output: data/csv/et_billiard_phase_structure.csv
Run:
```bash
cargo run --release --bin et-billiard -- --n-levels 4,5,6 --n-steps 10000 --seed 42
```

---

## E-013: Sky-Limit-Set Correspondence (Experiment C)

Method: Compare ET skybox pattern invariants to Coxeter group limit set invariants.
  Step 1: For each CD level N=4,5,6, compute the skybox (G x G grid) and extract:
    - box-kite count (= N-1)
    - emptiness clustering exponent (connected-component size distribution)
    - DMZ density (fraction of cells that are DMZ)
    - fill/hide period-doubling structure (from Four Corners rule)
  Step 2: For candidate Coxeter groups (A_{N-1}, B_{N-1}, D_{N-1}), compute limit set
    invariants from the Coxeter matrix:
    - rank
    - fractal dimension of limit set
    - invariant measure density
  Step 3: Compare skybox invariants (Step 1) to limit set invariants (Step 2) via
    correlation and matching tests.
  Prediction (C-477): systematic correspondence between skybox pattern invariants and
    Coxeter limit set invariants at matching rank.
  Falsifier: no Coxeter group at any rank produces invariants matching the ET skybox.
Input: None (ET computation + Coxeter matrix computation)
Output: data/csv/sky_limit_set_comparison.csv
Run:
```bash
cargo run --release --bin sky-limit-set -- --n-levels 4,5,6 --coxeter-types A,B,D
```

---
