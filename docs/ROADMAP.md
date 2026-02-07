# open_gororoba Roadmap

**Consolidated**: 2026-02-06
**Supersedes**: `RUST_MIGRATION_PLAN.md`, `RUST_REFACTOR_PLAN.md`, `docs/RESEARCH_ROADMAP.md` (all deleted)
**Companions**: `docs/TODO.md` (sprint), `docs/NEXT_ACTIONS.md` (priority queue)

This document merges the architectural evolution, crate ecosystem, migration
history, and forward-looking research plans into a single source of truth.
Every major claim becomes (1) a cited statement and (2) a reproducible test.

---

## 1. Architectural Evolution

### 1.1 Where We Started

The project began as a Python physics workbench (`gemini_physics/`) with a
monolithic Rust kernel (`gororoba_kernels`) providing hot-path acceleration
via PyO3.  The kernel contained ~325 lines of Cayley-Dickson algebra code
and ~400 lines of tests, exposed through NumPy array conversions.

**Original gororoba_kernels inventory** (historical):

| Function | Purpose | Lines |
|----------|---------|-------|
| `cd_multiply` | Cayley-Dickson multiplication | ~30 |
| `cd_conjugate` | Conjugation | ~10 |
| `cd_norm_sq` | Squared norm | ~5 |
| `cd_associator` | Associator A(a,b,c) | ~15 |
| `cd_associator_norm` | \|\|A(a,b,c)\|\| | ~5 |
| `batch_associator_norms` | Batch \|\|A\|\| | ~25 |
| `batch_associator_norms_sq` | Batch \|\|A\|\|^2 | ~20 |
| `left_mult_operator` | L_a matrix | ~20 |
| `find_zero_divisors` | 2-blade ZD search | ~35 |
| `find_zero_divisors_3blade` | 3-blade ZD search | ~40 |
| `find_zero_divisors_general_form` | Random ZD search | ~45 |
| `count_pathion_zero_divisors` | ZD counts | ~10 |
| `zd_spectrum_analysis` | Norm histogram | ~35 |
| `measure_associator_density` | Non-assoc % | ~30 |

**Original Python Sprint History**:
- Sprint 1 (Gravastar Stability): Polytropic EoS, anisotropic Bowers-Liang,
  closed claims C-008, C-022, C-023, C-077, C-078.
- Sprint 2 (CD Algebra Mass Spectra): Furey Cl(8) assignment, Tang sedenion-SU(5)
  mass ratios, pathion 32D ZD spectrum.

### 1.2 The Refactoring Path (Phase A -> D)

The migration followed four conceptual phases:

**Phase A: Extract Core Algebra to Pure Rust Library**
Split the monolithic gororoba_kernels into domain-specific crates with no
Python dependency.  Cayley-Dickson, associators, and zero-divisor search
became `algebra_core`; Clifford algebra became a separate module.

**Phase B: Python Bindings as Thin Wrapper**
Created `gororoba_py` as a thin PyO3 bridge that delegates to domain crates
rather than reimplementing algorithms.  This replaced gororoba_kernels.

**Phase C: Migrate Physics Modules**
Ported 15 Python modules to Rust domain crates in priority order:
1. Tang mass ratios, ZD spectrum analysis (already used Rust kernels)
2. Clifford algebra (pure math, no physics deps)
3. Gravastar TOV (numerical ODE)
4. Fluid dynamics, fractional Laplacian, metamaterials, optics, GR

**Phase D: CLI Binaries**
Created `gororoba_cli` with 25 analysis binaries covering algebra, cosmology,
optics, quantum, statistics, and data fetching.

### 1.3 Where We Are Now

**Migration status**: COMPLETE.  All 15 Python modules ported to Rust.
**Blackhole C++ port**: COMPLETE.  18 gr_core modules (394 tests), 5 cosmology_core
modules (tov, eos, flrw, axiodilaton, observational), scattering, absorption,
spectral bands, synchrotron, doppler, gravitational waves, Hawking radiation,
Penrose process, coordinates, null constraint, energy-conserving integrator.
**Test count**: 1370 Rust tests + 7 doc-tests pass, 0 clippy warnings.
gororoba_kernels removed (2026-02-06).
**GPU compute**: CUDA ultrametric engine via cudarc 0.19.1 (RTX 4070 Ti, 10M triples/test).

```
open_gororoba/
  Cargo.toml                  # Workspace root (13 members)
  crates/
    algebra_core/             # Cayley-Dickson, Clifford, E8, box-kites, Reggiani, annihilator, M3
    cosmology_core/           # TOV, spectral dimensions, bounce, dimensional geometry, observational
    materials_core/           # Metamaterials, absorbers, refractive index, optical database
    optics_core/              # GRIN solver, ray tracing, warp metrics, benchmarks
    quantum_core/             # Tensor networks, MERA, fractional Schrodinger, Casimir, Grover
    gr_core/                  # Kerr geodesics, gravastar, shadow, Schwarzschild
    stats_core/               # Frechet, bootstrap, Haar, ultrametric (local, Baire, temporal, dendrogram, GPU)
    spectral_core/            # Fractional Laplacian (periodic/Dirichlet, 1D/2D/3D)
    lbm_core/                 # Lattice Boltzmann D2Q9
    control_core/             # Control theory utilities
    data_core/                # 18 dataset providers + provenance + parsers
    gororoba_py/              # PyO3 bindings (thin wrappers -> domain crates)
    gororoba_cli/             # 25 analysis binaries + fetch-datasets
  src/gemini_physics/         # Remaining Python (visualization, notebooks)
  data/external/              # Cached datasets + PROVENANCE.local.json
  docs/                       # Claims, bibliography, insights, roadmaps
```

### 1.4 Measured Performance Gains

Rust native performance replaced both Python/NumPy and the Numba JIT path.
These measurements were taken during the migration (gororoba_kernels era):

| Operation | Python (NumPy) | Rust | Speedup | Notes |
|-----------|---------------|------|---------|-------|
| cd_multiply (dim=16) | ~50 us | ~2 us | 25x | SIMD potential to ~0.5 us |
| batch_assoc (10k, dim=16) | ~800 ms | ~30 ms | 27x | rayon potential to ~5 ms |
| ZD search (dim=32) | N/A (too slow) | ~100s | -- | rayon potential to ~10s |
| Motif census (dim=16) | ~3.1s (Numba) | ~0.58s | 5.3x | Pure Rust, no warmup |
| Motif census (dim=32) | ~2.8s (Numba) | ~0.79s | 3.5x | Pure Rust, no warmup |

---

## 2. Crate Ecosystem

### 2.1 Workspace Dependencies (Current)

All external crates are declared in the root `Cargo.toml` under
`[workspace.dependencies]` and trickle down with `workspace = true`.

#### Core Numerics
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| nalgebra | 0.33 | Linear algebra | Active: SVD, eigenvalues, matrix ops |
| ndarray | 0.16 | N-dim arrays | Active: tensor operations |
| num-complex | 0.4 | Complex numbers | Active: Clifford, materials |
| num-traits | 0.2 | Generic numeric traits | Active: trait bounds |
| num-quaternion | 1.0 | Quaternion arithmetic | Active: rotation representations |
| num-bigint | 0.4 | Arbitrary precision | Active: Monster group order |

#### Algebra Extensions
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| wheel | 0.1.0 | Wheel algebra | Active: division-by-zero semantics |
| padic | 0.1.6 | P-adic numbers | Active: ultrametric computations |
| adic | 0.5.0 | P-adic arithmetic | Active: rootfinding |
| atlas-embeddings | 0.1.1 | E6/E7/E8 roots | Active: exceptional Lie groups |
| petgraph | 0.7 | Graph algorithms | Active: ZD graphs, motif census |

#### Statistics and Stochastics
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| statrs | 0.18 | Distributions | Active: Normal, chi-squared |
| rand | 0.8 | Random number generation | Active: Monte Carlo |
| rand_chacha | 0.3 | ChaCha20 PRNG | Active: deterministic seeding |
| rand_distr | 0.4 | Extended distributions | Active: sampling |
| hurst | 0.1.0 | Hurst exponent | Active: R/S method (wrap with catch_unwind) |
| diffusionx | 0.11.2 | fBm, Levy processes | Active: benchmarks only |
| kodama | 0.3.0 | Hierarchical clustering | Active: dendrograms |
| kiddo | 5.2.4 | k-d trees (AVX2) | Active: spatial queries |

#### Spectral Analysis and FFT
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| rustfft | 6.4.1 | FFT (AVX/SSE) | Active: spectral analysis |
| realfft | 3.5.0 | Real-to-complex FFT | Active: wrapper for rustfft |
| gauss-quad | 0.2.4 | Gauss-Legendre quadrature | Active: replaces Simpson |

#### Astronomical Data
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| fitsrs | 0.4.1 | FITS file reader | Active: astronomical images |
| votable | 0.7.0 | VOTable XML | Active: HEASARC catalogs |
| satkit | 0.9.3 | Satellite toolkit | Active: coordinate frames, gravity |

#### Quantum
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| quantum | 0.1.2 | Quantum simulator | Active: gate operations |
| qua_ten_net | 0.2.0 | Tensor networks | Active: MERA contraction |

#### Infrastructure
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| rayon | 1.10 | Data parallelism | Active: batch operations |
| wide | 0.7 | SIMD (f64x4) | Active: Cayley-Dickson |
| clap | 4.5 | CLI framework | Active: all binaries |
| serde/serde_json | 1.0 | Serialization | Active: provenance, configs |
| csv | 1.3 | CSV I/O | Active: data output |
| ureq | 3 | HTTP client | Active: dataset fetching |
| sha2 | 0.10 | Checksums | Active: data provenance |
| thiserror | 2.0 | Error types | Active: all crates |
| plotters | 0.3 | Visualization | Active: SVG output |

#### GPU Compute
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| cudarc | 0.19.1 | CUDA (dynamic loading) | Active: ultrametric GPU kernel (feature-gated) |

#### Elliptic Integrals
| Crate | Version | Domain | Status |
|-------|---------|--------|--------|
| ellip | 1.0.4 | Carlson symmetric + Legendre | Active: RF, RD, RJ, RC, RG |

### 2.2 Hand-Rolled Code (Justified)

These implementations STAY because no suitable crate exists or licensing prevents adoption:

| Implementation | Justification |
|---------------|---------------|
| RK4 steppers (6x) | Domain-specific structs, 15 lines each, non-stiff |
| Gram-Schmidt QR (3x) | Mezzadri Haar measure trick needs R diagonal sign |
| Frechet distance | No crate for `&[f64] -> &[f64]` signature |
| Bootstrap CI | Tightly integrated with claims framework |
| Hurst (fractal_analysis.rs) | `hurst` crate is GPL-3.0 (license conflict) |
| Cosmological distances | Domain-specific (Macquart DM), stale crate alternatives |

### 2.3 Crates to Evaluate (Future Work)

| Crate | Version | Use Case | Blocker |
|-------|---------|----------|---------|
| scilib | 1.0.0 | Bessel, Legendre, Ylm | Zero deps; evaluate for GR expansion |
| anofox-statistics | 0.4.1 | Permutation tests, block bootstrap | rand 0.8 compat needed |
| hifitime | 4.2.4 | TAI/UTC/TDB time scales | Needed for satellite data |
| argmin | latest | L-BFGS-B optimization | Evaluate for cosmology fitting |

---

## 3. Test Strategy

### 3.1 Current State

1370 Rust unit tests pass across 13 crates, plus 7 doc-tests (1377 total).
0 clippy warnings (warnings-as-errors is non-negotiable).

### 3.2 Testing Layers

**Unit tests**: Algebraic identities, known values, edge cases.
Every Rust module has `#[cfg(test)]` inline tests.

**Property-based tests** (proptest): Algebraic invariants that must hold
for all inputs.  Pattern:

```rust
proptest! {
    // CD multiplication is norm-multiplicative for division algebras
    #[test]
    fn cd_norm_multiplicative_octonion(a in vec_f64(8), b in vec_f64(8)) {
        let ab = cd_multiply(&a, &b);
        let norm_ab = cd_norm(&ab);
        let norm_a_norm_b = cd_norm(&a) * cd_norm(&b);
        prop_assert!((norm_ab - norm_a_norm_b).abs() < 1e-10);
    }

    // Quaternions are associative
    #[test]
    fn quaternion_associative(a in vec_f64(4), b in vec_f64(4), c in vec_f64(4)) {
        let assoc = cd_associator(&a, &b, &c);
        let norm = cd_norm(&assoc);
        prop_assert!(norm < 1e-10);
    }
}
```

**Benchmark suite** (criterion): Performance regression detection.  Pattern:

```rust
fn cd_multiply_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_multiply");
    for dim in [4, 8, 16, 32, 64] {
        group.bench_function(format!("dim_{}", dim), |b| {
            let a = random_element(dim);
            let x = random_element(dim);
            b.iter(|| cd_multiply(&a, &x))
        });
    }
    group.finish();
}
```

**Analysis binaries**: 25 executable experiments that produce CSV artifacts
and can be re-run for reproducibility.

### 3.3 Quality Gates

```bash
# Non-negotiable before any commit:
cargo clippy --workspace -j$(nproc) -- -D warnings
cargo test --workspace -j$(nproc)

# Additional gates:
make ascii-check          # No Unicode in source
python -m pytest tests/   # Python integration tests
```

---

## 4. Current Sprint

### 4.1 Blockers (Source-First Research)

No deletions; only append/clarify sources:

- [ ] Wheels (division-by-zero) vs wheel graphs vs wheeled operads (CX-017)
- [ ] de Marrais primary sources + terminology alignment (CX-002)
- [ ] Reggiani alignment: Z(S) vs ZD(S) repo terminology (CX-003)
- [ ] Fractional Laplacian sources: Riesz vs spectral vs extension (CX-004)
- [ ] p-adic operator sources: Vladimirov/Kozyrev (CX-006)
- [ ] Exceptional/Jordan/Albert references to correct overclaims (CX-007)

### 4.2 Implementation

- [ ] Wheels axioms checker + unit tests (CX-017)
- [ ] XOR-balanced search extension + tests (CX-003)
- [ ] Motif census: extend to 64D/128D exact and 256D sampled + plot (CX-002)
- [ ] Visualization hygiene and artifact saving policy (CX-019)
- [ ] Fast basis-element multiplication table generator (16D/32D) with cache + checksum

### 4.3 Dataset Pillars (Active)

Major providers done: Pantheon+, Union3, DESI, GWOSC, Fermi GBM, Gaia, SDSS,
NANOGrav, Planck.

Remaining:
- [ ] `fetch-datasets --pillar {candle,map,image,...}` grouping flags
- [ ] Parser-level schema checks for every newly added dataset
- [ ] Deterministic row-count / column-integrity tests per provider
- [ ] Benchmark scripts for parser throughput on large catalogs
- [ ] Link dataset pillars to claim IDs in CLAIMS_EVIDENCE_MATRIX.md

See `docs/ULTRA_ROADMAP.md` Section H for the full granular checklist.

---

## 5. GR Module Expansion (Blackhole C++ Port)

### 5.1 Context

The Blackhole C++ codebase is an EXTERNAL repository containing verified GR
computations.  It is NOT checked into this repo.  The port brings these
computations into `crates/gr_core/` as pure Rust, with full test coverage
and provenance to the original C++ source.

**STATUS: COMPLETE** (2026-02-06).  All 6 layers (30 tasks) finished.
18 gr_core modules with 394 tests, plus 5 cosmology_core modules.
See gr_core/src/lib.rs for the full module listing.

### 5.2 Dependency Graph

The port follows a strict dependency ordering.  Tasks reference the project
task list (see `Task #N` identifiers).

```
Layer 0 (Foundation):
  #48  Research crates for GR tensor computation and elliptic integrals
  #63  Port Blackhole constants.h physical constants to gr_core
  #64  Create gr_core module organization plan (mod.rs structure)

Layer 1 (Core GR):
  #37  Port connection.h (generic Christoffel computation)    [needs #48]
  #28  Port Schwarzschild Christoffel symbols                  [needs #48, #37]
  #30  Port Carlson elliptic integrals to algebra_core         [needs #48]

Layer 2 (Kerr Family):
  #29  Port Kerr Christoffel symbols + full metric             [needs #48, #37, #28]
  #38  Port coordinate systems module                          [needs #37]
  #39  Port Kerr-Newman and Kerr-de Sitter verified modules    [needs #29]

Layer 3 (Physics on Metrics):
  #33  Port Novikov-Thorne accretion disk                      [needs #30, #29]
  #34  Port Penrose process                                    [needs #29]
  #35  Port Doppler beaming + relativistic effects             [needs #29, #38]
  #36  Port synchrotron radiation model                        [needs #29]
  #40  Port energy-conserving geodesic integrator (verified/)  [needs #46, #37]
  #46  Port null constraint checker (verified/)                [needs (standalone)]

Layer 4 (Applications):
  #31  Port gravitational wave calculations
  #32  Port Hawking radiation
  #41  Port nubhlight GRMHD metric/connection                 [needs #37, #38]
  #44  Port Blackhole GPU ray-tracer (CPU version)             [needs #29, #33, #35, #45]
  #45  Port absorption models                                  [needs #36, #35]

Layer 5 (Stellar / Cosmology):
  #42  Port CompactStar TOV equations and EOS models           [needs #67]
  #43  Port Blackhole cosmology module (FLRW + perturbation)   [needs (standalone)]
  #60  Port parameter_adjustment.h optimization                [needs (standalone)]
  #61  Port LUT infrastructure                                 [needs #29]
  #62  Port noise.cpp/h stochastic models to stats_core        [needs (standalone)]
  #65  Port axiodilaton.h (dilaton gravity)                    [needs #37]
  #66  Port cosmology.hpp (FRW metrics)                        [needs #43]
  #67  Port TOV with tidal deformability (tov.h)               [needs #68]
  #68  Port verified EOS functions (eos.hpp)                   [needs (standalone)]

Layer 6 (Formal Verification):
  #47  Port Rocq proof infrastructure references               [needs #39, #40, #46]
  #49  Evaluate per-component ODE tolerances                   [needs (standalone)]
```

### 5.3 Crate Research Required

Before implementing any C++ port task, search crates.io/lib.rs:

| Topic | What to search | Likely outcome |
|-------|---------------|----------------|
| Christoffel symbols | Tensor computation crates | Domain-specific; likely hand-roll |
| Carlson elliptic integrals | `elliptic`, `special-fun`, `scilib` | Evaluate scilib 1.0 first |
| GRMHD | Magneto-hydrodynamic crates | Nothing suitable; port from C++ |
| EOS tables | Nuclear EOS crates | Nothing suitable; port from C++ |
| Geodesic integration | ODE solver crates | Use ode_solvers 0.6.1 (Dopri5) |

### 5.4 Target Module Structure

```rust
// crates/gr_core/src/lib.rs (proposed expansion)
pub mod kerr;              // Kerr metric geodesics (EXISTING)
pub mod shadow;            // Black hole shadow (EXISTING)
pub mod gravastar;         // Gravastar structure (EXISTING)
pub mod schwarzschild;     // Schwarzschild Christoffel (NEW from #28)
pub mod connection;        // Generic Christoffel computation (NEW from #37)
pub mod coordinates;       // Coordinate system transforms (NEW from #38)
pub mod kerr_newman;       // Kerr-Newman family (NEW from #39)
pub mod novikov_thorne;    // Accretion disk model (NEW from #33)
pub mod penrose;           // Penrose process (NEW from #34)
pub mod geodesic;          // Energy-conserving integrator (NEW from #40)
pub mod synchrotron;       // Synchrotron radiation (NEW from #36)
pub mod hawking;           // Hawking radiation (NEW from #32)
pub mod grmhd;             // GRMHD connections (NEW from #41)
pub mod constants;         // Physical constants (NEW from #63)
```

---

## 6. Research Workstreams (Forward-Looking)

### 6.1 Quantum F4 Circuit Design
- **Goal**: Implement full F4 Weyl group operations on the 48-root quantum state.
- **Next**: Translate F4 permutation group into CNOT/SWAP gate sequence.
- **Crate**: `quantum` (0.1.2) for gate simulation.

### 6.2 Holographic Entropy Simulation
- **Goal**: Run recursive entropy PDE on a hyperbolic tensor network.
- **Next**: Map entropy PDE onto hyperbolic graph in `quantum_core/`.
- **Crate**: `qua_ten_net` (0.2.0) for tensor network contraction.

### 6.3 Publication Pipeline
- **Goal**: Assemble whitepaper from verified computational results.
- **Next**: Generate PDF from `MATHEMATICAL_FORMALISM.tex`.
- **Prerequisite**: Complete source-first research passes (Section 4.1).

### 6.4 Cross-Domain Ultrametric Analysis -- COMPLETE (I-011)
- **Result**: GPU exploration (10M triples x 1000 permutations x 9 catalogs).
  82/472 tests significant at BH-FDR<0.05.
- **Key finding**: Ultrametricity is NOT radio-transient-specific (old I-008 was wrong).
  Galactic kinematics dominates (60/82 hits from Hipparcos + Gaia proper motions).
- **Catalogs with signal**: Hipparcos (48/114), Gaia (12/114), CHIME (8/8),
  ATNF (6/8), GWOSC (4/66), SDSS (2/52), Pantheon+ (2/22).
- **No signal**: McGill magnetars (N=20 too small), Fermi GBM (isotropic).
- **Data**: `data/csv/c071g_exploration.csv`, `data/csv/c071g_exploration_gpu_10M_1000perm.csv`.
- **Crate**: `stats_core/ultrametric/gpu.rs` (CUDA kernel via cudarc 0.19.1).

### 6.5 Standard Model Embedding (Speculative)
- **Goal**: Embed SU(3) x SU(2) x U(1) generators into G2 subalgebra of F4.
- **Prerequisite**: Complete Reggiani alignment (Section 4.1).
- **Status**: Tracked in CLAIMS_EVIDENCE_MATRIX.md as speculative.

---

## 7. Remaining Workstreams from ULTRA_ROADMAP.md

These sections are tracked in detail in `docs/ULTRA_ROADMAP.md` and summarized
here for completeness.

### 7.1 Reproducibility & Quality Gates (Section A)
- [x] Package gemini_physics, pytest, ruff, CI, artifact verifier
- [ ] Extend ruff linting phased to all `src/*.py`

### 7.2 Claims -> Evidence (Section B)
- [ ] Primary-source citation for every claim in CLAIMS_EVIDENCE_MATRIX.md
- [ ] Reggiani Z(S)/ZD(S) terminology alignment

### 7.3 Materials Science (Section D)
- [ ] Second dataset backend (OQMD or NOMAD) for cross-validation
- [ ] Element-property featurization and predictive baselines

### 7.4 Coq/Rocq Formalization (Section F)
- [x] Buildable stub + compile workflow
- [ ] Decide semantics for `has_right`/`reachable_delegation`
- [ ] Prove a minimal non-trivial theorem end-to-end

### 7.5 Documentation & Paper (Section G)
- [ ] "Paper-ready" LaTeX build pipeline (`make latex`)
- [ ] Structured "hypotheses + tests + results" format

### 7.6 Dataset Pillars (Section H)
Full granular checklist in `docs/ULTRA_ROADMAP.md` Section H.
Major providers operational; schema checks and benchmarks remain.

---

## 8. Long-Term Vision

| Horizon | Goal | Dependencies | Status |
|---------|------|--------------|--------|
| ~~Near~~ | ~~Complete GR module Layers 0-6~~ | ~~Task #48 crate research~~ | **DONE** (2026-02-06) |
| ~~Near~~ | ~~Cross-domain ultrametric analysis (9 catalogs)~~ | ~~Dataset providers~~ | **DONE** (I-011, 82/472 sig) |
| ~~Near~~ | ~~GPU acceleration (CUDA ultrametric kernel)~~ | ~~cudarc 0.19.1~~ | **DONE** (RTX 4070 Ti) |
| Near | Schema checks for all 18 dataset providers | Parser code in data_core | In progress |
| Near | Extend motif census to 64D/128D | algebra_core | Planned |
| Medium | WASM target for browser visualization | Crate compatibility audit | Future |
| Medium | Hardware quantum circuit run (IBM Eagle 127q) | F4 circuit design | Future |
| Medium | Publication pipeline (LaTeX from verified results) | Source-first research | Future |
| Medium | GPU tensor network contraction (cudarc) | Extend GPU path beyond ultrametric | Future |

---

## 9. External References

- [Scientific Computing in Rust](https://scientificcomputing.rs/monthly/2025-12)
- [Rewrite it in Rust: Computational Physics](https://arxiv.org/abs/2410.19146)
- [Baez: Octonions](https://math.ucr.edu/home/baez/octonions/)
- [Reggiani: arXiv:2411.18881](https://arxiv.org/abs/2411.18881) (Sedenion zero-divisor manifold)
- [de Marrais: Box-kite geometry](https://arxiv.org/abs/math-ph/0207006)

---

## Appendix: Document Lineage

This document synthesizes and supersedes:

| Original | Lines | Key Contribution | Disposition |
|----------|-------|-------------------|-------------|
| `RUST_MIGRATION_PLAN.md` | 268 | Crate architecture, phase structure, test strategy | **Deleted** (absorbed into Sections 1-3) |
| `RUST_REFACTOR_PLAN.md` | 293 | Sprint history, benchmarks, proptest/criterion patterns | **Deleted** (absorbed into Sections 1, 3) |
| `docs/RESEARCH_ROADMAP.md` | 41 | Research workstreams, future directions | **Deleted** (absorbed into Section 6) |
| `docs/engineering/GEMINI_FUNCTION_MIGRATION_LEDGER.md` | 130 | Codex migration audit (inaccurate) | **Deleted** (wrapper anti-pattern throughout) |
| `docs/ULTRA_ROADMAP.md` | 112 | Granular claim-to-evidence checklists | Active companion (Section 7 detail) |
| `docs/TODO.md` | 41 | Current sprint tracker | Active companion (sprint-level) |
| `docs/NEXT_ACTIONS.md` | 60 | Priority queue | Active companion (top-5) |

`docs/ULTRA_ROADMAP.md`, `docs/TODO.md`, and `docs/NEXT_ACTIONS.md` continue
as active companions for granular tracking.
