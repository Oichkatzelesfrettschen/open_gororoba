<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/roadmap.toml, registry/roadmap_narrative.toml -->

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
**Claims infrastructure**: COMPLETE.  Rust claims-audit + claims-verify binaries
replace 11 Python scripts.  118 backfill items confirmed Verified (2026-02-07).
**Test count**: 2239 Rust tests pass (unit + doc + integration), 0 clippy warnings.
gororoba_kernels removed (2026-02-06).
**GPU compute**: CUDA ultrametric engine via cudarc 0.19.1 (RTX 4070 Ti, 10M triples/test).
**Claims**: 477 (423 verified, 38 refuted, 2 partial, 1 inconclusive, 13 closed).
**Insights**: I-001..I-017. **Experiments**: E-001..E-013. **Binaries**: 40.

```
open_gororoba/
  Cargo.toml                  # Workspace root (15 members)
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
    docpipe/                  # PDF/document extraction pipeline
    gororoba_cli/             # 40 analysis binaries + fetch-datasets
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

### 1.5 Documentation Registry Evolution (Sprint 13)

The repository now maintains a central markdown knowledge index:

- `registry/knowledge_sources.toml` (generated, deterministic index of all tracked markdown files)
- Generator: `src/scripts/analysis/build_knowledge_sources_registry.py`
- Central corpus migration: `registry/knowledge/docs/*.toml` with manifest at
  `registry/knowledge/documents.toml`
- Build hook: `make registry` now runs markdown indexing, TOML corpus capture,
  TOML-driven mirror export, mirror freshness verification, and `registry-check`
- Claims support normalization is now TOML-first:
  - `registry/claims_tasks.toml` (authoritative)
  - `registry/claims_domains.toml` (authoritative)
  - `registry/claim_tickets.toml` (authoritative)
  - Legacy ingest path retained as explicit maintenance command:
    `make registry-ingest-legacy`
- Generated mirrors now include:
  - `docs/generated/CLAIMS_TASKS_REGISTRY_MIRROR.md`
  - `docs/generated/CLAIMS_DOMAINS_REGISTRY_MIRROR.md`
  - `docs/generated/CLAIM_TICKETS_REGISTRY_MIRROR.md`
- Legacy claims-support files are now generated from TOML:
  - `docs/CLAIMS_TASKS.md`
  - `docs/claims/INDEX.md`
  - `docs/claims/CLAIMS_DOMAIN_MAP.csv`
  - `docs/claims/by_domain/*.md`
- Registry generators now emit deterministic stamps to avoid one-time event churn
  across repeated `make registry` runs.

Current policy:

- TOML remains the machine-parseable source of truth for structured registries.
- Markdown remains human-facing narrative and mirror material.
- Mirror docs should be generated from TOML where possible to reduce drift.

Immediate next steps:

1. TOML -> markdown exporters are now in place under `docs/generated/`.
2. Claims-support legacy docs are generated from TOML during `make registry`.
3. Finalize governance for remaining narrative overlays (`docs/INSIGHTS.md`,
   `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`).
4. Keep generated markdown explicitly marked with regeneration commands.

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

1628 Rust tests pass across 13 crates (unit + integration + doc-tests), 2 GPU-gated ignored.
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

- [x] Wheels (division-by-zero) vs wheel graphs vs wheeled operads (CX-017) -- DONE
- [x] de Marrais primary sources + terminology alignment (CX-002) -- DONE
- [x] Reggiani alignment: Z(S) vs ZD(S) repo terminology (CX-003) -- DONE
- [x] Fractional Laplacian sources: Riesz vs spectral vs extension (CX-004) -- DONE
- [x] p-adic operator sources: Vladimirov/Kozyrev (CX-006) -- DONE
- [x] Exceptional/Jordan/Albert references to correct overclaims (CX-007) -- DONE

### 4.2 Implementation

- [x] Wheels axioms checker + unit tests (CX-017) -- DONE
- [x] XOR-balanced search extension + tests (CX-003) -- DONE (10 regression tests)
- [x] Motif census: extend to 64D/128D/256D exact + scaling laws (CX-002) -- DONE (16 regression tests)
- [x] Visualization hygiene and artifact saving policy (CX-019) -- DONE
- [ ] Fast basis-element multiplication table generator (16D/32D) with cache + checksum

### 4.3 Dataset Pillars (Active)

Major providers done: Pantheon+, Union3, DESI, GWOSC, Fermi GBM, Gaia, SDSS,
NANOGrav, Planck.

Remaining:
- [x] `fetch-datasets --pillar {candle,map,image,...}` grouping flags -- DONE
- [x] Parser-level schema checks for every newly added dataset -- DONE (21 tests)
- [x] Deterministic row-count / column-integrity tests per provider -- DONE (D3 Sprint 5)
- [x] Benchmark scripts for parser throughput on large catalogs -- DONE (Section H)
- [x] Link dataset pillars to claim IDs in CLAIMS_EVIDENCE_MATRIX.md -- DONE
- [x] Validate all fetch-datasets providers against live endpoints -- DONE (2026-02-07)
  - 22/30 providers tested, all pass. 3 broken URLs fixed (WMM, GGM05S, GRACE-FO).
  - 8 large providers (multi-GB) are on-demand only.

See `docs/ULTRA_ROADMAP.md` Section H for the full granular checklist (21/21 complete).

### 4.4 New Concept Sources (CX-026..028, Sprint 6)

Download papers, results, and datasets for newly extracted concepts:
- [x] CX-026: Quantum inequalities (Ford-Roman-Pfenning) -- DONE (4 papers)
  - Papers: Pfenning & Ford (1997), Fewster (2012), Kontou & Sanders (2020), Fewster & Roman (2003)
  - Data: QI constant C=3/(32pi^2), tau^4 scaling verification
- [x] CX-027: Light-ion QGP (O-O/Ne-Ne) -- DONE (4 papers + HEPData RAA)
  - Papers: CMS (2510.09864), Mazeliauskas (2509.07008), Arleo & Falmagne (2411.13258), Pablos & Takacs (2509.19430)
  - Data: CMS O-O RAA 44 pT bins (3-103.6 GeV), ALICE Pb-Pb 0-5% baseline
  - Dataset: `data/external/cms_oo_raa/` (4 CSV files from HEPData ins3068407)
- [x] CX-028: Subluminal positive-energy warp drives -- DONE (3 new + 2 existing)
  - Papers: Santiago-Schuster-Visser (2105.03079), Fuchs et al. (2407.18908), Finazzi et al. (0904.0141)
  - Already had: Bobrick & Martire (2021), Lentz (2021), Alcubierre (2000), Smolyaninov (2010)
  - Data: Energy estimates, bubble thickness constraints (from paper text)

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
- ~~Extend ruff linting phased to all `src/*.py`~~ -- Obsolete (Python migration complete; only viz scripts remain)

### 7.2 Claims -> Evidence (Section B)
- [ ] Primary-source citation for every claim in CLAIMS_EVIDENCE_MATRIX.md
- [x] Reggiani Z(S)/ZD(S) terminology alignment -- DONE (CX-003, Sprint 4)

### 7.3 Materials Science (Section D)
- [x] Second dataset backend: AFLOW (aflowlib.duke.edu) -- full database via AFLUX REST API
- [x] JARVIS-DFT registered as DatasetProvider in fetch-datasets CLI
- [x] Magpie-style composition featurizer (54-dim: 10 properties x 5 stats + 4 global)
- [x] OLS linear regression baselines (SVD-based, formation energy + band gap)
- [x] materials-baseline CLI binary (cross-database comparison)

### 7.4 Coq/Rocq Formalization (Section F)
- [x] Buildable stub + compile workflow
- [ ] Decide semantics for `has_right`/`reachable_delegation`
- [ ] Prove a minimal non-trivial theorem end-to-end

### 7.5 Documentation & Paper (Section G)
- [ ] "Paper-ready" LaTeX build pipeline (`make latex`)
- [ ] Structured "hypotheses + tests + results" format

### 7.6 Dataset Pillars (Section H) -- COMPLETE
Full granular checklist in `docs/ULTRA_ROADMAP.md` Section H (21/21 done).
30 dataset providers, 6 EHT multi-format bundles (UVFITS+CSV+TXT), 3 URL fixes applied.

### 7.7 Finite Projective Geometry Integration (Sprint 7)

Maps Cayley-Dickson zero-divisor motif structure to PG(n-2,2) finite projective spaces:

- [x] PG(m,2) finite projective space implementation (points, lines, incidence)
- [x] PG-to-motif-component bijection via XOR-key extraction
- [x] Graph invariants on MotifComponent (adjacency matrix, spectrum, diameter, girth)
- [x] GF(2)-linear bit-predicate for motif class separation
- [x] Sign-twist cancellation predicate formalization

**Claim coverage**: C-444 (bijection), C-445 (bit-predicate), C-446 (sign-twist)
**Insight**: I-013 (PG explains scaling laws)

### 7.8 Adaptive Permutation Testing (Sprint 7)

Besag-Clifford (1991) sequential stopping for the ultrametric pipeline:

- [x] Null model abstraction (ColumnIndependent, RowPermutation, ToroidalShift, RandomRotation)
- [x] Adaptive permutation testing with binomial CI stopping rule
- [x] Attribute subset search extracted to library function
- [x] Squared-distance optimization (CPU + GPU, ~30% speedup)

**Claim coverage**: C-447 (type-I error preservation)
**Insight**: I-014 (squared-distance optimization)

---

### 7.9 Lattice Codebook Filtration (Sprint 9)

**Goal**: Implement and verify the monograph's 8 falsifiable theses (A-H) about
Cayley-Dickson lattice codebook structure.

| Thesis | Topic | Claim | Status |
|--------|-------|-------|--------|
| A | Codebook parity | C-458 | **Verified** (4 dims) |
| B | Filtration nesting | C-459 | **Verified** (strict subsets) |
| C | Prefix-cut characterization | C-460 | **Verified** (lex prefix!) |
| D | Scalar shadow action | C-465, C-466 | **Partial** (addition verified, multiplication open) |
| E | XOR partner law | C-462 | **Verified** (dim=64) |
| F | Parity-clique law | C-463 | **Verified** (dims 16, 32) |
| G | Spectral fingerprints | C-464 | **Verified** (known graphs) |
| H | Null-model identity | C-467 | **Verified** (3 tests) |

**Novel discovery**: All filtration transitions are lexicographic prefix cuts
(simpler than anticipated decision-trie rules). S_base = 2187 = 3^7.

**Conversation extraction**: 12 files in convos/ processed; 3 new markdown
extracts in docs/external_sources/ (inverse CD, wheel taxonomy, sedenion ZD).

**Open question**: Multiplication coupling rho(b) in GL(8,Z) (C-466).

**Insight**: I-015 (monograph theses verification)
**Test count**: ~1690+ (20 new cd_external + 3 stats_core)

### 7.10 De Marrais Emanation Architecture (Sprint 10)

**Goal**: Implement the full de Marrais "lacuna map" (L1-L18) as a single
coherent Rust module covering emanation tables, twist mechanics, lanyard
taxonomy, PSL(2,7) navigation, semiotic geometry, and brocade normalization.

**Module**: `crates/algebra_core/src/emanation.rs` (~4400 lines, 113 tests)

| Layer | Items | Topic | Claims |
|-------|-------|-------|--------|
| Product engine | L1 | CDP signed products, quadrant recursion | C-468 |
| Table generation | L2-L4 | Tone rows, ETs, strutted ETs, sparsity | C-469 |
| Twist mechanics | L5-L6 | H*/V* operations, PSL(2,7) graph | -- |
| Lanyard taxonomy | L7-L8 | Sails, tray-racks, blues, quincunx, Trip Sync | -- |
| Semiotic geometry | L9-L14 | Strut-opposite kernel, CT boundary, loop duality, Sky, Eco | C-475 |
| Orientation | L15-L18 | Oriented Trip Sync, signed graph, delta, brocade | C-470..C-474 |

**Key findings**:
- DMZ geometry is sign-concordance (12 edges/BK), not octahedral adjacency (9)
- Sail-loop duality is combinatorial (Fano incidence), not dynamical (twist orbits)
- All 7 BKs admit PSL(2,7) shorthand embedding (oriented Trip Sync universal)
- Delta reachability matches twist reachability but pair-level correspondence is nuanced

**Open questions**: twist-delta pair correspondence, full lanyard classification,
brocade CPO count explanation.

**Insight**: I-016 (De Marrais Emanation Architecture)
**Test count**: 1806 (113 new emanation + prior 1693)
**Claims**: C-468..C-475 (8 new, total 475)

---

## 8. Long-Term Vision

| Horizon | Goal | Dependencies | Status |
|---------|------|--------------|--------|
| ~~Near~~ | ~~Complete GR module Layers 0-6~~ | ~~Task #48 crate research~~ | **DONE** (2026-02-06) |
| ~~Near~~ | ~~Cross-domain ultrametric analysis (9 catalogs)~~ | ~~Dataset providers~~ | **DONE** (I-011, 82/472 sig) |
| ~~Near~~ | ~~GPU acceleration (CUDA ultrametric kernel)~~ | ~~cudarc 0.19.1~~ | **DONE** (RTX 4070 Ti) |
| ~~Near~~ | ~~Schema checks for all 18 dataset providers~~ | ~~Parser code in data_core~~ | **DONE** (21 tests, 3 URL fixes) |
| ~~Near~~ | ~~Extend motif census to 64D/128D/256D exact~~ | ~~algebra_core~~ | **DONE** (16 regression tests, 5 scaling laws) |
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
