# Rust Migration Plan: open_gororoba

> **HISTORICAL**: This document is preserved for provenance.  The authoritative
> consolidated roadmap is [`docs/ROADMAP.md`](docs/ROADMAP.md).
> Migration status: COMPLETE (all 15 modules ported, 855 Rust tests, 0 clippy warnings).

## Overview

Complete migration of Python physics stack to native Rust with domain-specific crate organization.

**Current state**: 98 Rust tests in `gororoba_kernels` + 154+ Python tests in `tests/`
**Target state**: 500+ Rust tests across specialized crates, Python for visualization only

## Crate Architecture

```
open_gororoba/
  crates/
    algebra_core/           # Cayley-Dickson, Clifford, E8 lattice
    cosmology_core/         # TOV, spectral dimensions, bounce cosmology
    materials_core/         # Metamaterials, absorbers, refractive index
    optics_core/            # GRIN solver, ray tracing, warp metrics
    quantum_core/           # Tensor networks, MERA, fractional Schrodinger
    gr_core/                # Kerr geodesics, gravastar, shadow calculations
    stats_core/             # Frechet, bootstrap, Haar unitaries
    gororoba_py/            # PyO3 unified bindings (cdylib)
```

## External Crate Dependencies

### Core Numerics
| Crate | Version | Domain | Notes |
|-------|---------|--------|-------|
| nalgebra | 0.33 | Linear algebra | Already integrated |
| ndarray | 0.16 | N-dim arrays | NumPy-like API |
| ndarray-linalg | 0.17 | Linear algebra ops | SVD, eigenvalues |
| num-complex | 0.4 | Complex numbers | Already integrated |

### ODE/PDE Solvers
| Crate | Version | Domain | Notes |
|-------|---------|--------|-------|
| russell_ode | latest | Stiff ODE/DAE | Radau5 for gravastar TOV |
| ode-solvers | 0.4 | Explicit RK | Dopri5, DOP853 |
| differential-equations | 0.1 | DDE, SDE | Stochastic support |

### Tensor Networks / Quantum
| Crate | Version | Domain | Notes |
|-------|---------|--------|-------|
| quantrs2 | 0.1-rc | Tensor networks | MERA, contraction optimization |
| ndarray-einsum | latest | Einstein summation | Tensor contraction |

### Graph / Topology
| Crate | Version | Domain | Notes |
|-------|---------|--------|-------|
| petgraph | 0.7 | Graph algorithms | Already integrated |
| topology | latest | Homology | Chain complexes |

### Statistics
| Crate | Version | Domain | Notes |
|-------|---------|--------|-------|
| statrs | 0.18 | Distributions | Already integrated |
| rand_distr | 0.4 | Sampling | Extended distributions |

### Physics Simulation
| Crate | Version | Domain | Notes |
|-------|---------|--------|-------|
| aether | latest | Reference frames | Quaternions, rigid bodies |
| uom | 0.36 | Units of measure | Type-safe physics |

### Parallel / GPU
| Crate | Version | Domain | Notes |
|-------|---------|--------|-------|
| rayon | 1.10 | Data parallelism | Already integrated |
| wgpu | 23.x | GPU compute | WebGPU for tensor ops |

## Phase 1: algebra_core (PARTIALLY DONE)

### Current (gororoba_kernels)
- [x] Cayley-Dickson multiplication
- [x] Associators and norms
- [x] Zero-divisor detection (2-blade)
- [x] Cl(8) gamma matrices
- [x] Batch operations with rayon

### To Add
- [ ] E8 lattice root system (`e8_lattice.py`)
- [ ] Box-kite symmetry groups (`de_marrais_boxkites.py`)
- [ ] PSL(2,7) action on box-kites (`algebra/psl_2_7.py`)
- [ ] Nilpotent orbit classification (`nilpotent_orbits.py`)
- [ ] Group theory utilities (`group_theory.py`)
- [ ] Wheels algebra (`wheels.py`)

### Migration Path
```rust
// New: crates/algebra_core/src/lib.rs
pub mod cayley_dickson;    // from gororoba_kernels/algebra.rs
pub mod clifford;          // from gororoba_kernels/clifford.rs
pub mod e8_lattice;        // from gemini_physics/e8_lattice.py
pub mod boxkites;          // from gemini_physics/de_marrais_boxkites.py
pub mod psl_2_7;           // from gemini_physics/algebra/psl_2_7.py
pub mod nilpotent;         // from gemini_physics/nilpotent_orbits.py
```

## Phase 2: cosmology_core

### Python Sources
- `gemini_physics/cosmology.py` (519 lines)
- `gemini_physics/fractal_cosmology.py` (434 lines)
- `gemini_physics/genesis_gravastar_bridge.py` (519 lines)
- `gravastar_tov.py` (existing)

### Rust Modules
```rust
pub mod tov;               // Polytropic/anisotropic TOV (from gravastar.rs)
pub mod spectral_dim;      // Calcagni, CDT spectral dimensions
pub mod bounce;            // Bounce cosmology fitting
pub mod genesis;           // Genesis soliton extraction
pub mod friedmann;         // FLRW cosmology
```

### Key Algorithms
- TOV integration with russell_ode (stiff solver)
- Spectral dimension flow computation
- Hubble parameter fitting to Pantheon+/DESI data

## Phase 3: materials_core

### Python Sources
- `gemini_physics/metamaterial.py` (374 lines)
- `gemini_physics/cd_absorber_map.py` (387 lines)
- `gemini_physics/materials/database.py`
- `gemini_physics/materials_jarvis.py`

### Rust Modules
```rust
pub mod absorber;          // ZD-to-layer mapping
pub mod refractive;        // Complex refractive index
pub mod salisbury;         // Salisbury screen model
pub mod tcmt;              // Temporal coupled mode theory
pub mod jarvis;            // JARVIS DFT database interface
```

### Physics Models
- Perfect absorption via critical coupling
- Metamaterial layer stack optimization
- Transfer matrix method for multilayers

## Phase 4: optics_core

### Python Sources
- `gemini_physics/optics/grin_solver.py` (153 lines)
- `gemini_physics/optics/grin_benchmarks.py` (323 lines)
- `scripts/visualization/animate_warp_v9_sapphire.py`

### Rust Modules
```rust
pub mod grin;              // Graded-index ray tracing
pub mod warp_metric;       // Alcubierre warp geometry
pub mod sapphire;          // Sapphire optical properties
pub mod ray_trace;         // General ray tracing engine
```

### Key Algorithms
- RK4 geodesic integration
- Graded-index GRIN lens simulation
- Complex refractive index from Drude model

## Phase 5: quantum_core

### Python Sources
- `gemini_physics/mera.py` (507 lines)
- `gemini_physics/tensor_networks.py`
- `quantum/fractional_schrodinger.py`
- `quantum/fracton_code.py`

### Rust Modules
```rust
pub mod mera;              // MERA tensor network (quantrs2)
pub mod entropy;           // von Neumann entropy
pub mod frac_schrodinger;  // Fractional Schrodinger
pub mod fracton;           // Fracton topological codes
```

### Dependencies
- quantrs2 for tensor contraction
- ndarray-einsum for Einstein summation
- russell_ode for time evolution

## Phase 6: gr_core

### Python Sources
- `gemini_physics/gr/kerr_geodesic.py` (365 lines)
- `gravastar_tov.py`
- `quantum/advanced/pseudospectrum_slice.py`

### Rust Modules
```rust
pub mod kerr;              // Kerr metric geodesics
pub mod shadow;            // Black hole shadow calculation
pub mod gravastar;         // Gravastar structure (from cosmology)
pub mod penrose;           // Penrose diagrams (optional)
```

### Key Algorithms
- Geodesic integration in Kerr spacetime
- Photon ring and shadow boundary
- ISCO, photon sphere calculations

## Phase 7: Integration and Bindings

### gororoba_py (unified Python bindings)
```rust
// crates/gororoba_py/src/lib.rs
use pyo3::prelude::*;

#[pymodule]
fn gororoba(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_submodule(algebra_core::python::module)?;
    m.add_submodule(cosmology_core::python::module)?;
    m.add_submodule(materials_core::python::module)?;
    // ... etc
    Ok(())
}
```

### CLI Tools
```
gororoba-tov        # Gravastar/TOV solver
gororoba-mera       # MERA entropy analysis
gororoba-grin       # GRIN optics simulation
gororoba-kerr       # Kerr geodesic tracer
gororoba-zd         # Zero-divisor analysis
```

## Test Migration Strategy

### Phase 1: Rust-first
1. Migrate each Python test to Rust
2. Use deterministic seeds for reproducibility
3. Compare CSV outputs byte-for-byte

### Phase 2: Property-based
1. Add proptest for algebraic invariants
2. Fuzz inputs for numerical stability
3. Benchmark critical paths with criterion

### Phase 3: Integration
1. Python tests call Rust via PyO3
2. Verify numerical equivalence
3. Deprecate pure-Python implementations

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1 | 2 weeks | algebra_core complete, 150+ Rust tests |
| 2 | 2 weeks | cosmology_core with russell_ode |
| 3 | 1 week | materials_core absorber stack |
| 4 | 1 week | optics_core GRIN solver |
| 5 | 2 weeks | quantum_core with quantrs2 |
| 6 | 1 week | gr_core Kerr geodesics |
| 7 | 1 week | Integration, CLI, docs |

**Total**: ~10 weeks

## Sources

- [Scientific Computing in Rust](https://scientificcomputing.rs/monthly/2025-12)
- [russell_ode - ODE/DAE solvers](https://crates.io/crates/russell_ode)
- [QuantRS2 - Tensor networks](https://lib.rs/crates/quantrs2)
- [Aether - Physics simulation](https://github.com/AtmoPierce/aether)
- [Rewrite it in Rust: Computational Physics](https://arxiv.org/abs/2410.19146)
