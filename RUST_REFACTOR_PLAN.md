# Rust-First Architecture Refactoring Plan

## Current State Review

### What We've Built (Phase 7 Progress)

#### Sprint 1: Gravastar Stability (COMPLETE)
- F1: Polytropic shell EoS extension (`src/gravastar_tov.py`)
- F2: Anisotropic pressure with Bowers-Liang form
- R1: Closed 5 claims (C-008, C-022, C-023, C-077, C-078)
- **Output**: `gravastar_polytropic_sweep.csv`, `gravastar_anisotropic_stability.csv`

#### Sprint 2: CD Algebra Mass Spectra (COMPLETE)
- G3.1: Furey Cl(8) generation assignment (`clifford_algebra.py`)
- G3.2: Tang sedenion-SU(5) mass ratios (`tang_mass_ratios.py`)
- G3.3: Rust batch associator computation (`algebra.rs`)
- G3.4: Pathion 32D ZD spectrum (`zd_spectrum_analysis.py`)
- **Output**: `tang_lepton_mass_predictions.csv`, `zd_scaling_law.csv`

### Current Rust Crate: `gororoba_kernels`

#### Dependencies (Cargo.toml)
```toml
[dependencies]
pyo3 = { version = "0.25", features = ["extension-module"] }
numpy = "0.25"
rand = "0.8"
```

#### Implemented Functions
| Function | Purpose | Lines |
|----------|---------|-------|
| `cd_multiply` | Cayley-Dickson multiplication | ~30 |
| `cd_conjugate` | Conjugation | ~10 |
| `cd_norm_sq` | Squared norm | ~5 |
| `cd_associator` | Associator A(a,b,c) | ~15 |
| `cd_associator_norm` | ||A(a,b,c)|| | ~5 |
| `batch_associator_norms` | Batch ||A|| | ~25 |
| `batch_associator_norms_sq` | Batch ||A||^2 | ~20 |
| `left_mult_operator` | L_a matrix | ~20 |
| `find_zero_divisors` | 2-blade ZD search | ~35 |
| `find_zero_divisors_3blade` | 3-blade ZD search | ~40 |
| `find_zero_divisors_general_form` | Random ZD search | ~45 |
| `count_pathion_zero_divisors` | ZD counts | ~10 |
| `zd_spectrum_analysis` | Norm histogram | ~35 |
| `measure_associator_density` | Non-assoc % | ~30 |

**Total**: ~325 lines of Rust algebra code + ~400 lines of tests

---

## Recommended Additional Cargo Dependencies

### For Correctness and Safety

```toml
# Numerical precision
num-traits = "0.2"        # Generic numeric traits
num-complex = "0.4"       # Complex number support (for Cl(n))
nalgebra = "0.33"         # Linear algebra (matrices for Clifford)

# Parallelism (batch operations)
rayon = "1.10"            # Data parallelism for batch ops

# Error handling
thiserror = "2.0"         # Derive(Error) for custom errors
anyhow = "1.0"            # Flexible error handling

# Testing/benchmarking
criterion = "0.6"         # Microbenchmarks
proptest = "1.6"          # Property-based testing
approx = "0.5"            # Floating point comparison
```

### For Security (cargo audit compatible)

```toml
# Cryptographically secure RNG (if needed for Monte Carlo)
rand_chacha = "0.3"       # ChaCha20 PRNG backend

# Constant-time operations (if implementing crypto-adjacent code)
subtle = "2.6"            # Constant-time comparison
```

### For CLI/Binary (if building standalone tools)

```toml
# CLI framework
clap = { version = "4.5", features = ["derive"] }
indicatif = "0.17"        # Progress bars

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
csv = "1.3"               # CSV output
```

---

## Refactoring Strategy: Python -> Rust

### Phase A: Extract Core Algebra to Pure Rust Library

**Goal**: Create `gororoba-core` crate with no Python dependency.

```
src/gororoba-core/
  Cargo.toml
  src/
    lib.rs
    cayley_dickson.rs     # CD multiplication, conjugation, norm
    associator.rs         # Associator computation, batch ops
    zero_divisor.rs       # ZD search algorithms
    clifford.rs           # Cl(n) gamma matrices, ideals
    spectrum.rs           # Mass ratio analysis
```

**Cargo.toml** for `gororoba-core`:
```toml
[package]
name = "gororoba-core"
version = "0.2.0"
edition = "2024"

[dependencies]
num-traits = "0.2"
num-complex = "0.4"
nalgebra = "0.33"
rayon = "1.10"
rand = "0.9"
rand_chacha = "0.3"
thiserror = "2.0"

[dev-dependencies]
criterion = "0.6"
proptest = "1.6"
approx = "0.5"
```

### Phase B: Python Bindings as Thin Wrapper

**Goal**: `gororoba-py` crate depends on `gororoba-core`, exposes to Python.

```
src/gororoba-py/
  Cargo.toml
  src/
    lib.rs                # PyO3 module definition
    array_convert.rs      # numpy <-> nalgebra conversion
```

### Phase C: Migrate Physics Modules

Priority order (by complexity and test coverage):

1. **Tang mass ratios** (already uses Rust kernels heavily)
   - Move `tang_particle_assignment()` to Rust
   - Move `compute_lepton_associator_norms()` to Rust
   - Python becomes thin CLI/plotting layer

2. **ZD spectrum analysis** (already uses Rust kernels)
   - Move `classify_zd_by_indices()` to Rust
   - Move histogram computation to Rust

3. **Clifford algebra** (pure math, no physics deps)
   - Implement gamma matrix construction in Rust
   - Implement Cl(6) ideal decomposition in Rust
   - Use `nalgebra` for matrix operations

4. **Gravastar TOV** (numerical ODE solver)
   - Implement RK4/RK45 solver in Rust
   - Move polytropic EoS to Rust
   - Move anisotropic TOV to Rust

### Phase D: CLI Binary

**Goal**: `gororoba` CLI for running analyses without Python.

```
src/gororoba-cli/
  Cargo.toml
  src/
    main.rs
    commands/
      associator.rs       # Associator density measurement
      zero_divisor.rs     # ZD search
      gravastar.rs        # TOV solver
      spectrum.rs         # Mass ratio analysis
```

---

## Migration Checklist

### Immediate (This Session)

- [ ] Add recommended dependencies to Cargo.toml
- [ ] Add `rayon` parallelism to `batch_associator_norms`
- [ ] Implement `criterion` benchmarks for core ops
- [ ] Add `proptest` property tests for CD algebra laws

### Short Term (Next Sprint)

- [ ] Create `gororoba-core` workspace member
- [ ] Move algebra code to pure Rust (no PyO3)
- [ ] Implement Clifford algebra in Rust with `nalgebra`
- [ ] Add comprehensive property tests

### Medium Term

- [ ] Implement gravastar TOV solver in Rust
- [ ] Create CLI binary with `clap`
- [ ] Add CSV output with `csv` crate
- [ ] Implement progress bars with `indicatif`

### Long Term

- [ ] Full Python elimination for core physics
- [ ] WASM target for browser visualization
- [ ] GPU acceleration with `wgpu` or CUDA bindings

---

## Test Strategy for Rust Migration

### Property-Based Tests (proptest)

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

### Benchmark Suite (criterion)

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

---

## Expected Performance Gains

| Operation | Python (NumPy) | Rust (current) | Rust (optimized) |
|-----------|---------------|----------------|------------------|
| cd_multiply (dim=16) | ~50 us | ~2 us | ~0.5 us (SIMD) |
| batch_assoc (10k, dim=16) | ~800 ms | ~30 ms | ~5 ms (rayon) |
| ZD search (dim=32) | N/A | ~100s | ~10s (rayon) |

---

## File Structure After Refactor

```
open_gororoba/
  Cargo.toml              # Workspace root
  src/
    gororoba-core/        # Pure Rust algebra/physics
    gororoba-py/          # Python bindings
    gororoba-cli/         # CLI binary
  python/                 # Thin Python layer (plotting, notebooks)
  tests/                  # Integration tests
  benches/                # Criterion benchmarks
  data/csv/               # Output artifacts
  docs/                   # Documentation
```
