# Core Abstractions: Unified Design Patterns Across Physics Domains

**Status:** Phase 7 reference document
**Date:** 2026-02-05
**Provenance:** Synthesized from optics_core, quantum_core, stats_core implementations
**Related:** GRAND_SYNTHESIS.md (physical claims), AGENTS.md (build/test commands)

---

## Overview

Four recurring design motifs emerge across the physics stack. These abstractions
provide a conceptual backbone for understanding how different modules relate:

| Abstraction | Physical Role | Key Property | Example Modules |
|-------------|---------------|--------------|-----------------|
| **Coupler** | Energy/information transfer | Coupling strength | TCMT external port, Casimir plate |
| **Resonator** | Mode confinement | Characteristic frequency | KerrCavity, SLM-focal system |
| **Amplifier** | Signal gain | Nonlinear feedback | Kerr bistability, Casimir transistor |
| **Verifier** | Statistical validation | Null hypothesis testing | ED/MMD, bootstrap CI |

---

## 1. Coupler: Energy and Information Transfer

A **Coupler** mediates energy or information exchange between subsystems. The
essential parameters are: coupling strength, directionality, and bandwidth.

### 1.1 TCMT External Coupling (optics_core/src/tcmt.rs)

The external coupling port in TCMT connects the cavity to an input/output waveguide:

```text
Input field:  s_in -> [sqrt(2/tau_e)] -> cavity mode a
Output field: s_out = sqrt(2/tau_e) * a - s_in
```

**Parameters:**
- `tau_e`: External coupling lifetime (s) -- larger = weaker coupling
- `Q_e = omega_0 * tau_e / 2`: External quality factor
- `gamma_e = 1/tau_e`: External decay rate (rad/s)

**Rust signature:**
```rust
impl KerrCavity {
    pub fn tau_external(&self) -> f64 {
        self.q_external * 2.0 / self.omega_0
    }
}
```

**Physical interpretation:** The coupler sets the rate at which energy leaks
into/from the resonator. Critical coupling (Q_e = Q_0) maximizes power transfer.

### 1.2 Casimir Plate Coupler (quantum_core/src/casimir.rs)

In the sphere-plate-sphere transistor, the plate acts as a mechanical coupler:

```text
Source sphere -> [F_source] -> Plate displacement -> [F_drain] -> Drain sphere
```

**Parameters:**
- `plate_spring_constant`: Mechanical compliance (N/m)
- Cross-coupling gain: `G = (k_source * k_drain) / k_plate^2`

**Rust signature:**
```rust
pub fn cross_coupling_additive(
    r_source: f64, r_drain: f64,
    gap_source: f64, gap_drain: f64,
    plate_spring: f64,
) -> (f64, f64, f64)  // (force_source, force_drain, coupling)
```

**Physical interpretation:** The plate mediates vacuum fluctuation information
between the spheres. A more compliant plate (smaller k) increases coupling.

### 1.3 WGS Propagator Coupler (optics_core/src/phase_retrieval.rs)

Fourier propagation couples the SLM plane to the focal plane:

```text
SLM plane: E_slm(x,y) --[FFT]--> Focal plane: E_focal(kx,ky)
```

**Parameters:**
- `grid_size`: Resolution of the coupling transform
- Pixel pitch: Determines the k-space range

**Rust signature:**
```rust
pub fn wgs_discrete(
    targets: &[TargetSpot],
    grid_size: usize,
    config: &WgsConfig,
) -> WgsResult
```

**Physical interpretation:** The FFT is an optical coupler -- light at each SLM
pixel contributes to all focal spots via constructive/destructive interference.

---

## 2. Resonator: Mode Confinement

A **Resonator** confines energy into discrete modes with characteristic frequencies.
The essential parameters are: resonance frequency, quality factor, and mode volume.

### 2.1 Kerr Cavity Resonator (optics_core/src/tcmt.rs)

```text
Cavity mode:  a(t) * exp(-i*omega_0*t)
Mode energy:  |a|^2 (Joules, normalized to photon energy for quantum)
```

**Parameters:**
- `omega_0`: Resonance angular frequency (rad/s)
- `Q_0`: Intrinsic quality factor (material/radiation loss)
- `Q_e`: External quality factor (coupling loss)
- `V_eff`: Effective mode volume (m^3)
- `Q_tot = 1/(1/Q_0 + 1/Q_e)`: Total quality factor

**Rust struct:**
```rust
pub struct KerrCavity {
    pub omega_0: f64,
    pub q_intrinsic: f64,
    pub q_external: f64,
    pub n_linear: f64,
    pub n2: f64,
    pub v_eff: f64,
}
```

**Linewidth relation:** `Delta_omega = omega_0 / Q_tot`

### 2.2 Casimir Gap Resonator (quantum_core/src/casimir.rs)

The sphere-plate gap acts as a vacuum mode resonator:

```text
Allowed modes:  k_n ~ n*pi/d  (standing waves between surfaces)
Mode density:   Modified by geometry, causing Casimir force
```

**Parameters:**
- `gap`: Surface-to-surface separation (m)
- `radius`: Sphere radius determines mode volume
- PFA validity: `R/d > 132` for 1% accuracy

**Rust struct:**
```rust
pub struct SpherePlateSphere {
    pub source: Sphere,
    pub plate: Plate,
    pub drain: Sphere,
}
```

### 2.3 Optical Tweezer Array (optics_core/src/phase_retrieval.rs)

Discrete target spots form a resonator array:

```text
Each spot:  Focused light -> Optical trap potential
Array:      Multiple independent resonators at target positions
```

**Parameters:**
- `TargetSpot.x, TargetSpot.y`: Spot position in focal plane
- `TargetSpot.amplitude`: Target intensity (trap depth)

**Rust struct:**
```rust
pub struct TargetSpot {
    pub x: f64,
    pub y: f64,
    pub amplitude: f64,
}
```

---

## 3. Amplifier: Signal Gain via Nonlinear Feedback

An **Amplifier** provides gain through nonlinear dynamics. The essential parameters
are: threshold, saturation, and stability.

### 3.1 Kerr Optical Bistability (optics_core/src/tcmt.rs)

Intensity-dependent refractive index creates positive feedback:

```text
High input -> Cavity heats -> Refractive index shifts -> Resonance tunes
           -> More power couples -> More heating -> BISTABLE
```

**Parameters:**
- `n2`: Kerr coefficient (m^2/W) -- positive = self-focusing
- `gamma_kerr = n2 * omega_0 * c / (n0^2 * V_eff)`: Nonlinear frequency shift rate
- Bistability threshold: `|Omega| > sqrt(3)` (normalized detuning)

**Normalized steady-state equation (Liu et al. 2013, Eq. 5):**
```text
u^2 = y * [(y - Omega)^2 + 1]

where:
  u^2 = normalized input power
  y   = normalized cavity energy
  Omega = 2*delta / gamma_tot (normalized detuning)
```

**Rust signature:**
```rust
pub fn solve_normalized_cubic(u_squared: f64, omega: f64) -> NormalizedSteadyState
pub fn bistability_bounds(omega: f64) -> (bool, f64, f64)  // (is_bistable, y_min, y_max)
```

**Stability criterion:** For cubic f(y), stable solutions have f'(y) > 0.

### 3.2 Casimir Transistor Gain (quantum_core/src/casimir.rs)

Plate displacement amplifies force changes:

```text
Source moves dx -> Plate displaced dy = F_source/k_plate
              -> Drain gap changes -> Drain force changes dF_drain
              -> Gain G = dF_drain / dF_source
```

**Parameters:**
- `plate_spring_constant`: Mechanical gain control
- Spring constant ratio sets amplification

**Rust signature:**
```rust
pub fn transistor_gain_additive(
    r_drain: f64,
    gap_drain: f64,
    plate_spring: f64,
) -> f64
```

**Gain expression:** `G = k_Casimir / k_plate` where `k_Casimir = 3*CASIMIR_COEFF*R/d^4`

### 3.3 Grover Amplitude Amplification (Planned: quantum_core)

Quantum amplitude amplification provides quadratic speedup:

```text
Initial:      |psi> = sum_x alpha_x |x>
After k iters: Amplitude of marked state grows as sin((2k+1)*theta)
```

**Parameters:**
- `oracle`: Marks target state with phase flip
- `diffusion`: Inversion about mean
- `iterations`: `k ~ pi/(4*theta)` for theta = arcsin(sqrt(M/N))

**Planned Rust signature:**
```rust
pub fn grover_iterate<O, D>(state: &mut Vec<Complex64>, oracle: O, diffusion: D, k: usize)
where O: Fn(&mut Vec<Complex64>), D: Fn(&mut Vec<Complex64>)
```

---

## 4. Verifier: Statistical Validation

A **Verifier** tests claims against null hypotheses. The essential parameters are:
test statistic, null distribution, and decision threshold.

### 4.1 Frechet Distance Test (stats_core/src/lib.rs)

Tests spectrum shape similarity accounting for curve ordering:

```text
Observed spectrum S_obs vs Reference S_ref
Null hypothesis: S_obs comes from random monotonic process
```

**Parameters:**
- `n_permutations`: Null distribution sample size
- `significance`: Decision threshold (typically 0.05)

**Rust signature:**
```rust
pub fn frechet_null_test(
    observed: &[f64],
    reference: &[f64],
    n_permutations: usize,
    seed: u64,
) -> FrechetNullTestResult
```

**Decision rule:** If p > 0.05, claim is refuted.

### 4.2 Bootstrap Confidence Intervals (stats_core/src/lib.rs)

Estimates parameter uncertainty via resampling:

```text
Point estimate theta_hat from data
Bootstrap: Resample -> Recompute -> Distribution of theta_hat*
CI: [theta_hat*_alpha/2, theta_hat*_1-alpha/2]
```

**Parameters:**
- `n_bootstrap`: Number of resamples (typically 1000+)
- `confidence`: Coverage level (typically 0.95)

**Rust signature:**
```rust
pub fn bootstrap_ci<F>(
    data: &[f64],
    statistic_fn: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> BootstrapCIResult
where F: Fn(&[f64]) -> f64
```

**Uncertainty flag:** CI width > 50% of point estimate triggers "uncertain" status.

### 4.3 Energy Distance + MMD Two-Sample Test (stats_core/src/lib.rs)

Tests whether two samples come from the same distribution:

```text
Energy Distance: ED = (2/nx*ny)*sum(DXY) - (1/nx^2)*sum(DXX) - (1/ny^2)*sum(DYY)
MMD (RBF kernel): MMD^2 = (1/nx^2)*sum(KXX) + (1/ny^2)*sum(KYY) - (2/nx*ny)*sum(KXY)
```

**Parameters:**
- `sigma`: Kernel bandwidth (median heuristic default)
- `n_permutations`: Null distribution size
- Fisher's method combines p-values

**Rust signature:**
```rust
pub fn two_sample_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: u64,
) -> TwoSampleTestResult
```

**Efficiency:** Precomputes distance/kernel matrices; permutations reuse O(n) sums.

### 4.4 Haar-Measure Unitary Test (stats_core/src/lib.rs)

Tests whether a predicted matrix is closer to a reference than random:

```text
Null hypothesis: Predicted matrix is no closer to PMNS than random U(n)
Test statistic: Frobenius distance ||predicted - reference||_F
```

**Parameters:**
- `n_permutations`: Number of random unitaries to generate
- Reference matrix (e.g., PMNS for neutrino mixing)

**Rust signature:**
```rust
pub fn test_pmns_prediction(
    predicted: &DMatrix<Complex64>,
    n_permutations: usize,
    seed: u64,
) -> PmnsComparisonResult
```

---

## Cross-Cutting Patterns

### Dimensional Analysis

All physics modules use SI units with explicit constants:

```rust
// quantum_core/src/casimir.rs
pub const HBAR: f64 = 1.054571817e-34;  // J*s
pub const C: f64 = 299792458.0;         // m/s

// optics_core/src/tcmt.rs
const C_LIGHT: f64 = 299792458.0;       // m/s
```

### Normalized Forms

Dimensionless equations simplify analysis:

| Module | Normalization | Physical Meaning |
|--------|---------------|------------------|
| TCMT | `Omega = 2*delta/gamma_tot` | Detuning in linewidths |
| TCMT | `u^2 = P_in * 2*gamma_e / (gamma_k * P_crit)` | Power ratio |
| Casimir | `R/d` | Shape parameter for PFA validity |

### Seeded Randomness

Reproducibility via explicit seeds:

```rust
use rand_chacha::ChaCha8Rng;
let mut rng = ChaCha8Rng::seed_from_u64(seed);
```

---

## Module Cross-References

| Abstraction | optics_core | quantum_core | stats_core |
|-------------|-------------|--------------|------------|
| Coupler | `KerrCavity::tau_external()` | `cross_coupling_additive()` | - |
| Resonator | `KerrCavity` | `SpherePlateSphere` | - |
| Amplifier | `solve_normalized_cubic()` | `transistor_gain_additive()` | - |
| Verifier | - | - | `two_sample_test()` |

---

## Future Extensions

1. **Grover Amplifier** (Task #78): Implement `grover_iterate()` in quantum_core
2. **Thermal Coupler** (Task #76): Add temperature-dependent TCMT dynamics
3. **Lifshitz Resonator** (Task #75): Frequency-dependent Casimir via reflection coefficients

## Implemented: Claims Gates (stats_core/src/claims_gates.rs)

The claims_gates module provides CI-integrated verification infrastructure:

```rust
use stats_core::{GateRegistry, pvalue_gate, run_frechet_gate};

let mut registry = GateRegistry::new();
registry.add(pvalue_gate("C-070", 0.01, 1000, 0.05, "Frechet match"));
assert!(registry.all_pass());
```

**Gate Types:**
- `pvalue_gate()`: Pass if p < threshold (significant difference)
- `inverse_pvalue_gate()`: Pass if p >= threshold (null not rejected)
- `bootstrap_ci_gate()`: Pass if estimate in range, uncertain if CI too wide
- `frechet_gate()`: Wrap FrechetNullTestResult
- `two_sample_gate()`: Wrap ED+MMD combined test
- `metric_gate()`: Custom metric with configurable direction

**Output:** `GateRegistry::to_json()` produces machine-readable verdicts for CI parsing.

---

## Literature

- Liu et al., Opt. Express 21(20), 23687-23699 (2013) -- TCMT + Kerr bistability
- Suh et al., IEEE J. Quantum Electron. 40, 1511 (2004) -- TCMT foundations
- Xu et al., Nature Communications 13, 6148 (2022) -- Casimir transistor
- Bordag et al., Physics Reports 353, 1 (2001) -- Casimir review
- Chaibub Neto & Prisco (2024), arXiv:2406.06488 -- Efficient permutation tests
- Efron & Tibshirani (1993) -- Bootstrap methods
- Gerchberg & Saxton, Optik 35, 237 (1972) -- Phase retrieval
- Grover, Proc. STOC (1996) -- Quantum amplitude amplification
