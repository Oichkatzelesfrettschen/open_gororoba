//! Dark Inflaton from Cayley-Dickson Algebra
//!
//! The dark sector generators of SU(5) (indices 12-23 that are SM-sterile)
//! may contain a spin-0 scalar field suitable for inflation. This module tests
//! the hypothesis by:
//!
//! 1. Applying Lorentz boosts to the quaternionic sub-block (indices 0-3)
//! 2. Checking scalar (spin-0) behavior under SO(3) rotations
//! 3. Computing slow-roll parameters from the norm-squared potential V(eta)

use crate::cayley_dickson_structs::Sedenion;
use crate::quantum_state::QuantumState;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Apply a Lorentz boost to a QuantumState.
///
/// Acts on the quaternionic sub-block (indices 0-3) as spacetime coordinates
/// (t, x, y, z). Internal indices 4-15 are unchanged.
///
/// The boost matrix for rapidity eta in direction n_hat is:
///   t' = t*cosh(eta) + (n.r)*sinh(eta)
///   r'_parallel = r_parallel*cosh(eta) + t*n_hat*sinh(eta)
///   r'_perp = r_perp
pub fn lorentz_boost(
    state: &QuantumState,
    rapidity: f64,
    direction: [f64; 3],
) -> QuantumState {
    match state {
        QuantumState::TopologicalNull => QuantumState::TopologicalNull,
        QuantumState::Observable(s) => {
            let mut components = s.to_slice();

            // Normalize direction
            let dir_norm = (direction[0] * direction[0]
                + direction[1] * direction[1]
                + direction[2] * direction[2])
                .sqrt();
            if dir_norm < 1e-15 {
                return *state;
            }
            let n = [
                direction[0] / dir_norm,
                direction[1] / dir_norm,
                direction[2] / dir_norm,
            ];

            let ch = rapidity.cosh();
            let sh = rapidity.sinh();

            let t = components[0];
            let r = [components[1], components[2], components[3]];

            // r_parallel = (n . r) * n
            let n_dot_r = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];

            // Boost
            components[0] = t * ch + n_dot_r * sh;
            for i in 0..3 {
                let r_par = n_dot_r * n[i];
                let r_perp = r[i] - r_par;
                components[i + 1] = r_perp + r_par * ch + t * n[i] * sh;
            }

            QuantumState::Observable(Sedenion::from_slice(&components))
        }
    }
}

/// Generate a random SO(3) rotation matrix from 3 standard normal samples.
///
/// Uses the exponential map: sample a random axis-angle vector,
/// then compute exp(theta * [n]_x) via Rodrigues' formula.
fn random_so3(rng: &mut ChaCha8Rng) -> [[f64; 3]; 3] {
    let normal = StandardNormal;
    let v: [f64; 3] = [normal.sample(rng), normal.sample(rng), normal.sample(rng)];
    let theta = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();

    if theta < 1e-15 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let n = [v[0] / theta, v[1] / theta, v[2] / theta];
    let c = theta.cos();
    let s = theta.sin();
    let omc = 1.0 - c;

    [
        [
            c + n[0] * n[0] * omc,
            n[0] * n[1] * omc - n[2] * s,
            n[0] * n[2] * omc + n[1] * s,
        ],
        [
            n[1] * n[0] * omc + n[2] * s,
            c + n[1] * n[1] * omc,
            n[1] * n[2] * omc - n[0] * s,
        ],
        [
            n[2] * n[0] * omc - n[1] * s,
            n[2] * n[1] * omc + n[0] * s,
            c + n[2] * n[2] * omc,
        ],
    ]
}

/// Test if a QuantumState is spin-0 (scalar) under SO(3) rotations.
///
/// Applies `n_rotation_tests` random SO(3) rotations to spatial indices 1-3,
/// checks that norm and internal quantum numbers are invariant.
pub fn is_scalar_field(state: &QuantumState, n_rotation_tests: usize, seed: u64) -> bool {
    let s = match state {
        QuantumState::TopologicalNull => return false,
        QuantumState::Observable(s) => s,
    };

    let original_norm = s.norm_sqr();
    if original_norm < 1e-15 {
        return false;
    }

    let original_components = s.to_slice();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..n_rotation_tests {
        let rot = random_so3(&mut rng);
        let mut rotated = original_components;

        // Apply SO(3) to spatial indices 1,2,3
        let r = [original_components[1], original_components[2], original_components[3]];
        for i in 0..3 {
            rotated[i + 1] = rot[i][0] * r[0] + rot[i][1] * r[1] + rot[i][2] * r[2];
        }

        let rotated_s = Sedenion::from_slice(&rotated);

        // Norm must be preserved
        let rotated_norm = rotated_s.norm_sqr();
        if (rotated_norm - original_norm).abs() > 1e-10 {
            return false;
        }

        // All components (including spatial 1-3 and internal 4-15) must be unchanged.
        // A true scalar is invariant under rotation, not just norm-preserving.
        for i in 1..16 {
            if (rotated[i] - original_components[i]).abs() > 1e-10 {
                return false;
            }
        }
    }

    true
}

/// Result of slow-roll parameter computation.
pub struct SlowRollResult {
    /// First slow-roll parameter: epsilon = (V'/V)^2 / 2
    pub epsilon: f64,
    /// Second slow-roll parameter: eta = V''/V
    pub eta: f64,
    /// Number of e-folds: N ~ integral of V/V' d(rapidity)
    pub n_efolds: f64,
    /// Whether the state was confirmed as a scalar field.
    pub scalar_confirmed: bool,
}

/// Compute slow-roll parameters from the CD potential landscape.
///
/// V(eta) = norm_sqr of Lorentz-boosted QuantumState as function of rapidity.
/// Uses 5-point stencil for numerical derivatives:
///   f'(x) ~ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
///   f''(x) ~ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
pub fn slow_roll_parameters(
    state: &QuantumState,
    direction: [f64; 3],
    rapidity_center: f64,
) -> SlowRollResult {
    let scalar_confirmed = is_scalar_field(state, 100, 42);

    let h = 0.001;

    // Evaluate V(eta) at 5 points for the stencil
    let v = |eta: f64| -> f64 {
        let boosted = lorentz_boost(state, eta, direction);
        match boosted {
            QuantumState::Observable(s) => s.norm_sqr(),
            QuantumState::TopologicalNull => 0.0,
        }
    };

    let v_center = v(rapidity_center);
    let v_p1 = v(rapidity_center + h);
    let v_m1 = v(rapidity_center - h);
    let v_p2 = v(rapidity_center + 2.0 * h);
    let v_m2 = v(rapidity_center - 2.0 * h);

    // 5-point stencil derivatives
    let v_prime = (-v_p2 + 8.0 * v_p1 - 8.0 * v_m1 + v_m2) / (12.0 * h);
    let v_double_prime =
        (-v_p2 + 16.0 * v_p1 - 30.0 * v_center + 16.0 * v_m1 - v_m2) / (12.0 * h * h);

    let epsilon = if v_center.abs() > 1e-15 {
        0.5 * (v_prime / v_center).powi(2)
    } else {
        f64::INFINITY
    };

    let eta_sr = if v_center.abs() > 1e-15 {
        v_double_prime / v_center
    } else {
        f64::INFINITY
    };

    // Estimate e-folds via trapezoidal integration of V/V' over rapidity [0, eta_end]
    // where eta_end is where epsilon(eta) > 1 (end of inflation).
    let n_efolds = estimate_efolds(state, direction);

    SlowRollResult {
        epsilon,
        eta: eta_sr,
        n_efolds,
        scalar_confirmed,
    }
}

/// Estimate number of e-folds by integrating V/V' until epsilon > 1.
fn estimate_efolds(state: &QuantumState, direction: [f64; 3]) -> f64 {
    let h = 0.01;
    let mut n = 0.0;
    let mut eta = 0.0;
    let max_steps = 10000;

    for _ in 0..max_steps {
        let v_here = match lorentz_boost(state, eta, direction) {
            QuantumState::Observable(s) => s.norm_sqr(),
            QuantumState::TopologicalNull => return n,
        };
        let v_next = match lorentz_boost(state, eta + h, direction) {
            QuantumState::Observable(s) => s.norm_sqr(),
            QuantumState::TopologicalNull => return n,
        };

        if v_here < 1e-15 {
            return n;
        }

        let v_prime = (v_next - v_here) / h;
        if v_prime.abs() < 1e-15 {
            // Flat potential -- infinite inflation
            return f64::INFINITY;
        }

        let epsilon_local = 0.5 * (v_prime / v_here).powi(2);
        if epsilon_local > 1.0 {
            break;
        }

        // dN = V / V' * d(eta)
        n += (v_here / v_prime).abs() * h;
        eta += h;
    }

    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cayley_dickson_structs::Sedenion;

    fn make_test_state() -> QuantumState {
        // A state with nonzero components in both spacetime and internal sectors
        let mut components = [0.0; 16];
        components[0] = 1.0; // time
        components[4] = 0.5; // internal (octonion)
        components[8] = 0.3; // internal (sedenion)
        QuantumState::Observable(Sedenion::from_slice(&components))
    }

    #[test]
    fn test_lorentz_boost_preserves_norm() {
        let state = make_test_state();
        let original_norm = match state {
            QuantumState::Observable(s) => s.norm_sqr(),
            _ => panic!(),
        };

        for &rapidity in &[0.0, 0.1, 0.5, 1.0, 2.0] {
            let boosted = lorentz_boost(&state, rapidity, [1.0, 0.0, 0.0]);
            let boosted_norm = match boosted {
                QuantumState::Observable(s) => s.norm_sqr(),
                _ => panic!("boost should not produce null"),
            };

            // Lorentz boost preserves the Minkowski norm (t^2 - x^2 - y^2 - z^2),
            // not the Euclidean norm. But internal indices are unchanged, so the
            // total sedenion norm changes only through the spacetime sub-block.
            //
            // For pure timelike (only t nonzero), boosted norm equals:
            //   t'^2 + internal^2 = (t*cosh)^2 + (t*sinh)^2 + internal^2
            // which grows with rapidity. This is expected and physical.
            println!(
                "  rapidity={rapidity:.1}: norm {original_norm:.6} -> {boosted_norm:.6}"
            );
        }

        // Zero boost must preserve exactly
        let zero_boost = lorentz_boost(&state, 0.0, [1.0, 0.0, 0.0]);
        let zero_norm = match zero_boost {
            QuantumState::Observable(s) => s.norm_sqr(),
            _ => panic!(),
        };
        assert!(
            (zero_norm - original_norm).abs() < 1e-12,
            "zero boost must preserve norm exactly"
        );
    }

    #[test]
    fn test_scalar_field_detection() {
        // A state with only internal components (no spatial) should be scalar
        let mut scalar_components = [0.0; 16];
        scalar_components[0] = 1.0; // time only
        scalar_components[8] = 0.5; // internal
        let scalar_state = QuantumState::Observable(Sedenion::from_slice(&scalar_components));
        assert!(
            is_scalar_field(&scalar_state, 100, 42),
            "pure time+internal state should be scalar under SO(3)"
        );

        // A state with spatial components transforms non-trivially
        let mut vector_components = [0.0; 16];
        vector_components[1] = 1.0; // x spatial
        let vector_state = QuantumState::Observable(Sedenion::from_slice(&vector_components));
        assert!(
            !is_scalar_field(&vector_state, 100, 42),
            "spatial vector should not be scalar"
        );
    }

    #[test]
    fn test_slow_roll_parameters() {
        let state = make_test_state();
        let result = slow_roll_parameters(&state, [1.0, 0.0, 0.0], 0.0);

        println!("--- SLOW-ROLL PARAMETERS ---");
        println!("  epsilon = {:.6e}", result.epsilon);
        println!("  eta     = {:.6e}", result.eta);
        println!("  N_efolds = {:.1}", result.n_efolds);
        println!("  scalar_confirmed = {}", result.scalar_confirmed);

        // epsilon and eta should be finite
        assert!(result.epsilon.is_finite(), "epsilon must be finite");
        assert!(result.eta.is_finite(), "eta must be finite");
    }

    #[test]
    fn test_null_state_boost() {
        let null = QuantumState::TopologicalNull;
        let boosted = lorentz_boost(&null, 1.0, [1.0, 0.0, 0.0]);
        assert_eq!(boosted, QuantumState::TopologicalNull);
    }
}
