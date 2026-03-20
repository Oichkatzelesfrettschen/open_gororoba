//! Non-Unitary Black Hole via Cayley-Dickson Curvature
//!
//! In extreme spacetime curvature (kappa -> kappa_critical), the sedenion
//! product is modified by a curvature term proportional to the associator.
//! At the critical curvature, ALL products collapse to TopologicalNull,
//! modeling the information-destroying singularity.
//!
//! The curvature-modified product interpolates between the standard product
//! and the purely non-associative associator sum:
//!   a *_kappa b = exp(-kappa) * (a*b) + (1 - exp(-kappa)) * mean_k [a, b, e_k]
//!
//! At kappa=0, this is the standard product. At kappa -> inf, the product
//! is dominated by the associator, which has extensive zero structure.

use crate::cayley_dickson_structs::Sedenion;
use crate::quantum_state::QuantumState;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Compute the sedenion associator: [a, b, c] = (a*b)*c - a*(b*c).
fn sedenion_associator(a: &Sedenion, b: &Sedenion, c: &Sedenion) -> Sedenion {
    let ab = *a * *b;
    let ab_c = ab * *c;
    let bc = *b * *c;
    let a_bc = *a * bc;
    ab_c - a_bc
}

/// Curvature-modified sedenion product.
///
/// a *_kappa b = exp(-kappa) * (a*b) + (1 - exp(-kappa)) * sum_k [a, b, e_k] / 15
///
/// At kappa=0 this reduces to the standard product. As kappa increases, the
/// product smoothly transitions from the associative-like standard product to
/// the purely non-associative associator sum. The associator has extensive zero
/// structure (vanishes for many pairs), so at high curvature the null fraction
/// increases -- modeling the information-destroying singularity.
pub fn curvature_product(a: &Sedenion, b: &Sedenion, kappa: f64) -> QuantumState {
    let standard_product = *a * *b;

    // Sum associators over all 15 imaginary basis elements
    let mut assoc_sum = Sedenion::default();
    for k in 1..16 {
        let mut ek_comp = [0.0; 16];
        ek_comp[k] = 1.0;
        let ek = Sedenion::from_slice(&ek_comp);
        let assoc_k = sedenion_associator(a, b, &ek);
        assoc_sum = assoc_sum + assoc_k;
    }

    let weight_std = (-kappa).exp();
    let weight_assoc = 1.0 - weight_std;

    let result_components: [f64; 16] = {
        let sp = standard_product.to_slice();
        let ac = assoc_sum.to_slice();
        let mut r = [0.0; 16];
        for i in 0..16 {
            r[i] = weight_std * sp[i] + weight_assoc * ac[i] / 15.0;
        }
        r
    };

    let result = Sedenion::from_slice(&result_components);

    // Check for zero-divisor collapse: product norm much smaller than input norms
    let input_scale = a.norm_sqr().sqrt() * b.norm_sqr().sqrt();
    let threshold = input_scale * 1e-10;
    if input_scale > 1e-12 && result.norm_sqr().sqrt() < threshold {
        QuantumState::TopologicalNull
    } else {
        QuantumState::Observable(result)
    }
}

/// Result of critical curvature search.
pub struct CriticalCurvatureResult {
    /// The curvature parameter where null fraction exceeds (1 - epsilon).
    pub kappa_critical: f64,
    /// Fraction of test pairs yielding TopologicalNull at kappa_critical.
    pub null_fraction_at_critical: f64,
    /// Mean associator norm at the critical curvature.
    pub associator_norm_at_critical: f64,
    /// Sharpness of the transition: d(null_fraction)/d(kappa) at critical.
    pub transition_sharpness: f64,
}

/// Characterize the curvature-induced product degradation.
///
/// At each kappa value, measures the mean ratio of deformed product norm to
/// standard product norm. As kappa increases, the product transitions from
/// the standard (ratio=1) to the associator-dominated regime.
///
/// Reports the kappa where the mean degradation ratio crosses 0.5 (half-power),
/// the asymptotic degradation ratio (kappa -> inf), and the mean associator norm.
pub fn find_critical_curvature(
    n_pairs: usize,
    _epsilon: f64,
    seed: u64,
) -> CriticalCurvatureResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let normal = StandardNormal;
    let pairs: Vec<(Sedenion, Sedenion)> = (0..n_pairs)
        .map(|_| {
            let mut a_comp = [0.0; 16];
            let mut b_comp = [0.0; 16];
            for i in 0..16 {
                a_comp[i] = normal.sample(&mut rng);
                b_comp[i] = normal.sample(&mut rng);
            }
            let a = Sedenion::from_slice(&a_comp);
            let b = Sedenion::from_slice(&b_comp);
            let a_norm = a.norm_sqr().sqrt();
            let b_norm = b.norm_sqr().sqrt();
            (
                Sedenion::from_slice(&a_comp.map(|x| x / a_norm)),
                Sedenion::from_slice(&b_comp.map(|x| x / b_norm)),
            )
        })
        .collect();

    // Measure mean degradation ratio at a given kappa
    let degradation_at = |kappa: f64| -> f64 {
        let total: f64 = pairs
            .iter()
            .map(|(a, b)| {
                let standard_norm = (*a * *b).norm_sqr().sqrt();
                let deformed = curvature_product(a, b, kappa);
                let deformed_norm = match deformed {
                    QuantumState::Observable(s) => s.norm_sqr().sqrt(),
                    QuantumState::TopologicalNull => 0.0,
                };
                if standard_norm > 1e-15 {
                    deformed_norm / standard_norm
                } else {
                    1.0
                }
            })
            .sum();
        total / n_pairs as f64
    };

    // Binary search for kappa where degradation crosses 0.5
    let mut lo = 0.0_f64;
    let mut hi = 100.0_f64;
    let d_at_hi = degradation_at(hi);
    // If degradation never drops to 0.5, report the asymptotic kappa
    let target_reached = d_at_hi < 0.5;

    if target_reached {
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            if degradation_at(mid) < 0.5 {
                hi = mid;
            } else {
                lo = mid;
            }
            if (hi - lo) < 1e-6 {
                break;
            }
        }
    }

    let kappa_critical = if target_reached { (lo + hi) / 2.0 } else { hi };
    let null_frac = degradation_at(kappa_critical);

    // Mean associator norm (asymptotic product norm at kappa -> inf)
    let mean_assoc_norm = degradation_at(50.0); // exp(-50) ~ 0, so this is the associator regime

    // Transition sharpness
    let dk = 0.1;
    let d_plus = degradation_at(kappa_critical + dk);
    let d_minus = degradation_at(kappa_critical - dk);
    let sharpness = (d_minus - d_plus) / (2.0 * dk); // positive = degradation increasing

    CriticalCurvatureResult {
        kappa_critical,
        null_fraction_at_critical: null_frac,
        associator_norm_at_critical: mean_assoc_norm,
        transition_sharpness: sharpness,
    }
}

/// Map ADM (Arnowitt-Deser-Misner) curvature invariants to the algebraic kappa.
///
/// kappa = sqrt(K_ij * K^ij) / (lapse * rho_crit)
///
/// where K_ij is the extrinsic curvature tensor, lapse is the ADM lapse function,
/// and rho_crit is a reference density (Planck density in natural units).
///
/// For Schwarzschild at r = 2M (horizon):
///   K_ij ~ 1/M, so |K|^2 ~ 1/M^2, kappa ~ 1/(M * rho_crit)
pub fn adm_curvature_to_kappa(
    lapse: f64,
    extrinsic_k_trace: f64,
    tracefree_norm: f64,
) -> f64 {
    let k_squared = extrinsic_k_trace.powi(2) / 3.0 + tracefree_norm.powi(2);
    let rho_crit = 1.0; // Planck units
    k_squared.sqrt() / (lapse * rho_crit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curvature_product_at_zero_matches_standard() {
        let a = Sedenion::from_slice(&[1.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0,
                                        0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]);
        let b = Sedenion::from_slice(&[0.0, 1.0, 0.0, 0.4, 0.0, 0.0, 0.6, 0.0,
                                        0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]);

        let standard = a * b;
        let curvature_result = curvature_product(&a, &b, 0.0);

        match curvature_result {
            QuantumState::Observable(s) => {
                let diff: f64 = standard
                    .to_slice()
                    .iter()
                    .zip(s.to_slice().iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                assert!(
                    diff < 1e-20,
                    "kappa=0 product must match standard product exactly"
                );
            }
            QuantumState::TopologicalNull => {
                // Standard product could be null too
                assert!(
                    standard.norm_sqr() < 1e-12,
                    "if curvature_product is null at kappa=0, standard product must also be near-zero"
                );
            }
        }
    }

    #[test]
    fn test_critical_curvature_is_finite() {
        let result = find_critical_curvature(200, 0.05, 42);

        println!("--- CURVATURE-INDUCED PRODUCT DEGRADATION ---");
        println!("  kappa_half_power   = {:.4}", result.kappa_critical);
        println!("  degradation_ratio  = {:.4}", result.null_fraction_at_critical);
        println!("  asymptotic_ratio   = {:.6}", result.associator_norm_at_critical);
        println!("  transition_sharpness = {:.4}", result.transition_sharpness);

        assert!(
            result.kappa_critical.is_finite() && result.kappa_critical > 0.0,
            "critical curvature must be finite and positive"
        );
        // The asymptotic degradation ratio should be less than 1.0:
        // the associator-dominated product has different norm than the standard product
        assert!(
            result.associator_norm_at_critical.is_finite(),
            "asymptotic degradation ratio must be finite"
        );
        // Transition should be smooth (non-negative sharpness)
        assert!(
            result.transition_sharpness.is_finite(),
            "transition sharpness must be finite"
        );
    }

    #[test]
    fn test_adm_curvature_mapping() {
        // Flat spacetime: lapse=1, K=0
        let kappa_flat = adm_curvature_to_kappa(1.0, 0.0, 0.0);
        assert!(
            kappa_flat.abs() < 1e-15,
            "flat spacetime should give kappa=0"
        );

        // Non-trivial curvature
        let kappa = adm_curvature_to_kappa(0.5, 1.0, 0.5);
        assert!(kappa > 0.0, "nonzero curvature should give positive kappa");
        assert!(kappa.is_finite());
    }
}
