//! Non-Unitary Black Hole via Cayley-Dickson Curvature
//!
//! In extreme spacetime curvature (kappa -> kappa_critical), the sedenion
//! product is modified by a curvature term proportional to the associator.
//! At the critical curvature, ALL products collapse to TopologicalNull,
//! modeling the information-destroying singularity.
//!
//! The curvature-modified product is:
//!   a *_kappa b = a*b + kappa * sum_k associator(a, b, e_k)
//!
//! where the sum runs over all 15 imaginary basis elements e_1..e_15.

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
/// a *_kappa b = a*b + kappa * sum_k associator(a, b, e_k)
///
/// The sum runs over all 15 imaginary basis elements (e_1..e_15). Using a single
/// real unit e_0 would give zero (the real unit always associates). The sum over
/// all imaginary units provides a basis-independent curvature coupling that
/// activates the full non-associative structure of the sedenion algebra.
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

    let result_components: [f64; 16] = {
        let sp = standard_product.to_slice();
        let ac = assoc_sum.to_slice();
        let mut r = [0.0; 16];
        for i in 0..16 {
            r[i] = sp[i] + kappa * ac[i];
        }
        r
    };

    let result = Sedenion::from_slice(&result_components);

    // Check for zero-divisor collapse
    if a.norm_sqr() > 1e-12 && b.norm_sqr() > 1e-12 && result.norm_sqr() < 1e-12 {
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

/// Find the critical curvature where ALL products collapse to TopologicalNull.
///
/// Uses binary search on kappa in [0, max_kappa]. At each kappa, tests
/// `n_pairs` random sedenion pairs, computes the null fraction. Critical
/// curvature = where null_fraction > (1 - epsilon).
pub fn find_critical_curvature(
    n_pairs: usize,
    epsilon: f64,
    seed: u64,
) -> CriticalCurvatureResult {
    let max_kappa = 1000.0;
    let mut lo = 0.0_f64;
    let mut hi = max_kappa;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Pre-generate random sedenion pairs
    let normal = StandardNormal;
    let pairs: Vec<(Sedenion, Sedenion)> = (0..n_pairs)
        .map(|_| {
            let mut a_comp = [0.0; 16];
            let mut b_comp = [0.0; 16];
            for i in 0..16 {
                a_comp[i] = normal.sample(&mut rng);
                b_comp[i] = normal.sample(&mut rng);
            }
            // Normalize to unit sedenions
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

    let null_fraction_at = |kappa: f64| -> f64 {
        let null_count = pairs
            .iter()
            .filter(|(a, b)| curvature_product(a, b, kappa) == QuantumState::TopologicalNull)
            .count();
        null_count as f64 / n_pairs as f64
    };

    // Binary search for critical kappa
    let target = 1.0 - epsilon;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let frac = null_fraction_at(mid);
        if frac >= target {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo) < 1e-6 {
            break;
        }
    }

    let kappa_critical = (lo + hi) / 2.0;
    let null_frac = null_fraction_at(kappa_critical);

    // Compute mean associator norm at critical kappa (sum over imaginary units)
    let mean_assoc_norm: f64 = {
        let total: f64 = pairs
            .iter()
            .map(|(a, b)| {
                let mut assoc_sum = Sedenion::default();
                for k in 1..16 {
                    let mut ek_comp = [0.0; 16];
                    ek_comp[k] = 1.0;
                    let ek = Sedenion::from_slice(&ek_comp);
                    assoc_sum = assoc_sum + sedenion_associator(a, b, &ek);
                }
                assoc_sum.norm_sqr().sqrt()
            })
            .sum();
        total / n_pairs as f64
    };

    // Estimate transition sharpness via finite difference
    let dk = 0.01;
    let frac_plus = null_fraction_at(kappa_critical + dk);
    let frac_minus = null_fraction_at(kappa_critical - dk);
    let sharpness = (frac_plus - frac_minus) / (2.0 * dk);

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

        println!("--- CRITICAL CURVATURE SEARCH ---");
        println!("  kappa_critical     = {:.4}", result.kappa_critical);
        println!("  null_fraction      = {:.4}", result.null_fraction_at_critical);
        println!("  mean_assoc_norm    = {:.6}", result.associator_norm_at_critical);
        println!("  transition_sharpness = {:.4}", result.transition_sharpness);

        assert!(
            result.kappa_critical.is_finite() && result.kappa_critical > 0.0,
            "critical curvature must be finite and positive"
        );
        assert!(
            result.null_fraction_at_critical >= 0.9,
            "null fraction at critical should be near 1.0, got {}",
            result.null_fraction_at_critical
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
