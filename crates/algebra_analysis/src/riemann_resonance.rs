//! Riemann Resonance Generator: prime-harmonic analysis of CD zero-divisor
//! eigenvalue spectra.
//!
//! The zero-divisor graph adjacency matrix A(dim) has eigenvalues whose spacing
//! statistics encode algebraic structure. This module tests whether CD dimensions
//! at 2^p (p prime: 2,3,5,7) exhibit qualitatively different spectral statistics
//! from 2^q (q composite: 4,6,8).

use crate::quantum_chaos::{nnsd_census, NnsdResult, SpacingVerdict};

/// Classification of a CD dimension by the primality of its doubling exponent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoublingClass {
    /// dim = 2^p where p is prime (e.g. 4=2^2, 8=2^3, 32=2^5, 128=2^7)
    PrimePower,
    /// dim = 2^q where q is composite (e.g. 16=2^4, 64=2^6, 256=2^8)
    CompositePower,
    /// dim = 2^1 (complex numbers) -- trivially commutative, excluded from comparison
    Trivial,
}

impl std::fmt::Display for DoublingClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrimePower => write!(f, "prime"),
            Self::CompositePower => write!(f, "composite"),
            Self::Trivial => write!(f, "trivial"),
        }
    }
}

/// Result for a single dimension in the resonance analysis.
#[derive(Debug, Clone)]
pub struct ResonanceResult {
    /// CD dimension (power of 2).
    pub dim: usize,
    /// The exponent k where dim = 2^k.
    pub exponent: u32,
    /// Whether the exponent is prime.
    pub doubling_class: DoublingClass,
    /// NNSD analysis result (None if dim has no ZD graph, e.g. quaternion/octonion).
    pub nnsd: Option<NnsdResult>,
    /// Largest eigenvalue gap (spectral gap) in the ZD Laplacian.
    pub spectral_gap: Option<f64>,
}

/// Summary comparing prime-power vs composite-power spectral statistics.
#[derive(Debug, Clone)]
pub struct ResonanceComparison {
    /// All individual results.
    pub results: Vec<ResonanceResult>,
    /// Mean Brody q for prime-power dims (excluding degenerate).
    pub mean_q_prime: Option<f64>,
    /// Mean Brody q for composite-power dims (excluding degenerate).
    pub mean_q_composite: Option<f64>,
    /// Whether prime-power dims have strictly higher mean q.
    pub prime_higher_q: Option<bool>,
}

fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut i = 5;
    while i * i <= n {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

fn classify_dim(dim: usize) -> Option<(u32, DoublingClass)> {
    if !dim.is_power_of_two() || dim < 2 {
        return None;
    }
    let exp = dim.trailing_zeros();
    let class = match exp {
        0 => return None, // dim=1 not a CD algebra
        1 => DoublingClass::Trivial,
        _ if is_prime(exp) => DoublingClass::PrimePower,
        _ => DoublingClass::CompositePower,
    };
    Some((exp, class))
}

/// Compute the largest eigenvalue gap from an NNSD result's spacings.
fn largest_spacing(nnsd: &NnsdResult) -> f64 {
    nnsd.spacings
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
}

/// Run the Riemann Resonance analysis across the standard CD dimensions.
///
/// Default dims: [4, 8, 16, 32, 64, 128, 256].
pub fn riemann_resonance_sweep(dims: &[usize]) -> ResonanceComparison {
    // Run nnsd_census for all dims at once (reuses the existing pipeline).
    let nnsd_results = nnsd_census(dims);

    let mut results = Vec::with_capacity(dims.len());

    for &dim in dims {
        let (exponent, doubling_class) = match classify_dim(dim) {
            Some(pair) => pair,
            None => continue,
        };

        // Find the NNSD result for this dim (nnsd_census may skip dims with no ZD).
        let nnsd = nnsd_results.iter().find(|r| r.cd_dim == dim).cloned();

        let spectral_gap = nnsd.as_ref().map(largest_spacing);

        results.push(ResonanceResult {
            dim,
            exponent,
            doubling_class,
            nnsd,
            spectral_gap,
        });
    }

    // Compute mean Brody q for prime vs composite dims.
    let prime_qs: Vec<f64> = results
        .iter()
        .filter(|r| r.doubling_class == DoublingClass::PrimePower)
        .filter_map(|r| {
            r.nnsd.as_ref().and_then(|n| {
                if matches!(n.verdict, SpacingVerdict::DegenerateIntegrable) {
                    None
                } else {
                    Some(n.brody_q)
                }
            })
        })
        .collect();

    let composite_qs: Vec<f64> = results
        .iter()
        .filter(|r| r.doubling_class == DoublingClass::CompositePower)
        .filter_map(|r| {
            r.nnsd.as_ref().and_then(|n| {
                if matches!(n.verdict, SpacingVerdict::DegenerateIntegrable) {
                    None
                } else {
                    Some(n.brody_q)
                }
            })
        })
        .collect();

    let mean_q_prime = if prime_qs.is_empty() {
        None
    } else {
        Some(prime_qs.iter().sum::<f64>() / prime_qs.len() as f64)
    };

    let mean_q_composite = if composite_qs.is_empty() {
        None
    } else {
        Some(composite_qs.iter().sum::<f64>() / composite_qs.len() as f64)
    };

    let prime_higher_q = match (mean_q_prime, mean_q_composite) {
        (Some(p), Some(c)) => Some(p > c),
        _ => None,
    };

    ResonanceComparison {
        results,
        mean_q_prime,
        mean_q_composite,
        prime_higher_q,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_known_dims() {
        // 4 = 2^2 (2 is prime)
        assert_eq!(classify_dim(4), Some((2, DoublingClass::PrimePower)));
        // 8 = 2^3 (3 is prime)
        assert_eq!(classify_dim(8), Some((3, DoublingClass::PrimePower)));
        // 16 = 2^4 (4 is composite)
        assert_eq!(classify_dim(16), Some((4, DoublingClass::CompositePower)));
        // 32 = 2^5 (5 is prime)
        assert_eq!(classify_dim(32), Some((5, DoublingClass::PrimePower)));
        // 64 = 2^6 (6 is composite)
        assert_eq!(classify_dim(64), Some((6, DoublingClass::CompositePower)));
        // 128 = 2^7 (7 is prime)
        assert_eq!(classify_dim(128), Some((7, DoublingClass::PrimePower)));
        // 256 = 2^8 (8 is composite)
        assert_eq!(classify_dim(256), Some((8, DoublingClass::CompositePower)));
        // 2 = 2^1 (trivial)
        assert_eq!(classify_dim(2), Some((1, DoublingClass::Trivial)));
    }

    #[test]
    fn classify_non_power_of_two() {
        assert_eq!(classify_dim(3), None);
        assert_eq!(classify_dim(6), None);
        assert_eq!(classify_dim(0), None);
    }

    #[test]
    fn is_prime_basic() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(6));
        assert!(is_prime(7));
        assert!(!is_prime(8));
    }

    #[test]
    fn small_sweep_runs() {
        // Only test dims 16 and 32 to keep test fast.
        let comp = riemann_resonance_sweep(&[16, 32]);
        assert_eq!(comp.results.len(), 2);
        assert_eq!(comp.results[0].dim, 16);
        assert_eq!(comp.results[0].doubling_class, DoublingClass::CompositePower);
        assert_eq!(comp.results[1].dim, 32);
        assert_eq!(comp.results[1].doubling_class, DoublingClass::PrimePower);
    }
}
