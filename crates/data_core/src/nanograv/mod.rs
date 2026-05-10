//! NANOGrav pulsar timing data structures and analysis engines.
//!
//! - `timing`: release loader, residuals, DM/DMX parsers, sky vectors.
//! - `timing_model`: structured `.par` file parser (astrometry, dispersion, noise).
//! - `refit`: phase-1 wideband refit solver (nalgebra least-squares).
//! - `engine`: independent timing engine (faer/anise barycentric correction).
//!
//! The `engine` submodule requires the `nanograv-engine` feature because it
//! depends on `faer` (dense linear algebra) and `anise` (NAIF/SPICE ephemeris).
//!
//! ## HD Synthetic Draws
//!
//! [`generate_hd_synthetic_draws`] wraps the Cholesky-based multivariate normal
//! sampler so CLI binaries do not need a direct nalgebra dep.

use anyhow::{Context, Result};
use nalgebra::{Cholesky, DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub mod refit;
pub mod timing;
pub mod timing_model;

#[cfg(feature = "nanograv-engine")]
pub mod engine;

/// Sample draws from a multivariate normal distribution defined by a
/// pre-computed correlation matrix.
///
/// WHY: The Cholesky factorization and DVector sampling require nalgebra, which
/// is a heavy dep.  CLI binaries that already compute the HD correlation matrix
/// (using `stats_core::astrophysics`) pass it here as a plain row-major
/// `&[Vec<f64>]` and receive results as `Vec<Vec<f64>>`.  No nalgebra types
/// escape across the boundary.
///
/// A small jitter (1e-9) is added to the diagonal before factorization to
/// ensure positive-definiteness under floating-point rounding.
///
/// Returns `Err` if the matrix is not positive definite after jittering.
pub fn generate_hd_synthetic_draws(
    correlation_rows: &[Vec<f64>],
    draws: usize,
    rng: &mut impl Rng,
) -> Result<Vec<Vec<f64>>> {
    let n = correlation_rows.len();
    let mut gamma = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            gamma[(i, j)] = *correlation_rows[i].get(j).unwrap_or(&0.0);
        }
        gamma[(i, i)] += 1e-9;
    }
    let chol = Cholesky::new(gamma).context("HD correlation matrix is not positive definite")?;
    let lower = chol.l();

    let mut result = Vec::with_capacity(draws);
    for _ in 0..draws {
        let z = DVector::from_iterator(n, (0..n).map(|_| StandardNormal.sample(rng)));
        let sample = &lower * z;
        result.push(sample.iter().copied().collect());
    }
    Ok(result)
}
