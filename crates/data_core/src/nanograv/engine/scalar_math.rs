//! Pure scalar / small-array math helpers used across the NANOGrav
//! timing engine.
//!
//! None of these depend on engine types or external crates beyond
//! `f64`'s built-in methods. Kept in their own submodule so the main
//! engine.rs file stays focused on the physical timing model.
//!
//! Functions:
//!   * `wrap_cycles`     -- fold a phase residual into (-0.5, 0.5]
//!   * `fract`           -- non-negative fractional part of an f64
//!   * `dot3`            -- 3D dot product
//!   * `norm3`           -- 3D Euclidean norm
//!   * `matern_three_halves`  -- Matern-3/2 covariance kernel
//!   * `gaussian_kernel`      -- isotropic Gaussian kernel
//!   * `median_value`    -- finite-positive median (used as a robust scale)
//!   * `fractional_improvement` -- (before - after) / |before|
//!   * `synthesis_score` -- weighted combination of raw / weighted / DM RMS

pub(super) fn wrap_cycles(value: f64) -> f64 {
    let wrapped = value - value.round();
    if wrapped >= 0.5 {
        wrapped - 1.0
    } else if wrapped < -0.5 {
        wrapped + 1.0
    } else {
        wrapped
    }
}

pub(super) fn fract(value: f64) -> f64 {
    value - value.floor()
}

pub(super) fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn norm3(value: [f64; 3]) -> f64 {
    dot3(value, value).sqrt()
}

pub(super) fn matern_three_halves(scaled_distance: f64) -> f64 {
    let x = 3.0_f64.sqrt() * scaled_distance.abs();
    (1.0 + x) * (-x).exp()
}

pub(super) fn gaussian_kernel(scaled_distance: f64) -> f64 {
    (-0.5 * scaled_distance * scaled_distance).exp()
}

pub(super) fn median_value(values: &[f64]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    finite.sort_by(|left, right| left.total_cmp(right));
    finite
        .get(finite.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(0.0)
}

pub(super) fn fractional_improvement(before: f64, after: f64) -> f64 {
    if !before.is_finite() || before.abs() < 1.0e-18 {
        0.0
    } else {
        (before - after) / before.abs()
    }
}

pub(super) fn synthesis_score(raw: f64, weighted: f64, dm: Option<f64>) -> f64 {
    let dm_component = dm.unwrap_or(0.0);
    0.45 * raw + 0.45 * weighted + 0.10 * dm_component
}

/// Kepler equation: solve M = E - e*sin(E) for E.
/// Newton-Raphson with 24 iterations and 1e-14 tolerance.
pub(super) fn solve_kepler(mean_anomaly: f64, ecc: f64) -> f64 {
    let mut eccentric = mean_anomaly;
    for _ in 0..24 {
        let f = eccentric - ecc * eccentric.sin() - mean_anomaly;
        let fp = 1.0 - ecc * eccentric.cos();
        let delta = f / fp;
        eccentric -= delta;
        if delta.abs() < 1.0e-14 {
            break;
        }
    }
    eccentric
}

/// Convert eccentric anomaly E to true anomaly nu via the standard
/// half-angle identity. Guards 1 - e against zero division.
pub(super) fn true_anomaly_from_eccentric(eccentric_anomaly: f64, ecc: f64) -> f64 {
    let root = ((1.0 + ecc) / (1.0 - ecc).max(1.0e-12)).sqrt();
    2.0 * (root * (0.5 * eccentric_anomaly).tan()).atan()
}
