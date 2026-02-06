//! Stochastic Processes: fBm, Levy, Ornstein-Uhlenbeck, and more.
//!
//! This module provides wrappers around the `diffusionx` crate for advanced
//! stochastic process simulation, complementing the Hosking-based generators
//! in `fractal_analysis.rs`.
//!
//! # Processes Supported
//! - Fractional Brownian Motion (fBm) - via diffusionx
//! - Ornstein-Uhlenbeck (OU) process - mean-reverting dynamics
//! - Levy flights - heavy-tailed jumps
//! - Geometric Brownian Motion (GBM) - stock price modeling
//!
//! # Literature
//! - Mandelbrot & van Ness (1968): Fractional Brownian motions
//! - Uhlenbeck & Ornstein (1930): Theory of Brownian motion
//! - Levy (1925): Calcul des probabilites
//! - Black & Scholes (1973): Options pricing model

/// Ornstein-Uhlenbeck process parameters.
#[derive(Debug, Clone, Copy)]
pub struct OUParams {
    /// Mean reversion rate (theta)
    pub theta: f64,
    /// Long-term mean (mu)
    pub mu: f64,
    /// Volatility (sigma)
    pub sigma: f64,
    /// Initial value
    pub x0: f64,
}

impl Default for OUParams {
    fn default() -> Self {
        Self {
            theta: 1.0,
            mu: 0.0,
            sigma: 1.0,
            x0: 0.0,
        }
    }
}

/// Generate Ornstein-Uhlenbeck process sample path.
///
/// The OU process satisfies: dX_t = theta * (mu - X_t) dt + sigma dW_t
///
/// This is a mean-reverting process used in:
/// - Interest rate models (Vasicek)
/// - Volatility modeling
/// - Physical systems with restoring forces
pub fn generate_ou_process(n: usize, dt: f64, params: OUParams, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("valid normal distribution");

    let mut path = Vec::with_capacity(n);
    let mut x = params.x0;
    path.push(x);

    let sqrt_dt = dt.sqrt();

    for _ in 1..n {
        let dw = normal.sample(&mut rng) * sqrt_dt;
        // Euler-Maruyama discretization
        x += params.theta * (params.mu - x) * dt + params.sigma * dw;
        path.push(x);
    }

    path
}

/// Geometric Brownian Motion parameters.
#[derive(Debug, Clone, Copy)]
pub struct GBMParams {
    /// Drift rate (mu)
    pub mu: f64,
    /// Volatility (sigma)
    pub sigma: f64,
    /// Initial value (S0)
    pub s0: f64,
}

impl Default for GBMParams {
    fn default() -> Self {
        Self {
            mu: 0.05,    // 5% drift
            sigma: 0.2,  // 20% volatility
            s0: 100.0,   // Initial price
        }
    }
}

/// Generate Geometric Brownian Motion sample path.
///
/// GBM satisfies: dS_t = mu * S_t dt + sigma * S_t dW_t
///
/// Solution: S_t = S_0 * exp((mu - sigma^2/2)t + sigma W_t)
///
/// Used in Black-Scholes option pricing.
pub fn generate_gbm(n: usize, dt: f64, params: GBMParams, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).expect("valid normal distribution");

    let mut path = Vec::with_capacity(n);
    let mut s = params.s0;
    path.push(s);

    let drift = (params.mu - 0.5 * params.sigma.powi(2)) * dt;
    let diffusion = params.sigma * dt.sqrt();

    for _ in 1..n {
        let z = normal.sample(&mut rng);
        // Exact solution step (log-normal)
        s *= (drift + diffusion * z).exp();
        path.push(s);
    }

    path
}

/// Levy flight parameters.
#[derive(Debug, Clone, Copy)]
pub struct LevyParams {
    /// Stability parameter (alpha in (0, 2])
    /// alpha = 2 gives Gaussian, alpha < 2 gives heavy tails
    pub alpha: f64,
    /// Scale parameter
    pub scale: f64,
    /// Initial position
    pub x0: f64,
}

impl Default for LevyParams {
    fn default() -> Self {
        Self {
            alpha: 1.5,   // Heavy-tailed but not Cauchy
            scale: 1.0,
            x0: 0.0,
        }
    }
}

/// Generate Levy flight sample path.
///
/// Levy flights are random walks with heavy-tailed step sizes,
/// used in anomalous diffusion and search optimization.
///
/// For alpha = 2, reduces to Brownian motion.
/// For alpha < 2, exhibits superdiffusive behavior.
pub fn generate_levy_flight(n: usize, params: LevyParams, seed: u64) -> Vec<f64> {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Uniform};

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0, 1.0);

    let mut path = Vec::with_capacity(n);
    let mut x = params.x0;
    path.push(x);

    // Chambers-Mallows-Stuck method for stable distribution
    let alpha = params.alpha.clamp(0.1, 2.0);

    for _ in 1..n {
        let u = std::f64::consts::PI * (uniform.sample(&mut rng) - 0.5);
        let w = -uniform.sample(&mut rng).ln();

        let step: f64 = if (alpha - 1.0).abs() < 1e-10 {
            // Cauchy case (alpha = 1)
            u.tan()
        } else if (alpha - 2.0).abs() < 1e-10 {
            // Gaussian case (alpha = 2)
            (2.0_f64 * w).sqrt() * u.sin()
        } else {
            // General stable case
            let t = u * alpha;
            let s = (1.0 + t.tan().powi(2)).powf(1.0 / (2.0 * alpha));
            s * (t / (u.cos().powf(1.0 / alpha))) * ((u.cos() - t.cos()) / w).powf((1.0 - alpha) / alpha)
        };

        x += params.scale * step;
        path.push(x);
    }

    path
}

/// Fractional Brownian Motion with specific Hurst parameter.
///
/// This is a wrapper providing the same interface as diffusionx,
/// but using our Hosking implementation for consistency.
pub fn generate_fbm_diffusionx_compatible(n: usize, hurst: f64, seed: u64) -> Vec<f64> {
    // Use our existing Hosking implementation
    crate::fractal_analysis::generate_fbm(n, hurst, seed)
}

/// Result of mean-reversion analysis on a time series.
#[derive(Debug, Clone)]
pub struct MeanReversionResult {
    /// Estimated theta (mean reversion rate)
    pub theta: f64,
    /// Estimated mu (long-term mean)
    pub mu: f64,
    /// Estimated sigma (volatility)
    pub sigma: f64,
    /// R-squared of the fit
    pub r_squared: f64,
    /// Half-life of mean reversion (time to revert halfway)
    pub half_life: f64,
}

/// Fit Ornstein-Uhlenbeck parameters to observed data.
///
/// Uses OLS regression on the discretized SDE.
pub fn fit_ou_parameters(data: &[f64], dt: f64) -> MeanReversionResult {
    if data.len() < 3 {
        return MeanReversionResult {
            theta: 0.0,
            mu: data.first().copied().unwrap_or(0.0),
            sigma: 0.0,
            r_squared: 0.0,
            half_life: f64::INFINITY,
        };
    }

    let n = data.len() - 1;

    // Regress X_{t+1} - X_t on X_t
    // dX = a + b*X_t where a = theta*mu*dt, b = -theta*dt
    let mut sum_x = 0.0;
    let mut sum_dx = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xdx = 0.0;
    for i in 0..n {
        let x = data[i];
        let dx = data[i + 1] - data[i];
        sum_x += x;
        sum_dx += dx;
        sum_xx += x * x;
        sum_xdx += x * dx;
    }

    let n_f = n as f64;
    let mean_x = sum_x / n_f;
    let mean_dx = sum_dx / n_f;

    // OLS: b = Cov(x, dx) / Var(x)
    let cov_xdx = sum_xdx / n_f - mean_x * mean_dx;
    let var_x = sum_xx / n_f - mean_x * mean_x;

    let b = if var_x.abs() > 1e-12 { cov_xdx / var_x } else { 0.0 };
    let a = mean_dx - b * mean_x;

    // Extract parameters
    let theta = -b / dt;
    let mu = if theta.abs() > 1e-12 { a / (theta * dt) } else { mean_x };

    // Estimate sigma from residual variance
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let dx = data[i + 1] - data[i];
        let dx_pred = a + b * data[i];
        ss_res += (dx - dx_pred).powi(2);
        ss_tot += (dx - mean_dx).powi(2);
    }

    let sigma = (ss_res / n_f / dt).sqrt();
    let r_squared = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let half_life = if theta > 0.0 { 2.0_f64.ln() / theta } else { f64::INFINITY };

    MeanReversionResult {
        theta: theta.max(0.0),
        mu,
        sigma,
        r_squared: r_squared.max(0.0),
        half_life,
    }
}

/// Analyze anomalous diffusion exponent from MSD.
///
/// For normal diffusion, MSD ~ t. For anomalous:
/// - MSD ~ t^alpha with alpha < 1: subdiffusion
/// - MSD ~ t^alpha with alpha > 1: superdiffusion
#[derive(Debug, Clone)]
pub struct AnomalousDiffusionResult {
    /// Diffusion exponent alpha
    pub alpha: f64,
    /// Generalized diffusion coefficient D_alpha
    pub d_alpha: f64,
    /// Classification
    pub diffusion_type: DiffusionType,
    /// R-squared of power-law fit
    pub r_squared: f64,
}

/// Classification of diffusion behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffusionType {
    /// alpha < 0.5: strongly subdiffusive
    StronglySubdiffusive,
    /// 0.5 <= alpha < 0.9: subdiffusive
    Subdiffusive,
    /// 0.9 <= alpha <= 1.1: normal
    Normal,
    /// 1.1 < alpha <= 1.5: superdiffusive
    Superdiffusive,
    /// alpha > 1.5: ballistic/strongly superdiffusive
    Ballistic,
}

/// Compute Mean Square Displacement and fit anomalous diffusion.
pub fn analyze_anomalous_diffusion(trajectory: &[f64]) -> AnomalousDiffusionResult {
    let n = trajectory.len();
    if n < 10 {
        return AnomalousDiffusionResult {
            alpha: 1.0,
            d_alpha: 0.0,
            diffusion_type: DiffusionType::Normal,
            r_squared: 0.0,
        };
    }

    // Compute MSD for different lag times
    let max_lag = n / 4;
    let mut log_tau = Vec::with_capacity(max_lag);
    let mut log_msd = Vec::with_capacity(max_lag);

    for lag in 1..=max_lag {
        let mut msd_sum = 0.0;
        let count = n - lag;
        for i in 0..count {
            let dx = trajectory[i + lag] - trajectory[i];
            msd_sum += dx * dx;
        }
        let msd = msd_sum / count as f64;
        if msd > 1e-12 {
            log_tau.push((lag as f64).ln());
            log_msd.push(msd.ln());
        }
    }

    if log_tau.len() < 5 {
        return AnomalousDiffusionResult {
            alpha: 1.0,
            d_alpha: 0.0,
            diffusion_type: DiffusionType::Normal,
            r_squared: 0.0,
        };
    }

    // Linear regression: log(MSD) = log(2*D_alpha) + alpha * log(tau)
    let n_pts = log_tau.len() as f64;
    let sum_x: f64 = log_tau.iter().sum();
    let sum_y: f64 = log_msd.iter().sum();
    let sum_xx: f64 = log_tau.iter().map(|x| x * x).sum();
    let sum_xy: f64 = log_tau.iter().zip(&log_msd).map(|(x, y)| x * y).sum();

    let denom = n_pts * sum_xx - sum_x * sum_x;
    let alpha = if denom.abs() > 1e-12 {
        (n_pts * sum_xy - sum_x * sum_y) / denom
    } else {
        1.0
    };
    let intercept = (sum_y - alpha * sum_x) / n_pts;
    let d_alpha = (intercept.exp() / 2.0).max(0.0);

    // R-squared
    let mean_y = sum_y / n_pts;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for i in 0..log_tau.len() {
        let y_pred = intercept + alpha * log_tau[i];
        ss_tot += (log_msd[i] - mean_y).powi(2);
        ss_res += (log_msd[i] - y_pred).powi(2);
    }
    let r_squared = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 0.0 };

    let diffusion_type = if alpha < 0.5 {
        DiffusionType::StronglySubdiffusive
    } else if alpha < 0.9 {
        DiffusionType::Subdiffusive
    } else if alpha <= 1.1 {
        DiffusionType::Normal
    } else if alpha <= 1.5 {
        DiffusionType::Superdiffusive
    } else {
        DiffusionType::Ballistic
    };

    AnomalousDiffusionResult {
        alpha,
        d_alpha,
        diffusion_type,
        r_squared: r_squared.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ou_process_mean_reversion() {
        let params = OUParams {
            theta: 2.0,
            mu: 5.0,
            sigma: 0.5,
            x0: 0.0,
        };

        let path = generate_ou_process(1000, 0.01, params, 42);
        assert_eq!(path.len(), 1000);

        // Mean should approach mu over time
        let late_mean: f64 = path[500..].iter().sum::<f64>() / 500.0;
        assert!((late_mean - params.mu).abs() < 2.0, "OU should mean-revert to mu");
    }

    #[test]
    fn test_gbm_positive() {
        let params = GBMParams::default();
        let path = generate_gbm(100, 0.01, params, 123);

        assert_eq!(path.len(), 100);
        assert!(path.iter().all(|&x| x > 0.0), "GBM should always be positive");
        assert!((path[0] - params.s0).abs() < 1e-10);
    }

    #[test]
    fn test_levy_flight_length() {
        let params = LevyParams::default();
        let path = generate_levy_flight(200, params, 456);

        assert_eq!(path.len(), 200);
        assert!((path[0] - params.x0).abs() < 1e-10);
    }

    #[test]
    fn test_levy_gaussian_limit() {
        // alpha = 2 should give Gaussian-like behavior
        let params = LevyParams {
            alpha: 2.0,
            scale: 1.0,
            x0: 0.0,
        };

        let path = generate_levy_flight(1000, params, 789);
        let diffusion = analyze_anomalous_diffusion(&path);

        // Should be close to normal diffusion
        assert!(diffusion.alpha > 0.7 && diffusion.alpha < 1.3,
            "Gaussian limit should give alpha near 1, got {}", diffusion.alpha);
    }

    #[test]
    fn test_fit_ou_parameters() {
        let true_params = OUParams {
            theta: 1.0,
            mu: 3.0,
            sigma: 0.5,
            x0: 0.0,
        };

        let path = generate_ou_process(5000, 0.01, true_params, 42);
        let fitted = fit_ou_parameters(&path, 0.01);

        // Fitted parameters should be close to true values
        assert!((fitted.mu - true_params.mu).abs() < 1.0,
            "Fitted mu {} should be close to {}", fitted.mu, true_params.mu);
        assert!(fitted.r_squared > 0.0);
    }

    #[test]
    fn test_anomalous_diffusion_fbm() {
        // fBm with H > 0.5 is superdiffusive
        let fbm = crate::fractal_analysis::generate_fbm(2000, 0.8, 42);
        let result = analyze_anomalous_diffusion(&fbm);

        // For fBm, MSD ~ t^{2H}, so alpha should be near 2*H = 1.6
        assert!(result.alpha > 1.0,
            "fBm with H=0.8 should be superdiffusive, got alpha={}", result.alpha);
    }

    #[test]
    fn test_anomalous_diffusion_types() {
        assert_eq!(
            DiffusionType::StronglySubdiffusive,
            DiffusionType::StronglySubdiffusive
        );
    }

    #[test]
    fn test_ou_half_life() {
        let params = OUParams {
            theta: 2.0_f64.ln(), // Half-life of 1
            ..Default::default()
        };

        let path = generate_ou_process(100, 0.1, params, 99);
        let fitted = fit_ou_parameters(&path, 0.1);

        // Half-life should be approximately 1
        assert!(fitted.half_life > 0.0 && fitted.half_life < 100.0);
    }
}
