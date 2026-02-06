//! State estimation and filtering.
//!
//! Provides Kalman-style filtering for state estimation from noisy measurements.
//! Key abstraction: StateEstimator trait with predict/update cycle.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors in state estimation.
#[derive(Debug, Clone, Error)]
pub enum FilterError {
    /// Dimension mismatch in matrices.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Covariance matrix not positive definite.
    #[error("Covariance matrix not positive definite")]
    NotPositiveDefinite,

    /// Numerical instability detected.
    #[error("Numerical instability: {reason}")]
    NumericalInstability { reason: String },
}

/// Result of a prediction step.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Predicted state mean.
    pub x_pred: DVector<f64>,
    /// Predicted state covariance.
    pub p_pred: DMatrix<f64>,
}

/// Result of an update step.
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// Updated state mean.
    pub x_post: DVector<f64>,
    /// Updated state covariance.
    pub p_post: DMatrix<f64>,
    /// Innovation (measurement residual).
    pub innovation: DVector<f64>,
    /// Innovation covariance.
    pub innovation_cov: DMatrix<f64>,
    /// Kalman gain used.
    pub kalman_gain: DMatrix<f64>,
}

/// Current filter state.
#[derive(Debug, Clone)]
pub struct FilterState {
    /// State estimate mean.
    pub x: DVector<f64>,
    /// State estimate covariance.
    pub p: DMatrix<f64>,
    /// Number of updates performed.
    pub n_updates: usize,
}

impl FilterState {
    /// Create a new filter state.
    pub fn new(x: DVector<f64>, p: DMatrix<f64>) -> Self {
        Self { x, p, n_updates: 0 }
    }

    /// State dimension.
    pub fn dim(&self) -> usize {
        self.x.len()
    }

    /// Get standard deviations (sqrt of diagonal of P).
    pub fn std_devs(&self) -> DVector<f64> {
        DVector::from_iterator(self.dim(), self.p.diagonal().iter().map(|v| v.sqrt()))
    }
}

/// A state estimator with predict/update cycle.
pub trait StateEstimator {
    /// Predict state forward given control input.
    fn predict(&mut self, u: &DVector<f64>, dt: f64) -> PredictionResult;

    /// Update state given measurement.
    fn update(&mut self, z: &DVector<f64>) -> Result<UpdateResult, FilterError>;

    /// Get current filter state.
    fn state(&self) -> &FilterState;

    /// Reset to initial conditions.
    fn reset(&mut self);
}

/// Standard Kalman filter for linear systems.
///
/// System model:
///   x_{k+1} = A * x_k + B * u_k + w_k,  w_k ~ N(0, Q)
///   z_k = H * x_k + v_k,                v_k ~ N(0, R)
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State transition matrix A (n x n).
    pub a: DMatrix<f64>,
    /// Control input matrix B (n x m).
    pub b: DMatrix<f64>,
    /// Observation matrix H (p x n).
    pub h: DMatrix<f64>,
    /// Process noise covariance Q (n x n).
    pub q: DMatrix<f64>,
    /// Measurement noise covariance R (p x p).
    pub r: DMatrix<f64>,
    /// Current filter state.
    state: FilterState,
    /// Initial state for reset.
    initial_state: FilterState,
}

impl KalmanFilter {
    /// Create a new Kalman filter.
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        h: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        x0: DVector<f64>,
        p0: DMatrix<f64>,
    ) -> Self {
        let initial = FilterState::new(x0.clone(), p0.clone());
        Self {
            a,
            b,
            h,
            q,
            r,
            state: FilterState::new(x0, p0),
            initial_state: initial,
        }
    }

    /// Create with identity matrices for simple cases.
    pub fn simple(n: usize, process_noise: f64, measurement_noise: f64) -> Self {
        let a = DMatrix::identity(n, n);
        let b = DMatrix::zeros(n, 1);
        let h = DMatrix::identity(n, n);
        let q = DMatrix::identity(n, n) * process_noise;
        let r = DMatrix::identity(n, n) * measurement_noise;
        let x0 = DVector::zeros(n);
        let p0 = DMatrix::identity(n, n);

        Self::new(a, b, h, q, r, x0, p0)
    }

    /// State dimension.
    pub fn state_dim(&self) -> usize {
        self.a.nrows()
    }

    /// Input dimension.
    pub fn input_dim(&self) -> usize {
        self.b.ncols()
    }

    /// Measurement dimension.
    pub fn measurement_dim(&self) -> usize {
        self.h.nrows()
    }
}

impl StateEstimator for KalmanFilter {
    fn predict(&mut self, u: &DVector<f64>, _dt: f64) -> PredictionResult {
        // x_pred = A * x + B * u
        let x_pred = &self.a * &self.state.x + &self.b * u;

        // P_pred = A * P * A^T + Q
        let p_pred = &self.a * &self.state.p * self.a.transpose() + &self.q;

        // Update state
        self.state.x = x_pred.clone();
        self.state.p = p_pred.clone();

        PredictionResult { x_pred, p_pred }
    }

    fn update(&mut self, z: &DVector<f64>) -> Result<UpdateResult, FilterError> {
        // Innovation: y = z - H * x
        let innovation = z - &self.h * &self.state.x;

        // Innovation covariance: S = H * P * H^T + R
        let innovation_cov = &self.h * &self.state.p * self.h.transpose() + &self.r;

        // Kalman gain: K = P * H^T * S^{-1}
        let s_inv = innovation_cov.clone().try_inverse().ok_or_else(|| {
            FilterError::NumericalInstability {
                reason: "Innovation covariance not invertible".to_string(),
            }
        })?;

        let kalman_gain = &self.state.p * self.h.transpose() * &s_inv;

        // Update state: x_post = x + K * y
        let x_post = &self.state.x + &kalman_gain * &innovation;

        // Update covariance: P_post = (I - K * H) * P
        let n = self.state_dim();
        let i_kh = DMatrix::identity(n, n) - &kalman_gain * &self.h;
        let p_post = &i_kh * &self.state.p;

        // Store updated state
        self.state.x = x_post.clone();
        self.state.p = p_post.clone();
        self.state.n_updates += 1;

        Ok(UpdateResult {
            x_post,
            p_post,
            innovation,
            innovation_cov,
            kalman_gain,
        })
    }

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = self.initial_state.clone();
    }
}

/// Extended Kalman filter for nonlinear systems.
///
/// Requires user-provided Jacobians for the nonlinear dynamics.
#[derive(Debug, Clone)]
pub struct ExtendedKalmanFilter {
    /// Process noise covariance Q (n x n).
    pub q: DMatrix<f64>,
    /// Measurement noise covariance R (p x p).
    pub r: DMatrix<f64>,
    /// Current filter state.
    state: FilterState,
    /// Initial state for reset.
    initial_state: FilterState,
    /// Cached state Jacobian (set by user before predict).
    f_jacobian: Option<DMatrix<f64>>,
    /// Cached observation Jacobian (set by user before update).
    h_jacobian: Option<DMatrix<f64>>,
}

impl ExtendedKalmanFilter {
    /// Create a new EKF.
    pub fn new(
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        x0: DVector<f64>,
        p0: DMatrix<f64>,
    ) -> Self {
        let initial = FilterState::new(x0.clone(), p0.clone());
        Self {
            q,
            r,
            state: FilterState::new(x0, p0),
            initial_state: initial,
            f_jacobian: None,
            h_jacobian: None,
        }
    }

    /// Set the state transition Jacobian F = df/dx evaluated at current state.
    pub fn set_state_jacobian(&mut self, f: DMatrix<f64>) {
        self.f_jacobian = Some(f);
    }

    /// Set the observation Jacobian H = dh/dx evaluated at current state.
    pub fn set_observation_jacobian(&mut self, h: DMatrix<f64>) {
        self.h_jacobian = Some(h);
    }

    /// Update state directly (for use with external nonlinear propagation).
    pub fn set_predicted_state(&mut self, x_pred: DVector<f64>) {
        self.state.x = x_pred;
    }

    /// State dimension.
    pub fn state_dim(&self) -> usize {
        self.state.x.len()
    }
}

impl StateEstimator for ExtendedKalmanFilter {
    fn predict(&mut self, _u: &DVector<f64>, _dt: f64) -> PredictionResult {
        // For EKF, user must:
        // 1. Propagate state externally: x_pred = f(x, u)
        // 2. Compute Jacobian F = df/dx at x
        // 3. Call set_predicted_state(x_pred) and set_state_jacobian(F)
        // Then call this to update covariance.

        let f = self.f_jacobian.as_ref().expect(
            "EKF predict requires state Jacobian. Call set_state_jacobian first.",
        );

        // P_pred = F * P * F^T + Q
        let p_pred = f * &self.state.p * f.transpose() + &self.q;
        self.state.p = p_pred.clone();

        PredictionResult {
            x_pred: self.state.x.clone(),
            p_pred,
        }
    }

    fn update(&mut self, z: &DVector<f64>) -> Result<UpdateResult, FilterError> {
        let h = self.h_jacobian.as_ref().ok_or_else(|| FilterError::NumericalInstability {
            reason: "EKF update requires observation Jacobian. Call set_observation_jacobian first.".to_string(),
        })?;

        // For EKF, user must compute h(x) externally.
        // Here we assume z is the innovation = z_measured - h(x_pred).
        // If z is the raw measurement, user should pass z - h(x).
        let innovation = z.clone();

        // Innovation covariance: S = H * P * H^T + R
        let innovation_cov = h * &self.state.p * h.transpose() + &self.r;

        // Kalman gain
        let s_inv = innovation_cov.clone().try_inverse().ok_or_else(|| {
            FilterError::NumericalInstability {
                reason: "Innovation covariance not invertible".to_string(),
            }
        })?;

        let kalman_gain = &self.state.p * h.transpose() * &s_inv;

        // Update state
        let x_post = &self.state.x + &kalman_gain * &innovation;

        // Joseph form for numerical stability
        let n = self.state_dim();
        let i_kh = DMatrix::identity(n, n) - &kalman_gain * h;
        let p_post = &i_kh * &self.state.p * i_kh.transpose()
            + &kalman_gain * &self.r * kalman_gain.transpose();

        self.state.x = x_post.clone();
        self.state.p = p_post.clone();
        self.state.n_updates += 1;

        Ok(UpdateResult {
            x_post,
            p_post,
            innovation,
            innovation_cov,
            kalman_gain,
        })
    }

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn reset(&mut self) {
        self.state = self.initial_state.clone();
        self.f_jacobian = None;
        self.h_jacobian = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kalman_filter_creation() {
        let kf = KalmanFilter::simple(2, 0.1, 0.5);
        assert_eq!(kf.state_dim(), 2);
        assert_eq!(kf.measurement_dim(), 2);
    }

    #[test]
    fn test_kalman_filter_predict() {
        let mut kf = KalmanFilter::simple(2, 0.1, 0.5);
        let u = DVector::zeros(1);
        let result = kf.predict(&u, 0.01);

        // State should remain zero
        assert_relative_eq!(result.x_pred.norm(), 0.0, epsilon = 1e-10);

        // Covariance should grow by Q
        assert!(result.p_pred[(0, 0)] > 1.0);
    }

    #[test]
    fn test_kalman_filter_update() {
        let mut kf = KalmanFilter::simple(2, 0.1, 0.5);

        // Predict
        let u = DVector::zeros(1);
        kf.predict(&u, 0.01);

        // Update with measurement
        let z = DVector::from_vec(vec![1.0, 0.5]);
        let result = kf.update(&z).unwrap();

        // State should move toward measurement
        assert!(result.x_post[0] > 0.0);
        assert!(result.x_post[1] > 0.0);

        // Covariance should shrink
        assert!(result.p_post[(0, 0)] < kf.state.p[(0, 0)] + 0.1);
    }

    #[test]
    fn test_kalman_filter_convergence() {
        let mut kf = KalmanFilter::simple(1, 0.01, 0.1);

        let true_state = 5.0;
        let u = DVector::zeros(1);

        // Run several predict-update cycles
        for _ in 0..20 {
            kf.predict(&u, 0.01);

            // Noisy measurement (here just true state for simplicity)
            let z = DVector::from_vec(vec![true_state]);
            kf.update(&z).unwrap();
        }

        // Should converge near true state
        assert_relative_eq!(kf.state().x[0], true_state, epsilon = 0.5);
    }

    #[test]
    fn test_filter_state_std_devs() {
        let p = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 9.0, 16.0]));
        let x = DVector::zeros(3);
        let state = FilterState::new(x, p);

        let std = state.std_devs();
        assert_relative_eq!(std[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(std[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(std[2], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kalman_filter_reset() {
        let mut kf = KalmanFilter::simple(2, 0.1, 0.5);

        // Run some updates
        let u = DVector::zeros(1);
        kf.predict(&u, 0.01);
        let z = DVector::from_vec(vec![1.0, 0.5]);
        kf.update(&z).unwrap();

        // Reset
        kf.reset();

        // Should be back to initial
        assert_eq!(kf.state().n_updates, 0);
        assert_relative_eq!(kf.state().x.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ekf_creation() {
        let q = DMatrix::identity(2, 2) * 0.1;
        let r = DMatrix::identity(2, 2) * 0.5;
        let x0 = DVector::zeros(2);
        let p0 = DMatrix::identity(2, 2);

        let ekf = ExtendedKalmanFilter::new(q, r, x0, p0);
        assert_eq!(ekf.state_dim(), 2);
    }
}
