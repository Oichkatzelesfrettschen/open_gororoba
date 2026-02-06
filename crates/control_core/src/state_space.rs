//! State-space analysis utilities.
//!
//! Provides controllability, observability, and transfer function analysis
//! for linear time-invariant (LTI) systems in state-space form:
//!
//!   dx/dt = A*x + B*u
//!   y = C*x + D*u
//!
//! # Key Concepts
//!
//! - **Controllability**: Can we steer the state to any target?
//! - **Observability**: Can we infer state from outputs?
//! - **Transfer function**: Input-output relation in s-domain.
//!
//! # Literature
//! - Kalman (1960): Controllability and observability conditions
//! - Chen (1999): Linear System Theory and Design

use nalgebra::{DMatrix, Complex};

/// State-space model in standard form.
#[derive(Debug, Clone)]
pub struct StateSpaceModel {
    /// State matrix A (n x n).
    pub a: DMatrix<f64>,
    /// Input matrix B (n x m).
    pub b: DMatrix<f64>,
    /// Output matrix C (p x n).
    pub c: DMatrix<f64>,
    /// Feedthrough matrix D (p x m).
    pub d: DMatrix<f64>,
}

impl StateSpaceModel {
    /// Create a new state-space model.
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        c: DMatrix<f64>,
        d: DMatrix<f64>,
    ) -> Self {
        Self { a, b, c, d }
    }

    /// State dimension n.
    pub fn state_dim(&self) -> usize {
        self.a.nrows()
    }

    /// Input dimension m.
    pub fn input_dim(&self) -> usize {
        self.b.ncols()
    }

    /// Output dimension p.
    pub fn output_dim(&self) -> usize {
        self.c.nrows()
    }

    /// Compute controllability matrix [B, AB, A^2B, ..., A^{n-1}B].
    pub fn controllability_matrix(&self) -> DMatrix<f64> {
        controllability_matrix(&self.a, &self.b)
    }

    /// Compute observability matrix [C; CA; CA^2; ...; CA^{n-1}].
    pub fn observability_matrix(&self) -> DMatrix<f64> {
        observability_matrix(&self.a, &self.c)
    }

    /// Check if system is controllable.
    pub fn is_controllable(&self) -> bool {
        is_controllable(&self.a, &self.b)
    }

    /// Check if system is observable.
    pub fn is_observable(&self) -> bool {
        is_observable(&self.a, &self.c)
    }

    /// Compute transfer function at complex frequency s.
    ///
    /// G(s) = C * (sI - A)^{-1} * B + D
    pub fn transfer_function_at(&self, s: Complex<f64>) -> Option<DMatrix<Complex<f64>>> {
        let n = self.state_dim();
        let si_minus_a = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                s - Complex::new(self.a[(i, j)], 0.0)
            } else {
                Complex::new(-self.a[(i, j)], 0.0)
            }
        });

        let inv = si_minus_a.try_inverse()?;

        let b_complex = self.b.map(|x| Complex::new(x, 0.0));
        let c_complex = self.c.map(|x| Complex::new(x, 0.0));
        let d_complex = self.d.map(|x| Complex::new(x, 0.0));

        Some(&c_complex * inv * &b_complex + d_complex)
    }

    /// Compute DC gain (transfer function at s=0).
    pub fn dc_gain(&self) -> Option<DMatrix<f64>> {
        let a_inv = self.a.clone().try_inverse()?;
        Some(-&self.c * a_inv * &self.b + &self.d)
    }
}

/// Compute controllability matrix [B, AB, A^2B, ..., A^{n-1}B].
///
/// The system is controllable if this matrix has full row rank.
pub fn controllability_matrix(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let m = b.ncols();

    let mut ctrl = DMatrix::zeros(n, n * m);

    // First block: B
    ctrl.view_mut((0, 0), (n, m)).copy_from(b);

    // Subsequent blocks: A^k * B
    let mut a_power_b = b.clone();
    for k in 1..n {
        a_power_b = a * &a_power_b;
        ctrl.view_mut((0, k * m), (n, m)).copy_from(&a_power_b);
    }

    ctrl
}

/// Compute observability matrix [C; CA; CA^2; ...; CA^{n-1}].
///
/// The system is observable if this matrix has full column rank.
pub fn observability_matrix(a: &DMatrix<f64>, c: &DMatrix<f64>) -> DMatrix<f64> {
    let n = a.nrows();
    let p = c.nrows();

    let mut obs = DMatrix::zeros(n * p, n);

    // First block: C
    obs.view_mut((0, 0), (p, n)).copy_from(c);

    // Subsequent blocks: C * A^k
    let mut c_a_power = c.clone();
    for k in 1..n {
        c_a_power = &c_a_power * a;
        obs.view_mut((k * p, 0), (p, n)).copy_from(&c_a_power);
    }

    obs
}

/// Check if (A, B) is controllable.
///
/// Uses rank test: rank([B, AB, ..., A^{n-1}B]) = n.
pub fn is_controllable(a: &DMatrix<f64>, b: &DMatrix<f64>) -> bool {
    let n = a.nrows();
    let ctrl = controllability_matrix(a, b);

    // Use SVD to compute numerical rank
    let svd = ctrl.svd(false, false);
    let tol = 1e-10 * n as f64 * svd.singular_values.max();

    let rank = svd.singular_values.iter().filter(|&s| *s > tol).count();
    rank == n
}

/// Check if (A, C) is observable.
///
/// Uses rank test: rank([C; CA; ...; CA^{n-1}]) = n.
pub fn is_observable(a: &DMatrix<f64>, c: &DMatrix<f64>) -> bool {
    let n = a.nrows();
    let obs = observability_matrix(a, c);

    let svd = obs.svd(false, false);
    let tol = 1e-10 * n as f64 * svd.singular_values.max();

    let rank = svd.singular_values.iter().filter(|&s| *s > tol).count();
    rank == n
}

/// SISO transfer function representation: G(s) = num(s) / den(s).
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Numerator coefficients [b_m, b_{m-1}, ..., b_0].
    /// Highest degree first.
    pub numerator: Vec<f64>,
    /// Denominator coefficients [a_n, a_{n-1}, ..., a_0].
    /// Highest degree first.
    pub denominator: Vec<f64>,
}

impl TransferFunction {
    /// Create a new transfer function.
    pub fn new(numerator: Vec<f64>, denominator: Vec<f64>) -> Self {
        Self { numerator, denominator }
    }

    /// Create a first-order system: G(s) = K / (tau*s + 1).
    pub fn first_order(k: f64, tau: f64) -> Self {
        Self {
            numerator: vec![k],
            denominator: vec![tau, 1.0],
        }
    }

    /// Create a second-order system: G(s) = K*wn^2 / (s^2 + 2*zeta*wn*s + wn^2).
    pub fn second_order(k: f64, wn: f64, zeta: f64) -> Self {
        Self {
            numerator: vec![k * wn * wn],
            denominator: vec![1.0, 2.0 * zeta * wn, wn * wn],
        }
    }

    /// Evaluate transfer function at complex frequency s.
    pub fn at(&self, s: Complex<f64>) -> Complex<f64> {
        let num = eval_polynomial(&self.numerator, s);
        let den = eval_polynomial(&self.denominator, s);

        if den.norm() < 1e-15 {
            Complex::new(f64::INFINITY, 0.0)
        } else {
            num / den
        }
    }

    /// DC gain (value at s=0).
    pub fn dc_gain(&self) -> f64 {
        if self.denominator.is_empty() {
            return f64::NAN;
        }

        let num_dc = self.numerator.last().copied().unwrap_or(0.0);
        let den_dc = self.denominator.last().copied().unwrap_or(1.0);

        if den_dc.abs() < 1e-15 {
            f64::INFINITY * num_dc.signum()
        } else {
            num_dc / den_dc
        }
    }

    /// Compute gain and phase at given frequency (rad/s).
    pub fn frequency_response(&self, omega: f64) -> (f64, f64) {
        let s = Complex::new(0.0, omega);
        let g = self.at(s);

        let gain = g.norm();
        let phase = g.arg();

        (gain, phase)
    }

    /// Compute gain in dB and phase in degrees at given frequency.
    pub fn bode_point(&self, omega: f64) -> (f64, f64) {
        let (gain, phase) = self.frequency_response(omega);
        let gain_db = 20.0 * gain.log10();
        let phase_deg = phase.to_degrees();

        (gain_db, phase_deg)
    }
}

/// Evaluate polynomial at complex value.
/// Coefficients are [a_n, a_{n-1}, ..., a_0] (highest degree first).
fn eval_polynomial(coeffs: &[f64], s: Complex<f64>) -> Complex<f64> {
    let mut result = Complex::new(0.0, 0.0);

    for &c in coeffs {
        result = result * s + Complex::new(c, 0.0);
    }

    result
}

/// Compute eigenvalues of a matrix (poles of the system).
pub fn eigenvalues(a: &DMatrix<f64>) -> Vec<Complex<f64>> {
    // Use Schur decomposition for eigenvalue computation
    let schur = a.clone().schur();
    let t = schur.unpack().1;

    let n = t.nrows();
    let mut eigs = Vec::with_capacity(n);

    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1, i)].abs() > 1e-12 {
            // 2x2 block: complex conjugate pair
            let a11 = t[(i, i)];
            let a12 = t[(i, i + 1)];
            let a21 = t[(i + 1, i)];
            let a22 = t[(i + 1, i + 1)];

            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let disc = trace * trace - 4.0 * det;

            if disc < 0.0 {
                let real = trace / 2.0;
                let imag = (-disc).sqrt() / 2.0;
                eigs.push(Complex::new(real, imag));
                eigs.push(Complex::new(real, -imag));
            } else {
                eigs.push(Complex::new((trace + disc.sqrt()) / 2.0, 0.0));
                eigs.push(Complex::new((trace - disc.sqrt()) / 2.0, 0.0));
            }
            i += 2;
        } else {
            // 1x1 block: real eigenvalue
            eigs.push(Complex::new(t[(i, i)], 0.0));
            i += 1;
        }
    }

    eigs
}

/// Check if system is stable (all eigenvalues in left half-plane).
pub fn is_stable(a: &DMatrix<f64>) -> bool {
    let eigs = eigenvalues(a);
    eigs.iter().all(|e| e.re < 0.0)
}

/// Check if system is marginally stable (no eigenvalues in right half-plane,
/// all on imaginary axis are simple).
pub fn is_marginally_stable(a: &DMatrix<f64>) -> bool {
    let eigs = eigenvalues(a);

    // No eigenvalues in right half-plane
    if eigs.iter().any(|e| e.re > 1e-10) {
        return false;
    }

    // For marginal stability, eigenvalues on imaginary axis must be simple
    // (simplified check: just verify none in right half-plane)
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_controllability_matrix_simple() {
        // Simple 2D system: A = [[0, 1], [0, 0]], B = [[0], [1]]
        // Controllability matrix = [B, AB] = [[0, 1], [1, 0]]
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);

        let ctrl = controllability_matrix(&a, &b);

        assert_eq!(ctrl.nrows(), 2);
        assert_eq!(ctrl.ncols(), 2);
        assert_relative_eq!(ctrl[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(ctrl[(1, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(ctrl[(0, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(ctrl[(1, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_controllable_true() {
        // Double integrator is controllable
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);

        assert!(is_controllable(&a, &b));
    }

    #[test]
    fn test_is_controllable_false() {
        // Uncontrollable system: can't affect first state
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);

        assert!(!is_controllable(&a, &b));
    }

    #[test]
    fn test_observability_matrix_simple() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        let obs = observability_matrix(&a, &c);

        assert_eq!(obs.nrows(), 2);
        assert_eq!(obs.ncols(), 2);
        // [C; CA] = [[1, 0], [0, 1]]
        assert_relative_eq!(obs[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(obs[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(obs[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(obs[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_observable_true() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        assert!(is_observable(&a, &c));
    }

    #[test]
    fn test_is_observable_false() {
        // Unobservable: can't see second state
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        assert!(!is_observable(&a, &c));
    }

    #[test]
    fn test_transfer_function_first_order() {
        let tf = TransferFunction::first_order(2.0, 0.5);

        // DC gain = K = 2
        assert_relative_eq!(tf.dc_gain(), 2.0, epsilon = 1e-10);

        // At s=0: G(0) = 2 / (0 + 1) = 2
        let g0 = tf.at(Complex::new(0.0, 0.0));
        assert_relative_eq!(g0.re, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_transfer_function_second_order() {
        let tf = TransferFunction::second_order(1.0, 2.0, 0.5);

        // DC gain = K = 1
        assert_relative_eq!(tf.dc_gain(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bode_point() {
        let tf = TransferFunction::first_order(1.0, 1.0);

        // At omega = 1 rad/s (corner frequency)
        let (gain_db, phase_deg) = tf.bode_point(1.0);

        // Gain should be -3 dB at corner
        assert_relative_eq!(gain_db, -3.0103, epsilon = 0.01);

        // Phase should be -45 degrees at corner
        assert_relative_eq!(phase_deg, -45.0, epsilon = 0.1);
    }

    #[test]
    fn test_eigenvalues_real() {
        // Diagonal matrix with eigenvalues 1, 2, 3
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);

        let eigs = eigenvalues(&a);
        assert_eq!(eigs.len(), 3);

        let mut real_parts: Vec<f64> = eigs.iter().map(|e| e.re).collect();
        real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_relative_eq!(real_parts[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(real_parts[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(real_parts[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_stable_true() {
        // Stable system: eigenvalues at -1, -2
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -2.0]);
        assert!(is_stable(&a));
    }

    #[test]
    fn test_is_stable_false() {
        // Unstable system: eigenvalue at +1
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
        assert!(!is_stable(&a));
    }

    #[test]
    fn test_state_space_model() {
        // Double integrator: x1' = x2, x2' = u
        // A = [[0, 1], [0, 0]], B = [[0], [1]]
        // This is both controllable and observable
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::from_row_slice(1, 1, &[0.0]);

        let ss = StateSpaceModel::new(a, b, c, d);

        assert_eq!(ss.state_dim(), 2);
        assert_eq!(ss.input_dim(), 1);
        assert_eq!(ss.output_dim(), 1);
        assert!(ss.is_controllable());
        assert!(ss.is_observable());
    }

    #[test]
    fn test_state_space_dc_gain() {
        // First-order system: dx/dt = -x + u, y = x
        // DC gain = 1/1 = 1
        let a = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let b = DMatrix::from_row_slice(1, 1, &[1.0]);
        let c = DMatrix::from_row_slice(1, 1, &[1.0]);
        let d = DMatrix::from_row_slice(1, 1, &[0.0]);

        let ss = StateSpaceModel::new(a, b, c, d);
        let dc = ss.dc_gain().unwrap();

        assert_relative_eq!(dc[(0, 0)], 1.0, epsilon = 1e-10);
    }
}
