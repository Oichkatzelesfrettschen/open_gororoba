//! Full fixed-rank worldline matrices for a pure magnetic background.
//!
//! The matrices follow Appendix B of Ahmadiniaz et al., arXiv:2601.23279.
//! The scalar coefficient functions are kept private to this module so that
//! the tensor path cannot silently inherit a projected channel convention.
//! Contact terms are retained as explicit coefficients of delta(0).

use num_complex::Complex64;

use super::tensor_types::{ComplexLorentzMatrix, LORENTZ_DIMENSION};

const SMALL_FIELD_THRESHOLD: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PureMagneticWorldline {
    pub g_b: ComplexLorentzMatrix,
    pub dot_g_b: ComplexLorentzMatrix,
    pub ddot_g_b: ComplexLorentzMatrix,
    pub g_f: ComplexLorentzMatrix,
    pub dot_g_f: ComplexLorentzMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistributionalMatrix {
    pub regular: ComplexLorentzMatrix,
    pub delta_zero_coefficient: ComplexLorentzMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoincidenceLimits {
    pub g_b: ComplexLorentzMatrix,
    pub dot_g_b: ComplexLorentzMatrix,
    pub ddot_g_b: DistributionalMatrix,
    pub g_f: ComplexLorentzMatrix,
    pub dot_g_f: DistributionalMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldlineInputError {
    InvalidProperTime,
    InvalidModulus,
    NonFiniteInput,
}

pub fn identity() -> ComplexLorentzMatrix {
    ComplexLorentzMatrix::identity()
}

pub fn g_plus() -> ComplexLorentzMatrix {
    diagonal([1.0, 1.0, 0.0, 0.0])
}

pub fn g_minus() -> ComplexLorentzMatrix {
    diagonal([0.0, 0.0, 1.0, 1.0])
}

pub fn r_plus() -> ComplexLorentzMatrix {
    let mut matrix = ComplexLorentzMatrix::zeros();
    matrix[(0, 1)] = real(1.0);
    matrix[(1, 0)] = real(-1.0);
    matrix
}

pub fn scalar_determinant_factor(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    if z.abs() < SMALL_FIELD_THRESHOLD {
        1.0 - z * z / 6.0 + z.powi(4) / 120.0
    } else {
        z / z.sinh()
    }
}

pub fn spinor_determinant_factor(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    if z.abs() < SMALL_FIELD_THRESHOLD {
        1.0 + z * z / 3.0 - z.powi(4) / 45.0
    } else {
        z / z.tanh()
    }
}

pub fn anti_periodic_determinant_factor(z: f64) -> f64 {
    if z.abs() < SMALL_FIELD_THRESHOLD {
        1.0 + z * z / 2.0 + z.powi(4) / 24.0
    } else {
        z.cosh()
    }
}

/// Construct the noncoincident periodic and anti-periodic matrices.
pub fn pure_magnetic_worldline(
    z: f64,
    u: f64,
    proper_time: f64,
) -> Result<PureMagneticWorldline, WorldlineInputError> {
    validate_inputs(z, u, proper_time)?;
    let v = 1.0 - 2.0 * u;
    let g_b_free = proper_time * u * (1.0 - u);
    let dot_g_b_free = v;
    let ddot_g_b_free = -2.0 / proper_time;
    let gf_sign = sign_of_worldline_difference(u);
    let projectors = g_plus();
    let complement = g_minus();
    let field_plane_rotation = imaginary_matrix(&r_plus());

    let (g_b, dot_g_b, ddot_g_b) = if z.abs() < SMALL_FIELD_THRESHOLD {
        (
            scale(&identity(), g_b_free),
            scale(&identity(), dot_g_b_free),
            scale(&identity(), ddot_g_b_free),
        )
    } else {
        let s_b = bosonic_s(z, u);
        let a_b = bosonic_a(z, u);
        let g_b = add(
            &add(
                &scale(&complement, g_b_free),
                &scale(&projectors, -proper_time * a_b / (2.0 * z)),
            ),
            &scale(&field_plane_rotation, proper_time * (s_b - v) / (2.0 * z)),
        );
        let dot_g_b = add(
            &add(&scale(&complement, dot_g_b_free), &scale(&projectors, s_b)),
            &scale(&field_plane_rotation, -a_b),
        );
        let ddot_g_b = add(
            &add(
                &scale(&identity(), ddot_g_b_free),
                &scale(&projectors, -2.0 * z * a_b / proper_time),
            ),
            &scale(
                &field_plane_rotation,
                2.0 * z * bosonic_s(z, u) / proper_time,
            ),
        );
        (g_b, dot_g_b, ddot_g_b)
    };

    let (g_f, dot_g_f) = if z.abs() < SMALL_FIELD_THRESHOLD {
        (scale(&identity(), gf_sign), ComplexLorentzMatrix::zeros())
    } else {
        let s_f = fermionic_s(z, u, gf_sign);
        let a_f = fermionic_a(z, u, gf_sign);
        let g_f = add(
            &add(&scale(&complement, gf_sign), &scale(&projectors, s_f)),
            &scale(&field_plane_rotation, -a_f),
        );
        let dot_g_f = add(
            &scale(&projectors, -2.0 * z * a_f / proper_time),
            &scale(&field_plane_rotation, 2.0 * z * s_f / proper_time),
        );
        (g_f, dot_g_f)
    };

    Ok(PureMagneticWorldline {
        g_b,
        dot_g_b,
        ddot_g_b,
        g_f,
        dot_g_f,
    })
}

/// Return the source coincidence limits with delta(0) coefficients separate.
pub fn pure_magnetic_coincidence(
    z: f64,
    proper_time: f64,
) -> Result<CoincidenceLimits, WorldlineInputError> {
    if !z.is_finite() || !proper_time.is_finite() || proper_time <= 0.0 {
        return Err(if !z.is_finite() {
            WorldlineInputError::NonFiniteInput
        } else {
            WorldlineInputError::InvalidProperTime
        });
    }

    let projectors = g_plus();
    let field_plane_rotation = imaginary_matrix(&r_plus());
    let contact_coefficient = scale(&identity(), 2.0);
    let (g_b, dot_g_b, ddot_g_b_regular, g_f, dot_g_f_regular) = if z.abs() < SMALL_FIELD_THRESHOLD
    {
        (
            scale(&identity(), -proper_time / 6.0),
            ComplexLorentzMatrix::zeros(),
            scale(&identity(), -2.0 / proper_time),
            ComplexLorentzMatrix::zeros(),
            ComplexLorentzMatrix::zeros(),
        )
    } else {
        let a_b = bosonic_a_at_coincidence(z);
        (
            add(
                &scale(&identity(), -proper_time / 6.0),
                &scale(&projectors, -proper_time * (a_b - z / 3.0) / (2.0 * z)),
            ),
            scale(&field_plane_rotation, -a_b),
            add(
                &add(
                    &scale(&identity(), -2.0 / proper_time),
                    &scale(&projectors, -2.0 * z * a_b / proper_time),
                ),
                &ComplexLorentzMatrix::zeros(),
            ),
            scale(&field_plane_rotation, -z.tanh()),
            scale(&projectors, -2.0 * z * z.tanh() / proper_time),
        )
    };
    Ok(CoincidenceLimits {
        g_b,
        dot_g_b,
        ddot_g_b: DistributionalMatrix {
            regular: ddot_g_b_regular,
            delta_zero_coefficient: contact_coefficient,
        },
        g_f,
        dot_g_f: DistributionalMatrix {
            regular: dot_g_f_regular,
            delta_zero_coefficient: contact_coefficient,
        },
    })
}

fn bosonic_s(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < SMALL_FIELD_THRESHOLD {
        let z2 = z * z;
        let v2 = v * v;
        v * (1.0 + (v2 - 1.0) * z2 / 6.0)
    } else {
        (z * v).sinh() / z.sinh()
    }
}

fn bosonic_a(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < SMALL_FIELD_THRESHOLD {
        z * (v * v - 1.0 / 3.0) / 2.0
    } else {
        (z * v).cosh() / z.sinh() - 1.0 / z
    }
}

fn bosonic_a_at_coincidence(z: f64) -> f64 {
    if z.abs() < SMALL_FIELD_THRESHOLD {
        z / 3.0 - z.powi(3) / 45.0
    } else {
        1.0 / z.tanh() - 1.0 / z
    }
}

fn fermionic_s(z: f64, u: f64, gf_sign: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    gf_sign * (z * v).cosh() / z.cosh()
}

fn fermionic_a(z: f64, u: f64, gf_sign: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    gf_sign * (z * v).sinh() / z.cosh()
}

fn sign_of_worldline_difference(u: f64) -> f64 {
    if u > 0.0 {
        1.0
    } else if u < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn validate_inputs(z: f64, u: f64, proper_time: f64) -> Result<(), WorldlineInputError> {
    if !z.is_finite() || !u.is_finite() || !proper_time.is_finite() {
        return Err(WorldlineInputError::NonFiniteInput);
    }
    if !(0.0..=1.0).contains(&u) {
        return Err(WorldlineInputError::InvalidModulus);
    }
    if proper_time <= 0.0 {
        return Err(WorldlineInputError::InvalidProperTime);
    }
    Ok(())
}

fn diagonal(values: [f64; LORENTZ_DIMENSION]) -> ComplexLorentzMatrix {
    let mut matrix = ComplexLorentzMatrix::zeros();
    for index in 0..LORENTZ_DIMENSION {
        matrix[(index, index)] = real(values[index]);
    }
    matrix
}

fn real(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

fn scale(matrix: &ComplexLorentzMatrix, factor: f64) -> ComplexLorentzMatrix {
    matrix.map(|value| value * real(factor))
}

fn add(left: &ComplexLorentzMatrix, right: &ComplexLorentzMatrix) -> ComplexLorentzMatrix {
    left + right
}

fn imaginary_matrix(matrix: &ComplexLorentzMatrix) -> ComplexLorentzMatrix {
    matrix.map(|value| value * Complex64::new(0.0, 1.0))
}

#[cfg(test)]
fn matrix_difference_norm(left: &ComplexLorentzMatrix, right: &ComplexLorentzMatrix) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (*left - *right).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_matrix_close(
        left: &ComplexLorentzMatrix,
        right: &ComplexLorentzMatrix,
        tolerance: f64,
    ) {
        let difference = matrix_difference_norm(left, right);
        assert!(
            difference <= tolerance,
            "matrix difference {difference} > {tolerance}"
        );
    }

    #[test]
    fn projectors_partition_the_euclidean_space() {
        let plus = g_plus();
        let minus = g_minus();
        assert_matrix_close(&(plus + minus), &identity(), 1e-15);
        assert_matrix_close(&(plus * minus), &ComplexLorentzMatrix::zeros(), 1e-15);
        assert_matrix_close(&(plus * plus), &plus, 1e-15);
        assert_matrix_close(&(minus * minus), &minus, 1e-15);
    }

    #[test]
    fn magnetic_rotation_has_the_source_plane_algebra() {
        let rotation = r_plus();
        assert_matrix_close(&(rotation * rotation), &scale(&g_plus(), -1.0), 1e-15);
        assert_matrix_close(
            &(rotation * g_minus()),
            &ComplexLorentzMatrix::zeros(),
            1e-15,
        );
        assert_matrix_close(
            &(rotation.transpose() + rotation),
            &ComplexLorentzMatrix::zeros(),
            1e-15,
        );
    }

    #[test]
    fn pure_magnetic_matrices_obey_u_symmetries() {
        let forward = pure_magnetic_worldline(0.7, 0.23, 1.4).expect("valid source node");
        let reflected = pure_magnetic_worldline(0.7, 1.0 - 0.23, 1.4).expect("valid source node");
        assert_matrix_close(&forward.g_b, &reflected.g_b.transpose(), 1e-12);
        assert_matrix_close(&forward.dot_g_b, &(-reflected.dot_g_b.transpose()), 1e-12);
        assert_matrix_close(&forward.ddot_g_b, &reflected.ddot_g_b.transpose(), 1e-12);
        assert_matrix_close(&forward.g_f, &reflected.g_f.transpose(), 1e-12);
    }

    #[test]
    fn field_sign_changes_the_antisymmetric_matrix_part() {
        let positive = pure_magnetic_worldline(0.7, 0.23, 1.4).expect("valid source node");
        let negative = pure_magnetic_worldline(-0.7, 0.23, 1.4).expect("valid source node");
        assert_matrix_close(&negative.g_b, &positive.g_b.transpose(), 1e-12);
        assert_matrix_close(&negative.dot_g_b, &positive.dot_g_b.transpose(), 1e-12);
        assert_matrix_close(&negative.ddot_g_b, &positive.ddot_g_b.transpose(), 1e-12);
        assert_matrix_close(&negative.g_f, &positive.g_f.transpose(), 1e-12);
    }

    #[test]
    fn zero_field_limit_is_the_free_worldline() {
        let matrices = pure_magnetic_worldline(0.0, 0.23, 1.4).expect("valid source node");
        let expected_g = scale(&identity(), 1.4 * 0.23 * 0.77);
        let expected_dot = scale(&identity(), 1.0 - 2.0 * 0.23);
        let expected_ddot = scale(&identity(), -2.0 / 1.4);
        assert_matrix_close(&matrices.g_b, &expected_g, 1e-15);
        assert_matrix_close(&matrices.dot_g_b, &expected_dot, 1e-15);
        assert_matrix_close(&matrices.ddot_g_b, &expected_ddot, 1e-15);
        assert_matrix_close(&matrices.g_f, &identity(), 1e-15);
        assert_matrix_close(&matrices.dot_g_f, &ComplexLorentzMatrix::zeros(), 1e-15);
    }

    #[test]
    fn derivatives_match_finite_difference_away_from_contacts() {
        let z = 0.7;
        let u = 0.23;
        let proper_time = 1.4;
        let step = 1.0e-6;
        let left = pure_magnetic_worldline(z, u - step, proper_time)
            .expect("valid left derivative node")
            .g_b;
        let right = pure_magnetic_worldline(z, u + step, proper_time)
            .expect("valid right derivative node")
            .g_b;
        let center = pure_magnetic_worldline(z, u, proper_time).expect("valid center node");
        let numerical = (right - left) * real(1.0 / (2.0 * step * proper_time));
        assert_matrix_close(&numerical, &center.dot_g_b, 1e-8);
    }

    #[test]
    fn coincidence_limits_keep_contact_coefficients_explicit() {
        let limits = pure_magnetic_coincidence(0.7, 1.4).expect("valid coincidence");
        assert_matrix_close(
            &limits.ddot_g_b.delta_zero_coefficient,
            &scale(&identity(), 2.0),
            1e-15,
        );
        assert_matrix_close(
            &limits.dot_g_f.delta_zero_coefficient,
            &scale(&identity(), 2.0),
            1e-15,
        );
        assert_matrix_close(&limits.dot_g_b, &(-limits.dot_g_b.transpose()), 1e-15);
    }

    #[test]
    fn determinant_weak_field_limits_are_unity() {
        assert!((scalar_determinant_factor(0.0) - 1.0).abs() < 1e-15);
        assert!((spinor_determinant_factor(0.0) - 1.0).abs() < 1e-15);
        assert!((anti_periodic_determinant_factor(0.0) - 1.0).abs() < 1e-15);
        assert!((scalar_determinant_factor(1e-4) - 1.0).abs() < 1e-8);
        assert!((spinor_determinant_factor(1e-4) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn invalid_worldline_nodes_are_rejected() {
        assert_eq!(
            pure_magnetic_worldline(0.7, -0.1, 1.4),
            Err(WorldlineInputError::InvalidModulus)
        );
        assert_eq!(
            pure_magnetic_worldline(0.7, 0.2, 0.0),
            Err(WorldlineInputError::InvalidProperTime)
        );
    }
}
