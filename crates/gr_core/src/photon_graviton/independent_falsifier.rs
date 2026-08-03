//! Test-only dense-index source transcription for cross-implementation checks.
//!
//! This module deliberately duplicates the Appendix B scalar formulas and
//! Eq. 4.11 assembly. It does not call the production tensor constructors,
//! contraction methods, J assembly, or projected legacy amplitudes.

use num_complex::Complex64;
use std::{fmt::Write as _, fs};

use super::{
    irreducible_tensor::irreducible_integrand,
    tensor_integrands::TensorLoopConfig,
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, MomentumRule, ShellMode, WardKinematics,
    },
    types::LoopType,
};

const DIMENSION: usize = 4;
const MATRIX_COMPONENTS: usize = DIMENSION * DIMENSION;
const RANK_THREE_COMPONENTS: usize = DIMENSION * DIMENSION * DIMENSION;
type DenseMatrix = [Complex64; MATRIX_COMPONENTS];
type DenseRankThree = [Complex64; RANK_THREE_COMPONENTS];
const FIXTURE_HASH: &str = "ea08b72d402e59c06c9e50bd0e4570f6e5b7d2cc49f5ae76fe7dc92f74e621a8";

#[derive(Debug, Clone, Copy)]
struct DenseWorldline {
    dot_g_b: DenseMatrix,
    ddot_g_b: DenseMatrix,
    g_f: DenseMatrix,
    dot_g_f: DenseMatrix,
    coincidence_dot_g_b: DenseMatrix,
    coincidence_ddot_g_b: DenseMatrix,
    coincidence_g_f: DenseMatrix,
    coincidence_dot_g_f: DenseMatrix,
}

fn zero_matrix() -> DenseMatrix {
    [Complex64::new(0.0, 0.0); MATRIX_COMPONENTS]
}

fn zero_rank_three() -> DenseRankThree {
    [Complex64::new(0.0, 0.0); RANK_THREE_COMPONENTS]
}

fn matrix_index(row: usize, column: usize) -> usize {
    row * DIMENSION + column
}

fn rank_three_index(mu: usize, nu: usize, alpha: usize) -> usize {
    (mu * DIMENSION + nu) * DIMENSION + alpha
}

fn matrix_from_fn(mut function: impl FnMut(usize, usize) -> Complex64) -> DenseMatrix {
    let mut matrix = zero_matrix();
    for row in 0..DIMENSION {
        for column in 0..DIMENSION {
            matrix[matrix_index(row, column)] = function(row, column);
        }
    }
    matrix
}

fn rank_three_from_fn(
    mut function: impl FnMut(usize, usize, usize) -> Complex64,
) -> DenseRankThree {
    let mut tensor = zero_rank_three();
    for mu in 0..DIMENSION {
        for nu in 0..DIMENSION {
            for alpha in 0..DIMENSION {
                tensor[rank_three_index(mu, nu, alpha)] = function(mu, nu, alpha);
            }
        }
    }
    tensor
}

fn matrix_even(matrix: &DenseMatrix) -> DenseMatrix {
    matrix_from_fn(|row, column| {
        (matrix[matrix_index(row, column)] + matrix[matrix_index(column, row)])
            * Complex64::new(0.5, 0.0)
    })
}

fn matrix_odd(matrix: &DenseMatrix) -> DenseMatrix {
    matrix_from_fn(|row, column| {
        (matrix[matrix_index(row, column)] - matrix[matrix_index(column, row)])
            * Complex64::new(0.5, 0.0)
    })
}

fn matrix_add(left: &DenseMatrix, right: &DenseMatrix) -> DenseMatrix {
    matrix_from_fn(|row, column| left[matrix_index(row, column)] + right[matrix_index(row, column)])
}

fn matrix_subtract(left: &DenseMatrix, right: &DenseMatrix) -> DenseMatrix {
    matrix_from_fn(|row, column| left[matrix_index(row, column)] - right[matrix_index(row, column)])
}

fn matrix_scale(matrix: &DenseMatrix, factor: Complex64) -> DenseMatrix {
    matrix_from_fn(|row, column| matrix[matrix_index(row, column)] * factor)
}

fn rank_three_add(left: &DenseRankThree, right: &DenseRankThree) -> DenseRankThree {
    rank_three_from_fn(|mu, nu, alpha| {
        left[rank_three_index(mu, nu, alpha)] + right[rank_three_index(mu, nu, alpha)]
    })
}

fn right_contract(matrix: &DenseMatrix, vector: &[Complex64; DIMENSION]) -> [Complex64; DIMENSION] {
    let mut result = [Complex64::new(0.0, 0.0); DIMENSION];
    for row in 0..DIMENSION {
        for column in 0..DIMENSION {
            result[row] += matrix[matrix_index(row, column)] * vector[column];
        }
    }
    result
}

fn left_contract(vector: &[Complex64; DIMENSION], matrix: &DenseMatrix) -> [Complex64; DIMENSION] {
    let mut result = [Complex64::new(0.0, 0.0); DIMENSION];
    for row in 0..DIMENSION {
        for column in 0..DIMENSION {
            result[column] += vector[row] * matrix[matrix_index(row, column)];
        }
    }
    result
}

fn bilinear(vector: &[Complex64; DIMENSION], matrix: &DenseMatrix) -> Complex64 {
    left_contract(vector, matrix)
        .iter()
        .zip(vector.iter())
        .map(|(left, right)| *left * *right)
        .sum()
}

fn diagonal(values: [f64; DIMENSION]) -> DenseMatrix {
    matrix_from_fn(|row, column| {
        if row == column {
            Complex64::new(values[row], 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

fn g_plus() -> DenseMatrix {
    diagonal([1.0, 1.0, 0.0, 0.0])
}

fn g_minus() -> DenseMatrix {
    diagonal([0.0, 0.0, 1.0, 1.0])
}

fn imaginary_r_plus() -> DenseMatrix {
    matrix_from_fn(|row, column| match (row, column) {
        (0, 1) => Complex64::new(0.0, 1.0),
        (1, 0) => Complex64::new(0.0, -1.0),
        _ => Complex64::new(0.0, 0.0),
    })
}

fn bosonic_s(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1.0e-6 {
        v * (1.0 + (v * v - 1.0) * z * z / 6.0)
    } else {
        (z * v).sinh() / z.sinh()
    }
}

fn bosonic_a(z: f64, u: f64) -> f64 {
    let v = 1.0 - 2.0 * u;
    if z.abs() < 1.0e-6 {
        z * (v * v - 1.0 / 3.0) / 2.0
    } else {
        (z * v).cosh() / z.sinh() - 1.0 / z
    }
}

fn bosonic_a_coincidence(z: f64) -> f64 {
    if z.abs() < 1.0e-6 {
        z / 3.0 - z.powi(3) / 45.0
    } else {
        1.0 / z.tanh() - 1.0 / z
    }
}

fn fermionic_s(z: f64, u: f64) -> f64 {
    (z * (1.0 - 2.0 * u)).cosh() / z.cosh()
}

fn fermionic_a(z: f64, u: f64) -> f64 {
    (z * (1.0 - 2.0 * u)).sinh() / z.cosh()
}

fn pure_magnetic_worldline(z: f64, u: f64, proper_time: f64) -> DenseWorldline {
    let identity = diagonal([1.0, 1.0, 1.0, 1.0]);
    let plus = g_plus();
    let minus = g_minus();
    let imaginary_rotation = imaginary_r_plus();
    let v = 1.0 - 2.0 * u;
    let free_ddot = -2.0 / proper_time;
    let s_b = bosonic_s(z, u);
    let a_b = bosonic_a(z, u);
    let dot_g_b = matrix_add(
        &matrix_add(
            &matrix_scale(&minus, Complex64::new(v, 0.0)),
            &matrix_scale(&plus, Complex64::new(s_b, 0.0)),
        ),
        &matrix_scale(&imaginary_rotation, Complex64::new(-a_b, 0.0)),
    );
    let ddot_g_b = matrix_add(
        &matrix_add(
            &matrix_scale(&identity, Complex64::new(free_ddot, 0.0)),
            &matrix_scale(&plus, Complex64::new(-2.0 * z * a_b / proper_time, 0.0)),
        ),
        &matrix_scale(
            &imaginary_rotation,
            Complex64::new(2.0 * z * s_b / proper_time, 0.0),
        ),
    );
    let s_f = fermionic_s(z, u);
    let a_f = fermionic_a(z, u);
    let g_f = matrix_add(
        &matrix_add(
            &matrix_scale(&minus, Complex64::new(1.0, 0.0)),
            &matrix_scale(&plus, Complex64::new(s_f, 0.0)),
        ),
        &matrix_scale(&imaginary_rotation, Complex64::new(-a_f, 0.0)),
    );
    let dot_g_f = matrix_add(
        &matrix_scale(&plus, Complex64::new(-2.0 * z * a_f / proper_time, 0.0)),
        &matrix_scale(
            &imaginary_rotation,
            Complex64::new(2.0 * z * s_f / proper_time, 0.0),
        ),
    );
    let a_coinc = bosonic_a_coincidence(z);
    let coincidence_dot_g_b = matrix_scale(&imaginary_rotation, Complex64::new(-a_coinc, 0.0));
    let coincidence_ddot_g_b = matrix_add(
        &matrix_scale(&identity, Complex64::new(-2.0 / proper_time, 0.0)),
        &matrix_scale(&plus, Complex64::new(-2.0 * z * a_coinc / proper_time, 0.0)),
    );
    let coincidence_g_f = matrix_scale(&imaginary_rotation, Complex64::new(-z.tanh(), 0.0));
    let coincidence_dot_g_f = matrix_scale(
        &plus,
        Complex64::new(-2.0 * z * z.tanh() / proper_time, 0.0),
    );
    DenseWorldline {
        dot_g_b,
        ddot_g_b,
        g_f,
        dot_g_f,
        coincidence_dot_g_b,
        coincidence_ddot_g_b,
        coincidence_g_f,
        coincidence_dot_g_f,
    }
}

fn symmetrize(tensor: &DenseRankThree) -> DenseRankThree {
    rank_three_from_fn(|mu, nu, alpha| {
        (tensor[rank_three_index(mu, nu, alpha)] + tensor[rank_three_index(nu, mu, alpha)])
            * Complex64::new(0.5, 0.0)
    })
}

fn source_integrand(
    k: &[Complex64; DIMENSION],
    z: f64,
    proper_time: f64,
    u: f64,
    loop_type: LoopType,
) -> DenseRankThree {
    let node = pure_magnetic_worldline(z, u, proper_time);
    let zero = zero_matrix();
    let ddot_s_b11 = matrix_even(&node.coincidence_ddot_g_b);
    let ddot_s_b12 = matrix_even(&node.ddot_g_b);
    let ddot_a_b12 = matrix_odd(&node.ddot_g_b);
    let dot_s_b12 = matrix_even(&node.dot_g_b);
    let dot_a_b12 = matrix_subtract(
        &matrix_odd(&node.dot_g_b),
        &matrix_odd(&node.coincidence_dot_g_b),
    );
    let include_fermion = matches!(loop_type, LoopType::Spinor);
    let dot_s_f11 = if include_fermion {
        matrix_even(&node.coincidence_dot_g_f)
    } else {
        zero
    };
    let a_f11 = if include_fermion {
        matrix_odd(&node.coincidence_g_f)
    } else {
        zero
    };
    let s_f12 = if include_fermion {
        matrix_even(&node.g_f)
    } else {
        zero
    };
    let a_f12 = if include_fermion {
        matrix_odd(&node.g_f)
    } else {
        zero
    };
    let dot_s_f12 = if include_fermion {
        matrix_even(&node.dot_g_f)
    } else {
        zero
    };
    let dot_a_f12 = if include_fermion {
        matrix_odd(&node.dot_g_f)
    } else {
        zero
    };
    let bar_dot_a_plus_a_f11 = matrix_add(&dot_a_b12, &a_f11);
    let k_dot_bar_dot_a_plus_a_f11 = left_contract(k, &bar_dot_a_plus_a_f11);
    let bar_dot_a_plus_a_f11_k = right_contract(&bar_dot_a_plus_a_f11, k);
    let dot_s_b_k = right_contract(&dot_s_b12, k);
    let k_dot_dot_s_b = left_contract(k, &dot_s_b12);
    let s_f_k = right_contract(&s_f12, k);
    let k_dot_s_f = left_contract(k, &s_f12);
    let bar_dot_a_k = right_contract(&dot_a_b12, k);
    let a_f_k = right_contract(&a_f12, k);
    let ddot_s_b_k = right_contract(&ddot_s_b12, k);
    let ddot_a_b_k = right_contract(&ddot_a_b12, k);
    let dot_a_f_k = right_contract(&dot_a_f12, k);
    let dot_s_f_k = right_contract(&dot_s_f12, k);
    let dot_s_b_bilinear = bilinear(k, &dot_s_b12);
    let s_f_bilinear = bilinear(k, &s_f12);
    let k_dot_a_f = left_contract(k, &a_f12);

    let j1 = symmetrize(&rank_three_from_fn(|mu, nu, alpha| {
        (ddot_s_b11[matrix_index(mu, nu)] - dot_s_f11[matrix_index(mu, nu)])
            * k_dot_bar_dot_a_plus_a_f11[alpha]
    }));
    let j2 = symmetrize(&rank_three_from_fn(|mu, nu, alpha| {
        -dot_s_b12[matrix_index(mu, alpha)] * ddot_a_b_k[nu]
            + s_f12[matrix_index(mu, alpha)] * dot_a_f_k[nu]
            + ddot_a_b12[matrix_index(nu, alpha)] * dot_s_b_k[mu]
            - dot_a_f12[matrix_index(nu, alpha)] * s_f_k[mu]
            - dot_a_b12[matrix_index(mu, alpha)] * ddot_s_b_k[nu]
            + a_f12[matrix_index(mu, alpha)] * dot_s_f_k[nu]
            + ddot_s_b12[matrix_index(nu, alpha)] * bar_dot_a_plus_a_f11_k[mu]
            - dot_s_f12[matrix_index(nu, alpha)] * a_f_k[mu]
    }));
    let j3 = symmetrize(&rank_three_from_fn(|mu, nu, alpha| {
        let first_bracket = dot_s_b_k[nu] * k_dot_bar_dot_a_plus_a_f11[alpha]
            - s_f_k[nu] * k_dot_a_f[alpha]
            + bar_dot_a_plus_a_f11_k[nu] * k_dot_dot_s_b[alpha]
            - a_f_k[nu] * k_dot_s_f[alpha]
            - dot_a_b12[matrix_index(nu, alpha)] * dot_s_b_bilinear
            + a_f12[matrix_index(nu, alpha)] * s_f_bilinear;
        let second_bracket = dot_s_b_k[nu] * k_dot_dot_s_b[alpha] - s_f_k[nu] * k_dot_s_f[alpha]
            + bar_dot_a_plus_a_f11_k[nu] * k_dot_bar_dot_a_plus_a_f11[alpha]
            - a_f_k[nu] * k_dot_a_f[alpha]
            - dot_s_b12[matrix_index(nu, alpha)] * dot_s_b_bilinear
            + s_f12[matrix_index(nu, alpha)] * s_f_bilinear;
        -dot_s_b_k[mu] * first_bracket - bar_dot_a_k[mu] * second_bracket
    }));
    rank_three_add(&rank_three_add(&j1, &j2), &j3)
}

fn fixture() -> WardKinematics {
    let mut field = ComplexLorentzMatrix::zeros();
    field[(0, 1)] = Complex64::new(0.1, 0.0);
    field[(1, 0)] = Complex64::new(-0.1, 0.0);
    let k = ComplexFourVector::from([
        Complex64::new(0.15, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.2, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    WardKinematics::new(
        k,
        -k,
        ComplexFourVector::zeros(),
        ComplexLorentzMatrix::identity(),
        ComplexFourVector::zeros(),
        field,
        Complex64::new(0.0625, 0.0),
        ShellMode::OffShell,
        MomentumRule::ConstantBackgroundConversion,
        false,
        1.0e-12,
    )
    .expect("valid independent fixture")
}

fn complex_text(value: Complex64) -> String {
    format!("{:.17e}{:+.17e}i", value.re, value.im)
}

#[test]
fn dense_source_reference_agrees_component_by_component() {
    let kinematics = fixture();
    let output_path = std::env::var("P1_INDEPENDENT_FIXTURES_OUTPUT").ok();
    let mut retained_output = String::new();
    if output_path.is_some() {
        writeln!(
            retained_output,
            "# Dense-index reference component fixtures."
        )
        .expect("write fixture header");
        writeln!(retained_output, "[meta]").expect("write fixture metadata");
        writeln!(
            retained_output,
            "artifact_id = \"p1-photon-graviton-independent-component-fixtures\""
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "source_equations = [\"B.13-B.33\", \"B.43-B.57\", \"4.9-4.11\"]"
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "generator_source = \"crates/gr_core/src/photon_graviton/independent_falsifier.rs\""
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "generator_version = \"Rust 1.97.0; dense fixed-array implementation\""
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "generation_command = \"RUSTC_WRAPPER= CARGO_TARGET_DIR=.cache/exp-p1-target cargo test -p gr_core --lib photon_graviton::independent_falsifier::dense_source_reference_agrees_component_by_component -- --nocapture\""
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "component_order = \"row-major ((mu * 4 + nu) * 4 + alpha)\""
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "comparison = \"production node versus independently assembled dense reference, component by component\""
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "agreement_classification = \"cross-implementation survival, not external replication\""
        )
        .expect("write fixture metadata");
        writeln!(
            retained_output,
            "fixture_input = \"D=4;B/Bcr=0.1;t=0.8;u=0.27;k=(0.15,0,0.2,0);loop=scalar+spinor;equations=B.13-B.33,B.43-B.57,4.9-4.11\""
        )
        .expect("write fixture metadata");
        writeln!(retained_output, "fixture_hash = \"{FIXTURE_HASH}\"")
            .expect("write fixture metadata");
    }
    for loop_type in [LoopType::Scalar, LoopType::Spinor] {
        let proper_time = 0.8;
        let u = 0.27;
        let production = irreducible_integrand(
            &kinematics,
            loop_type,
            proper_time,
            u,
            TensorLoopConfig::unit_natural(),
            super::irreducible_tensor::IrreducibleMutation::None,
        )
        .expect("production source node")
        .total;
        let reference = source_integrand(
            &[
                kinematics.k[0],
                kinematics.k[1],
                kinematics.k[2],
                kinematics.k[3],
            ],
            0.1 * proper_time,
            proper_time,
            u,
            loop_type,
        );
        let difference = production
            .components()
            .iter()
            .zip(reference.iter())
            .map(|(left, right)| (*left - *right).norm_sqr())
            .sum::<f64>()
            .sqrt();
        println!(
            "independent loop={loop_type:?} max_l2_difference={difference:.16e} reference_components={reference:?}"
        );
        if output_path.is_some() {
            writeln!(retained_output, "\n[[fixture]]").expect("write fixture record");
            writeln!(retained_output, "loop_type = \"{loop_type:?}\"").expect("write loop type");
            writeln!(retained_output, "max_l2_difference = {:.17e}", difference)
                .expect("write difference");
            writeln!(retained_output, "expected_components = [").expect("write components");
            for (index, component) in reference.iter().enumerate() {
                let separator = if index + 1 == reference.len() {
                    ""
                } else {
                    ","
                };
                writeln!(
                    retained_output,
                    "  \"{}\"{separator}",
                    complex_text(*component)
                )
                .expect("write component");
            }
            writeln!(retained_output, "]").expect("close components");
        }
        assert!(
            difference < 1.0e-12,
            "independent reference disagrees for {loop_type:?}: {difference}"
        );
    }
    if let Some(output_path) = output_path {
        fs::write(&output_path, retained_output).expect("retain independent fixtures");
        println!("p1_independent_fixtures_output={output_path}");
    }
}
