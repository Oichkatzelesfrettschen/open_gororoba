//! Fixed-proper-time source oracle with periodic midpoint contact products.

use gauss_quad::GaussLegendre;
use gr_core::photon_graviton::{
    irreducible_tensor::{
        IrreducibleMutation, irreducible_contact_integrand, irreducible_integrand,
    },
    tensor_integrands::TensorLoopConfig,
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, ComplexRankThreeTensor, MomentumRule, ShellMode,
        WardKinematics,
    },
    types::LoopType,
};
use num_complex::Complex64;
use std::num::NonZeroUsize;

type Matrix = ComplexLorentzMatrix;
type Vector = ComplexFourVector;
type Tensor = ComplexRankThreeTensor;

struct SourceNode {
    bar_g: Matrix,
    bar_dot_g: Matrix,
    ddot_g: Matrix,
    ddot_coincidence: Matrix,
    fermion: Matrix,
    dot_fermion: Matrix,
    fermion_coincidence: Matrix,
    dot_fermion_coincidence: Matrix,
}

fn source_node(time: f64, modulus: f64, field: f64) -> SourceNode {
    let field_time = field * time;
    let velocity = 1.0 - 2.0 * modulus;
    let boson_even = (field_time * velocity).sinh() / field_time.sinh();
    let boson_odd = (field_time * velocity).cosh() / field_time.sinh() - 1.0 / field_time;
    let coincidence_odd = 1.0 / field_time.tanh() - 1.0 / field_time;
    let fermion_even = (field_time * velocity).cosh() / field_time.cosh();
    let fermion_odd = (field_time * velocity).sinh() / field_time.cosh();
    let plane = Matrix::from_diagonal(&Vector::from([1.0, 1.0, 0.0, 0.0].map(Complex64::from)));
    let complement = Matrix::identity() - plane;
    let mut rotation = Matrix::zeros();
    rotation[(0, 1)] = Complex64::new(0.0, 1.0);
    rotation[(1, 0)] = Complex64::new(0.0, -1.0);
    let scale = |matrix: Matrix, value: f64| matrix * Complex64::from(value);
    // Appendix B.2 and B.6 use the same additive-constant convention.
    let bar_g = scale(complement, time * modulus * (1.0 - modulus))
        + scale(
            plane,
            -time * (boson_odd - coincidence_odd) / (2.0 * field_time),
        )
        + scale(
            rotation,
            time * (boson_even - velocity) / (2.0 * field_time),
        );
    SourceNode {
        bar_g,
        bar_dot_g: scale(complement, velocity) + scale(plane, boson_even)
            - scale(rotation, boson_odd - coincidence_odd),
        ddot_g: scale(complement, -2.0 / time)
            + scale(
                plane,
                -2.0 * field_time * (field_time * velocity).cosh() / (time * field_time.sinh()),
            )
            + scale(rotation, 2.0 * field_time * boson_even / time),
        ddot_coincidence: scale(complement, -2.0 / time)
            + scale(plane, -2.0 * field_time / (time * field_time.tanh())),
        fermion: complement + scale(plane, fermion_even) - scale(rotation, fermion_odd),
        dot_fermion: scale(plane, -2.0 * field_time * fermion_odd / time)
            + scale(rotation, 2.0 * field_time * fermion_even / time),
        fermion_coincidence: -scale(rotation, field_time.tanh()),
        dot_fermion_coincidence: scale(plane, -2.0 * field_time * field_time.tanh() / time),
    }
}

fn unreduced(node: &SourceNode, momentum: &Vector, spinor: bool) -> Tensor {
    let right = node.bar_dot_g * momentum;
    let left = node.bar_dot_g.transpose() * momentum;
    let fermion_right = node.fermion * momentum;
    let fermion_left = node.fermion.transpose() * momentum;
    let coincidence_right = node.fermion_coincidence * momentum;
    let derivative_right = node.dot_fermion * momentum;
    let fermion_bilinear: Complex64 = momentum
        .iter()
        .zip(fermion_right.iter())
        .map(|(left, right)| left * right)
        .sum();
    let orbital = |row, column, photon| {
        node.ddot_coincidence[(row, column)] * left[photon]
            + node.ddot_g[(row, photon)] * right[column]
            + node.ddot_g[(column, photon)] * right[row]
            - right[row] * right[column] * left[photon]
    };
    // Eq. 4.4 is symmetrized only after all seven source groups are assembled.
    let fermionic = |row, column, photon| {
        (node.dot_fermion_coincidence[(row, column)] + coincidence_right[column] * right[row])
            * coincidence_right[photon]
            - fermion_right[row] * node.dot_fermion[(column, photon)]
            + node.fermion[(row, photon)] * derivative_right[column]
            + (-node.fermion[(column, photon)] * fermion_bilinear
                + fermion_right[column] * fermion_left[photon])
                * right[row]
            - node.dot_fermion_coincidence[(row, column)] * left[photon]
            - (node.ddot_coincidence[(row, column)] - right[row] * right[column])
                * coincidence_right[photon]
            + (node.ddot_g[(row, photon)] - right[row] * left[photon]) * coincidence_right[column]
    };
    Tensor::from_fn(|row, column, photon| {
        orbital(row, column, photon)
            + if spinor {
                (fermionic(row, column, photon) + fermionic(column, row, photon)) * 0.5
            } else {
                Complex64::default()
            }
    })
}

fn contact(time: f64, field: f64, momentum: &Vector, spinor: bool) -> Tensor {
    contact_with_weights(time, field, momentum, spinor, 1.0, 1.0)
}

fn contact_with_weights(
    time: f64,
    field: f64,
    momentum: &Vector,
    spinor: bool,
    boson_weight: f64,
    fermion_weight: f64,
) -> Tensor {
    // Products at delta12 use the midpoint value, with total periodic weight1.
    // Ghost-subtracted coincidence derivatives stay regular; only B.4/B.36
    // noncoincident derivatives receive the delta12 coefficient2I.
    let mut midpoint = source_node(time, 0.0, field);
    midpoint.bar_g = Matrix::zeros();
    midpoint.bar_dot_g = Matrix::zeros();
    midpoint.fermion = midpoint.fermion_coincidence;
    let without = unreduced(&midpoint, momentum, spinor);
    midpoint.ddot_g += Matrix::identity() * Complex64::from(2.0 * boson_weight);
    midpoint.dot_fermion += Matrix::identity() * Complex64::from(2.0 * fermion_weight);
    let with = unreduced(&midpoint, momentum, spinor);
    Tensor::from_fn(|row, column, photon| {
        (with.get(row, column, photon) - without.get(row, column, photon)) / time
    })
}

fn fixture() -> WardKinematics {
    let momentum = Vector::from([0.15, -0.11, 0.2, 0.07].map(Complex64::from));
    let mut field = Matrix::zeros();
    field[(0, 1)] = Complex64::from(0.1);
    field[(1, 0)] = Complex64::from(-0.1);
    WardKinematics::new(
        momentum,
        -momentum,
        Vector::from([0.11, 0.15, 0.0, 0.0].map(Complex64::from)),
        Matrix::identity(),
        Vector::from([0.2, 0.1, -0.3, 0.4].map(Complex64::from)),
        field,
        momentum.iter().map(|value| value * value).sum(),
        ShellMode::OffShell,
        MomentumRule::ConstantBackgroundConversion,
        false,
        1e-12,
    )
    .unwrap()
}

fn norm(tensor: &Tensor) -> f64 {
    tensor
        .components()
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}
fn difference(left: &Tensor, right: &Tensor) -> Tensor {
    Tensor::from_fn(|row, column, photon| {
        left.get(row, column, photon) - right.get(row, column, photon)
    })
}
fn gauge_norm(tensor: &Tensor, momentum: &Vector) -> f64 {
    (0..16)
        .map(|index| {
            (0..4)
                .map(|photon| tensor.get(index / 4, index % 4, photon) * momentum[photon])
                .sum::<Complex64>()
                .norm_sqr()
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn scalar_source_momentum_mapping_preserves_orbital_terms_and_counterterm() {
    let kinematics = fixture();
    let new_momentum = kinematics.k;
    let old_momentum = -new_momentum;
    for time in [0.07, 0.3, 1.2] {
        for modulus in [0.17, 0.39] {
            let node = source_node(time, modulus, 0.1);
            let old_right = node.bar_dot_g * old_momentum;
            let old_left = node.bar_dot_g.transpose() * old_momentum;
            // gr-qc/0412095 Eq. 3.5, xi_bar=0, k=k_graviton.
            let old = Tensor::from_fn(|row, column, photon| {
                -node.ddot_coincidence[(row, column)] * old_left[photon]
                    - node.ddot_g[(row, photon)] * old_right[column]
                    - node.ddot_g[(column, photon)] * old_right[row]
                    + old_right[row] * old_right[column] * old_left[photon]
            });
            let new = unreduced(&node, &new_momentum, false);
            let error = norm(&difference(&old, &new));
            let wrong_momentum = unreduced(&node, &old_momentum, false);
            println!(
                "scalar_mapping time={time} u={modulus} orbital_error={error:.17e} omit_momentum_reversal={:.17e}",
                norm(&difference(&old, &wrong_momentum))
            );
            assert!(error < 1e-12);
            assert!(norm(&difference(&old, &wrong_momentum)) > 0.01);
        }
    }
    let tree = |momentum: &Vector| {
        let field_momentum = kinematics.field_strength * momentum;
        Tensor::from_fn(|row, column, photon| {
            kinematics.field_strength[(row, photon)] * momentum[column]
                + kinematics.field_strength[(column, photon)] * momentum[row]
                - field_momentum[row] * f64::from(column == photon)
                - field_momentum[column] * f64::from(row == photon)
                + field_momentum[photon] * f64::from(row == column)
        })
    };
    let old_tree = tree(&old_momentum);
    let new_tree = tree(&new_momentum);
    // Eq. 3.11 has +2i e C_old/3; equal loop prefactors leave -2i e C_new/3.
    let mapped_error = Tensor::from_fn(|row, column, photon| {
        (old_tree.get(row, column, photon) + new_tree.get(row, column, photon))
            * Complex64::new(0.0, 2.0 / 3.0)
    });
    assert!(norm(&mapped_error) < 1e-15);
    assert!(norm(&new_tree) > 0.01);
}

#[test]
fn unreduced_source_contact_and_production_are_separate() {
    let kinematics = fixture();
    let nodes = GaussLegendre::new(NonZeroUsize::new(96).unwrap());
    let mut production_defects = Vec::new();
    for time in [0.07, 0.3, 1.2] {
        for loop_type in [LoopType::Scalar, LoopType::Spinor] {
            let spinor = loop_type == LoopType::Spinor;
            let mut source_regular = Tensor::zero();
            let mut production_regular = Tensor::zero();
            let mut production_source_exponent = Tensor::zero();
            for &(abscissa, weight) in nodes.as_node_weight_pairs() {
                let modulus = (abscissa + 1.0) * 0.5;
                let node = source_node(time, modulus, 0.1);
                let exponent = (-(kinematics.k.transpose() * node.bar_g * kinematics.k)[0]).exp();
                let raw = unreduced(&node, &kinematics.k, spinor);
                let production = irreducible_integrand(
                    &kinematics,
                    loop_type,
                    time,
                    modulus,
                    TensorLoopConfig::unit_natural(),
                    IrreducibleMutation::None,
                )
                .unwrap();
                for row in 0..4 {
                    for column in 0..4 {
                        for photon in 0..4 {
                            source_regular.set(
                                row,
                                column,
                                photon,
                                source_regular.get(row, column, photon)
                                    + raw.get(row, column, photon) * exponent * (weight * 0.5),
                            );
                            production_regular.set(
                                row,
                                column,
                                photon,
                                production_regular.get(row, column, photon)
                                    + production.total.get(row, column, photon)
                                        * production.exponent
                                        * (weight * 0.5),
                            );
                            production_source_exponent.set(
                                row,
                                column,
                                photon,
                                production_source_exponent.get(row, column, photon)
                                    + production.total.get(row, column, photon)
                                        * exponent
                                        * (weight * 0.5),
                            );
                        }
                    }
                }
            }
            let contact = contact(time, 0.1, &kinematics.k, spinor);
            let production_contact = irreducible_contact_integrand(
                &kinematics,
                loop_type,
                time,
                TensorLoopConfig::unit_natural(),
                IrreducibleMutation::None,
            )
            .unwrap();
            assert!(norm(&difference(&contact, &production_contact)) < 1e-12);
            let production_complete = Tensor::from_fn(|row, column, photon| {
                production_regular.get(row, column, photon)
                    + production_contact.get(row, column, photon)
            });
            if spinor {
                for (label, boson_weight, fermion_weight) in
                    [("omit_B4", 0.0, 1.0), ("omit_B36", 1.0, 0.0)]
                {
                    let omitted = contact_with_weights(
                        time,
                        0.1,
                        &kinematics.k,
                        spinor,
                        boson_weight,
                        fermion_weight,
                    );
                    let tensor_delta = norm(&difference(&contact, &omitted));
                    let changed = Tensor::from_fn(|row, column, photon| {
                        source_regular.get(row, column, photon) + omitted.get(row, column, photon)
                    });
                    println!(
                        "contact_omission time={time} mode={label} tensor_delta={tensor_delta:.17e} gauge={:.17e}",
                        gauge_norm(&changed, &kinematics.k)
                    );
                    assert!(tensor_delta > 1e-4);
                }
            }
            let complete = Tensor::from_fn(|row, column, photon| {
                source_regular.get(row, column, photon) + contact.get(row, column, photon)
            });
            let raw_tilde_error = norm(&difference(&source_regular, &production_source_exponent));
            println!(
                "fixed_time time={time} loop={loop_type:?} raw_tilde_error={raw_tilde_error:.17e} contact_norm={:.17e} regular_gauge={:.17e} complete_gauge={:.17e} production_regular_difference={:.17e}",
                norm(&contact),
                gauge_norm(&source_regular, &kinematics.k),
                gauge_norm(&complete, &kinematics.k),
                norm(&difference(&complete, &production_regular))
            );
            assert!(raw_tilde_error < 1e-10);
            assert!(gauge_norm(&complete, &kinematics.k) < 1e-11);
            if spinor {
                assert!(gauge_norm(&source_regular, &kinematics.k) > 1e-4);
            }
            production_defects.push(norm(&difference(&complete, &production_complete)));
            println!(
                "production_complete time={time} loop={loop_type:?} source_error={:.17e}",
                production_defects.last().unwrap()
            );
        }
    }
    assert!(
        production_defects.iter().all(|defect| *defect < 1e-10),
        "production must retain source exponent and distribution"
    );
}

#[test]
fn signed_total_derivative_endpoints_cancel_without_dropping_each_endpoint() {
    let momentum = fixture().k;
    for time in [0.07, 0.3, 1.2] {
        let primitive = |modulus| {
            let node = source_node(time, modulus, 0.1);
            let right = node.bar_dot_g * momentum;
            let exponent = (-(momentum.transpose() * node.bar_g * momentum)[0]).exp();
            Tensor::from_fn(|row, column, photon| {
                -0.5 * (node.bar_dot_g[(row, photon)] * right[column]
                    + node.bar_dot_g[(column, photon)] * right[row])
                    * exponent
                    / time
            })
        };
        let lower = primitive(0.0);
        let upper = primitive(1.0);
        let jump = difference(&upper, &lower);
        println!(
            "endpoint time={time} signed_jump={:.17e} omit_lower={:.17e} omit_upper={:.17e}",
            norm(&jump),
            norm(&upper),
            norm(&lower)
        );
        assert!(norm(&jump) < 1e-12);
        assert!(norm(&upper) > 0.1 && norm(&lower) > 0.1);
    }
}

#[test]
fn source_uv_coefficients_survive_the_small_field_branch() {
    let kinematics = fixture();
    let nodes = GaussLegendre::new(NonZeroUsize::new(96).unwrap());
    let tree = kinematics.field_strength[(0, 1)] * kinematics.k[0];
    for loop_type in [LoopType::Scalar, LoopType::Spinor] {
        for time in [0.01001, 0.00999, 1e-4, 1e-6, 1e-8] {
            let mut regular = Complex64::default();
            for &(abscissa, weight) in nodes.as_node_weight_pairs() {
                let node = irreducible_integrand(
                    &kinematics,
                    loop_type,
                    time,
                    (abscissa + 1.0) * 0.5,
                    TensorLoopConfig::unit_natural(),
                    IrreducibleMutation::None,
                )
                .unwrap();
                regular += node.total.get(0, 0, 1)
                    * node.exponent
                    * node.determinant_factor
                    * (weight * 0.5);
            }
            let contact = irreducible_contact_integrand(
                &kinematics,
                loop_type,
                time,
                TensorLoopConfig::unit_natural(),
                IrreducibleMutation::None,
            )
            .unwrap();
            let determinant = match loop_type {
                LoopType::Scalar => (0.1 * time) / (0.1 * time).sinh(),
                LoopType::Spinor => (0.1 * time) / (0.1 * time).tanh(),
            };
            let regular_coefficient = regular / (Complex64::new(0.0, 1.0) * tree);
            let coefficient =
                (regular + contact.get(0, 0, 1) * determinant) / (Complex64::new(0.0, 1.0) * tree);
            let subtraction = match loop_type {
                LoopType::Scalar => -2.0 / 3.0,
                LoopType::Spinor => 4.0 / 3.0,
            };
            let subtraction_remainder = (coefficient + subtraction).norm();
            println!(
                "uv_complete loop={loop_type:?} time={time:.17e} coefficient={:.17e} subtraction_remainder={subtraction_remainder:.17e} omit_contact_coefficient={:.17e}",
                coefficient.re, regular_coefficient.re
            );
            assert!(subtraction_remainder < 1e-3);
            if loop_type == LoopType::Spinor {
                assert!((regular_coefficient + subtraction).norm() > 1.9);
            }
            if time <= 1e-6 {
                assert!(subtraction_remainder < 1e-7);
            }
        }
    }
}
