//! Rank-three irreducible photon-graviton tensors.
//!
//! Ahmadiniaz et al., arXiv:2601.23279v1, equations 4.9 through 4.11
//! supply the J1, J2,
//! and J3 structures before the graviton indices are symmetrized. Scalar
//! loops use the orbital terms and the source global scalar-loop factor.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::{
    quadrature::QuadratureConfig,
    tensor_integrands::{
        SourceWorldlineNode, TensorEvaluationError, TensorLoopConfig, bilinear,
        double_integrate_rank_three_with_contact, even, left_contract, rank_three_add,
        rank_three_scale, right_contract, scalar_determinant, source_worldline_node,
        validate_tensor_inputs,
    },
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, ComplexRankThreeTensor, WardKinematics,
    },
    types::LoopType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrreducibleMutation {
    None,
    FlipJ1Sign,
    FlipJ2Sign,
    FlipJ3Sign,
    OmitJ1,
    OmitJ2,
    OmitJ3,
    TransposeAntisymmetricProjector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrreducibleNode {
    pub j1: ComplexRankThreeTensor,
    pub j2: ComplexRankThreeTensor,
    pub j3: ComplexRankThreeTensor,
    pub total: ComplexRankThreeTensor,
    pub exponent: Complex64,
    pub determinant_factor: f64,
}

/// Assemble one source integrand node without reducing the rank-three tensor.
pub fn irreducible_integrand(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    proper_time: f64,
    u: f64,
    loop_config: TensorLoopConfig,
    mutation: IrreducibleMutation,
) -> Result<IrreducibleNode, TensorEvaluationError> {
    let (_, magnetic_field) = validate_tensor_inputs(
        kinematics,
        loop_config,
        &QuadratureConfig {
            n_u: 1,
            n_t: 1,
            t_min: proper_time,
            t_max: proper_time + 1.0,
        },
    )?;
    if !proper_time.is_finite() || proper_time <= 0.0 || !u.is_finite() || !(0.0..=1.0).contains(&u)
    {
        return Err(TensorEvaluationError::InvalidQuadrature);
    }
    let node = source_worldline_node(magnetic_field, loop_config.charge, proper_time, u)?;
    let node = mutate_worldline_node(node, mutation);
    let (j1, j2, j3) = match loop_type {
        LoopType::Scalar => scalar_j_structures(&node, &kinematics.k),
        LoopType::Spinor => spinor_j_structures(&node, &kinematics.k),
    };
    let (j1, j2, j3) = apply_j_mutation(j1, j2, j3, mutation);
    let total = rank_three_add(&rank_three_add(&j1, &j2), &j3);
    let exponent = (-bilinear(&kinematics.k, &node.bar_g_b12, &kinematics.k)).exp();
    let determinant_factor =
        scalar_determinant(loop_type, loop_config.charge * magnetic_field * proper_time);
    Ok(IrreducibleNode {
        j1,
        j2,
        j3,
        total,
        exponent,
        determinant_factor,
    })
}

/// Integrate the unrenormalized rank-three source tensor.
pub fn irreducible_tensor_unrenormalized(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError> {
    irreducible_tensor_unrenormalized_with_mutation(
        kinematics,
        loop_type,
        loop_config,
        quadrature,
        IrreducibleMutation::None,
    )
}

/// Integrate the unrenormalized tensor with an explicit source mutation.
///
/// The mutation remains on the same source-owned path as the control tensor,
/// so a zero-producing legacy shortcut cannot make a declared defect pass.
pub fn irreducible_tensor_unrenormalized_with_mutation(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
    mutation: IrreducibleMutation,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError> {
    let (loop_config, magnetic_field) =
        validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let kinematics = *kinematics;
    let tensor = double_integrate_rank_three_with_contact(
        |proper_time, u| {
            let node = mutate_worldline_node(
                source_worldline_node(magnetic_field, loop_config.charge, proper_time, u)?,
                mutation,
            );
            let (j1, j2, j3) = match loop_type {
                LoopType::Scalar => scalar_j_structures(&node, &kinematics.k),
                LoopType::Spinor => spinor_j_structures(&node, &kinematics.k),
            };
            let (j1, j2, j3) = apply_j_mutation(j1, j2, j3, mutation);
            let total = rank_three_add(&rank_three_add(&j1, &j2), &j3);
            let exponent = (-bilinear(&kinematics.k, &node.bar_g_b12, &kinematics.k)).exp();
            let factor = Complex64::new(
                scalar_determinant(loop_type, loop_config.charge * magnetic_field * proper_time),
                0.0,
            ) * exponent;
            Ok(rank_three_scale(&total, factor))
        },
        |proper_time| {
            let contact = irreducible_contact_integrand(
                &kinematics,
                loop_type,
                proper_time,
                loop_config,
                mutation,
            )?;
            Ok(rank_three_scale(
                &contact,
                Complex64::new(
                    scalar_determinant(
                        loop_type,
                        loop_config.charge * magnetic_field * proper_time,
                    ),
                    0.0,
                ),
            ))
        },
        loop_config.mass,
        1.0,
        quadrature,
    )?;
    Ok(rank_three_scale(
        &tensor,
        irreducible_prefactor(loop_type, loop_config),
    ))
}

/// Integrate the distinct renormalized rank-three path with the source tree
/// counterterm. The unrenormalized and renormalized APIs do not share a
/// projected scalar result.
pub fn irreducible_tensor_renormalized(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    loop_config: TensorLoopConfig,
    quadrature: &QuadratureConfig,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError> {
    let (loop_config, magnetic_field) =
        validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let kinematics = *kinematics;
    let tensor = double_integrate_rank_three_with_contact(
        |proper_time, u| {
            let node = source_worldline_node(magnetic_field, loop_config.charge, proper_time, u)?;
            let (j1, j2, j3) = match loop_type {
                LoopType::Scalar => scalar_j_structures(&node, &kinematics.k),
                LoopType::Spinor => spinor_j_structures(&node, &kinematics.k),
            };
            let total = rank_three_add(&rank_three_add(&j1, &j2), &j3);
            let exponent = (-bilinear(&kinematics.k, &node.bar_g_b12, &kinematics.k)).exp();
            let determinant =
                scalar_determinant(loop_type, loop_config.charge * magnetic_field * proper_time);
            let source_term = rank_three_scale(&total, Complex64::new(determinant, 0.0) * exponent);
            // gr-qc/0412095 Eq. 3.11 defines k as the graviton momentum.
            // Reversing it to the photon convention changes C's sign, so
            // the scalar +2/3 coefficient becomes -2/3. The spinor Eq. 4.10
            // coefficient in arXiv:2601.23279v1 is already photon-oriented.
            let subtraction_coefficient = match loop_type {
                LoopType::Scalar => -2.0 / 3.0,
                LoopType::Spinor => 4.0 / 3.0,
            };
            let counterterm = rank_three_scale(
                &tree_tensor(&kinematics.field_strength, &kinematics.k),
                // Eq. 4.10 uses dT/T after both insertion times are rescaled;
                // the T^2 factor in Eq. 4.9 has already entered the measure.
                Complex64::new(0.0, subtraction_coefficient * loop_config.charge),
            );
            Ok(rank_three_add(&source_term, &counterterm))
        },
        |proper_time| {
            let contact = irreducible_contact_integrand(
                &kinematics,
                loop_type,
                proper_time,
                loop_config,
                IrreducibleMutation::None,
            )?;
            Ok(rank_three_scale(
                &contact,
                Complex64::new(
                    scalar_determinant(
                        loop_type,
                        loop_config.charge * magnetic_field * proper_time,
                    ),
                    0.0,
                ),
            ))
        },
        loop_config.mass,
        1.0,
        quadrature,
    )?;
    Ok(rank_three_scale(
        &tensor,
        irreducible_prefactor(loop_type, loop_config),
    ))
}

/// Analytically integrate the J2 delta12 terms with periodic midpoint products.
/// B.4 and B.36 each supply 2*delta12*I. At coincidence bar-dot-G_B=0 and
/// G_F12=G_F11, leaving symmetrized G_F11*k with the delta(tau)=delta(u)/T
/// Jacobian. The determinant and proper-time measure remain outside this API.
pub fn irreducible_contact_integrand(
    kinematics: &WardKinematics,
    loop_type: LoopType,
    proper_time: f64,
    loop_config: TensorLoopConfig,
    mutation: IrreducibleMutation,
) -> Result<ComplexRankThreeTensor, TensorEvaluationError> {
    let (_, magnetic_field) = validate_tensor_inputs(
        kinematics,
        loop_config,
        &QuadratureConfig {
            n_u: 1,
            n_t: 1,
            t_min: proper_time,
            t_max: proper_time + 1.0,
        },
    )?;
    if loop_type == LoopType::Scalar || mutation == IrreducibleMutation::OmitJ2 {
        return Ok(ComplexRankThreeTensor::zero());
    }
    let node = source_worldline_node(magnetic_field, loop_config.charge, proper_time, 0.0)?;
    let sign = if mutation == IrreducibleMutation::FlipJ2Sign {
        -1.0
    } else {
        1.0
    };
    Ok(ComplexRankThreeTensor::from_fn(|row, column, photon| {
        (node.coincidence.g_f[(row, photon)] * kinematics.k[column]
            + node.coincidence.g_f[(column, photon)] * kinematics.k[row])
            * (sign / proper_time)
    }))
}

fn irreducible_prefactor(loop_type: LoopType, loop_config: TensorLoopConfig) -> Complex64 {
    let spinor = -loop_config.charge * loop_config.kappa / (32.0 * PI * PI);
    match loop_type {
        LoopType::Spinor => Complex64::new(spinor, 0.0),
        LoopType::Scalar => Complex64::new(-0.5 * spinor, 0.0),
    }
}

fn mutate_worldline_node(
    mut node: SourceWorldlineNode,
    mutation: IrreducibleMutation,
) -> SourceWorldlineNode {
    if matches!(
        mutation,
        IrreducibleMutation::TransposeAntisymmetricProjector
    ) {
        // Transpose the source dot-G_B field projector before rebuilding its
        // even and odd parts. This keeps the mutation on the tensor path
        // rather than changing an unused legacy-derived cache.
        node.full.dot_g_b = node.full.dot_g_b.transpose();
        node.dot_s_b12 = even(&node.full.dot_g_b);
        node.dot_a_b12 = super::tensor_integrands::odd(&node.full.dot_g_b);
        node.bar_dot_g_b12 = node.full.dot_g_b - node.coincidence.dot_g_b;
        node.bar_dot_g_b21 = -node.full.dot_g_b.transpose() - node.coincidence.dot_g_b;
    }
    node
}

fn apply_j_mutation(
    j1: ComplexRankThreeTensor,
    j2: ComplexRankThreeTensor,
    j3: ComplexRankThreeTensor,
    mutation: IrreducibleMutation,
) -> (
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
) {
    let zero = ComplexRankThreeTensor::zero();
    match mutation {
        IrreducibleMutation::FlipJ1Sign => {
            (rank_three_scale(&j1, -Complex64::new(1.0, 0.0)), j2, j3)
        }
        IrreducibleMutation::FlipJ2Sign => {
            (j1, rank_three_scale(&j2, -Complex64::new(1.0, 0.0)), j3)
        }
        IrreducibleMutation::FlipJ3Sign => {
            (j1, j2, rank_three_scale(&j3, -Complex64::new(1.0, 0.0)))
        }
        IrreducibleMutation::OmitJ1 => (zero, j2, j3),
        IrreducibleMutation::OmitJ2 => (j1, zero, j3),
        IrreducibleMutation::OmitJ3 => (j1, j2, zero),
        IrreducibleMutation::None | IrreducibleMutation::TransposeAntisymmetricProjector => {
            (j1, j2, j3)
        }
    }
}

fn scalar_j_structures(
    node: &SourceWorldlineNode,
    k: &ComplexFourVector,
) -> (
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
) {
    tilde_j_structures(node, k, false)
}

fn spinor_j_structures(
    node: &SourceWorldlineNode,
    k: &ComplexFourVector,
) -> (
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
) {
    tilde_j_structures(node, k, true)
}

fn tilde_j_structures(
    node: &SourceWorldlineNode,
    k: &ComplexFourVector,
    include_fermion: bool,
) -> (
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
) {
    let zero = ComplexLorentzMatrix::zeros();
    let ddot_s_b11 = even(&node.coincidence.ddot_g_b.regular);
    let ddot_s_b12 = even(&node.full.ddot_g_b);
    let ddot_a_b12 = super::tensor_integrands::odd(&node.full.ddot_g_b);
    let dot_s_b12 = node.dot_s_b12;
    let dot_a_b12 = node.dot_a_b12 - super::tensor_integrands::odd(&node.coincidence.dot_g_b);
    let dot_s_f11 = if include_fermion {
        even(&node.coincidence.dot_g_f.regular)
    } else {
        zero
    };
    let a_f11 = if include_fermion {
        super::tensor_integrands::odd(&node.coincidence.g_f)
    } else {
        zero
    };
    let s_f12 = if include_fermion { node.s_f12 } else { zero };
    let a_f12 = if include_fermion { node.a_f12 } else { zero };
    let dot_s_f12 = if include_fermion {
        node.dot_s_f12
    } else {
        zero
    };
    let dot_a_f12 = if include_fermion {
        node.dot_a_f12
    } else {
        zero
    };
    let bar_dot_a_plus_a_f11 = dot_a_b12 + a_f11;
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
    let dot_s_b_bilinear = bilinear(k, &dot_s_b12, k);
    let s_f_bilinear = bilinear(k, &s_f12, k);
    let k_dot_a_f = left_contract(k, &a_f12);
    let k_dot_s_f_cached = k_dot_s_f;

    let j1 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        (ddot_s_b11[(mu, nu)] - dot_s_f11[(mu, nu)]) * k_dot_bar_dot_a_plus_a_f11[alpha]
    }));
    let j2 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        -dot_s_b12[(mu, alpha)] * ddot_a_b_k[nu]
            + s_f12[(mu, alpha)] * dot_a_f_k[nu]
            + ddot_a_b12[(nu, alpha)] * dot_s_b_k[mu]
            - dot_a_f12[(nu, alpha)] * s_f_k[mu]
            - dot_a_b12[(mu, alpha)] * ddot_s_b_k[nu]
            + a_f12[(mu, alpha)] * dot_s_f_k[nu]
            + ddot_s_b12[(nu, alpha)] * bar_dot_a_plus_a_f11_k[mu]
            - dot_s_f12[(nu, alpha)] * a_f_k[mu]
    }));
    let j3 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        let first_bracket = dot_s_b_k[nu] * k_dot_bar_dot_a_plus_a_f11[alpha]
            - s_f_k[nu] * k_dot_a_f[alpha]
            + bar_dot_a_plus_a_f11_k[nu] * k_dot_dot_s_b[alpha]
            - a_f_k[nu] * k_dot_s_f_cached[alpha]
            - dot_a_b12[(nu, alpha)] * dot_s_b_bilinear
            + a_f12[(nu, alpha)] * s_f_bilinear;
        let second_bracket = dot_s_b_k[nu] * k_dot_dot_s_b[alpha]
            - s_f_k[nu] * k_dot_s_f_cached[alpha]
            + bar_dot_a_plus_a_f11_k[nu] * k_dot_bar_dot_a_plus_a_f11[alpha]
            - a_f_k[nu] * k_dot_a_f[alpha]
            - dot_s_b12[(nu, alpha)] * dot_s_b_bilinear
            + s_f12[(nu, alpha)] * s_f_bilinear;
        -dot_s_b_k[mu] * first_bracket - bar_dot_a_k[mu] * second_bracket
    }));
    (j1, j2, j3)
}

fn symmetrize(tensor: ComplexRankThreeTensor) -> ComplexRankThreeTensor {
    ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        (tensor.get(mu, nu, alpha) + tensor.get(nu, mu, alpha)) * Complex64::new(0.5, 0.0)
    })
}

fn tree_tensor(
    field_strength: &ComplexLorentzMatrix,
    k: &ComplexFourVector,
) -> ComplexRankThreeTensor {
    let field_k = right_contract(field_strength, k);
    ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        let first = field_strength[(mu, alpha)] * k[nu];
        let second = field_strength[(nu, alpha)] * k[mu];
        let third = if nu == alpha {
            field_k[mu]
        } else {
            Complex64::new(0.0, 0.0)
        };
        let fourth = if mu == alpha {
            field_k[nu]
        } else {
            Complex64::new(0.0, 0.0)
        };
        let fifth = if mu == nu {
            field_k[alpha]
        } else {
            Complex64::new(0.0, 0.0)
        };
        first + second - third - fourth + fifth
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

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
            ComplexFourVector::from([
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            ComplexLorentzMatrix::identity(),
            ComplexFourVector::from([
                Complex64::new(0.2, 0.0),
                Complex64::new(0.1, 0.0),
                Complex64::new(-0.3, 0.0),
                Complex64::new(0.4, 0.0),
            ]),
            field,
            Complex64::new(0.0625, 0.0),
            super::super::tensor_types::ShellMode::OffShell,
            super::super::tensor_types::MomentumRule::ConstantBackgroundConversion,
            false,
            1.0e-12,
        )
        .expect("valid tensor fixture")
    }

    #[test]
    fn j_structures_retain_rank_and_symmetrization() {
        let node = irreducible_integrand(
            &fixture(),
            LoopType::Spinor,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
            IrreducibleMutation::None,
        )
        .expect("valid irreducible node");
        assert!(
            node.total
                .components()
                .iter()
                .any(|value| value.norm() > 1.0e-14)
        );
        for mu in 0..4 {
            for nu in 0..4 {
                for alpha in 0..4 {
                    assert_eq!(node.total.get(mu, nu, alpha), node.total.get(nu, mu, alpha));
                }
            }
        }
    }

    #[test]
    fn each_j_mutation_changes_a_non_degenerate_fixture() {
        let base = irreducible_integrand(
            &fixture(),
            LoopType::Spinor,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
            IrreducibleMutation::None,
        )
        .expect("base node");
        for mutation in [
            IrreducibleMutation::FlipJ1Sign,
            IrreducibleMutation::FlipJ2Sign,
            IrreducibleMutation::FlipJ3Sign,
            IrreducibleMutation::OmitJ1,
            IrreducibleMutation::OmitJ2,
            IrreducibleMutation::OmitJ3,
            IrreducibleMutation::TransposeAntisymmetricProjector,
        ] {
            let changed = irreducible_integrand(
                &fixture(),
                LoopType::Spinor,
                0.8,
                0.27,
                TensorLoopConfig::unit_natural(),
                mutation,
            )
            .expect("mutated node");
            let difference = base
                .total
                .components()
                .iter()
                .zip(changed.total.components().iter())
                .map(|(left, right)| (*left - *right).norm_sqr())
                .sum::<f64>()
                .sqrt();
            assert!(
                difference > 1.0e-12,
                "mutation {mutation:?} was not detected"
            );
        }
    }

    #[test]
    fn scalar_path_uses_orbital_rank_three_terms() {
        let node = irreducible_integrand(
            &fixture(),
            LoopType::Scalar,
            0.8,
            0.27,
            TensorLoopConfig::unit_natural(),
            IrreducibleMutation::None,
        )
        .expect("scalar node");
        assert!(node.j1.components().iter().any(|value| value.norm() > 0.0));
        assert!(node.j2.components().iter().any(|value| value.norm() > 0.0));
        assert!(node.j3.components().iter().any(|value| value.norm() > 0.0));
    }
}
