//! Rank-three irreducible photon-graviton tensors.
//!
//! Equations 4.9, 4.10, and 4.11 are assembled from the complete J1, J2,
//! and J3 structures before the graviton indices are symmetrized. Scalar
//! loops use the orbital terms and the source global scalar-loop factor.

use num_complex::Complex64;
use std::f64::consts::PI;

use super::{
    quadrature::QuadratureConfig,
    tensor_integrands::{
        SourceWorldlineNode, TensorEvaluationError, TensorLoopConfig, bilinear,
        double_integrate_rank_three, rank_three_add, rank_three_scale, right_contract,
        scalar_determinant, source_worldline_node, validate_tensor_inputs,
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
    let (loop_config, magnetic_field) =
        validate_tensor_inputs(kinematics, loop_config, quadrature)?;
    let kinematics = *kinematics;
    let tensor = double_integrate_rank_three(
        |proper_time, u| {
            let node = source_worldline_node(magnetic_field, loop_config.charge, proper_time, u)?;
            let (j1, j2, j3) = match loop_type {
                LoopType::Scalar => scalar_j_structures(&node, &kinematics.k),
                LoopType::Spinor => spinor_j_structures(&node, &kinematics.k),
            };
            let total = rank_three_add(&rank_three_add(&j1, &j2), &j3);
            let exponent = (-bilinear(&kinematics.k, &node.bar_g_b12, &kinematics.k)).exp();
            let factor = Complex64::new(
                scalar_determinant(loop_type, loop_config.charge * magnetic_field * proper_time),
                0.0,
            ) * exponent;
            Ok(rank_three_scale(&total, factor))
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
    let tensor = double_integrate_rank_three(
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
            let counterterm = rank_three_scale(
                &tree_tensor(&kinematics.field_strength, &kinematics.k),
                Complex64::new(0.0, 4.0 / 3.0 * loop_config.charge * proper_time.powi(2)),
            );
            Ok(rank_three_add(&source_term, &counterterm))
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
        node.bar_dot_g_b12 = node.bar_dot_g_b12.transpose();
        node.bar_dot_g_b21 = node.bar_dot_g_b21.transpose();
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
    let ddot_b11 = node.coincidence.ddot_g_b.regular;
    let bar_dot_k = right_contract(&node.bar_dot_g_b12, k);
    let k_dot_bar_dot = super::tensor_integrands::left_contract(k, &node.bar_dot_g_b12);
    let j1 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        ddot_b11[(mu, nu)] * k_dot_bar_dot[alpha]
    }));
    let j2 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        node.full.ddot_g_b[(mu, alpha)] * bar_dot_k[nu]
            + node.full.ddot_g_b[(nu, alpha)] * bar_dot_k[mu]
    }));
    let j3 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        -bar_dot_k[mu] * bar_dot_k[nu] * k_dot_bar_dot[alpha]
    }));
    (j1, j2, j3)
}

fn spinor_j_structures(
    node: &SourceWorldlineNode,
    k: &ComplexFourVector,
) -> (
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
    ComplexRankThreeTensor,
) {
    let ddot_b11 = node.coincidence.ddot_g_b.regular;
    let dot_f11 = node.coincidence.dot_g_f.regular;
    let gf11 = node.coincidence.g_f;
    let gf22 = node.coincidence.g_f;
    let gf12 = node.full.g_f;
    let gf21 = -gf12.transpose();
    let dot_gf12 = node.full.dot_g_f;
    let ddot_b12 = node.full.ddot_g_b;
    let bar_dot12 = node.bar_dot_g_b12;
    let bar_dot21 = node.bar_dot_g_b21;
    let first_vector = right_contract(&(bar_dot21 + gf22), k);
    let bar_dot12_k = right_contract(&bar_dot12, k);
    let gf12_k = right_contract(&gf12, k);
    let gf21_k = right_contract(&gf21, k);
    let dot_gf12_k = right_contract(&dot_gf12, k);
    let bar_dot12_gf11_k = right_contract(&(bar_dot12 + gf11), k);
    let dot_bilinear = bilinear(k, &node.full.dot_g_b, k);
    let gf_bilinear = bilinear(k, &gf12, k);

    let j1 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        -(ddot_b11[(mu, nu)] - dot_f11[(mu, nu)]) * first_vector[alpha]
    }));
    let j2 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        -bar_dot12[(mu, alpha)] * right_contract(&ddot_b12, k)[nu]
            + gf12[(mu, alpha)] * dot_gf12_k[nu]
            + ddot_b12[(nu, alpha)] * bar_dot12_gf11_k[mu]
            - dot_gf12[(nu, alpha)] * gf12_k[mu]
    }));
    let j3 = symmetrize(ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
        bar_dot12_k[mu]
            * (bar_dot12_gf11_k[nu] * first_vector[alpha] - gf12_k[nu] * gf21_k[alpha]
                + bar_dot12[(nu, alpha)] * dot_bilinear
                - gf12[(nu, alpha)] * gf_bilinear)
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
