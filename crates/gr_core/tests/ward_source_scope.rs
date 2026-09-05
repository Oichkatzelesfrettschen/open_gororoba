//! Source-expression checks keep subtraction and polarization scopes separate.

use gauss_quad::GaussLegendre;
use gr_core::photon_graviton::{
    QuadratureConfig,
    external_tensor::{external_tensor_off_shell, external_tensor_unrenormalized_off_shell},
    irreducible_tensor::{
        IrreducibleMutation, irreducible_integrand, irreducible_tensor_renormalized,
        irreducible_tensor_unrenormalized,
    },
    tensor_integrands::TensorLoopConfig,
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, ComplexRankThreeTensor, MomentumRule, ShellMode,
        WardKinematics,
    },
    types::LoopType,
};
use num_complex::Complex64;
use std::{f64::consts::PI, num::NonZeroUsize};

#[test]
fn orbital_and_spinor_uv_coefficients_are_separate() {
    let fixture = subtraction_fixture();
    let config = TensorLoopConfig::unit_natural();
    let quad = GaussLegendre::new(NonZeroUsize::new(64).unwrap());
    for loop_type in [LoopType::Scalar, LoopType::Spinor] {
        for time in [0.1, 0.03, 0.01, 0.003, 0.001] {
            let mut integral = Complex64::default();
            for &(node, weight) in quad.as_node_weight_pairs() {
                let value = irreducible_integrand(
                    &fixture,
                    loop_type,
                    time,
                    0.5 + 0.5 * node,
                    config,
                    IrreducibleMutation::None,
                )
                .unwrap();
                integral += value.total.get(0, 0, 1)
                    * value.exponent
                    * value.determinant_factor
                    * (0.5 * weight);
            }
            // A.13 at indices (0,0,1) gives B*k[0].
            let tree = fixture.field_strength[(0, 1)] * fixture.k[0];
            let coefficient = integral / (Complex64::new(0.0, config.charge) * tree);
            println!(
                "uv_coefficient loop={loop_type:?} time={time:.17e} real={:.17e} imaginary={:.17e}",
                coefficient.re, coefficient.im
            );
            assert!(coefficient.norm().is_finite());
            if time == 0.001 {
                assert!((coefficient.re - 2.0 / 3.0).abs() < 1e-4);
                if loop_type == LoopType::Spinor {
                    assert!(
                        (coefficient.re + 4.0 / 3.0).abs() > 1.9,
                        "regular nodes alone retain a noncancelling source UV coefficient"
                    );
                }
            }
        }
    }
}

fn subtraction_fixture() -> WardKinematics {
    let momentum = ComplexFourVector::from([0.15, -0.11, 0.2, 0.07].map(Complex64::from));
    let mut field = ComplexLorentzMatrix::zeros();
    field[(0, 1)] = Complex64::from(0.1);
    field[(1, 0)] = Complex64::from(-0.1);
    WardKinematics::new(
        momentum,
        -momentum,
        ComplexFourVector::from([0.11, 0.15, 0.0, 0.0].map(Complex64::from)),
        ComplexLorentzMatrix::identity(),
        ComplexFourVector::from([0.2, 0.1, -0.3, 0.4].map(Complex64::from)),
        field,
        momentum.iter().map(|value| value * value).sum(),
        ShellMode::OffShell,
        MomentumRule::ConstantBackgroundConversion,
        false,
        1e-12,
    )
    .unwrap()
}

fn transverse_fixture(
    virtuality: f64,
    polarization_index: usize,
    gauge_index: usize,
) -> WardKinematics {
    let root_five = 5.0_f64.sqrt();
    let spatial = [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0];
    let basis = [
        [1.0 / root_five, -2.0 / root_five, 0.0],
        [
            4.0 / (3.0 * root_five),
            2.0 / (3.0 * root_five),
            -5.0 / (3.0 * root_five),
        ],
    ];
    let momentum = ComplexFourVector::from([
        Complex64::from(spatial[0]),
        Complex64::from(spatial[1]),
        Complex64::from(spatial[2]),
        Complex64::new(0.0, (1.0 - virtuality).sqrt()),
    ]);
    let polarization = ComplexFourVector::from_fn(|index, _| {
        Complex64::from(if index < 3 {
            basis[polarization_index][index]
        } else {
            0.0
        })
    });
    let gauge = ComplexFourVector::from_fn(|index, _| {
        Complex64::from(if index < 3 {
            basis[gauge_index][index]
        } else {
            0.0
        })
    });
    let mut field = ComplexLorentzMatrix::zeros();
    field[(0, 1)] = Complex64::from(0.1);
    field[(1, 0)] = Complex64::from(-0.1);
    WardKinematics::new(
        momentum,
        -momentum,
        polarization,
        ComplexLorentzMatrix::identity(),
        gauge,
        field,
        Complex64::from(virtuality),
        ShellMode::OffShell,
        MomentumRule::ConstantBackgroundConversion,
        true,
        1e-12,
    )
    .unwrap()
}

fn independent_gravity(
    tensor: &ComplexRankThreeTensor,
    fixture: &WardKinematics,
) -> [Complex64; 4] {
    std::array::from_fn(|photon| {
        (0..16)
            .rev()
            .map(|index| {
                let row = index / 4;
                let column = index % 4;
                tensor.get(row, column, photon)
                    * (fixture.k0[row] * fixture.zeta0[column]
                        + fixture.k0[column] * fixture.zeta0[row])
            })
            .sum()
    })
}

fn projection(vector: &[Complex64; 4], polarization: &ComplexFourVector) -> Complex64 {
    (0..4)
        .rev()
        .map(|index| vector[index] * polarization[index])
        .sum()
}

fn zero_intercept(values: &[Complex64; 3], virtualities: &[f64; 3]) -> Complex64 {
    (0..3)
        .map(|index| {
            let weight: f64 = (0..3)
                .filter(|other| *other != index)
                .map(|other| -virtualities[other] / (virtualities[index] - virtualities[other]))
                .product();
            values[index] * weight
        })
        .sum()
}

#[test]
fn transverse_source_identity_retains_diagram_omissions_and_vector_defects() {
    let config = TensorLoopConfig::unit_natural();
    let virtualities = [0.02, 0.01, 0.005];
    let quadrature = QuadratureConfig {
        n_u: 48,
        n_t: 96,
        t_min: 1e-4,
        t_max: 20.0,
    };
    let mut discriminating_pairs = 0;
    for loop_type in [LoopType::Scalar, LoopType::Spinor] {
        for renormalized in [false, true] {
            for polarization_index in 0..2 {
                for gauge_index in 0..2 {
                    let mut irreducible_samples = [Complex64::default(); 3];
                    let mut external_samples = [Complex64::default(); 3];
                    let mut vector_norms = [0.0; 3];
                    for (index, virtuality) in virtualities.iter().enumerate() {
                        let fixture =
                            transverse_fixture(*virtuality, polarization_index, gauge_index);
                        let irreducible = if renormalized {
                            irreducible_tensor_renormalized(
                                &fixture,
                                loop_type,
                                config,
                                &quadrature,
                            )
                        } else {
                            irreducible_tensor_unrenormalized(
                                &fixture,
                                loop_type,
                                config,
                                &quadrature,
                            )
                        }
                        .unwrap();
                        let external = if renormalized {
                            external_tensor_off_shell(&fixture, loop_type, config, &quadrature)
                        } else {
                            external_tensor_unrenormalized_off_shell(
                                &fixture,
                                loop_type,
                                config,
                                &quadrature,
                            )
                        }
                        .unwrap();
                        let irreducible_vector = independent_gravity(&irreducible, &fixture);
                        let external_vector = independent_gravity(&external, &fixture);
                        irreducible_samples[index] =
                            projection(&irreducible_vector, &fixture.epsilon);
                        external_samples[index] = projection(&external_vector, &fixture.epsilon);
                        vector_norms[index] = (0..4)
                            .map(|component| {
                                (irreducible_vector[component] + external_vector[component])
                                    .norm_sqr()
                            })
                            .sum::<f64>()
                            .sqrt();
                    }
                    let irreducible_limit = zero_intercept(&irreducible_samples, &virtualities);
                    let external_limit = zero_intercept(&external_samples, &virtualities);
                    let combined_limit = irreducible_limit + external_limit;
                    let scale = irreducible_limit.norm().max(external_limit.norm());
                    let omission_discriminates =
                        irreducible_limit.norm() > 1e-8 && external_limit.norm() > 1e-8;
                    discriminating_pairs += usize::from(omission_discriminates);
                    println!(
                        "transverse loop={loop_type:?} renormalized={renormalized} polarization={polarization_index} zeta={gauge_index} combined={:.17e} normalized={:.17e} omit_external={:.17e} omit_irreducible={:.17e} vector_norm_last={:.17e} omission_discriminates={omission_discriminates}",
                        combined_limit.norm(),
                        combined_limit.norm() / scale.max(f64::MIN_POSITIVE),
                        irreducible_limit.norm(),
                        external_limit.norm(),
                        vector_norms[2]
                    );
                    assert!(combined_limit.norm().is_finite());
                    if omission_discriminates {
                        let passes_source_gate =
                            combined_limit.norm() <= 1e-12 && combined_limit.norm() / scale <= 1e-6;
                        assert!(passes_source_gate);
                    }
                }
            }
        }
    }
    assert!(
        discriminating_pairs >= 4,
        "generic controls must change transverse amplitudes"
    );
}

fn simpson(function: impl Fn(f64) -> f64, lower: f64, upper: f64) -> f64 {
    let intervals = 16384;
    let step = (upper - lower) / intervals as f64;
    let interior: f64 = (1..intervals)
        .map(|index| {
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            weight * function(lower + index as f64 * step)
        })
        .sum();
    (function(lower) + interior + function(upper)) * step / 3.0
}

#[test]
fn rescaled_counterterm_matches_source_measure() {
    let fixture = subtraction_fixture();
    let quadrature = QuadratureConfig {
        n_u: 24,
        n_t: 96,
        t_min: 0.125,
        t_max: 4.0,
    };
    for mass in [0.8, 1.3] {
        let config = TensorLoopConfig {
            mass,
            ..TensorLoopConfig::unit_natural()
        };
        let source_integral = simpson(
            |time| (-mass * mass * time).exp() / time,
            quadrature.t_min,
            quadrature.t_max,
        );
        let wrong_measure_integral = simpson(
            |time| (-mass * mass * time).exp() * time,
            quadrature.t_min,
            quadrature.t_max,
        );
        for loop_type in [LoopType::Spinor, LoopType::Scalar] {
            let bare = irreducible_tensor_unrenormalized(&fixture, loop_type, config, &quadrature)
                .unwrap();
            let renormalized =
                irreducible_tensor_renormalized(&fixture, loop_type, config, &quadrature).unwrap();
            let spinor_prefactor = -config.charge * config.kappa / (32.0 * PI * PI);
            let prefactor = match loop_type {
                LoopType::Spinor => spinor_prefactor,
                LoopType::Scalar => -0.5 * spinor_prefactor,
            };
            let mut squared_error = 0.0;
            let mut source_squared_norm = 0.0;
            let mut wrong_squared_error = 0.0;
            for row in 0..4 {
                for column in 0..4 {
                    for photon in 0..4 {
                        // Source Eq. A.13 derives the tree tensor independently.
                        let field_times = |index| {
                            (0..4)
                                .map(|other| {
                                    fixture.field_strength[(index, other)] * fixture.k[other]
                                })
                                .sum::<Complex64>()
                        };
                        let tree = fixture.field_strength[(row, photon)] * fixture.k[column]
                            + fixture.field_strength[(column, photon)] * fixture.k[row]
                            - field_times(row) * f64::from(column == photon)
                            - field_times(column) * f64::from(row == photon)
                            + field_times(photon) * f64::from(row == column);
                        // Eq. 4.10 gives the spinor coefficient. The scalar
                        // coefficient maps gr-qc/0412095 Eq. 3.11 through
                        // k_old=-k_photon, so C_old=-C_photon.
                        let source_coefficient = match loop_type {
                            LoopType::Spinor => 4.0 / 3.0,
                            LoopType::Scalar => -2.0 / 3.0,
                        };
                        let coefficient = tree
                            * Complex64::new(0.0, prefactor * source_coefficient * config.charge);
                        let expected = coefficient * source_integral;
                        let observed =
                            renormalized.get(row, column, photon) - bare.get(row, column, photon);
                        squared_error += (observed - expected).norm_sqr();
                        wrong_squared_error +=
                            (observed - coefficient * wrong_measure_integral).norm_sqr();
                        source_squared_norm += expected.norm_sqr();
                    }
                }
            }
            println!(
                "counterterm loop={loop_type:?} mass={mass} source_relative_error={:.17e} wrong_measure_relative_error={:.17e} omitted_norm={:.17e}",
                (squared_error / source_squared_norm).sqrt(),
                (wrong_squared_error / source_squared_norm).sqrt(),
                source_squared_norm.sqrt()
            );
            assert!(
                (squared_error / source_squared_norm).sqrt() < 1e-9,
                "rescaled source counterterm differs"
            );
            assert!((wrong_squared_error / source_squared_norm).sqrt() > 0.05);
            assert!(
                source_squared_norm.sqrt() > 1e-5,
                "omission control must discriminate"
            );
        }
    }
}
