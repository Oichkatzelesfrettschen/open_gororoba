//! Independent component contractions separate gauge and gravitational predicates.
//! Physics tensors and quadrature remain shared with gr_core; the oracle checks
//! contraction, lower-point assembly, omissions, and extrapolation arithmetic.

use gr_core::photon_graviton::{
    QuadratureConfig,
    external_tensor::external_tensor_off_shell,
    gravitational_ward::{gravitational_ward_off_shell, on_shell_combined_virtuality_ladder},
    irreducible_tensor::{
        IrreducibleMutation, irreducible_tensor_renormalized,
        irreducible_tensor_unrenormalized_with_mutation,
    },
    one_photon::one_photon_amplitude,
    tensor_integrands::TensorLoopConfig,
    tensor_types::{
        ComplexFourVector, ComplexLorentzMatrix, ComplexRankThreeTensor, MomentumRule,
        ResidualTolerance, ShellMode, WardKinematics,
    },
    tensor_ward::gauge_ward_integrated_residuals,
    types::LoopType,
    vacuum_pol_tensor::vacuum_polarization_tensor_unrenormalized,
};
use num_complex::Complex64;
use std::fmt::Write as _;

fn fixture(virtuality: Option<f64>) -> WardKinematics {
    let mut field = ComplexLorentzMatrix::zeros();
    field[(0, 1)] = Complex64::new(0.1, 0.0);
    field[(1, 0)] = Complex64::new(-0.1, 0.0);
    let mut momentum = ComplexFourVector::zeros();
    let mut gauge_parameter = ComplexFourVector::zeros();
    if let Some(value) = virtuality {
        momentum[0] = Complex64::new(1.0, 0.0);
        momentum[3] = Complex64::new(0.0, (1.0 - value).sqrt());
        gauge_parameter[1] = Complex64::new(1.0, 0.0);
    } else {
        momentum[0] = Complex64::new(0.15, 0.0);
        momentum[2] = Complex64::new(0.2, 0.0);
        gauge_parameter =
            ComplexFourVector::from([0.2, 0.1, -0.3, 0.4].map(|value| Complex64::new(value, 0.0)));
    }
    let mut polarization = ComplexFourVector::zeros();
    polarization[1] = Complex64::new(1.0, 0.0);
    WardKinematics::new(
        momentum,
        -momentum,
        polarization,
        ComplexLorentzMatrix::identity(),
        gauge_parameter,
        field,
        Complex64::new(virtuality.unwrap_or(0.0625), 0.0),
        ShellMode::OffShell,
        MomentumRule::ConstantBackgroundConversion,
        virtuality.is_some(),
        1e-12,
    )
    .unwrap()
}

fn norm(values: &[Complex64]) -> f64 {
    values
        .iter()
        .fold(0.0, |total, value| total.hypot(value.re).hypot(value.im))
}

fn photon_contract(tensor: &ComplexRankThreeTensor, kinematics: &WardKinematics) -> Vec<Complex64> {
    (0..16)
        .map(|index| {
            (0..4)
                .rev()
                .map(|photon| tensor.get(index / 4, index % 4, photon) * kinematics.k[photon])
                .sum()
        })
        .collect()
}

fn gravity_contract(
    tensor: &ComplexRankThreeTensor,
    kinematics: &WardKinematics,
) -> [Complex64; 4] {
    std::array::from_fn(|photon| {
        (0..16)
            .rev()
            .map(|index| {
                let row = index / 4;
                let column = index % 4;
                tensor.get(row, column, photon)
                    * (kinematics.k0[row] * kinematics.zeta0[column]
                        + kinematics.k0[column] * kinematics.zeta0[row])
            })
            .sum()
    })
}

fn difference(left: &[Complex64; 4], right: &[Complex64; 4]) -> [Complex64; 4] {
    std::array::from_fn(|index| left[index] - right[index])
}

fn intercept(abscissae: &[f64], values: &[[Complex64; 4]]) -> [Complex64; 4] {
    // Lagrange interpolation evaluates the polynomial at zero directly.
    std::array::from_fn(|component| {
        abscissae
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let weight: f64 = abscissae
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(other, abscissa)| -abscissa / (abscissae[index] - abscissae[other]))
                    .product();
                values[index][component] * weight
            })
            .sum()
    })
}

#[test]
fn independent_ward_residuals_preserve_failures_and_omission_scope() {
    let loop_config = TensorLoopConfig::unit_natural();
    let quadrature = QuadratureConfig::fast();
    let tolerance = ResidualTolerance::new(1e-12, 1e-6).unwrap();
    let kinematics = fixture(None);
    let mut report = String::from(
        "schema_version = 1\nevidence_layer = \"implementation_conformance\"\nindependence = \"Independent component contraction, RHS assembly and Lagrange intercept; shared physics tensors and quadrature\"\nerror_budget_status = \"heuristic refinement sensitivity; certified quadrature and extrapolation remainder unresolved\"\n",
    );
    for loop_type in [LoopType::Scalar, LoopType::Spinor] {
        let tensor = irreducible_tensor_unrenormalized_with_mutation(
            &kinematics,
            loop_type,
            loop_config,
            &quadrature,
            IrreducibleMutation::None,
        )
        .unwrap();
        let gauge = photon_contract(&tensor, &kinematics);
        let gauge_absolute = norm(&gauge);
        let gauge_normalized = gauge_absolute / norm(tensor.components());
        let recorded = gauge_ward_integrated_residuals(
            &kinematics,
            loop_type,
            loop_config,
            &quadrature,
            tolerance,
        )
        .unwrap();
        for row in 0..4 {
            for column in 0..4 {
                assert!(
                    (gauge[row * 4 + column] - recorded[0].contracted_components[(row, column)])
                        .norm()
                        < 1e-14
                );
            }
        }
        assert!((gauge_normalized - recorded[0].normalized_norm).abs() < 1e-14);
        let gauge_passes =
            gauge_absolute <= tolerance.absolute && gauge_normalized <= tolerance.normalized;
        assert_eq!(gauge_passes, loop_type == LoopType::Scalar);
        writeln!(report, "\n[[gauge]]\nloop_type = \"{loop_type:?}\"\nabsolute = {gauge_absolute:.17e}\nnormalized = {gauge_normalized:.17e}\npasses = {gauge_passes}").unwrap();
        for mutation in [
            IrreducibleMutation::OmitJ1,
            IrreducibleMutation::OmitJ2,
            IrreducibleMutation::OmitJ3,
        ] {
            let changed = irreducible_tensor_unrenormalized_with_mutation(
                &kinematics,
                loop_type,
                loop_config,
                &quadrature,
                mutation,
            )
            .unwrap();
            let omitted_norm = norm(&photon_contract(&changed, &kinematics));
            let tensor_delta: Vec<_> = changed
                .components()
                .iter()
                .zip(tensor.components())
                .map(|(changed, original)| changed - original)
                .collect();
            let tensor_delta_norm = norm(&tensor_delta);
            assert!(tensor_delta_norm > 1e-10);
            writeln!(report, "\n[[gauge_omission]]\nloop_type = \"{loop_type:?}\"\nmutation = \"{mutation:?}\"\nabsolute = {omitted_norm:.17e}\ntensor_delta_norm = {tensor_delta_norm:.17e}").unwrap();
        }

        let gravitational = gravitational_ward_off_shell(
            &kinematics,
            loop_type,
            loop_config,
            &quadrature,
            tolerance,
        )
        .unwrap();
        let lhs = gravity_contract(&tensor, &kinematics);
        let vacuum = vacuum_polarization_tensor_unrenormalized(
            &kinematics,
            loop_type,
            loop_config,
            &quadrature,
        )
        .unwrap();
        let lower_momentum = kinematics.k0 + kinematics.k;
        let mut one_photon = [Complex64::new(0.0, 0.0); 4];
        let mut two_photon = [Complex64::new(0.0, 0.0); 4];
        for photon in 0..4 {
            let effective = ComplexFourVector::from_fn(|row, _| {
                (0..4)
                    .map(|column| {
                        let field = kinematics.k[row] * f64::from(column == photon)
                            - f64::from(row == photon) * kinematics.k[column];
                        loop_config.kappa * field * kinematics.zeta0[column]
                    })
                    .sum()
            });
            let amplitude = one_photon_amplitude(
                &kinematics.field_strength,
                lower_momentum,
                effective,
                loop_type,
                loop_config,
                &quadrature,
            )
            .unwrap();
            assert!(amplitude.momentum_delta_support);
            one_photon[photon] = amplitude.coefficient;
            two_photon[photon] = (0..16)
                .rev()
                .map(|index| {
                    let row = index / 4;
                    let column = index % 4;
                    Complex64::new(0.0, -loop_config.kappa)
                        * kinematics.field_strength[(row, column)]
                        * kinematics.zeta0[column]
                        * vacuum[(row, photon)]
                })
                .sum();
            assert!(
                (one_photon[photon] - gravitational.residual.one_photon_rhs_components[photon])
                    .norm()
                    < 1e-14
            );
            assert!(
                (two_photon[photon] - gravitational.residual.two_photon_rhs_components[photon])
                    .norm()
                    < 1e-14
            );
        }
        let rhs = std::array::from_fn(|index| one_photon[index] + two_photon[index]);
        let absolute = norm(&difference(&lhs, &rhs));
        let normalized = absolute / norm(&lhs).max(norm(&one_photon)).max(norm(&two_photon));
        assert!((absolute - gravitational.residual.absolute_defect).abs() < 1e-14);
        let omit_one_photon = norm(&difference(&lhs, &two_photon));
        let omit_two_photon = norm(&difference(&lhs, &one_photon));
        assert_eq!(norm(&one_photon), 0.0);
        assert!(omit_two_photon > 1e-6);
        writeln!(report, "\n[[off_shell_gravitational]]\nloop_type = \"{loop_type:?}\"\nabsolute = {absolute:.17e}\nnormalized = {normalized:.17e}\none_photon_norm = {:.17e}\ntwo_photon_norm = {:.17e}\nomit_one_photon_absolute = {omit_one_photon:.17e}\nomit_two_photon_absolute = {omit_two_photon:.17e}\none_photon_omission_discriminates = false", norm(&one_photon), norm(&two_photon)).unwrap();

        let abscissae = [0.10, 0.05, 0.01, 0.005];
        let ladder: Vec<_> = abscissae
            .iter()
            .map(|value| fixture(Some(*value)))
            .collect();
        let retained = on_shell_combined_virtuality_ladder(
            &ladder,
            loop_type,
            loop_config,
            &quadrature,
            tolerance,
        )
        .unwrap();
        let mut combined_samples = Vec::new();
        let mut irreducible_samples = Vec::new();
        let mut external_samples = Vec::new();
        let mut refined_samples = Vec::new();
        let refined_quadrature = QuadratureConfig {
            n_u: 64,
            n_t: 128,
            ..quadrature
        };
        for point in &ladder {
            let irreducible =
                irreducible_tensor_renormalized(point, loop_type, loop_config, &quadrature)
                    .unwrap();
            let external =
                external_tensor_off_shell(point, loop_type, loop_config, &quadrature).unwrap();
            let irreducible_components = gravity_contract(&irreducible, point);
            let external_components = gravity_contract(&external, point);
            irreducible_samples.push(irreducible_components);
            external_samples.push(external_components);
            combined_samples.push(std::array::from_fn(|index| {
                irreducible_components[index] + external_components[index]
            }));
            let refined_irreducible =
                irreducible_tensor_renormalized(point, loop_type, loop_config, &refined_quadrature)
                    .unwrap();
            let refined_external =
                external_tensor_off_shell(point, loop_type, loop_config, &refined_quadrature)
                    .unwrap();
            let refined_irreducible = gravity_contract(&refined_irreducible, point);
            let refined_external = gravity_contract(&refined_external, point);
            refined_samples.push(std::array::from_fn(|index| {
                refined_irreducible[index] + refined_external[index]
            }));
        }
        let linear = intercept(&abscissae[2..], &combined_samples[2..]);
        let quadratic = intercept(&abscissae[1..], &combined_samples[1..]);
        let refined = intercept(&abscissae[2..], &refined_samples[2..]);
        let previous = intercept(&abscissae[1..3], &combined_samples[1..3]);
        let linear_norm = norm(&linear);
        let quadratic_spread = norm(&difference(&quadratic, &linear));
        let quadrature_spread = norm(&difference(&refined, &linear));
        let ladder_spread = norm(&difference(&previous, &linear));
        let component_difference: Vec<_> = linear
            .iter()
            .zip(retained.extrapolated_components.iter())
            .map(|(independent, production)| independent - production)
            .collect();
        assert!(norm(&component_difference) < 1e-12);
        assert!(!retained.passes);
        assert!(linear_norm > 1e-3);
        let linear_real: Vec<_> = linear.iter().map(|value| value.re).collect();
        let linear_imaginary: Vec<_> = linear.iter().map(|value| value.im).collect();
        let omit_external = norm(&intercept(&abscissae[2..], &irreducible_samples[2..]));
        let omit_irreducible = norm(&intercept(&abscissae[2..], &external_samples[2..]));
        writeln!(report, "\n[[on_shell_limit]]\nloop_type = \"{loop_type:?}\"\nvirtualities = [0.1, 0.05, 0.01, 0.005]\nlinear_norm = {linear_norm:.17e}\nquadratic_norm = {:.17e}\nquadratic_spread = {quadratic_spread:.17e}\nquadrature_spread = {quadrature_spread:.17e}\nladder_spread = {ladder_spread:.17e}\nomit_external_norm = {omit_external:.17e}\nomit_irreducible_norm = {omit_irreducible:.17e}\npasses_original_gate = false\ncertified_zero_exclusion = false\nlinear_components_real = {linear_real:?}\nlinear_components_imaginary = {linear_imaginary:?}\nrefined_linear_norm = {:.17e}", norm(&quadratic), norm(&refined)).unwrap();
    }
    if let Ok(path) = std::env::var("WARD_INDEPENDENT_OUTPUT") {
        std::fs::write(path, &report).unwrap();
    }
    println!("{report}");
}
