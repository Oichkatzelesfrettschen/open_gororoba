//! Retain source-faithful Ruan-Fan channel and Mie oracle evidence.
//!
//! The analytical reference calculations intentionally duplicate the source
//! equations instead of calling the production assembly. The Mie reference
//! controls independently assemble homogeneous and single-cylinder boundary
//! equations before comparing production channel components.

use num_complex::Complex64;
use optics_core::{
    ChannelEvaluation, ConcentricCylinder, CylinderLayer, CylindricalPolarization,
    DimensionlessFanoChannel, FanoChannel, FanoDrudeParams, MaterialRole, SourceCouplingParameters,
    bessel_j, bessel_j_prime, evaluate_channel, evaluate_dimensionless_channel,
    evaluate_source_constraints, extract_fano_params, hankel_1, hankel_1_prime, hankel_2,
    hankel_2_prime, ruan_fan_mdm_fig4, ruan_fan_mdm_fig5, try_mie_scattering,
    try_scattering_channel,
};
use std::{env, error::Error, f64::consts::PI, fmt::Write as _, fs, path::PathBuf};

const FANO_COMPONENT_TOLERANCE: f64 = 1e-12;
const MIE_DIRECT_TOLERANCE: f64 = 1e-8;
const INTERFACE_TOLERANCE: f64 = 1e-8;
const BOUNDARY_MUTATION_TOLERANCE: f64 = 1e-10;

type EvidenceResult<T> = Result<T, Box<dyn Error>>;
type GeometryConstructor = fn(&FanoDrudeParams) -> ConcentricCylinder;
type GeometryCase = (&'static str, GeometryConstructor, Vec<f64>);

#[derive(Debug, Clone, Copy)]
struct IndependentChannelEvaluation {
    reflection: Complex64,
    scattering: Complex64,
    scattering_observable: f64,
    absorption_from_s: f64,
    absorption_from_r: f64,
    absorption_closed_form: f64,
    extinction_from_s: f64,
}

fn finite_real(value: f64) -> String {
    assert!(value.is_finite(), "non-finite evidence value");
    if value == 0.0 {
        "0.0".to_string()
    } else {
        format!("{value:.17e}")
    }
}

fn complex_defect(left: Complex64, right: Complex64) -> f64 {
    (left - right).norm()
}

fn independent_channel(channel: FanoChannel, omega: f64) -> IndependentChannelEvaluation {
    let detuning = channel.omega_0 - omega;
    let phase = Complex64::new(0.0, channel.phi).exp();
    let numerator = Complex64::new(channel.gamma_0 - channel.gamma, detuning);
    let denominator = Complex64::new(channel.gamma_0 + channel.gamma, detuning);
    let reflection = phase * numerator / denominator;
    let scattering = (reflection - Complex64::new(1.0, 0.0)) / 2.0;
    let scattering_observable = scattering.norm_sqr();
    let absorption_from_s = -(scattering.re + scattering_observable);
    let absorption_from_r = (1.0 - reflection.norm_sqr()) / 4.0;
    let absorption_closed_form = channel.gamma_0 * channel.gamma
        / ((omega - channel.omega_0).powi(2) + (channel.gamma_0 + channel.gamma).powi(2));
    let extinction_from_s = -scattering.re;
    IndependentChannelEvaluation {
        reflection,
        scattering,
        scattering_observable,
        absorption_from_s,
        absorption_from_r,
        absorption_closed_form,
        extinction_from_s,
    }
}

fn channel_component_defect(
    production: &ChannelEvaluation,
    independent: IndependentChannelEvaluation,
) -> f64 {
    let residuals = production.residuals;
    [
        complex_defect(production.amplitudes.reflection, independent.reflection),
        complex_defect(production.amplitudes.scattering, independent.scattering),
        (production.cross_sections.scattering - independent.scattering_observable).abs(),
        (residuals.absorption_from_s - independent.absorption_from_s).abs(),
        (residuals.absorption_from_r - independent.absorption_from_r).abs(),
        (residuals.absorption_closed_form - independent.absorption_closed_form).abs(),
        (residuals.extinction_from_s - independent.extinction_from_s).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn write_fano_grid(output: &mut String) -> (f64, usize) {
    let phi_values = [
        ("zero", 0.0),
        ("positive_half_pi", PI / 2.0),
        ("pi", PI),
        ("negative_half_pi", -PI / 2.0),
    ];
    let ratio_values = [0.0, 0.1, 1.0, 10.0];
    let detuning_values = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0];
    let mut maximum_defect: f64 = 0.0;
    let mut row_count = 0;

    for (phi_name, phi) in phi_values {
        for gamma_ratio in ratio_values {
            for x in detuning_values {
                let channel = FanoChannel {
                    omega_0: 1.0,
                    gamma: 1.0,
                    gamma_0: gamma_ratio,
                    phi,
                    l: 0,
                };
                let production = evaluate_dimensionless_channel(
                    &DimensionlessFanoChannel {
                        gamma_0_over_gamma: gamma_ratio,
                        phi,
                    },
                    x,
                )
                .expect("frozen analytical grid input is valid");
                let independent = independent_channel(channel, 1.0 + x);
                let residuals = production.residuals;
                let defect = channel_component_defect(&production, independent);
                maximum_defect = maximum_defect.max(defect);
                row_count += 1;
                writeln!(
                    output,
                    "[[fano_row]]\nphi_name = \"{phi_name}\"\nphi = {}\ngamma0_over_gamma = {}\nx = {}\nproduction_r_re = {}\nproduction_r_im = {}\nreference_r_re = {}\nreference_r_im = {}\nproduction_s_re = {}\nproduction_s_im = {}\nreference_s_re = {}\nreference_s_im = {}\nproduction_c_sct = {}\nreference_c_sct = {}\nproduction_absorption_from_s = {}\nreference_absorption_from_s = {}\nproduction_absorption_from_r = {}\nreference_absorption_from_r = {}\nproduction_absorption_closed_form = {}\nreference_absorption_closed_form = {}\nproduction_extinction = {}\nreference_extinction = {}\nproduction_balance_defect = {}\nproduction_absorption_representation_defect = {}\nproduction_flux_representation_defect = {}\ncomponent_max_defect = {}\n",
                    finite_real(phi),
                    finite_real(gamma_ratio),
                    finite_real(x),
                    finite_real(production.amplitudes.reflection.re),
                    finite_real(production.amplitudes.reflection.im),
                    finite_real(independent.reflection.re),
                    finite_real(independent.reflection.im),
                    finite_real(production.amplitudes.scattering.re),
                    finite_real(production.amplitudes.scattering.im),
                    finite_real(independent.scattering.re),
                    finite_real(independent.scattering.im),
                    finite_real(production.cross_sections.scattering),
                    finite_real(independent.scattering_observable),
                    finite_real(residuals.absorption_from_s),
                    finite_real(independent.absorption_from_s),
                    finite_real(residuals.absorption_from_r),
                    finite_real(independent.absorption_from_r),
                    finite_real(residuals.absorption_closed_form),
                    finite_real(independent.absorption_closed_form),
                    finite_real(residuals.extinction_from_s),
                    finite_real(independent.extinction_from_s),
                    finite_real(residuals.balance_defect),
                    finite_real(residuals.absorption_representation_defect),
                    finite_real(residuals.flux_representation_defect),
                    finite_real(defect),
                )
                .expect("write analytical evidence row");
            }
        }
    }
    (maximum_defect, row_count)
}

fn source_coupling(phi: f64, gamma_0: f64) -> SourceCouplingParameters {
    let eta = Complex64::from_polar((2.0_f64).sqrt(), (phi + PI) / 2.0);
    SourceCouplingParameters {
        background: Complex64::from_polar(1.0, phi),
        kappa: eta,
        eta,
        gamma: 1.0,
        omega_0: 1.0,
        gamma_0,
    }
}

fn write_source_constraints(output: &mut String) -> (f64, usize) {
    let mut maximum_defect: f64 = 0.0;
    let mut row_count = 0;
    for (phi_name, phi) in [("zero", 0.0), ("positive_half_pi", PI / 2.0), ("pi", PI)] {
        for gamma_0_over_gamma in [0.0, 0.1, 1.0] {
            let channel = FanoChannel {
                omega_0: 1.0,
                gamma: 1.0,
                gamma_0: gamma_0_over_gamma,
                phi,
                l: 0,
            };
            let residuals = evaluate_source_constraints(
                &channel,
                &source_coupling(phi, gamma_0_over_gamma),
                1.0,
            )
            .expect("frozen source coupling input is valid");
            let defect = residuals
                .eta_norm_defect
                .abs()
                .max(residuals.time_reversal_drive_defect.norm())
                .max(residuals.time_reversal_background_defect.norm())
                .max(residuals.reciprocal_coupling_defect.norm());
            maximum_defect = maximum_defect.max(defect);
            row_count += 1;
            writeln!(
                output,
                "[[source_constraint_row]]\nphi_name = \"{phi_name}\"\nphi = {}\ngamma0_over_gamma = {}\nlossless_flux_defect = {}\npassive_contractivity_excess = {}\neta_norm_defect = {}\ntime_reversal_drive_re = {}\ntime_reversal_drive_im = {}\ntime_reversal_background_re = {}\ntime_reversal_background_im = {}\nreciprocal_coupling_re = {}\nreciprocal_coupling_im = {}\nvalidity_ratio = {}\ncomponent_max_defect = {}\n",
                finite_real(phi),
                finite_real(gamma_0_over_gamma),
                finite_real(residuals.lossless_flux_defect),
                finite_real(residuals.passive_contractivity_excess),
                finite_real(residuals.eta_norm_defect),
                finite_real(residuals.time_reversal_drive_defect.re),
                finite_real(residuals.time_reversal_drive_defect.im),
                finite_real(residuals.time_reversal_background_defect.re),
                finite_real(residuals.time_reversal_background_defect.im),
                finite_real(residuals.reciprocal_coupling_defect.re),
                finite_real(residuals.reciprocal_coupling_defect.im),
                finite_real(residuals.validity_ratio),
                finite_real(defect),
            )
            .expect("write source constraint row");
        }
    }
    (maximum_defect, row_count)
}

fn write_mutation(output: &mut String, name: &str, observed: f64, threshold: f64, rationale: &str) {
    let detected = observed > threshold;
    writeln!(
        output,
        "[[mutation_control]]\nname = \"{name}\"\nobserved_defect = {}\ndetection_threshold = {}\ndetected = {detected}\nrationale = \"{rationale}\"\n",
        finite_real(observed),
        finite_real(threshold),
    )
    .expect("write mutation control");
    assert!(detected, "mutation control was not detected: {name}");
}

fn write_channel_mutations(output: &mut String) {
    let representative = FanoChannel {
        omega_0: 1.0,
        gamma: 1.0,
        gamma_0: 1.0,
        phi: PI / 2.0,
        l: 0,
    };
    let production = evaluate_channel(&representative, 2.0).expect("valid mutation fixture");
    let independent = independent_channel(representative, 2.0);
    write_mutation(
        output,
        "omit_absorption_minus_s_squared",
        (production.residuals.absorption_from_s + production.amplitudes.scattering.norm_sqr())
            .abs(),
        FANO_COMPONENT_TOLERANCE,
        "The omitted term changes the source absorption at equal damping.",
    );
    let corrupted_absorption = production.residuals.absorption_from_s + 0.1;
    write_mutation(
        output,
        "construct_extinction_from_corrupted_absorption",
        (production.cross_sections.extinction
            - (production.cross_sections.scattering + corrupted_absorption))
            .abs(),
        FANO_COMPONENT_TOLERANCE,
        "Independent extinction exposes corrupted balance bookkeeping.",
    );
    let conjugated = independent_channel(
        FanoChannel {
            phi: -representative.phi,
            ..representative
        },
        2.0,
    );
    write_mutation(
        output,
        "conjugate_background_phase",
        complex_defect(independent.scattering, conjugated.scattering),
        FANO_COMPONENT_TOLERANCE,
        "The complex phase component changes under an incorrect conjugation.",
    );
    let reversed_detuning = independent_channel(representative, 0.0);
    write_mutation(
        output,
        "reverse_detuning_sign",
        complex_defect(independent.scattering, reversed_detuning.scattering),
        FANO_COMPONENT_TOLERANCE,
        "The complex amplitude changes under an incorrect detuning orientation.",
    );
    let plus_scattering = (independent.reflection + 1.0) / 2.0;
    write_mutation(
        output,
        "use_r_plus_one_over_two",
        complex_defect(independent.scattering, plus_scattering),
        FANO_COMPONENT_TOLERANCE,
        "The source amplitude relation uses R minus one.",
    );
    let negative_ratio = evaluate_dimensionless_channel(
        &DimensionlessFanoChannel {
            gamma_0_over_gamma: -0.1,
            phi: 0.0,
        },
        0.0,
    );
    write_mutation(
        output,
        "allow_negative_intrinsic_loss",
        if negative_ratio.is_err() { 1.0 } else { 0.0 },
        0.5,
        "The passive input boundary rejects negative intrinsic loss.",
    );
    let mut coupling = source_coupling(PI / 2.0, 0.1);
    coupling.kappa += Complex64::new(0.2, 0.0);
    let kappa_residual = evaluate_source_constraints(&representative, &coupling, 2.0)
        .expect("mutated coupling remains finite")
        .reciprocal_coupling_defect
        .norm();
    write_mutation(
        output,
        "change_kappa_without_changing_eta",
        kappa_residual,
        FANO_COMPONENT_TOLERANCE,
        "Reciprocal input and output coupling remain a separate predicate.",
    );
    let mut background_mutation = source_coupling(PI / 2.0, 0.1);
    background_mutation.background = Complex64::from_polar(1.0, 0.0);
    let background_residual =
        evaluate_source_constraints(&representative, &background_mutation, 2.0)
            .expect("mutated background remains finite")
            .time_reversal_background_defect
            .norm();
    write_mutation(
        output,
        "alter_background_phase_at_fixed_modulus",
        background_residual,
        FANO_COMPONENT_TOLERANCE,
        "Time-reversal background phase is not inferred from modulus alone.",
    );
    write_mutation(
        output,
        "change_complex_phase_preserving_scalar_scattering",
        complex_defect(independent.scattering, independent.scattering.conj()),
        FANO_COMPONENT_TOLERANCE,
        "Component comparison detects a phase mutation that scalar norms miss.",
    );
}

fn direct_single_hz_reference(l: i32, omega: f64) -> Complex64 {
    let epsilon = Complex64::new(4.0, 0.0);
    let radius = 1.0;
    let k0 = Complex64::new(omega, 0.0);
    let k = k0 * epsilon.sqrt();
    let x0 = k0 * radius;
    let x = k * radius;
    let q_inside = k / epsilon;
    let q_outside = k0;
    let numerator = q_outside * bessel_j(l, x) * hankel_2_prime(l, x0)
        - q_inside * bessel_j_prime(l, x) * hankel_2(l, x0);
    let denominator = q_inside * bessel_j_prime(l, x) * hankel_1(l, x0)
        - q_outside * bessel_j(l, x) * hankel_1_prime(l, x0);
    numerator / denominator
}

fn single_dielectric_geometry() -> ConcentricCylinder {
    ConcentricCylinder {
        layers: vec![CylinderLayer::nonmagnetic(
            1.0,
            Complex64::new(4.0, 0.0),
            MaterialRole::Dielectric,
        )],
        eps_ext: Complex64::new(1.0, 0.0),
        mu_ext: Complex64::new(1.0, 0.0),
        polarization: CylindricalPolarization::HzTm,
        metal_drude: None,
    }
}

fn write_mie_reference_controls(output: &mut String) -> f64 {
    let geometry = single_dielectric_geometry();
    let mut maximum_defect: f64 = 0.0;
    for l in 0..=2 {
        let production = try_scattering_channel(&geometry, l, 2.0)
            .expect("valid direct single-cylinder fixture");
        let reference = direct_single_hz_reference(l, 2.0);
        let defect = complex_defect(production.r_l, reference);
        maximum_defect = maximum_defect.max(defect);
        writeln!(
            output,
            "[[mie_reference_control]]\nname = \"direct_single_hz\"\nl = {l}\nomega = {}\nproduction_r_re = {}\nproduction_r_im = {}\nreference_r_re = {}\nreference_r_im = {}\ncomponent_defect = {}\n",
            finite_real(2.0),
            finite_real(production.r_l.re),
            finite_real(production.r_l.im),
            finite_real(reference.re),
            finite_real(reference.im),
            finite_real(defect),
        )
        .expect("write direct Mie reference row");
    }

    for l in 0..=3 {
        let homogeneous = ConcentricCylinder {
            layers: vec![CylinderLayer::nonmagnetic(
                1.0,
                Complex64::new(1.0, 0.0),
                MaterialRole::Dielectric,
            )],
            eps_ext: Complex64::new(1.0, 0.0),
            mu_ext: Complex64::new(1.0, 0.0),
            polarization: CylindricalPolarization::HzTm,
            metal_drude: None,
        };
        let result =
            try_scattering_channel(&homogeneous, l, 2.0).expect("valid homogeneous fixture");
        let defect = result.s_l.norm();
        maximum_defect = maximum_defect.max(defect);
        writeln!(
            output,
            "[[mie_reference_control]]\nname = \"homogeneous_no_scattering\"\nl = {l}\nomega = {}\nscattering_norm = {}\ninterface_residual = {}\n",
            finite_real(2.0),
            finite_real(defect),
            finite_real(result.interface_residual.max_component),
        )
        .expect("write homogeneous Mie reference row");
    }
    maximum_defect
}

fn material_role_name(role: MaterialRole) -> &'static str {
    match role {
        MaterialRole::Metal => "metal",
        MaterialRole::Dielectric => "dielectric",
        MaterialRole::Generic => "generic",
    }
}

fn write_geometry_record(output: &mut String, name: &str, geom: &ConcentricCylinder) {
    writeln!(
        output,
        "[[geometry]]\nname = \"{name}\"\npolarization = \"HzTm\"\neps_ext_re = {}\neps_ext_im = {}\n",
        finite_real(geom.eps_ext.re),
        finite_real(geom.eps_ext.im),
    )
    .expect("write geometry header");
    for (index, layer) in geom.layers.iter().enumerate() {
        writeln!(
            output,
            "[[geometry_layer]]\ngeometry = \"{name}\"\nindex = {index}\nrole = \"{}\"\nouter_radius = {}\nepsilon_re = {}\nepsilon_im = {}\n",
            material_role_name(layer.material),
            finite_real(layer.outer_radius),
            finite_real(layer.epsilon.re),
            finite_real(layer.epsilon.im),
        )
        .expect("write geometry layer");
    }
}

fn write_mie_rows(output: &mut String) -> (f64, f64, usize) {
    let drude_values = [("lossless", 0.0), ("passive", 0.001)];
    let cases: [GeometryCase; 2] = [
        ("fig4", ruan_fan_mdm_fig4, vec![0.154, 0.1552, 0.1564]),
        ("fig5", ruan_fan_mdm_fig5, vec![0.22, 0.2265, 0.233]),
    ];
    let mut maximum_balance_defect: f64 = 0.0;
    let mut maximum_interface_defect: f64 = 0.0;
    let mut row_count = 0;
    for (geometry_name, constructor, frequencies) in cases {
        let geometry = constructor(&FanoDrudeParams {
            omega_p: 1.0,
            gamma_d: 0.001,
        });
        write_geometry_record(output, geometry_name, &geometry);
        for (loss_name, gamma_d) in drude_values {
            let drude = FanoDrudeParams {
                omega_p: 1.0,
                gamma_d,
            };
            for omega in &frequencies {
                let mut run_geometry = geometry.clone();
                run_geometry.metal_drude = Some(drude);
                let result =
                    try_mie_scattering(&run_geometry, *omega, 2).expect("valid source MDM fixture");
                maximum_balance_defect =
                    maximum_balance_defect.max(result.observable_residuals.balance_defect.abs());
                maximum_interface_defect = maximum_interface_defect.max(
                    result
                        .channels
                        .iter()
                        .map(|row| row.interface_residual.max_component)
                        .fold(0.0, f64::max),
                );
                for channel in result.channels {
                    row_count += 1;
                    writeln!(
                        output,
                        "[[mie_channel_row]]\ngeometry = \"{geometry_name}\"\nloss_model = \"{loss_name}\"\ngamma_d = {}\nomega = {}\nl = {}\nr_re = {}\nr_im = {}\ns_re = {}\ns_im = {}\nabs_r_squared = {}\nc_sct = {}\nc_abs_from_s = {}\nc_abs_from_flux = {}\nc_ext = {}\nbalance_defect = {}\ninterface_field_re = {}\ninterface_field_im = {}\ninterface_flux_re = {}\ninterface_flux_im = {}\ninterface_max_component = {}\nconditioning_indicator = {}\n",
                        finite_real(gamma_d),
                        finite_real(*omega),
                        channel.l,
                        finite_real(channel.r_l.re),
                        finite_real(channel.r_l.im),
                        finite_real(channel.s_l.re),
                        finite_real(channel.s_l.im),
                        finite_real(channel.r_l.norm_sqr()),
                        finite_real(channel.cross_sections.scattering),
                        finite_real(channel.cross_sections.absorption),
                        finite_real(channel.absorption_from_flux),
                        finite_real(channel.cross_sections.extinction),
                        finite_real(channel.balance_defect),
                        finite_real(channel.interface_residual.field_defect.re),
                        finite_real(channel.interface_residual.field_defect.im),
                        finite_real(channel.interface_residual.flux_defect.re),
                        finite_real(channel.interface_residual.flux_defect.im),
                        finite_real(channel.interface_residual.max_component),
                        finite_real(channel.conditioning_indicator),
                    )
                    .expect("write Mie channel row");
                }
                writeln!(
                    output,
                    "[[mie_total_row]]\ngeometry = \"{geometry_name}\"\nloss_model = \"{loss_name}\"\ngamma_d = {}\nomega = {}\nc_sct = {}\nc_abs = {}\nc_ext = {}\nbalance_defect = {}\nabsorption_representation_defect = {}\nflux_representation_defect = {}\n",
                    finite_real(gamma_d),
                    finite_real(*omega),
                    finite_real(result.cross_sections.c_sct),
                    finite_real(result.cross_sections.c_abs),
                    finite_real(result.cross_sections.c_ext),
                    finite_real(result.observable_residuals.balance_defect),
                    finite_real(result.observable_residuals.absorption_representation_defect),
                    finite_real(result.observable_residuals.flux_representation_defect),
                )
                .expect("write Mie total row");
            }
        }
    }
    (maximum_balance_defect, maximum_interface_defect, row_count)
}

fn write_mie_mutations(output: &mut String) {
    let drude = FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.001,
    };
    let source_geometry = ruan_fan_mdm_fig4(&drude);
    let source =
        try_scattering_channel(&source_geometry, 0, 0.1552).expect("valid MDM mutation fixture");
    let old_absorption = -source.s_l.re;
    write_mutation(
        output,
        "mie_omit_absorption_minus_s_squared",
        (old_absorption - source.cross_sections.absorption).abs(),
        FANO_COMPONENT_TOLERANCE,
        "Mie channel absorption retains the negative |S_l|^2 term.",
    );
    let old_extinction = source.cross_sections.scattering + old_absorption;
    write_mutation(
        output,
        "mie_extinction_bookkeeping_identity",
        (source.cross_sections.extinction - old_extinction).abs(),
        FANO_COMPONENT_TOLERANCE,
        "Independent extinction rejects the legacy constructed identity.",
    );

    let mut swapped_geometry = source_geometry.clone();
    swapped_geometry.layers[0].material = MaterialRole::Dielectric;
    swapped_geometry.layers[0].epsilon = Complex64::new(12.96, 0.0);
    swapped_geometry.layers[1].material = MaterialRole::Metal;
    let swapped = try_scattering_channel(&swapped_geometry, 0, 0.1552)
        .expect("swapped geometry remains numerically valid");
    write_mutation(
        output,
        "swap_metal_and_dielectric_region_roles",
        complex_defect(source.s_l, swapped.s_l),
        BOUNDARY_MUTATION_TOLERANCE,
        "The source figure order changes the complex response.",
    );

    let mut wrong_polarization = source_geometry.clone();
    wrong_polarization.polarization = CylindricalPolarization::EzTe;
    let te = try_scattering_channel(&wrong_polarization, 0, 0.1552)
        .expect("alternate polarization remains numerically valid");
    write_mutation(
        output,
        "use_unweighted_or_wrong_polarization_state",
        complex_defect(source.s_l, te.s_l),
        BOUNDARY_MUTATION_TOLERANCE,
        "The Hz and Ez boundary contracts are distinct.",
    );
}

fn write_heuristic_quarantine(output: &mut String) {
    let drude = FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.001,
    };
    let geometry = ruan_fan_mdm_fig4(&drude);
    let frequencies = [
        0.14, 0.14375, 0.1475, 0.15125, 0.155, 0.15875, 0.1625, 0.16625, 0.17,
    ];
    let results = frequencies
        .iter()
        .map(|omega| try_mie_scattering(&geometry, *omega, 2).expect("valid heuristic fixture"))
        .collect::<Vec<_>>();
    let extracted = extract_fano_params(&frequencies, &results, 0);
    writeln!(
        output,
        "[heuristic_extractor_quarantine]\nname = \"extract_fano_params\"\nstatus = \"characterization_only\"\nvalidating_path = false\nsource_parameter_method = \"complex roots of the boundary equation plus uniform metallic-cylinder background phase\"\nlegacy_method = \"peak, one-sided HWHM, endpoint phase\"\nquarter_range_fallback = false\ncharacterization_fixture_count = {}\nreturned_characterization_value = {}\n",
        frequencies.len(),
        extracted.is_some(),
    )
    .expect("write heuristic quarantine");
    if let Some(channel) = extracted {
        writeln!(
            output,
            "heuristic_omega_0 = {}\nheuristic_gamma = {}\nheuristic_gamma_0 = {}\nheuristic_phi = {}\n",
            finite_real(channel.omega_0),
            finite_real(channel.gamma),
            finite_real(channel.gamma_0),
            finite_real(channel.phi),
        )
        .expect("write heuristic characterization");
    }
}

fn main() -> EvidenceResult<()> {
    let mut arguments = env::args().skip(1);
    let output_path = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: p2a-channel-oracle-evidence <output.toml> <code-commit-sha>")?;
    let code_commit_sha = arguments
        .next()
        .ok_or("usage: p2a-channel-oracle-evidence <output.toml> <code-commit-sha>")?;
    if arguments.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    if code_commit_sha.len() != 40 || !code_commit_sha.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("code commit must be a 40-character hexadecimal SHA".into());
    }

    let mut output = String::new();
    writeln!(
        output,
        "format = \"p2a-ruan-fan-channel-oracle-evidence-v1\"\nscientific_status = \"oracle repair evidence; C-849 and C-850 remain unadjudicated\"\ncode_commit_sha = \"{code_commit_sha}\"\nsource_id = \"0909.3323v2\"\nsource_pdf_sha256 = \"a355dc5a9358d05e6eeae3475c4722a37fb3d521fa457e6aac474d71a06d5c9a\"\nsource_equation_ledger = \"data/output/audit/2026-08-04/ruan-fan-source-equation-figure-ledger.toml\"\nindependent_bessel_fixture = \"data/output/audit/2026-08-04/bessel-independent-fixtures.toml\"\nindependent_formula_path = \"p2a_channel_oracle_evidence::independent_channel and direct_single_hz_reference\"\nproduction_path = \"optics_core::evaluate_channel and optics_core::try_scattering_channel\"\n",
    )?;
    writeln!(
        output,
        "[grid]\nphi = [0.0, 1.5707963267948966, 3.141592653589793, -1.5707963267948966]\ngamma0_over_gamma = [0.0, 0.1, 1.0, 10.0]\nx = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]\nfrozen = true\n",
    )?;
    writeln!(
        output,
        "[tolerance_policy]\nfano_component_absolute = {}\nmie_direct_complex = {}\ninterface_component_absolute = {}\nboundary_mutation = {}\npolicy_frozen_before_final_grid = true\nbasis = \"focused analytical calibration, independent direct-cylinder comparison, and declared source-domain fixture tolerances\"\n",
        finite_real(FANO_COMPONENT_TOLERANCE),
        finite_real(MIE_DIRECT_TOLERANCE),
        finite_real(INTERFACE_TOLERANCE),
        finite_real(BOUNDARY_MUTATION_TOLERANCE),
    )?;

    let (maximum_fano_defect, fano_rows) = write_fano_grid(&mut output);
    let (maximum_constraint_defect, constraint_rows) = write_source_constraints(&mut output);
    write_channel_mutations(&mut output);
    let maximum_mie_reference_defect = write_mie_reference_controls(&mut output);
    let (maximum_mie_balance_defect, maximum_interface_defect, mie_rows) =
        write_mie_rows(&mut output);
    write_mie_mutations(&mut output);
    write_heuristic_quarantine(&mut output);

    writeln!(
        output,
        "[summary]\nfano_row_count = {fano_rows}\nsource_constraint_row_count = {constraint_rows}\nmie_channel_row_count = {mie_rows}\nmaximum_fano_reference_defect = {}\nmaximum_source_constraint_defect = {}\nmaximum_mie_direct_reference_defect = {}\nmaximum_mie_balance_defect = {}\nmaximum_interface_defect = {}\nall_mutation_controls_detected = true\nsource_parameter_fit_executed = false\nheld_out_paper_comparison_executed = false\n",
        finite_real(maximum_fano_defect),
        finite_real(maximum_constraint_defect),
        finite_real(maximum_mie_reference_defect),
        finite_real(maximum_mie_balance_defect),
        finite_real(maximum_interface_defect),
    )?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, output)?;
    println!("wrote {}", output_path.display());
    Ok(())
}
