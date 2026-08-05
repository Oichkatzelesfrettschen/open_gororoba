//! Independent dense-index reference for the source SFWM amplitudes.
//!
//! The equations are intentionally duplicated here. The test does not call
//! source constructors, mismatch assembly, amplitude assembly, or legacy
//! magnitude APIs to produce the reference values. It compares every complex
//! component and rate factor against the source-owned implementation.

use num_complex::Complex64;
use optics_core::{
    SfwmSourceParameters, SourceCoherenceAnchors, SourceMismatchAudit, SourceWavevectorMismatches,
    WavevectorMismatch, source_amplitudes, source_rates,
};
use std::f64::consts::PI;

fn reference_phase_matching(delta_k_per_um: f64, thickness_um: f64) -> Complex64 {
    let argument = delta_k_per_um * thickness_um / 2.0;
    let value = if argument == 0.0 {
        thickness_um
    } else {
        argument.sin() / (delta_k_per_um / 2.0)
    };
    Complex64::new(value, 0.0)
}

#[derive(Clone, Copy)]
struct ReferenceAmplitudeInputs {
    chi2_m_per_v: f64,
    chi3_m2_per_v2: f64,
    pump_field_squared: f64,
    n_sh: f64,
    lambda_pump_um: f64,
}

fn reference_inputs(parameters: &SfwmSourceParameters) -> ReferenceAmplitudeInputs {
    ReferenceAmplitudeInputs {
        chi2_m_per_v: parameters.chi2_m_per_v,
        chi3_m2_per_v2: parameters.chi3_m2_per_v2,
        pump_field_squared: parameters.pump_field_squared,
        n_sh: parameters.n_sh,
        lambda_pump_um: parameters.lambda_pump_um,
    }
}

fn reference_amplitudes(
    inputs: ReferenceAmplitudeInputs,
    mismatches: SourceWavevectorMismatches,
    thickness_um: f64,
) -> (Complex64, Complex64) {
    let f_sfwm = reference_phase_matching(mismatches.sfwm.value_per_um, thickness_um);
    let f_spdc = reference_phase_matching(mismatches.spdc.value_per_um, thickness_um);
    let phase_spdc = Complex64::from_polar(1.0, mismatches.spdc.value_per_um * thickness_um / 2.0);
    let phase_shg = Complex64::from_polar(1.0, mismatches.shg.value_per_um * thickness_um / 2.0);
    let lambda_sh_um = inputs.lambda_pump_um / 2.0;
    let prefactor = 2.0 * PI * inputs.chi2_m_per_v.powi(2) * inputs.pump_field_squared
        / (inputs.n_sh * lambda_sh_um * mismatches.shg.value_per_um);
    let a_cas = prefactor * phase_spdc * (phase_shg * f_sfwm - f_spdc);
    let a_dir = inputs.chi3_m2_per_v2
        * inputs.pump_field_squared
        * Complex64::from_polar(1.0, mismatches.sfwm.value_per_um * thickness_um / 2.0)
        * f_sfwm;
    (a_cas, a_dir)
}

fn assert_component_close(actual: Complex64, expected: Complex64, label: &str) {
    let tolerance = 5.0e-14 * (1.0 + expected.norm());
    assert!(
        (actual.re - expected.re).abs() <= tolerance,
        "{label} real component differs: actual={actual:?} expected={expected:?}"
    );
    assert!(
        (actual.im - expected.im).abs() <= tolerance,
        "{label} imaginary component differs: actual={actual:?} expected={expected:?}"
    );
}

fn fixture() -> (SfwmSourceParameters, SourceWavevectorMismatches) {
    let parameters = SfwmSourceParameters {
        chi2_m_per_v: 2.5e-11,
        chi3_m2_per_v2: 1.5e-20,
        pump_field_squared: 1.0,
        n_pump: 2.156,
        n_signal: 2.19,
        n_idler: 2.14,
        n_sh: 2.20,
        lambda_pump_um: 1.030,
        lambda_signal_um: 0.770,
        lambda_idler_um: 1.550,
    };
    let mismatches = SourceWavevectorMismatches {
        sfwm: WavevectorMismatch {
            value_per_um: PI / 33.3,
        },
        shg: WavevectorMismatch {
            value_per_um: PI / 3.1,
        },
        spdc: WavevectorMismatch {
            value_per_um: PI / 33.3 - PI / 3.1,
        },
    };
    (parameters, mismatches)
}

#[test]
fn independent_reference_matches_all_complex_components() {
    let (parameters, mismatches) = fixture();
    for thickness_um in [0.1, 3.7, 10.0, 25.3, 75.0] {
        let actual =
            source_amplitudes(&parameters, mismatches, thickness_um).expect("valid source fixture");
        let (expected_cas, expected_dir) =
            reference_amplitudes(reference_inputs(&parameters), mismatches, thickness_um);
        assert_component_close(actual.a_cas, expected_cas, "A_cas");
        assert_component_close(actual.a_dir, expected_dir, "A_dir");
        let rates = source_rates(actual).expect("valid source rates");
        let expected_r_cas = (4.0 / 9.0) * expected_cas.norm_sqr();
        let expected_r_dir = (9.0 / 16.0) * expected_dir.norm_sqr();
        assert!((rates.r_cas - expected_r_cas).abs() <= 5.0e-14 * (1.0 + expected_r_cas));
        assert!((rates.r_dir - expected_r_dir).abs() <= 5.0e-14 * (1.0 + expected_r_dir));
    }
}

#[test]
fn omitted_eq6_prefactor_is_detected() {
    let (parameters, mismatches) = fixture();
    let actual = source_amplitudes(&parameters, mismatches, 10.0).expect("valid source fixture");
    let (mut mutated_cas, _) =
        reference_amplitudes(reference_inputs(&parameters), mismatches, 10.0);
    let omitted_prefactor = 2.0 * PI
        / (parameters.n_sh * (parameters.lambda_pump_um / 2.0) * mismatches.shg.value_per_um);
    mutated_cas /= omitted_prefactor;
    assert!((actual.a_cas - mutated_cas).norm() > 1.0e-25);
}

#[test]
fn positive_only_spdc_mismatch_is_detected() {
    let (parameters, mismatches) = fixture();
    let actual = source_amplitudes(&parameters, mismatches, 10.0).expect("valid source fixture");
    let mut positive_only_mismatches = mismatches;
    positive_only_mismatches.spdc.value_per_um = positive_only_mismatches.spdc.value_per_um.abs();
    let (mutated_cas, _) = reference_amplitudes(
        reference_inputs(&parameters),
        positive_only_mismatches,
        10.0,
    );
    assert!((actual.a_cas - mutated_cas).norm() > 1.0e-25);
}

#[test]
fn source_susceptibility_substitution_is_detected() {
    let (parameters, mismatches) = fixture();
    let actual = source_amplitudes(&parameters, mismatches, 10.0).expect("valid source fixture");
    let mut substituted_inputs = reference_inputs(&parameters);
    substituted_inputs.chi2_m_per_v = 27.0e-12;
    substituted_inputs.chi3_m2_per_v2 = 2.4e-21;
    let (mutated_cas, mutated_dir) = reference_amplitudes(substituted_inputs, mismatches, 10.0);
    let actual_ratio = source_rates(actual)
        .expect("valid source rates")
        .ratio_cas_to_dir
        .expect("nonzero direct rate");
    let mutated_ratio =
        (4.0 / 9.0 * mutated_cas.norm_sqr()) / (9.0 / 16.0 * mutated_dir.norm_sqr());
    assert!((actual_ratio - mutated_ratio).abs() > 1.0e-3);
}

fn assert_high_precision_component_close(actual: f64, expected: f64, label: &str) {
    let scale = expected.abs().max(1.0e-300);
    assert!(
        (actual - expected).abs() <= 5.0e-11 * scale + 1.0e-45,
        "{label} differs: actual={actual:.17e} expected={expected:.17e}"
    );
}

#[test]
fn independent_mpmath_fixtures_match_componentwise() {
    let fixture_text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/output/audit/2026-08-04/sfwm-p2c1-results/independent-mpmath-fixtures.csv"
    ));
    let extraordinary_n_sh = 2.241115644643947;
    let paper_mismatches =
        SourceMismatchAudit::from_source_anchors(SourceCoherenceAnchors::son_chekhova())
            .expect("valid paper anchors")
            .mismatches;
    let sellmeier_mismatches = SourceWavevectorMismatches {
        sfwm: WavevectorMismatch {
            value_per_um: -0.12457975988585446,
        },
        shg: WavevectorMismatch {
            value_per_um: -1.0215433250264354,
        },
        spdc: WavevectorMismatch {
            value_per_um: 0.896963565140581,
        },
    };
    for line in fixture_text.lines().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        assert_eq!(fields.len(), 12, "fixture row has the wrong column count");
        let case = fields[0];
        let thickness_um: f64 = fields[1].parse().expect("fixture thickness");
        let mismatches = match case {
            "paper-input" => paper_mismatches,
            "sellmeier-derived" => sellmeier_mismatches,
            other => panic!("unknown fixture case {other}"),
        };
        let parameters = SfwmSourceParameters {
            chi2_m_per_v: 2.5e-11,
            chi3_m2_per_v2: 1.5e-20,
            pump_field_squared: 1.0,
            n_pump: 2.1573850547172952,
            n_signal: 2.178987531046263,
            n_idler: 2.1375596497855565,
            n_sh: extraordinary_n_sh,
            lambda_pump_um: 1.030,
            lambda_signal_um: 0.770,
            lambda_idler_um: 1.550,
        };
        let actual = source_amplitudes(&parameters, mismatches, thickness_um)
            .expect("valid high-precision fixture inputs");
        assert_high_precision_component_close(
            actual.a_cas.re,
            fields[5].parse().expect("fixture cascaded real component"),
            "A_cas real",
        );
        assert_high_precision_component_close(
            actual.a_cas.im,
            fields[6]
                .parse()
                .expect("fixture cascaded imaginary component"),
            "A_cas imaginary",
        );
        assert_high_precision_component_close(
            actual.a_dir.re,
            fields[7].parse().expect("fixture direct real component"),
            "A_dir real",
        );
        assert_high_precision_component_close(
            actual.a_dir.im,
            fields[8]
                .parse()
                .expect("fixture direct imaginary component"),
            "A_dir imaginary",
        );
        let rates = source_rates(actual).expect("valid fixture rates");
        assert_high_precision_component_close(
            rates.r_cas,
            fields[9].parse().expect("fixture cascaded rate"),
            "R_cas",
        );
        assert_high_precision_component_close(
            rates.r_dir,
            fields[10].parse().expect("fixture direct rate"),
            "R_dir",
        );
        assert_high_precision_component_close(
            rates.ratio_cas_to_dir.expect("fixture ratio"),
            fields[11].parse().expect("fixture ratio"),
            "R_cas/R_dir",
        );
    }
}

#[test]
fn corrected_paper_input_reproduces_the_source_ratio_boundary() {
    let parameters = SfwmSourceParameters {
        chi2_m_per_v: 2.5e-11,
        chi3_m2_per_v2: 1.5e-20,
        pump_field_squared: 1.0,
        n_pump: 2.1573850547172952,
        n_signal: 2.178987531046263,
        n_idler: 2.1375596497855565,
        n_sh: 2.241115644643947,
        lambda_pump_um: 1.030,
        lambda_signal_um: 0.770,
        lambda_idler_um: 1.550,
    };
    let mismatches =
        SourceMismatchAudit::from_source_anchors(SourceCoherenceAnchors::son_chekhova())
            .expect("valid paper anchors")
            .mismatches;
    let amplitudes =
        source_amplitudes(&parameters, mismatches, 10.0).expect("valid paper source amplitudes");
    let ratio = source_rates(amplitudes)
        .expect("valid paper source rates")
        .ratio_cas_to_dir
        .expect("nonzero direct rate");
    assert!(ratio < 0.05, "corrected ratio is {ratio:.17e}");
    assert!(
        1.0 / ratio > 5.0,
        "direct-to-cascaded ratio is {ratio:.17e}"
    );
    assert!((ratio - 0.0477442355934858).abs() < 1.0e-14);
}

#[test]
fn corrected_phase_matching_preserves_source_fringe_structure() {
    let parameters = SfwmSourceParameters {
        chi2_m_per_v: 2.5e-11,
        chi3_m2_per_v2: 1.5e-20,
        pump_field_squared: 1.0,
        n_pump: 2.1573850547172952,
        n_signal: 2.178987531046263,
        n_idler: 2.1375596497855565,
        n_sh: 2.241115644643947,
        lambda_pump_um: 1.030,
        lambda_signal_um: 0.770,
        lambda_idler_um: 1.550,
    };
    let mismatches =
        SourceMismatchAudit::from_source_anchors(SourceCoherenceAnchors::son_chekhova())
            .expect("valid paper anchors")
            .mismatches;

    let mut previous_sfwm = 0.0;
    for index in 0..=3330 {
        let thickness_um = index as f64 * 0.01;
        let amplitudes = source_amplitudes(&parameters, mismatches, thickness_um)
            .expect("valid phase-matching point");
        let sfwm = amplitudes.f_sfwm.norm_sqr();
        assert!(
            sfwm + 1.0e-10 * (1.0 + sfwm) >= previous_sfwm,
            "SFWM phase matching is not monotonic at {thickness_um} um"
        );
        previous_sfwm = sfwm;
    }

    let mut shg_maxima = Vec::new();
    let mut values = Vec::new();
    for index in 0..=2000 {
        let thickness_um = index as f64 * 0.01;
        let amplitudes = source_amplitudes(&parameters, mismatches, thickness_um)
            .expect("valid phase-matching point");
        values.push((thickness_um, amplitudes.f_shg.norm_sqr()));
    }
    for window in values.windows(3) {
        if window[1].1 > window[0].1 && window[1].1 > window[2].1 {
            shg_maxima.push(window[1].0);
        }
    }
    assert!(shg_maxima.len() >= 2);
    assert!((shg_maxima[0] - 3.1).abs() < 0.02);
    assert!((shg_maxima[1] - shg_maxima[0] - 6.2).abs() < 0.03);
}
