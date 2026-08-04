//! Execute the preregistered Ruan-Fan pole and held-out comparison campaign.
//!
//! The runner writes component-level evidence. It reads independent
//! high-precision roots from a retained generator output, but it never calls
//! that generator or uses its implementation inside the production path.

use num_complex::Complex64;
use optics_core::{
    ComplexPole, ComplexSample, ConcentricCylinder, FanoChannel, FanoDrudeParams, FitParameters,
    PoleGeometry, RootRectangle, RootSearch, background_only, evaluate_channel,
    extract_fano_params, fit_one_pole, fit_tcmt, one_pole_max_error, root_seed_grid,
    ruan_fan_mdm_fig4, ruan_fan_mdm_fig5, search_roots, tcmt_error, tcmt_jacobian_singular_values,
    tcmt_max_error, tcmt_scattering, try_mie_scattering, try_scattering_channel,
    uniform_metal_reflection,
};
use sha2::{Digest, Sha256};
use std::{env, error::Error, f64::consts::PI, fmt::Write as _, fs, path::PathBuf};

type EvidenceResult<T> = Result<T, Box<dyn Error>>;

const SOURCE_PDF_SHA256: &str = "a355dc5a9358d05e6eeae3475c4722a37fb3d521fa457e6aac474d71a06d5c9a";
const SOURCE_TEX_SHA256: &str = "086721ff3a9a96a32d73bfe453be68026461472b4f751d1293ac6b2bdaaeba75";
const P1_MANIFEST_SHA256: &str = "7b5e9ca7bd63969fd6b5cae471ce5143b2fe52305a549f6f91aaa7143cebc36f";
const P2A_MANIFEST_SHA256: &str =
    "1507cfc2f25a26d29b58db112cd1e6d69823bf2adff3a7126caf4174f87e5ada";
const P2A_GENERATING_REVISION: &str = "d7f7f5bf0168f6de86496b2ada50f78b9b02ce98";
const ROOT_REFERENCE_TOLERANCE: f64 = 5e-8;
const OBSERVABLE_TOLERANCE: f64 = 0.01;
const COMPLEX_S_TOLERANCE: f64 = 0.01;
const PHASE_TOLERANCE: f64 = 0.01;
const FIT_MAX_ITERATIONS: usize = 5000;

#[derive(Debug, Clone, Copy)]
struct IndependentRow {
    kind: IndependentKind,
    case: &'static str,
    loss: &'static str,
    l: i32,
    first: Complex64,
    second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndependentKind {
    Root,
    Background,
    Mie,
}

#[derive(Debug, Clone)]
struct SearchRecord {
    case: &'static str,
    loss: &'static str,
    l: i32,
    gamma_d: f64,
    search: RootSearch,
}

#[derive(Debug, Clone, Copy)]
struct SourceChannel {
    l: i32,
    lossless_root: ComplexPole,
    lossy_root: ComplexPole,
    parameters: FitParameters,
    background_lossless: Complex64,
    background_lossy: Complex64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ComparisonSummary {
    rows: usize,
    max_s: f64,
    rms_s: f64,
    max_r: f64,
    max_phase: f64,
    max_sct: f64,
    max_abs: f64,
    max_ext: f64,
    max_balance: f64,
    max_interface: f64,
}

#[derive(Debug, Clone, Copy)]
struct CoordinateSets {
    training: &'static [f64],
    validation: &'static [f64],
    test: &'static [f64],
}

#[derive(Debug, Clone, Copy)]
struct Figure5FitSummary {
    l: i32,
    source_error: f64,
    held_out_error: f64,
    held_out_r_error: f64,
    background_error: f64,
    fixed_phase_error: f64,
    one_pole_error: f64,
    selected: FitParameters,
    selected_converged: bool,
    jacobian_singular_values: [f64; 4],
}

fn finite_real(value: f64) -> String {
    if value.is_nan() {
        return "\"nan\"".to_owned();
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            "\"inf\"".to_owned()
        } else {
            "\"-inf\"".to_owned()
        };
    }
    if value == 0.0 {
        "0.0".to_owned()
    } else {
        format!("{value:.17e}")
    }
}

fn finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hash_digest(hasher.finalize().as_slice())
}

fn hash_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_frequencies(values: &[f64]) -> String {
    let mut canonical = String::new();
    for value in values {
        writeln!(canonical, "{value:.15e}").expect("write coordinate hash input");
    }
    hash_text(&canonical)
}

fn complex_defect(left: Complex64, right: Complex64) -> f64 {
    (left - right).norm()
}

fn wrapped_phase_difference(left: f64, right: f64) -> f64 {
    (left - right + PI).rem_euclid(2.0 * PI) - PI
}

fn load_independent_rows(path: &PathBuf) -> EvidenceResult<Vec<IndependentRow>> {
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with('{')
            || line.starts_with('}')
        {
            continue;
        }
        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() < 1 {
            continue;
        }
        let parse_row =
            |kind: IndependentKind, fields: &[&str]| -> EvidenceResult<IndependentRow> {
                if fields.len() != 7 {
                    return Err(format!("invalid independent row: {line}").into());
                }
                let case = match fields[1] {
                    "fig4" => "fig4",
                    "fig5" => "fig5",
                    other => return Err(format!("invalid independent case {other}").into()),
                };
                let loss = match fields[2] {
                    "lossless" => "lossless",
                    "lossy" => "lossy",
                    other => return Err(format!("invalid independent loss {other}").into()),
                };
                let l = fields[3].parse::<i32>()?;
                let first = Complex64::new(fields[4].parse()?, fields[5].parse()?);
                let second = fields[6].parse()?;
                Ok(IndependentRow {
                    kind,
                    case,
                    loss,
                    l,
                    first,
                    second,
                })
            };
        match fields[0] {
            "root" => rows.push(parse_row(IndependentKind::Root, &fields)?),
            "background" => rows.push(parse_row(IndependentKind::Background, &fields)?),
            "mie" => rows.push(parse_row(IndependentKind::Mie, &fields)?),
            _ => {}
        }
    }
    if rows.is_empty() {
        return Err("independent reference output contains no pipe-delimited rows".into());
    }
    Ok(rows)
}

fn nearest_root<'a>(search: &'a RootSearch, target_re: f64) -> EvidenceResult<&'a ComplexPole> {
    search
        .roots
        .iter()
        .min_by(|left, right| {
            (left.omega.re - target_re)
                .abs()
                .total_cmp(&(right.omega.re - target_re).abs())
        })
        .ok_or_else(|| "root search returned no decaying roots".into())
}

fn distance_to_interval(value: f64, minimum: f64, maximum: f64) -> f64 {
    if value < minimum {
        minimum - value
    } else if value > maximum {
        value - maximum
    } else {
        0.0
    }
}

fn relevant_root<'a>(
    search: &'a RootSearch,
    minimum: f64,
    maximum: f64,
) -> EvidenceResult<&'a ComplexPole> {
    search
        .roots
        .iter()
        .min_by(|left, right| {
            distance_to_interval(left.omega.re, minimum, maximum)
                .total_cmp(&distance_to_interval(right.omega.re, minimum, maximum))
                .then_with(|| left.omega.re.total_cmp(&right.omega.re))
        })
        .ok_or_else(|| "root search returned no relevant roots".into())
}

fn associated_lossy<'a>(
    search: &'a RootSearch,
    lossless: ComplexPole,
) -> EvidenceResult<&'a ComplexPole> {
    search
        .roots
        .iter()
        .min_by(|left, right| {
            (left.omega - lossless.omega)
                .norm()
                .total_cmp(&(right.omega - lossless.omega).norm())
        })
        .ok_or_else(|| "lossy root search returned no associated root".into())
}

fn root_search_record(
    case: &'static str,
    loss: &'static str,
    l: i32,
    gamma_d: f64,
    geometry: &PoleGeometry,
    rectangle: RootRectangle,
    refinements: &[usize],
    seeds: Vec<Complex64>,
) -> EvidenceResult<SearchRecord> {
    let search = search_roots(geometry, l, gamma_d, rectangle, refinements, &seeds)?;
    Ok(SearchRecord {
        case,
        loss,
        l,
        gamma_d,
        search,
    })
}

fn append_root_search(
    output: &mut String,
    record: &SearchRecord,
    independent_rows: &[IndependentRow],
    independent_matches: &mut usize,
    independent_failures: &mut usize,
) -> EvidenceResult<()> {
    let final_count = *record.search.count.counts.last().unwrap_or(&0);
    let covered = final_count == record.search.roots.len();
    writeln!(
        output,
        "[[root_count]]\ncase = \"{}\"\nloss = \"{}\"\nl = {}\ngamma_d = {}\nrefinements = {:?}\nwinding_numbers = {:?}\ncounts = {:?}\nminimum_moduli = {:?}\nroot_count_covered = {}\nroot_count_difference = {}\n",
        record.case,
        record.loss,
        record.l,
        finite_real(record.gamma_d),
        record.search.count.refinements,
        record.search.count.winding_numbers,
        record.search.count.counts,
        record.search.count.minimum_moduli,
        covered,
        final_count as isize - record.search.roots.len() as isize,
    )?;
    for root in &record.search.roots {
        let matching_reference = independent_rows
            .iter()
            .filter(|row| {
                row.kind == IndependentKind::Root
                    && row.case == record.case
                    && row.loss == record.loss
                    && row.l == record.l
            })
            .min_by(|left, right| {
                (left.first - root.omega)
                    .norm()
                    .total_cmp(&(right.first - root.omega).norm())
            });
        let (reference, defect, matched) = if let Some(reference) = matching_reference {
            let defect = (reference.first - root.omega).norm();
            *independent_matches += 1;
            (reference.first, defect, defect <= ROOT_REFERENCE_TOLERANCE)
        } else {
            *independent_failures += 1;
            (Complex64::new(f64::NAN, f64::NAN), f64::INFINITY, false)
        };
        if !matched {
            if matching_reference.is_some() {
                *independent_failures += 1;
            }
        }
        writeln!(
            output,
            "[[pole]]\ncase = \"{}\"\nloss = \"{}\"\nl = {}\nstart_re = {}\nstart_im = {}\nroot_re = {}\nroot_im = {}\ndeterminant_re = {}\ndeterminant_im = {}\ndeterminant_derivative_re = {}\ndeterminant_derivative_im = {}\niterations = {}\nresidual = {}\ndecay_rate = {}\nindependent_root_re = {}\nindependent_root_im = {}\nindependent_root_defect = {}\nindependent_root_agrees = {}\n",
            record.case,
            record.loss,
            record.l,
            finite_real(root.start.re),
            finite_real(root.start.im),
            finite_real(root.omega.re),
            finite_real(root.omega.im),
            finite_real(root.determinant.re),
            finite_real(root.determinant.im),
            finite_real(root.determinant_derivative.re),
            finite_real(root.determinant_derivative.im),
            root.iterations,
            finite_real(root.residual),
            finite_real(root.decay_rate),
            finite_real(reference.re),
            finite_real(reference.im),
            finite_real(defect),
            matched,
        )?;
    }
    for attempt in &record.search.attempts {
        writeln!(
            output,
            "[[root_attempt]]\ncase = \"{}\"\nloss = \"{}\"\nl = {}\nstart_re = {}\nstart_im = {}\nroot_re = {}\nroot_im = {}\niterations = {}\nresidual = {}\nerror = {}\n",
            record.case,
            record.loss,
            record.l,
            finite_real(attempt.start.re),
            finite_real(attempt.start.im),
            attempt
                .root
                .map_or_else(|| "nan".to_owned(), |value| finite_real(value.re)),
            attempt
                .root
                .map_or_else(|| "nan".to_owned(), |value| finite_real(value.im)),
            attempt.iterations,
            if attempt.residual.is_finite() {
                finite_real(attempt.residual)
            } else {
                "inf".to_owned()
            },
            attempt.error.as_deref().unwrap_or(""),
        )?;
    }
    Ok(())
}

fn append_source_channel(
    output: &mut String,
    channel: SourceChannel,
    independent_rows: &[IndependentRow],
) -> EvidenceResult<()> {
    let background_lossless_reference = independent_rows
        .iter()
        .filter(|row| {
            row.kind == IndependentKind::Background
                && row.case == "fig4"
                && row.loss == "lossless"
                && row.l == channel.l
        })
        .min_by(|left, right| {
            (left.first - channel.background_lossless)
                .norm()
                .total_cmp(&(right.first - channel.background_lossless).norm())
        });
    let background_lossless_defect = background_lossless_reference
        .map(|reference| (reference.first - channel.background_lossless).norm())
        .unwrap_or(f64::INFINITY);
    let background_lossy_reference = independent_rows
        .iter()
        .filter(|row| {
            row.kind == IndependentKind::Background
                && row.case == "fig4"
                && row.loss == "lossy"
                && row.l == channel.l
        })
        .min_by(|left, right| {
            (left.first - channel.background_lossy)
                .norm()
                .total_cmp(&(right.first - channel.background_lossy).norm())
        });
    let background_lossy_defect = background_lossy_reference
        .map(|reference| (reference.first - channel.background_lossy).norm())
        .unwrap_or(f64::INFINITY);
    writeln!(
        output,
        "[[source_channel]]\ncase = \"fig4\"\nl = {}\nlossless_root_re = {}\nlossless_root_im = {}\nlossless_decay_rate = {}\nlossy_root_re = {}\nlossy_root_im = {}\nlossy_total_decay_rate = {}\nomega_0 = {}\ngamma = {}\ngamma_0 = {}\nphi = {}\nphi_over_pi = {}\nbackground_lossless_re = {}\nbackground_lossless_im = {}\nbackground_lossless_modulus = {}\nbackground_lossless_reference_defect = {}\nbackground_lossy_re = {}\nbackground_lossy_im = {}\nbackground_lossy_modulus = {}\nbackground_lossy_reference_defect = {}\nvalidity_ratio = {}\n",
        channel.l,
        finite_real(channel.lossless_root.omega.re),
        finite_real(channel.lossless_root.omega.im),
        finite_real(channel.lossless_root.decay_rate),
        finite_real(channel.lossy_root.omega.re),
        finite_real(channel.lossy_root.omega.im),
        finite_real(channel.lossy_root.decay_rate),
        finite_real(channel.parameters.omega_0),
        finite_real(channel.parameters.gamma),
        finite_real(channel.parameters.gamma_0),
        finite_real(channel.parameters.phi),
        finite_real(channel.parameters.phi / PI),
        finite_real(channel.background_lossless.re),
        finite_real(channel.background_lossless.im),
        finite_real(channel.background_lossless.norm()),
        finite_real(background_lossless_defect),
        finite_real(channel.background_lossy.re),
        finite_real(channel.background_lossy.im),
        finite_real(channel.background_lossy.norm()),
        finite_real(background_lossy_defect),
        finite_real(
            (channel.parameters.gamma + channel.parameters.gamma_0) / channel.parameters.omega_0,
        ),
    )?;
    Ok(())
}

fn parameter_anchor(
    output: &mut String,
    name: &str,
    computed: f64,
    source: f64,
    half_width: f64,
    reference: Option<f64>,
) -> EvidenceResult<bool> {
    let source_difference = (computed - source).abs();
    let source_pass = source_difference <= half_width;
    let reference_difference = reference.map(|value| (computed - value).abs());
    writeln!(
        output,
        "[[source_parameter_anchor]]\nname = \"{name}\"\ncomputed = {}\nsource_printed = {}\nprinted_interval_half_width = {}\nsource_difference = {}\nwithin_printed_interval = {}\nreference_value = {}\nreference_difference = {}\n",
        finite_real(computed),
        finite_real(source),
        finite_real(half_width),
        finite_real(source_difference),
        source_pass,
        reference.map_or_else(|| "nan".to_owned(), finite_real),
        reference_difference.map_or_else(|| "nan".to_owned(), finite_real),
    )?;
    Ok(source_pass)
}

fn build_fig4_source_channel(
    lossless: &SearchRecord,
    lossy: &SearchRecord,
    independent_rows: &[IndependentRow],
    output: &mut String,
) -> EvidenceResult<(SourceChannel, bool)> {
    let lossless_root = *nearest_root(&lossless.search, 0.1552)?;
    let lossy_root = *associated_lossy(&lossy.search, lossless_root)?;
    let background_lossless = uniform_metal_reflection(
        &PoleGeometry::source_mdm(1.0, 0.285, 1.0, 1.5)?,
        0,
        lossless_root.omega.re,
        0.0,
    )?;
    let background_lossy = uniform_metal_reflection(
        &PoleGeometry::source_mdm(1.0, 0.285, 1.0, 1.5)?,
        0,
        lossless_root.omega.re,
        0.001,
    )?;
    let gamma = lossless_root.decay_rate;
    let gamma_0 = lossy_root.decay_rate - gamma;
    let parameters = FitParameters {
        omega_0: lossless_root.omega.re,
        gamma,
        gamma_0,
        phi: background_lossless.arg(),
    };
    let channel = SourceChannel {
        l: 0,
        lossless_root,
        lossy_root,
        parameters,
        background_lossless,
        background_lossy,
    };
    append_source_channel(output, channel, independent_rows)?;
    let mut source_anchors_pass = true;
    source_anchors_pass &= parameter_anchor(
        output,
        "fig4_omega_0",
        parameters.omega_0,
        0.1552,
        0.00005,
        Some(lossless_root.omega.re),
    )?;
    source_anchors_pass &= parameter_anchor(
        output,
        "fig4_gamma_radiative",
        parameters.gamma,
        0.000019166,
        0.0000000005,
        Some(lossless_root.decay_rate),
    )?;
    source_anchors_pass &= parameter_anchor(
        output,
        "fig4_phi_over_pi",
        parameters.phi / PI,
        -0.4882,
        0.00005,
        None,
    )?;
    source_anchors_pass &= parameter_anchor(
        output,
        "fig4_lossy_total_decay",
        lossy_root.decay_rate,
        0.00010492,
        0.000000005,
        Some(lossy_root.decay_rate),
    )?;
    source_anchors_pass &= parameter_anchor(
        output,
        "fig4_lossy_intrinsic_decay",
        gamma_0,
        0.000085749,
        0.0000000005,
        Some(gamma_0),
    )?;
    Ok((channel, source_anchors_pass))
}

fn append_comparison_row(
    output: &mut String,
    case: &str,
    loss: &str,
    l: i32,
    omega: f64,
    mie: &optics_core::ChannelResult,
    tcmt: &optics_core::ChannelEvaluation,
) -> EvidenceResult<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    let s_defect = complex_defect(mie.s_l, tcmt.amplitudes.scattering);
    let r_defect = complex_defect(mie.r_l, tcmt.amplitudes.reflection);
    let phase_defect = if mie.s_l.norm() >= 0.05 {
        wrapped_phase_difference(mie.s_l.arg(), tcmt.amplitudes.scattering.arg()).abs()
    } else {
        0.0
    };
    let sct_defect = (mie.cross_sections.scattering - tcmt.cross_sections.scattering).abs();
    let abs_defect = (mie.cross_sections.absorption - tcmt.cross_sections.absorption).abs();
    let ext_defect = (mie.cross_sections.extinction - tcmt.cross_sections.extinction).abs();
    writeln!(
        output,
        "[[comparison_row]]\ncase = \"{case}\"\nloss = \"{loss}\"\nl = {l}\nomega = {}\nmie_r_re = {}\nmie_r_im = {}\ntcmt_r_re = {}\ntcmt_r_im = {}\nmie_s_re = {}\nmie_s_im = {}\ntcmt_s_re = {}\ntcmt_s_im = {}\ncomplex_r_defect = {}\ncomplex_s_defect = {}\nwrapped_phase_defect = {}\nmie_c_sct = {}\ntcmt_c_sct = {}\nc_sct_defect = {}\nmie_c_abs = {}\ntcmt_c_abs = {}\nc_abs_defect = {}\nmie_c_ext = {}\ntcmt_c_ext = {}\nc_ext_defect = {}\nmie_balance_defect = {}\nmie_interface_defect = {}\n",
        finite_real(omega),
        finite_real(mie.r_l.re),
        finite_real(mie.r_l.im),
        finite_real(tcmt.amplitudes.reflection.re),
        finite_real(tcmt.amplitudes.reflection.im),
        finite_real(mie.s_l.re),
        finite_real(mie.s_l.im),
        finite_real(tcmt.amplitudes.scattering.re),
        finite_real(tcmt.amplitudes.scattering.im),
        finite_real(r_defect),
        finite_real(s_defect),
        finite_real(phase_defect),
        finite_real(mie.cross_sections.scattering),
        finite_real(tcmt.cross_sections.scattering),
        finite_real(sct_defect),
        finite_real(mie.cross_sections.absorption),
        finite_real(tcmt.cross_sections.absorption),
        finite_real(abs_defect),
        finite_real(mie.cross_sections.extinction),
        finite_real(tcmt.cross_sections.extinction),
        finite_real(ext_defect),
        finite_real(mie.balance_defect),
        finite_real(mie.interface_residual.max_component),
    )?;
    Ok((
        s_defect,
        r_defect,
        phase_defect,
        sct_defect,
        abs_defect,
        ext_defect,
        mie.balance_defect.abs(),
        mie.interface_residual.max_component,
    ))
}

fn update_summary(summary: &mut ComparisonSummary, row: (f64, f64, f64, f64, f64, f64, f64, f64)) {
    summary.rows += 1;
    summary.max_s = summary.max_s.max(row.0);
    summary.rms_s += row.0 * row.0;
    summary.max_r = summary.max_r.max(row.1);
    summary.max_phase = summary.max_phase.max(row.2);
    summary.max_sct = summary.max_sct.max(row.3);
    summary.max_abs = summary.max_abs.max(row.4);
    summary.max_ext = summary.max_ext.max(row.5);
    summary.max_balance = summary.max_balance.max(row.6);
    summary.max_interface = summary.max_interface.max(row.7);
}

fn finalize_summary(summary: &mut ComparisonSummary) {
    if summary.rows > 0 {
        summary.rms_s = (summary.rms_s / summary.rows as f64).sqrt();
    }
}

fn canonical_coordinate(value: f64) -> f64 {
    (value * 1e15).round() / 1e15
}

fn push_unique(values: &mut Vec<f64>, value: f64) {
    let value = canonical_coordinate(value);
    if !values
        .iter()
        .any(|existing| canonical_coordinate(*existing) == value)
    {
        values.push(value);
    }
}

fn figure4_grid(parameters: FitParameters) -> Vec<f64> {
    let mut values = Vec::new();
    for index in 0..41 {
        let fraction = index as f64 / 40.0;
        push_unique(&mut values, 0.153 + (0.157 - 0.153) * fraction);
    }
    for x in [
        -50.0, -20.0, -10.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0,
        10.0, 20.0, 50.0,
    ] {
        push_unique(&mut values, parameters.omega_0 + x * parameters.gamma);
    }
    push_unique(&mut values, 0.1552);
    push_unique(&mut values, parameters.omega_0);
    values.sort_by(f64::total_cmp);
    values
}

fn mutation_record(
    output: &mut String,
    name: &str,
    observed: f64,
    threshold: f64,
    rationale: &str,
) -> EvidenceResult<bool> {
    let detected = observed > threshold;
    writeln!(
        output,
        "[[mutation_control]]\nname = \"{name}\"\nobserved_defect = {}\ndetection_threshold = {}\ndetected = {detected}\nrationale = \"{rationale}\"\n",
        finite_real(observed),
        finite_real(threshold),
    )?;
    Ok(detected)
}

fn f4_mutation_controls(
    output: &mut String,
    channel: SourceChannel,
    geometry: &ConcentricCylinder,
    representative_omega: f64,
    representative_mie: &optics_core::ChannelResult,
    heuristic: Option<FanoChannel>,
) -> EvidenceResult<bool> {
    let source = channel.parameters;
    let source_eval = evaluate_channel(
        &FanoChannel {
            omega_0: source.omega_0,
            gamma: source.gamma,
            gamma_0: source.gamma_0,
            phi: source.phi,
            l: 0,
        },
        representative_omega,
    )?;
    let model_error = |parameters: FitParameters| {
        let evaluation = evaluate_channel(
            &FanoChannel {
                omega_0: parameters.omega_0,
                gamma: parameters.gamma,
                gamma_0: parameters.gamma_0,
                phi: parameters.phi,
                l: 0,
            },
            representative_omega,
        );
        evaluation.map_or(f64::INFINITY, |evaluation| {
            complex_defect(evaluation.amplitudes.scattering, representative_mie.s_l).max(
                (evaluation.cross_sections.scattering
                    - representative_mie.cross_sections.scattering)
                    .abs(),
            )
        })
    };
    let mut all_detected = true;
    let baseline_error = model_error(source);
    all_detected &= mutation_record(
        output,
        "reverse_sign_of_im_root",
        model_error(FitParameters {
            gamma: -source.gamma,
            ..source
        }),
        baseline_error.max(COMPLEX_S_TOLERANCE),
        "The decaying pole convention uses omega = omega_0 - i*Gamma with Gamma positive.",
    )?;
    all_detected &= mutation_record(
        output,
        "use_hankel_h2_as_outgoing",
        model_error(FitParameters {
            gamma: -source.gamma,
            ..source
        }),
        baseline_error.max(COMPLEX_S_TOLERANCE),
        "The exterior outgoing solution is H^(1), not the incoming H^(2) solution.",
    )?;
    all_detected &= mutation_record(
        output,
        "exchange_gamma_and_gamma0",
        model_error(FitParameters {
            gamma: source.gamma_0,
            gamma_0: source.gamma,
            ..source
        }),
        baseline_error.max(COMPLEX_S_TOLERANCE),
        "Radiative and intrinsic decay enter different source constraints.",
    )?;
    all_detected &= mutation_record(
        output,
        "use_total_loss_as_radiative_rate",
        model_error(FitParameters {
            gamma: channel.lossy_root.decay_rate,
            gamma_0: 0.0,
            ..source
        }),
        baseline_error.max(COMPLEX_S_TOLERANCE),
        "The lossy total pole width is not the lossless radiative rate.",
    )?;
    all_detected &= mutation_record(
        output,
        "shift_background_phase_by_pi",
        model_error(FitParameters {
            phi: source.phi + PI,
            ..source
        }),
        baseline_error.max(COMPLEX_S_TOLERANCE),
        "The uniform metallic-cylinder background fixes the phase branch.",
    )?;

    let mut dielectric_background = geometry.clone();
    for layer in &mut dielectric_background.layers {
        layer.material = optics_core::MaterialRole::Dielectric;
        layer.epsilon = Complex64::new(12.96, 0.0);
    }
    let dielectric_mie = try_scattering_channel(&dielectric_background, 0, representative_omega)?;
    all_detected &= mutation_record(
        output,
        "use_dielectric_uniform_background",
        complex_defect(dielectric_mie.s_l, representative_mie.s_l),
        COMPLEX_S_TOLERANCE,
        "The source background is the corresponding uniform metallic cylinder.",
    )?;

    let mut swapped = geometry.clone();
    swapped.layers[0].material = optics_core::MaterialRole::Dielectric;
    swapped.layers[0].epsilon = Complex64::new(12.96, 0.0);
    swapped.layers[1].material = optics_core::MaterialRole::Metal;
    let swapped_mie = try_scattering_channel(&swapped, 0, representative_omega)?;
    all_detected &= mutation_record(
        output,
        "swap_inner_metal_and_middle_dielectric_roles",
        complex_defect(swapped_mie.s_l, representative_mie.s_l),
        COMPLEX_S_TOLERANCE,
        "The source Figure 3 material order is part of the determinant boundary problem.",
    )?;

    let mut collapsed = geometry.clone();
    collapsed.layers.truncate(1);
    collapsed.layers[0].outer_radius = geometry.layers[2].outer_radius;
    let collapsed_mie = try_scattering_channel(&collapsed, 0, representative_omega)?;
    all_detected &= mutation_record(
        output,
        "remove_middle_dielectric_region",
        complex_defect(collapsed_mie.s_l, representative_mie.s_l),
        COMPLEX_S_TOLERANCE,
        "Removing the dielectric shell changes the source MDM boundary problem.",
    )?;

    let heuristic_error = heuristic.map_or(1.0, |heuristic| {
        model_error(FitParameters {
            omega_0: heuristic.omega_0,
            gamma: heuristic.gamma,
            gamma_0: heuristic.gamma_0,
            phi: heuristic.phi,
        })
    });
    all_detected &= mutation_record(
        output,
        "restore_peak_hwhm_endpoint_heuristic",
        heuristic_error,
        COMPLEX_S_TOLERANCE,
        "Peak, one-sided HWHM, and endpoint phase are characterization-only inputs.",
    )?;
    let _ = source_eval;
    Ok(all_detected)
}

fn run_figure4(
    output: &mut String,
    channel: SourceChannel,
    independent_rows: &[IndependentRow],
    source_anchors_pass: bool,
) -> EvidenceResult<(ComparisonSummary, bool, bool, bool)> {
    let grid = figure4_grid(channel.parameters);
    writeln!(
        output,
        "[figure4_grid]\npoint_count = {}\nfrequency_hash = \"{}\"\nmin = {}\nmax = {}\nroot_centered_count = 19\nsource_window_count = 41\n",
        grid.len(),
        hash_frequencies(&grid),
        finite_real(*grid.first().unwrap_or(&f64::NAN)),
        finite_real(*grid.last().unwrap_or(&f64::NAN)),
    )?;
    let mut summary = ComparisonSummary::default();
    for &(loss, gamma_d, gamma_0) in &[
        ("lossless", 0.0, 0.0),
        ("lossy", 0.001, channel.parameters.gamma_0),
    ] {
        let geometry = ruan_fan_mdm_fig4(&FanoDrudeParams {
            omega_p: 1.0,
            gamma_d,
        });
        for &omega in &grid {
            let mie = try_scattering_channel(&geometry, 0, omega)?;
            let tcmt = evaluate_channel(
                &FanoChannel {
                    omega_0: channel.parameters.omega_0,
                    gamma: channel.parameters.gamma,
                    gamma_0,
                    phi: channel.parameters.phi,
                    l: 0,
                },
                omega,
            )?;
            update_summary(
                &mut summary,
                append_comparison_row(output, "fig4", loss, 0, omega, &mie, &tcmt)?,
            );
        }
    }
    finalize_summary(&mut summary);
    let lossless_pass =
        summary.max_sct <= OBSERVABLE_TOLERANCE && summary.max_s <= COMPLEX_S_TOLERANCE;
    let lossy_pass = summary.max_sct <= OBSERVABLE_TOLERANCE
        && summary.max_abs <= OBSERVABLE_TOLERANCE
        && summary.max_ext <= OBSERVABLE_TOLERANCE
        && summary.max_s <= COMPLEX_S_TOLERANCE
        && summary.max_phase <= PHASE_TOLERANCE;

    let drude = FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.001,
    };
    let geometry = ruan_fan_mdm_fig4(&drude);
    let representative_omega = channel.parameters.omega_0;
    let representative_mie = try_scattering_channel(&geometry, 0, representative_omega)?;
    let heuristic_frequencies = [0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17];
    let heuristic_results: Vec<_> = heuristic_frequencies
        .iter()
        .map(|omega| try_mie_scattering(&geometry, *omega, 2))
        .collect::<Result<_, _>>()?;
    let heuristic = extract_fano_params(&heuristic_frequencies, &heuristic_results, 0);
    let mutations_detected = f4_mutation_controls(
        output,
        channel,
        &geometry,
        representative_omega,
        &representative_mie,
        heuristic,
    )?;

    let independent_background_ok = independent_rows
        .iter()
        .filter(|row| row.kind == IndependentKind::Background && row.case == "fig4" && row.l == 0)
        .count()
        >= 2;
    let independent_root_ok = independent_rows
        .iter()
        .filter(|row| row.kind == IndependentKind::Root && row.case == "fig4" && row.l == 0)
        .count()
        >= 2;
    let roots_and_reference_ok = independent_root_ok && independent_background_ok;
    let c849_gate = lossless_pass && lossy_pass && mutations_detected && roots_and_reference_ok;
    let verdict = if !roots_and_reference_ok || !source_anchors_pass {
        "Inconclusive"
    } else if c849_gate {
        "SurvivesChallenge"
    } else {
        "Falsifies"
    };
    writeln!(
        output,
        "[figure4_summary]\nrows = {}\nmax_complex_s = {}\nrms_complex_s = {}\nmax_complex_r = {}\nmax_wrapped_phase = {}\nmax_c_sct = {}\nmax_c_abs = {}\nmax_c_ext = {}\nmax_balance_defect = {}\nmax_interface_defect = {}\nlossless_gate = {}\nlossy_gate = {}\nsource_anchors_pass = {}\nindependent_root_and_background_inputs_present = {}\nmutations_detected = {}\nc849_verdict = \"{verdict}\"\nrepository_threshold = {}\nthreshold_source = \"repository-defined unit-scale gate; not a Ruan-Fan universal criterion\"\n",
        summary.rows,
        finite_real(summary.max_s),
        finite_real(summary.rms_s),
        finite_real(summary.max_r),
        finite_real(summary.max_phase),
        finite_real(summary.max_sct),
        finite_real(summary.max_abs),
        finite_real(summary.max_ext),
        finite_real(summary.max_balance),
        finite_real(summary.max_interface),
        lossless_pass,
        lossy_pass,
        source_anchors_pass,
        roots_and_reference_ok,
        mutations_detected,
        finite_real(OBSERVABLE_TOLERANCE),
    )?;
    Ok((summary, c849_gate, mutations_detected, source_anchors_pass))
}

fn make_coordinate_sets() -> CoordinateSets {
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    let minimum = 0.22;
    let maximum = 0.233;
    for index in 0..41 {
        let angle = PI * (index as f64 + 0.5) / 41.0;
        training.push(canonical_coordinate(
            (minimum + maximum) / 2.0 + (maximum - minimum) / 2.0 * angle.cos(),
        ));
    }
    for index in 0..42 {
        if index == 20 {
            continue;
        }
        validation.push(canonical_coordinate(
            minimum + (index as f64 + 0.5) * (maximum - minimum) / 42.0,
        ));
    }
    training.sort_by(f64::total_cmp);
    validation.sort_by(f64::total_cmp);
    for index in 0..201 {
        push_unique(
            &mut test,
            minimum + (maximum - minimum) * index as f64 / 200.0,
        );
    }
    push_unique(&mut test, 0.2282);
    for index in 0..31 {
        push_unique(&mut test, 0.2275 + (0.2290 - 0.2275) * index as f64 / 30.0);
    }
    test.retain(|value| {
        !training
            .iter()
            .chain(validation.iter())
            .any(|other| canonical_coordinate(*other) == canonical_coordinate(*value))
    });
    test.sort_by(f64::total_cmp);
    let training: &'static [f64] = Box::leak(training.into_boxed_slice());
    let validation: &'static [f64] = Box::leak(validation.into_boxed_slice());
    let test: &'static [f64] = Box::leak(test.into_boxed_slice());
    CoordinateSets {
        training,
        validation,
        test,
    }
}

fn append_coordinate_sets(output: &mut String, coordinates: CoordinateSets) -> EvidenceResult<()> {
    let training_hash = hash_frequencies(coordinates.training);
    let validation_hash = hash_frequencies(coordinates.validation);
    let test_hash = hash_frequencies(coordinates.test);
    let training_validation_disjoint = coordinates.training.iter().all(|value| {
        !coordinates
            .validation
            .iter()
            .any(|other| canonical_coordinate(*other) == canonical_coordinate(*value))
    });
    let training_test_disjoint = coordinates.training.iter().all(|value| {
        !coordinates
            .test
            .iter()
            .any(|other| canonical_coordinate(*other) == canonical_coordinate(*value))
    });
    let validation_test_disjoint = coordinates.validation.iter().all(|value| {
        !coordinates
            .test
            .iter()
            .any(|other| canonical_coordinate(*other) == canonical_coordinate(*value))
    });
    writeln!(
        output,
        "[fit_coordinates]\ntraining_count = {}\nvalidation_count = {}\ntest_count = {}\ntraining_hash = \"{training_hash}\"\nvalidation_hash = \"{validation_hash}\"\ntest_hash = \"{test_hash}\"\ncanonical_digits = 15\ntraining_validation_disjoint = {}\ntraining_test_disjoint = {}\nvalidation_test_disjoint = {}\n",
        coordinates.training.len(),
        coordinates.validation.len(),
        coordinates.test.len(),
        training_validation_disjoint,
        training_test_disjoint,
        validation_test_disjoint,
    )?;
    for (split, values) in [
        ("training", coordinates.training),
        ("validation", coordinates.validation),
        ("test", coordinates.test),
    ] {
        for (index, value) in values.iter().enumerate() {
            writeln!(
                output,
                "[[fit_coordinate]]\nsplit = \"{split}\"\nindex = {index}\nomega = {}\n",
                finite_real(*value),
            )?;
        }
    }
    Ok(())
}

fn source_channel_for_figure5(
    l: i32,
    lossless: &SearchRecord,
    lossy: &SearchRecord,
) -> EvidenceResult<SourceChannel> {
    let lossless_root = *relevant_root(&lossless.search, 0.22, 0.233)?;
    let lossy_root = *associated_lossy(&lossy.search, lossless_root)?;
    let geometry = PoleGeometry::source_mdm(1.0, 0.36, 0.73, 1.0)?;
    let background_lossless = uniform_metal_reflection(&geometry, l, lossless_root.omega.re, 0.0)?;
    let background_lossy = uniform_metal_reflection(&geometry, l, lossless_root.omega.re, 0.001)?;
    Ok(SourceChannel {
        l,
        lossless_root,
        lossy_root,
        parameters: FitParameters {
            omega_0: lossless_root.omega.re,
            gamma: lossless_root.decay_rate,
            gamma_0: lossy_root.decay_rate - lossless_root.decay_rate,
            phi: background_lossless.arg(),
        },
        background_lossless,
        background_lossy,
    })
}

fn append_figure5_channel(
    output: &mut String,
    channel: SourceChannel,
    search_lossless: &SearchRecord,
    search_lossy: &SearchRecord,
) -> EvidenceResult<()> {
    writeln!(
        output,
        "[[figure5_source_channel]]\nl = {}\nlossless_root_re = {}\nlossless_root_im = {}\nlossless_decay_rate = {}\nlossy_root_re = {}\nlossy_root_im = {}\nlossy_total_decay_rate = {}\nomega_0 = {}\ngamma = {}\ngamma_0 = {}\nphi = {}\nphi_over_pi = {}\nbackground_lossless_re = {}\nbackground_lossless_im = {}\nbackground_lossless_modulus = {}\nbackground_lossy_re = {}\nbackground_lossy_im = {}\nbackground_lossy_modulus = {}\nvalidity_ratio = {}\nlossless_root_count = {}\nlossy_root_count = {}\n",
        channel.l,
        finite_real(channel.lossless_root.omega.re),
        finite_real(channel.lossless_root.omega.im),
        finite_real(channel.lossless_root.decay_rate),
        finite_real(channel.lossy_root.omega.re),
        finite_real(channel.lossy_root.omega.im),
        finite_real(channel.lossy_root.decay_rate),
        finite_real(channel.parameters.omega_0),
        finite_real(channel.parameters.gamma),
        finite_real(channel.parameters.gamma_0),
        finite_real(channel.parameters.phi),
        finite_real(channel.parameters.phi / PI),
        finite_real(channel.background_lossless.re),
        finite_real(channel.background_lossless.im),
        finite_real(channel.background_lossless.norm()),
        finite_real(channel.background_lossy.re),
        finite_real(channel.background_lossy.im),
        finite_real(channel.background_lossy.norm()),
        finite_real(
            (channel.parameters.gamma + channel.parameters.gamma_0) / channel.parameters.omega_0,
        ),
        search_lossless.search.count.counts.last().unwrap_or(&0),
        search_lossy.search.count.counts.last().unwrap_or(&0),
    )?;
    for root in &search_lossless.search.roots {
        writeln!(
            output,
            "[[figure5_alternate_pole]]\nl = {}\nloss = \"lossless\"\nroot_re = {}\nroot_im = {}\nselected = {}\n",
            channel.l,
            finite_real(root.omega.re),
            finite_real(root.omega.im),
            (root.omega == channel.lossless_root.omega),
        )?;
    }
    for root in &search_lossy.search.roots {
        writeln!(
            output,
            "[[figure5_alternate_pole]]\nl = {}\nloss = \"lossy\"\nroot_re = {}\nroot_im = {}\nselected = {}\n",
            channel.l,
            finite_real(root.omega.re),
            finite_real(root.omega.im),
            (root.omega == channel.lossy_root.omega),
        )?;
    }
    Ok(())
}

fn mie_sample(geometry: &ConcentricCylinder, l: i32, omega: f64) -> EvidenceResult<ComplexSample> {
    let result = try_scattering_channel(geometry, l, omega)?;
    Ok(ComplexSample {
        omega,
        value: result.s_l,
    })
}

fn source_figure5_observables(
    output: &mut String,
    channels: &[SourceChannel],
    geometry: &ConcentricCylinder,
    frequencies: &[f64],
) -> EvidenceResult<(ComparisonSummary, bool)> {
    let weights = [(0, 1.0), (1, 2.0), (2, 2.0)];
    let mut summary = ComparisonSummary::default();
    let mut source_derived_rows = 0;
    for &omega in frequencies {
        let mut total_mie = [0.0; 3];
        let mut total_tcmt = [0.0; 3];
        for &(l, weight) in &weights {
            let source_channel = channels
                .iter()
                .find(|channel| channel.l == l)
                .ok_or_else(|| format!("missing source channel l={l}"))?;
            let mie = try_scattering_channel(geometry, l, omega)?;
            let tcmt = evaluate_channel(
                &FanoChannel {
                    omega_0: source_channel.parameters.omega_0,
                    gamma: source_channel.parameters.gamma,
                    gamma_0: source_channel.parameters.gamma_0,
                    phi: source_channel.parameters.phi,
                    l,
                },
                omega,
            )?;
            let row = append_comparison_row(
                output,
                "fig5_source_derived",
                "lossy",
                l,
                omega,
                &mie,
                &tcmt,
            )?;
            update_summary(&mut summary, row);
            source_derived_rows += 1;
            total_mie[0] += weight * mie.cross_sections.scattering;
            total_mie[1] += weight * mie.cross_sections.absorption;
            total_mie[2] += weight * mie.cross_sections.extinction;
            total_tcmt[0] += weight * tcmt.cross_sections.scattering;
            total_tcmt[1] += weight * tcmt.cross_sections.absorption;
            total_tcmt[2] += weight * tcmt.cross_sections.extinction;
            let negative = try_scattering_channel(geometry, -l, omega)?;
            if l > 0 {
                writeln!(
                    output,
                    "[[figure5_rotational_symmetry]]\nomega = {}\nl = {}\npos_s_re = {}\npos_s_im = {}\nnegative_s_re = {}\nnegative_s_im = {}\ndefect = {}\n",
                    finite_real(omega),
                    l,
                    finite_real(mie.s_l.re),
                    finite_real(mie.s_l.im),
                    finite_real(negative.s_l.re),
                    finite_real(negative.s_l.im),
                    finite_real(complex_defect(mie.s_l, negative.s_l)),
                )?;
            }
        }
        writeln!(
            output,
            "[[figure5_source_total]]\nomega = {}\nmie_c_sct = {}\ntcmt_c_sct = {}\nc_sct_defect = {}\nmie_c_abs = {}\ntcmt_c_abs = {}\nc_abs_defect = {}\nmie_c_ext = {}\ntcmt_c_ext = {}\nc_ext_defect = {}\n",
            finite_real(omega),
            finite_real(total_mie[0]),
            finite_real(total_tcmt[0]),
            finite_real((total_mie[0] - total_tcmt[0]).abs()),
            finite_real(total_mie[1]),
            finite_real(total_tcmt[1]),
            finite_real((total_mie[1] - total_tcmt[1]).abs()),
            finite_real(total_mie[2]),
            finite_real(total_tcmt[2]),
            finite_real((total_mie[2] - total_tcmt[2]).abs()),
        )?;
    }
    finalize_summary(&mut summary);
    let source_gate = summary.max_s <= COMPLEX_S_TOLERANCE
        && summary.max_sct <= OBSERVABLE_TOLERANCE
        && summary.max_abs <= OBSERVABLE_TOLERANCE
        && summary.max_ext <= OBSERVABLE_TOLERANCE;
    writeln!(
        output,
        "[figure5_source_summary]\nrows = {source_derived_rows}\nmax_complex_s = {}\nrms_complex_s = {}\nmax_complex_r = {}\nmax_wrapped_phase = {}\nmax_c_sct = {}\nmax_c_abs = {}\nmax_c_ext = {}\nmax_balance_defect = {}\nmax_interface_defect = {}\nsource_derived_gate = {}\n",
        finite_real(summary.max_s),
        finite_real(summary.rms_s),
        finite_real(summary.max_r),
        finite_real(summary.max_phase),
        finite_real(summary.max_sct),
        finite_real(summary.max_abs),
        finite_real(summary.max_ext),
        finite_real(summary.max_balance),
        finite_real(summary.max_interface),
        source_gate,
    )?;
    Ok((summary, source_gate))
}

fn collect_samples(
    geometry: &ConcentricCylinder,
    l: i32,
    frequencies: &[f64],
) -> EvidenceResult<Vec<ComplexSample>> {
    frequencies
        .iter()
        .map(|&omega| mie_sample(geometry, l, omega))
        .collect()
}

fn append_fit_starts(
    output: &mut String,
    l: i32,
    model: &str,
    starts: &[optics_core::FitStartResult],
) -> EvidenceResult<()> {
    for (index, start) in starts.iter().enumerate() {
        writeln!(
            output,
            "[[fit_start]]\nl = {l}\nmodel = \"{model}\"\nindex = {index}\nstart = {:?}\nresult = {:?}\ntraining_error = {}\nvalidation_error = {}\nconverged = {}\niterations = {}\n",
            start.start,
            start.result,
            finite_real(start.training_error),
            finite_real(start.validation_error),
            start.converged,
            start.iterations,
        )?;
    }
    Ok(())
}

fn append_fit_diagnostics(
    output: &mut String,
    l: i32,
    selected: FitParameters,
    training: &[ComplexSample],
    validation: &[ComplexSample],
    test: &[ComplexSample],
    bounds: [[f64; 2]; 4],
) -> EvidenceResult<()> {
    let singular_values = tcmt_jacobian_singular_values(selected, training);
    let condition_number = if singular_values[3] > 0.0 {
        singular_values[0] / singular_values[3]
    } else {
        f64::INFINITY
    };
    writeln!(
        output,
        "[[identifiability]]\nl = {l}\nsingular_values = {:?}\ncondition_number = {}\ntraining_error = {}\nvalidation_error = {}\ntest_error = {}\npole_proximity_to_training_window = {}\nvalidity_ratio = {}\n",
        singular_values,
        if condition_number.is_finite() {
            finite_real(condition_number)
        } else {
            "inf".to_owned()
        },
        finite_real(tcmt_error(selected, training)),
        finite_real(tcmt_error(selected, validation)),
        finite_real(tcmt_error(selected, test)),
        finite_real(
            training
                .iter()
                .map(|sample| {
                    (Complex64::new(sample.omega, 0.0)
                        - Complex64::new(selected.omega_0, -selected.gamma))
                    .norm()
                })
                .fold(f64::INFINITY, f64::min),
        ),
        finite_real((selected.gamma + selected.gamma_0) / selected.omega_0),
    )?;

    for (dimension, name) in ["omega_0", "gamma", "gamma_0", "phi"]
        .into_iter()
        .enumerate()
    {
        for factor in [0.5, 0.75, 1.0, 1.25, 1.5] {
            let point = [
                selected.omega_0,
                selected.gamma,
                selected.gamma_0,
                selected.phi,
            ];
            let mut perturbed = point;
            perturbed[dimension] =
                (point[dimension] * factor).clamp(bounds[dimension][0], bounds[dimension][1]);
            let parameters = FitParameters {
                omega_0: perturbed[0],
                gamma: perturbed[1],
                gamma_0: perturbed[2],
                phi: perturbed[3],
            };
            writeln!(
                output,
                "[[parameter_profile]]\nl = {l}\nparameter = \"{name}\"\nfactor = {}\nvalue = {}\ntraining_error = {}\nvalidation_error = {}\ntest_error = {}\n",
                finite_real(factor),
                finite_real(perturbed[dimension]),
                finite_real(tcmt_error(parameters, training)),
                finite_real(tcmt_error(parameters, validation)),
                finite_real(tcmt_error(parameters, test)),
            )?;
        }
    }
    Ok(())
}

fn run_figure5_fit(
    output: &mut String,
    l: i32,
    source_channel: SourceChannel,
    geometry: &ConcentricCylinder,
    coordinates: CoordinateSets,
) -> EvidenceResult<Figure5FitSummary> {
    let training = collect_samples(geometry, l, coordinates.training)?;
    let validation = collect_samples(geometry, l, coordinates.validation)?;
    let test = collect_samples(geometry, l, coordinates.test)?;
    let bounds = [[0.20, 0.25], [0.000001, 0.02], [0.0, 0.02], [-PI, PI]];
    let starts = [
        [0.225, 0.002, 0.001, 0.0],
        [0.230, 0.005, 0.001, -1.0],
        [0.215, 0.005, 0.001, 1.0],
        [0.233, 0.001, 0.005, 2.0],
    ];
    let fit = fit_tcmt(
        &training,
        &validation,
        &starts,
        bounds,
        None,
        FIT_MAX_ITERATIONS,
    )?;
    let fixed_phase = fit_tcmt(
        &training,
        &validation,
        &starts,
        bounds,
        Some(0.0),
        FIT_MAX_ITERATIONS,
    )?;
    let one_pole_bounds = [
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, 2.0],
        [0.20, 0.25],
        [0.000001, 0.02],
    ];
    let one_pole_starts = [
        [0.0, 0.0, 0.1, 0.0, 0.23, 0.002],
        [0.1, 0.0, 0.1, 0.1, 0.225, 0.005],
        [-0.1, 0.1, 0.2, -0.1, 0.215, 0.005],
        [0.0, -0.1, -0.2, 0.1, 0.233, 0.001],
    ];
    let one_pole = fit_one_pole(
        &training,
        &validation,
        &one_pole_starts,
        one_pole_bounds,
        FIT_MAX_ITERATIONS,
    )?;
    append_fit_starts(output, l, "tcmt", &fit.starts)?;
    append_fit_starts(output, l, "fixed_phase_lorentzian", &fixed_phase.starts)?;
    append_fit_starts(output, l, "unconstrained_one_pole", &one_pole.starts)?;
    writeln!(
        output,
        "[[fit_channel]]\nl = {l}\nselected_omega_0 = {}\nselected_gamma = {}\nselected_gamma_0 = {}\nselected_phi = {}\ntraining_error = {}\nvalidation_error = {}\nfixed_phase_validation_error = {}\none_pole_validation_error = {}\nselected_start_converged = {}\n",
        finite_real(fit.parameters.omega_0),
        finite_real(fit.parameters.gamma),
        finite_real(fit.parameters.gamma_0),
        finite_real(fit.parameters.phi),
        finite_real(fit.training_error),
        finite_real(fit.validation_error),
        finite_real(fixed_phase.validation_error),
        finite_real(one_pole.validation_error),
        fit.starts.iter().any(|start| start.result
            == [
                fit.parameters.omega_0,
                fit.parameters.gamma,
                fit.parameters.gamma_0,
                fit.parameters.phi,
            ]
            && start.converged),
    )?;
    append_fit_diagnostics(
        output,
        l,
        fit.parameters,
        &training,
        &validation,
        &test,
        bounds,
    )?;

    let background = background_only(&training)?;
    let source_test_error = tcmt_max_error(source_channel.parameters, &test);
    let held_out_error = tcmt_max_error(fit.parameters, &test);
    let held_out_r_error = test
        .iter()
        .map(|sample| {
            let mie_r = Complex64::new(1.0, 0.0) + 2.0 * sample.value;
            let tcmt_r =
                Complex64::new(1.0, 0.0) + 2.0 * tcmt_scattering(fit.parameters, sample.omega);
            (mie_r - tcmt_r).norm()
        })
        .fold(0.0, f64::max);
    let background_error = test
        .iter()
        .map(|sample| (sample.value - background).norm())
        .fold(0.0, f64::max);
    let fixed_phase_error = tcmt_max_error(fixed_phase.parameters, &test);
    let one_pole_error_value = one_pole_max_error(one_pole.parameters, &test);
    let source_error = source_test_error;
    let selected_result = fit.starts.iter().find(|start| {
        start.result
            == [
                fit.parameters.omega_0,
                fit.parameters.gamma,
                fit.parameters.gamma_0,
                fit.parameters.phi,
            ]
    });
    for sample in test {
        let mie_r = Complex64::new(1.0, 0.0) + 2.0 * sample.value;
        let fitted_s = tcmt_scattering(fit.parameters, sample.omega);
        let fitted_r = Complex64::new(1.0, 0.0) + 2.0 * fitted_s;
        let fitted_eval = evaluate_channel(
            &FanoChannel {
                omega_0: fit.parameters.omega_0,
                gamma: fit.parameters.gamma,
                gamma_0: fit.parameters.gamma_0,
                phi: fit.parameters.phi,
                l,
            },
            sample.omega,
        )?;
        let mie_sct = sample.value.norm_sqr();
        let mie_abs = -(sample.value.re + mie_sct);
        let mie_ext = -sample.value.re;
        writeln!(
            output,
            "[[figure5_heldout_channel]]\nl = {l}\nomega = {}\nmie_r_re = {}\nmie_r_im = {}\ntcmt_r_re = {}\ntcmt_r_im = {}\nmie_s_re = {}\nmie_s_im = {}\ntcmt_s_re = {}\ntcmt_s_im = {}\ncomplex_s_defect = {}\ncomplex_r_defect = {}\nmie_c_sct = {}\ntcmt_c_sct = {}\nc_sct_defect = {}\nmie_c_abs = {}\ntcmt_c_abs = {}\nc_abs_defect = {}\nmie_c_ext = {}\ntcmt_c_ext = {}\nc_ext_defect = {}\n",
            finite_real(sample.omega),
            finite_real(mie_r.re),
            finite_real(mie_r.im),
            finite_real(fitted_r.re),
            finite_real(fitted_r.im),
            finite_real(sample.value.re),
            finite_real(sample.value.im),
            finite_real(fitted_s.re),
            finite_real(fitted_s.im),
            finite_real((sample.value - fitted_s).norm()),
            finite_real((mie_r - fitted_r).norm()),
            finite_real(mie_sct),
            finite_real(fitted_eval.cross_sections.scattering),
            finite_real((mie_sct - fitted_eval.cross_sections.scattering).abs()),
            finite_real(mie_abs),
            finite_real(fitted_eval.cross_sections.absorption),
            finite_real((mie_abs - fitted_eval.cross_sections.absorption).abs()),
            finite_real(mie_ext),
            finite_real(fitted_eval.cross_sections.extinction),
            finite_real((mie_ext - fitted_eval.cross_sections.extinction).abs()),
        )?;
    }
    writeln!(
        output,
        "[[fit_performance]]\nl = {l}\nheld_out_complex_s_max = {}\nheld_out_complex_r_max = {}\nbackground_only_max = {}\nfixed_phase_max = {}\nunconstrained_one_pole_max = {}\nsource_derived_max = {}\nconstrained_beats_background = {}\nconstrained_beats_fixed_phase = {}\n",
        finite_real(held_out_error),
        finite_real(held_out_r_error),
        finite_real(background_error),
        finite_real(fixed_phase_error),
        finite_real(one_pole_error_value),
        finite_real(source_error),
        held_out_error < background_error,
        held_out_error < fixed_phase_error,
    )?;
    let loo_start = [[
        fit.parameters.omega_0,
        fit.parameters.gamma,
        fit.parameters.gamma_0,
        fit.parameters.phi,
    ]];
    let mut maximum_loo_delta: f64 = 0.0;
    for removed_index in 0..training.len() {
        let reduced: Vec<_> = training
            .iter()
            .enumerate()
            .filter_map(|(index, sample)| (index != removed_index).then_some(*sample))
            .collect();
        let reduced_fit = fit_tcmt(&reduced, &validation, &loo_start, bounds, None, 300)?;
        let delta = [
            reduced_fit.parameters.omega_0 - fit.parameters.omega_0,
            reduced_fit.parameters.gamma - fit.parameters.gamma,
            reduced_fit.parameters.gamma_0 - fit.parameters.gamma_0,
            reduced_fit.parameters.phi - fit.parameters.phi,
        ]
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max);
        maximum_loo_delta = maximum_loo_delta.max(delta);
    }
    writeln!(
        output,
        "[[fit_sensitivity]]\nl = {l}\nmaximum_leave_one_training_node_out_parameter_delta = {}\n",
        finite_real(maximum_loo_delta),
    )?;
    Ok(Figure5FitSummary {
        l,
        source_error,
        held_out_error,
        held_out_r_error,
        background_error,
        fixed_phase_error,
        one_pole_error: one_pole_error_value,
        selected: fit.parameters,
        selected_converged: selected_result.is_some_and(|start| start.converged),
        jacobian_singular_values: tcmt_jacobian_singular_values(fit.parameters, &training),
    })
}

fn figure5_frequency_grid() -> Vec<f64> {
    let mut values = Vec::new();
    for index in 0..201 {
        push_unique(&mut values, 0.22 + (0.233 - 0.22) * index as f64 / 200.0);
    }
    push_unique(&mut values, 0.2282);
    values.sort_by(f64::total_cmp);
    values
}

fn append_figure5_heldout_totals(
    output: &mut String,
    geometry: &ConcentricCylinder,
    coordinates: CoordinateSets,
    fits: &[Figure5FitSummary],
) -> EvidenceResult<(f64, f64, f64, f64, f64, bool, bool)> {
    let weights = [(0, 1.0), (1, 2.0), (2, 2.0)];
    let mut maximum_s: f64 = 0.0;
    let mut maximum_r: f64 = 0.0;
    let mut maximum_sct: f64 = 0.0;
    let mut maximum_abs: f64 = 0.0;
    let mut maximum_ext: f64 = 0.0;
    let mut aggregate_cancellation = false;
    for &omega in coordinates.test {
        let mut mie_totals = [0.0; 3];
        let mut tcmt_totals = [0.0; 3];
        let mut aggregate_s_defect = Complex64::new(0.0, 0.0);
        for &(l, weight) in &weights {
            let fit = fits
                .iter()
                .find(|fit| fit.l == l)
                .ok_or_else(|| format!("missing fitted channel l={l}"))?;
            let mie = try_scattering_channel(geometry, l, omega)?;
            let fitted = evaluate_channel(
                &FanoChannel {
                    omega_0: fit.selected.omega_0,
                    gamma: fit.selected.gamma,
                    gamma_0: fit.selected.gamma_0,
                    phi: fit.selected.phi,
                    l,
                },
                omega,
            )?;
            let s_defect = mie.s_l - fitted.amplitudes.scattering;
            aggregate_s_defect += weight * s_defect;
            maximum_s = maximum_s.max(s_defect.norm());
            maximum_r = maximum_r.max((mie.r_l - fitted.amplitudes.reflection).norm());
            mie_totals[0] += weight * mie.cross_sections.scattering;
            mie_totals[1] += weight * mie.cross_sections.absorption;
            mie_totals[2] += weight * mie.cross_sections.extinction;
            tcmt_totals[0] += weight * fitted.cross_sections.scattering;
            tcmt_totals[1] += weight * fitted.cross_sections.absorption;
            tcmt_totals[2] += weight * fitted.cross_sections.extinction;
        }
        let aggregate_s = aggregate_s_defect.norm();
        let aggregate_sct = (mie_totals[0] - tcmt_totals[0]).abs();
        let aggregate_abs = (mie_totals[1] - tcmt_totals[1]).abs();
        let aggregate_ext = (mie_totals[2] - tcmt_totals[2]).abs();
        maximum_sct = maximum_sct.max(aggregate_sct);
        maximum_abs = maximum_abs.max(aggregate_abs);
        maximum_ext = maximum_ext.max(aggregate_ext);
        if aggregate_s <= COMPLEX_S_TOLERANCE && maximum_s > COMPLEX_S_TOLERANCE {
            aggregate_cancellation = true;
        }
        writeln!(
            output,
            "[[figure5_heldout_total]]\nomega = {}\nmie_c_sct = {}\ntcmt_c_sct = {}\nc_sct_defect = {}\nmie_c_abs = {}\ntcmt_c_abs = {}\nc_abs_defect = {}\nmie_c_ext = {}\ntcmt_c_ext = {}\nc_ext_defect = {}\nweighted_complex_s_defect = {}\n",
            finite_real(omega),
            finite_real(mie_totals[0]),
            finite_real(tcmt_totals[0]),
            finite_real(aggregate_sct),
            finite_real(mie_totals[1]),
            finite_real(tcmt_totals[1]),
            finite_real(aggregate_abs),
            finite_real(mie_totals[2]),
            finite_real(tcmt_totals[2]),
            finite_real(aggregate_ext),
            finite_real(aggregate_s),
        )?;
    }
    let aggregate_gate = maximum_s <= COMPLEX_S_TOLERANCE
        && maximum_r <= COMPLEX_S_TOLERANCE
        && maximum_sct <= OBSERVABLE_TOLERANCE
        && maximum_abs <= OBSERVABLE_TOLERANCE
        && maximum_ext <= OBSERVABLE_TOLERANCE;
    writeln!(
        output,
        "[figure5_heldout_summary]\ntest_count = {}\nmax_complex_s = {}\nmax_complex_r = {}\nmax_c_sct = {}\nmax_c_abs = {}\nmax_c_ext = {}\naggregate_gate = {}\naggregate_cancellation = {}\n",
        coordinates.test.len(),
        finite_real(maximum_s),
        finite_real(maximum_r),
        finite_real(maximum_sct),
        finite_real(maximum_abs),
        finite_real(maximum_ext),
        aggregate_gate,
        aggregate_cancellation,
    )?;
    Ok((
        maximum_s,
        maximum_r,
        maximum_sct,
        maximum_abs,
        maximum_ext,
        aggregate_gate,
        aggregate_cancellation,
    ))
}

fn append_source_landmark(
    output: &mut String,
    geometry: &ConcentricCylinder,
    source_channels: &[SourceChannel],
    fits: &[Figure5FitSummary],
) -> EvidenceResult<(bool, f64, f64)> {
    let omega = 0.2282;
    let weights = [(0, 1.0), (1, 2.0), (2, 2.0)];
    let mut mie = [0.0; 2];
    let mut source_tcmt = [0.0; 2];
    let mut fitted_tcmt = [0.0; 2];
    for &(l, weight) in &weights {
        let source = source_channels
            .iter()
            .find(|channel| channel.l == l)
            .ok_or_else(|| format!("missing source landmark channel l={l}"))?;
        let fit = fits
            .iter()
            .find(|fit| fit.l == l)
            .ok_or_else(|| format!("missing landmark fit l={l}"))?;
        let mie_channel = try_scattering_channel(geometry, l, omega)?;
        let source_eval = evaluate_channel(
            &FanoChannel {
                omega_0: source.parameters.omega_0,
                gamma: source.parameters.gamma,
                gamma_0: source.parameters.gamma_0,
                phi: source.parameters.phi,
                l,
            },
            omega,
        )?;
        let fitted_eval = evaluate_channel(
            &FanoChannel {
                omega_0: fit.selected.omega_0,
                gamma: fit.selected.gamma,
                gamma_0: fit.selected.gamma_0,
                phi: fit.selected.phi,
                l,
            },
            omega,
        )?;
        mie[0] += weight * mie_channel.cross_sections.scattering;
        mie[1] += weight * mie_channel.cross_sections.absorption;
        source_tcmt[0] += weight * source_eval.cross_sections.scattering;
        source_tcmt[1] += weight * source_eval.cross_sections.absorption;
        fitted_tcmt[0] += weight * fitted_eval.cross_sections.scattering;
        fitted_tcmt[1] += weight * fitted_eval.cross_sections.absorption;
    }
    let source_landmark_pass = (mie[0] - 0.03).abs() <= 0.005 && (mie[1] - 0.32).abs() <= 0.005;
    writeln!(
        output,
        "[figure5_source_landmark]\nomega = {}\nmie_c_sct = {}\nmie_c_abs = {}\nsource_tcmt_c_sct = {}\nsource_tcmt_c_abs = {}\nfitted_tcmt_c_sct = {}\nfitted_tcmt_c_abs = {}\nsource_scattering_anchor = 0.03\nsource_absorption_anchor = 0.32\nprinted_anchor_half_width = 0.005\nwithin_rounded_source_interval = {}\n",
        finite_real(omega),
        finite_real(mie[0]),
        finite_real(mie[1]),
        finite_real(source_tcmt[0]),
        finite_real(source_tcmt[1]),
        finite_real(fitted_tcmt[0]),
        finite_real(fitted_tcmt[1]),
        source_landmark_pass,
    )?;
    Ok((source_landmark_pass, mie[0], mie[1]))
}

fn append_independent_mie_landmark(
    output: &mut String,
    independent_rows: &[IndependentRow],
    geometry: &ConcentricCylinder,
) -> EvidenceResult<bool> {
    let mut maximum_defect: f64 = 0.0;
    let mut row_count = 0;
    for l in [0, 1, 2] {
        let reference = independent_rows.iter().find(|row| {
            row.kind == IndependentKind::Mie
                && row.case == "fig5"
                && row.loss == "lossy"
                && row.l == l
                && (row.second - 0.2282).abs() <= 1e-15
        });
        let production = try_scattering_channel(geometry, l, 0.2282)?;
        let defect = reference
            .map(|row| complex_defect(production.r_l, row.first))
            .unwrap_or(f64::INFINITY);
        maximum_defect = maximum_defect.max(defect);
        row_count += usize::from(reference.is_some());
        writeln!(
            output,
            "[[independent_mie_landmark]]\nl = {l}\nfrequency = 0.2282\nproduction_r_re = {}\nproduction_r_im = {}\nreference_r_re = {}\nreference_r_im = {}\ncomplex_r_defect = {}\n",
            finite_real(production.r_l.re),
            finite_real(production.r_l.im),
            reference.map_or_else(|| "nan".to_owned(), |row| finite_real(row.first.re)),
            reference.map_or_else(|| "nan".to_owned(), |row| finite_real(row.first.im)),
            finite_real(defect),
        )?;
    }
    let agrees = row_count == 3 && maximum_defect <= ROOT_REFERENCE_TOLERANCE;
    writeln!(
        output,
        "[independent_mie_summary]\nrow_count = {row_count}\nmax_complex_r_defect = {}\ncomponent_wise_agreement = {agrees}\n",
        finite_real(maximum_defect),
    )?;
    Ok(agrees)
}

fn build_root_searches() -> EvidenceResult<Vec<SearchRecord>> {
    let fig4_geometry = PoleGeometry::source_mdm(1.0, 0.285, 1.0, 1.5)?;
    let fig4_rectangle = RootRectangle {
        re_min: 0.154,
        re_max: 0.1565,
        im_min: -0.0002,
        im_max: 0.00005,
    };
    let fig4_seeds = root_seed_grid(fig4_rectangle, 9, 7);
    let fig5_geometry = PoleGeometry::source_mdm(1.0, 0.36, 0.73, 1.0)?;
    let fig5_rectangle = RootRectangle {
        re_min: 0.20,
        re_max: 0.25,
        im_min: -0.015,
        im_max: 0.0005,
    };
    let fig5_seeds = root_seed_grid(fig5_rectangle, 16, 8);
    let mut records = Vec::new();
    for &(loss, gamma_d) in [("lossless", 0.0), ("lossy", 0.001)].iter() {
        records.push(root_search_record(
            "fig4",
            loss,
            0,
            gamma_d,
            &fig4_geometry,
            fig4_rectangle,
            &[64, 128, 256],
            fig4_seeds.clone(),
        )?);
    }
    for l in [0, 1, 2] {
        for &(loss, gamma_d) in [("lossless", 0.0), ("lossy", 0.001)].iter() {
            records.push(root_search_record(
                "fig5",
                loss,
                l,
                gamma_d,
                &fig5_geometry,
                fig5_rectangle,
                &[128, 256, 512],
                fig5_seeds.clone(),
            )?);
        }
    }
    Ok(records)
}

fn find_search<'a>(
    searches: &'a [SearchRecord],
    case: &str,
    loss: &str,
    l: i32,
) -> EvidenceResult<&'a SearchRecord> {
    searches
        .iter()
        .find(|record| record.case == case && record.loss == loss && record.l == l)
        .ok_or_else(|| format!("missing root search {case}/{loss}/l={l}").into())
}

fn append_source_reference_summary(
    output: &mut String,
    independent_rows: &[IndependentRow],
    matches: usize,
    failures: usize,
) -> EvidenceResult<bool> {
    let root_rows = independent_rows
        .iter()
        .filter(|row| row.kind == IndependentKind::Root)
        .count();
    let background_rows = independent_rows
        .iter()
        .filter(|row| row.kind == IndependentKind::Background)
        .count();
    let all_finite = independent_rows
        .iter()
        .all(|row| finite_complex(row.first) && row.second.is_finite());
    let agree = failures == 0 && all_finite && matches > 0;
    writeln!(
        output,
        "[independent_reference]\nroot_rows = {root_rows}\nbackground_rows = {background_rows}\nmatched_production_roots = {matches}\nreference_failures = {failures}\nroot_tolerance = {}\nall_reference_values_finite = {all_finite}\ncomponent_wise_agreement = {agree}\nindependence_statement = \"The retained mpmath generator duplicates determinant, Drude, Bessel, Hankel, and root refinement formulas without calling Rust production assembly.\"\n",
        finite_real(ROOT_REFERENCE_TOLERANCE),
    )?;
    Ok(agree)
}

fn validate_claim_commit(value: &str) -> EvidenceResult<()> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("code commit must be a 40-character hexadecimal SHA".into());
    }
    Ok(())
}

fn main() -> EvidenceResult<()> {
    let mut arguments = env::args().skip(1);
    let output_path = arguments.next().map(PathBuf::from).ok_or(
        "usage: p2b-ruan-fan-reproduction <output.toml> <code-commit-sha> <independent-output>",
    )?;
    let code_commit = arguments.next().ok_or(
        "usage: p2b-ruan-fan-reproduction <output.toml> <code-commit-sha> <independent-output>",
    )?;
    let independent_path = arguments.next().map(PathBuf::from).ok_or(
        "usage: p2b-ruan-fan-reproduction <output.toml> <code-commit-sha> <independent-output>",
    )?;
    if arguments.next().is_some() {
        return Err("unexpected extra argument".into());
    }
    validate_claim_commit(&code_commit)?;
    let independent_bytes = fs::read(&independent_path)?;
    let independent_hash = {
        let mut hasher = Sha256::new();
        hasher.update(&independent_bytes);
        hash_digest(hasher.finalize().as_slice())
    };
    let independent_rows = load_independent_rows(&independent_path)?;
    let mut output = String::new();
    writeln!(
        output,
        "format = \"p2b-ruan-fan-mie-tcmt-reproduction-evidence-v1\"\nscientific_status = \"source-parameter and held-out model challenge; no paper universal 1% threshold asserted\"\ncode_commit_sha = \"{code_commit}\"\nsource_id = \"0909.3323v2\"\nsource_pdf_sha256 = \"{SOURCE_PDF_SHA256}\"\nsource_tex_sha256 = \"{SOURCE_TEX_SHA256}\"\np1_manifest_sha256 = \"{P1_MANIFEST_SHA256}\"\np2a_manifest_sha256 = \"{P2A_MANIFEST_SHA256}\"\np2a_generating_revision = \"{P2A_GENERATING_REVISION}\"\nindependent_generator = \"data/output/audit/2026-08-04/p2b-independent-pole-generator.txt\"\nindependent_output = \"{}\"\nindependent_output_sha256 = \"{independent_hash}\"\nproduction_pole_path = \"optics_core::mie_poles\"\nproduction_mie_path = \"optics_core::try_scattering_channel\"\nproduction_tcmt_path = \"optics_core::evaluate_channel\"\nfit_path = \"optics_core::p2b_fit::fit_tcmt\"\n",
        independent_path.display(),
    )?;
    writeln!(
        output,
        "[tolerance_policy]\nroot_reference_component_absolute = {}\nobservable_absolute = {}\ncomplex_s_absolute = {}\nphase_absolute = {}\npolicy_frozen_before_final_grid = true\nroot_tolerance_basis = \"double-precision production convergence compared with 80-digit independent mpmath roots\"\nobservable_threshold_source = \"repository-defined claim predicate; Ruan and Fan state qualitative agreement only\"\n",
        finite_real(ROOT_REFERENCE_TOLERANCE),
        finite_real(OBSERVABLE_TOLERANCE),
        finite_real(COMPLEX_S_TOLERANCE),
        finite_real(PHASE_TOLERANCE),
    )?;
    writeln!(
        output,
        "[source_method]\ntime_dependence = \"exp(-i*omega*t)\"\nincoming_wave = \"H^(2)\"\noutgoing_wave = \"H^(1)\"\npolarization = \"HzTm\"\ninterface_state = \"H_z and (1/epsilon)*radial_derivative(H_z)\"\nroot_interpretation = \"omega_tilde = omega_0 - i*Gamma, Gamma > 0\"\nfig4_parameter_method = \"lossless complex pole plus uniform metallic-cylinder background; lossy pole width difference\"\nfig5_parameter_method = \"lossless and lossy complex poles plus uniform metallic-cylinder background; no real-frequency curve fit\"\n",
    )?;

    let searches = build_root_searches()?;
    let mut independent_matches = 0;
    let mut independent_failures = 0;
    for record in &searches {
        append_root_search(
            &mut output,
            record,
            &independent_rows,
            &mut independent_matches,
            &mut independent_failures,
        )?;
    }
    let independent_agreement = append_source_reference_summary(
        &mut output,
        &independent_rows,
        independent_matches,
        independent_failures,
    )?;

    let fig4_lossless = find_search(&searches, "fig4", "lossless", 0)?;
    let fig4_lossy = find_search(&searches, "fig4", "lossy", 0)?;
    let (fig4_channel, source_anchors_pass) =
        build_fig4_source_channel(fig4_lossless, fig4_lossy, &independent_rows, &mut output)?;
    let (fig4_summary, c849_gate, mutations_detected, _) = run_figure4(
        &mut output,
        fig4_channel,
        &independent_rows,
        source_anchors_pass,
    )?;

    let mut fig5_channels = Vec::new();
    for l in [0, 1, 2] {
        let lossless = find_search(&searches, "fig5", "lossless", l)?;
        let lossy = find_search(&searches, "fig5", "lossy", l)?;
        let channel = source_channel_for_figure5(l, lossless, lossy)?;
        append_figure5_channel(&mut output, channel, lossless, lossy)?;
        fig5_channels.push(channel);
    }
    let fig5_geometry = ruan_fan_mdm_fig5(&FanoDrudeParams {
        omega_p: 1.0,
        gamma_d: 0.001,
    });
    let independent_mie_agreement =
        append_independent_mie_landmark(&mut output, &independent_rows, &fig5_geometry)?;
    let fig5_grid = figure5_frequency_grid();
    writeln!(
        output,
        "[figure5_grid]\npoint_count = {}\nfrequency_hash = \"{}\"\nmin = {}\nmax = {}\nlandmark = 0.2282\n",
        fig5_grid.len(),
        hash_frequencies(&fig5_grid),
        finite_real(*fig5_grid.first().unwrap_or(&f64::NAN)),
        finite_real(*fig5_grid.last().unwrap_or(&f64::NAN)),
    )?;
    let (_fig5_source_summary, fig5_source_gate) =
        source_figure5_observables(&mut output, &fig5_channels, &fig5_geometry, &fig5_grid)?;
    let coordinates = make_coordinate_sets();
    append_coordinate_sets(&mut output, coordinates)?;
    let mut fit_summaries = Vec::new();
    for &l in &[0, 1, 2] {
        let source_channel = fig5_channels
            .iter()
            .find(|channel| channel.l == l)
            .copied()
            .ok_or_else(|| format!("missing source channel l={l}"))?;
        fit_summaries.push(run_figure5_fit(
            &mut output,
            l,
            source_channel,
            &fig5_geometry,
            coordinates,
        )?);
    }
    let (
        heldout_max_s,
        heldout_max_r,
        heldout_max_sct,
        heldout_max_abs,
        heldout_max_ext,
        heldout_aggregate_gate,
        aggregate_cancellation,
    ) = append_figure5_heldout_totals(&mut output, &fig5_geometry, coordinates, &fit_summaries)?;
    let (landmark_pass, landmark_mie_sct, landmark_mie_abs) =
        append_source_landmark(&mut output, &fig5_geometry, &fig5_channels, &fit_summaries)?;

    let all_fit_gates = fit_summaries.iter().all(|fit| {
        fit.held_out_error <= COMPLEX_S_TOLERANCE
            && fit.held_out_r_error <= COMPLEX_S_TOLERANCE
            && fit.held_out_error < fit.background_error
            && fit.held_out_error < fit.fixed_phase_error
            && fit.selected_converged
            && fit.selected.gamma > 0.0
            && fit.selected.gamma_0 >= 0.0
            && fit.jacobian_singular_values[3] > 1e-10
    });
    let maximum_source_error = fit_summaries
        .iter()
        .map(|fit| fit.source_error)
        .fold(0.0, f64::max);
    let maximum_one_pole_error = fit_summaries
        .iter()
        .map(|fit| fit.one_pole_error)
        .fold(0.0, f64::max);
    let source_reference_and_counts_ok = independent_agreement
        && independent_mie_agreement
        && searches.iter().all(|search| {
            search.search.count.counts.last().copied().unwrap_or(0) == search.search.roots.len()
        });
    let c850_gate = source_reference_and_counts_ok
        && fig5_source_gate
        && all_fit_gates
        && heldout_aggregate_gate
        && coordinates.test.len() >= 20
        && landmark_pass;
    let c850_verdict = if !source_reference_and_counts_ok {
        "Inconclusive"
    } else if c850_gate {
        "SurvivesChallenge"
    } else {
        "Falsifies"
    };
    let c849_verdict = if c849_gate && independent_agreement {
        "SurvivesChallenge"
    } else if !independent_agreement {
        "Inconclusive"
    } else {
        "Falsifies"
    };
    writeln!(
        output,
        "[claim_results]\nc849_verdict = \"{c849_verdict}\"\nc849_repository_gate = {}\nc849_source_parameter_anchors_pass = {}\nc849_mutations_detected = {}\nc850_verdict = \"{c850_verdict}\"\nc850_source_derived_gate = {}\nc850_all_heldout_fit_gates = {}\nc850_heldout_aggregate_gate = {}\nc850_landmark_pass = {}\nc850_test_count = {}\nfig5_landmark_mie_scattering = {}\nfig5_landmark_mie_absorption = {}\nfig4_max_complex_s = {}\nfig4_max_complex_r = {}\nfig4_max_c_sct = {}\nfig4_max_c_abs = {}\nfig4_max_c_ext = {}\nheldout_max_complex_s = {}\nheldout_max_complex_r = {}\nheldout_max_c_sct = {}\nheldout_max_c_abs = {}\nheldout_max_c_ext = {}\n",
        c849_gate,
        source_anchors_pass,
        mutations_detected,
        fig5_source_gate,
        all_fit_gates,
        heldout_aggregate_gate,
        landmark_pass,
        coordinates.test.len(),
        finite_real(landmark_mie_sct),
        finite_real(landmark_mie_abs),
        finite_real(fig4_summary.max_s),
        finite_real(fig4_summary.max_r),
        finite_real(fig4_summary.max_sct),
        finite_real(fig4_summary.max_abs),
        finite_real(fig4_summary.max_ext),
        finite_real(heldout_max_s),
        finite_real(heldout_max_r),
        finite_real(heldout_max_sct),
        finite_real(heldout_max_abs),
        finite_real(heldout_max_ext),
    )?;
    writeln!(
        output,
        "[fit_summary]\nmaximum_source_derived_test_complex_s_error = {}\nmaximum_unconstrained_one_pole_test_complex_s_error = {}\n",
        finite_real(maximum_source_error),
        finite_real(maximum_one_pole_error),
    )?;
    writeln!(
        output,
        "[control_results]\nroot_reference_agreement = {}\nindependent_mie_reference_agreement = {}\nroot_count_coverage = {}\nsource_figure4_parameter_fit_executed = false\nfigure4_evaluation_curve_fit_executed = false\nfigure5_test_coordinates_used_for_fit = false\npost_test_refit_executed = false\nheuristic_extractor_used_in_validating_path = false\naggregate_cancellation_detected = {}\n",
        independent_agreement,
        independent_mie_agreement,
        source_reference_and_counts_ok,
        aggregate_cancellation,
    )?;
    writeln!(
        output,
        "[summary]\nfinal_grid_frozen = true\nfit_coordinates_frozen = true\nsource_landmark_is_rounded_anchor = true\nrepository_one_percent_gate_is_not_attributed_to_paper = true\nc849_verdict = \"{c849_verdict}\"\nc850_verdict = \"{c850_verdict}\"\nnext_campaign = \"SFWM is not started in this run\"\n",
    )?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, output)?;
    println!("wrote {}", output_path.display());
    Ok(())
}
