//! Generate the signed, complex Son-Chekhova SFWM source reproduction.
//!
//! The binary freezes its thickness coordinates before evaluating either
//! source case. It writes component-level CSV rows and a typed TOML summary.
//! The legacy SFWM API is intentionally absent from this call path.

use anyhow::{Context, Result, bail};
use clap::Parser;
use materials_core::{SellmeierParams, linbo3_extraordinary_sellmeier, linbo3_ordinary_sellmeier};
use optics_core::{
    SfwmSourceAmplitudes, SfwmSourceParameters, SfwmSourceRates, SourceCoherenceAnchors,
    SourceMismatchAudit, SourceWavevectorMismatches, evaluate_source_case,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

const SOURCE_PDF_SHA256: &str = "25c92caa576a805711fc0050291f8f15977147be927f32b4b8e3ae01112ec8e8";
const SOURCE_TEX_SHA256: &str = "9670f2b48f4efa5e5167aded959a53af34f233765aa0e37c268d0f3ba7b26d8f";
const CHI2_SOURCE: f64 = 2.5e-11;
const CHI3_SOURCE: f64 = 1.5e-20;
const PUMP_WAVELENGTH_UM: f64 = 1.030;
const SIGNAL_WAVELENGTH_UM: f64 = 0.770;
const IDLER_WAVELENGTH_UM: f64 = 1.550;
const THICKNESS_STEP_UM: f64 = 0.1;
const THICKNESS_COUNT: usize = 1001;
const THICKNESS_AT_RATIO_UM: f64 = 10.0;
const REFINEMENT_STEPS_UM: [f64; 3] = [0.1, 0.05, 0.01];

#[derive(Debug, Parser)]
#[command(name = "sfwm-source-reproduction")]
#[command(about = "Generate signed complex SFWM source-reproduction artifacts")]
struct Args {
    /// Directory for the retained P2C1 source-reproduction artifacts.
    #[arg(long, default_value = "data/output/audit/2026-08-04/sfwm-p2c1-results")]
    output_dir: PathBuf,

    /// Freeze and hash the declared grid without evaluating amplitudes.
    #[arg(long)]
    freeze_only: bool,
}

#[derive(Debug, Clone, Copy)]
struct SourceCase {
    name: &'static str,
    parameters: SfwmSourceParameters,
    mismatches: SourceWavevectorMismatches,
}

#[derive(Debug, Serialize)]
struct ComplexRecord {
    real: f64,
    imaginary: f64,
    magnitude: f64,
}

#[derive(Debug, Serialize)]
struct RateRecord {
    cascaded: f64,
    direct: f64,
    ratio: Option<f64>,
}

#[derive(Debug, Serialize)]
struct FringeRecord {
    expected_first_maximum_um: [Option<f64>; 3],
    observed_first_grid_maximum_um: [Option<f64>; 3],
    first_maximum_abs_error_um: [Option<f64>; 3],
    expected_period_um: [Option<f64>; 3],
    observed_period_um: [Option<f64>; 3],
    period_abs_error_um: [Option<f64>; 3],
}

#[derive(Debug, Serialize)]
struct GridRefinementRecord {
    step_um: f64,
    first_maximum_abs_error_um: [Option<f64>; 3],
    period_abs_error_um: [Option<f64>; 3],
}

#[derive(Debug, Serialize)]
struct CaseRecord {
    name: String,
    mismatch_per_um: [f64; 3],
    identity_defect_per_um: f64,
    normalized_identity_defect: f64,
    coherence_lengths_um: [Option<f64>; 3],
    rate_at_10_um: RateRecord,
    amplitudes_at_10_um: [ComplexRecord; 2],
    fringes: FringeRecord,
    grid_refinement: Vec<GridRefinementRecord>,
}

#[derive(Debug, Serialize)]
struct IndexRecord {
    branch: String,
    nominal_pump: f64,
    nominal_signal: f64,
    nominal_idler: f64,
    nominal_sh: f64,
    pump_wavelength_envelope: [f64; 2],
    signal_wavelength_envelope: [f64; 2],
    idler_wavelength_envelope: [f64; 2],
    sh_wavelength_envelope: [f64; 2],
    uncertainty_status: String,
}

#[derive(Debug, Serialize)]
struct UncertaintyRecord {
    chi2_half_width_m_per_v: f64,
    chi3_half_width_m2_per_v2: f64,
    chi2_interval_m_per_v: [f64; 2],
    chi3_interval_m2_per_v2: [f64; 2],
    pump_wavelength_half_width_um: f64,
    signal_wavelength_half_width_um: f64,
    idler_wavelength_half_width_um: f64,
    pump_wavelength_interval_um: [f64; 2],
    signal_wavelength_interval_um: [f64; 2],
    idler_wavelength_interval_um: [f64; 2],
    thickness_half_width_um: f64,
    thickness_interval_um: [f64; 2],
    paper_ratio_interval_at_10_um: [f64; 2],
    sellmeier_ratio_interval_at_10_um: [f64; 2],
    thickness_status: String,
    index_status: String,
    propagated_status: String,
}

#[derive(Debug, Serialize)]
struct Summary {
    schema: String,
    source_pdf_sha256: String,
    source_tex_sha256: String,
    grid_sha256: String,
    grid_count: usize,
    grid_start_um: f64,
    grid_stop_um: f64,
    grid_step_um: f64,
    grid_refinement_steps_um: [f64; 3],
    source_chi2_m_per_v: f64,
    source_chi3_m2_per_v2: f64,
    source_anchor_coherence_lengths_um: [f64; 3],
    source_anchor_derived_coherence_lengths_um: [f64; 3],
    source_anchor_defects_um: [f64; 3],
    extraordinary_indices: IndexRecord,
    ordinary_indices: IndexRecord,
    uncertainty: UncertaintyRecord,
    cases: Vec<CaseRecord>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn frozen_thickness_grid() -> Vec<f64> {
    (0..THICKNESS_COUNT)
        .map(|index| index as f64 * THICKNESS_STEP_UM)
        .collect()
}

fn freeze_grid(output_dir: &Path, grid: &[f64]) -> Result<String> {
    let grid_path = output_dir.join("thickness-grid-0-to-100um-step-0.1um.txt");
    let contents = grid
        .iter()
        .map(|thickness| format!("{thickness:.1}\n"))
        .collect::<String>();
    if grid_path.exists() {
        let existing = fs::read_to_string(&grid_path)
            .with_context(|| format!("read frozen grid {}", grid_path.display()))?;
        if existing != contents {
            bail!("frozen thickness grid differs from the preregistered grid");
        }
    } else {
        fs::write(&grid_path, contents.as_bytes())
            .with_context(|| format!("write frozen grid {}", grid_path.display()))?;
    }
    Ok(sha256_hex(contents.as_bytes()))
}

fn complex_record(value: num_complex::Complex64) -> ComplexRecord {
    ComplexRecord {
        real: value.re,
        imaginary: value.im,
        magnitude: value.norm(),
    }
}

fn rate_record(value: SfwmSourceRates) -> RateRecord {
    RateRecord {
        cascaded: value.r_cas,
        direct: value.r_dir,
        ratio: value.ratio_cas_to_dir,
    }
}

fn component_csv_row(
    thickness_um: f64,
    amplitudes: SfwmSourceAmplitudes,
    rates: SfwmSourceRates,
    mismatches: SourceWavevectorMismatches,
) -> String {
    let values = [
        thickness_um,
        mismatches.sfwm.value_per_um,
        mismatches.shg.value_per_um,
        mismatches.spdc.value_per_um,
        amplitudes.f_sfwm.re,
        amplitudes.f_sfwm.im,
        amplitudes.f_shg.re,
        amplitudes.f_shg.im,
        amplitudes.f_spdc.re,
        amplitudes.f_spdc.im,
        amplitudes.a_cas.re,
        amplitudes.a_cas.im,
        amplitudes.a_dir.re,
        amplitudes.a_dir.im,
        amplitudes.a_cas.norm_sqr(),
        amplitudes.a_dir.norm_sqr(),
        rates.r_cas,
        rates.r_dir,
    ];
    let mut row = values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>();
    row.push(
        rates
            .ratio_cas_to_dir
            .map_or_else(|| "undefined".to_string(), |value| format!("{value:.17e}")),
    );
    row.join(",")
}

fn local_maxima(rows: &[(f64, f64)]) -> Vec<f64> {
    rows.windows(3)
        .filter_map(|window| {
            if window[1].1 > window[0].1 && window[1].1 > window[2].1 {
                Some(window[1].0)
            } else {
                None
            }
        })
        .collect()
}

fn fringe_record(
    grid: &[f64],
    amplitudes: &[SfwmSourceAmplitudes],
    mismatches: SourceWavevectorMismatches,
) -> FringeRecord {
    let values = [
        amplitudes
            .iter()
            .map(|value| value.f_sfwm.norm_sqr())
            .collect::<Vec<_>>(),
        amplitudes
            .iter()
            .map(|value| value.f_shg.norm_sqr())
            .collect::<Vec<_>>(),
        amplitudes
            .iter()
            .map(|value| value.f_spdc.norm_sqr())
            .collect::<Vec<_>>(),
    ];
    let maxima = values
        .iter()
        .map(|values| {
            local_maxima(
                &grid
                    .iter()
                    .copied()
                    .zip(values.iter().copied())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let expected = [
        mismatches.sfwm.coherence_length_um(),
        mismatches.shg.coherence_length_um(),
        mismatches.spdc.coherence_length_um(),
    ];
    let periods = expected.map(|value| value.map(|coherence| 2.0 * coherence));
    let observed_periods = maxima
        .iter()
        .map(|values| {
            if values.len() >= 2 {
                Some(values[1] - values[0])
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let first_maximum_abs_error_um = std::array::from_fn(|index| {
        expected[index]
            .zip(maxima[index].first().copied())
            .map(|(expected, observed)| (expected - observed).abs())
    });
    let period_abs_error_um = std::array::from_fn(|index| {
        periods[index]
            .zip(observed_periods[index])
            .map(|(expected, observed)| (expected - observed).abs())
    });
    FringeRecord {
        expected_first_maximum_um: expected,
        observed_first_grid_maximum_um: [
            maxima[0].first().copied(),
            maxima[1].first().copied(),
            maxima[2].first().copied(),
        ],
        first_maximum_abs_error_um,
        expected_period_um: periods,
        observed_period_um: [
            observed_periods[0],
            observed_periods[1],
            observed_periods[2],
        ],
        period_abs_error_um,
    }
}

fn evaluate_case_grid(
    case: SourceCase,
    grid: &[f64],
) -> Result<(Vec<SfwmSourceAmplitudes>, Vec<SfwmSourceRates>)> {
    if grid.is_empty() {
        bail!("cannot evaluate an empty thickness grid");
    }
    let mut amplitudes = Vec::with_capacity(grid.len());
    let mut rates = Vec::with_capacity(grid.len());
    for &thickness_um in grid {
        let (amplitude, rate) =
            evaluate_source_case(&case.parameters, case.mismatches, thickness_um)
                .with_context(|| format!("evaluate {} at {thickness_um} um", case.name))?;
        amplitudes.push(amplitude);
        rates.push(rate);
    }
    Ok((amplitudes, rates))
}

fn refinement_grid(step_um: f64) -> Result<Vec<f64>> {
    if !step_um.is_finite() || step_um <= 0.0 {
        bail!("grid refinement step must be finite and positive");
    }
    let count = (100.0 / step_um).round() as usize + 1;
    Ok((0..count).map(|index| index as f64 * step_um).collect())
}

fn grid_refinement(case: SourceCase) -> Result<Vec<GridRefinementRecord>> {
    REFINEMENT_STEPS_UM
        .into_iter()
        .map(|step_um| {
            let grid = refinement_grid(step_um)?;
            let (amplitudes, _) = evaluate_case_grid(case, &grid)?;
            let fringe = fringe_record(&grid, &amplitudes, case.mismatches);
            Ok(GridRefinementRecord {
                step_um,
                first_maximum_abs_error_um: fringe.first_maximum_abs_error_um,
                period_abs_error_um: fringe.period_abs_error_um,
            })
        })
        .collect()
}

fn write_case(output_dir: &Path, case: SourceCase, grid: &[f64]) -> Result<CaseRecord> {
    let (amplitudes, rates) = evaluate_case_grid(case, grid)?;
    let mut rows = Vec::with_capacity(grid.len());
    for ((&thickness_um, &amplitude), &rate) in grid.iter().zip(&amplitudes).zip(&rates) {
        rows.push(component_csv_row(
            thickness_um,
            amplitude,
            rate,
            case.mismatches,
        ));
    }
    let csv_header = "thickness_um,dk_sfwm_per_um,dk_shg_per_um,dk_spdc_per_um,f_sfwm_re,f_sfwm_im,f_shg_re,f_shg_im,f_spdc_re,f_spdc_im,a_cas_re,a_cas_im,a_dir_re,a_dir_im,a_cas_abs_sq,a_dir_abs_sq,r_cas,r_dir,ratio_cas_to_dir";
    let csv_contents = format!("{}\n{}\n", csv_header, rows.join("\n"));
    let csv_path = output_dir.join(format!("{}-thickness-components.csv", case.name));
    fs::write(&csv_path, csv_contents)
        .with_context(|| format!("write component output {}", csv_path.display()))?;

    let ratio_index = grid
        .iter()
        .position(|thickness| *thickness == THICKNESS_AT_RATIO_UM)
        .context("frozen grid does not contain the 10 um ratio point")?;
    let amplitude_at_ratio = amplitudes[ratio_index];
    let rate_at_ratio = rates[ratio_index];
    let coherence_lengths = [
        case.mismatches.sfwm.coherence_length_um(),
        case.mismatches.shg.coherence_length_um(),
        case.mismatches.spdc.coherence_length_um(),
    ];
    Ok(CaseRecord {
        name: case.name.to_string(),
        mismatch_per_um: [
            case.mismatches.sfwm.value_per_um,
            case.mismatches.shg.value_per_um,
            case.mismatches.spdc.value_per_um,
        ],
        identity_defect_per_um: case.mismatches.identity_defect_per_um(),
        normalized_identity_defect: case.mismatches.normalized_identity_defect(),
        coherence_lengths_um: coherence_lengths,
        rate_at_10_um: rate_record(rate_at_ratio),
        amplitudes_at_10_um: [
            complex_record(amplitude_at_ratio.a_cas),
            complex_record(amplitude_at_ratio.a_dir),
        ],
        fringes: fringe_record(grid, &amplitudes, case.mismatches),
        grid_refinement: grid_refinement(case)?,
    })
}

fn index_envelope(sellmeier: &SellmeierParams, center_um: f64, half_width_um: f64) -> [f64; 2] {
    let values = [
        sellmeier.refractive_index(center_um - half_width_um),
        sellmeier.refractive_index(center_um),
        sellmeier.refractive_index(center_um + half_width_um),
    ];
    [
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ]
}

fn index_record(branch: &str, sellmeier: &SellmeierParams) -> IndexRecord {
    IndexRecord {
        branch: branch.to_string(),
        nominal_pump: sellmeier.refractive_index(PUMP_WAVELENGTH_UM),
        nominal_signal: sellmeier.refractive_index(SIGNAL_WAVELENGTH_UM),
        nominal_idler: sellmeier.refractive_index(IDLER_WAVELENGTH_UM),
        nominal_sh: sellmeier.refractive_index(PUMP_WAVELENGTH_UM / 2.0),
        pump_wavelength_envelope: index_envelope(sellmeier, PUMP_WAVELENGTH_UM, 0.005),
        signal_wavelength_envelope: index_envelope(sellmeier, SIGNAL_WAVELENGTH_UM, 0.005),
        idler_wavelength_envelope: index_envelope(sellmeier, IDLER_WAVELENGTH_UM, 0.025),
        sh_wavelength_envelope: index_envelope(sellmeier, PUMP_WAVELENGTH_UM / 2.0, 0.0025),
        uncertainty_status: "Deterministic Sellmeier and wavelength-band envelope; no statistical index uncertainty is supplied by the source.".to_string(),
    }
}

fn ratio_at_thickness(
    parameters: SfwmSourceParameters,
    mismatches: SourceWavevectorMismatches,
) -> Result<f64> {
    let (_, rates) = evaluate_source_case(&parameters, mismatches, THICKNESS_AT_RATIO_UM)?;
    rates
        .ratio_cas_to_dir
        .context("nonzero direct rate in uncertainty corner")
}

fn interval(values: &[f64]) -> Result<[f64; 2]> {
    if values.is_empty() {
        bail!("cannot build an interval from no values");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("uncertainty interval contains a non-finite value");
    }
    Ok([
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ])
}

fn uncertainty_record(
    extraordinary: &SellmeierParams,
    paper_mismatches: SourceWavevectorMismatches,
) -> Result<UncertaintyRecord> {
    let chi2_interval = [2.45e-11, 2.55e-11];
    let chi3_interval = [1.45e-20, 1.55e-20];
    let pump_interval = [1.025, 1.035];
    let signal_interval = [0.765, 0.775];
    let idler_interval = [1.525, 1.575];
    let mut paper_ratios = Vec::new();
    let mut sellmeier_ratios = Vec::new();
    for &chi2 in &chi2_interval {
        for &chi3 in &chi3_interval {
            for &pump in &pump_interval {
                let n_sh = extraordinary.refractive_index(pump / 2.0);
                let paper_parameters = SfwmSourceParameters::new(
                    chi2,
                    chi3,
                    1.0,
                    extraordinary.refractive_index(pump),
                    extraordinary.refractive_index(SIGNAL_WAVELENGTH_UM),
                    extraordinary.refractive_index(IDLER_WAVELENGTH_UM),
                    n_sh,
                    pump,
                    SIGNAL_WAVELENGTH_UM,
                    IDLER_WAVELENGTH_UM,
                )?;
                paper_ratios.push(ratio_at_thickness(paper_parameters, paper_mismatches)?);
            }
            for &pump in &pump_interval {
                for &signal in &signal_interval {
                    for &idler in &idler_interval {
                        let sellmeier_parameters = SfwmSourceParameters::new(
                            chi2,
                            chi3,
                            1.0,
                            extraordinary.refractive_index(pump),
                            extraordinary.refractive_index(signal),
                            extraordinary.refractive_index(idler),
                            extraordinary.refractive_index(pump / 2.0),
                            pump,
                            signal,
                            idler,
                        )?;
                        let sellmeier_mismatches = sellmeier_parameters.wavevector_mismatches()?;
                        sellmeier_ratios.push(ratio_at_thickness(
                            sellmeier_parameters,
                            sellmeier_mismatches,
                        )?);
                    }
                }
            }
        }
    }
    Ok(UncertaintyRecord {
        chi2_half_width_m_per_v: 0.05e-11,
        chi3_half_width_m2_per_v2: 0.05e-20,
        chi2_interval_m_per_v: chi2_interval,
        chi3_interval_m2_per_v2: chi3_interval,
        pump_wavelength_half_width_um: 0.005,
        signal_wavelength_half_width_um: 0.005,
        idler_wavelength_half_width_um: 0.025,
        pump_wavelength_interval_um: pump_interval,
        signal_wavelength_interval_um: signal_interval,
        idler_wavelength_interval_um: idler_interval,
        thickness_half_width_um: 0.0,
        thickness_interval_um: [THICKNESS_AT_RATIO_UM, THICKNESS_AT_RATIO_UM],
        paper_ratio_interval_at_10_um: interval(&paper_ratios)?,
        sellmeier_ratio_interval_at_10_um: interval(&sellmeier_ratios)?,
        thickness_status: "No fabrication uncertainty is supplied by the public source; no spread is invented.".to_string(),
        index_status: "Branch and wavelength-band envelopes are retained; statistical index uncertainty is unavailable.".to_string(),
        propagated_status: "Deterministic endpoint corner envelope, not a statistical confidence interval.".to_string(),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create output directory {}", args.output_dir.display()))?;
    let grid = frozen_thickness_grid();
    let grid_sha256 = freeze_grid(&args.output_dir, &grid)?;
    if args.freeze_only {
        println!("grid_sha256={grid_sha256}");
        println!("grid_count={}", grid.len());
        return Ok(());
    }

    let extraordinary = linbo3_extraordinary_sellmeier();
    let ordinary = linbo3_ordinary_sellmeier();
    let n_pump = extraordinary.refractive_index(PUMP_WAVELENGTH_UM);
    let n_signal = extraordinary.refractive_index(SIGNAL_WAVELENGTH_UM);
    let n_idler = extraordinary.refractive_index(IDLER_WAVELENGTH_UM);
    let n_sh = extraordinary.refractive_index(PUMP_WAVELENGTH_UM / 2.0);
    let parameters = SfwmSourceParameters::new(
        CHI2_SOURCE,
        CHI3_SOURCE,
        1.0,
        n_pump,
        n_signal,
        n_idler,
        n_sh,
        PUMP_WAVELENGTH_UM,
        SIGNAL_WAVELENGTH_UM,
        IDLER_WAVELENGTH_UM,
    )?;
    let source_anchor_audit =
        SourceMismatchAudit::from_source_anchors(SourceCoherenceAnchors::son_chekhova())?;
    let paper_case = SourceCase {
        name: "paper-input",
        parameters,
        mismatches: source_anchor_audit.mismatches,
    };
    let sellmeier_case = SourceCase {
        name: "sellmeier-derived",
        parameters,
        mismatches: parameters.wavevector_mismatches()?,
    };
    let cases = vec![
        write_case(&args.output_dir, paper_case, &grid)?,
        write_case(&args.output_dir, sellmeier_case, &grid)?,
    ];
    let summary = Summary {
        schema: "sfwm-source-reproduction-v1".to_string(),
        source_pdf_sha256: SOURCE_PDF_SHA256.to_string(),
        source_tex_sha256: SOURCE_TEX_SHA256.to_string(),
        grid_sha256: grid_sha256.clone(),
        grid_count: grid.len(),
        grid_start_um: grid[0],
        grid_stop_um: *grid.last().context("nonempty frozen grid")?,
        grid_step_um: THICKNESS_STEP_UM,
        grid_refinement_steps_um: REFINEMENT_STEPS_UM,
        source_chi2_m_per_v: CHI2_SOURCE,
        source_chi3_m2_per_v2: CHI3_SOURCE,
        source_anchor_coherence_lengths_um: [33.3, 3.1, 3.4],
        source_anchor_derived_coherence_lengths_um: source_anchor_audit
            .derived_coherence_lengths_um,
        source_anchor_defects_um: source_anchor_audit.coherence_length_defects_um,
        extraordinary_indices: index_record("extraordinary", &extraordinary),
        ordinary_indices: index_record("ordinary", &ordinary),
        uncertainty: uncertainty_record(&extraordinary, source_anchor_audit.mismatches)?,
        cases,
    };
    let summary_text = toml::to_string_pretty(&summary).context("serialize source summary")?;
    let summary_path = args.output_dir.join("source-reproduction-summary.toml");
    fs::write(&summary_path, summary_text)
        .with_context(|| format!("write source summary {}", summary_path.display()))?;
    println!("grid_sha256={grid_sha256}");
    println!("summary={}", summary_path.display());
    println!("source_pdf_sha256={SOURCE_PDF_SHA256}");
    println!("source_tex_sha256={SOURCE_TEX_SHA256}");
    Ok(())
}
