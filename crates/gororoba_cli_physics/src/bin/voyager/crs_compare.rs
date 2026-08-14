use chrono::{DateTime, Duration, Utc};
use clap::Args;
use csv::Reader;
use data_core::catalogs::voyager_crs::{VoyagerCrsRecord, parse_voyager_crs};
use gororoba_cli_physics::voyager_arrow::{MissionPhase, TrajectoryFeeder, default_repo_root};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Args, Debug)]
pub struct Cli {
    /// Comparison mode: shape, shape-arrow, or absolute.
    #[arg(long, default_value = "shape")]
    mode: String,

    /// Root containing voyager{1,2}/crs/... legacy rate files.
    #[arg(long, default_value = "data/external/voyager")]
    crs_root: PathBuf,

    /// cr-modulation-sweep summary CSV with columns r_au,phi_gv for shape mode.
    #[arg(long)]
    phi_summary: Option<PathBuf>,

    /// Normalized observed-flux CSV for absolute mode.
    /// Expected columns:
    /// spacecraft,decimal_year,distance_au,channel_index,kinetic_energy_gev,flux,flux_error
    #[arg(long)]
    observed_flux_csv: Option<PathBuf>,

    /// Staging output directory used to discover the default observed flux CSV.
    #[arg(long, default_value = "data/output/heliosphere/staging")]
    stage_dir: PathBuf,

    /// Baseline modeled flux CSV from cr-modulation-sweep.
    #[arg(long)]
    model_flux_csv: Option<PathBuf>,

    /// Optional DM-template modeled flux CSV from cr-modulation-sweep.
    #[arg(long)]
    dm_model_flux_csv: Option<PathBuf>,

    /// Output CSV path.
    #[arg(
        long,
        default_value = "data/output/heliosphere/limits/voyager_crs_compare.csv"
    )]
    out: PathBuf,

    /// Governed mission phase to discover promoted Arrow telemetry.
    #[arg(long)]
    mission_phase: Option<String>,

    /// Spacecraft number for promoted Arrow telemetry.
    #[arg(long)]
    spacecraft: Option<u8>,

    /// Promoted Arrow product identifier, e.g. LD1_RATE.
    #[arg(long)]
    product_id: Option<String>,

    /// Value column inside the promoted Arrow file.
    #[arg(long)]
    value_column: Option<String>,

    /// Optional repo root used for promoted Arrow discovery.
    #[arg(long)]
    repo_root: Option<PathBuf>,

    /// RFC3339 window start for promoted Arrow comparison.
    #[arg(long)]
    window_start: Option<String>,

    /// Window length in hours for promoted Arrow comparison.
    #[arg(long, default_value_t = 72)]
    window_hours: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct PhiRow {
    r_au: f64,
    phi_gv: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ShapeCompareRow {
    spacecraft: u8,
    path: String,
    records: usize,
    fill_rate: f64,
    median_distance_au: f64,
    median_proton_ch1_rate: f64,
    nearest_phi_gv: f64,
}

#[derive(Clone, Debug, Serialize)]
struct EncounterWindowRow {
    spacecraft: u8,
    mission_phase: String,
    product_id: String,
    value_column: String,
    input: String,
    rows_total: usize,
    window_start: String,
    window_end: String,
    window_start_index: usize,
    window_end_index: usize,
    window_row_count: usize,
    valid_fraction: f64,
    mean_value: f64,
    median_value: f64,
    min_value: f64,
    max_value: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct ObservedFluxRow {
    spacecraft: u8,
    decimal_year: f64,
    distance_au: f64,
    channel_index: usize,
    kinetic_energy_gev: f64,
    flux: f64,
    flux_error: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelFluxRow {
    #[serde(rename = "step")]
    _step: usize,
    r_au: f64,
    #[serde(rename = "x")]
    _x: usize,
    #[serde(rename = "y")]
    _y: usize,
    #[serde(rename = "z")]
    _z: usize,
    #[serde(rename = "rigidity_gv")]
    _rigidity_gv: f64,
    kinetic_energy_gev: f64,
    flux_j: f64,
    modulation_phi: f64,
}

#[derive(Clone, Debug, Serialize)]
struct AbsoluteCompareRow {
    spacecraft: u8,
    decimal_year: f64,
    distance_au: f64,
    channel_index: usize,
    kinetic_energy_gev: f64,
    observed_flux: f64,
    observed_error: f64,
    baseline_flux: f64,
    dm_template_flux: f64,
    phi_gv: f64,
    residual_sigma: f64,
}

#[derive(Clone, Debug, Serialize)]
struct AbsoluteFitSummary {
    rows_used: usize,
    chi2_null: f64,
    chi2_best: f64,
    amplitude_hat: f64,
    amplitude_95: f64,
    sigma_v_scale_95: f64,
}

fn finite_median(mut values: Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn finite_mean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for value in values {
        if value.is_finite() {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn detect_spacecraft(path: &Path) -> Option<u8> {
    let text = path.to_string_lossy();
    if text.contains("voyager1") || text.contains("vy1") {
        Some(1)
    } else if text.contains("voyager2") || text.contains("vy2") {
        Some(2)
    } else {
        None
    }
}

fn collect_crs_files(root: &Path) -> Vec<PathBuf> {
    fn recurse(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                recurse(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "asc")
                && path.to_string_lossy().contains("/crs/")
            {
                out.push(path);
            }
        }
    }

    let mut files = Vec::new();
    recurse(root, &mut files);
    files.sort();
    files
}

fn load_phi_rows(path: &Path) -> Vec<PhiRow> {
    let mut reader = Reader::from_path(path).expect("failed to open phi summary");
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        let row: PhiRow = row.expect("failed to parse phi summary row");
        rows.push(row);
    }
    rows
}

fn nearest_phi(phi_rows: &[PhiRow], distance_au: f64) -> f64 {
    phi_rows
        .iter()
        .min_by(|a, b| {
            (a.r_au - distance_au)
                .abs()
                .partial_cmp(&(b.r_au - distance_au).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|row| row.phi_gv)
        .unwrap_or(f64::NAN)
}

fn load_crs_records(path: &Path) -> (u8, Vec<VoyagerCrsRecord>) {
    let spacecraft = detect_spacecraft(path).expect("unable to infer spacecraft");
    let content = fs::read_to_string(path).expect("failed to read CRS file");
    let (records, _skipped) = parse_voyager_crs(&content, spacecraft);
    (spacecraft, records)
}

fn compare_shape_file(path: &Path, phi_rows: &[PhiRow]) -> ShapeCompareRow {
    let (spacecraft, records) = load_crs_records(path);
    let fill_rate = if records.is_empty() {
        0.0
    } else {
        records.iter().filter(|record| record.fill_flag).count() as f64 / records.len() as f64
    };
    let median_distance_au =
        finite_median(records.iter().map(|record| record.distance_au).collect());
    let median_proton_ch1_rate = finite_median(
        records
            .iter()
            .map(|record| record.proton_rates[0])
            .collect(),
    );

    ShapeCompareRow {
        spacecraft,
        path: path.to_string_lossy().into_owned(),
        records: records.len(),
        fill_rate,
        median_distance_au,
        median_proton_ch1_rate,
        nearest_phi_gv: nearest_phi(phi_rows, median_distance_au),
    }
}

fn load_csv_rows<T: for<'de> Deserialize<'de>>(path: &Path) -> Vec<T> {
    let mut reader = Reader::from_path(path)
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", path.display()));
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row.unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display())));
    }
    rows
}

fn parse_mission_phase(text: &str) -> MissionPhase {
    match text.to_ascii_lowercase().as_str() {
        "jupiter_encounter" | "jupiter-encounter" => MissionPhase::JupiterEncounter,
        other => panic!("unsupported mission phase {other}"),
    }
}

fn parse_rfc3339_utc(text: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(text)
        .unwrap_or_else(|err| panic!("failed to parse RFC3339 timestamp {text}: {err}"))
        .with_timezone(&Utc)
}

fn build_encounter_window_row(
    repo_root: &Path,
    mission_phase: MissionPhase,
    spacecraft: u8,
    product_id: &str,
    value_column: &str,
    window_start: DateTime<Utc>,
    window_hours: u64,
) -> EncounterWindowRow {
    let feeder = TrajectoryFeeder::open_mission_phase(
        repo_root,
        mission_phase,
        spacecraft,
        product_id,
        value_column.to_string(),
    )
    .unwrap_or_else(|err| {
        panic!(
            "failed to open promoted Arrow telemetry for spacecraft {} product {}: {err}",
            spacecraft, product_id
        )
    });
    let window_end = window_start + Duration::hours(window_hours as i64);
    let start_ms = window_start.timestamp_millis();
    let end_ms = window_end.timestamp_millis();
    let (window_start_index, window_end_index) = feeder
        .window_bounds(start_ms, end_ms)
        .unwrap_or_else(|err| panic!("failed to compute window bounds: {err}"))
        .unwrap_or_else(|| {
            panic!(
                "no promoted Arrow samples found for {} between {} and {}",
                product_id,
                window_start.to_rfc3339(),
                window_end.to_rfc3339()
            )
        });

    let mut finite_values = Vec::new();
    for index in window_start_index..=window_end_index {
        if let Some(value) = feeder
            .dataset()
            .float64_value(value_column, index)
            .unwrap_or_else(|err| panic!("failed to read {value_column} at row {index}: {err}"))
        {
            finite_values.push(value);
        }
    }
    let window_row_count = window_end_index - window_start_index + 1;
    let valid_fraction = finite_values.len() as f64 / window_row_count as f64;
    let median_value = finite_median(finite_values.clone());
    let mean_value = finite_mean(&finite_values);
    let min_value = finite_values
        .iter()
        .copied()
        .reduce(f64::min)
        .unwrap_or(f64::NAN);
    let max_value = finite_values
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(f64::NAN);

    EncounterWindowRow {
        spacecraft,
        mission_phase: mission_phase.as_path_component().to_string(),
        product_id: product_id.to_string(),
        value_column: value_column.to_string(),
        input: feeder.dataset().path().display().to_string(),
        rows_total: feeder.dataset().num_rows(),
        window_start: window_start.to_rfc3339(),
        window_end: window_end.to_rfc3339(),
        window_start_index,
        window_end_index,
        window_row_count,
        valid_fraction,
        mean_value,
        median_value,
        min_value,
        max_value,
    }
}

fn nearest_model_row(
    rows: &[ModelFluxRow],
    distance_au: f64,
    kinetic_energy_gev: f64,
) -> Option<&ModelFluxRow> {
    rows.iter()
        .filter(|row| row.flux_j.is_finite())
        .min_by(|a, b| {
            let da = (a.r_au - distance_au).abs()
                + (a.kinetic_energy_gev / kinetic_energy_gev).ln().abs();
            let db = (b.r_au - distance_au).abs()
                + (b.kinetic_energy_gev / kinetic_energy_gev).ln().abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn build_absolute_rows(
    observed: &[ObservedFluxRow],
    baseline: &[ModelFluxRow],
    dm_template: Option<&[ModelFluxRow]>,
) -> Vec<AbsoluteCompareRow> {
    let mut rows = Vec::new();
    for obs in observed {
        if !(obs.flux.is_finite()
            && obs.flux_error.is_finite()
            && obs.flux_error > 0.0
            && obs.distance_au.is_finite()
            && obs.kinetic_energy_gev.is_finite()
            && obs.kinetic_energy_gev > 0.0)
        {
            continue;
        }
        let baseline_row = nearest_model_row(baseline, obs.distance_au, obs.kinetic_energy_gev)
            .unwrap_or_else(|| {
                panic!(
                    "no baseline model row near r={} K={}",
                    obs.distance_au, obs.kinetic_energy_gev
                )
            });
        let dm_flux = dm_template
            .and_then(|rows| nearest_model_row(rows, obs.distance_au, obs.kinetic_energy_gev))
            .map(|row| row.flux_j)
            .unwrap_or(0.0);
        let residual_sigma = (obs.flux - baseline_row.flux_j) / obs.flux_error;
        rows.push(AbsoluteCompareRow {
            spacecraft: obs.spacecraft,
            decimal_year: obs.decimal_year,
            distance_au: obs.distance_au,
            channel_index: obs.channel_index,
            kinetic_energy_gev: obs.kinetic_energy_gev,
            observed_flux: obs.flux,
            observed_error: obs.flux_error,
            baseline_flux: baseline_row.flux_j,
            dm_template_flux: dm_flux,
            phi_gv: baseline_row.modulation_phi,
            residual_sigma,
        });
    }
    rows
}

fn fit_upper_limit(rows: &[AbsoluteCompareRow], sigma_v_reference: f64) -> AbsoluteFitSummary {
    let mut weighted_signal_sq = 0.0;
    let mut weighted_signal_residual = 0.0;
    let mut chi2_null = 0.0;

    for row in rows {
        if !(row.observed_error.is_finite() && row.observed_error > 0.0) {
            continue;
        }
        let variance = row.observed_error * row.observed_error;
        let weight = 1.0 / variance;
        let residual = row.observed_flux - row.baseline_flux;
        chi2_null += residual * residual * weight;
        weighted_signal_sq += row.dm_template_flux * row.dm_template_flux * weight;
        weighted_signal_residual += row.dm_template_flux * residual * weight;
    }

    let amplitude_unclamped = if weighted_signal_sq > 0.0 {
        weighted_signal_residual / weighted_signal_sq
    } else {
        0.0
    };
    let amplitude_hat = amplitude_unclamped.max(0.0);
    // When amplitude was negative and clamped to 0, the best-fit chi2 at
    // amplitude=0 equals chi2_null.  Only use the analytic minimum when the
    // unclamped optimum is non-negative.
    let chi2_best = if weighted_signal_sq > 0.0 && amplitude_unclamped >= 0.0 {
        chi2_null - weighted_signal_residual * weighted_signal_residual / weighted_signal_sq
    } else {
        chi2_null
    };
    let amplitude_95 = if weighted_signal_sq > 0.0 {
        amplitude_hat + (2.71 / weighted_signal_sq).sqrt()
    } else {
        0.0
    };

    AbsoluteFitSummary {
        rows_used: rows.len(),
        chi2_null,
        chi2_best,
        amplitude_hat,
        amplitude_95,
        sigma_v_scale_95: amplitude_95 * sigma_v_reference,
    }
}

pub fn run(cli: Cli) {
    let mode = cli.mode.to_lowercase();

    if let Some(parent) = cli.out.parent() {
        fs::create_dir_all(parent).expect("failed to create output directory");
    }

    match mode.as_str() {
        "absolute" => {
            let observed_path_buf = cli
                .observed_flux_csv
                .clone()
                .unwrap_or_else(|| cli.stage_dir.join("voyager_crs_flux_observed.csv"));
            let observed_path = observed_path_buf.as_path();
            let baseline_path = cli
                .model_flux_csv
                .as_deref()
                .expect("--model-flux-csv is required for absolute mode");
            assert!(
                observed_path.exists(),
                "absolute mode requires a staged observed flux CSV at {} (or pass --observed-flux-csv)",
                observed_path.display()
            );
            let observed = load_csv_rows::<ObservedFluxRow>(observed_path);
            let baseline = load_csv_rows::<ModelFluxRow>(baseline_path);
            let dm_template = cli
                .dm_model_flux_csv
                .as_deref()
                .map(load_csv_rows::<ModelFluxRow>);
            let rows = build_absolute_rows(&observed, &baseline, dm_template.as_deref());
            assert!(
                !rows.is_empty(),
                "absolute mode did not produce any comparable rows"
            );

            let mut writer =
                csv::Writer::from_path(&cli.out).expect("failed to create compare CSV");
            for row in &rows {
                writer
                    .serialize(row)
                    .expect("failed to write absolute compare row");
            }
            writer.flush().expect("failed to flush compare CSV");

            if dm_template.is_some() {
                let summary = fit_upper_limit(&rows, 3.0e-26);
                let summary_path = cli.out.with_extension("toml");
                let payload =
                    toml::to_string_pretty(&summary).expect("failed to serialize summary");
                fs::write(&summary_path, payload).expect("failed to write absolute summary");
                println!("wrote {}", summary_path.display());
            }
            println!("wrote {}", cli.out.display());
        }
        "shape-arrow" => {
            let repo_root = cli
                .repo_root
                .clone()
                .unwrap_or_else(|| default_repo_root().expect("failed to resolve repo root"));
            let mission_phase = parse_mission_phase(
                cli.mission_phase
                    .as_deref()
                    .expect("--mission-phase is required for shape-arrow mode"),
            );
            let spacecraft = cli
                .spacecraft
                .expect("--spacecraft is required for shape-arrow mode");
            let product_id = cli
                .product_id
                .as_deref()
                .expect("--product-id is required for shape-arrow mode");
            let value_column = cli
                .value_column
                .as_deref()
                .expect("--value-column is required for shape-arrow mode");
            let window_start = parse_rfc3339_utc(
                cli.window_start
                    .as_deref()
                    .expect("--window-start is required for shape-arrow mode"),
            );

            let row = build_encounter_window_row(
                &repo_root,
                mission_phase,
                spacecraft,
                product_id,
                value_column,
                window_start,
                cli.window_hours,
            );

            let mut writer =
                csv::Writer::from_path(&cli.out).expect("failed to create compare CSV");
            writer
                .serialize(row)
                .expect("failed to write encounter compare row");
            writer.flush().expect("failed to flush compare CSV");
            println!("wrote {}", cli.out.display());
        }
        _ => {
            let phi_path = cli
                .phi_summary
                .as_deref()
                .expect("--phi-summary is required for shape mode");
            let phi_rows = load_phi_rows(phi_path);
            let crs_files = collect_crs_files(&cli.crs_root);
            assert!(
                !crs_files.is_empty(),
                "no CRS files found under {}",
                cli.crs_root.display()
            );
            let mut writer =
                csv::Writer::from_path(&cli.out).expect("failed to create compare CSV");
            for file in &crs_files {
                writer
                    .serialize(compare_shape_file(file, &phi_rows))
                    .expect("failed to write shape compare row");
            }
            writer.flush().expect("failed to flush compare CSV");
            println!("wrote {}", cli.out.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_phi_selects_closest_radius() {
        let rows = vec![
            PhiRow {
                r_au: 1.0,
                phi_gv: 0.6,
            },
            PhiRow {
                r_au: 80.0,
                phi_gv: 0.1,
            },
            PhiRow {
                r_au: 100.0,
                phi_gv: 0.05,
            },
        ];
        assert!((nearest_phi(&rows, 84.0) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_finite_median_ignores_nan() {
        let median = finite_median(vec![f64::NAN, 5.0, 1.0, 3.0]);
        assert!((median - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_fit_upper_limit_recovers_positive_signal() {
        let rows = vec![
            AbsoluteCompareRow {
                spacecraft: 1,
                decimal_year: 2010.0,
                distance_au: 100.0,
                channel_index: 1,
                kinetic_energy_gev: 0.5,
                observed_flux: 12.0,
                observed_error: 1.0,
                baseline_flux: 10.0,
                dm_template_flux: 2.0,
                phi_gv: 0.1,
                residual_sigma: 2.0,
            },
            AbsoluteCompareRow {
                spacecraft: 1,
                decimal_year: 2010.0,
                distance_au: 110.0,
                channel_index: 2,
                kinetic_energy_gev: 1.0,
                observed_flux: 18.0,
                observed_error: 2.0,
                baseline_flux: 16.0,
                dm_template_flux: 4.0,
                phi_gv: 0.1,
                residual_sigma: 1.0,
            },
        ];
        let summary = fit_upper_limit(&rows, 3.0e-26);
        assert!(summary.amplitude_hat > 0.0);
        assert!(summary.amplitude_95 > summary.amplitude_hat);
    }

    #[test]
    fn test_build_encounter_window_row_summarizes_promoted_arrow_window() {
        use arrow_array::{
            ArrayRef, RecordBatch,
            builder::{Float64Builder, TimestampMillisecondBuilder},
        };
        use arrow_ipc::writer::FileWriter;
        use arrow_schema::{DataType, Field, Schema, TimeUnit};
        use std::sync::Arc;

        let root = std::env::temp_dir().join(format!(
            "voyager_crs_compare_window_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let phase_dir = root.join("data/output/heliosphere/voyager/jupiter_encounter/voyager2");
        fs::create_dir_all(&phase_dir).expect("create phase dir");
        let arrow_path = phase_dir.join("VG2-J-CRS-5-SUMM-FLUX-V1.0_ld1_rate.tab.arrow");

        let mut time_builder = TimestampMillisecondBuilder::new();
        let t0 = parse_rfc3339_utc("1979-07-03T00:00:00Z").timestamp_millis();
        time_builder.append_value(t0);
        time_builder.append_value(t0 + 3_600_000);
        time_builder.append_value(t0 + 7_200_000);
        let mut value_builder = Float64Builder::new();
        value_builder.append_value(3.0);
        value_builder.append_null();
        value_builder.append_value(5.0);
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "TIME",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new("LD1 RATE", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(time_builder.finish()) as ArrayRef,
                Arc::new(value_builder.finish()) as ArrayRef,
            ],
        )
        .expect("batch");
        let file = std::fs::File::create(&arrow_path).expect("create arrow");
        let mut writer = FileWriter::try_new(file, &schema).expect("writer");
        writer.write(&batch).expect("write");
        writer.finish().expect("finish");

        let row = build_encounter_window_row(
            &root,
            MissionPhase::JupiterEncounter,
            2,
            "LD1_RATE",
            "LD1 RATE",
            parse_rfc3339_utc("1979-07-03T00:00:00Z"),
            2,
        );
        assert_eq!(row.window_row_count, 3);
        assert!((row.valid_fraction - (2.0 / 3.0)).abs() < 1e-12);
        assert!((row.mean_value - 4.0).abs() < 1e-12);
        assert!((row.median_value - 5.0).abs() < 1e-12 || (row.median_value - 3.0).abs() < 1e-12);

        let _ = fs::remove_dir_all(root);
    }
}
