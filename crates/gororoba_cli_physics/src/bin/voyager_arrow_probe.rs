use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use gororoba_cli_physics::voyager_arrow::{MissionPhase, TrajectoryFeeder, default_repo_root};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "voyager-arrow-probe")]
#[command(
    about = "Memory-map a Voyager Arrow artifact, assert schema, and locate telemetry bounds for a target timestamp."
)]
struct Cli {
    /// Arrow IPC file emitted by ingest-pds-crs --export-format arrow.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Repository root used for mission-phase discovery. Defaults to current dir.
    #[arg(long)]
    repo_root: Option<PathBuf>,

    /// Mission phase selector for promoted Arrow discovery.
    #[arg(long)]
    mission_phase: Option<String>,

    /// Voyager spacecraft number for promoted Arrow discovery.
    #[arg(long)]
    spacecraft: Option<u8>,

    /// Product identifier for promoted Arrow discovery, e.g. LD1_RATE.
    #[arg(long)]
    product_id: Option<String>,

    /// Required Float64 column to assert and inspect.
    #[arg(long)]
    value_column: String,

    /// Target timestamp in RFC3339 (UTC).
    #[arg(long)]
    target_time: String,

    /// Optional symmetric window width in hours around the target time.
    #[arg(long)]
    window_hours: Option<f64>,
}

#[derive(Serialize)]
struct ProbeReport {
    input: String,
    file_len_bytes: usize,
    rows: usize,
    value_column: String,
    target_time: String,
    lower_index: usize,
    upper_index: usize,
    lower_time_ms: i64,
    upper_time_ms: i64,
    lower_valid: bool,
    upper_valid: bool,
    lower_value: Option<f64>,
    upper_value: Option<f64>,
    interpolated_value: Option<f64>,
    window_start_index: Option<usize>,
    window_end_index: Option<usize>,
    window_row_count: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let target_ms = DateTime::parse_from_rfc3339(&cli.target_time)
        .with_context(|| format!("failed to parse target time {}", cli.target_time))?
        .with_timezone(&Utc)
        .timestamp_millis();

    let feeder = if let Some(input) = &cli.input {
        TrajectoryFeeder::open_input(input, cli.value_column.clone())?
    } else {
        let repo_root = cli.repo_root.unwrap_or(default_repo_root()?);
        let mission_phase = match cli.mission_phase.as_deref() {
            Some("jupiter_encounter") => MissionPhase::JupiterEncounter,
            Some(other) => anyhow::bail!("unsupported mission phase {}", other),
            None => anyhow::bail!(
                "either --input or --mission-phase/--spacecraft/--product-id is required"
            ),
        };
        let spacecraft = cli
            .spacecraft
            .context("--spacecraft is required for mission-phase discovery")?;
        let product_id = cli
            .product_id
            .as_deref()
            .context("--product-id is required for mission-phase discovery")?;
        TrajectoryFeeder::open_mission_phase(
            repo_root,
            mission_phase,
            spacecraft,
            product_id,
            cli.value_column.clone(),
        )?
    };

    let dataset = feeder.dataset();
    let sample = feeder
        .sample_linear(target_ms)?
        .context("Arrow file had no rows")?;
    let times = dataset.timestamp_values()?;
    let (window_start_index, window_end_index, window_row_count) =
        if let Some(window_hours) = cli.window_hours {
            let half_window_ms = (window_hours * 0.5 * 3600.0 * 1000.0).round() as i64;
            if let Some((start, end)) =
                feeder.window_bounds(target_ms - half_window_ms, target_ms + half_window_ms)?
            {
                (Some(start), Some(end), Some(end - start + 1))
            } else {
                (None, None, Some(0))
            }
        } else {
            (None, None, None)
        };

    let report = ProbeReport {
        input: dataset.path().display().to_string(),
        file_len_bytes: dataset.file_len_bytes(),
        rows: dataset.num_rows(),
        value_column: cli.value_column.clone(),
        target_time: cli.target_time,
        lower_index: sample.lower_index,
        upper_index: sample.upper_index,
        lower_time_ms: times[sample.lower_index],
        upper_time_ms: times[sample.upper_index],
        lower_valid: dataset.is_valid(&cli.value_column, sample.lower_index)?,
        upper_valid: dataset.is_valid(&cli.value_column, sample.upper_index)?,
        lower_value: sample.lower_value,
        upper_value: sample.upper_value,
        interpolated_value: sample.interpolated_value,
        window_start_index,
        window_end_index,
        window_row_count,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
