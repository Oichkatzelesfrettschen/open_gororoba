use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, Duration, Utc};
use clap::Parser;
use data_core::catalogs::voyager::VoyagerSpacecraft;
use gororoba_cli_physics::{
    heliosphere_boundary::{ChronologyGate, DatasetBounds},
    voyager_arrow::{MissionPhase, TrajectoryFeeder, default_repo_root},
    voyager_encounter::{
        FusedEncounterFeeder, GridMappingConfig, VoyagerSpatialFeeder, heliocentric_to_lattice,
    },
};
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(name = "voyager-encounter-track")]
#[command(
    about = "Fuse promoted encounter Arrow telemetry with merged Voyager spatial coordinates and map them into lattice indices."
)]
struct Cli {
    /// Repository root used for promoted Arrow discovery. Defaults to current dir.
    #[arg(long)]
    repo_root: Option<PathBuf>,

    /// Mission phase selector for promoted Arrow discovery.
    #[arg(long, default_value = "jupiter_encounter")]
    mission_phase: String,

    /// Voyager spacecraft number for promoted Arrow discovery.
    #[arg(long)]
    spacecraft: u8,

    /// Product identifier for promoted Arrow discovery, e.g. LD1_RATE.
    #[arg(long)]
    product_id: String,

    /// Required Float64 column to inspect.
    #[arg(long)]
    value_column: String,

    /// Optional explicit path to the Voyager merged hourly file carrying
    /// r_au/lat_deg/lon_deg. When omitted, the binary discovers a governed
    /// staged file under data/external/voyager/... using the window start year.
    #[arg(long)]
    voyager_merged_file: Option<PathBuf>,

    /// Window start timestamp in RFC3339.
    #[arg(long)]
    window_start: String,

    /// Window length in hours.
    #[arg(long, default_value_t = 72.0)]
    window_hours: f64,

    /// Sampling cadence in minutes for the fused track output.
    #[arg(long, default_value_t = 60.0)]
    step_minutes: f64,

    #[arg(long, default_value_t = 128)]
    nx: usize,
    #[arg(long, default_value_t = 32)]
    ny: usize,
    #[arg(long, default_value_t = 32)]
    nz: usize,
    #[arg(long, default_value_t = 1.0)]
    r_min_au: f64,
    #[arg(long, default_value_t = 100.0)]
    r_max_au: f64,
    #[arg(long, default_value_t = -40.0)]
    lat_min_deg: f64,
    #[arg(long, default_value_t = 40.0)]
    lat_max_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    lon_min_deg: f64,
    #[arg(long, default_value_t = 360.0)]
    lon_max_deg: f64,
    #[arg(long, default_value_t = true)]
    log_radius: bool,

    /// Output CSV path for the fused encounter track.
    #[arg(
        long,
        default_value = "data/output/heliosphere/limits/voyager_encounter_track.csv"
    )]
    out: PathBuf,
}

#[derive(Serialize)]
struct TrackRow {
    timestamp: String,
    instrument_value: Option<f64>,
    r_au: Option<f64>,
    lat_deg: Option<f64>,
    lon_deg: Option<f64>,
    x: Option<usize>,
    y: Option<usize>,
    z: Option<usize>,
    in_bounds: bool,
}

fn parse_mission_phase(text: &str) -> MissionPhase {
    match text.to_ascii_lowercase().as_str() {
        "jupiter_encounter" | "jupiter-encounter" => MissionPhase::JupiterEncounter,
        other => panic!("unsupported mission phase {other}"),
    }
}

fn parse_spacecraft(number: u8) -> VoyagerSpacecraft {
    match number {
        1 => VoyagerSpacecraft::V1,
        2 => VoyagerSpacecraft::V2,
        other => panic!("unsupported Voyager spacecraft {other}"),
    }
}

fn discover_voyager_merged_file(
    repo_root: &std::path::Path,
    spacecraft: u8,
    year: i32,
) -> Result<PathBuf> {
    let (subdir, prefix) = match spacecraft {
        1 => ("voyager1", "vy1"),
        2 => ("voyager2", "vy2"),
        other => anyhow::bail!("unsupported Voyager spacecraft {other}"),
    };
    let root = repo_root.join("data/external/voyager").join(subdir);
    let candidates = [
        root.join(format!("{prefix}_{year}_amda_merged_hourly.asc")),
        root.join(format!("{prefix}_{year}.asc")),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "no staged Voyager merged file found for spacecraft {} year {} under {}",
        spacecraft,
        year,
        root.display()
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = cli.repo_root.unwrap_or(default_repo_root()?);
    let mission_phase = parse_mission_phase(&cli.mission_phase);
    let window_start = DateTime::parse_from_rfc3339(&cli.window_start)
        .with_context(|| format!("failed to parse {}", cli.window_start))?
        .with_timezone(&Utc);
    let voyager_merged_file = match cli.voyager_merged_file {
        Some(path) => path,
        None => discover_voyager_merged_file(&repo_root, cli.spacecraft, window_start.year())?,
    };
    let telemetry = TrajectoryFeeder::open_mission_phase(
        &repo_root,
        mission_phase,
        cli.spacecraft,
        &cli.product_id,
        cli.value_column.clone(),
    )?;
    let spatial = VoyagerSpatialFeeder::from_voyager_merged(
        &voyager_merged_file,
        parse_spacecraft(cli.spacecraft),
    )?;
    let chronology = ChronologyGate::from_dataset_bounds(vec![
        DatasetBounds {
            label: format!("encounter_{}", cli.product_id),
            bounds: telemetry.time_bounds()?,
        },
        DatasetBounds {
            label: format!("voyager{}_merged", cli.spacecraft),
            bounds: spatial
                .time_bounds()
                .context("Voyager spatial feeder had no temporal bounds")?,
        },
    ])?;
    let window_end_ms =
        window_start.timestamp_millis() + (cli.window_hours * 3600.0 * 1000.0).round() as i64;
    chronology.require_window(window_start.timestamp_millis(), window_end_ms)?;
    let fused = FusedEncounterFeeder::new(telemetry, spatial);

    let step_ms = (cli.step_minutes * 60.0 * 1000.0).round() as i64;
    let steps = ((cli.window_hours * 60.0) / cli.step_minutes).round() as usize;
    let grid = GridMappingConfig {
        nx: cli.nx,
        ny: cli.ny,
        nz: cli.nz,
        r_min_au: cli.r_min_au,
        r_max_au: cli.r_max_au,
        lat_min_deg: cli.lat_min_deg,
        lat_max_deg: cli.lat_max_deg,
        lon_min_deg: cli.lon_min_deg,
        lon_max_deg: cli.lon_max_deg,
        log_radius: cli.log_radius,
    };

    if let Some(parent) = cli.out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut writer = csv::Writer::from_path(&cli.out)
        .with_context(|| format!("failed to create {}", cli.out.display()))?;

    for step in 0..=steps {
        let timestamp_ms = window_start.timestamp_millis() + step as i64 * step_ms;
        let timestamp = window_start + Duration::milliseconds(step as i64 * step_ms);
        let sample = fused.sample(timestamp_ms)?;
        let (instrument_value, r_au, lat_deg, lon_deg) = if let Some(sample) = sample {
            (
                sample.instrument_value,
                sample.r_au,
                sample.lat_deg,
                sample.lon_deg,
            )
        } else {
            (None, None, None, None)
        };
        let lattice = match (r_au, lat_deg, lon_deg) {
            (Some(r), Some(lat), Some(lon)) => heliocentric_to_lattice(r, lat, lon, grid),
            _ => None,
        };
        writer.serialize(TrackRow {
            timestamp: timestamp.to_rfc3339(),
            instrument_value,
            r_au,
            lat_deg,
            lon_deg,
            x: lattice.map(|idx| idx.x),
            y: lattice.map(|idx| idx.y),
            z: lattice.map(|idx| idx.z),
            in_bounds: lattice.is_some(),
        })?;
    }
    writer.flush()?;
    println!("wrote {}", cli.out.display());
    Ok(())
}
