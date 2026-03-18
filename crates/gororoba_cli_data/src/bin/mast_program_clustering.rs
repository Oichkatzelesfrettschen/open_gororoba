use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, Duration, TimeZone, Utc};
use clap::Parser;
use data_core::catalogs::{
    hst::{HstPublicObservation, parse_hst_public_metadata_csv},
    jwst::{JwstPublicObservation, parse_jwst_public_metadata_csv},
};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "mast-program-clustering",
    about = "Cluster bounded JWST/HST metadata by program, release time, and instrument"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long)]
    out_csv: Option<PathBuf>,

    #[arg(long)]
    out_report: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct ProgramObservation {
    mission: String,
    proposal_id: String,
    instrument_name: String,
    target_name: String,
    release_time_utc: String,
    ra_deg: f64,
    dec_deg: f64,
    filters: String,
}

#[derive(Debug, Serialize)]
struct ClusterRow {
    mission: String,
    proposal_id: String,
    instrument_name: String,
    release_year: i32,
    observation_count: usize,
    target_count: usize,
    filter_token_count_mean: f64,
    mean_ra_deg: f64,
    mean_dec_deg: f64,
    release_start_utc: String,
    release_end_utc: String,
}

#[derive(Debug, Serialize)]
struct ClusterReport {
    generated_at_utc: String,
    missions: Vec<String>,
    cluster_count: usize,
    source_paths: Vec<String>,
    notes: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let date = Utc::now().date_naive();
    let out_csv = cli.out_csv.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!("mast_program_clustering_{}.csv", date))
    });
    let out_report = cli.out_report.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!("mast_program_clustering_{}.toml", date))
    });

    let jwst_path = cli.repo_root.join("data/external/jwst_public_observations.csv");
    let hst_path = cli.repo_root.join("data/external/hst_public_observations.csv");
    let mut source_paths = Vec::new();
    let mut observations = Vec::new();

    if jwst_path.exists() {
        source_paths.push(jwst_path.display().to_string());
        observations.extend(
            parse_jwst_public_metadata_csv(&jwst_path)?
                .into_iter()
                .filter_map(program_observation_from_jwst),
        );
    }
    if hst_path.exists() {
        source_paths.push(hst_path.display().to_string());
        observations.extend(
            parse_hst_public_metadata_csv(&hst_path)?
                .into_iter()
                .filter_map(program_observation_from_hst),
        );
    }

    let clusters = cluster_programs(&observations);
    if let Some(parent) = out_csv.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = out_report.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut writer = csv::Writer::from_path(&out_csv)?;
    for row in &clusters {
        writer.serialize(row)?;
    }
    writer.flush()?;

    let mut missions = clusters.iter().map(|row| row.mission.clone()).collect::<Vec<_>>();
    missions.sort();
    missions.dedup();
    let report = ClusterReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        missions,
        cluster_count: clusters.len(),
        source_paths,
        notes: vec![
            "Clusters are keyed by mission + proposal_id + instrument_name + release_year.".to_string(),
            "This is a bounded program/time clustering lane over public MAST metadata, not a full archive workflow."
                .to_string(),
        ],
    };
    fs::write(&out_report, toml::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", out_report.display()))?;
    println!("clusters = {}", clusters.len());
    println!("csv = {}", out_csv.display());
    println!("report = {}", out_report.display());
    Ok(())
}

fn cluster_programs(observations: &[ProgramObservation]) -> Vec<ClusterRow> {
    let mut groups: BTreeMap<(String, String, String, i32), Vec<&ProgramObservation>> =
        BTreeMap::new();
    for observation in observations {
        let release_year = parse_release_time(&observation.release_time_utc)
            .map(|dt| dt.year())
            .unwrap_or(0);
        groups
            .entry((
                observation.mission.clone(),
                observation.proposal_id.clone(),
                observation.instrument_name.clone(),
                release_year,
            ))
            .or_default()
            .push(observation);
    }

    groups
        .into_iter()
        .map(|((mission, proposal_id, instrument_name, release_year), group)| {
            let observation_count = group.len();
            let mut targets = group
                .iter()
                .map(|row| row.target_name.clone())
                .collect::<Vec<_>>();
            targets.sort();
            targets.dedup();
            let filter_token_count_mean = group
                .iter()
                .map(|row| {
                    row.filters
                        .split('|')
                        .filter(|token| !token.trim().is_empty())
                        .count() as f64
                })
                .sum::<f64>()
                / observation_count as f64;
            let mean_ra_deg =
                group.iter().map(|row| row.ra_deg).sum::<f64>() / observation_count as f64;
            let mean_dec_deg =
                group.iter().map(|row| row.dec_deg).sum::<f64>() / observation_count as f64;
            let mut release_times = group
                .iter()
                .filter_map(|row| parse_release_time(&row.release_time_utc))
                .collect::<Vec<_>>();
            release_times.sort();
            let release_start_utc = release_times
                .first()
                .map(DateTime::<Utc>::to_rfc3339)
                .unwrap_or_default();
            let release_end_utc = release_times
                .last()
                .map(DateTime::<Utc>::to_rfc3339)
                .unwrap_or_default();
            ClusterRow {
                mission,
                proposal_id,
                instrument_name,
                release_year,
                observation_count,
                target_count: targets.len(),
                filter_token_count_mean,
                mean_ra_deg,
                mean_dec_deg,
                release_start_utc,
                release_end_utc,
            }
        })
        .collect()
}

fn program_observation_from_jwst(row: JwstPublicObservation) -> Option<ProgramObservation> {
    if !row.s_ra.is_finite() || !row.s_dec.is_finite() || row.obsid.trim().is_empty() {
        return None;
    }
    Some(ProgramObservation {
        mission: "JWST".to_string(),
        proposal_id: row.proposal_id,
        instrument_name: row.instrument_name,
        target_name: row.target_name,
        release_time_utc: row.t_obs_release,
        ra_deg: row.s_ra,
        dec_deg: row.s_dec,
        filters: row.filters,
    })
}

fn program_observation_from_hst(row: HstPublicObservation) -> Option<ProgramObservation> {
    if !row.s_ra.is_finite() || !row.s_dec.is_finite() || row.obsid.trim().is_empty() {
        return None;
    }
    Some(ProgramObservation {
        mission: "HST".to_string(),
        proposal_id: row.proposal_id,
        instrument_name: row.instrument_name,
        target_name: row.target_name,
        release_time_utc: row.t_obs_release,
        ra_deg: row.s_ra,
        dec_deg: row.s_dec,
        filters: row.filters,
    })
}

fn parse_release_time(value: &str) -> Option<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Some(dt.with_timezone(&Utc));
    }
    let mjd = value.trim().parse::<f64>().ok()?;
    let epoch = Utc.with_ymd_and_hms(1858, 11, 17, 0, 0, 0).single()?;
    let millis = (mjd * 86_400_000.0).round() as i64;
    Some(epoch + Duration::milliseconds(millis))
}
