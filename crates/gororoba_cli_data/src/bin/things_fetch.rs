use anyhow::{Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use data_core::{
    DatasetProvider, FetchConfig,
    catalogs::things::{
        ThingsPreferredCubesProvider, ThingsTablesProvider, build_things_hi_metadata,
        discover_things_cube_manifest, parse_things_galaxies, parse_things_hi_spectra,
        preferred_things_cube_entries, write_things_cube_manifest_csv,
    },
};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(name = "things-fetch")]
#[command(about = "Rust-native THINGS tables/cubes acquisition and phase-2 staging")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Tables {
        #[arg(long, default_value = "data/external")]
        output_dir: PathBuf,
        #[arg(
            long,
            default_value_t = true,
            action = ArgAction::Set,
            num_args = 0..=1,
            default_missing_value = "true"
        )]
        skip_existing: bool,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Manifest {
        #[arg(
            long,
            default_value = "data/external/things/things_cube_manifest_all.csv"
        )]
        output: PathBuf,
        #[arg(long, default_value_t = false)]
        preferred_only: bool,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Metadata {
        #[arg(long, default_value = "data/external/things/table1.dat")]
        table1: PathBuf,
        #[arg(long, default_value = "data/external/things/table4.dat")]
        table4: PathBuf,
        #[arg(long, default_value = "data/external/things/things_metadata.csv")]
        output: PathBuf,
        #[arg(long, default_value_t = 6.0)]
        default_beam_fwhm_arcsec: f64,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Cubes {
        #[arg(long, default_value = "data/external")]
        output_dir: PathBuf,
        #[arg(
            long,
            default_value_t = true,
            action = ArgAction::Set,
            num_args = 0..=1,
            default_missing_value = "true"
        )]
        skip_existing: bool,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    Phase2 {
        #[arg(long, default_value = "data/external")]
        output_dir: PathBuf,
        #[arg(
            long,
            default_value_t = true,
            action = ArgAction::Set,
            num_args = 0..=1,
            default_missing_value = "true"
        )]
        skip_existing: bool,
        #[arg(long, default_value_t = 6.0)]
        default_beam_fwhm_arcsec: f64,
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Debug, Serialize)]
struct ThingsFetchReport {
    generated_at_utc: String,
    mode: String,
    output_root: String,
    tables_dir: Option<String>,
    metadata_csv: Option<String>,
    manifest_csv: Option<String>,
    cubes_dir: Option<String>,
    table_galaxy_count: Option<usize>,
    manifest_entry_count: Option<usize>,
    preferred_cube_count: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Tables {
            output_dir,
            skip_existing,
            report,
        } => {
            let config = fetch_config(output_dir.clone(), skip_existing);
            let provider = ThingsTablesProvider;
            let tables_dir = provider.fetch(&config)?;
            let report_model = ThingsFetchReport {
                generated_at_utc: chrono::Utc::now().to_rfc3339(),
                mode: "tables".to_string(),
                output_root: output_dir.display().to_string(),
                tables_dir: Some(tables_dir.display().to_string()),
                metadata_csv: None,
                manifest_csv: None,
                cubes_dir: None,
                table_galaxy_count: None,
                manifest_entry_count: None,
                preferred_cube_count: None,
            };
            let report_path = report
                .unwrap_or_else(|| PathBuf::from("reports/things_tables_fetch_2026-03-13.toml"));
            write_toml_report(&report_path, &report_model)?;
            println!("Tables: {}", tables_dir.display());
            println!("Report: {}", report_path.display());
        }
        Command::Manifest {
            output,
            preferred_only,
            report,
        } => {
            let mut manifest = discover_things_cube_manifest()?;
            if preferred_only {
                manifest = preferred_things_cube_entries(&manifest);
            }
            write_things_cube_manifest_csv(&output, &manifest)?;
            let report_model = ThingsFetchReport {
                generated_at_utc: chrono::Utc::now().to_rfc3339(),
                mode: if preferred_only {
                    "manifest_preferred".to_string()
                } else {
                    "manifest_all".to_string()
                },
                output_root: output
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .display()
                    .to_string(),
                tables_dir: None,
                metadata_csv: None,
                manifest_csv: Some(output.display().to_string()),
                cubes_dir: None,
                table_galaxy_count: None,
                manifest_entry_count: Some(manifest.len()),
                preferred_cube_count: Some(
                    manifest
                        .iter()
                        .filter(|entry| entry.product_kind == "CUBE")
                        .count(),
                ),
            };
            let report_path =
                report.unwrap_or_else(|| PathBuf::from("reports/things_manifest_2026-03-13.toml"));
            write_toml_report(&report_path, &report_model)?;
            println!("Manifest rows: {}", manifest.len());
            println!("CSV:           {}", output.display());
            println!("Report:        {}", report_path.display());
        }
        Command::Metadata {
            table1,
            table4,
            output,
            default_beam_fwhm_arcsec,
            report,
        } => {
            let metadata_count =
                materialize_metadata(&table1, &table4, &output, default_beam_fwhm_arcsec)?;
            let report_model = ThingsFetchReport {
                generated_at_utc: chrono::Utc::now().to_rfc3339(),
                mode: "metadata".to_string(),
                output_root: output
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .display()
                    .to_string(),
                tables_dir: Some(
                    table1
                        .parent()
                        .unwrap_or_else(|| Path::new("."))
                        .display()
                        .to_string(),
                ),
                metadata_csv: Some(output.display().to_string()),
                manifest_csv: None,
                cubes_dir: None,
                table_galaxy_count: Some(metadata_count),
                manifest_entry_count: None,
                preferred_cube_count: None,
            };
            let report_path =
                report.unwrap_or_else(|| PathBuf::from("reports/things_metadata_2026-03-13.toml"));
            write_toml_report(&report_path, &report_model)?;
            println!("Metadata rows: {}", metadata_count);
            println!("CSV:           {}", output.display());
            println!("Report:        {}", report_path.display());
        }
        Command::Cubes {
            output_dir,
            skip_existing,
            report,
        } => {
            let config = fetch_config(output_dir.clone(), skip_existing);
            let provider = ThingsPreferredCubesProvider;
            let cubes_dir = provider.fetch(&config)?;
            let manifest_path = output_dir.join("things").join("things_cube_manifest.csv");
            let preferred_cube_count = if manifest_path.exists() {
                csv::Reader::from_path(&manifest_path)?
                    .records()
                    .filter(|row| row.is_ok())
                    .count()
            } else {
                0
            };
            let report_model = ThingsFetchReport {
                generated_at_utc: chrono::Utc::now().to_rfc3339(),
                mode: "cubes".to_string(),
                output_root: output_dir.display().to_string(),
                tables_dir: None,
                metadata_csv: None,
                manifest_csv: Some(manifest_path.display().to_string()),
                cubes_dir: Some(cubes_dir.display().to_string()),
                table_galaxy_count: None,
                manifest_entry_count: Some(preferred_cube_count),
                preferred_cube_count: Some(preferred_cube_count),
            };
            let report_path = report
                .unwrap_or_else(|| PathBuf::from("reports/things_cubes_fetch_2026-03-13.toml"));
            write_toml_report(&report_path, &report_model)?;
            println!("Preferred cubes: {}", preferred_cube_count);
            println!("Cubes:           {}", cubes_dir.display());
            println!("Report:          {}", report_path.display());
        }
        Command::Phase2 {
            output_dir,
            skip_existing,
            default_beam_fwhm_arcsec,
            report,
        } => {
            let config = fetch_config(output_dir.clone(), skip_existing);
            let tables_provider = ThingsTablesProvider;
            let tables_dir = tables_provider.fetch(&config)?;
            let metadata_csv = output_dir.join("things").join("things_metadata.csv");
            let metadata_count = materialize_metadata(
                &tables_dir.join("table1.dat"),
                &tables_dir.join("table4.dat"),
                &metadata_csv,
                default_beam_fwhm_arcsec,
            )?;
            let cubes_provider = ThingsPreferredCubesProvider;
            let cubes_dir = cubes_provider.fetch(&config)?;
            let manifest_path = output_dir.join("things").join("things_cube_manifest.csv");
            let preferred_cube_count = csv::Reader::from_path(&manifest_path)?
                .records()
                .filter(|row| row.is_ok())
                .count();
            let report_model = ThingsFetchReport {
                generated_at_utc: chrono::Utc::now().to_rfc3339(),
                mode: "phase2".to_string(),
                output_root: output_dir.display().to_string(),
                tables_dir: Some(tables_dir.display().to_string()),
                metadata_csv: Some(metadata_csv.display().to_string()),
                manifest_csv: Some(manifest_path.display().to_string()),
                cubes_dir: Some(cubes_dir.display().to_string()),
                table_galaxy_count: Some(metadata_count),
                manifest_entry_count: Some(preferred_cube_count),
                preferred_cube_count: Some(preferred_cube_count),
            };
            let report_path = report
                .unwrap_or_else(|| PathBuf::from("reports/things_phase2_fetch_2026-03-13.toml"));
            write_toml_report(&report_path, &report_model)?;
            println!("Tables:          {}", tables_dir.display());
            println!("Metadata rows:   {}", metadata_count);
            println!("Preferred cubes: {}", preferred_cube_count);
            println!("Cubes:           {}", cubes_dir.display());
            println!("Report:          {}", report_path.display());
        }
    }
    Ok(())
}

fn fetch_config(output_dir: PathBuf, skip_existing: bool) -> FetchConfig {
    FetchConfig {
        output_dir,
        skip_existing,
        verify_checksums: true,
    }
}

fn materialize_metadata(
    table1: &Path,
    table4: &Path,
    output: &Path,
    default_beam_fwhm_arcsec: f64,
) -> Result<usize> {
    let galaxies = parse_things_galaxies(table1).map_err(anyhow::Error::msg)?;
    let spectra = parse_things_hi_spectra(table4).map_err(anyhow::Error::msg)?;
    let rows = build_things_hi_metadata(&galaxies, &spectra, default_beam_fwhm_arcsec);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(output)
        .with_context(|| format!("open metadata output {}", output.display()))?;
    writer.write_record([
        "name",
        "ra_deg",
        "dec_deg",
        "distance_mpc",
        "inclination_deg",
        "pa_deg",
        "beam_fwhm_arcsec",
        "channel_width_km_s",
        "v_sys_km_s",
    ])?;
    for row in &rows {
        writer.write_record([
            row.name.as_str(),
            &format!("{:.6}", row.ra_deg),
            &format!("{:.6}", row.dec_deg),
            &format!("{:.6}", row.distance_mpc),
            &format!("{:.3}", row.inclination_deg),
            &format!("{:.3}", row.pa_deg),
            &format!("{:.3}", row.beam_fwhm_arcsec),
            &format!("{:.4}", row.channel_width_km_s),
            &format!("{:.4}", row.v_sys_km_s),
        ])?;
    }
    writer.flush()?;
    Ok(rows.len())
}

fn write_toml_report<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(value)?;
    fs::write(path, text)?;
    Ok(())
}
