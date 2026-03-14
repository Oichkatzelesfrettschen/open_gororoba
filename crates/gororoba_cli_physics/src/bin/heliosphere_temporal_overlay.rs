use anyhow::{Context, Result, bail};
use chrono::Datelike;
use clap::Parser;
use data_core::{
    catalogs::{
        ace_mag::{AceMagHourly, average_to_hourly, parse_ace_mag_file, parse_ace_mag_hapi_file},
        bepicolombo::parse_bepicolombo_file,
        cassini::parse_cassini_cruise_file,
        helios::{HeliosSpacecraft, parse_helios_file},
        ibex::{IbexOrbitRecord, parse_ibex_ena_file, parse_ibex_orbit_file},
        juno::parse_juno_cruise_file,
        new_horizons::parse_nh_swap_file,
        omni::{OmniRecord, parse_omni_file},
        psp::parse_psp_file,
        solar_orbiter::parse_solar_orbiter_file,
        solar_wind::{SwepamRecord, parse_swepam_file},
        spdf_merged::SpdfMergedRecord,
        stereo_plastic::{
            StereoMagRecord, StereoPlasticRecord, parse_stereo_magplasma_file,
            parse_stereo_plastic_file,
        },
        ulysses::parse_ulysses_file,
        voyager::{VoyagerSpacecraft, parse_voyager_file},
        voyager_crs_flux::{VoyagerCrsFluxRecord, parse_voyager_crs_flux_csv},
        wind_swe::{WindMfiRecord, WindSweRecord, parse_wind_mfi_file, parse_wind_swe_file},
    },
    time_bounds::{TimeBounds, bounds_from_omni, format_timestamp_ms},
};
use rusqlite::Connection;
use serde::Serialize;
use std::{
    collections::{BTreeSet, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(name = "heliosphere-temporal-overlay")]
#[command(about = "Time-aligned OMNI / spacecraft overlay and fleet coverage report")]
struct Cli {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    control_plane_db: PathBuf,

    #[arg(long, default_value = "data/external/omni2")]
    omni_dir: PathBuf,

    #[arg(long, default_value = "data/external/voyager1")]
    voyager1_dir: PathBuf,

    #[arg(long, default_value = "data/external/voyager2")]
    voyager2_dir: PathBuf,

    #[arg(
        long,
        default_value = "data/external/voyager_crs/voyager1_crs_daily_flux_2016_2016.csv"
    )]
    voyager1_crs: PathBuf,

    #[arg(
        long,
        default_value = "data/external/voyager_crs/voyager2_crs_daily_flux_2016_2016.csv"
    )]
    voyager2_crs: PathBuf,

    #[arg(long, default_value = "data/external/ace_mag")]
    ace_mag_dir: PathBuf,

    #[arg(long, default_value = "data/external/ace_swepam")]
    ace_swepam_dir: PathBuf,

    #[arg(long, default_value = "data/external/wind_mfi")]
    wind_mfi_dir: PathBuf,

    #[arg(long, default_value = "data/external/wind_swe")]
    wind_swe_dir: PathBuf,

    #[arg(long, default_value = "data/external/stereo_plastic")]
    stereo_plastic_dir: PathBuf,

    #[arg(long, default_value = "data/external/stereo_impact")]
    stereo_impact_dir: PathBuf,

    #[arg(long, default_value = "data/external/ulysses")]
    ulysses_dir: PathBuf,

    #[arg(long, default_value = "data/external/helios/helios1")]
    helios1_dir: PathBuf,

    #[arg(long, default_value = "data/external/helios/helios2")]
    helios2_dir: PathBuf,

    #[arg(long, default_value = "data/external/cassini")]
    cassini_dir: PathBuf,

    #[arg(long, default_value = "data/external/juno")]
    juno_dir: PathBuf,

    #[arg(long, default_value = "data/external/new_horizons")]
    new_horizons_dir: PathBuf,

    #[arg(long, default_value = "data/external/ibex")]
    ibex_dir: PathBuf,

    #[arg(long, default_value = "data/external/ibex/orbits")]
    ibex_orbit_dir: PathBuf,

    #[arg(long, default_value = "data/external/psp")]
    psp_dir: PathBuf,

    #[arg(long, default_value = "data/external/solar_orbiter")]
    solar_orbiter_dir: PathBuf,

    #[arg(long, default_value = "data/external/bepicolombo")]
    bepicolombo_dir: PathBuf,

    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct DatasetWindow {
    label: String,
    path: String,
    availability: String,
    row_count: usize,
    start_utc: Option<String>,
    end_utc: Option<String>,
    cadence_seconds: Option<f64>,
    notes: Option<String>,
}

#[derive(Debug, Clone)]
struct OverlaySeries {
    label: String,
    path: String,
    availability: bool,
    row_count: usize,
    bounds: Option<TimeBounds>,
    day_keys: BTreeSet<i32>,
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct MissionOverlay {
    mission: String,
    overlay_mode: String,
    components: Vec<DatasetWindow>,
    simultaneous_day_bins: usize,
    simultaneous_start_day: Option<String>,
    simultaneous_end_day: Option<String>,
    temporal_classification: String,
    mean_heliocentric_distance_au: Option<f64>,
    min_heliocentric_distance_au: Option<f64>,
    max_heliocentric_distance_au: Option<f64>,
    notes: Option<String>,
}

#[derive(Debug, Serialize)]
struct FleetCoverageRow {
    mission: String,
    provider_names: Vec<String>,
    fetch_registered: bool,
    cache_present: bool,
    parser_working: bool,
    source_contract_migrated: bool,
    overlay_3d_implemented: bool,
    overlay_4d_implemented: bool,
    cross_domain_integrated: bool,
    notes: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct CoverageStatus {
    parser_working: bool,
    source_contract_migrated: bool,
    overlay_3d_implemented: bool,
    overlay_4d_implemented: bool,
    cross_domain_integrated: bool,
}

#[derive(Debug, Serialize)]
struct HeliosphereTemporalOverlayReport {
    generated_at_utc: String,
    overlay_definition: String,
    control_plane_db_path: String,
    datasets: Vec<DatasetWindow>,
    overlays: Vec<MissionOverlay>,
    fleet_coverage: Vec<FleetCoverageRow>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report_path = cli.report.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "heliosphere_temporal_overlay_{}.toml",
            chrono::Utc::now().date_naive()
        ))
    });

    let contract_ids = load_contract_ids(&cli.control_plane_db)?;

    let omni = load_omni_records(&cli.omni_dir)?;
    let omni_series = overlay_series(
        "OMNI hourly",
        &cli.omni_dir,
        omni.len(),
        bounds_from_omni(&omni),
        omni_day_keys(&omni),
        None,
    );

    let voyager1 = load_merged_optional(
        &cli.voyager1_dir,
        |path| file_name_starts_with(path, "vy1_") && file_name_ends_with(path, ".asc"),
        |path| Ok(parse_voyager_file(path, VoyagerSpacecraft::V1)?),
    )?;
    let voyager2 = load_merged_optional(
        &cli.voyager2_dir,
        |path| file_name_starts_with(path, "vy2_") && file_name_ends_with(path, ".asc"),
        |path| Ok(parse_voyager_file(path, VoyagerSpacecraft::V2)?),
    )?;
    let voyager1_crs = load_crs_flux_records_optional(&cli.voyager1_crs, 1)?;
    let voyager2_crs = load_crs_flux_records_optional(&cli.voyager2_crs, 2)?;

    let ace_mag = load_ace_mag_hourly_optional(&cli.ace_mag_dir)?;
    let ace_swepam = load_records_optional(
        &cli.ace_swepam_dir,
        |path| {
            (file_name_starts_with(path, "swepam_h2s_") && file_name_ends_with(path, ".txt"))
                || (file_name_starts_with(path, "ac_h2_swe_") && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_swepam_file(path)?),
    )?;
    let wind_mfi = load_records_optional(
        &cli.wind_mfi_dir,
        |path| file_name_ends_with(path, "_wind_mag_1hour.asc"),
        |path| Ok(parse_wind_mfi_file(path)?),
    )?;
    let wind_swe = load_records_optional(
        &cli.wind_swe_dir,
        |path| file_name_starts_with(path, "wind_kp_unspike") && file_name_ends_with(path, ".txt"),
        |path| Ok(parse_wind_swe_file(path)?),
    )?;
    let stereo_plastic = load_records_optional(
        &cli.stereo_plastic_dir,
        |path| file_name_ends_with(path, ".txt") || file_name_ends_with(path, ".csv"),
        |path| Ok(parse_stereo_plastic_file(path)?),
    )?;
    let stereo_mag = load_records_optional(
        &cli.stereo_impact_dir,
        |path| file_name_ends_with(path, ".txt") || file_name_ends_with(path, ".csv"),
        |path| Ok(parse_stereo_magplasma_file(path)?),
    )?;
    let ulysses = load_merged_optional(
        &cli.ulysses_dir,
        |path| {
            (file_name_starts_with(path, "uly_") && file_name_ends_with(path, ".asc"))
                || (file_name_starts_with(path, "uy_coho1hr_") && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_ulysses_file(path)?),
    )?;
    let helios1 = load_merged_optional(
        &cli.helios1_dir,
        |path| {
            (file_name_starts_with(path, "he1_") && file_name_ends_with(path, ".asc"))
                || (file_name_starts_with(path, "helios1_") && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_helios_file(path, HeliosSpacecraft::H1)?),
    )?;
    let helios2 = load_merged_optional(
        &cli.helios2_dir,
        |path| {
            (file_name_starts_with(path, "he2_") && file_name_ends_with(path, ".asc"))
                || (file_name_starts_with(path, "helios2_") && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_helios_file(path, HeliosSpacecraft::H2)?),
    )?;
    let cassini = load_merged_optional(
        &cli.cassini_dir,
        |path| file_name_starts_with(path, "cassini_") && file_name_ends_with(path, ".asc"),
        |path| Ok(parse_cassini_cruise_file(path)?),
    )?;
    let juno = load_merged_optional(
        &cli.juno_dir,
        |path| {
            (file_name_starts_with(path, "juno_") && file_name_ends_with(path, ".asc"))
                || (file_name_starts_with(path, "juno_helio1hr_position_")
                    && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_juno_cruise_file(path)?),
    )?;
    let new_horizons = load_merged_optional(
        &cli.new_horizons_dir,
        |path| {
            (file_name_starts_with(path, "nh_swap_") && file_name_ends_with(path, ".asc"))
                || (file_name_starts_with(path, "new_horizons_helio1hr_position_")
                    && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_nh_swap_file(path)?),
    )?;
    let ibex_path = first_matching_file(&cli.ibex_dir, |path| {
        file_name_ends_with(path, ".csv") || file_name_ends_with(path, "-flux.txt")
    })?;
    let ibex_pixel_count = if let Some(path) = ibex_path.as_ref() {
        Some(
            parse_ibex_ena_file(path, 1.1, 1, 2009, "Hi")
                .with_context(|| format!("parse IBEX file {}", path.display()))?
                .pixels
                .len(),
        )
    } else {
        None
    };
    let ibex_orbit = load_records_optional(
        &cli.ibex_orbit_dir,
        |path| file_name_starts_with(path, "ibex_or_ssc_") && file_name_ends_with(path, ".csv"),
        |path| Ok(parse_ibex_orbit_file(path)?),
    )?;
    let psp = load_merged_optional(
        &cli.psp_dir,
        |path| {
            (file_name_starts_with(path, "psp_") && file_name_ends_with(path, ".asc"))
                || (file_name_starts_with(path, "psp_coho1hr_")
                    && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_psp_file(path)?),
    )?;
    let solar_orbiter = load_merged_optional(
        &cli.solar_orbiter_dir,
        |path| {
            (file_name_starts_with(path, "solo_") && file_name_ends_with(path, ".asc"))
                || (file_name_starts_with(path, "solo_coho1hr_")
                    && file_name_ends_with(path, ".csv"))
        },
        |path| Ok(parse_solar_orbiter_file(path)?),
    )?;
    let bepicolombo = load_merged_optional(
        &cli.bepicolombo_dir,
        |path| {
            file_name_starts_with(path, "bepicolombo_helio1hr_position_")
                && file_name_ends_with(path, ".csv")
        },
        |path| Ok(parse_bepicolombo_file(path)?),
    )?;

    let voyager1_series = merged_overlay_series(
        "Voyager 1 merged hourly",
        &cli.voyager1_dir,
        voyager1.as_deref(),
        None,
    );
    let voyager2_series = merged_overlay_series(
        "Voyager 2 merged hourly",
        &cli.voyager2_dir,
        voyager2.as_deref(),
        None,
    );
    let voyager1_crs_series = crs_overlay_series(
        "Voyager 1 CRS daily flux",
        &cli.voyager1_crs,
        voyager1_crs.as_deref(),
        Some(
            "Daily-flux calibration lane; used as the chronology-tight third component."
                .to_string(),
        ),
    );
    let voyager2_crs_series = crs_overlay_series(
        "Voyager 2 CRS daily flux",
        &cli.voyager2_crs,
        voyager2_crs.as_deref(),
        Some(
            "Daily-flux calibration lane; current local cache stops at 2016 for Voyager 2."
                .to_string(),
        ),
    );
    let ace_mag_series = ace_mag_overlay_series(&cli.ace_mag_dir, ace_mag.as_deref());
    let ace_swepam_series = swepam_overlay_series(&cli.ace_swepam_dir, ace_swepam.as_deref());
    let wind_mfi_series = wind_mfi_overlay_series(&cli.wind_mfi_dir, wind_mfi.as_deref());
    let wind_swe_series = wind_swe_overlay_series(&cli.wind_swe_dir, wind_swe.as_deref());
    let stereo_plastic_series =
        stereo_plastic_overlay_series(&cli.stereo_plastic_dir, stereo_plastic.as_deref());
    let stereo_mag_series =
        stereo_mag_overlay_series(&cli.stereo_impact_dir, stereo_mag.as_deref());
    let ulysses_series =
        merged_overlay_series(
            "Ulysses merged hourly",
            &cli.ulysses_dir,
            ulysses.as_deref(),
            Some("Official SPDF merged yearly files now cover the 1997-2009 post-Jupiter era for same-epoch fleet comparisons.".to_string()),
        );
    let helios1_series = merged_overlay_series(
        "Helios 1 merged hourly",
        &cli.helios1_dir,
        helios1.as_deref(),
        Some("Official SPDF merged yearly files are now the primary Rust fetch path; temporal overlap with OMNI still remains historically absent.".to_string()),
    );
    let helios2_series = merged_overlay_series(
        "Helios 2 merged hourly",
        &cli.helios2_dir,
        helios2.as_deref(),
        Some("Official SPDF merged yearly files are now the primary Rust fetch path; temporal overlap with OMNI still remains historically absent.".to_string()),
    );
    let cassini_series = merged_overlay_series(
        "Cassini cruise merged hourly",
        &cli.cassini_dir,
        cassini.as_deref(),
        Some(
            "Cruise lane is a governed hybrid with modeled plasma and measured trajectory/MAG."
                .to_string(),
        ),
    );
    let juno_series = merged_overlay_series(
        "Juno cruise merged hourly",
        &cli.juno_dir,
        juno.as_deref(),
        None,
    );
    let new_horizons_series = merged_overlay_series(
        "New Horizons SWAP hourly",
        &cli.new_horizons_dir,
        new_horizons.as_deref(),
        Some("SWAP lane provides plasma and trajectory but no in-situ magnetometer.".to_string()),
    );
    let ibex_series = overlay_series(
        "IBEX ENA sky maps",
        &cli.ibex_dir,
        ibex_pixel_count.unwrap_or(0),
        None,
        BTreeSet::new(),
        Some("ENA sky maps are governed here as a sky-map lane, not a heliocentric time-series overlay.".to_string()),
    );
    let ibex_orbit_series = ibex_orbit_overlay_series(&cli.ibex_orbit_dir, ibex_orbit.as_deref());
    let psp_series = merged_overlay_series("PSP merged hourly", &cli.psp_dir, psp.as_deref(), None);
    let solar_orbiter_series = merged_overlay_series(
        "Solar Orbiter merged hourly",
        &cli.solar_orbiter_dir,
        solar_orbiter.as_deref(),
        Some("Modern ESA/NASA inner-heliosphere lane sourced from the official CDAWeb merged hourly HAPI feed.".to_string()),
    );
    let bepicolombo_series = merged_overlay_series(
        "BepiColombo position hourly",
        &cli.bepicolombo_dir,
        bepicolombo.as_deref(),
        Some("Modern ESA/JAXA cruise-position lane sourced from the official CDAWeb heliocentric hourly support feed.".to_string()),
    );

    let overlays = vec![
        build_overlay(
            "Voyager 1",
            "omni+voyager+crs",
            vec![
                omni_series.clone(),
                voyager1_series.clone(),
                voyager1_crs_series.clone(),
            ],
            voyager1.as_deref(),
            None,
        ),
        build_overlay(
            "Voyager 2",
            "omni+voyager+crs",
            vec![
                omni_series.clone(),
                voyager2_series.clone(),
                voyager2_crs_series.clone(),
            ],
            voyager2.as_deref(),
            Some("Voyager 2 currently has a chronology-tight triple overlay through the 2016 CRS flux lane.".to_string()),
        ),
        build_overlay(
            "ACE",
            "omni+mag+plasma",
            vec![
                omni_series.clone(),
                ace_mag_series.clone(),
                ace_swepam_series.clone(),
            ],
            None,
            Some("Near-Earth cross-validation lane anchored at L1; geometry is effectively 1 AU rather than an outer-heliosphere baseline.".to_string()),
        ),
        build_overlay(
            "WIND",
            "omni+mfi+swe",
            vec![
                omni_series.clone(),
                wind_mfi_series.clone(),
                wind_swe_series.clone(),
            ],
            None,
            Some("Near-Earth cross-validation lane anchored at L1; geometry is effectively 1 AU rather than an outer-heliosphere baseline.".to_string()),
        ),
        build_overlay(
            "STEREO-A",
            "omni+plastic+mag",
            vec![
                omni_series.clone(),
                stereo_plastic_series.clone(),
                stereo_mag_series.clone(),
            ],
            None,
            Some("Spatial triangulation lane; full 4D overlay requires both PLASTIC and governed MAG text exports.".to_string()),
        ),
        build_overlay(
            "Ulysses",
            "omni+mission",
            vec![omni_series.clone(), ulysses_series.clone()],
            ulysses.as_deref(),
            Some("High-latitude heliosphere lane now targets the 1997-2009 second and third polar-scan era from the official SPDF archive.".to_string()),
        ),
        build_overlay(
            "Helios 1",
            "omni+mission",
            vec![omni_series.clone(), helios1_series.clone()],
            helios1.as_deref(),
            Some("Inner-heliosphere lane; temporal overlap with OMNI is not expected because Helios predates the OMNI local windows in this repo.".to_string()),
        ),
        build_overlay(
            "Helios 2",
            "omni+mission",
            vec![omni_series.clone(), helios2_series.clone()],
            helios2.as_deref(),
            Some("Inner-heliosphere lane; temporal overlap with OMNI is not expected because Helios predates the OMNI local windows in this repo.".to_string()),
        ),
        build_overlay(
            "Cassini",
            "omni+mission",
            vec![omni_series.clone(), cassini_series.clone()],
            cassini.as_deref(),
            Some("Late-cruise hybrid lane; overlaps OMNI in 1998-2004 when both are staged.".to_string()),
        ),
        build_overlay(
            "Juno",
            "omni+mission",
            vec![omni_series.clone(), juno_series.clone()],
            juno.as_deref(),
            Some("Cruise lane; 2016 is the strongest same-time overlap target with OMNI and Voyager-era comparison packs.".to_string()),
        ),
        build_overlay(
            "New Horizons",
            "omni+mission",
            vec![omni_series.clone(), new_horizons_series.clone()],
            new_horizons.as_deref(),
            Some("Outer-heliosphere plasma lane without in-situ magnetometer; temporal overlap is still scientifically useful for same-epoch boundary comparisons.".to_string()),
        ),
        build_overlay(
            "IBEX Orbit Companion",
            "omni+earth_orbit_support",
            vec![omni_series.clone(), ibex_orbit_series.clone()],
            None,
            Some("IBEX remains a sky-map mission in the science matrix, but its SSC orbit support series now provides a real same-epoch companion lane.".to_string()),
        ),
        build_overlay(
            "Parker Solar Probe",
            "omni+mission",
            vec![omni_series.clone(), psp_series.clone()],
            psp.as_deref(),
            Some("Modern inner-heliosphere lane sourced from official CDAWeb HAPI instead of the old AMDA bridge.".to_string()),
        ),
        build_overlay(
            "Solar Orbiter",
            "omni+mission",
            vec![omni_series.clone(), solar_orbiter_series.clone()],
            solar_orbiter.as_deref(),
            Some("Modern ESA/NASA inner-heliosphere lane sourced from the official CDAWeb merged hourly feed.".to_string()),
        ),
        build_overlay(
            "BepiColombo",
            "omni+mission",
            vec![omni_series.clone(), bepicolombo_series.clone()],
            bepicolombo.as_deref(),
            Some("Modern ESA/JAXA cruise-position lane sourced from the official CDAWeb heliocentric hourly support feed.".to_string()),
        ),
    ];

    let fleet_coverage = vec![
        build_coverage_row(
            "Voyager 1",
            &["VoyagerProvider", "VoyagerCrsFluxProvider"],
            &[&cli.voyager1_dir, &cli.voyager1_crs],
            CoverageStatus {
                parser_working: voyager1.is_some() && voyager1_crs.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-VOYAGER1-MERGED-AMDA-DERIVED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[0].simultaneous_day_bins > 0,
            },
            Some("Merged hourly + calibrated CRS daily flux are both present in the executed overlay.".to_string()),
        ),
        build_coverage_row(
            "Voyager 2",
            &["VoyagerProvider", "VoyagerCrsFluxProvider"],
            &[&cli.voyager2_dir, &cli.voyager2_crs],
            CoverageStatus {
                parser_working: voyager2.is_some() && voyager2_crs.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-VOYAGER2-MERGED-AMDA-DERIVED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[1].simultaneous_day_bins > 0,
            },
            Some("Voyager 2 remains the current best full outer-heliosphere same-time overlay lane.".to_string()),
        ),
        build_coverage_row(
            "ACE",
            &["AceMagProvider", "AceSwepamProvider"],
            &[&cli.ace_mag_dir, &cli.ace_swepam_dir],
            CoverageStatus {
                parser_working: ace_mag.is_some() && ace_swepam.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-ACE-SWEPAM-HOURLY", "SRC-ACE-MAG-HOURLY"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[2].simultaneous_day_bins > 0,
            },
            Some("Near-Earth HAPI lane now uses distinct SQLite-authored ACE MAG and ACE SWEPAM contracts.".to_string()),
        ),
        build_coverage_row(
            "WIND",
            &["WindMfiProvider", "WindSweProvider"],
            &[&cli.wind_mfi_dir, &cli.wind_swe_dir],
            CoverageStatus {
                parser_working: wind_mfi.is_some() && wind_swe.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-WIND-MFI-HOURLY", "SRC-WIND-SWE-KP"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[3].simultaneous_day_bins > 0,
            },
            Some("WIND now has dedicated SQLite-authored MFI and SWE source contracts alongside the executed OMNI overlap lane.".to_string()),
        ),
        build_coverage_row(
            "STEREO-A",
            &["StereoPlasticProvider", "StereoMagProvider"],
            &[&cli.stereo_plastic_dir, &cli.stereo_impact_dir],
            CoverageStatus {
                parser_working: stereo_plastic.is_some() && stereo_mag.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-STEREO-A-PLASTIC-1HR", "SRC-STEREO-A-IMPACT-MAG"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[4].simultaneous_day_bins > 0,
            },
            Some("STEREO-A now uses paired HAPI-driven PLASTIC and merged MAG/plasma contracts; real parity still requires both components to parse in the same run.".to_string()),
        ),
        build_coverage_row(
            "Ulysses",
            &["UlyssesProvider"],
            &[&cli.ulysses_dir],
            CoverageStatus {
                parser_working: ulysses.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-ULYSSES-SPDF-MERGED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[5].simultaneous_day_bins > 0,
            },
            Some("Second and third Ulysses polar-scan years are now sourced from the official SPDF merged archive.".to_string()),
        ),
        build_coverage_row(
            "Helios 1",
            &["HeliosProvider"],
            &[&cli.helios1_dir],
            CoverageStatus {
                parser_working: helios1.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-HELIOS1-SPDF-MERGED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[6].simultaneous_day_bins > 0,
            },
            Some("Official SPDF merged files are now fetchable in the Rust path, even though same-epoch OMNI overlap remains unavailable.".to_string()),
        ),
        build_coverage_row(
            "Helios 2",
            &["HeliosProvider"],
            &[&cli.helios2_dir],
            CoverageStatus {
                parser_working: helios2.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-HELIOS2-SPDF-MERGED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[7].simultaneous_day_bins > 0,
            },
            Some("Official SPDF merged files are now fetchable in the Rust path, even though same-epoch OMNI overlap remains unavailable.".to_string()),
        ),
        build_coverage_row(
            "Cassini",
            &["CassiniCruiseProvider"],
            &[&cli.cassini_dir],
            CoverageStatus {
                parser_working: cassini.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-CASSINI-CRUISE-AMDA-HYBRID"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[8].simultaneous_day_bins > 0,
            },
            None,
        ),
        build_coverage_row(
            "Juno",
            &["JunoCruiseProvider"],
            &[&cli.juno_dir],
            CoverageStatus {
                parser_working: juno.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-JUNO-AMDA-CRUISE-DERIVED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[9].simultaneous_day_bins > 0,
            },
            None,
        ),
        build_coverage_row(
            "New Horizons",
            &["NhSwapProvider"],
            &[&cli.new_horizons_dir],
            CoverageStatus {
                parser_working: new_horizons.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-NH-SINGLE-SOURCE-FRAGILE"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[10].simultaneous_day_bins > 0,
            },
            Some("Current lane is plasma+trajectory only; magnetometer parity with Voyager is not applicable.".to_string()),
        ),
        build_coverage_row(
            "IBEX",
            &["IbexProvider", "IbexOrbitProvider"],
            &[&cli.ibex_dir, &cli.ibex_orbit_dir],
            CoverageStatus {
                parser_working: ibex_pixel_count.is_some() || ibex_orbit.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-IBEX-ENA-MAPS", "SRC-IBEX-ORBIT-SSC"],
                ),
                overlay_3d_implemented: false,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[11].simultaneous_day_bins > 0,
            },
            Some("IBEX still contributes ENA sky maps, and now also has an official orbit support time-series for same-epoch overlay classification.".to_string()),
        ),
        build_coverage_row(
            "Parker Solar Probe",
            &["PspProvider"],
            &[&cli.psp_dir],
            CoverageStatus {
                parser_working: psp.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-PSP-COHO1HR-MERGED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[12].simultaneous_day_bins > 0,
            },
            Some("PSP now uses the official CDAWeb merged hourly feed in the Rust fleet lane.".to_string()),
        ),
        build_coverage_row(
            "Solar Orbiter",
            &["SolarOrbiterProvider"],
            &[&cli.solar_orbiter_dir],
            CoverageStatus {
                parser_working: solar_orbiter.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-SOLO-COHO1HR-MERGED"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[13].simultaneous_day_bins > 0,
            },
            Some("Solar Orbiter now uses the official CDAWeb merged hourly plasma+magnetic-field feed in the Rust fleet lane.".to_string()),
        ),
        build_coverage_row(
            "BepiColombo",
            &["BepicolomboProvider"],
            &[&cli.bepicolombo_dir],
            CoverageStatus {
                parser_working: bepicolombo.is_some(),
                source_contract_migrated: contracts_present(
                    &contract_ids,
                    &["SRC-BEPICOLOMBO-HELIO1HR-POSITION"],
                ),
                overlay_3d_implemented: true,
                overlay_4d_implemented: true,
                cross_domain_integrated: overlays[14].simultaneous_day_bins > 0,
            },
            Some("BepiColombo now has a governed heliocentric hourly support lane for same-epoch fleet overlays.".to_string()),
        ),
    ];

    let datasets = vec![
        dataset_window_from_series(&omni_series),
        dataset_window_from_series(&voyager1_series),
        dataset_window_from_series(&voyager2_series),
        dataset_window_from_series(&voyager1_crs_series),
        dataset_window_from_series(&voyager2_crs_series),
        dataset_window_from_series(&ace_mag_series),
        dataset_window_from_series(&ace_swepam_series),
        dataset_window_from_series(&wind_mfi_series),
        dataset_window_from_series(&wind_swe_series),
        dataset_window_from_series(&stereo_plastic_series),
        dataset_window_from_series(&stereo_mag_series),
        dataset_window_from_series(&ulysses_series),
        dataset_window_from_series(&helios1_series),
        dataset_window_from_series(&helios2_series),
        dataset_window_from_series(&cassini_series),
        dataset_window_from_series(&juno_series),
        dataset_window_from_series(&new_horizons_series),
        dataset_window_from_series(&ibex_series),
        dataset_window_from_series(&ibex_orbit_series),
        dataset_window_from_series(&psp_series),
        dataset_window_from_series(&solar_orbiter_series),
        dataset_window_from_series(&bepicolombo_series),
    ];

    let report = HeliosphereTemporalOverlayReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        overlay_definition: "4D overlay = shared observed epoch windows across heliocentric or near-Earth telemetry lanes using true measurement times. Near-Earth missions use OMNI as the common boundary clock; outer-heliosphere missions retain heliocentric distance summaries where available.".to_string(),
        control_plane_db_path: cli.control_plane_db.display().to_string(),
        datasets,
        overlays,
        fleet_coverage,
    };

    write_toml_report(&report_path, &report)?;
    println!("Datasets: {}", report.datasets.len());
    for overlay in &report.overlays {
        println!(
            "{} simultaneous day bins: {} ({})",
            overlay.mission, overlay.simultaneous_day_bins, overlay.temporal_classification
        );
    }
    println!("Fleet coverage rows: {}", report.fleet_coverage.len());
    println!("Report: {}", report_path.display());
    Ok(())
}

fn load_contract_ids(path: &Path) -> Result<HashSet<String>> {
    let conn = Connection::open(path)
        .with_context(|| format!("open control-plane DB {}", path.display()))?;
    let mut stmt = conn
        .prepare("select id from external_source_contracts")
        .context("prepare external_source_contracts query")?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .context("query external_source_contracts ids")?;
    let mut ids = HashSet::new();
    for row in rows {
        ids.insert(row.context("read external_source_contracts id")?);
    }
    Ok(ids)
}

fn contracts_present(contract_ids: &HashSet<String>, expected: &[&str]) -> bool {
    expected.iter().all(|id| contract_ids.contains(*id))
}

fn collect_matching_paths<F>(dir: &Path, predicate: F) -> Result<Vec<PathBuf>>
where
    F: Fn(&Path) -> bool,
{
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && predicate(path))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn first_matching_file<F>(dir: &Path, predicate: F) -> Result<Option<PathBuf>>
where
    F: Fn(&Path) -> bool,
{
    let mut paths = collect_matching_paths(dir, predicate)?;
    Ok(paths.drain(..).next())
}

fn file_name_starts_with(path: &Path, prefix: &str) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|name| name.starts_with(prefix))
        .unwrap_or(false)
}

fn file_name_ends_with(path: &Path, suffix: &str) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|name| name.ends_with(suffix))
        .unwrap_or(false)
}

fn load_records_optional<T, F, P>(dir: &Path, predicate: P, parser: F) -> Result<Option<Vec<T>>>
where
    F: Fn(&Path) -> Result<Vec<T>>,
    P: Fn(&Path) -> bool,
{
    let paths = collect_matching_paths(dir, predicate)?;
    if paths.is_empty() {
        return Ok(None);
    }
    let mut rows = Vec::new();
    for path in paths {
        rows.extend(parser(&path).with_context(|| format!("parse {}", path.display()))?);
    }
    if rows.is_empty() {
        Ok(None)
    } else {
        Ok(Some(rows))
    }
}

fn load_merged_optional<F, P>(
    dir: &Path,
    predicate: P,
    parser: F,
) -> Result<Option<Vec<SpdfMergedRecord>>>
where
    F: Fn(&Path) -> Result<Vec<SpdfMergedRecord>>,
    P: Fn(&Path) -> bool,
{
    load_records_optional(dir, predicate, parser)
}

fn load_ace_mag_hourly_optional(dir: &Path) -> Result<Option<Vec<AceMagHourly>>> {
    let paths = collect_matching_paths(dir, |path| {
        (file_name_starts_with(path, "ACE_MAG16_") && file_name_ends_with(path, ".txt"))
            || (file_name_starts_with(path, "ac_h2_mfi_") && file_name_ends_with(path, ".csv"))
    })?;
    if paths.is_empty() {
        return Ok(None);
    }
    let mut hourly = Vec::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) == Some("csv") {
            hourly.extend(
                parse_ace_mag_hapi_file(&path)
                    .with_context(|| format!("parse {}", path.display()))?,
            );
        } else {
            let raw =
                parse_ace_mag_file(&path).with_context(|| format!("parse {}", path.display()))?;
            hourly.extend(average_to_hourly(&raw));
        }
    }
    if hourly.is_empty() {
        Ok(None)
    } else {
        Ok(Some(hourly))
    }
}

fn load_omni_records(dir: &Path) -> Result<Vec<OmniRecord>> {
    if !dir.exists() {
        bail!("OMNI directory not found: {}", dir.display());
    }
    let mut paths = collect_matching_paths(dir, |path| {
        let name = path.file_name().and_then(|v| v.to_str()).unwrap_or("");
        (name.starts_with("omni2_") && name.ends_with(".dat"))
            || (name.starts_with("omni2_") && name.ends_with("_amda_hourly.csv"))
    })?;
    if paths.is_empty() {
        bail!("No OMNI files were found under {}", dir.display());
    }
    let mut records = Vec::new();
    for path in paths.drain(..) {
        records.extend(parse_omni_file(&path)?);
    }
    if records.is_empty() {
        bail!("No OMNI files were parseable under {}", dir.display());
    }
    Ok(records)
}

fn load_crs_flux_records_optional(
    path: &Path,
    spacecraft: u8,
) -> Result<Option<Vec<VoyagerCrsFluxRecord>>> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read CRS flux file {}", path.display()))?;
    let trimmed = raw.trim_start();
    if trimmed.starts_with('{')
        && trimmed.contains("\"code\": 1201")
        && trimmed.to_ascii_lowercase().contains("no data")
    {
        return Ok(None);
    }
    let (records, skipped) = parse_voyager_crs_flux_csv(&raw, spacecraft);
    if records.is_empty() {
        bail!(
            "No CRS flux rows were parseable from {} (skipped={})",
            path.display(),
            skipped
        );
    }
    Ok(Some(records))
}

fn overlay_series(
    label: &str,
    path: &Path,
    row_count: usize,
    bounds: Option<TimeBounds>,
    day_keys: BTreeSet<i32>,
    notes: Option<String>,
) -> OverlaySeries {
    OverlaySeries {
        label: label.to_string(),
        path: path.display().to_string(),
        availability: row_count > 0,
        row_count,
        bounds,
        day_keys,
        notes,
    }
}

fn merged_overlay_series(
    label: &str,
    path: &Path,
    records: Option<&[SpdfMergedRecord]>,
    notes: Option<String>,
) -> OverlaySeries {
    overlay_series(
        label,
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_merged),
        records.map(merged_day_keys).unwrap_or_default(),
        notes.or_else(|| {
            if path.exists() && records.is_none() {
                Some("Path exists but no parseable merged hourly records were found.".to_string())
            } else if !path.exists() {
                Some("Dataset path is not staged locally.".to_string())
            } else {
                None
            }
        }),
    )
}

fn crs_overlay_series(
    label: &str,
    path: &Path,
    records: Option<&[VoyagerCrsFluxRecord]>,
    notes: Option<String>,
) -> OverlaySeries {
    overlay_series(
        label,
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_crs_flux),
        records.map(crs_day_keys).unwrap_or_default(),
        notes.or_else(|| {
            if !path.exists() {
                Some("CRS daily-flux file is not staged locally.".to_string())
            } else {
                None
            }
        }),
    )
}

fn ace_mag_overlay_series(path: &Path, records: Option<&[AceMagHourly]>) -> OverlaySeries {
    overlay_series(
        "ACE MAG hourly",
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_ace_mag),
        records.map(ace_mag_day_keys).unwrap_or_default(),
        if !path.exists() {
            Some("ACE MAG browse files are not staged locally.".to_string())
        } else {
            None
        },
    )
}

fn swepam_overlay_series(path: &Path, records: Option<&[SwepamRecord]>) -> OverlaySeries {
    overlay_series(
        "ACE SWEPAM hourly",
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_swepam),
        records.map(swepam_day_keys).unwrap_or_default(),
        if !path.exists() {
            Some("ACE SWEPAM yearly files are not staged locally.".to_string())
        } else {
            None
        },
    )
}

fn wind_mfi_overlay_series(path: &Path, records: Option<&[WindMfiRecord]>) -> OverlaySeries {
    overlay_series(
        "WIND MFI hourly",
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_wind_mfi),
        records.map(wind_mfi_day_keys).unwrap_or_default(),
        if !path.exists() {
            Some("WIND MFI monthly files are not staged locally.".to_string())
        } else {
            None
        },
    )
}

fn wind_swe_overlay_series(path: &Path, records: Option<&[WindSweRecord]>) -> OverlaySeries {
    overlay_series(
        "WIND SWE plasma",
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_wind_swe),
        records.map(wind_swe_day_keys).unwrap_or_default(),
        if !path.exists() {
            Some("WIND SWE yearly files are not staged locally.".to_string())
        } else {
            None
        },
    )
}

fn stereo_plastic_overlay_series(
    path: &Path,
    records: Option<&[StereoPlasticRecord]>,
) -> OverlaySeries {
    overlay_series(
        "STEREO-A PLASTIC hourly",
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_stereo_plastic),
        records.map(stereo_plastic_day_keys).unwrap_or_default(),
        if !path.exists() {
            Some("STEREO-A PLASTIC files are not staged locally.".to_string())
        } else {
            None
        },
    )
}

fn stereo_mag_overlay_series(path: &Path, records: Option<&[StereoMagRecord]>) -> OverlaySeries {
    overlay_series(
        "STEREO-A merged MAG/plasma",
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_stereo_mag),
        records.map(stereo_mag_day_keys).unwrap_or_default(),
        if !path.exists() {
            Some("STEREO-A merged MAG/plasma files are not staged locally.".to_string())
        } else {
            Some("Current Rust lane uses the SQLite-authored HAPI contract for the merged STEREO-A MAG/plasma feed.".to_string())
        },
    )
}

fn ibex_orbit_overlay_series(path: &Path, records: Option<&[IbexOrbitRecord]>) -> OverlaySeries {
    overlay_series(
        "IBEX orbit support",
        path,
        records.map(|rows| rows.len()).unwrap_or(0),
        records.and_then(bounds_from_ibex_orbit),
        records.map(ibex_orbit_day_keys).unwrap_or_default(),
        if !path.exists() {
            Some("IBEX orbit support files are not staged locally.".to_string())
        } else {
            Some(
                "SSC orbit series complements the ENA sky-map family for same-epoch overlay only."
                    .to_string(),
            )
        },
    )
}

fn dataset_window_from_series(series: &OverlaySeries) -> DatasetWindow {
    DatasetWindow {
        label: series.label.clone(),
        path: series.path.clone(),
        availability: if series.availability {
            "available".to_string()
        } else {
            "missing".to_string()
        },
        row_count: series.row_count,
        start_utc: series
            .bounds
            .as_ref()
            .map(|bounds| format_timestamp_ms(bounds.start_ms)),
        end_utc: series
            .bounds
            .as_ref()
            .map(|bounds| format_timestamp_ms(bounds.end_ms)),
        cadence_seconds: series
            .bounds
            .as_ref()
            .and_then(|bounds| bounds.cadence_seconds),
        notes: series.notes.clone(),
    }
}

fn build_overlay(
    mission: &str,
    overlay_mode: &str,
    components: Vec<OverlaySeries>,
    distance_records: Option<&[SpdfMergedRecord]>,
    notes: Option<String>,
) -> MissionOverlay {
    let available_sets = components
        .iter()
        .filter(|series| series.availability)
        .map(|series| &series.day_keys)
        .collect::<Vec<_>>();
    let all_available = components.iter().all(|series| series.availability);
    let simultaneous = if all_available && !available_sets.is_empty() {
        intersect_day_sets(&available_sets)
    } else {
        BTreeSet::new()
    };
    let available_bounds = components
        .iter()
        .filter_map(|series| series.bounds.clone())
        .collect::<Vec<_>>();
    let temporal_classification = if all_available && !simultaneous.is_empty() {
        "simultaneous"
    } else if all_available && bounds_overlap(&available_bounds) {
        "near_contemporaneous"
    } else if available_bounds.len() >= 2 && bounds_overlap(&available_bounds) {
        "campaign_overlap_only"
    } else if components.iter().any(|series| series.availability) {
        "no_temporal_overlap"
    } else {
        "temporal_unknown"
    };

    let mean_distance =
        distance_records.and_then(|records| mean_distance_on_days(records, &simultaneous));
    let min_distance =
        distance_records.and_then(|records| min_distance_on_days(records, &simultaneous));
    let max_distance =
        distance_records.and_then(|records| max_distance_on_days(records, &simultaneous));

    MissionOverlay {
        mission: mission.to_string(),
        overlay_mode: overlay_mode.to_string(),
        components: components
            .iter()
            .map(dataset_window_from_series)
            .collect::<Vec<_>>(),
        simultaneous_day_bins: simultaneous.len(),
        simultaneous_start_day: simultaneous
            .iter()
            .next()
            .and_then(|key| day_key_to_date_string(*key)),
        simultaneous_end_day: simultaneous
            .iter()
            .next_back()
            .and_then(|key| day_key_to_date_string(*key)),
        temporal_classification: temporal_classification.to_string(),
        mean_heliocentric_distance_au: mean_distance,
        min_heliocentric_distance_au: min_distance,
        max_heliocentric_distance_au: max_distance,
        notes,
    }
}

fn build_coverage_row(
    mission: &str,
    provider_names: &[&str],
    cache_paths: &[&Path],
    status: CoverageStatus,
    notes: Option<String>,
) -> FleetCoverageRow {
    FleetCoverageRow {
        mission: mission.to_string(),
        provider_names: provider_names
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        fetch_registered: true,
        cache_present: cache_paths.iter().any(|path| path.exists()),
        parser_working: status.parser_working,
        source_contract_migrated: status.source_contract_migrated,
        overlay_3d_implemented: status.overlay_3d_implemented,
        overlay_4d_implemented: status.overlay_4d_implemented,
        cross_domain_integrated: status.cross_domain_integrated,
        notes,
    }
}

fn intersect_day_sets(sets: &[&BTreeSet<i32>]) -> BTreeSet<i32> {
    let Some((first, rest)) = sets.split_first() else {
        return BTreeSet::new();
    };
    let mut intersection = (*first).clone();
    for set in rest {
        intersection = intersection
            .intersection(set)
            .copied()
            .collect::<BTreeSet<_>>();
        if intersection.is_empty() {
            break;
        }
    }
    intersection
}

fn bounds_overlap(bounds: &[TimeBounds]) -> bool {
    if bounds.len() < 2 {
        return false;
    }
    TimeBounds::intersect_all(bounds).is_some()
}

fn ydh_timestamp_ms(year: i32, doy: u32, hour: u32) -> Option<i64> {
    let date = chrono::NaiveDate::from_yo_opt(year, doy)?;
    let datetime = date.and_hms_opt(hour, 0, 0)?;
    Some(datetime.and_utc().timestamp_millis())
}

fn ymdh_timestamp_ms(year: i32, month: u32, day: u32, hour: u32) -> Option<i64> {
    let date = chrono::NaiveDate::from_ymd_opt(year, month, day)?;
    let datetime = date.and_hms_opt(hour, 0, 0)?;
    Some(datetime.and_utc().timestamp_millis())
}

fn bounds_from_timestamps(mut timestamps: Vec<i64>) -> Option<TimeBounds> {
    timestamps.sort();
    TimeBounds::from_sorted_epoch_ms(&timestamps)
}

fn merged_timestamp_ms(record: &SpdfMergedRecord) -> Option<i64> {
    ydh_timestamp_ms(record.year as i32, record.doy as u32, record.hour as u32)
}

fn bounds_from_merged(records: &[SpdfMergedRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(merged_timestamp_ms)
            .collect::<Vec<_>>(),
    )
}

fn merged_day_keys(records: &[SpdfMergedRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn ace_mag_timestamp_ms(record: &AceMagHourly) -> Option<i64> {
    ydh_timestamp_ms(record.year as i32, record.doy as u32, record.hour as u32)
}

fn bounds_from_ace_mag(records: &[AceMagHourly]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(ace_mag_timestamp_ms)
            .collect::<Vec<_>>(),
    )
}

fn ace_mag_day_keys(records: &[AceMagHourly]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn swepam_year(record: &SwepamRecord) -> i32 {
    record.decimal_year.floor() as i32
}

fn swepam_timestamp_ms(record: &SwepamRecord) -> Option<i64> {
    ydh_timestamp_ms(swepam_year(record), record.doy as u32, record.hour as u32)
}

fn bounds_from_swepam(records: &[SwepamRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(swepam_timestamp_ms)
            .collect::<Vec<_>>(),
    )
}

fn swepam_day_keys(records: &[SwepamRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| swepam_year(record) * 1000 + record.doy as i32)
        .collect()
}

fn bounds_from_wind_mfi(records: &[WindMfiRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(|record| {
                ymdh_timestamp_ms(
                    record.year as i32,
                    record.month as u32,
                    record.day as u32,
                    record.hour as u32,
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn wind_mfi_day_keys(records: &[WindMfiRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .filter_map(|record| {
            chrono::NaiveDate::from_ymd_opt(
                record.year as i32,
                record.month as u32,
                record.day as u32,
            )
            .map(|date| record.year as i32 * 1000 + date.ordinal() as i32)
        })
        .collect()
}

fn bounds_from_wind_swe(records: &[WindSweRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(wind_swe_timestamp_ms)
            .collect::<Vec<_>>(),
    )
}

fn wind_swe_day_keys(records: &[WindSweRecord]) -> BTreeSet<i32> {
    records.iter().filter_map(wind_swe_day_key).collect()
}

fn wind_swe_day_key(record: &WindSweRecord) -> Option<i32> {
    let doy = record.decimal_doy.floor() as i32;
    if !(1..=366).contains(&doy) {
        return None;
    }
    Some(record.year as i32 * 1000 + doy)
}

fn wind_swe_timestamp_ms(record: &WindSweRecord) -> Option<i64> {
    let doy = record.decimal_doy.floor() as u32;
    let frac = record.decimal_doy - record.decimal_doy.floor();
    let hour = (frac * 24.0).floor() as u32;
    ydh_timestamp_ms(record.year as i32, doy, hour.min(23))
}

fn bounds_from_stereo_plastic(records: &[StereoPlasticRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(|record| {
                ydh_timestamp_ms(record.year as i32, record.doy as u32, record.hour as u32)
            })
            .collect::<Vec<_>>(),
    )
}

fn stereo_plastic_day_keys(records: &[StereoPlasticRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn bounds_from_stereo_mag(records: &[StereoMagRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(|record| {
                ydh_timestamp_ms(record.year as i32, record.doy as u32, record.hour as u32)
            })
            .collect::<Vec<_>>(),
    )
}

fn stereo_mag_day_keys(records: &[StereoMagRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn bounds_from_ibex_orbit(records: &[IbexOrbitRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(|record| {
                ydh_timestamp_ms(record.year as i32, record.doy as u32, record.hour as u32)
            })
            .collect::<Vec<_>>(),
    )
}

fn ibex_orbit_day_keys(records: &[IbexOrbitRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn bounds_from_crs_flux(records: &[VoyagerCrsFluxRecord]) -> Option<TimeBounds> {
    bounds_from_timestamps(
        records
            .iter()
            .filter_map(|record| decimal_year_to_day_key(record.decimal_year))
            .filter_map(day_key_to_timestamp_ms)
            .collect::<Vec<_>>(),
    )
}

fn omni_day_keys(records: &[OmniRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .map(|record| record.year as i32 * 1000 + record.doy as i32)
        .collect()
}

fn crs_day_keys(records: &[VoyagerCrsFluxRecord]) -> BTreeSet<i32> {
    records
        .iter()
        .filter_map(|record| decimal_year_to_day_key(record.decimal_year))
        .collect()
}

fn decimal_year_to_day_key(decimal_year: f64) -> Option<i32> {
    if !decimal_year.is_finite() {
        return None;
    }
    let year = decimal_year.floor() as i32;
    let frac = decimal_year - year as f64;
    let days_in_year = if is_leap_year(year) { 366.0 } else { 365.0 };
    let doy = (frac * days_in_year).floor() as i32 + 1;
    if !(1..=366).contains(&doy) {
        return None;
    }
    Some(year * 1000 + doy)
}

fn day_key_to_date_string(day_key: i32) -> Option<String> {
    let year = day_key / 1000;
    let doy = (day_key % 1000) as u32;
    chrono::NaiveDate::from_yo_opt(year, doy).map(|date| date.to_string())
}

fn day_key_to_timestamp_ms(day_key: i32) -> Option<i64> {
    let year = day_key / 1000;
    let doy = (day_key % 1000) as u32;
    ydh_timestamp_ms(year, doy, 0)
}

fn mean_distance_on_days(records: &[SpdfMergedRecord], day_keys: &BTreeSet<i32>) -> Option<f64> {
    let values = records
        .iter()
        .filter(|record| day_keys.contains(&(record.year as i32 * 1000 + record.doy as i32)))
        .map(|record| record.distance_au)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn min_distance_on_days(records: &[SpdfMergedRecord], day_keys: &BTreeSet<i32>) -> Option<f64> {
    records
        .iter()
        .filter(|record| day_keys.contains(&(record.year as i32 * 1000 + record.doy as i32)))
        .map(|record| record.distance_au)
        .filter(|value| value.is_finite())
        .reduce(f64::min)
}

fn max_distance_on_days(records: &[SpdfMergedRecord], day_keys: &BTreeSet<i32>) -> Option<f64> {
    records
        .iter()
        .filter(|record| day_keys.contains(&(record.year as i32 * 1000 + record.doy as i32)))
        .map(|record| record.distance_au)
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

const fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn write_toml_report<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, toml::to_string_pretty(value)?)?;
    Ok(())
}
