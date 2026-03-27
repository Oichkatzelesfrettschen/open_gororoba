use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, Utc};
use clap::Parser;
use csv::WriterBuilder;
use data_core::{
    HeliosphereFeatureCube, HeliosphereFeatureCubeManifest, HeliosphereFeatureRow,
    catalogs::{
        ace_mag::{ace_mag_to_omni, parse_ace_mag_hapi_file},
        cassini::{cassini_to_omni, parse_cassini_cruise_file},
        helios::{HeliosSpacecraft, helios_to_omni, parse_helios_file},
        ibex::{parse_ibex_ena_file, parse_ibex_orbit_file},
        imap::{
            parse_imap_helio1hr_file, parse_imap_hi_h90_file, parse_imap_ialirt_file,
            parse_imap_ialirt_live_day,
        },
        imp8::{imp8_to_omni, parse_imp8_file},
        juno::{juno_to_omni, parse_juno_cruise_file},
        new_horizons::{nh_swap_to_omni, parse_nh_swap_file},
        omni::{OmniRecord, parse_omni_file},
        psp::{parse_psp_file, psp_to_omni},
        psp_fields::parse_psp_fields_file,
        psp_spc::parse_psp_spc_l3i_file,
        psp_spi::parse_psp_spi_mom_file,
        psp_sqtn::parse_psp_sqtn_file,
        soho_celias::{
            parse_soho_celias_bundle_file, parse_soho_celias_cdf_file,
            parse_soho_lasco_img_hdr_file, soho_to_hourly_omni,
        },
        solar_orbiter::{parse_solar_orbiter_file, solar_orbiter_to_omni},
        solar_orbiter_mag::parse_solar_orbiter_mag_file,
        solar_orbiter_rpw::parse_solar_orbiter_rpw_file,
        solar_orbiter_rpw_density::parse_solar_orbiter_rpw_density_file,
        solar_orbiter_rpw_hfr::parse_solar_orbiter_rpw_hfr_file,
        solar_orbiter_rpw_tnr::parse_solar_orbiter_rpw_tnr_file,
        solar_orbiter_swa::parse_solar_orbiter_swa_file,
        solar_wind::parse_swepam_file,
        stereo_plastic::{
            average_stereo_mag_hourly, parse_stereo_magplasma_file, parse_stereo_plastic_file,
            stereo_to_omni,
        },
        ulysses::{parse_ulysses_file, ulysses_to_omni},
        voyager::{VoyagerSpacecraft, parse_voyager_file, voyager_to_omni},
        voyager_crs_flux::{VoyagerCrsFluxRecord, parse_voyager_crs_flux_csv},
        voyager_pws::parse_voyager_pws_file,
        wind_swe::{merge_wind_swe_mfi, parse_wind_mfi_file, parse_wind_swe_file},
    },
    cdf_support::filename_date_yyyymmdd,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-feature-cube",
    about = "Build normalized heliosphere feature cubes from real mission data"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "fleet2016")]
    window: String,

    #[arg(long)]
    out_csv: Option<PathBuf>,

    #[arg(long)]
    out_manifest: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let window = cli.window.trim().to_ascii_lowercase();
    let date = Utc::now().date_naive();
    let out_csv = cli.out_csv.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!("heliosphere_feature_cube_{window}_{date}.csv"))
    });
    let out_manifest = cli.out_manifest.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!("heliosphere_feature_cube_{window}_{date}.toml"))
    });

    let cube = build_cube(&cli.repo_root, &window)?;
    if let Some(parent) = out_csv.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = out_manifest.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(&out_csv)
        .with_context(|| format!("create {}", out_csv.display()))?;
    for row in &cube.rows {
        writer.serialize(row)?;
    }
    writer.flush()?;

    fs::write(&out_manifest, toml::to_string_pretty(&cube.manifest)?)
        .with_context(|| format!("write {}", out_manifest.display()))?;

    println!("window = {}", cube.manifest.window_name);
    println!("rows = {}", cube.manifest.row_count);
    println!("missions = {}", cube.manifest.missions.join(", "));
    println!("products = {}", cube.manifest.products.join(", "));
    println!("csv = {}", out_csv.display());
    println!("manifest = {}", out_manifest.display());
    Ok(())
}

fn build_cube(repo_root: &Path, window: &str) -> Result<HeliosphereFeatureCube> {
    let mut rows = Vec::new();
    let mut sources = Vec::new();
    let mut notes = Vec::new();
    match window {
        "fleet2016" => build_fleet2016(repo_root, &mut rows, &mut sources, &mut notes)?,
        "inner1976" => build_inner1976(repo_root, &mut rows, &mut sources, &mut notes)?,
        "modern2020" => build_modern2020(repo_root, &mut rows, &mut sources, &mut notes)?,
        "outer2001" => build_outer2001(repo_root, &mut rows, &mut sources, &mut notes)?,
        "boundary2009" => build_boundary2009(repo_root, &mut rows, &mut sources, &mut notes)?,
        "remote2024" => build_remote2024(repo_root, &mut rows, &mut sources, &mut notes)?,
        "imap2025" => build_imap2025(repo_root, &mut rows, &mut sources, &mut notes)?,
        "imap2026" => build_imap2026(repo_root, &mut rows, &mut sources, &mut notes)?,
        "psp2025" => build_psp2025(repo_root, &mut rows, &mut sources, &mut notes)?,
        "mms2024" => build_mms2024(repo_root, &mut rows, &mut sources, &mut notes)?,
        "full-heliosphere" => {
            build_inner1976(repo_root, &mut rows, &mut sources, &mut notes)?;
            build_outer2001(repo_root, &mut rows, &mut sources, &mut notes)?;
            build_fleet2016(repo_root, &mut rows, &mut sources, &mut notes)?;
            build_modern2020(repo_root, &mut rows, &mut sources, &mut notes)?;
        }
        "densified" => {
            // Start from the full-heliosphere baseline
            build_inner1976(repo_root, &mut rows, &mut sources, &mut notes)?;
            build_outer2001(repo_root, &mut rows, &mut sources, &mut notes)?;
            build_fleet2016(repo_root, &mut rows, &mut sources, &mut notes)?;
            build_modern2020(repo_root, &mut rows, &mut sources, &mut notes)?;
            // Add all available Voyager and Ulysses year files not already covered
            ingest_all_voyager_years(repo_root, &mut rows, &mut sources, &mut notes)?;
            ingest_all_ulysses_years(repo_root, &mut rows, &mut sources, &mut notes)?;
            ingest_all_nh_swap_years(repo_root, &mut rows, &mut sources, &mut notes)?;
            ingest_all_cassini_cruise_years(repo_root, &mut rows, &mut sources, &mut notes)?;
        }
        other => bail!(
            "unsupported window {other}; expected fleet2016, inner1976, modern2020, outer2001, boundary2009, remote2024, imap2025, imap2026, psp2025, mms2024, full-heliosphere, or densified"
        ),
    }

    rows.sort_by_key(|row| {
        (
            row.year,
            row.doy,
            row.hour,
            row.mission.clone(),
            row.product.clone(),
        )
    });

    let missions = collect_sorted(rows.iter().map(|row| row.mission.clone()));
    let products = collect_sorted(
        rows.iter()
            .map(|row| format!("{}:{}", row.mission, row.product)),
    );
    let temporal_start_utc = rows
        .first()
        .and_then(|row| timestamp_utc(row.year, row.doy, row.hour));
    let temporal_end_utc = rows
        .last()
        .and_then(|row| timestamp_utc(row.year, row.doy, row.hour));

    Ok(HeliosphereFeatureCube {
        manifest: HeliosphereFeatureCubeManifest {
            window_name: window.to_string(),
            generated_at_utc: Utc::now().to_rfc3339(),
            temporal_start_utc,
            temporal_end_utc,
            row_count: rows.len(),
            missions,
            products,
            channel_names: data_core::HELIOSPHERE_CHANNEL_NAMES
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            source_paths: collect_sorted(sources),
            notes,
        },
        rows,
    })
}

fn build_fleet2016(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let omni_path = repo_root.join("data/external/omni2/omni2_2016_amda_hourly.csv");
    let omni = parse_omni_file(&omni_path)?;
    sources.push(rel(repo_root, &omni_path));
    push_omni_rows(rows, "fleet2016", "OMNI", "Hourly merged", &omni);

    let ace_mag_path = repo_root.join("data/external/ace_mag/ac_h2_mfi_2016.csv");
    if ace_mag_path.exists() {
        let ace_mag = parse_ace_mag_hapi_file(&ace_mag_path)?;
        sources.push(rel(repo_root, &ace_mag_path));
        push_omni_rows(
            rows,
            "fleet2016",
            "ACE",
            "MAG hourly",
            &ace_mag_to_omni(&ace_mag),
        );
    }

    let ace_swe_path = repo_root.join("data/external/ace_swepam/ac_h2_swe_2016.csv");
    if ace_swe_path.exists() {
        let ace_swe = parse_swepam_file(&ace_swe_path)?;
        sources.push(rel(repo_root, &ace_swe_path));
        for record in ace_swe {
            rows.push(HeliosphereFeatureRow {
                window_name: "fleet2016".to_string(),
                mission: "ACE".to_string(),
                product: "SWEPAM hourly".to_string(),
                year: record.decimal_year.floor() as u16,
                doy: record.doy,
                hour: record.hour,
                r_au: 1.0,
                lat_deg: f64::NAN,
                lon_deg: f64::NAN,
                density_cm3: record.proton_density,
                speed_kms: record.bulk_speed,
                temperature_k: record.ion_temperature,
                bx: f64::NAN,
                by: f64::NAN,
                bz: f64::NAN,
                b_mag: f64::NAN,
                crs_flux: f64::NAN,
                spectral_mean: f64::NAN,
                spectral_peak: f64::NAN,
                map_flux_mean: f64::NAN,
                map_flux_std: f64::NAN,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            });
        }
    }

    let wind_swe_path = repo_root.join("data/external/wind_swe/wind_kp_unspike2016.txt");
    let wind_mfi_paths =
        collect_matching_files(&repo_root.join("data/external/wind_mfi"), |path| {
            file_name_ends_with(path, "_wind_mag_1hour.asc")
        });
    if wind_swe_path.exists() && !wind_mfi_paths.is_empty() {
        let swe = parse_wind_swe_file(&wind_swe_path)?;
        let mut mfi = Vec::new();
        for path in &wind_mfi_paths {
            mfi.extend(parse_wind_mfi_file(path)?);
            sources.push(rel(repo_root, path));
        }
        sources.push(rel(repo_root, &wind_swe_path));
        push_omni_rows(
            rows,
            "fleet2016",
            "WIND",
            "SWE+MFI merged",
            &merge_wind_swe_mfi(&swe, &mfi),
        );
    }

    let stereo_plastic_path =
        repo_root.join("data/external/stereo_plastic/sta_l2_pla_1dmax_1hr_2016.csv");
    let stereo_mag_path =
        repo_root.join("data/external/stereo_impact/sta_coho1hr_merged_mag_plasma_2016.csv");
    if stereo_plastic_path.exists() && stereo_mag_path.exists() {
        let plastic = parse_stereo_plastic_file(&stereo_plastic_path)?;
        let mag = average_stereo_mag_hourly(&parse_stereo_magplasma_file(&stereo_mag_path)?);
        sources.push(rel(repo_root, &stereo_plastic_path));
        sources.push(rel(repo_root, &stereo_mag_path));
        push_omni_rows(
            rows,
            "fleet2016",
            "STEREO-A",
            "PLASTIC+MAG merged",
            &stereo_to_omni(&plastic, &mag, 0.0),
        );
    }

    let v1_path = repo_root.join("data/external/voyager1/vy1_2016.asc");
    let v2_path = repo_root.join("data/external/voyager2/vy2_2016.asc");
    let v1_omni = if v1_path.exists() {
        let merged = parse_voyager_file(&v1_path, VoyagerSpacecraft::V1)?;
        sources.push(rel(repo_root, &v1_path));
        let omni_rows = voyager_to_omni(&merged);
        push_omni_rows(rows, "fleet2016", "Voyager 1", "Merged support", &omni_rows);
        Some(omni_rows)
    } else {
        None
    };
    let v2_omni = if v2_path.exists() {
        let merged = parse_voyager_file(&v2_path, VoyagerSpacecraft::V2)?;
        sources.push(rel(repo_root, &v2_path));
        let omni_rows = voyager_to_omni(&merged);
        push_omni_rows(rows, "fleet2016", "Voyager 2", "Merged support", &omni_rows);
        Some(omni_rows)
    } else {
        None
    };

    let v1_support = v1_omni
        .as_deref()
        .map(build_support_index)
        .unwrap_or_default();
    let v2_support = v2_omni
        .as_deref()
        .map(build_support_index)
        .unwrap_or_default();

    let v1_crs_path =
        repo_root.join("data/external/voyager_crs/voyager1_crs_daily_flux_2016_2016.csv");
    if v1_crs_path.exists() {
        let crs_text = fs::read_to_string(&v1_crs_path)
            .with_context(|| format!("read {}", v1_crs_path.display()))?;
        let (records, _skipped) = parse_voyager_crs_flux_csv(&crs_text, 1);
        sources.push(rel(repo_root, &v1_crs_path));
        push_crs_rows(rows, "fleet2016", "Voyager 1", &records);
    }
    let v2_crs_path =
        repo_root.join("data/external/voyager_crs/voyager2_crs_daily_flux_2016_2016.csv");
    if v2_crs_path.exists() {
        let crs_text = fs::read_to_string(&v2_crs_path)
            .with_context(|| format!("read {}", v2_crs_path.display()))?;
        let (records, _skipped) = parse_voyager_crs_flux_csv(&crs_text, 2);
        sources.push(rel(repo_root, &v2_crs_path));
        push_crs_rows(rows, "fleet2016", "Voyager 2", &records);
    }

    let v1_pws_path = repo_root.join("data/external/voyager/pws/v1/v1_pws_lr_2016.csv");
    if v1_pws_path.exists() {
        let records = parse_voyager_pws_file(&v1_pws_path)?;
        sources.push(rel(repo_root, &v1_pws_path));
        push_pws_rows(rows, "fleet2016", "Voyager 1", &records, &v1_support);
    } else {
        notes.push("Voyager 1 PWS file missing; run fetch-datasets for 'Voyager PWS Low Rate' to populate it.".to_string());
    }
    let v2_pws_path = repo_root.join("data/external/voyager/pws/v2/v2_pws_lr_2016.csv");
    if v2_pws_path.exists() {
        let records = parse_voyager_pws_file(&v2_pws_path)?;
        sources.push(rel(repo_root, &v2_pws_path));
        push_pws_rows(rows, "fleet2016", "Voyager 2", &records, &v2_support);
    } else {
        notes.push("Voyager 2 PWS file missing; run fetch-datasets for 'Voyager PWS Low Rate' to populate it.".to_string());
    }

    let juno_path = repo_root.join("data/external/juno/juno_helio1hr_position_2016.csv");
    if juno_path.exists() {
        let records = parse_juno_cruise_file(&juno_path)?;
        sources.push(rel(repo_root, &juno_path));
        push_omni_rows(
            rows,
            "fleet2016",
            "Juno",
            "Cruise support",
            &juno_to_omni(&records),
        );
    }

    let nh_path =
        repo_root.join("data/external/new_horizons/new_horizons_helio1hr_position_2016.csv");
    if nh_path.exists() {
        let records = parse_nh_swap_file(&nh_path)?;
        sources.push(rel(repo_root, &nh_path));
        push_omni_rows(
            rows,
            "fleet2016",
            "New Horizons",
            "SWAP+support",
            &nh_swap_to_omni(&records),
        );
    }

    let ibex_orbit_path = repo_root.join("data/external/ibex/orbits/ibex_or_ssc_2016.csv");
    if ibex_orbit_path.exists() {
        let records = parse_ibex_orbit_file(&ibex_orbit_path)?;
        sources.push(rel(repo_root, &ibex_orbit_path));
        for record in records {
            rows.push(HeliosphereFeatureRow {
                window_name: "fleet2016".to_string(),
                mission: "IBEX".to_string(),
                product: "Orbit support".to_string(),
                year: record.year,
                doy: record.doy,
                hour: record.hour,
                r_au: record.radius_re / 23_454.8,
                lat_deg: f64::NAN,
                lon_deg: f64::NAN,
                density_cm3: f64::NAN,
                speed_kms: f64::NAN,
                temperature_k: f64::NAN,
                bx: f64::NAN,
                by: f64::NAN,
                bz: f64::NAN,
                b_mag: f64::NAN,
                crs_flux: f64::NAN,
                spectral_mean: f64::NAN,
                spectral_peak: f64::NAN,
                map_flux_mean: f64::NAN,
                map_flux_std: f64::NAN,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            });
        }
    }

    Ok(())
}

fn build_inner1976(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let imp8_paths = collect_matching_files(&repo_root.join("data/external/imp8"), |path| {
        file_name_starts_with(path, "imp_min_merge1976") && file_name_ends_with(path, ".asc")
    });
    if imp8_paths.is_empty() {
        notes.push(
            "IMP 8 merged 1-minute files for 1976 not found; run fetch-datasets for 'IMP 8 Merged 1-minute' to populate them."
                .to_string(),
        );
    } else {
        let mut imp8_hourly = Vec::new();
        for path in &imp8_paths {
            imp8_hourly.extend(parse_imp8_file(path)?);
            sources.push(rel(repo_root, path));
        }
        push_omni_rows(
            rows,
            "inner1976",
            "IMP 8",
            "Merged 1-minute hourly aggregate",
            &imp8_to_omni(&imp8_hourly),
        );
    }

    let helios1_paths =
        collect_matching_files(&repo_root.join("data/external/helios/helios1"), |path| {
            file_name_starts_with(path, "he1_1976") && file_name_ends_with(path, ".asc")
        });
    if helios1_paths.is_empty() {
        notes.push(
            "Helios 1 1976 merged files not found; run fetch-datasets for 'Helios 1 Merged Hourly' to populate them."
                .to_string(),
        );
    } else {
        let mut all = Vec::new();
        for path in &helios1_paths {
            all.extend(parse_helios_file(path, HeliosSpacecraft::H1)?);
            sources.push(rel(repo_root, path));
        }
        push_omni_rows(
            rows,
            "inner1976",
            "Helios 1",
            "Merged mission support",
            &helios_to_omni(&all),
        );
    }

    let helios2_paths =
        collect_matching_files(&repo_root.join("data/external/helios/helios2"), |path| {
            file_name_starts_with(path, "he2_1976") && file_name_ends_with(path, ".asc")
        });
    if helios2_paths.is_empty() {
        notes.push(
            "Helios 2 1976 merged files not found; run fetch-datasets for 'Helios 2 Merged Hourly' to populate them."
                .to_string(),
        );
    } else {
        let mut all = Vec::new();
        for path in &helios2_paths {
            all.extend(parse_helios_file(path, HeliosSpacecraft::H2)?);
            sources.push(rel(repo_root, path));
        }
        push_omni_rows(
            rows,
            "inner1976",
            "Helios 2",
            "Merged mission support",
            &helios_to_omni(&all),
        );
    }

    notes.push(
        "inner1976 is the executed historical same-epoch lane for Helios 1/2, anchored to the official SPDF IMP 8 merged reference archive rather than modern OMNI windows."
            .to_string(),
    );
    Ok(())
}

fn build_modern2020(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let omni_path = repo_root.join("data/external/omni2/omni2_2020_amda_hourly.csv");
    if omni_path.exists() {
        let omni = parse_omni_file(&omni_path)?;
        sources.push(rel(repo_root, &omni_path));
        push_omni_rows(rows, "modern2020", "OMNI", "Hourly merged", &omni);
    }

    let soho_bundle_path =
        repo_root.join("data/external/soho/celias/CELIAS_Proton_Monitor_5min.tar.gz");
    let soho_cdf_paths = collect_matching_files(
        &repo_root.join("data/external/soho/celias_pm_5min/2020"),
        |path| file_name_ends_with(path, ".cdf"),
    );
    if !soho_cdf_paths.is_empty() {
        let mut soho_rows = Vec::new();
        for path in &soho_cdf_paths {
            soho_rows.extend(parse_soho_celias_cdf_file(path)?);
            sources.push(rel(repo_root, path));
        }
        let hourly = soho_to_hourly_omni(&soho_rows);
        push_omni_rows(
            rows,
            "modern2020",
            "SOHO",
            "CELIAS daily CDF hourly",
            &hourly,
        );
        notes.push(format!(
            "SOHO modern2020 used {} native daily CELIAS CDF files.",
            soho_cdf_paths.len()
        ));
        if soho_cdf_paths.len() < 366 {
            notes.push(format!(
                "SOHO modern2020 native daily CDF coverage is still partial for 2020 ({} of 366 daily files cached).",
                soho_cdf_paths.len()
            ));
        }
    } else if soho_bundle_path.exists() {
        let soho = parse_soho_celias_bundle_file(&soho_bundle_path)?;
        sources.push(rel(repo_root, &soho_bundle_path));
        let hourly = soho_to_hourly_omni(
            &soho
                .into_iter()
                .filter(|row| row.year == 2020)
                .collect::<Vec<_>>(),
        );
        push_omni_rows(rows, "modern2020", "SOHO", "CELIAS bundle hourly", &hourly);
        notes.push("SOHO modern2020 fell back to the mission-long bundle because no 2020 daily CDFs were cached.".to_string());
    }

    let psp_path = repo_root.join("data/external/psp/psp_coho1hr_merged_mag_plasma_2020.csv");
    let psp_omni = if psp_path.exists() {
        let merged = parse_psp_file(&psp_path)?;
        sources.push(rel(repo_root, &psp_path));
        let omni_rows = psp_to_omni(&merged);
        push_omni_rows(
            rows,
            "modern2020",
            "Parker Solar Probe",
            "Merged hourly",
            &omni_rows,
        );
        Some(omni_rows)
    } else {
        None
    };

    let psp_support = psp_omni
        .as_deref()
        .map(build_support_index)
        .unwrap_or_default();
    let mut psp_fields_paths = collect_matching_files(
        &repo_root.join("data/external/psp/fields_l2_mag_rtn_1min"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "psp_fld_l2_mag_rtn_1min_2020")
        },
    );
    let modern_psp_days = psp_fields_paths
        .iter()
        .filter_map(|path| file_date_key(path))
        .collect::<BTreeSet<_>>();
    psp_fields_paths.extend(collect_matching_files(
        &repo_root.join("data/external/psp/berkeley_fields"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "psp_fld_l2_mag_rtn_2020_")
                && file_date_key(path)
                    .map(|date| !modern_psp_days.contains(&date))
                    .unwrap_or(true)
        },
    ));
    if psp_fields_paths.is_empty() {
        psp_fields_paths = collect_matching_files(
            &repo_root.join("data/external/psp/fields_l2_mag_rtn_1min"),
            |path| {
                path.extension().and_then(|value| value.to_str()) == Some("cdf")
                    && file_name_starts_with(path, "psp_fld_l2_mag_rtn_1min_2020")
            },
        );
    }
    if !psp_fields_paths.is_empty() {
        let mut parsed_psp_fields = 0usize;
        let mut skipped_psp_fields = 0usize;
        for path in &psp_fields_paths {
            match parse_psp_fields_file(path) {
                Ok(records) => {
                    parsed_psp_fields += 1;
                    sources.push(rel(repo_root, path));
                    for record in records {
                        let support = psp_support.get(&(record.year, record.doy, record.hour));
                        rows.push(HeliosphereFeatureRow {
                            window_name: "modern2020".to_string(),
                            mission: "Parker Solar Probe".to_string(),
                            product: "FIELDS MAG RTN".to_string(),
                            year: record.year,
                            doy: record.doy,
                            hour: record.hour,
                            r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                            lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                            lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                            density_cm3: f64::NAN,
                            speed_kms: f64::NAN,
                            temperature_k: f64::NAN,
                            bx: record.br,
                            by: record.bt,
                            bz: record.bn,
                            b_mag: record.b_magnitude,
                            crs_flux: f64::NAN,
                            spectral_mean: f64::NAN,
                            spectral_peak: f64::NAN,
                            map_flux_mean: f64::NAN,
                            map_flux_std: f64::NAN,
                            event_score: None,
                            event_mask: None,
                            event_segment_id: None,
                        });
                    }
                }
                Err(err) => {
                    skipped_psp_fields += 1;
                    notes.push(format!(
                        "PSP FIELDS file {} was skipped in modern2020: {}",
                        rel(repo_root, path),
                        err
                    ));
                }
            }
        }
        notes.push(format!(
            "PSP FIELDS modern2020 currently uses {} bounded daily files, preferring parser-friendly daily CSV mirrors over staged direct CDFs.",
            parsed_psp_fields
        ));
        if skipped_psp_fields > 0 {
            notes.push(format!(
                "PSP FIELDS modern2020 skipped {} daily files after parser validation.",
                skipped_psp_fields
            ));
        }
        if parsed_psp_fields < 60 {
            notes.push(format!(
                "PSP FIELDS modern2020 coverage is still partial for the bounded Jan-Feb 2020 target window ({} of 60 daily files executed).",
                parsed_psp_fields
            ));
        }
    } else {
        notes.push("PSP FIELDS files missing; run fetch-datasets for 'Parker Solar Probe FIELDS MAG RTN' to populate them.".to_string());
    }

    let solo_path =
        repo_root.join("data/external/solar_orbiter/solo_coho1hr_merged_mag_plasma_2020.csv");
    let solo_omni = if solo_path.exists() {
        let merged = parse_solar_orbiter_file(&solo_path)?;
        sources.push(rel(repo_root, &solo_path));
        let omni_rows = solar_orbiter_to_omni(&merged);
        push_omni_rows(
            rows,
            "modern2020",
            "Solar Orbiter",
            "Merged hourly",
            &omni_rows,
        );
        Some(omni_rows)
    } else {
        None
    };
    let solo_support = solo_omni
        .as_deref()
        .map(build_support_index)
        .unwrap_or_default();
    let solo_swa_path =
        repo_root.join("data/external/solo/soar_swa_pas/solo_l2_swa_pas_grnd_mom_2020.csv");
    if solo_swa_path.exists() {
        let records = parse_solar_orbiter_swa_file(&solo_swa_path)?;
        sources.push(rel(repo_root, &solo_swa_path));
        for record in records {
            let support = solo_support.get(&(record.year, record.doy, record.hour));
            rows.push(HeliosphereFeatureRow {
                window_name: "modern2020".to_string(),
                mission: "Solar Orbiter".to_string(),
                product: "SWA-PAS ground moments".to_string(),
                year: record.year,
                doy: record.doy,
                hour: record.hour,
                r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                density_cm3: record.proton_density,
                speed_kms: record.bulk_speed,
                temperature_k: record.proton_temperature,
                bx: f64::NAN,
                by: f64::NAN,
                bz: f64::NAN,
                b_mag: f64::NAN,
                crs_flux: f64::NAN,
                spectral_mean: f64::NAN,
                spectral_peak: f64::NAN,
                map_flux_mean: f64::NAN,
                map_flux_std: f64::NAN,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            });
        }
    } else {
        notes.push("Solar Orbiter SWA-PAS file missing; run fetch-datasets for 'Solar Orbiter SWA-PAS Ground Moments' to populate it.".to_string());
    }

    let solo_mag_paths = collect_matching_files(
        &repo_root.join("data/external/solar_orbiter/mag_rtn_normal_1min"),
        |path| file_name_starts_with(path, "solo_l2_mag_rtn_normal_1minute_2020"),
    );
    if !solo_mag_paths.is_empty() {
        for path in &solo_mag_paths {
            let records = parse_solar_orbiter_mag_file(path)?;
            sources.push(rel(repo_root, path));
            for record in records {
                let support = solo_support.get(&(record.year, record.doy, record.hour));
                rows.push(HeliosphereFeatureRow {
                    window_name: "modern2020".to_string(),
                    mission: "Solar Orbiter".to_string(),
                    product: "MAG RTN 1-minute".to_string(),
                    year: record.year,
                    doy: record.doy,
                    hour: record.hour,
                    r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                    lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                    lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                    density_cm3: f64::NAN,
                    speed_kms: f64::NAN,
                    temperature_k: f64::NAN,
                    bx: record.br,
                    by: record.bt,
                    bz: record.bn,
                    b_mag: record.b_magnitude,
                    crs_flux: f64::NAN,
                    spectral_mean: f64::NAN,
                    spectral_peak: f64::NAN,
                    map_flux_mean: f64::NAN,
                    map_flux_std: f64::NAN,
                    event_score: None,
                    event_mask: None,
                    event_segment_id: None,
                });
            }
        }
        notes.push(format!(
            "Solar Orbiter modern2020 includes {} MAG RTN 1-minute daily files.",
            solo_mag_paths.len()
        ));
    } else {
        notes.push("Solar Orbiter MAG RTN 1-minute files missing; run fetch-datasets for 'Solar Orbiter MAG RTN 1-minute (2020)' to populate them.".to_string());
    }

    let solo_rpw_paths = collect_matching_files(
        &repo_root.join("data/external/solar_orbiter/rpw_bia_scpot_10s"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "solo_l3_rpw-bia-scpot-10-seconds_2020")
        },
    );
    if !solo_rpw_paths.is_empty() {
        let mut parsed_solo_rpw = 0usize;
        let mut skipped_solo_rpw = 0usize;
        for path in &solo_rpw_paths {
            match parse_solar_orbiter_rpw_file(path) {
                Ok(records) => {
                    parsed_solo_rpw += 1;
                    sources.push(rel(repo_root, path));
                    for record in records {
                        let support = solo_support.get(&(record.year, record.doy, record.hour));
                        rows.push(HeliosphereFeatureRow {
                            window_name: "modern2020".to_string(),
                            mission: "Solar Orbiter".to_string(),
                            product: "RPW BIA SCPOT 10-second".to_string(),
                            year: record.year,
                            doy: record.doy,
                            hour: record.hour,
                            r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                            lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                            lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                            density_cm3: f64::NAN,
                            speed_kms: f64::NAN,
                            temperature_k: f64::NAN,
                            bx: f64::NAN,
                            by: f64::NAN,
                            bz: f64::NAN,
                            b_mag: f64::NAN,
                            crs_flux: f64::NAN,
                            spectral_mean: record.scpot,
                            spectral_peak: record.psp,
                            map_flux_mean: f64::NAN,
                            map_flux_std: f64::NAN,
                            event_score: None,
                            event_mask: None,
                            event_segment_id: None,
                        });
                    }
                }
                Err(err) => {
                    skipped_solo_rpw += 1;
                    notes.push(format!(
                        "Solar Orbiter RPW file {} was skipped in modern2020: {}",
                        rel(repo_root, path),
                        err
                    ));
                }
            }
        }
        notes.push(format!(
            "Solar Orbiter modern2020 includes {} RPW BIA SCPOT daily CSV mirrors, with the official direct CDFs retained only for provenance staging.",
            parsed_solo_rpw
        ));
        if skipped_solo_rpw > 0 {
            notes.push(format!(
                "Solar Orbiter modern2020 skipped {} RPW BIA SCPOT daily CSV mirrors after parser validation.",
                skipped_solo_rpw
            ));
        }
    } else {
        notes.push("Solar Orbiter RPW BIA SCPOT files missing; run fetch-datasets for 'Solar Orbiter RPW BIA SCPOT 10-second (2020)' to populate them.".to_string());
    }

    let solo_density_paths = collect_matching_files(
        &repo_root.join("data/external/solar_orbiter/rpw_bia_density"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "solo_l3_rpw-bia-density_2020")
        },
    );
    if !solo_density_paths.is_empty() {
        let mut parsed_solo_density = 0usize;
        let mut skipped_solo_density = 0usize;
        for path in &solo_density_paths {
            match parse_solar_orbiter_rpw_density_file(path) {
                Ok(records) => {
                    parsed_solo_density += 1;
                    sources.push(rel(repo_root, path));
                    for record in records {
                        let support = solo_support.get(&(record.year, record.doy, record.hour));
                        rows.push(HeliosphereFeatureRow {
                            window_name: "modern2020".to_string(),
                            mission: "Solar Orbiter".to_string(),
                            product: "RPW BIA density".to_string(),
                            year: record.year,
                            doy: record.doy,
                            hour: record.hour,
                            r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                            lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                            lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                            density_cm3: record.density_cm3,
                            speed_kms: f64::NAN,
                            temperature_k: f64::NAN,
                            bx: f64::NAN,
                            by: f64::NAN,
                            bz: f64::NAN,
                            b_mag: f64::NAN,
                            crs_flux: f64::NAN,
                            spectral_mean: f64::NAN,
                            spectral_peak: f64::NAN,
                            map_flux_mean: f64::NAN,
                            map_flux_std: f64::NAN,
                            event_score: None,
                            event_mask: None,
                            event_segment_id: None,
                        });
                    }
                }
                Err(err) => {
                    skipped_solo_density += 1;
                    notes.push(format!(
                        "Solar Orbiter RPW density file {} was skipped in modern2020: {}",
                        rel(repo_root, path),
                        err
                    ));
                }
            }
        }
        notes.push(format!(
            "Solar Orbiter modern2020 includes {} RPW BIA density daily CSV mirrors.",
            parsed_solo_density
        ));
        if skipped_solo_density > 0 {
            notes.push(format!(
                "Solar Orbiter modern2020 skipped {} RPW BIA density daily CSV mirrors after parser validation.",
                skipped_solo_density
            ));
        }
    } else {
        notes.push("Solar Orbiter RPW BIA density files missing; run fetch-datasets for 'Solar Orbiter RPW BIA Density (2020)' to populate them.".to_string());
    }

    let solo_hfr_paths = collect_matching_files(
        &repo_root.join("data/external/solar_orbiter/rpw_hfr_surv_flux/2020"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "solo_l3_rpw-hfr-surv-flux_2020")
        },
    );
    if !solo_hfr_paths.is_empty() {
        let mut parsed_solo_hfr = 0usize;
        let mut skipped_solo_hfr = 0usize;
        for path in &solo_hfr_paths {
            match parse_solar_orbiter_rpw_hfr_file(path) {
                Ok(records) => {
                    parsed_solo_hfr += 1;
                    sources.push(rel(repo_root, path));
                    for record in records {
                        rows.push(HeliosphereFeatureRow {
                            window_name: "modern2020".to_string(),
                            mission: "Solar Orbiter".to_string(),
                            product: "RPW HFR survey flux".to_string(),
                            year: record.year,
                            doy: record.doy,
                            hour: record.hour,
                            r_au: record.r_au,
                            lat_deg: record.lat_deg,
                            lon_deg: record.lon_deg,
                            density_cm3: f64::NAN,
                            speed_kms: f64::NAN,
                            temperature_k: f64::NAN,
                            bx: f64::NAN,
                            by: f64::NAN,
                            bz: f64::NAN,
                            b_mag: f64::NAN,
                            crs_flux: f64::NAN,
                            spectral_mean: record.spectral_mean,
                            spectral_peak: record.spectral_peak,
                            map_flux_mean: f64::NAN,
                            map_flux_std: f64::NAN,
                            event_score: None,
                            event_mask: None,
                            event_segment_id: None,
                        });
                    }
                }
                Err(err) => {
                    skipped_solo_hfr += 1;
                    notes.push(format!(
                        "Solar Orbiter RPW HFR file {} was skipped in modern2020: {}",
                        rel(repo_root, path),
                        err
                    ));
                }
            }
        }
        notes.push(format!(
            "Solar Orbiter modern2020 includes {} RPW HFR survey-flux daily CSV mirrors.",
            parsed_solo_hfr
        ));
        if skipped_solo_hfr > 0 {
            notes.push(format!(
                "Solar Orbiter modern2020 skipped {} RPW HFR survey-flux daily CSV mirrors after parser validation.",
                skipped_solo_hfr
            ));
        }
    } else {
        notes.push("Solar Orbiter RPW HFR survey-flux files missing; run fetch-datasets for 'Solar Orbiter RPW HFR Survey Flux (2020)' to populate them.".to_string());
    }

    let solo_tnr_paths = collect_matching_files(
        &repo_root.join("data/external/solar_orbiter/rpw_tnr_surv_flux/2020"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "solo_l3_rpw-tnr-surv-flux_2020")
        },
    );
    if !solo_tnr_paths.is_empty() {
        let mut parsed_solo_tnr = 0usize;
        let mut skipped_solo_tnr = 0usize;
        for path in &solo_tnr_paths {
            match parse_solar_orbiter_rpw_tnr_file(path) {
                Ok(records) => {
                    parsed_solo_tnr += 1;
                    sources.push(rel(repo_root, path));
                    for record in records {
                        rows.push(HeliosphereFeatureRow {
                            window_name: "modern2020".to_string(),
                            mission: "Solar Orbiter".to_string(),
                            product: "RPW TNR survey flux".to_string(),
                            year: record.year,
                            doy: record.doy,
                            hour: record.hour,
                            r_au: record.r_au,
                            lat_deg: record.lat_deg,
                            lon_deg: record.lon_deg,
                            density_cm3: f64::NAN,
                            speed_kms: f64::NAN,
                            temperature_k: f64::NAN,
                            bx: f64::NAN,
                            by: f64::NAN,
                            bz: f64::NAN,
                            b_mag: f64::NAN,
                            crs_flux: f64::NAN,
                            spectral_mean: record.spectral_mean,
                            spectral_peak: record.spectral_peak,
                            map_flux_mean: f64::NAN,
                            map_flux_std: f64::NAN,
                            event_score: None,
                            event_mask: None,
                            event_segment_id: None,
                        });
                    }
                }
                Err(err) => {
                    skipped_solo_tnr += 1;
                    notes.push(format!(
                        "Solar Orbiter RPW TNR file {} was skipped in modern2020: {}",
                        rel(repo_root, path),
                        err
                    ));
                }
            }
        }
        notes.push(format!(
            "Solar Orbiter modern2020 includes {} RPW TNR survey-flux daily CSV mirrors.",
            parsed_solo_tnr
        ));
        if skipped_solo_tnr > 0 {
            notes.push(format!(
                "Solar Orbiter modern2020 skipped {} RPW TNR survey-flux daily CSV mirrors after parser validation.",
                skipped_solo_tnr
            ));
        }
    } else {
        notes.push("Solar Orbiter RPW TNR survey-flux files missing; run fetch-datasets for 'Solar Orbiter RPW TNR Survey Flux (2020)' to populate them.".to_string());
    }

    let bepi_path =
        repo_root.join("data/external/bepicolombo/bepicolombo_helio1hr_position_2020.csv");
    if bepi_path.exists() {
        let records = data_core::catalogs::bepicolombo::parse_bepicolombo_file(&bepi_path)?;
        sources.push(rel(repo_root, &bepi_path));
        push_omni_rows(
            rows,
            "modern2020",
            "BepiColombo",
            "Helio1hr support",
            &data_core::catalogs::bepicolombo::bepicolombo_to_omni(&records),
        );
    }

    Ok(())
}

fn build_outer2001(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let omni_path = repo_root.join("data/external/omni2/omni2_2001_amda_hourly.csv");
    if omni_path.exists() {
        let omni = parse_omni_file(&omni_path)?;
        sources.push(rel(repo_root, &omni_path));
        push_omni_rows(rows, "outer2001", "OMNI", "Hourly merged", &omni);
    } else {
        notes.push(
            "OMNI 2001 file missing; run fetch-datasets for 'OMNI Hourly' to populate it."
                .to_string(),
        );
    }

    let ulysses_path = repo_root.join("data/external/ulysses/uly_2001.asc");
    if ulysses_path.exists() {
        let merged = parse_ulysses_file(&ulysses_path)?;
        sources.push(rel(repo_root, &ulysses_path));
        push_omni_rows(
            rows,
            "outer2001",
            "Ulysses",
            "Merged mission support",
            &ulysses_to_omni(&merged),
        );
    } else {
        notes.push("Ulysses 2001 merged file missing; run fetch-datasets for 'Ulysses Merged Hourly' to populate it.".to_string());
    }

    let cassini_path = repo_root.join("data/external/cassini/cassini_2001_amda_cruise_hourly.asc");
    if cassini_path.exists() {
        let merged = parse_cassini_cruise_file(&cassini_path)?;
        sources.push(rel(repo_root, &cassini_path));
        push_omni_rows(
            rows,
            "outer2001",
            "Cassini",
            "Cruise hybrid",
            &cassini_to_omni(&merged),
        );
    } else {
        notes.push("Cassini 2001 cruise file missing; run fetch-datasets for 'Cassini Cruise Merged Hourly' to populate it.".to_string());
    }

    Ok(())
}

fn build_boundary2009(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let ibex_flux_path =
        collect_matching_files(&repo_root.join("data/external/ibex/release17"), |path| {
            file_name_ends_with(path, "-flux.txt")
        })
        .into_iter()
        .next();
    if let Some(path) = ibex_flux_path {
        let map = parse_ibex_ena_file(&path, 1.1, 1, 2009, "Lo")?;
        sources.push(rel(repo_root, &path));
        let fluxes: Vec<f64> = map
            .pixels
            .iter()
            .map(|pixel| pixel.flux)
            .filter(|value| value.is_finite())
            .collect();
        let mean = mean(&fluxes);
        let std = stddev(&fluxes, mean);
        rows.push(HeliosphereFeatureRow {
            window_name: "boundary2009".to_string(),
            mission: "IBEX".to_string(),
            product: "ENA sky-map summary".to_string(),
            year: 2009,
            doy: 1,
            hour: 0,
            r_au: f64::NAN,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
            density_cm3: f64::NAN,
            speed_kms: f64::NAN,
            temperature_k: f64::NAN,
            bx: f64::NAN,
            by: f64::NAN,
            bz: f64::NAN,
            b_mag: f64::NAN,
            crs_flux: f64::NAN,
            spectral_mean: f64::NAN,
            spectral_peak: f64::NAN,
            map_flux_mean: mean,
            map_flux_std: std,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        });
    } else {
        notes.push("IBEX release17 flux maps not found; run fetch-datasets for 'IBEX ENA Sky Maps' to populate them.".to_string());
    }
    Ok(())
}

fn build_remote2024(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let header_paths = collect_matching_files(
        &repo_root.join("data/external/soho/lasco/level05"),
        |path| file_name_ends_with(path, "img_hdr.txt"),
    );
    if header_paths.is_empty() {
        notes.push(
            "SOHO LASCO sample headers not found; run fetch-datasets for 'SOHO LASCO L0.5 Day Sample' to populate them."
                .to_string(),
        );
        return Ok(());
    }

    let mut grouped: BTreeMap<(String, u16, u16, u8), Vec<f64>> = BTreeMap::new();
    for path in &header_paths {
        let records = parse_soho_lasco_img_hdr_file(path)?;
        sources.push(rel(repo_root, path));
        for record in records.into_iter().filter(|record| record.year == 2024) {
            grouped
                .entry((record.camera.clone(), record.year, record.doy, record.hour))
                .or_default()
                .push(record.exposure_seconds);
        }
    }

    for ((camera, year, doy, hour), exposures) in grouped {
        let exposure_mean = mean(&exposures);
        rows.push(HeliosphereFeatureRow {
            window_name: "remote2024".to_string(),
            mission: "SOHO".to_string(),
            product: format!("LASCO {camera} sample"),
            year,
            doy,
            hour,
            r_au: 1.0,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
            density_cm3: f64::NAN,
            speed_kms: f64::NAN,
            temperature_k: f64::NAN,
            bx: f64::NAN,
            by: f64::NAN,
            bz: f64::NAN,
            b_mag: f64::NAN,
            crs_flux: f64::NAN,
            spectral_mean: f64::NAN,
            spectral_peak: f64::NAN,
            map_flux_mean: exposure_mean,
            map_flux_std: stddev(&exposures, exposure_mean),
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        });
    }

    Ok(())
}

fn build_imap2025(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let helio_paths =
        collect_matching_files(&repo_root.join("data/external/imap/helio1hr"), |path| {
            file_name_ends_with(path, ".cdf")
        });
    if helio_paths.is_empty() {
        notes.push(
            "IMAP helio1hr CDFs not found; run fetch-datasets for 'IMAP Helio1hr Position' to populate them."
                .to_string(),
        );
        return Ok(());
    }

    for path in &helio_paths {
        let records = parse_imap_helio1hr_file(path)?;
        sources.push(rel(repo_root, path));
        for record in records {
            rows.push(HeliosphereFeatureRow {
                window_name: "imap2025".to_string(),
                mission: "IMAP".to_string(),
                product: "Helio1hr support".to_string(),
                year: record.year,
                doy: record.doy,
                hour: record.hour,
                r_au: record.r_au,
                lat_deg: record.lat_deg,
                lon_deg: record.lon_deg,
                density_cm3: f64::NAN,
                speed_kms: f64::NAN,
                temperature_k: f64::NAN,
                bx: f64::NAN,
                by: f64::NAN,
                bz: f64::NAN,
                b_mag: f64::NAN,
                crs_flux: f64::NAN,
                spectral_mean: f64::NAN,
                spectral_peak: f64::NAN,
                map_flux_mean: f64::NAN,
                map_flux_std: f64::NAN,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            });
        }
    }
    Ok(())
}

fn build_imap2026(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let science_paths = collect_matching_files(
        &repo_root.join("data/external/imap/ialirt/space_weather"),
        |path| {
            file_name_starts_with(path, "imap_ialirt_space_weather_science_")
                && file_name_ends_with(path, ".json")
        },
    );
    if !science_paths.is_empty() {
        let spacecraft_map = collect_matching_files(
            &repo_root.join("data/external/imap/ialirt/space_weather"),
            |path| {
                file_name_starts_with(path, "imap_ialirt_space_weather_spacecraft_")
                    && file_name_ends_with(path, ".json")
            },
        )
        .into_iter()
        .filter_map(|path| file_date_key(&path).map(|date| (date, path)))
        .collect::<BTreeMap<_, _>>();
        let mut parsed_live_days = 0usize;
        let mut skipped_live_days = 0usize;
        for science_path in &science_paths {
            let Some(date_key) = file_date_key(science_path) else {
                skipped_live_days += 1;
                notes.push(format!(
                    "Skipping IMAP I-ALiRT live JSON file {} because its date could not be inferred.",
                    rel(repo_root, science_path)
                ));
                continue;
            };
            let Some(spacecraft_path) = spacecraft_map.get(&date_key) else {
                skipped_live_days += 1;
                notes.push(format!(
                    "Skipping IMAP I-ALiRT live JSON file {} because the matching spacecraft support JSON is missing.",
                    rel(repo_root, science_path)
                ));
                continue;
            };
            match parse_imap_ialirt_live_day(science_path, Some(spacecraft_path.as_path())) {
                Ok(records) => {
                    parsed_live_days += 1;
                    sources.push(rel(repo_root, science_path));
                    sources.push(rel(repo_root, spacecraft_path));
                    for record in records {
                        rows.push(HeliosphereFeatureRow {
                            window_name: "imap2026".to_string(),
                            mission: "IMAP".to_string(),
                            product: "I-ALiRT live API".to_string(),
                            year: record.year,
                            doy: record.doy,
                            hour: record.hour,
                            r_au: record.r_au,
                            lat_deg: record.lat_deg,
                            lon_deg: record.lon_deg,
                            density_cm3: record.pseudo_density,
                            speed_kms: record.pseudo_speed,
                            temperature_k: record.pseudo_temperature,
                            bx: record.br,
                            by: record.bt,
                            bz: record.bn,
                            b_mag: record.b_magnitude,
                            crs_flux: f64::NAN,
                            spectral_mean: record.spectral_mean,
                            spectral_peak: record.spectral_peak,
                            map_flux_mean: f64::NAN,
                            map_flux_std: f64::NAN,
                            event_score: None,
                            event_mask: None,
                            event_segment_id: None,
                        });
                    }
                }
                Err(err) => {
                    skipped_live_days += 1;
                    notes.push(format!(
                        "Skipping IMAP I-ALiRT live JSON day {} because {}",
                        rel(repo_root, science_path),
                        err
                    ));
                }
            }
        }
        notes.push(format!(
            "IMAP I-ALiRT imap2026 prefers the official public live JSON API and executed {} day-pairs successfully; {} day-pairs were skipped.",
            parsed_live_days,
            skipped_live_days
        ));
    }

    let ialirt_paths = collect_matching_files(
        &repo_root.join("data/external/imap/ialirt/l1/realtime"),
        |path| file_name_ends_with(path, ".cdf"),
    );
    if science_paths.is_empty() && ialirt_paths.is_empty() {
        notes.push(
            "IMAP I-ALiRT live JSON and realtime CDF files were not found; run fetch-datasets for 'IMAP I-ALiRT L1 Realtime' to populate them."
                .to_string(),
        );
    } else if science_paths.is_empty() {
        let mut parsed_ialirt_files = 0usize;
        let mut skipped_ialirt_files = 0usize;
        for path in &ialirt_paths {
            match parse_imap_ialirt_file(path) {
                Ok(records) => {
                    parsed_ialirt_files += 1;
                    sources.push(rel(repo_root, path));
                    for record in records {
                        rows.push(HeliosphereFeatureRow {
                            window_name: "imap2026".to_string(),
                            mission: "IMAP".to_string(),
                            product: "I-ALiRT realtime".to_string(),
                            year: record.year,
                            doy: record.doy,
                            hour: record.hour,
                            r_au: record.r_au,
                            lat_deg: record.lat_deg,
                            lon_deg: record.lon_deg,
                            density_cm3: record.pseudo_density,
                            speed_kms: record.pseudo_speed,
                            temperature_k: record.pseudo_temperature,
                            bx: record.br,
                            by: record.bt,
                            bz: record.bn,
                            b_mag: record.b_magnitude,
                            crs_flux: f64::NAN,
                            spectral_mean: record.spectral_mean,
                            spectral_peak: record.spectral_peak,
                            map_flux_mean: f64::NAN,
                            map_flux_std: f64::NAN,
                            event_score: None,
                            event_mask: None,
                            event_segment_id: None,
                        });
                    }
                }
                Err(err) => {
                    skipped_ialirt_files += 1;
                    notes.push(format!(
                        "Skipping IMAP I-ALiRT file {} because {}",
                        rel(repo_root, path),
                        err
                    ));
                }
            }
        }
        notes.push(format!(
            "IMAP I-ALiRT realtime staged {} daily CDF files; {} parsed successfully and {} were skipped by the current Rust CDF reader.",
            ialirt_paths.len(),
            parsed_ialirt_files,
            skipped_ialirt_files
        ));
    }

    let ena_paths =
        collect_matching_files(&repo_root.join("data/external/imap/hi/l2/h90"), |path| {
            file_name_ends_with(path, ".cdf")
        });
    if ena_paths.is_empty() {
        notes.push(
            "IMAP-Hi h90 CDFs not found; run fetch-datasets for 'IMAP-Hi L2 ENA h90' to populate them."
                .to_string(),
        );
        return Ok(());
    }

    for path in &ena_paths {
        let summary = parse_imap_hi_h90_file(path)?;
        sources.push(rel(repo_root, path));
        rows.push(HeliosphereFeatureRow {
            window_name: "imap2026".to_string(),
            mission: "IMAP".to_string(),
            product: "IMAP-Hi h90 ENA summary".to_string(),
            year: summary.year,
            doy: summary.doy,
            hour: summary.hour,
            r_au: f64::NAN,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
            density_cm3: f64::NAN,
            speed_kms: f64::NAN,
            temperature_k: f64::NAN,
            bx: f64::NAN,
            by: f64::NAN,
            bz: f64::NAN,
            b_mag: f64::NAN,
            crs_flux: f64::NAN,
            spectral_mean: f64::NAN,
            spectral_peak: f64::NAN,
            map_flux_mean: summary.map_flux_mean,
            map_flux_std: summary.map_flux_std,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        });
    }
    Ok(())
}

fn build_psp2025(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let psp_path = repo_root.join("data/external/psp/psp_coho1hr_merged_mag_plasma_2025.csv");
    let psp_omni = if psp_path.exists() {
        let merged = parse_psp_file(&psp_path)?;
        sources.push(rel(repo_root, &psp_path));
        let omni_rows = psp_to_omni(&merged);
        push_omni_rows(
            rows,
            "psp2025",
            "Parker Solar Probe",
            "Merged hourly",
            &omni_rows,
        );
        Some(omni_rows)
    } else {
        notes.push("PSP merged 2025 file missing; run fetch-datasets for 'Parker Solar Probe Merged Hourly (2025)' to populate it.".to_string());
        None
    };

    let psp_support = psp_omni
        .as_deref()
        .map(build_support_index)
        .unwrap_or_default();
    let sqtn_paths = collect_matching_files(
        &repo_root.join("data/external/psp/sqtn_rfs_v1v2/2025"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "psp_fld_l3_sqtn_rfs_v1v2_2025")
        },
    );
    if sqtn_paths.is_empty() {
        notes.push("PSP SQTN 2025 files missing; run fetch-datasets for 'Parker Solar Probe SQTN RFS V1V2' to populate them.".to_string());
        return Ok(());
    }

    let mut parsed_sqtn = 0usize;
    let mut skipped_sqtn = 0usize;
    for path in &sqtn_paths {
        match parse_psp_sqtn_file(path) {
            Ok(records) => {
                parsed_sqtn += 1;
                sources.push(rel(repo_root, path));
                for record in records {
                    let support = psp_support.get(&(record.year, record.doy, record.hour));
                    rows.push(HeliosphereFeatureRow {
                        window_name: "psp2025".to_string(),
                        mission: "Parker Solar Probe".to_string(),
                        product: "SQTN RFS V1V2".to_string(),
                        year: record.year,
                        doy: record.doy,
                        hour: record.hour,
                        r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                        lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                        lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                        density_cm3: record.electron_density_cm3,
                        speed_kms: f64::NAN,
                        temperature_k: record.electron_core_temperature_k,
                        bx: f64::NAN,
                        by: f64::NAN,
                        bz: f64::NAN,
                        b_mag: f64::NAN,
                        crs_flux: f64::NAN,
                        spectral_mean: f64::NAN,
                        spectral_peak: f64::NAN,
                        map_flux_mean: f64::NAN,
                        map_flux_std: f64::NAN,
                        event_score: None,
                        event_mask: None,
                        event_segment_id: None,
                    });
                }
            }
            Err(err) => {
                skipped_sqtn += 1;
                notes.push(format!(
                    "PSP SQTN file {} was skipped in psp2025: {}",
                    rel(repo_root, path),
                    err
                ));
            }
        }
    }
    notes.push(format!(
        "PSP psp2025 includes {} SQTN daily files, with {} skipped after parser validation.",
        parsed_sqtn, skipped_sqtn
    ));

    let spi_paths = collect_matching_files(
        &repo_root.join("data/external/psp/sweap_spi_sf00_l3_mom/2025"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "psp_swp_spi_sf00_l3_mom_2025")
        },
    );
    if spi_paths.is_empty() {
        notes.push("PSP SWEAP SPI SF00 L3 moment files missing; run fetch-datasets for 'Parker Solar Probe SWEAP SPI SF00 L3 moments' to populate them.".to_string());
        return Ok(());
    }

    let mut parsed_spi = 0usize;
    let mut skipped_spi = 0usize;
    for path in &spi_paths {
        match parse_psp_spi_mom_file(path) {
            Ok(records) => {
                parsed_spi += 1;
                sources.push(rel(repo_root, path));
                for record in records {
                    let support = psp_support.get(&(record.year, record.doy, record.hour));
                    rows.push(HeliosphereFeatureRow {
                        window_name: "psp2025".to_string(),
                        mission: "Parker Solar Probe".to_string(),
                        product: "SWEAP SPI SF00 L3 moments".to_string(),
                        year: record.year,
                        doy: record.doy,
                        hour: record.hour,
                        r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                        lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                        lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                        density_cm3: record.density_cm3,
                        speed_kms: record.speed_kms,
                        temperature_k: record.temperature_k,
                        bx: f64::NAN,
                        by: f64::NAN,
                        bz: f64::NAN,
                        b_mag: f64::NAN,
                        crs_flux: f64::NAN,
                        spectral_mean: f64::NAN,
                        spectral_peak: f64::NAN,
                        map_flux_mean: f64::NAN,
                        map_flux_std: f64::NAN,
                        event_score: None,
                        event_mask: None,
                        event_segment_id: None,
                    });
                }
            }
            Err(err) => {
                skipped_spi += 1;
                notes.push(format!(
                    "PSP SPI moment file {} was skipped in psp2025: {}",
                    rel(repo_root, path),
                    err
                ));
            }
        }
    }
    notes.push(format!(
        "PSP psp2025 includes {} SWEAP SPI SF00 L3 moment daily files, with {} skipped after parser validation.",
        parsed_spi, skipped_spi
    ));

    let spc_paths = collect_matching_files(
        &repo_root.join("data/external/psp/sweap_spc_l3i/2025"),
        |path| {
            path.extension().and_then(|value| value.to_str()) == Some("csv")
                && file_name_starts_with(path, "psp_swp_spc_l3i_2025")
        },
    );
    let mut parsed_spc = 0usize;
    let mut skipped_spc = 0usize;
    for path in &spc_paths {
        match parse_psp_spc_l3i_file(path) {
            Ok(records) => {
                parsed_spc += 1;
                sources.push(rel(repo_root, path));
                for record in records {
                    let support = psp_support.get(&(record.year, record.doy, record.hour));
                    rows.push(HeliosphereFeatureRow {
                        window_name: "psp2025".to_string(),
                        mission: "Parker Solar Probe".to_string(),
                        product: "SWEAP SPC L3 ion moments".to_string(),
                        year: record.year,
                        doy: record.doy,
                        hour: record.hour,
                        r_au: support.map(|value| value.0).unwrap_or(f64::NAN),
                        lat_deg: support.map(|value| value.1).unwrap_or(f64::NAN),
                        lon_deg: support.map(|value| value.2).unwrap_or(f64::NAN),
                        density_cm3: record.density_cm3,
                        speed_kms: record.speed_kms,
                        temperature_k: record.temperature_k,
                        bx: f64::NAN,
                        by: f64::NAN,
                        bz: f64::NAN,
                        b_mag: f64::NAN,
                        crs_flux: f64::NAN,
                        spectral_mean: f64::NAN,
                        spectral_peak: f64::NAN,
                        map_flux_mean: f64::NAN,
                        map_flux_std: f64::NAN,
                        event_score: None,
                        event_mask: None,
                        event_segment_id: None,
                    });
                }
            }
            Err(err) => {
                skipped_spc += 1;
                notes.push(format!(
                    "PSP SPC ion-moment file {} was skipped in psp2025: {}",
                    rel(repo_root, path),
                    err
                ));
            }
        }
    }
    notes.push(format!(
        "PSP psp2025 includes {} SWEAP SPC L3 ion-moment daily files, with {} skipped after parser validation.",
        parsed_spc, skipped_spc
    ));
    Ok(())
}

fn build_mms2024(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let mms_paths = collect_matching_files(&repo_root.join("data/external/mms"), |path| {
        file_name_starts_with(path, "mms1_fgm_srvy_l2_2024") && file_name_ends_with(path, ".csv")
    });
    if mms_paths.is_empty() {
        notes.push("MMS FGM 2024 files not found; run fetch-datasets for 'MMS1 FGM Survey L2 (1-day sample)' to populate them.".to_string());
        return Ok(());
    }

    for path in &mms_paths {
        let text = fs::read_to_string(path)?;
        let records = data_core::catalogs::mms::parse_mms_fgm_hapi_csv(&text);
        sources.push(rel(repo_root, path));
        let hourly = data_core::catalogs::mms::average_to_hourly(&records);
        for record in hourly {
            rows.push(HeliosphereFeatureRow {
                window_name: "mms2024".to_string(),
                mission: "MMS".to_string(),
                product: "FGM Survey L2 hourly aggregate".to_string(),
                year: record.year,
                doy: record.doy,
                hour: record.hour,
                r_au: 1.0, // MMS is in Earth orbit, roughly 1 AU heliocentric
                lat_deg: f64::NAN,
                lon_deg: f64::NAN,
                density_cm3: f64::NAN,
                speed_kms: f64::NAN,
                temperature_k: f64::NAN,
                bx: record.bx_gse,
                by: record.by_gse,
                bz: record.bz_gse,
                b_mag: record.b_magnitude,
                crs_flux: f64::NAN,
                spectral_mean: f64::NAN,
                spectral_peak: f64::NAN,
                map_flux_mean: f64::NAN,
                map_flux_std: f64::NAN,
                event_score: None,
                event_mask: None,
                event_segment_id: None,
            });
        }
    }
    Ok(())
}

fn push_omni_rows(
    rows: &mut Vec<HeliosphereFeatureRow>,
    window_name: &str,
    mission: &str,
    product: &str,
    records: &[OmniRecord],
) {
    for record in records {
        let b_mag = if record.b_magnitude.is_finite() {
            record.b_magnitude
        } else if record.bx_gse.is_finite()
            || record.by_gse.is_finite()
            || record.bz_gse.is_finite()
        {
            let bx = if record.bx_gse.is_finite() {
                record.bx_gse
            } else {
                0.0
            };
            let by = if record.by_gse.is_finite() {
                record.by_gse
            } else {
                0.0
            };
            let bz = if record.bz_gse.is_finite() {
                record.bz_gse
            } else {
                0.0
            };
            (bx * bx + by * by + bz * bz).sqrt()
        } else {
            f64::NAN
        };
        rows.push(HeliosphereFeatureRow {
            window_name: window_name.to_string(),
            mission: mission.to_string(),
            product: product.to_string(),
            year: record.year,
            doy: record.doy,
            hour: record.hour,
            r_au: record.r_au,
            lat_deg: record.lat_deg,
            lon_deg: record.lon_deg,
            density_cm3: record.proton_density,
            speed_kms: record.bulk_speed,
            temperature_k: record.proton_temperature,
            bx: record.bx_gse,
            by: record.by_gse,
            bz: record.bz_gse,
            b_mag,
            crs_flux: f64::NAN,
            spectral_mean: f64::NAN,
            spectral_peak: f64::NAN,
            map_flux_mean: f64::NAN,
            map_flux_std: f64::NAN,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        });
    }
}

fn push_crs_rows(
    rows: &mut Vec<HeliosphereFeatureRow>,
    window_name: &str,
    mission: &str,
    records: &[VoyagerCrsFluxRecord],
) {
    for record in records {
        let Some((year, doy, hour)) = decimal_year_to_doy_hour(record.decimal_year) else {
            continue;
        };
        let flux = record
            .proton_flux
            .iter()
            .copied()
            .find(|value| value.is_finite())
            .unwrap_or(f64::NAN);
        rows.push(HeliosphereFeatureRow {
            window_name: window_name.to_string(),
            mission: mission.to_string(),
            product: "CRS daily flux".to_string(),
            year,
            doy,
            hour,
            r_au: record.distance_au,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
            density_cm3: f64::NAN,
            speed_kms: f64::NAN,
            temperature_k: f64::NAN,
            bx: f64::NAN,
            by: f64::NAN,
            bz: f64::NAN,
            b_mag: f64::NAN,
            crs_flux: flux,
            spectral_mean: f64::NAN,
            spectral_peak: f64::NAN,
            map_flux_mean: f64::NAN,
            map_flux_std: f64::NAN,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        });
    }
}

fn push_pws_rows(
    rows: &mut Vec<HeliosphereFeatureRow>,
    window_name: &str,
    mission: &str,
    records: &[data_core::catalogs::voyager_pws::VoyagerPwsRecord],
    support: &BTreeMap<(u16, u16, u8), (f64, f64, f64)>,
) {
    for record in records {
        let support_value = support.get(&(record.year, record.doy, record.hour));
        rows.push(HeliosphereFeatureRow {
            window_name: window_name.to_string(),
            mission: mission.to_string(),
            product: "PWS low-rate spectra".to_string(),
            year: record.year,
            doy: record.doy,
            hour: record.hour,
            r_au: support_value.map(|value| value.0).unwrap_or(f64::NAN),
            lat_deg: support_value.map(|value| value.1).unwrap_or(f64::NAN),
            lon_deg: support_value.map(|value| value.2).unwrap_or(f64::NAN),
            density_cm3: f64::NAN,
            speed_kms: f64::NAN,
            temperature_k: f64::NAN,
            bx: f64::NAN,
            by: f64::NAN,
            bz: f64::NAN,
            b_mag: f64::NAN,
            crs_flux: f64::NAN,
            spectral_mean: record.spectral_mean,
            spectral_peak: record.spectral_peak,
            map_flux_mean: f64::NAN,
            map_flux_std: f64::NAN,
            event_score: None,
            event_mask: None,
            event_segment_id: None,
        });
    }
}

fn build_support_index(records: &[OmniRecord]) -> BTreeMap<(u16, u16, u8), (f64, f64, f64)> {
    records
        .iter()
        .map(|record| {
            (
                (record.year, record.doy, record.hour),
                (record.r_au, record.lat_deg, record.lon_deg),
            )
        })
        .collect()
}

fn collect_matching_files<F>(root: &Path, predicate: F) -> Vec<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    let mut paths = Vec::new();
    if !root.exists() {
        return paths;
    }
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();
        if path.is_file() && predicate(path) {
            paths.push(path.to_path_buf());
        }
    }
    paths.sort();
    paths
}

fn file_name_ends_with(path: &Path, suffix: &str) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.ends_with(suffix))
        .unwrap_or(false)
}

fn file_name_starts_with(path: &Path, prefix: &str) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.starts_with(prefix))
        .unwrap_or(false)
}

fn file_date_key(path: &Path) -> Option<(u16, u8, u8)> {
    if let Some(date) = filename_date_yyyymmdd(path) {
        return Some(date);
    }
    let text = path.file_name()?.to_string_lossy();
    let digits = text
        .chars()
        .filter(|value| value.is_ascii_digit())
        .collect::<String>();
    for idx in 0..digits.len().saturating_sub(7) {
        let slice = &digits[idx..idx + 8];
        let year = slice[0..4].parse::<u16>().ok()?;
        let month = slice[4..6].parse::<u8>().ok()?;
        let day = slice[6..8].parse::<u8>().ok()?;
        if NaiveDate::from_ymd_opt(i32::from(year), u32::from(month), u32::from(day)).is_some() {
            return Some((year, month, day));
        }
    }
    None
}

fn rel(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn collect_sorted(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut set = BTreeSet::new();
    for value in values {
        set.insert(value);
    }
    set.into_iter().collect()
}

fn timestamp_utc(year: u16, doy: u16, hour: u8) -> Option<String> {
    let date = NaiveDate::from_yo_opt(year as i32, doy as u32)?;
    Some(format!("{}T{:02}:00:00Z", date.format("%Y-%m-%d"), hour))
}

fn decimal_year_to_doy_hour(decimal_year: f64) -> Option<(u16, u16, u8)> {
    if !decimal_year.is_finite() {
        return None;
    }
    let year = decimal_year.floor() as i32;
    let leap = NaiveDate::from_ymd_opt(year, 2, 29).is_some();
    let days = if leap { 366.0 } else { 365.0 };
    let total_hours = ((decimal_year - year as f64) * days * 24.0).round() as i64;
    let doy = total_hours.div_euclid(24) + 1;
    let hour = total_hours.rem_euclid(24);
    Some((year as u16, doy as u16, hour as u8))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 || !mean.is_finite() {
        return f64::NAN;
    }
    let var = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

/// Scan all `vy1_YYYY.asc` / `vy2_YYYY.asc` files in `data/external/voyager{1,2}/`
/// and ingest any year not already present in `rows` for that mission.
fn ingest_all_voyager_years(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let existing: BTreeSet<(String, u16)> = rows
        .iter()
        .filter(|r| r.mission.starts_with("Voyager"))
        .map(|r| (r.mission.clone(), r.year))
        .collect();

    for (sc, prefix, dir) in [
        (VoyagerSpacecraft::V1, "vy1", "voyager1"),
        (VoyagerSpacecraft::V2, "vy2", "voyager2"),
    ] {
        let mission = match sc {
            VoyagerSpacecraft::V1 => "Voyager 1",
            VoyagerSpacecraft::V2 => "Voyager 2",
        };
        let dir_path = repo_root.join("data/external").join(dir);
        if !dir_path.exists() {
            continue;
        }
        for entry in WalkDir::new(&dir_path).min_depth(1).max_depth(1) {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy();
            if !name.starts_with(prefix) || !name.ends_with(".asc") {
                continue;
            }
            // Extract year from filename: vy1_YYYY.asc
            let year_str = name
                .strip_prefix(&format!("{prefix}_"))
                .and_then(|s| s.strip_suffix(".asc"));
            let Some(year_str) = year_str else { continue };
            let Ok(year) = year_str.parse::<u16>() else { continue };
            if existing.contains(&(mission.to_string(), year)) {
                continue;
            }
            let path = entry.path().to_path_buf();
            match parse_voyager_file(&path, sc) {
                Ok(merged) => {
                    sources.push(rel(repo_root, &path));
                    push_omni_rows(
                        rows,
                        "densified",
                        mission,
                        "Merged hourly",
                        &voyager_to_omni(&merged),
                    );
                }
                Err(e) => {
                    notes.push(format!("WARN: failed to parse {}: {e}", path.display()));
                }
            }
        }
    }
    Ok(())
}

/// Scan all `uly_YYYY.asc` and `uy_coho1hr_*.csv` files in `data/external/ulysses/`
/// and ingest any year not already present in `rows`.
fn ingest_all_ulysses_years(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let existing: BTreeSet<u16> = rows
        .iter()
        .filter(|r| r.mission == "Ulysses")
        .map(|r| r.year)
        .collect();

    let dir_path = repo_root.join("data/external/ulysses");
    if !dir_path.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(&dir_path).min_depth(1).max_depth(1) {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy();
        let year = if let Some(s) = name.strip_prefix("uly_").and_then(|s| s.strip_suffix(".asc")) {
            s.parse::<u16>().ok()
        } else if let Some(s) = name
            .strip_prefix("uy_coho1hr_merged_mag_plasma_")
            .and_then(|s| s.strip_suffix(".csv"))
        {
            s.parse::<u16>().ok()
        } else {
            None
        };
        let Some(year) = year else { continue };
        if existing.contains(&year) {
            continue;
        }
        let path = entry.path().to_path_buf();
        match parse_ulysses_file(&path) {
            Ok(merged) => {
                sources.push(rel(repo_root, &path));
                push_omni_rows(
                    rows,
                    "densified",
                    "Ulysses",
                    "Merged hourly",
                    &ulysses_to_omni(&merged),
                );
            }
            Err(e) => {
                notes.push(format!("WARN: failed to parse {}: {e}", path.display()));
            }
        }
    }
    Ok(())
}

/// Scan all `new_horizons_helio1hr_position_YYYY.csv` in `data/external/new_horizons/`
/// and ingest any year not already present.
fn ingest_all_nh_swap_years(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let existing: BTreeSet<u16> = rows
        .iter()
        .filter(|r| r.mission == "New Horizons")
        .map(|r| r.year)
        .collect();

    let dir_path = repo_root.join("data/external/new_horizons");
    if !dir_path.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(&dir_path).min_depth(1).max_depth(1) {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy();
        let year = name
            .strip_prefix("new_horizons_helio1hr_position_")
            .and_then(|s| s.strip_suffix(".csv"))
            .and_then(|s| s.parse::<u16>().ok());
        let Some(year) = year else { continue };
        if existing.contains(&year) {
            continue;
        }
        let path = entry.path().to_path_buf();
        match parse_nh_swap_file(&path) {
            Ok(records) => {
                sources.push(rel(repo_root, &path));
                push_omni_rows(
                    rows,
                    "densified",
                    "New Horizons",
                    "SWAP+support",
                    &nh_swap_to_omni(&records),
                );
            }
            Err(e) => {
                notes.push(format!("WARN: failed to parse {}: {e}", path.display()));
            }
        }
    }
    Ok(())
}

/// Scan Cassini cruise year files in `data/external/cassini/` and ingest
/// any year not already present. Only cruise phase (1997-2004) -- Saturn
/// orbital data is magnetosphere-contaminated.
fn ingest_all_cassini_cruise_years(
    repo_root: &Path,
    rows: &mut Vec<HeliosphereFeatureRow>,
    sources: &mut Vec<String>,
    notes: &mut Vec<String>,
) -> Result<()> {
    let existing: BTreeSet<u16> = rows
        .iter()
        .filter(|r| r.mission == "Cassini")
        .map(|r| r.year)
        .collect();

    let dir_path = repo_root.join("data/external/cassini");
    if !dir_path.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(&dir_path).min_depth(1).max_depth(1) {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy();
        let year = name
            .strip_prefix("cassini_")
            .and_then(|s| s.strip_suffix("_amda_cruise_hourly.asc"))
            .and_then(|s| s.parse::<u16>().ok());
        let Some(year) = year else { continue };
        if existing.contains(&year) || year > 2004 {
            continue;
        }
        let path = entry.path().to_path_buf();
        match parse_cassini_cruise_file(&path) {
            Ok(records) => {
                sources.push(rel(repo_root, &path));
                push_omni_rows(
                    rows,
                    "densified",
                    "Cassini",
                    "Cruise hybrid",
                    &cassini_to_omni(&records),
                );
            }
            Err(e) => {
                notes.push(format!("WARN: failed to parse {}: {e}", path.display()));
            }
        }
    }
    Ok(())
}
