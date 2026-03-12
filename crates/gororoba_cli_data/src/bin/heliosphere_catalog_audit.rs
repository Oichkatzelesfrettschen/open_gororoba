use anyhow::{Context, Result};
use chrono::TimeZone;
use clap::Parser;
use data_core::{
    catalogs::{
        cassini::{cassini_to_omni, parse_cassini_cruise_file},
        fermi_gbm::parse_fermi_gbm_csv,
        omni::parse_omni_file,
        pioneer::{PioneerSpacecraft, parse_pioneer_file, pioneer_to_omni},
        soho_celias::parse_soho_celias_bundle_file,
        sorce::parse_sorce_csv,
        tsi::parse_tsi_csv,
        voyager::{VoyagerSpacecraft, parse_voyager_file, voyager_to_omni},
    },
    time_bounds::{TimeBounds, bounds_from_omni, bounds_from_soho_celias, format_epoch},
};
use hifitime::Epoch;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-catalog-audit",
    about = "Audit staged heliosphere and adjacent multi-messenger datasets, then compute real local overlap windows."
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(
        long,
        default_value = "data/output/heliosphere/heliosphere_catalog_audit.json"
    )]
    json_out: PathBuf,

    #[arg(
        long,
        default_value = "data/output/heliosphere/heliosphere_catalog_audit.md"
    )]
    markdown_out: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
struct DatasetAuditEntry {
    key: String,
    label: String,
    family: String,
    role: String,
    cadence: String,
    authority: String,
    local_path: String,
    staged: bool,
    catalog_status: String,
    acquisition_status: String,
    contract_status: String,
    satisfies_provider_contract: bool,
    row_count: Option<usize>,
    start_rfc3339: Option<String>,
    end_rfc3339: Option<String>,
    cadence_seconds: Option<f64>,
    notes: String,
}

struct DatasetAuditSpec<'a> {
    key: &'a str,
    label: &'a str,
    family: &'a str,
    role: &'a str,
    cadence: &'a str,
    authority: &'a str,
    local_path: &'a Path,
    row_count: Option<usize>,
    bounds: Option<TimeBounds>,
    notes: String,
}

#[derive(Clone, Copy, Debug)]
enum PioneerEncounterTarget {
    Jupiter,
    Saturn,
}

#[derive(Clone, Copy, Debug)]
enum PioneerCoverage {
    AnnualMerged,
    Encounter(PioneerEncounterTarget),
}

#[derive(Clone, Debug, Serialize)]
struct PackAuditEntry {
    name: String,
    required_keys: Vec<String>,
    optional_keys: Vec<String>,
    gap_tolerant_keys: Vec<String>,
    missing_required: Vec<String>,
    target_start_rfc3339: Option<String>,
    target_end_rfc3339: Option<String>,
    overlap_start_rfc3339: Option<String>,
    overlap_end_rfc3339: Option<String>,
    status: String,
    notes: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    generated_at: String,
    datasets: Vec<DatasetAuditEntry>,
    packs: Vec<PackAuditEntry>,
}

fn bounds_to_strings(bounds: &TimeBounds) -> (String, String) {
    (
        format_epoch(bounds.start_epoch()),
        format_epoch(bounds.end_epoch()),
    )
}

fn union_bounds(bounds: &[TimeBounds]) -> Option<TimeBounds> {
    let start_ms = bounds.iter().map(|b| b.start_ms).min()?;
    let end_ms = bounds.iter().map(|b| b.end_ms).max()?;
    let cadence_seconds = bounds.iter().find_map(|b| b.cadence_seconds);
    Some(TimeBounds {
        start_ms,
        end_ms,
        cadence_seconds,
    })
}

fn bounds_from_unsorted_epochs(epochs: &[Epoch]) -> Option<TimeBounds> {
    let mut ordered = epochs.to_vec();
    ordered.sort_by(|a, b| a.to_et_seconds().total_cmp(&b.to_et_seconds()));
    TimeBounds::from_sorted_epochs(&ordered)
}

fn entry_from_bounds(spec: DatasetAuditSpec<'_>) -> DatasetAuditEntry {
    let (start_rfc3339, end_rfc3339, cadence_seconds) = if let Some(bounds) = spec.bounds {
        let (start, end) = bounds_to_strings(&bounds);
        (Some(start), Some(end), bounds.cadence_seconds)
    } else {
        (None, None, None)
    };
    let staged = spec.local_path.exists();
    let has_science_rows = spec.row_count.unwrap_or(0) > 0;
    let acquisition_status = if has_science_rows {
        "staged"
    } else if staged {
        "partial"
    } else {
        "not_staged"
    };
    let contract_status = if has_science_rows {
        "satisfied"
    } else {
        "blocked"
    };
    DatasetAuditEntry {
        key: spec.key.to_string(),
        label: spec.label.to_string(),
        family: spec.family.to_string(),
        role: spec.role.to_string(),
        cadence: spec.cadence.to_string(),
        authority: spec.authority.to_string(),
        local_path: spec.local_path.display().to_string(),
        staged,
        catalog_status: "known".to_string(),
        acquisition_status: acquisition_status.to_string(),
        contract_status: contract_status.to_string(),
        satisfies_provider_contract: has_science_rows,
        row_count: spec.row_count,
        start_rfc3339,
        end_rfc3339,
        cadence_seconds,
        notes: spec.notes,
    }
}

fn override_contract_status(
    entry: &mut DatasetAuditEntry,
    acquisition_status: &str,
    contract_status: &str,
    satisfies_provider_contract: bool,
) {
    entry.acquisition_status = acquisition_status.to_string();
    entry.contract_status = contract_status.to_string();
    entry.satisfies_provider_contract = satisfies_provider_contract;
}

fn audit_soho_celias(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let path = repo_root.join("data/external/soho/celias/CELIAS_Proton_Monitor_5min.tar.gz");
    if !path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key: "soho_celias_bundle",
            label: "SOHO CELIAS Proton Monitor bundle",
            family: "heliosphere",
            role: "inner_boundary_primary",
            cadence: "native_5min_bundle",
            authority: "GSFC mission bundle / governed local",
            local_path: &path,
            row_count: None,
            bounds: None,
            notes: "Bundle not staged locally.".to_string(),
        }));
    }
    let records = parse_soho_celias_bundle_file(&path)?;
    let bounds = bounds_from_soho_celias(&records);
    Ok(entry_from_bounds(DatasetAuditSpec {
        key: "soho_celias_bundle",
        label: "SOHO CELIAS Proton Monitor bundle",
        family: "heliosphere",
        role: "inner_boundary_primary",
        cadence: "native_5min_bundle",
        authority: "GSFC mission bundle / governed local",
        local_path: &path,
        row_count: Some(records.len()),
        bounds,
        notes: "Native 5-minute inner-boundary lane; hourly normalization is derived downstream."
            .to_string(),
    }))
}

fn audit_tsis(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let path = repo_root.join("data/external/tsis1_tsi_daily.csv");
    if !path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key: "tsis1_daily",
            label: "TSIS-1 TSI Daily",
            family: "solar_context",
            role: "optional_secondary",
            cadence: "daily",
            authority: "LASP LISIRD",
            local_path: &path,
            row_count: None,
            bounds: None,
            notes: "TSIS file not staged locally.".to_string(),
        }));
    }
    let rows = parse_tsi_csv(&path)?;
    let bounds = TimeBounds::from_sorted_epochs(
        &rows
            .iter()
            .map(|row| Epoch::from_jde_utc(row.jd))
            .collect::<Vec<_>>(),
    );
    Ok(entry_from_bounds(DatasetAuditSpec {
        key: "tsis1_daily",
        label: "TSIS-1 TSI Daily",
        family: "solar_context",
        role: "optional_secondary",
        cadence: "daily",
        authority: "LASP LISIRD",
        local_path: &path,
        row_count: Some(rows.len()),
        bounds,
        notes: "Daily thermodynamic context layer for 2018+ packs.".to_string(),
    }))
}

fn audit_sorce(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let path = repo_root.join("data/external/sorce_tsi_daily.csv");
    if !path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key: "sorce_daily",
            label: "SORCE TSI Daily",
            family: "solar_context",
            role: "optional_secondary",
            cadence: "daily",
            authority: "LASP LISIRD",
            local_path: &path,
            row_count: None,
            bounds: None,
            notes: "SORCE file not staged locally.".to_string(),
        }));
    }
    let rows = parse_sorce_csv(&path)?;
    let bounds = TimeBounds::from_sorted_epochs(
        &rows
            .iter()
            .map(|row| Epoch::from_jde_utc(row.jd))
            .collect::<Vec<_>>(),
    );
    Ok(entry_from_bounds(DatasetAuditSpec {
        key: "sorce_daily",
        label: "SORCE TSI Daily",
        family: "solar_context",
        role: "optional_secondary",
        cadence: "daily",
        authority: "LASP LISIRD",
        local_path: &path,
        row_count: Some(rows.len()),
        bounds,
        notes:
            "Legacy radiative context layer overlapping the late Cassini cruise and heliopause eras."
                .to_string(),
    }))
}

fn audit_voyager_merged_file(
    key: &str,
    label: &str,
    path: &Path,
    spacecraft: VoyagerSpacecraft,
    role: &str,
) -> Result<DatasetAuditEntry> {
    if !path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key,
            label,
            family: "heliosphere",
            role,
            cadence: "hourly",
            authority: "AMDA governed fallback",
            local_path: path,
            row_count: None,
            bounds: None,
            notes: "Merged file not staged locally.".to_string(),
        }));
    }
    let raw = parse_voyager_file(path, spacecraft)?;
    let omni = voyager_to_omni(&raw);
    let bounds = bounds_from_omni(&omni);
    Ok(entry_from_bounds(DatasetAuditSpec {
        key,
        label,
        family: "heliosphere",
        role,
        cadence: "hourly",
        authority: "AMDA governed fallback",
        local_path: path,
        row_count: Some(omni.len()),
        bounds,
        notes: "Merged plasma/magnetic/trajectory lane.".to_string(),
    }))
}

fn audit_voyager_2017_2018(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let root = repo_root.join("data/external/voyager/voyager2");
    let paths = [
        root.join("vy2_2017_amda_merged_hourly.asc"),
        root.join("vy2_2018_amda_merged_hourly.asc"),
    ];
    let mut row_count = 0usize;
    let mut bounds = Vec::new();
    for path in &paths {
        if !path.exists() {
            continue;
        }
        let raw = parse_voyager_file(path, VoyagerSpacecraft::V2)?;
        let omni = voyager_to_omni(&raw);
        row_count += omni.len();
        if let Some(bound) = bounds_from_omni(&omni) {
            bounds.push(bound);
        }
    }
    Ok(entry_from_bounds(DatasetAuditSpec {
        key: "voyager2_2017_2018_merged",
        label: "Voyager 2 merged 2017-2018",
        family: "heliosphere",
        role: "outer_boundary_primary",
        cadence: "hourly",
        authority: "AMDA governed fallback",
        local_path: &root,
        row_count: if row_count > 0 { Some(row_count) } else { None },
        bounds: union_bounds(&bounds),
        notes: "Deep heliosheath / heliopause-era merged plasma and trajectory lane.".to_string(),
    }))
}

fn audit_omni_2017_2018(repo_root: &Path) -> Result<DatasetAuditEntry> {
    audit_omni_range(
        repo_root,
        "omni_2017_2018",
        "OMNI2 2017-2018",
        &[2017, 2018],
        "Bow-shock-propagated L1 hourly context for aligned heliopause packs, from canonical SPDF OMNI2 or governed AMDA HAPI fallback.",
    )
}

fn audit_omni_range(
    repo_root: &Path,
    key: &str,
    label: &str,
    years: &[i32],
    notes: &str,
) -> Result<DatasetAuditEntry> {
    let root = repo_root.join("data/external/omni2");
    let mut paths = Vec::new();
    let mut has_spdf = false;
    let mut has_amda = false;
    for year in years {
        let spdf_path = root.join(format!("omni2_{year}.dat"));
        let amda_path = root.join(format!("omni2_{year}_amda_hourly.csv"));
        if spdf_path.exists() {
            has_spdf = true;
        }
        if amda_path.exists() {
            has_amda = true;
        }
        paths.push(spdf_path);
        paths.push(amda_path);
    }
    let mut row_count = 0usize;
    let mut bounds = Vec::new();
    for path in &paths {
        if !path.exists() {
            continue;
        }
        let rows = parse_omni_file(path)?;
        row_count += rows.len();
        if let Some(bound) = bounds_from_omni(&rows) {
            bounds.push(bound);
        }
    }
    let staged = row_count > 0 && !bounds.is_empty();
    let authority = match (has_spdf, has_amda) {
        (true, true) => "SPDF OMNI + AMDA fallback",
        (true, false) => "SPDF OMNI",
        (false, true) => "AMDA OMNI fallback",
        (false, false) => "SPDF OMNI",
    };
    let source_summary = match (has_spdf, has_amda) {
        (true, true) => "Local source lineage: canonical SPDF OMNI2 plus governed AMDA fallback.",
        (true, false) => "Local source lineage: canonical SPDF OMNI2 only.",
        (false, true) => "Local source lineage: governed AMDA fallback only.",
        (false, false) => "Local source lineage: no staged OMNI files found.",
    };
    let mut entry = entry_from_bounds(DatasetAuditSpec {
        key,
        label,
        family: "heliosphere",
        role: "inner_boundary_primary",
        cadence: "hourly",
        authority,
        local_path: &root,
        row_count: if staged { Some(row_count) } else { None },
        bounds: union_bounds(&bounds),
        notes: format!("{notes} {source_summary}"),
    });
    entry.staged = staged;
    Ok(entry)
}

fn audit_gwosc(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let path = repo_root.join("data/external/gwosc_all_events.csv");
    if !path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key: "gwosc_all_events",
            label: "GWOSC combined GWTC",
            family: "multi_messenger",
            role: "optional_secondary",
            cadence: "event_catalog",
            authority: "GWOSC",
            local_path: &path,
            row_count: None,
            bounds: None,
            notes: "Combined GW catalog not staged locally.".to_string(),
        }));
    }
    let mut reader = csv::Reader::from_path(&path)?;
    let headers = reader.headers()?.clone();
    let idx_gps = headers
        .iter()
        .position(|header| header.trim().eq_ignore_ascii_case("gps"))
        .context("gwosc_all_events.csv missing gps column")?;
    let mut epochs = Vec::new();
    let mut rows = 0usize;
    for record in reader.records() {
        let record = record?;
        let gps_seconds = record
            .get(idx_gps)
            .and_then(|text| text.trim().parse::<f64>().ok());
        if let Some(gps_seconds) = gps_seconds {
            epochs.push(Epoch::from_gpst_seconds(gps_seconds));
            rows += 1;
        }
    }
    Ok(entry_from_bounds(DatasetAuditSpec {
        key: "gwosc_all_events",
        label: "GWOSC combined GWTC",
        family: "multi_messenger",
        role: "optional_secondary",
        cadence: "event_catalog",
        authority: "GWOSC",
        local_path: &path,
        row_count: Some(rows),
        bounds: bounds_from_unsorted_epochs(&epochs),
        notes: "Use this broader catalog for 2017 overlap, not GWTC-3 confident only.".to_string(),
    }))
}

fn parse_fermi_trigger_epoch(text: &str) -> Option<Epoch> {
    let dt = chrono::NaiveDateTime::parse_from_str(text.trim(), "%Y-%m-%d %H:%M:%S%.3f").ok()?;
    Some(Epoch::from_unix_milliseconds(
        chrono::Utc.from_utc_datetime(&dt).timestamp_millis() as f64,
    ))
}

fn audit_fermi(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let path = repo_root.join("data/external/fermi_gbm_grbs.csv");
    if !path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key: "fermi_gbm",
            label: "Fermi GBM Burst Catalog",
            family: "multi_messenger",
            role: "optional_secondary",
            cadence: "event_catalog",
            authority: "HEASARC",
            local_path: &path,
            row_count: None,
            bounds: None,
            notes: "Fermi GBM catalog not staged locally.".to_string(),
        }));
    }
    let rows = parse_fermi_gbm_csv(&path)?;
    let epochs: Vec<Epoch> = rows
        .iter()
        .filter_map(|row| parse_fermi_trigger_epoch(&row.trigger_time))
        .collect();
    Ok(entry_from_bounds(DatasetAuditSpec {
        key: "fermi_gbm",
        label: "Fermi GBM Burst Catalog",
        family: "multi_messenger",
        role: "optional_secondary",
        cadence: "event_catalog",
        authority: "HEASARC",
        local_path: &path,
        row_count: Some(rows.len()),
        bounds: bounds_from_unsorted_epochs(&epochs),
        notes: "Gamma-ray transient context layer for 2012+ windows.".to_string(),
    }))
}

fn audit_wow(repo_root: &Path) -> DatasetAuditEntry {
    let path = repo_root.join("data/external/wow_1977_printout.jpg");
    entry_from_bounds(DatasetAuditSpec {
        key: "wow_1977",
        label: "Wow! Signal printout",
        family: "historical_context",
        role: "context_only",
        cadence: "single_event_artifact",
        authority: "Ohio History Connection",
        local_path: &path,
        row_count: None,
        bounds: None,
        notes:
            "Historical epoch/context anchor only; not a plasma boundary or chronology provider."
                .to_string(),
    })
}

fn audit_placeholder(key: &str, label: &str, root: &Path, notes: &str) -> DatasetAuditEntry {
    entry_from_bounds(DatasetAuditSpec {
        key,
        label,
        family: "adjacent_heliosphere",
        role: "optional_secondary",
        cadence: "unknown_or_not_staged",
        authority: "cataloged_not_staged",
        local_path: root,
        row_count: None,
        bounds: None,
        notes: notes.to_string(),
    })
}

fn audit_pioneer_annual(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let _coverage = PioneerCoverage::AnnualMerged;
    let mut paths: Vec<PathBuf> = Vec::new();
    for pattern in [
        "data/external/pioneer/pioneer10/pds_ppi_merged/p10_*.TAB",
        "data/external/pioneer/pioneer11/pds_ppi_merged/p11_*.TAB",
    ] {
        let matches = glob::glob(&repo_root.join(pattern).display().to_string())
            .with_context(|| format!("expanding glob {}", pattern))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("collecting glob matches for {}", pattern))?;
        paths.extend(matches);
    }
    if paths.is_empty() {
        let mut entry = audit_placeholder(
            "pioneer_annual_merged",
            "Pioneer annual merged hourly",
            &repo_root.join("data/external/pioneer"),
            "Canonical annual Pioneer merged hourly lane is known, but no annual merged science bytes are staged locally.",
        );
        entry.staged = false;
        entry.catalog_status = "known".to_string();
        entry.acquisition_status = "partial".to_string();
        entry.contract_status = "blocked_host_unreachable".to_string();
        entry.satisfies_provider_contract = false;
        return Ok(entry);
    }
    paths.sort();

    let mut bounds = Vec::new();
    let mut row_count = 0usize;
    for path in &paths {
        let spacecraft = if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("p10_"))
        {
            PioneerSpacecraft::P10
        } else {
            PioneerSpacecraft::P11
        };
        let raw = parse_pioneer_file(path, spacecraft)?;
        let omni = pioneer_to_omni(&raw);
        row_count += omni.len();
        if let Some(dataset_bounds) = bounds_from_omni(&omni) {
            bounds.push(dataset_bounds);
        }
    }
    let mut entry = entry_from_bounds(DatasetAuditSpec {
        key: "pioneer_annual_merged",
        label: "Pioneer annual merged hourly",
        family: "adjacent_heliosphere",
        role: "optional_secondary",
        cadence: "hourly",
        authority: "PDS/PPI governed local",
        local_path: &repo_root.join("data/external/pioneer"),
        row_count: Some(row_count),
        bounds: union_bounds(&bounds),
        notes: "Reachable UCLA PDS/PPI annual merged Pioneer lane is now partially staged locally. This improves scientific coverage beyond metadata-only state, but the local staging is still a subset of the full annual source family and does not yet satisfy the full annual provider contract.".to_string(),
    });
    entry.acquisition_status = "partial".to_string();
    entry.contract_status = "ready_governed_partial_annual_lane".to_string();
    entry.satisfies_provider_contract = false;
    Ok(entry)
}

fn audit_pioneer_encounter(
    key: &str,
    label: &str,
    spacecraft: PioneerSpacecraft,
    target: PioneerEncounterTarget,
    data_path: &Path,
) -> Result<DatasetAuditEntry> {
    let coverage = PioneerCoverage::Encounter(target);
    if !data_path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key,
            label,
            family: "adjacent_heliosphere",
            role: "optional_secondary",
            cadence: "hourly_encounter",
            authority: "PDS/PPI governed local",
            local_path: data_path,
            row_count: None,
            bounds: None,
            notes: "Pioneer encounter subset not staged locally.".to_string(),
        }));
    }
    let raw = parse_pioneer_file(data_path, spacecraft)?;
    let omni = pioneer_to_omni(&raw);
    let notes = match coverage {
        PioneerCoverage::Encounter(PioneerEncounterTarget::Jupiter) => {
            "Reachable UCLA PDS/PPI Jupiter encounter subset preserving the original NSSDC merged Pioneer record format. Use as an encounter-window adjacent lane, not as a full annual sibling replacement.".to_string()
        }
        PioneerCoverage::Encounter(PioneerEncounterTarget::Saturn) => {
            "Reachable UCLA PDS/PPI Saturn encounter subset preserving the original NSSDC merged Pioneer record format. This is a disjoint 1979 context lane, not a same-window Jupiter replacement.".to_string()
        }
        PioneerCoverage::AnnualMerged => unreachable!(),
    };
    let mut entry = entry_from_bounds(DatasetAuditSpec {
        key,
        label,
        family: "adjacent_heliosphere",
        role: "optional_secondary",
        cadence: "hourly_encounter",
        authority: "PDS/PPI governed local",
        local_path: data_path,
        row_count: Some(omni.len()),
        bounds: bounds_from_omni(&omni),
        notes,
    });
    entry.catalog_status = "known".to_string();
    entry.acquisition_status = "staged".to_string();
    entry.contract_status = "ready_governed_adjacent_lane".to_string();
    entry.satisfies_provider_contract = false;
    Ok(entry)
}

fn audit_encounter_track(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let path = repo_root.join("data/output/heliosphere/limits/voyager_encounter_track.csv");
    if !path.exists() {
        return Ok(entry_from_bounds(DatasetAuditSpec {
            key: "voyager2_jupiter_track",
            label: "Voyager 2 Jupiter encounter fused track",
            family: "heliosphere_operational",
            role: "operational_validation",
            cadence: "hourly_track",
            authority: "governed local output",
            local_path: &path,
            row_count: None,
            bounds: None,
            notes: "Operational fused artifact not present.".to_string(),
        }));
    }
    let mut reader = csv::Reader::from_path(&path)?;
    let headers = reader.headers()?.clone();
    let idx_ts = headers
        .iter()
        .position(|header| header.trim().eq_ignore_ascii_case("timestamp"))
        .context("voyager_encounter_track.csv missing timestamp column")?;
    let mut epochs = Vec::new();
    let mut rows = 0usize;
    for record in reader.records() {
        let record = record?;
        if let Some(text) = record.get(idx_ts)
            && let Ok(epoch) = Epoch::from_str(text.trim())
        {
            epochs.push(epoch);
            rows += 1;
        }
    }
    Ok(entry_from_bounds(DatasetAuditSpec {
        key: "voyager2_jupiter_track",
        label: "Voyager 2 Jupiter encounter fused track",
        family: "heliosphere_operational",
        role: "operational_validation",
        cadence: "hourly_track",
        authority: "governed local output",
        local_path: &path,
        row_count: Some(rows),
        bounds: TimeBounds::from_sorted_epochs(&epochs),
        notes: "Operational fused telemetry-plus-position validation artifact.".to_string(),
    }))
}

fn audit_cassini_cruise(repo_root: &Path) -> Result<DatasetAuditEntry> {
    let root = repo_root.join("data/external/cassini");
    let paths = [
        root.join("cassini_1998_amda_cruise_hourly.asc"),
        root.join("cassini_1999_amda_cruise_hourly.asc"),
        root.join("cassini_2000_amda_cruise_hourly.asc"),
        root.join("cassini_2001_amda_cruise_hourly.asc"),
        root.join("cassini_2002_amda_cruise_hourly.asc"),
        root.join("cassini_2003_amda_cruise_hourly.asc"),
        root.join("cassini_2004_amda_cruise_hourly.asc"),
    ];
    let mut row_count = 0usize;
    let mut bounds = Vec::new();
    for path in &paths {
        if !path.exists() {
            continue;
        }
        let raw = parse_cassini_cruise_file(path)?;
        let omni = cassini_to_omni(&raw);
        row_count += omni.len();
        if let Some(bound) = bounds_from_omni(&omni) {
            bounds.push(bound);
        }
    }
    let mut entry = entry_from_bounds(DatasetAuditSpec {
        key: "cassini_cruise_1998_2004",
        label: "Cassini cruise hybrid 1998-2004",
        family: "adjacent_heliosphere",
        role: "outer_boundary_primary",
        cadence: "hourly",
        authority: "AMDA hybrid derived",
        local_path: &root,
        row_count: if row_count > 0 { Some(row_count) } else { None },
        bounds: union_bounds(&bounds),
        notes: "Governed Cassini cruise hourly lane derived from AMDA `cass-orb-cruise` (measured trajectory), `cass-mag-rtn60` (measured magnetic field), and `tao-cass-sw` (modeled solar-wind plasma). Full overlap begins in late 1998, so this supports a fully aligned late-cruise pack from 1999 onward rather than the full 1997 mission launch interval.".to_string(),
    });
    if entry.staged {
        override_contract_status(&mut entry, "staged", "ready_governed_hybrid_lane", false);
    }
    Ok(entry)
}

fn build_pack(
    name: &str,
    required_keys: &[&str],
    optional_keys: &[&str],
    gap_tolerant_keys: &[&str],
    datasets: &BTreeMap<String, DatasetAuditEntry>,
    target_start: Option<&str>,
    target_end: Option<&str>,
) -> PackAuditEntry {
    let missing_required: Vec<String> = required_keys
        .iter()
        .filter(|key| !datasets.get(**key).map(|d| d.staged).unwrap_or(false))
        .map(|key| (*key).to_string())
        .collect();

    let required_bounds: Vec<TimeBounds> = required_keys
        .iter()
        .filter_map(|key| datasets.get(*key))
        .filter(|dataset| dataset.staged)
        .filter_map(
            |dataset| match (&dataset.start_rfc3339, &dataset.end_rfc3339) {
                (Some(start), Some(end)) => {
                    let start_epoch = Epoch::from_str(start).ok()?;
                    let end_epoch = Epoch::from_str(end).ok()?;
                    Some(TimeBounds::from_sorted_epochs(&[start_epoch, end_epoch])?)
                }
                _ => None,
            },
        )
        .collect();

    let strict_required_bounds: Vec<TimeBounds> = required_keys
        .iter()
        .filter(|key| !gap_tolerant_keys.contains(key))
        .filter_map(|key| datasets.get(*key))
        .filter(|dataset| dataset.staged)
        .filter_map(
            |dataset| match (&dataset.start_rfc3339, &dataset.end_rfc3339) {
                (Some(start), Some(end)) => {
                    let start_epoch = Epoch::from_str(start).ok()?;
                    let end_epoch = Epoch::from_str(end).ok()?;
                    Some(TimeBounds::from_sorted_epochs(&[start_epoch, end_epoch])?)
                }
                _ => None,
            },
        )
        .collect();

    let overlap = if missing_required.is_empty() {
        TimeBounds::intersect_all(&required_bounds)
    } else {
        None
    };
    let (overlap_start_rfc3339, overlap_end_rfc3339) = if let Some(ref bounds) = overlap {
        let (start, end) = bounds_to_strings(bounds);
        (Some(start), Some(end))
    } else {
        (None, None)
    };

    let target_bounds = match (target_start, target_end) {
        (Some(start), Some(end)) => match (Epoch::from_str(start), Epoch::from_str(end)) {
            (Ok(start_epoch), Ok(end_epoch)) => {
                Some((start.to_string(), end.to_string(), start_epoch, end_epoch))
            }
            _ => None,
        },
        _ => None,
    };

    let strict_overlap = if missing_required.is_empty() {
        TimeBounds::intersect_all(&strict_required_bounds)
    } else {
        None
    };

    let gap_bounds: Vec<String> = gap_tolerant_keys
        .iter()
        .filter_map(|key| {
            datasets
                .get(*key)
                .map(|dataset| ((*key).to_string(), dataset))
        })
        .filter_map(
            |(key, dataset)| match (&dataset.start_rfc3339, &dataset.end_rfc3339) {
                (Some(start), Some(end)) => Some(format!("{key}=[{start}, {end}]")),
                _ => None,
            },
        )
        .collect();

    let (status, notes) = if !missing_required.is_empty() {
        (
            "blocked_missing_required".to_string(),
            format!(
                "Missing required staged datasets: {}",
                missing_required.join(", ")
            ),
        )
    } else if overlap_start_rfc3339.is_none() {
        (
            "blocked_no_overlap".to_string(),
            "Required datasets are staged but do not share a valid overlap window.".to_string(),
        )
    } else if let (Some(bounds), Some((_, _, target_start_epoch, target_end_epoch))) =
        (overlap.clone(), target_bounds.as_ref())
    {
        if !bounds.contains_epoch_window(*target_start_epoch, *target_end_epoch) {
            if !gap_tolerant_keys.is_empty()
                && strict_overlap.as_ref().is_some_and(|strict| {
                    strict.contains_epoch_window(*target_start_epoch, *target_end_epoch)
                })
            {
                (
                    "ready_gap_tolerant".to_string(),
                    format!(
                        "Required strict datasets cover the full requested pack window [{}, {}], but gap-tolerant datasets only have native coverage over {}. Outside those native intervals they must propagate missing/None values rather than fabricated samples.",
                        format_epoch(*target_start_epoch),
                        format_epoch(*target_end_epoch),
                        gap_bounds.join(", ")
                    ),
                )
            } else {
                (
                    "blocked_partial_window".to_string(),
                    format!(
                        "Required datasets overlap, but not for the full requested pack window [{}, {}].",
                        format_epoch(*target_start_epoch),
                        format_epoch(*target_end_epoch)
                    ),
                )
            }
        } else {
            (
                "ready".to_string(),
                "Required datasets are staged and share a valid overlap window.".to_string(),
            )
        }
    } else {
        (
            "ready".to_string(),
            "Required datasets are staged and share a valid overlap window.".to_string(),
        )
    };

    PackAuditEntry {
        name: name.to_string(),
        required_keys: required_keys.iter().map(|key| (*key).to_string()).collect(),
        optional_keys: optional_keys.iter().map(|key| (*key).to_string()).collect(),
        gap_tolerant_keys: gap_tolerant_keys
            .iter()
            .map(|key| (*key).to_string())
            .collect(),
        missing_required,
        target_start_rfc3339: target_bounds.as_ref().map(|(start, _, _, _)| start.clone()),
        target_end_rfc3339: target_bounds.as_ref().map(|(_, end, _, _)| end.clone()),
        overlap_start_rfc3339,
        overlap_end_rfc3339,
        status,
        notes,
    }
}

fn write_markdown(path: &Path, report: &AuditReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    out.push_str("# Heliosphere Catalog Audit\n\n");
    out.push_str(&format!("Generated at `{}`.\n\n", report.generated_at));
    out.push_str("## Datasets\n\n");
    out.push_str("| Key | Role | Staged | Catalog | Acquisition | Contract | Satisfies Contract | Cadence | Start | End | Notes |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for dataset in &report.datasets {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            dataset.key,
            dataset.role,
            if dataset.staged { "yes" } else { "no" },
            dataset.catalog_status,
            dataset.acquisition_status,
            dataset.contract_status,
            if dataset.satisfies_provider_contract {
                "yes"
            } else {
                "no"
            },
            dataset.cadence,
            dataset.start_rfc3339.as_deref().unwrap_or("-"),
            dataset.end_rfc3339.as_deref().unwrap_or("-"),
            dataset.notes.replace('|', "/"),
        ));
    }
    out.push_str("\n## Packs\n\n");
    out.push_str("| Pack | Status | Gap-Tolerant Keys | Target Start | Target End | Overlap Start | Overlap End | Missing Required |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for pack in &report.packs {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            pack.name,
            pack.status,
            if pack.gap_tolerant_keys.is_empty() {
                "-".to_string()
            } else {
                pack.gap_tolerant_keys.join(", ")
            },
            pack.target_start_rfc3339.as_deref().unwrap_or("-"),
            pack.target_end_rfc3339.as_deref().unwrap_or("-"),
            pack.overlap_start_rfc3339.as_deref().unwrap_or("-"),
            pack.overlap_end_rfc3339.as_deref().unwrap_or("-"),
            if pack.missing_required.is_empty() {
                "-".to_string()
            } else {
                pack.missing_required.join(", ")
            }
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = cli.repo_root;

    let datasets = vec![
        audit_soho_celias(&repo_root)?,
        audit_sorce(&repo_root)?,
        audit_tsis(&repo_root)?,
        audit_omni_range(
            &repo_root,
            "omni_1997_2004",
            "OMNI2 1997-2004",
            &[1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004],
            "Bow-shock-propagated L1 hourly context for the full Cassini launch-to-insertion era, from canonical SPDF OMNI2 or governed AMDA HAPI fallback.",
        )?,
        audit_omni_range(
            &repo_root,
            "omni_1999_2004",
            "OMNI2 1999-2004",
            &[1999, 2000, 2001, 2002, 2003, 2004],
            "Bow-shock-propagated L1 hourly context for the late Cassini cruise era, from canonical SPDF OMNI2 or governed AMDA HAPI fallback.",
        )?,
        audit_omni_range(
            &repo_root,
            "omni_2005_2016",
            "OMNI2 2005-2016",
            &[2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016],
            "Bow-shock-propagated L1 hourly context for the mid-mission continuous inner-boundary span, from governed AMDA HAPI fallback on this host.",
        )?,
        audit_omni_2017_2018(&repo_root)?,
        audit_omni_range(
            &repo_root,
            "omni_2019_2025",
            "OMNI2 2019-2025",
            &[2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "Bow-shock-propagated L1 hourly context for the post-heliopause modern era, spanning the governed AMDA 2019 fallback and canonical SPDF 2020-2025 yearly ASCII.",
        )?,
        audit_omni_range(
            &repo_root,
            "omni_1997_2025",
            "OMNI2 1997-2025",
            &[
                1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007,
                2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
                2019, 2020, 2021, 2022, 2023, 2024, 2025,
            ],
            "Continuous governed OMNI hourly inner-boundary lane across the full locally staged 1997-2025 span, with AMDA fallback for 1997-2019 and canonical SPDF yearly ASCII for 2020-2025.",
        )?,
        audit_voyager_merged_file(
            "voyager1_1979_merged",
            "Voyager 1 merged 1979",
            &repo_root.join("data/external/voyager/voyager1/vy1_1979_amda_merged_hourly.asc"),
            VoyagerSpacecraft::V1,
            "optional_secondary",
        )?,
        audit_voyager_merged_file(
            "voyager2_1979_merged",
            "Voyager 2 merged 1979",
            &repo_root.join("data/external/voyager/voyager2/vy2_1979_amda_merged_hourly.asc"),
            VoyagerSpacecraft::V2,
            "outer_boundary_primary",
        )?,
        audit_voyager_2017_2018(&repo_root)?,
        audit_encounter_track(&repo_root)?,
        audit_gwosc(&repo_root)?,
        audit_fermi(&repo_root)?,
        audit_wow(&repo_root),
        audit_pioneer_annual(&repo_root)?,
        audit_pioneer_encounter(
            "pioneer10_jupiter_1973_encounter",
            "Pioneer 10 Jupiter encounter 1973",
            PioneerSpacecraft::P10,
            PioneerEncounterTarget::Jupiter,
            &repo_root.join(
                "data/external/pioneer/pioneer10/jupiter_encounter_ppi/DATA/P10_JUP_HVM_PA_1HR_MERGED.TAB",
            ),
        )?,
        audit_pioneer_encounter(
            "pioneer11_jupiter_1974_encounter",
            "Pioneer 11 Jupiter encounter 1974",
            PioneerSpacecraft::P11,
            PioneerEncounterTarget::Jupiter,
            &repo_root.join(
                "data/external/pioneer/pioneer11/jupiter_encounter_ppi/DATA/P11_JUP_HVM_PLS_1HR_MERGED.TAB",
            ),
        )?,
        audit_pioneer_encounter(
            "pioneer11_saturn_1979_encounter",
            "Pioneer 11 Saturn encounter 1979",
            PioneerSpacecraft::P11,
            PioneerEncounterTarget::Saturn,
            &repo_root.join(
                "data/external/pioneer/pioneer11/saturn_encounter_ppi/DATA/P11_SAT_HVM_PA_1HR_MERGED.TAB",
            ),
        )?,
        audit_cassini_cruise(&repo_root)?,
    ];

    let dataset_map: BTreeMap<String, DatasetAuditEntry> = datasets
        .iter()
        .cloned()
        .map(|dataset| (dataset.key.clone(), dataset))
        .collect();

    let packs = vec![
        build_pack(
            "JUPITER_1979",
            &["voyager2_1979_merged", "voyager2_jupiter_track"],
            &[
                "voyager1_1979_merged",
                "pioneer_annual_merged",
                "pioneer11_saturn_1979_encounter",
                "wow_1977",
            ],
            &[],
            &dataset_map,
            None,
            None,
        ),
        build_pack(
            "PIONEER_10_JUPITER_1973",
            &["pioneer10_jupiter_1973_encounter"],
            &["wow_1977"],
            &[],
            &dataset_map,
            Some("1973-11-26T00:00:00Z"),
            Some("1973-12-31T23:00:00Z"),
        ),
        build_pack(
            "PIONEER_11_JUPITER_1974",
            &["pioneer11_jupiter_1974_encounter"],
            &["wow_1977"],
            &[],
            &dataset_map,
            Some("1974-11-03T00:00:00Z"),
            Some("1974-12-31T23:00:00Z"),
        ),
        build_pack(
            "PIONEER_11_SATURN_1979",
            &["pioneer11_saturn_1979_encounter"],
            &["voyager2_1979_merged"],
            &[],
            &dataset_map,
            Some("1979-07-31T00:00:00Z"),
            Some("1979-10-04T00:00:00Z"),
        ),
        build_pack(
            "HELIOPAUSE_2017_2018",
            &[
                "soho_celias_bundle",
                "omni_2017_2018",
                "voyager2_2017_2018_merged",
            ],
            &[
                "tsis1_daily",
                "sorce_daily",
                "gwosc_all_events",
                "fermi_gbm",
            ],
            &[],
            &dataset_map,
            Some("2017-01-01T00:00:00Z"),
            Some("2018-11-30T23:00:00Z"),
        ),
        build_pack(
            "CRUISE_1997_2004",
            &[
                "soho_celias_bundle",
                "omni_1997_2004",
                "cassini_cruise_1998_2004",
            ],
            &["sorce_daily"],
            &["cassini_cruise_1998_2004"],
            &dataset_map,
            Some("1997-11-15T00:00:00Z"),
            Some("2004-07-04T00:00:00Z"),
        ),
        build_pack(
            "CRUISE_1999_2004",
            &[
                "soho_celias_bundle",
                "omni_1999_2004",
                "cassini_cruise_1998_2004",
            ],
            &["sorce_daily"],
            &[],
            &dataset_map,
            Some("1999-01-01T00:00:00Z"),
            Some("2004-07-03T23:00:00Z"),
        ),
        build_pack(
            "INNER_BOUNDARY_1997_2023",
            &["soho_celias_bundle", "omni_1997_2025"],
            &["sorce_daily", "tsis1_daily"],
            &[],
            &dataset_map,
            Some("1997-01-01T00:00:00Z"),
            Some("2023-07-06T23:57:16Z"),
        ),
        build_pack(
            "RADIATIVE_2003_2020",
            &["soho_celias_bundle", "omni_1997_2025", "sorce_daily"],
            &["fermi_gbm"],
            &[],
            &dataset_map,
            Some("2003-02-25T12:00:00Z"),
            Some("2020-02-25T12:00:00Z"),
        ),
        build_pack(
            "TSI_CROSSCAL_2018_2020",
            &[
                "soho_celias_bundle",
                "omni_1997_2025",
                "sorce_daily",
                "tsis1_daily",
            ],
            &["fermi_gbm", "gwosc_all_events"],
            &[],
            &dataset_map,
            Some("2018-01-11T12:00:00Z"),
            Some("2020-02-25T12:00:00Z"),
        ),
        build_pack(
            "SOLAR_CYCLE24_2008_2019",
            &["soho_celias_bundle", "omni_1997_2025"],
            &["sorce_daily", "tsis1_daily", "fermi_gbm"],
            &[],
            &dataset_map,
            Some("2008-01-01T00:00:00Z"),
            Some("2019-12-31T23:00:00Z"),
        ),
        build_pack(
            "POST_HELIOPAUSE_2019_2023",
            &["soho_celias_bundle", "omni_2019_2025", "tsis1_daily"],
            &["sorce_daily", "fermi_gbm"],
            &[],
            &dataset_map,
            Some("2019-01-01T00:00:00Z"),
            Some("2023-07-06T23:57:16Z"),
        ),
    ];

    let report = AuditReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        datasets,
        packs,
    };

    if let Some(parent) = cli.json_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.json_out, serde_json::to_vec_pretty(&report)?)?;
    write_markdown(&cli.markdown_out, &report)?;

    println!("wrote {}", cli.json_out.display());
    println!("wrote {}", cli.markdown_out.display());
    Ok(())
}
