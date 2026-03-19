use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use csv::Writer;
use gororoba_cli_data::nanograv_timing_model::{
    BinaryFamily, ReleaseBand, TimingModel, load_release_timing_models,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "nanograv-timing-refit-preflight",
    about = "Typed .par preflight inventory for the NANOGrav 15-year timing-refit lane"
)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, value_enum, default_value_t = BandArg::Wideband)]
    band: BandArg,

    #[arg(long, default_value = "data/csv/nanograv_15yr_timing_refit_preflight.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value = "reports/nanograv_15yr_timing_refit_preflight.toml")]
    report_out: PathBuf,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum BandArg {
    Narrowband,
    Wideband,
}

impl From<BandArg> for ReleaseBand {
    fn from(value: BandArg) -> Self {
        match value {
            BandArg::Narrowband => ReleaseBand::Narrowband,
            BandArg::Wideband => ReleaseBand::Wideband,
        }
    }
}

#[derive(Debug, Clone)]
struct PreflightRow {
    solution_id: String,
    pulsar_id: String,
    band: &'static str,
    binary_family: String,
    ntoa: Option<usize>,
    chi2: Option<f64>,
    start_mjd: Option<f64>,
    finish_mjd: Option<f64>,
    span_days: Option<f64>,
    fit_parameter_count: usize,
    jump_count: usize,
    dmjump_count: usize,
    noise_term_count: usize,
    fd_term_count: usize,
    dmx_window_count: usize,
    has_equatorial_astrometry: bool,
    has_ecliptic_astrometry: bool,
    px_mas: Option<f64>,
    multi_solution_pulsar: bool,
    phase1_selected: bool,
    phase1_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct Phase1Selection {
    solution_id: String,
    pulsar_id: String,
    reason: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.root.exists() {
        bail!("timing release root not found: {}", args.root.display());
    }

    let band: ReleaseBand = args.band.into();
    let models = load_release_timing_models(&args.root, band)
        .with_context(|| format!("failed to load {:?} timing models", band))?;
    if models.is_empty() {
        bail!("no {:?} timing models found under {}", band, args.root.display());
    }

    let pulsar_solution_counts = count_solutions_per_pulsar(&models);
    let phase1 = select_phase1_subset(&models);
    let phase1_map = phase1
        .iter()
        .map(|entry| (entry.solution_id.clone(), entry.reason.clone()))
        .collect::<BTreeMap<_, _>>();
    let rows = models
        .iter()
        .map(|model| build_row(model, &pulsar_solution_counts, &phase1_map))
        .collect::<Vec<_>>();

    write_csv(&args.csv_out, &rows)?;
    write_report(&args.report_out, &args.root, band, &models, &rows, &phase1)?;

    println!("Timing models inventoried: {}", models.len());
    println!(
        "Unique pulsars: {}",
        pulsar_solution_counts.keys().collect::<BTreeSet<_>>().len()
    );
    println!("CSV: {}", args.csv_out.display());
    println!("Report: {}", args.report_out.display());
    Ok(())
}

fn count_solutions_per_pulsar(models: &[TimingModel]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for model in models {
        *counts.entry(model.pulsar_id.clone()).or_insert(0) += 1;
    }
    counts
}

fn build_row(
    model: &TimingModel,
    pulsar_solution_counts: &BTreeMap<String, usize>,
    phase1_map: &BTreeMap<String, String>,
) -> PreflightRow {
    let px_mas = model.astrometry.px.as_ref().and_then(|term| term.value);
    PreflightRow {
        solution_id: model.solution_id.clone(),
        pulsar_id: model.pulsar_id.clone(),
        band: model.band.as_str(),
        binary_family: model
            .binary_family
            .as_ref()
            .map_or_else(|| "isolated".to_string(), |family| family.as_str().to_string()),
        ntoa: model.ntoa,
        chi2: model.chi2,
        start_mjd: model.start_mjd,
        finish_mjd: model.finish_mjd,
        span_days: model.observing_span_days(),
        fit_parameter_count: model.fit_parameter_count(),
        jump_count: model.jumps.len(),
        dmjump_count: model.dmjumps.len(),
        noise_term_count: model.noise_terms.len(),
        fd_term_count: model.fd_terms.len(),
        dmx_window_count: model.dispersion.dmx_windows.len(),
        has_equatorial_astrometry: model.astrometry.raj.is_some() && model.astrometry.decj.is_some(),
        has_ecliptic_astrometry: model.astrometry.elong.is_some() && model.astrometry.elat.is_some(),
        px_mas,
        multi_solution_pulsar: pulsar_solution_counts.get(&model.pulsar_id).copied().unwrap_or(0) > 1,
        phase1_selected: phase1_map.contains_key(&model.solution_id),
        phase1_reason: phase1_map.get(&model.solution_id).cloned(),
    }
}

fn select_phase1_subset(models: &[TimingModel]) -> Vec<Phase1Selection> {
    let scout_anchors = [
        ("J1713+0747", "scout_anchor_highest_toa"),
        ("J1903+0327", "scout_anchor_largest_dmx_scatter"),
        ("J2214+3000", "scout_anchor_largest_wideband_dm_scatter"),
        ("J0709+0458", "scout_anchor_largest_full_residual_rms"),
    ];
    let mut selected = Vec::new();
    let mut seen_solution_ids = BTreeSet::new();
    let mut covered_families = BTreeSet::new();

    for (pulsar_id, reason) in scout_anchors {
        if let Some(model) = best_model_for_pulsar(models, pulsar_id)
            && seen_solution_ids.insert(model.solution_id.clone())
        {
            if let Some(family) = &model.binary_family {
                covered_families.insert(family.as_str().to_string());
            }
            selected.push(Phase1Selection {
                solution_id: model.solution_id.clone(),
                pulsar_id: model.pulsar_id.clone(),
                reason: reason.to_string(),
            });
        }
    }

    for family in [
        BinaryFamily::Bt,
        BinaryFamily::Dd,
        BinaryFamily::Ddk,
        BinaryFamily::Ell1,
        BinaryFamily::Ell1h,
    ] {
        if covered_families.contains(family.as_str()) {
            continue;
        }
        if let Some(model) = best_model_for_family(models, &family, &seen_solution_ids) {
            seen_solution_ids.insert(model.solution_id.clone());
            covered_families.insert(family.as_str().to_string());
            selected.push(Phase1Selection {
                solution_id: model.solution_id.clone(),
                pulsar_id: model.pulsar_id.clone(),
                reason: format!("family_representative_{}", family.as_str().to_ascii_lowercase()),
            });
        }
    }

    selected
}

fn best_model_for_pulsar<'a>(models: &'a [TimingModel], pulsar_id: &str) -> Option<&'a TimingModel> {
    models
        .iter()
        .filter(|model| model.pulsar_id == pulsar_id)
        .max_by(|left, right| compare_models(left, right))
}

fn best_model_for_family<'a>(
    models: &'a [TimingModel],
    family: &BinaryFamily,
    excluded_solution_ids: &BTreeSet<String>,
) -> Option<&'a TimingModel> {
    models
        .iter()
        .filter(|model| model.binary_family.as_ref() == Some(family))
        .filter(|model| !excluded_solution_ids.contains(&model.solution_id))
        .max_by(|left, right| compare_models(left, right))
}

fn compare_models(left: &TimingModel, right: &TimingModel) -> std::cmp::Ordering {
    left.ntoa
        .unwrap_or(0)
        .cmp(&right.ntoa.unwrap_or(0))
        .then_with(|| {
            left.fit_parameter_count()
                .cmp(&right.fit_parameter_count())
        })
}

fn write_csv(path: &Path, rows: &[PreflightRow]) -> Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = Writer::from_path(path)?;
    writer.write_record([
        "solution_id",
        "pulsar_id",
        "band",
        "binary_family",
        "ntoa",
        "chi2",
        "start_mjd",
        "finish_mjd",
        "span_days",
        "fit_parameter_count",
        "jump_count",
        "dmjump_count",
        "noise_term_count",
        "fd_term_count",
        "dmx_window_count",
        "has_equatorial_astrometry",
        "has_ecliptic_astrometry",
        "px_mas",
        "multi_solution_pulsar",
        "phase1_selected",
        "phase1_reason",
    ])?;
    for row in rows {
        writer.write_record([
            row.solution_id.as_str(),
            row.pulsar_id.as_str(),
            row.band,
            row.binary_family.as_str(),
            &format_opt_usize(row.ntoa),
            &format_opt_f64(row.chi2),
            &format_opt_f64(row.start_mjd),
            &format_opt_f64(row.finish_mjd),
            &format_opt_f64(row.span_days),
            &row.fit_parameter_count.to_string(),
            &row.jump_count.to_string(),
            &row.dmjump_count.to_string(),
            &row.noise_term_count.to_string(),
            &row.fd_term_count.to_string(),
            &row.dmx_window_count.to_string(),
            &row.has_equatorial_astrometry.to_string(),
            &row.has_ecliptic_astrometry.to_string(),
            &format_opt_f64(row.px_mas),
            &row.multi_solution_pulsar.to_string(),
            &row.phase1_selected.to_string(),
            row.phase1_reason.as_deref().unwrap_or(""),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_report(
    path: &Path,
    root: &Path,
    band: ReleaseBand,
    models: &[TimingModel],
    rows: &[PreflightRow],
    phase1: &[Phase1Selection],
) -> Result<()> {
    ensure_parent_dir(path)?;
    let unique_pulsars = models
        .iter()
        .map(|model| model.pulsar_id.as_str())
        .collect::<BTreeSet<_>>();
    let multi_solution_pulsars = rows
        .iter()
        .filter(|row| row.multi_solution_pulsar)
        .map(|row| row.pulsar_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut family_counts = BTreeMap::<String, usize>::new();
    for row in rows {
        *family_counts.entry(row.binary_family.clone()).or_insert(0) += 1;
    }
    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "title = \"NANOGrav timing refit preflight\"");
    let _ = writeln!(out, "root = {:?}", root.display().to_string());
    let _ = writeln!(out, "band = {:?}", band.as_str());
    let _ = writeln!(
        out,
        "scope_note = \"This lane inventories typed .par timing-model structure for solver planning. It is a refit preflight, not a TOA refit or regenerated-residual solution.\""
    );
    let _ = writeln!(out, "solution_count = {}", models.len());
    let _ = writeln!(out, "unique_pulsar_count = {}", unique_pulsars.len());
    let _ = writeln!(
        out,
        "multi_solution_pulsar_count = {}",
        multi_solution_pulsars.len()
    );
    let _ = writeln!(out, "phase1_subset_count = {}", phase1.len());

    for (family, count) in family_counts {
        let _ = writeln!(out, "family_{}_count = {}", sanitize_key(&family), count);
    }

    let _ = writeln!(out, "\n[[phase1_subset]]");
    for (index, selection) in phase1.iter().enumerate() {
        if index > 0 {
            let _ = writeln!(out, "\n[[phase1_subset]]");
        }
        let _ = writeln!(out, "solution_id = {:?}", selection.solution_id);
        let _ = writeln!(out, "pulsar_id = {:?}", selection.pulsar_id);
        let _ = writeln!(out, "reason = {:?}", selection.reason);
    }

    Ok(fs::write(path, out)?)
}

fn sanitize_key(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn format_opt_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.12}"))
        .unwrap_or_default()
}

fn format_opt_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(())
}
