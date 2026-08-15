use anyhow::{Context, Result, bail};
use csv::Writer;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_15yr_timing_inventory.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value = "reports/nanograv_15yr_timing_inventory.toml")]
    report_out: PathBuf,
}

#[derive(Default)]
struct TimingStats {
    file_count: usize,
    toa_count: usize,
    mjd_min: Option<f64>,
    mjd_max: Option<f64>,
    observatories: BTreeSet<String>,
    frontends: BTreeSet<String>,
    backends: BTreeSet<String>,
    uncertainty_us: RunningStats,
    pp_dm: RunningStats,
    pp_dme: RunningStats,
}

#[derive(Default)]
struct ParStats {
    file_count: usize,
    ntoa_max: Option<usize>,
    start_min: Option<f64>,
    finish_max: Option<f64>,
    f0_hz: Option<f64>,
    dm: Option<f64>,
    px_mas: Option<f64>,
    elong_deg: Option<f64>,
    elat_deg: Option<f64>,
    raj: Option<String>,
    decj: Option<String>,
    dmx_values: ValueSeries,
    dmx_window_start_min: Option<f64>,
    dmx_window_end_max: Option<f64>,
}

#[derive(Default)]
struct NoiseStats {
    chain_file_count: usize,
    chain_row_count: usize,
    parameter_count: usize,
    red_noise_log10_a_medians: ValueSeries,
    red_noise_gamma_medians: ValueSeries,
    efac_medians: ValueSeries,
    ecorr_medians: ValueSeries,
    equad_medians: ValueSeries,
    dmefac_medians: ValueSeries,
    dmequad_medians: ValueSeries,
    t2equad_medians: ValueSeries,
}

#[derive(Default)]
struct ResidualStats {
    full_file_count: usize,
    full_row_count: usize,
    full_mjd_min: Option<f64>,
    full_mjd_max: Option<f64>,
    avg_file_count: usize,
    avg_row_count: usize,
    avg_mjd_min: Option<f64>,
    avg_mjd_max: Option<f64>,
    full_residual_us: RunningStats,
    full_white_residual_us: RunningStats,
    full_uncertainty_us: RunningStats,
    avg_residual_us: RunningStats,
    avg_white_residual_us: RunningStats,
    avg_uncertainty_us: RunningStats,
}

#[derive(Default, Clone)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    sum_sq: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl RunningStats {
    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.sum_sq += value * value;
        update_min(&mut self.min, Some(value));
        update_max(&mut self.max, Some(value));
    }

    fn mean_value(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    fn stddev(&self) -> Option<f64> {
        (self.count > 1).then_some((self.m2 / (self.count as f64 - 1.0)).sqrt())
    }

    fn rms(&self) -> Option<f64> {
        (self.count > 0).then_some((self.sum_sq / self.count as f64).sqrt())
    }
}

#[derive(Default, Clone)]
struct ValueSeries {
    values: Vec<f64>,
}

impl ValueSeries {
    fn push(&mut self, value: f64) {
        if value.is_finite() {
            self.values.push(value);
        }
    }

    fn count(&self) -> usize {
        self.values.len()
    }

    fn mean(&self) -> Option<f64> {
        (!self.values.is_empty())
            .then(|| self.values.iter().sum::<f64>() / self.values.len() as f64)
    }

    fn stddev(&self) -> Option<f64> {
        if self.values.len() < 2 {
            return None;
        }
        let mean = self.mean()?;
        let variance = self
            .values
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f64>()
            / (self.values.len() as f64 - 1.0);
        Some(variance.sqrt())
    }

    fn min(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::min)
    }

    fn max(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::max)
    }

    fn median(&self) -> Option<f64> {
        quantile(&self.values, 0.5)
    }
}

#[derive(Default)]
struct PulsarInventory {
    pulsar: String,
    narrowband_timing: TimingStats,
    wideband_timing: TimingStats,
    narrowband_par: ParStats,
    wideband_par: ParStats,
    narrowband_noise: NoiseStats,
    wideband_noise: NoiseStats,
    residuals: ResidualStats,
}

impl PulsarInventory {
    fn new(pulsar: String) -> Self {
        Self {
            pulsar,
            ..Self::default()
        }
    }
}

pub fn run(args: Args) -> Result<()> {
    if !args.root.exists() {
        bail!("timing release root not found: {}", args.root.display());
    }

    let mut inventories: BTreeMap<String, PulsarInventory> = BTreeMap::new();
    walk_release(&args.root, &args.root, &mut inventories)?;

    if inventories.is_empty() {
        bail!("no pulsar products found under {}", args.root.display());
    }

    write_inventory_csv(&args.csv_out, inventories.values())?;
    write_report(
        &args.report_out,
        &args.root,
        &args.csv_out,
        inventories.values(),
    )?;

    println!("Pulsars inventoried: {}", inventories.len());
    println!("CSV: {}", args.csv_out.display());
    println!("Report: {}", args.report_out.display());
    Ok(())
}

fn walk_release(
    release_root: &Path,
    scan_dir: &Path,
    inventories: &mut BTreeMap<String, PulsarInventory>,
) -> Result<()> {
    for entry in
        fs::read_dir(scan_dir).with_context(|| format!("read_dir {}", scan_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_release(release_root, &path, inventories)?;
            continue;
        }
        inventory_file(release_root, &path, inventories)?;
    }
    Ok(())
}

fn inventory_file(
    root: &Path,
    path: &Path,
    inventories: &mut BTreeMap<String, PulsarInventory>,
) -> Result<()> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("strip_prefix {} from {}", root.display(), path.display()))?;
    let rel = relative.to_string_lossy();

    if rel.contains("narrowband/tim/") && rel.ends_with(".tim") {
        let pulsar = canonical_pulsar_name(path_stem_string(path)?);
        let inv = ensure_inventory(inventories, &pulsar);
        parse_tim_file(path, &mut inv.narrowband_timing)?;
        return Ok(());
    }
    if rel.contains("wideband/tim/") && rel.ends_with(".tim") {
        let pulsar = canonical_pulsar_name(path_stem_string(path)?);
        let inv = ensure_inventory(inventories, &pulsar);
        parse_tim_file(path, &mut inv.wideband_timing)?;
        return Ok(());
    }
    if rel.contains("narrowband/par/") && rel.ends_with(".par") {
        let pulsar = canonical_pulsar_name(path_stem_string(path)?);
        let inv = ensure_inventory(inventories, &pulsar);
        parse_par_file(path, &mut inv.narrowband_par)?;
        return Ok(());
    }
    if rel.contains("wideband/par/") && rel.ends_with(".par") {
        let pulsar = canonical_pulsar_name(path_stem_string(path)?);
        let inv = ensure_inventory(inventories, &pulsar);
        parse_par_file(path, &mut inv.wideband_par)?;
        return Ok(());
    }
    if rel.contains("narrowband/noise/") {
        let pulsar = canonical_pulsar_name(file_name_string(path)?);
        let inv = ensure_inventory(inventories, &pulsar);
        parse_noise_file(path, &mut inv.narrowband_noise)?;
        return Ok(());
    }
    if rel.contains("wideband/noise/") {
        let pulsar = canonical_pulsar_name(file_name_string(path)?);
        let inv = ensure_inventory(inventories, &pulsar);
        parse_noise_file(path, &mut inv.wideband_noise)?;
        return Ok(());
    }
    if rel.contains("residuals/") && rel.ends_with(".res") {
        let pulsar = canonical_pulsar_name(file_name_string(path)?);
        let inv = ensure_inventory(inventories, &pulsar);
        parse_residual_file(path, &mut inv.residuals)?;
    }
    Ok(())
}

fn ensure_inventory<'a>(
    inventories: &'a mut BTreeMap<String, PulsarInventory>,
    pulsar: &str,
) -> &'a mut PulsarInventory {
    inventories
        .entry(pulsar.to_string())
        .or_insert_with(|| PulsarInventory::new(pulsar.to_string()))
}

fn path_stem_string(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.to_string())
        .ok_or_else(|| anyhow::anyhow!("missing UTF-8 file stem for {}", path.display()))
}

fn file_name_string(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.to_string())
        .ok_or_else(|| anyhow::anyhow!("missing UTF-8 file name for {}", path.display()))
}

fn canonical_pulsar_name(raw: String) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if out.is_empty() {
            if ch == 'J' || ch == 'B' {
                out.push(ch);
            } else if ch.is_ascii_alphanumeric() {
                continue;
            } else {
                break;
            }
            continue;
        }
        if ch.is_ascii_digit() || ch == '+' || ch == '-' {
            out.push(ch);
        } else {
            break;
        }
    }
    if out.is_empty() { raw } else { out }
}

fn parse_tim_file(path: &Path, stats: &mut TimingStats) -> Result<()> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    stats.file_count += 1;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('C') || trimmed.starts_with("FORMAT") {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
        }
        let Ok(mjd) = fields[2].parse::<f64>() else {
            continue;
        };
        stats.toa_count += 1;
        update_min(&mut stats.mjd_min, Some(mjd));
        update_max(&mut stats.mjd_max, Some(mjd));
        if let Ok(uncertainty_us) = fields[3].parse::<f64>() {
            stats.uncertainty_us.push(uncertainty_us);
        }
        stats.observatories.insert(fields[4].to_string());
        collect_flag_value(&fields, "-fe", &mut stats.frontends);
        collect_flag_value(&fields, "-be", &mut stats.backends);
        collect_numeric_flag_value(&fields, "-pp_dm", &mut stats.pp_dm);
        collect_numeric_flag_value(&fields, "-pp_dme", &mut stats.pp_dme);
    }
    Ok(())
}

fn collect_flag_value(fields: &[&str], flag: &str, output: &mut BTreeSet<String>) {
    for window in fields.windows(2) {
        if window[0] == flag {
            output.insert(window[1].to_string());
        }
    }
}

fn collect_numeric_flag_value(fields: &[&str], flag: &str, output: &mut RunningStats) {
    for window in fields.windows(2) {
        if window[0] == flag
            && let Ok(value) = window[1].parse::<f64>()
        {
            output.push(value);
        }
    }
}

fn parse_par_file(path: &Path, stats: &mut ParStats) -> Result<()> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    stats.file_count += 1;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 2 {
            continue;
        }
        match fields[0] {
            "START" => update_min(&mut stats.start_min, parse_f64(fields[1])),
            "FINISH" => update_max(&mut stats.finish_max, parse_f64(fields[1])),
            "NTOA" => update_max_usize(&mut stats.ntoa_max, parse_usize(fields[1])),
            "F0" => set_if_some(&mut stats.f0_hz, parse_f64(fields[1])),
            "DM" => set_if_some(&mut stats.dm, parse_f64(fields[1])),
            "PX" => set_if_some(&mut stats.px_mas, parse_f64(fields[1])),
            "ELONG" => set_if_some(&mut stats.elong_deg, parse_f64(fields[1])),
            "ELAT" => set_if_some(&mut stats.elat_deg, parse_f64(fields[1])),
            "RAJ" => stats.raj = Some(fields[1].to_string()),
            "DECJ" => stats.decj = Some(fields[1].to_string()),
            key if key.starts_with("DMX_") && !key.starts_with("DMXR") => {
                if let Some(value) = parse_f64(fields[1]) {
                    stats.dmx_values.push(value);
                }
            }
            key if key.starts_with("DMXR1_") => {
                update_min(&mut stats.dmx_window_start_min, parse_f64(fields[1]));
            }
            key if key.starts_with("DMXR2_") => {
                update_max(&mut stats.dmx_window_end_max, parse_f64(fields[1]));
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_noise_file(path: &Path, stats: &mut NoiseStats) -> Result<()> {
    let name = file_name_string(path)?;
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    if name.ends_with(".pars.txt") {
        stats.parameter_count += content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .count();
    } else if name.contains(".chain_") && name.ends_with(".txt") {
        stats.chain_file_count += 1;
        parse_noise_chain(path, &content, stats)?;
    }
    Ok(())
}

fn parse_noise_chain(path: &Path, content: &str, stats: &mut NoiseStats) -> Result<()> {
    let pars_name = file_name_string(path)?.replace(".chain_1.txt", ".pars.txt");
    let pars_path = path.with_file_name(pars_name);
    let pars_content =
        fs::read_to_string(&pars_path).with_context(|| format!("read {}", pars_path.display()))?;
    let names: Vec<String> = pars_content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.to_string())
        .collect();
    if names.is_empty() {
        return Ok(());
    }

    let mut columns = vec![Vec::<f64>::new(); names.len()];
    for line in content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < names.len() {
            continue;
        }
        stats.chain_row_count += 1;
        for (idx, name) in names.iter().enumerate() {
            let Ok(value) = fields[idx].parse::<f64>() else {
                continue;
            };
            let target = if name.ends_with("red_noise_log10_A") {
                Some(&mut stats.red_noise_log10_a_medians)
            } else if name.ends_with("red_noise_gamma") {
                Some(&mut stats.red_noise_gamma_medians)
            } else if name.ends_with("_efac") {
                Some(&mut stats.efac_medians)
            } else if name.contains("log10_ecorr") {
                Some(&mut stats.ecorr_medians)
            } else if name.contains("log10_equad") {
                Some(&mut stats.equad_medians)
            } else if name.ends_with("_dmefac") {
                Some(&mut stats.dmefac_medians)
            } else if name.contains("log10_dmequad") {
                Some(&mut stats.dmequad_medians)
            } else if name.contains("log10_t2equad") {
                Some(&mut stats.t2equad_medians)
            } else {
                None
            };
            if target.is_some() {
                columns[idx].push(value);
            }
        }
    }

    for (idx, name) in names.iter().enumerate() {
        if columns[idx].is_empty() {
            continue;
        }
        let Some(median) = quantile(&columns[idx], 0.5) else {
            continue;
        };
        if name.ends_with("red_noise_log10_A") {
            stats.red_noise_log10_a_medians.push(median);
        } else if name.ends_with("red_noise_gamma") {
            stats.red_noise_gamma_medians.push(median);
        } else if name.ends_with("_efac") {
            stats.efac_medians.push(median);
        } else if name.contains("log10_ecorr") {
            stats.ecorr_medians.push(median);
        } else if name.contains("log10_equad") {
            stats.equad_medians.push(median);
        } else if name.ends_with("_dmefac") {
            stats.dmefac_medians.push(median);
        } else if name.contains("log10_dmequad") {
            stats.dmequad_medians.push(median);
        } else if name.contains("log10_t2equad") {
            stats.t2equad_medians.push(median);
        }
    }
    Ok(())
}

fn parse_residual_file(path: &Path, stats: &mut ResidualStats) -> Result<()> {
    let name = file_name_string(path)?;
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let is_full = name.contains(".full.");
    if is_full {
        stats.full_file_count += 1;
    } else {
        stats.avg_file_count += 1;
    }
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        let Ok(mjd) = fields[0].parse::<f64>() else {
            continue;
        };
        if is_full {
            stats.full_row_count += 1;
            update_min(&mut stats.full_mjd_min, Some(mjd));
            update_max(&mut stats.full_mjd_max, Some(mjd));
            if let Some(value) = parse_field_f64(&fields, 2) {
                stats.full_residual_us.push(value);
            }
            if fields.len() >= 7 {
                if let Some(value) = parse_field_f64(&fields, 3) {
                    stats.full_white_residual_us.push(value);
                }
                if let Some(value) = parse_field_f64(&fields, 4) {
                    stats.full_uncertainty_us.push(value);
                }
            } else if let Some(value) = parse_field_f64(&fields, 3) {
                stats.full_uncertainty_us.push(value);
            }
        } else {
            stats.avg_row_count += 1;
            update_min(&mut stats.avg_mjd_min, Some(mjd));
            update_max(&mut stats.avg_mjd_max, Some(mjd));
            if let Some(value) = parse_field_f64(&fields, 2) {
                stats.avg_residual_us.push(value);
            }
            if fields.len() >= 5 {
                if let Some(value) = parse_field_f64(&fields, 3) {
                    stats.avg_white_residual_us.push(value);
                }
                if let Some(value) = parse_field_f64(&fields, 4) {
                    stats.avg_uncertainty_us.push(value);
                }
            } else if let Some(value) = parse_field_f64(&fields, 3) {
                stats.avg_uncertainty_us.push(value);
            }
        }
    }
    Ok(())
}

fn parse_field_f64(fields: &[&str], index: usize) -> Option<f64> {
    fields
        .get(index)?
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn parse_f64(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|parsed| parsed.is_finite())
}

fn parse_usize(value: &str) -> Option<usize> {
    value.parse::<usize>().ok()
}

fn quantile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.len() == 1 {
        return sorted.first().copied();
    }
    let target = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = target.floor() as usize;
    let upper = target.ceil() as usize;
    if lower == upper {
        return sorted.get(lower).copied();
    }
    let frac = target - lower as f64;
    Some(sorted[lower] + frac * (sorted[upper] - sorted[lower]))
}

fn set_if_some(slot: &mut Option<f64>, value: Option<f64>) {
    if slot.is_none() {
        *slot = value;
    }
}

fn update_min(slot: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        match slot {
            Some(current) if value >= *current => {}
            _ => *slot = Some(value),
        }
    }
}

fn update_max(slot: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        match slot {
            Some(current) if value <= *current => {}
            _ => *slot = Some(value),
        }
    }
}

fn update_max_usize(slot: &mut Option<usize>, value: Option<usize>) {
    if let Some(value) = value {
        match slot {
            Some(current) if value <= *current => {}
            _ => *slot = Some(value),
        }
    }
}

fn write_inventory_csv<'a>(
    path: &Path,
    inventories: impl Iterator<Item = &'a PulsarInventory>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut writer = Writer::from_path(path).with_context(|| format!("open {}", path.display()))?;
    writer.write_record([
        "pulsar",
        "nb_tim_files",
        "nb_toa_count",
        "nb_mjd_min",
        "nb_mjd_max",
        "nb_observatories",
        "nb_frontends",
        "nb_backends",
        "nb_toa_unc_mean_us",
        "wb_tim_files",
        "wb_toa_count",
        "wb_mjd_min",
        "wb_mjd_max",
        "wb_observatories",
        "wb_frontends",
        "wb_backends",
        "wb_toa_unc_mean_us",
        "wb_pp_dm_count",
        "wb_pp_dm_mean",
        "wb_pp_dm_std",
        "wb_pp_dm_min",
        "wb_pp_dm_max",
        "wb_pp_dme_mean",
        "nb_par_files",
        "nb_par_ntoa_max",
        "nb_par_start_min",
        "nb_par_finish_max",
        "wb_par_files",
        "wb_par_ntoa_max",
        "wb_par_start_min",
        "wb_par_finish_max",
        "f0_hz",
        "dm",
        "px_mas",
        "elong_deg",
        "elat_deg",
        "dmx_count",
        "dmx_mean",
        "dmx_std",
        "dmx_min",
        "dmx_max",
        "dmx_epoch_span_days",
        "nb_noise_chain_files",
        "nb_noise_chain_rows",
        "nb_noise_parameter_count",
        "nb_red_noise_log10_a_median",
        "nb_red_noise_gamma_median",
        "nb_efac_median",
        "nb_log10_ecorr_median",
        "nb_log10_equad_median",
        "wb_noise_chain_files",
        "wb_noise_chain_rows",
        "wb_noise_parameter_count",
        "wb_red_noise_log10_a_median",
        "wb_red_noise_gamma_median",
        "wb_efac_median",
        "wb_dmefac_median",
        "wb_log10_dmequad_median",
        "wb_log10_t2equad_median",
        "res_full_files",
        "res_full_rows",
        "res_full_mjd_min",
        "res_full_mjd_max",
        "res_full_rms_us",
        "res_full_white_rms_us",
        "res_full_unc_mean_us",
        "res_full_whitening_gain",
        "res_avg_files",
        "res_avg_rows",
        "res_avg_mjd_min",
        "res_avg_mjd_max",
        "res_avg_rms_us",
        "res_avg_white_rms_us",
        "res_avg_unc_mean_us",
        "res_avg_whitening_gain",
    ])?;
    for inv in inventories {
        writer.write_record([
            inv.pulsar.clone(),
            inv.narrowband_timing.file_count.to_string(),
            inv.narrowband_timing.toa_count.to_string(),
            format_opt(inv.narrowband_timing.mjd_min),
            format_opt(inv.narrowband_timing.mjd_max),
            join_set(&inv.narrowband_timing.observatories),
            join_set(&inv.narrowband_timing.frontends),
            join_set(&inv.narrowband_timing.backends),
            format_opt(inv.narrowband_timing.uncertainty_us.mean_value()),
            inv.wideband_timing.file_count.to_string(),
            inv.wideband_timing.toa_count.to_string(),
            format_opt(inv.wideband_timing.mjd_min),
            format_opt(inv.wideband_timing.mjd_max),
            join_set(&inv.wideband_timing.observatories),
            join_set(&inv.wideband_timing.frontends),
            join_set(&inv.wideband_timing.backends),
            format_opt(inv.wideband_timing.uncertainty_us.mean_value()),
            inv.wideband_timing.pp_dm.count.to_string(),
            format_opt(inv.wideband_timing.pp_dm.mean_value()),
            format_opt(inv.wideband_timing.pp_dm.stddev()),
            format_opt(inv.wideband_timing.pp_dm.min),
            format_opt(inv.wideband_timing.pp_dm.max),
            format_opt(inv.wideband_timing.pp_dme.mean_value()),
            inv.narrowband_par.file_count.to_string(),
            format_opt_usize(inv.narrowband_par.ntoa_max),
            format_opt(inv.narrowband_par.start_min),
            format_opt(inv.narrowband_par.finish_max),
            inv.wideband_par.file_count.to_string(),
            format_opt_usize(inv.wideband_par.ntoa_max),
            format_opt(inv.wideband_par.start_min),
            format_opt(inv.wideband_par.finish_max),
            format_opt(inv.wideband_par.f0_hz.or(inv.narrowband_par.f0_hz)),
            format_opt(inv.wideband_par.dm.or(inv.narrowband_par.dm)),
            format_opt(inv.wideband_par.px_mas.or(inv.narrowband_par.px_mas)),
            format_opt(inv.wideband_par.elong_deg.or(inv.narrowband_par.elong_deg)),
            format_opt(inv.wideband_par.elat_deg.or(inv.narrowband_par.elat_deg)),
            inv.wideband_par
                .dmx_values
                .count()
                .max(inv.narrowband_par.dmx_values.count())
                .to_string(),
            format_opt(
                inv.wideband_par
                    .dmx_values
                    .mean()
                    .or(inv.narrowband_par.dmx_values.mean()),
            ),
            format_opt(
                inv.wideband_par
                    .dmx_values
                    .stddev()
                    .or(inv.narrowband_par.dmx_values.stddev()),
            ),
            format_opt(
                inv.wideband_par
                    .dmx_values
                    .min()
                    .or(inv.narrowband_par.dmx_values.min()),
            ),
            format_opt(
                inv.wideband_par
                    .dmx_values
                    .max()
                    .or(inv.narrowband_par.dmx_values.max()),
            ),
            format_opt(
                match (
                    inv.wideband_par
                        .dmx_window_start_min
                        .or(inv.narrowband_par.dmx_window_start_min),
                    inv.wideband_par
                        .dmx_window_end_max
                        .or(inv.narrowband_par.dmx_window_end_max),
                ) {
                    (Some(start), Some(end)) => Some(end - start),
                    _ => None,
                },
            ),
            inv.narrowband_noise.chain_file_count.to_string(),
            inv.narrowband_noise.chain_row_count.to_string(),
            inv.narrowband_noise.parameter_count.to_string(),
            format_opt(inv.narrowband_noise.red_noise_log10_a_medians.median()),
            format_opt(inv.narrowband_noise.red_noise_gamma_medians.median()),
            format_opt(inv.narrowband_noise.efac_medians.median()),
            format_opt(inv.narrowband_noise.ecorr_medians.median()),
            format_opt(inv.narrowband_noise.equad_medians.median()),
            inv.wideband_noise.chain_file_count.to_string(),
            inv.wideband_noise.chain_row_count.to_string(),
            inv.wideband_noise.parameter_count.to_string(),
            format_opt(inv.wideband_noise.red_noise_log10_a_medians.median()),
            format_opt(inv.wideband_noise.red_noise_gamma_medians.median()),
            format_opt(inv.wideband_noise.efac_medians.median()),
            format_opt(inv.wideband_noise.dmefac_medians.median()),
            format_opt(inv.wideband_noise.dmequad_medians.median()),
            format_opt(inv.wideband_noise.t2equad_medians.median()),
            inv.residuals.full_file_count.to_string(),
            inv.residuals.full_row_count.to_string(),
            format_opt(inv.residuals.full_mjd_min),
            format_opt(inv.residuals.full_mjd_max),
            format_opt(inv.residuals.full_residual_us.rms()),
            format_opt(inv.residuals.full_white_residual_us.rms()),
            format_opt(inv.residuals.full_uncertainty_us.mean_value()),
            format_opt(
                match (
                    inv.residuals.full_residual_us.rms(),
                    inv.residuals.full_white_residual_us.rms(),
                ) {
                    (Some(raw), Some(white)) if white > 0.0 => Some(raw / white),
                    _ => None,
                },
            ),
            inv.residuals.avg_file_count.to_string(),
            inv.residuals.avg_row_count.to_string(),
            format_opt(inv.residuals.avg_mjd_min),
            format_opt(inv.residuals.avg_mjd_max),
            format_opt(inv.residuals.avg_residual_us.rms()),
            format_opt(inv.residuals.avg_white_residual_us.rms()),
            format_opt(inv.residuals.avg_uncertainty_us.mean_value()),
            format_opt(
                match (
                    inv.residuals.avg_residual_us.rms(),
                    inv.residuals.avg_white_residual_us.rms(),
                ) {
                    (Some(raw), Some(white)) if white > 0.0 => Some(raw / white),
                    _ => None,
                },
            ),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_report<'a>(
    path: &Path,
    root: &Path,
    csv_out: &Path,
    inventories: impl Iterator<Item = &'a PulsarInventory>,
) -> Result<()> {
    let rows: Vec<&PulsarInventory> = inventories.collect();
    let total = rows.len();
    let nb_ready = rows
        .iter()
        .filter(|row| row.narrowband_timing.toa_count > 0 && row.narrowband_par.file_count > 0)
        .count();
    let wb_ready = rows
        .iter()
        .filter(|row| row.wideband_timing.toa_count > 0 && row.wideband_par.file_count > 0)
        .count();
    let both_ready = rows
        .iter()
        .filter(|row| {
            row.narrowband_timing.toa_count > 0
                && row.narrowband_par.file_count > 0
                && row.wideband_timing.toa_count > 0
                && row.wideband_par.file_count > 0
        })
        .count();
    let residual_ready = rows
        .iter()
        .filter(|row| row.residuals.full_file_count > 0 && row.residuals.avg_file_count > 0)
        .count();
    let dmx_ready = rows
        .iter()
        .filter(|row| {
            row.narrowband_par.dmx_values.count() > 0 || row.wideband_par.dmx_values.count() > 0
        })
        .count();
    let wideband_dm_ready = rows
        .iter()
        .filter(|row| row.wideband_timing.pp_dm.count > 0)
        .count();
    let parallax_ready = rows
        .iter()
        .filter(|row| preferred_px(row).is_some())
        .count();
    let whitened_residual_ready = rows
        .iter()
        .filter(|row| row.residuals.full_white_residual_us.count > 0)
        .count();
    let noise_ready = rows
        .iter()
        .filter(|row| {
            row.narrowband_noise.chain_file_count > 0
                || row.wideband_noise.chain_file_count > 0
                || row.narrowband_noise.parameter_count > 0
                || row.wideband_noise.parameter_count > 0
        })
        .count();
    let max_toa = rows
        .iter()
        .max_by_key(|row| row.narrowband_timing.toa_count + row.wideband_timing.toa_count);
    let max_dmx = rows
        .iter()
        .filter_map(|row| {
            let value = row
                .wideband_par
                .dmx_values
                .stddev()
                .or(row.narrowband_par.dmx_values.stddev())?;
            Some((row, value))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let max_wb_dm = rows
        .iter()
        .filter_map(|row| row.wideband_timing.pp_dm.stddev().map(|value| (row, value)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let max_full_rms = rows
        .iter()
        .filter_map(|row| {
            row.residuals
                .full_residual_us
                .rms()
                .map(|value| (row, value))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let max_parallax = rows
        .iter()
        .filter_map(|row| preferred_px(row).map(|value| (row, value)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let top_total_toa = top_ranked(&rows, 5, |row| Some(total_toa_count(row) as f64));
    let top_dmx = top_ranked(&rows, 5, |row| {
        row.wideband_par
            .dmx_values
            .stddev()
            .or(row.narrowband_par.dmx_values.stddev())
    });
    let top_wb_dm = top_ranked(&rows, 5, |row| row.wideband_timing.pp_dm.stddev());
    let top_full_rms = top_ranked(&rows, 5, |row| row.residuals.full_residual_us.rms());
    let top_whitening_gain = top_ranked(&rows, 5, full_whitening_gain);
    let top_red_noise = top_ranked(&rows, 5, |row| {
        row.narrowband_noise
            .red_noise_log10_a_medians
            .median()
            .or(row.wideband_noise.red_noise_log10_a_medians.median())
    });
    let top_parallax = top_ranked(&rows, 5, preferred_px);

    let mut out = String::new();
    let _ = writeln!(out, "root = {:?}", root.display().to_string());
    let _ = writeln!(out, "inventory_csv = {:?}", csv_out.display().to_string());
    let _ = writeln!(out, "pulsar_count = {}", total);
    let _ = writeln!(out, "narrowband_ready_count = {}", nb_ready);
    let _ = writeln!(out, "wideband_ready_count = {}", wb_ready);
    let _ = writeln!(out, "dual_band_ready_count = {}", both_ready);
    let _ = writeln!(out, "residual_ready_count = {}", residual_ready);
    let _ = writeln!(out, "noise_ready_count = {}", noise_ready);
    let _ = writeln!(out, "dmx_ready_count = {}", dmx_ready);
    let _ = writeln!(out, "wideband_dm_ready_count = {}", wideband_dm_ready);
    let _ = writeln!(out, "parallax_ready_count = {}", parallax_ready);
    let _ = writeln!(
        out,
        "whitened_full_residual_ready_count = {}",
        whitened_residual_ready
    );
    if let Some(max_toa) = max_toa {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_toa_pulsar]");
        let _ = writeln!(out, "name = {:?}", max_toa.pulsar);
        let _ = writeln!(
            out,
            "total_toa_count = {}",
            max_toa.narrowband_timing.toa_count + max_toa.wideband_timing.toa_count
        );
        let _ = writeln!(
            out,
            "narrowband_toa_count = {}",
            max_toa.narrowband_timing.toa_count
        );
        let _ = writeln!(
            out,
            "wideband_toa_count = {}",
            max_toa.wideband_timing.toa_count
        );
    }
    if let Some((row, value)) = max_full_rms {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_full_residual_rms]");
        let _ = writeln!(out, "name = {:?}", row.pulsar);
        let _ = writeln!(out, "rms_us = {:.12}", value);
    }
    if let Some((row, value)) = max_dmx {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_dmx_std]");
        let _ = writeln!(out, "name = {:?}", row.pulsar);
        let _ = writeln!(out, "dmx_std = {:.12}", value);
    }
    if let Some((row, value)) = max_wb_dm {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_wideband_dm_std]");
        let _ = writeln!(out, "name = {:?}", row.pulsar);
        let _ = writeln!(out, "pp_dm_std = {:.12}", value);
    }
    if let Some((row, value)) = max_parallax {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_parallax_mas]");
        let _ = writeln!(out, "name = {:?}", row.pulsar);
        let _ = writeln!(out, "px_mas = {:.12}", value);
    }
    write_ranked_metric(&mut out, "top_total_toa", "total_toa_count", &top_total_toa);
    write_ranked_metric(&mut out, "top_dmx_std", "dmx_std", &top_dmx);
    write_ranked_metric(&mut out, "top_wideband_dm_std", "pp_dm_std", &top_wb_dm);
    write_ranked_metric(&mut out, "top_full_residual_rms", "rms_us", &top_full_rms);
    write_ranked_metric(
        &mut out,
        "top_full_whitening_gain",
        "whitening_gain",
        &top_whitening_gain,
    );
    write_ranked_metric(
        &mut out,
        "top_red_noise_log10_a",
        "log10_a_median",
        &top_red_noise,
    );
    write_ranked_metric(&mut out, "top_parallax_mas", "px_mas", &top_parallax);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn top_ranked<'a, F>(
    rows: &'a [&'a PulsarInventory],
    top_n: usize,
    metric: F,
) -> Vec<(&'a PulsarInventory, f64)>
where
    F: Fn(&PulsarInventory) -> Option<f64>,
{
    let mut ranked = rows
        .iter()
        .filter_map(|row| metric(row).map(|value| (*row, value)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(top_n);
    ranked
}

fn write_ranked_metric(
    out: &mut String,
    section: &str,
    value_key: &str,
    ranked: &[(&PulsarInventory, f64)],
) {
    for (rank, (row, value)) in ranked.iter().enumerate() {
        let _ = writeln!(out);
        let _ = writeln!(out, "[[{section}]]");
        let _ = writeln!(out, "rank = {}", rank + 1);
        let _ = writeln!(out, "name = {:?}", row.pulsar);
        let _ = writeln!(out, "{value_key} = {:.12}", value);
        if value_key != "total_toa_count" {
            let _ = writeln!(out, "total_toa_count = {}", total_toa_count(row));
        }
        let _ = writeln!(
            out,
            "narrowband_toa_count = {}",
            row.narrowband_timing.toa_count
        );
        let _ = writeln!(
            out,
            "wideband_toa_count = {}",
            row.wideband_timing.toa_count
        );
        if value_key != "px_mas"
            && let Some(px_mas) = preferred_px(row)
        {
            let _ = writeln!(out, "px_mas = {:.12}", px_mas);
        }
        if value_key != "elong_deg"
            && let Some(elong_deg) = preferred_elong(row)
        {
            let _ = writeln!(out, "elong_deg = {:.12}", elong_deg);
        }
        if value_key != "elat_deg"
            && let Some(elat_deg) = preferred_elat(row)
        {
            let _ = writeln!(out, "elat_deg = {:.12}", elat_deg);
        }
    }
}

fn total_toa_count(row: &PulsarInventory) -> usize {
    row.narrowband_timing.toa_count + row.wideband_timing.toa_count
}

fn full_whitening_gain(row: &PulsarInventory) -> Option<f64> {
    match (
        row.residuals.full_residual_us.rms(),
        row.residuals.full_white_residual_us.rms(),
    ) {
        (Some(raw), Some(white)) if white > 0.0 => Some(raw / white),
        _ => None,
    }
}

fn preferred_px(row: &PulsarInventory) -> Option<f64> {
    row.wideband_par.px_mas.or(row.narrowband_par.px_mas)
}

fn preferred_elong(row: &PulsarInventory) -> Option<f64> {
    row.wideband_par.elong_deg.or(row.narrowband_par.elong_deg)
}

fn preferred_elat(row: &PulsarInventory) -> Option<f64> {
    row.wideband_par.elat_deg.or(row.narrowband_par.elat_deg)
}

fn format_opt(value: Option<f64>) -> String {
    value.map(|val| format!("{val:.12}")).unwrap_or_default()
}

fn format_opt_usize(value: Option<usize>) -> String {
    value.map(|val| val.to_string()).unwrap_or_default()
}

fn join_set(values: &BTreeSet<String>) -> String {
    values.iter().cloned().collect::<Vec<_>>().join("|")
}
