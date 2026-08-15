use anyhow::{Context, Result, bail};
use csv::Writer;
use gororoba_cli_data::nanograv_timing::{
    DmxPoint, PulsarTimingData, ResidualPoint, TimingModelMetadata, WidebandDmPoint, load_release,
};
use std::{
    collections::BTreeMap,
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

    #[arg(long, default_value = "data/csv/nanograv_15yr_propagation_pulsars.csv")]
    pulsar_csv_out: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_15yr_pairwise_hd_audit.csv")]
    pair_csv_out: PathBuf,

    #[arg(long, default_value = "reports/nanograv_15yr_propagation_audit.toml")]
    report_out: PathBuf,

    #[arg(long, default_value_t = 14.0)]
    match_tolerance_days: f64,

    #[arg(long, default_value_t = 30.0)]
    pair_bin_days: f64,

    #[arg(long, default_value_t = 8)]
    min_pair_overlap_bins: usize,
}

#[derive(Debug, Clone, Default)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min: Option<f64>,
    max: Option<f64>,
    sum_sq: f64,
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
        update_min(&mut self.min, value);
        update_max(&mut self.max, value);
    }

    fn mean_value(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    fn rms(&self) -> Option<f64> {
        (self.count > 0).then_some((self.sum_sq / self.count as f64).sqrt())
    }
}

#[derive(Debug, Clone)]
struct PulsarAuditRow {
    pulsar: String,
    elong_deg: Option<f64>,
    elat_deg: Option<f64>,
    px_mas: Option<f64>,
    dmx_count: usize,
    dmx_epoch_min: Option<f64>,
    dmx_epoch_max: Option<f64>,
    dmx_mean: Option<f64>,
    dmx_std: Option<f64>,
    dmx_slope_per_day: Option<f64>,
    wb_dm_count: usize,
    wb_dm_epoch_min: Option<f64>,
    wb_dm_epoch_max: Option<f64>,
    wb_dm_mean: Option<f64>,
    wb_dm_std: Option<f64>,
    wb_dm_slope_per_day: Option<f64>,
    avg_residual_count: usize,
    avg_residual_rms_us: Option<f64>,
    avg_white_rms_us: Option<f64>,
    avg_uncertainty_mean_us: Option<f64>,
    avg_abs_res_vs_unc_pearson: Option<f64>,
    avg_abs_res_vs_unc_spearman: Option<f64>,
    avg_abs_res_vs_dmx_pairs: usize,
    avg_abs_res_vs_dmx_pearson: Option<f64>,
    avg_abs_res_vs_dmx_spearman: Option<f64>,
    avg_abs_res_vs_wb_dm_pairs: usize,
    avg_abs_res_vs_wb_dm_pearson: Option<f64>,
    avg_abs_res_vs_wb_dm_spearman: Option<f64>,
}

#[derive(Debug, Clone)]
struct PairAuditRow {
    pulsar_a: String,
    pulsar_b: String,
    separation_deg: f64,
    hellings_downs: f64,
    overlap_bins: usize,
    avg_residual_pearson: f64,
    avg_white_residual_pearson: Option<f64>,
}

#[derive(Debug, Clone)]
struct PreparedPulsar {
    pulsar: String,
    sky_vector: Option<[f64; 3]>,
    raw_bins: BTreeMap<i64, f64>,
    white_bins: BTreeMap<i64, f64>,
    row: PulsarAuditRow,
}

pub fn run(args: Args) -> Result<()> {
    if !args.root.exists() {
        bail!("timing release root not found: {}", args.root.display());
    }

    let release = load_release(&args.root)?;
    if release.is_empty() {
        bail!(
            "no timing-release products found under {}",
            args.root.display()
        );
    }

    let mut prepared = release
        .values()
        .map(|data| prepare_pulsar(data, &args))
        .collect::<Result<Vec<_>>>()?;
    prepared.sort_by(|a, b| a.pulsar.cmp(&b.pulsar));

    let pair_rows = build_pair_rows(&prepared, &args);

    write_pulsar_csv(
        &args.pulsar_csv_out,
        prepared.iter().map(|entry| &entry.row),
    )?;
    write_pair_csv(&args.pair_csv_out, pair_rows.iter())?;
    write_report(
        &args.report_out,
        &args.root,
        &args.pulsar_csv_out,
        &args.pair_csv_out,
        &prepared,
        &pair_rows,
        &args,
    )?;

    println!("Pulsars audited: {}", prepared.len());
    println!("Pair rows emitted: {}", pair_rows.len());
    println!("Pulsar CSV: {}", args.pulsar_csv_out.display());
    println!("Pair CSV: {}", args.pair_csv_out.display());
    println!("Report: {}", args.report_out.display());
    Ok(())
}

fn prepare_pulsar(data: &PulsarTimingData, args: &Args) -> Result<PreparedPulsar> {
    let meta = preferred_metadata(data);
    let elong_deg = meta.elong_deg;
    let elat_deg = meta.elat_deg;
    let px_mas = meta.px_mas;
    let sky_vector = sky_vector(meta);

    let dmx_series = data
        .dmx
        .iter()
        .map(|point| (point.epoch_mjd, point.dmx_value))
        .collect::<Vec<_>>();
    let wb_dm_series = data
        .wideband_dm
        .iter()
        .map(|point| (point.mjd, point.pp_dm))
        .collect::<Vec<_>>();

    let mut avg_res_stats = RunningStats::default();
    let mut avg_white_stats = RunningStats::default();
    let mut avg_unc_stats = RunningStats::default();
    let mut abs_residuals = Vec::new();
    let mut uncertainties = Vec::new();
    for point in &data.avg_residuals {
        avg_res_stats.push(point.residual_us);
        if let Some(value) = point.white_residual_us {
            avg_white_stats.push(value);
        }
        if let Some(value) = point.uncertainty_us {
            avg_unc_stats.push(value);
            abs_residuals.push(point.residual_us.abs());
            uncertainties.push(value);
        }
    }

    let dmx_pairs = match_avg_residuals_to_dmx(&data.avg_residuals, &data.dmx);
    let wb_dm_pairs = match_avg_residuals_to_wb_dm(
        &data.avg_residuals,
        &data.wideband_dm,
        args.match_tolerance_days,
    );

    let row = PulsarAuditRow {
        pulsar: data.pulsar.clone(),
        elong_deg,
        elat_deg,
        px_mas,
        dmx_count: data.dmx.len(),
        dmx_epoch_min: data.dmx.first().map(|point| point.epoch_mjd),
        dmx_epoch_max: data.dmx.last().map(|point| point.epoch_mjd),
        dmx_mean: mean_of_pairs(&dmx_series),
        dmx_std: stddev_of_pairs(&dmx_series),
        dmx_slope_per_day: linear_slope(&dmx_series),
        wb_dm_count: data.wideband_dm.len(),
        wb_dm_epoch_min: data.wideband_dm.first().map(|point| point.mjd),
        wb_dm_epoch_max: data.wideband_dm.last().map(|point| point.mjd),
        wb_dm_mean: mean_of_pairs(&wb_dm_series),
        wb_dm_std: stddev_of_pairs(&wb_dm_series),
        wb_dm_slope_per_day: linear_slope(&wb_dm_series),
        avg_residual_count: data.avg_residuals.len(),
        avg_residual_rms_us: avg_res_stats.rms(),
        avg_white_rms_us: avg_white_stats.rms(),
        avg_uncertainty_mean_us: avg_unc_stats.mean_value(),
        avg_abs_res_vs_unc_pearson: pearson_correlation(&abs_residuals, &uncertainties),
        avg_abs_res_vs_unc_spearman: spearman_correlation(&abs_residuals, &uncertainties),
        avg_abs_res_vs_dmx_pairs: dmx_pairs.len(),
        avg_abs_res_vs_dmx_pearson: pairwise_abs_residual_correlation(
            &dmx_pairs,
            pearson_correlation,
        ),
        avg_abs_res_vs_dmx_spearman: pairwise_abs_residual_correlation(
            &dmx_pairs,
            spearman_correlation,
        ),
        avg_abs_res_vs_wb_dm_pairs: wb_dm_pairs.len(),
        avg_abs_res_vs_wb_dm_pearson: pairwise_abs_residual_correlation(
            &wb_dm_pairs,
            pearson_correlation,
        ),
        avg_abs_res_vs_wb_dm_spearman: pairwise_abs_residual_correlation(
            &wb_dm_pairs,
            spearman_correlation,
        ),
    };

    Ok(PreparedPulsar {
        pulsar: data.pulsar.clone(),
        sky_vector,
        raw_bins: bin_avg_residuals(&data.avg_residuals, args.pair_bin_days, false),
        white_bins: bin_avg_residuals(&data.avg_residuals, args.pair_bin_days, true),
        row,
    })
}

fn preferred_metadata(data: &PulsarTimingData) -> &TimingModelMetadata {
    if data.wideband_par.elong_deg.is_some()
        || data.wideband_par.elat_deg.is_some()
        || data.wideband_par.raj.is_some()
        || data.wideband_par.decj.is_some()
    {
        &data.wideband_par
    } else {
        &data.narrowband_par
    }
}

fn build_pair_rows(prepared: &[PreparedPulsar], args: &Args) -> Vec<PairAuditRow> {
    let mut rows = Vec::new();
    for left_index in 0..prepared.len() {
        for right_index in (left_index + 1)..prepared.len() {
            let left = &prepared[left_index];
            let right = &prepared[right_index];
            let (Some(left_vec), Some(right_vec)) = (left.sky_vector, right.sky_vector) else {
                continue;
            };
            let overlap = overlapping_values(&left.raw_bins, &right.raw_bins);
            if overlap.0.len() < args.min_pair_overlap_bins {
                continue;
            }
            let Some(avg_residual_pearson) = pearson_correlation(&overlap.0, &overlap.1) else {
                continue;
            };
            let white_overlap = overlapping_values(&left.white_bins, &right.white_bins);
            let avg_white_residual_pearson =
                pearson_correlation(&white_overlap.0, &white_overlap.1);
            let separation_rad = angular_separation(left_vec, right_vec);
            rows.push(PairAuditRow {
                pulsar_a: left.pulsar.clone(),
                pulsar_b: right.pulsar.clone(),
                separation_deg: separation_rad.to_degrees(),
                hellings_downs: hellings_downs(separation_rad),
                overlap_bins: overlap.0.len(),
                avg_residual_pearson,
                avg_white_residual_pearson,
            });
        }
    }
    rows.sort_by(|a, b| {
        b.overlap_bins.cmp(&a.overlap_bins).then_with(|| {
            b.avg_residual_pearson
                .abs()
                .total_cmp(&a.avg_residual_pearson.abs())
        })
    });
    rows
}

fn match_avg_residuals_to_dmx(
    residuals: &[ResidualPoint],
    dmx_points: &[DmxPoint],
) -> Vec<(f64, f64)> {
    let mut pairs = Vec::new();
    for residual in residuals {
        let matched =
            dmx_points.iter().find(
                |point| match (point.window_start_mjd, point.window_end_mjd) {
                    (Some(start), Some(end)) => residual.mjd >= start && residual.mjd <= end,
                    _ => false,
                },
            );
        if let Some(point) = matched {
            pairs.push((residual.residual_us.abs(), point.dmx_value));
        }
    }
    pairs
}

fn match_avg_residuals_to_wb_dm(
    residuals: &[ResidualPoint],
    wb_dm_points: &[WidebandDmPoint],
    tolerance_days: f64,
) -> Vec<(f64, f64)> {
    let mut pairs = Vec::new();
    for residual in residuals {
        let nearest = wb_dm_points
            .iter()
            .map(|point| ((point.mjd - residual.mjd).abs(), point))
            .filter(|(delta, _)| *delta <= tolerance_days)
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, point)| point);
        if let Some(point) = nearest {
            pairs.push((residual.residual_us.abs(), point.pp_dm));
        }
    }
    pairs
}

fn pairwise_abs_residual_correlation(
    pairs: &[(f64, f64)],
    correlation: fn(&[f64], &[f64]) -> Option<f64>,
) -> Option<f64> {
    if pairs.len() < 3 {
        return None;
    }
    let left = pairs.iter().map(|pair| pair.0).collect::<Vec<_>>();
    let right = pairs.iter().map(|pair| pair.1).collect::<Vec<_>>();
    correlation(&left, &right)
}

fn bin_avg_residuals(
    residuals: &[ResidualPoint],
    bin_days: f64,
    whitened: bool,
) -> BTreeMap<i64, f64> {
    let mut accumulators: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for point in residuals {
        let Some(value) = (if whitened {
            point.white_residual_us
        } else {
            Some(point.residual_us)
        }) else {
            continue;
        };
        let bin = (point.mjd / bin_days).floor() as i64;
        let weight = point
            .uncertainty_us
            .filter(|unc| *unc > 0.0)
            .map(|unc| 1.0 / (unc * unc))
            .unwrap_or(1.0);
        let entry = accumulators.entry(bin).or_insert((0.0, 0.0));
        entry.0 += weight * value;
        entry.1 += weight;
    }
    accumulators
        .into_iter()
        .filter_map(|(bin, (weighted_sum, weight_sum))| {
            (weight_sum > 0.0).then_some((bin, weighted_sum / weight_sum))
        })
        .collect()
}

fn overlapping_values(
    left: &BTreeMap<i64, f64>,
    right: &BTreeMap<i64, f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (bin, left_value) in left {
        if let Some(right_value) = right.get(bin) {
            xs.push(*left_value);
            ys.push(*right_value);
        }
    }
    (xs, ys)
}

fn sky_vector(meta: &TimingModelMetadata) -> Option<[f64; 3]> {
    if let (Some(elong_deg), Some(elat_deg)) = (meta.elong_deg, meta.elat_deg) {
        return Some(ecliptic_to_equatorial_vector(
            elong_deg.to_radians(),
            elat_deg.to_radians(),
        ));
    }
    let raj = meta.raj.as_deref()?;
    let decj = meta.decj.as_deref()?;
    let ra_rad = parse_hms_radians(raj)?;
    let dec_rad = parse_dms_radians(decj)?;
    Some([
        dec_rad.cos() * ra_rad.cos(),
        dec_rad.cos() * ra_rad.sin(),
        dec_rad.sin(),
    ])
}

fn ecliptic_to_equatorial_vector(longitude_rad: f64, latitude_rad: f64) -> [f64; 3] {
    let epsilon = 23.4392911_f64.to_radians();
    let x_ecl = latitude_rad.cos() * longitude_rad.cos();
    let y_ecl = latitude_rad.cos() * longitude_rad.sin();
    let z_ecl = latitude_rad.sin();
    [
        x_ecl,
        y_ecl * epsilon.cos() - z_ecl * epsilon.sin(),
        y_ecl * epsilon.sin() + z_ecl * epsilon.cos(),
    ]
}

fn parse_hms_radians(value: &str) -> Option<f64> {
    let fields = value.split(':').collect::<Vec<_>>();
    if fields.len() != 3 {
        return None;
    }
    let hours = fields[0].parse::<f64>().ok()?;
    let minutes = fields[1].parse::<f64>().ok()?;
    let seconds = fields[2].parse::<f64>().ok()?;
    let hours_total = hours + minutes / 60.0 + seconds / 3600.0;
    Some((hours_total * 15.0).to_radians())
}

fn parse_dms_radians(value: &str) -> Option<f64> {
    let fields = value.split(':').collect::<Vec<_>>();
    if fields.len() != 3 {
        return None;
    }
    let degrees = fields[0].parse::<f64>().ok()?;
    let minutes = fields[1].parse::<f64>().ok()?;
    let seconds = fields[2].parse::<f64>().ok()?;
    let sign = if degrees < 0.0 { -1.0 } else { 1.0 };
    let magnitude = degrees.abs() + minutes / 60.0 + seconds / 3600.0;
    Some((sign * magnitude).to_radians())
}

fn angular_separation(left: [f64; 3], right: [f64; 3]) -> f64 {
    let dot = (left[0] * right[0] + left[1] * right[1] + left[2] * right[2]).clamp(-1.0, 1.0);
    dot.acos()
}

fn hellings_downs(separation_rad: f64) -> f64 {
    let x = (1.0 - separation_rad.cos()) / 2.0;
    if x <= 0.0 {
        0.5
    } else {
        1.5 * x * x.ln() - 0.25 * x + 0.5
    }
}

fn mean_of_pairs(series: &[(f64, f64)]) -> Option<f64> {
    (!series.is_empty())
        .then(|| series.iter().map(|(_, value)| *value).sum::<f64>() / series.len() as f64)
}

fn stddev_of_pairs(series: &[(f64, f64)]) -> Option<f64> {
    if series.len() < 2 {
        return None;
    }
    let mean = mean_of_pairs(series)?;
    let variance = series
        .iter()
        .map(|(_, value)| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (series.len() as f64 - 1.0);
    Some(variance.sqrt())
}

fn linear_slope(series: &[(f64, f64)]) -> Option<f64> {
    if series.len() < 2 {
        return None;
    }
    let x_mean = series.iter().map(|(x, _)| *x).sum::<f64>() / series.len() as f64;
    let y_mean = series.iter().map(|(_, y)| *y).sum::<f64>() / series.len() as f64;
    let numerator = series
        .iter()
        .map(|(x, y)| (*x - x_mean) * (*y - y_mean))
        .sum::<f64>();
    let denominator = series
        .iter()
        .map(|(x, _)| {
            let centered = *x - x_mean;
            centered * centered
        })
        .sum::<f64>();
    (denominator > 0.0).then_some(numerator / denominator)
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut numerator = 0.0_f64;
    let mut x_var = 0.0_f64;
    let mut y_var = 0.0_f64;
    for (x, y) in xs.iter().zip(ys) {
        let dx = *x - x_mean;
        let dy = *y - y_mean;
        numerator += dx * dy;
        x_var += dx * dx;
        y_var += dy * dy;
    }
    let denominator = (x_var * y_var).sqrt();
    (denominator > 0.0).then_some(numerator / denominator)
}

fn spearman_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let rank_x = rank_values(xs)?;
    let rank_y = rank_values(ys)?;
    pearson_correlation(&rank_x, &rank_y)
}

fn rank_values(values: &[f64]) -> Option<Vec<f64>> {
    if values.len() < 2 {
        return None;
    }
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f64)>>();
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut ranks = vec![0.0_f64; values.len()];
    let mut index = 0usize;
    while index < indexed.len() {
        let mut upper = index + 1;
        while upper < indexed.len() && indexed[upper].1 == indexed[index].1 {
            upper += 1;
        }
        let rank = (index + upper - 1) as f64 / 2.0 + 1.0;
        for tied in &indexed[index..upper] {
            ranks[tied.0] = rank;
        }
        index = upper;
    }
    Some(ranks)
}

fn write_pulsar_csv<'a>(path: &Path, rows: impl Iterator<Item = &'a PulsarAuditRow>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut writer = Writer::from_path(path).with_context(|| format!("open {}", path.display()))?;
    writer.write_record([
        "pulsar",
        "elong_deg",
        "elat_deg",
        "px_mas",
        "dmx_count",
        "dmx_epoch_min",
        "dmx_epoch_max",
        "dmx_mean",
        "dmx_std",
        "dmx_slope_per_day",
        "wb_dm_count",
        "wb_dm_epoch_min",
        "wb_dm_epoch_max",
        "wb_dm_mean",
        "wb_dm_std",
        "wb_dm_slope_per_day",
        "avg_residual_count",
        "avg_residual_rms_us",
        "avg_white_rms_us",
        "avg_uncertainty_mean_us",
        "avg_abs_res_vs_unc_pearson",
        "avg_abs_res_vs_unc_spearman",
        "avg_abs_res_vs_dmx_pairs",
        "avg_abs_res_vs_dmx_pearson",
        "avg_abs_res_vs_dmx_spearman",
        "avg_abs_res_vs_wb_dm_pairs",
        "avg_abs_res_vs_wb_dm_pearson",
        "avg_abs_res_vs_wb_dm_spearman",
    ])?;
    for row in rows {
        writer.write_record([
            row.pulsar.clone(),
            format_opt(row.elong_deg),
            format_opt(row.elat_deg),
            format_opt(row.px_mas),
            row.dmx_count.to_string(),
            format_opt(row.dmx_epoch_min),
            format_opt(row.dmx_epoch_max),
            format_opt(row.dmx_mean),
            format_opt(row.dmx_std),
            format_opt(row.dmx_slope_per_day),
            row.wb_dm_count.to_string(),
            format_opt(row.wb_dm_epoch_min),
            format_opt(row.wb_dm_epoch_max),
            format_opt(row.wb_dm_mean),
            format_opt(row.wb_dm_std),
            format_opt(row.wb_dm_slope_per_day),
            row.avg_residual_count.to_string(),
            format_opt(row.avg_residual_rms_us),
            format_opt(row.avg_white_rms_us),
            format_opt(row.avg_uncertainty_mean_us),
            format_opt(row.avg_abs_res_vs_unc_pearson),
            format_opt(row.avg_abs_res_vs_unc_spearman),
            row.avg_abs_res_vs_dmx_pairs.to_string(),
            format_opt(row.avg_abs_res_vs_dmx_pearson),
            format_opt(row.avg_abs_res_vs_dmx_spearman),
            row.avg_abs_res_vs_wb_dm_pairs.to_string(),
            format_opt(row.avg_abs_res_vs_wb_dm_pearson),
            format_opt(row.avg_abs_res_vs_wb_dm_spearman),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_pair_csv<'a>(path: &Path, rows: impl Iterator<Item = &'a PairAuditRow>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut writer = Writer::from_path(path).with_context(|| format!("open {}", path.display()))?;
    writer.write_record([
        "pulsar_a",
        "pulsar_b",
        "separation_deg",
        "hellings_downs",
        "overlap_bins",
        "avg_residual_pearson",
        "avg_white_residual_pearson",
    ])?;
    for row in rows {
        writer.write_record([
            row.pulsar_a.clone(),
            row.pulsar_b.clone(),
            format!("{:.12}", row.separation_deg),
            format!("{:.12}", row.hellings_downs),
            row.overlap_bins.to_string(),
            format!("{:.12}", row.avg_residual_pearson),
            format_opt(row.avg_white_residual_pearson),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_report(
    path: &Path,
    root: &Path,
    pulsar_csv_out: &Path,
    pair_csv_out: &Path,
    prepared: &[PreparedPulsar],
    pair_rows: &[PairAuditRow],
    args: &Args,
) -> Result<()> {
    let max_dmx = prepared
        .iter()
        .filter_map(|entry| entry.row.dmx_std.map(|value| (entry, value)))
        .max_by(|left, right| left.1.total_cmp(&right.1));
    let max_wb_dm = prepared
        .iter()
        .filter_map(|entry| entry.row.wb_dm_std.map(|value| (entry, value)))
        .max_by(|left, right| left.1.total_cmp(&right.1));
    let max_residual = prepared
        .iter()
        .filter_map(|entry| entry.row.avg_residual_rms_us.map(|value| (entry, value)))
        .max_by(|left, right| left.1.total_cmp(&right.1));
    let max_parallax = prepared
        .iter()
        .filter_map(|entry| entry.row.px_mas.map(|value| (entry, value)))
        .max_by(|left, right| left.1.total_cmp(&right.1));
    let top_pair_overlap = pair_rows.first();
    let top_pair_corr = pair_rows.iter().max_by(|left, right| {
        left.avg_residual_pearson
            .abs()
            .total_cmp(&right.avg_residual_pearson.abs())
    });

    let mut out = String::new();
    let _ = writeln!(out, "root = {:?}", root.display().to_string());
    let _ = writeln!(
        out,
        "pulsar_csv = {:?}",
        pulsar_csv_out.display().to_string()
    );
    let _ = writeln!(out, "pair_csv = {:?}", pair_csv_out.display().to_string());
    let _ = writeln!(out, "pulsar_count = {}", prepared.len());
    let _ = writeln!(out, "pair_count = {}", pair_rows.len());
    let _ = writeln!(
        out,
        "match_tolerance_days = {:.3}",
        args.match_tolerance_days
    );
    let _ = writeln!(out, "pair_bin_days = {:.3}", args.pair_bin_days);
    let _ = writeln!(
        out,
        "min_pair_overlap_bins = {}",
        args.min_pair_overlap_bins
    );
    let _ = writeln!(
        out,
        "sky_ready_count = {}",
        prepared
            .iter()
            .filter(|entry| entry.sky_vector.is_some())
            .count()
    );
    let _ = writeln!(
        out,
        "dmx_ready_count = {}",
        prepared
            .iter()
            .filter(|entry| entry.row.dmx_count > 0)
            .count()
    );
    let _ = writeln!(
        out,
        "wideband_dm_ready_count = {}",
        prepared
            .iter()
            .filter(|entry| entry.row.wb_dm_count > 0)
            .count()
    );
    let _ = writeln!(
        out,
        "avg_residual_ready_count = {}",
        prepared
            .iter()
            .filter(|entry| entry.row.avg_residual_count > 0)
            .count()
    );
    let _ = writeln!(
        out,
        "residual_uncertainty_correlation_ready_count = {}",
        prepared
            .iter()
            .filter(|entry| entry.row.avg_abs_res_vs_unc_pearson.is_some())
            .count()
    );
    let _ = writeln!(
        out,
        "residual_dmx_correlation_ready_count = {}",
        prepared
            .iter()
            .filter(|entry| entry.row.avg_abs_res_vs_dmx_pearson.is_some())
            .count()
    );
    let _ = writeln!(
        out,
        "residual_wb_dm_correlation_ready_count = {}",
        prepared
            .iter()
            .filter(|entry| entry.row.avg_abs_res_vs_wb_dm_pearson.is_some())
            .count()
    );
    if let Some((entry, value)) = max_dmx {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_dmx_std]");
        let _ = writeln!(out, "name = {:?}", entry.pulsar);
        let _ = writeln!(out, "dmx_std = {:.12}", value);
    }
    if let Some((entry, value)) = max_wb_dm {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_wideband_dm_std]");
        let _ = writeln!(out, "name = {:?}", entry.pulsar);
        let _ = writeln!(out, "pp_dm_std = {:.12}", value);
    }
    if let Some((entry, value)) = max_residual {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_avg_residual_rms]");
        let _ = writeln!(out, "name = {:?}", entry.pulsar);
        let _ = writeln!(out, "rms_us = {:.12}", value);
    }
    if let Some((entry, value)) = max_parallax {
        let _ = writeln!(out);
        let _ = writeln!(out, "[max_parallax_mas]");
        let _ = writeln!(out, "name = {:?}", entry.pulsar);
        let _ = writeln!(out, "px_mas = {:.12}", value);
    }
    if let Some(pair) = top_pair_overlap {
        let _ = writeln!(out);
        let _ = writeln!(out, "[top_pair_overlap]");
        let _ = writeln!(out, "pulsar_a = {:?}", pair.pulsar_a);
        let _ = writeln!(out, "pulsar_b = {:?}", pair.pulsar_b);
        let _ = writeln!(out, "overlap_bins = {}", pair.overlap_bins);
        let _ = writeln!(out, "separation_deg = {:.12}", pair.separation_deg);
        let _ = writeln!(out, "hellings_downs = {:.12}", pair.hellings_downs);
        let _ = writeln!(
            out,
            "avg_residual_pearson = {:.12}",
            pair.avg_residual_pearson
        );
    }
    if let Some(pair) = top_pair_corr {
        let _ = writeln!(out);
        let _ = writeln!(out, "[top_abs_pair_correlation]");
        let _ = writeln!(out, "pulsar_a = {:?}", pair.pulsar_a);
        let _ = writeln!(out, "pulsar_b = {:?}", pair.pulsar_b);
        let _ = writeln!(out, "overlap_bins = {}", pair.overlap_bins);
        let _ = writeln!(out, "separation_deg = {:.12}", pair.separation_deg);
        let _ = writeln!(out, "hellings_downs = {:.12}", pair.hellings_downs);
        let _ = writeln!(
            out,
            "avg_residual_pearson = {:.12}",
            pair.avg_residual_pearson
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "[timing_model_refit_scope]");
    let _ = writeln!(out, "status = {:?}", "scoped_only_not_recomputed_in_repo");
    let _ = writeln!(
        out,
        "note = {:?}",
        "This audit consumes released post-fit products, DMX parses, wideband TOA DM estimates, and timing-model metadata. A true timing-model refit still needs a Rust-native timing engine that re-evaluates the timing model against TOAs with clock and solar-system ephemeris handling rather than reusing release residuals."
    );

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn format_opt(value: Option<f64>) -> String {
    value
        .map(|number| format!("{number:.12}"))
        .unwrap_or_default()
}

fn update_min(slot: &mut Option<f64>, value: f64) {
    match slot {
        Some(current) if value >= *current => {}
        _ => *slot = Some(value),
    }
}

fn update_max(slot: &mut Option<f64>, value: f64) {
    match slot {
        Some(current) if value <= *current => {}
        _ => *slot = Some(value),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        angular_separation, ecliptic_to_equatorial_vector, hellings_downs, parse_dms_radians,
        parse_hms_radians,
    };

    #[test]
    fn hellings_downs_zero_separation_is_half() {
        let value = hellings_downs(0.0);
        assert!((value - 0.5).abs() < 1.0e-12, "value={value}");
    }

    #[test]
    fn right_angle_separation_gives_negative_hd_value() {
        let left = ecliptic_to_equatorial_vector(0.0, 0.0);
        let right = ecliptic_to_equatorial_vector(90.0_f64.to_radians(), 0.0);
        let value = hellings_downs(angular_separation(left, right));
        assert!(value < 0.0, "value={value}");
    }

    #[test]
    fn parses_sexagesimal_angles() {
        let ra = parse_hms_radians("22:14:38.846").expect("ra");
        let dec = parse_dms_radians("+30:00:38.20").expect("dec");
        assert!(ra.is_finite());
        assert!(dec.is_finite());
    }
}
