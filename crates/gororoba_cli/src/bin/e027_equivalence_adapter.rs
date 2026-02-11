use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use csv::Writer;
use serde::Serialize;
use toml::Value;

#[derive(Parser, Debug)]
#[command(
    name = "e027-equivalence-adapter",
    about = "Extract E-027 TOML artifacts into proxy curves for homotopy-viscosity equivalence."
)]
struct Args {
    #[arg(long, default_value = "data")]
    data_root: PathBuf,

    #[arg(long, default_value = "data/equivalence/e027")]
    output_dir: PathBuf,

    #[arg(long, default_value_t = true)]
    include_forcing_runs: bool,

    #[arg(long, default_value_t = 0.375)]
    vacuum_attractor: f64,
}

#[derive(Debug, Serialize)]
struct BundleMetadata {
    data_root: String,
    output_dir: String,
    include_forcing_runs: bool,
    vacuum_attractor: f64,
    point_count: usize,
}

#[derive(Debug, Serialize)]
struct PointRecord {
    index: usize,
    source: String,
    label: String,
    lambda: f64,
    nu_base: f64,
    frustration_mean: f64,
    viscosity_proxy: f64,
    loss_proxy: f64,
}

#[derive(Debug, Serialize)]
struct CurveBundle {
    metadata: BundleMetadata,
    points: Vec<PointRecord>,
}

fn get_f64(table: &Value, path: &[&str]) -> Option<f64> {
    let mut current = table;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_float()
}

fn compute_viscosity_proxy(nu_base: f64, lambda: f64, frustration_mean: f64, vacuum_attractor: f64) -> f64 {
    let delta = frustration_mean - vacuum_attractor;
    nu_base * (-(lambda * delta * delta)).exp()
}

fn parse_single_run(path: &Path, vacuum_attractor: f64) -> Result<PointRecord, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let value: Value = raw
        .parse::<Value>()
        .map_err(|e| format!("failed to parse TOML {}: {e}", path.display()))?;

    let lambda = get_f64(&value, &["metadata", "lambda"])
        .ok_or_else(|| format!("missing metadata.lambda in {}", path.display()))?;
    let nu_base = get_f64(&value, &["metadata", "nu_base"])
        .ok_or_else(|| format!("missing metadata.nu_base in {}", path.display()))?;
    let frustration_mean = get_f64(&value, &["correlation", "mean_frustration_channels"]).ok_or_else(|| {
        format!(
            "missing correlation.mean_frustration_channels in {}",
            path.display()
        )
    })?;

    let viscosity_proxy = compute_viscosity_proxy(nu_base, lambda, frustration_mean, vacuum_attractor);
    let loss_proxy = (frustration_mean - vacuum_attractor).abs();

    let label = path
        .parent()
        .and_then(Path::file_name)
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    Ok(PointRecord {
        index: 0,
        source: path.display().to_string(),
        label,
        lambda,
        nu_base,
        frustration_mean,
        viscosity_proxy,
        loss_proxy,
    })
}

fn parse_forcing_runs(path: &Path, vacuum_attractor: f64) -> Result<Vec<PointRecord>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let value: Value = raw
        .parse::<Value>()
        .map_err(|e| format!("failed to parse TOML {}: {e}", path.display()))?;

    let runs = value
        .get("experiment")
        .and_then(|v| v.get("run"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing [[experiment.run]] in {}", path.display()))?;

    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
        let name = run
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("unnamed_run")
            .to_string();
        let lambda = run
            .get("lambda")
            .and_then(Value::as_float)
            .ok_or_else(|| format!("run {name}: missing lambda in {}", path.display()))?;
        let nu_base = run
            .get("nu_base")
            .and_then(Value::as_float)
            .ok_or_else(|| format!("run {name}: missing nu_base in {}", path.display()))?;
        let frustration_mean = get_f64(run, &["results", "frustration_mean"])
            .ok_or_else(|| format!("run {name}: missing results.frustration_mean in {}", path.display()))?;
        let viscosity_proxy = get_f64(run, &["results", "viscosity_mean"])
            .unwrap_or_else(|| compute_viscosity_proxy(nu_base, lambda, frustration_mean, vacuum_attractor));
        let loss_proxy = (frustration_mean - vacuum_attractor).abs();

        out.push(PointRecord {
            index: 0,
            source: path.display().to_string(),
            label: format!("forcing:{name}"),
            lambda,
            nu_base,
            frustration_mean,
            viscosity_proxy,
            loss_proxy,
        });
    }
    Ok(out)
}

fn write_curves(output_dir: &Path, points: &[PointRecord]) -> Result<(), String> {
    fs::create_dir_all(output_dir)
        .map_err(|e| format!("failed to create {}: {e}", output_dir.display()))?;
    let loss_path = output_dir.join("loss_curve.csv");
    let viscosity_path = output_dir.join("viscosity_curve.csv");

    let mut loss_wtr =
        Writer::from_path(&loss_path).map_err(|e| format!("failed to open {}: {e}", loss_path.display()))?;
    loss_wtr
        .write_record(["step", "loss"])
        .map_err(|e| format!("failed write header {}: {e}", loss_path.display()))?;

    let mut visc_wtr = Writer::from_path(&viscosity_path)
        .map_err(|e| format!("failed to open {}: {e}", viscosity_path.display()))?;
    visc_wtr
        .write_record(["step", "value"])
        .map_err(|e| format!("failed write header {}: {e}", viscosity_path.display()))?;

    for p in points {
        loss_wtr
            .write_record([p.index.to_string(), p.loss_proxy.to_string()])
            .map_err(|e| format!("failed writing {}: {e}", loss_path.display()))?;
        visc_wtr
            .write_record([p.index.to_string(), p.viscosity_proxy.to_string()])
            .map_err(|e| format!("failed writing {}: {e}", viscosity_path.display()))?;
    }

    loss_wtr
        .flush()
        .map_err(|e| format!("failed flushing {}: {e}", loss_path.display()))?;
    visc_wtr
        .flush()
        .map_err(|e| format!("failed flushing {}: {e}", viscosity_path.display()))?;
    Ok(())
}

fn write_bundle(output_dir: &Path, bundle: &CurveBundle) -> Result<(), String> {
    let path = output_dir.join("curve_bundle.toml");
    let text = toml::to_string_pretty(bundle).map_err(|e| format!("failed serializing TOML bundle: {e}"))?;
    fs::write(&path, text).map_err(|e| format!("failed writing {}: {e}", path.display()))?;
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let fixed_paths = [
        args.data_root.join("e027/e027_results.toml"),
        args.data_root.join("e027_no_forcing/e027_results.toml"),
        args.data_root.join("e027_forced/e027_results.toml"),
        args.data_root.join("e027_final/e027_results.toml"),
        args.data_root.join("e027_lambda200/e027_results.toml"),
    ];

    let mut points = Vec::new();
    for path in fixed_paths {
        if path.exists() {
            points.push(parse_single_run(&path, args.vacuum_attractor)?);
        }
    }

    let forcing_path = args.data_root.join("e027_forcing_results.toml");
    if args.include_forcing_runs && forcing_path.exists() {
        points.extend(parse_forcing_runs(&forcing_path, args.vacuum_attractor)?);
    }

    if points.len() < 3 {
        return Err(format!(
            "insufficient points extracted ({}). need at least 3.",
            points.len()
        ));
    }

    points.sort_by(|a, b| {
        a.lambda
            .partial_cmp(&b.lambda)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.label.cmp(&b.label))
    });

    for (idx, p) in points.iter_mut().enumerate() {
        p.index = idx;
    }

    write_curves(&args.output_dir, &points)?;
    let bundle = CurveBundle {
        metadata: BundleMetadata {
            data_root: args.data_root.display().to_string(),
            output_dir: args.output_dir.display().to_string(),
            include_forcing_runs: args.include_forcing_runs,
            vacuum_attractor: args.vacuum_attractor,
            point_count: points.len(),
        },
        points,
    };
    write_bundle(&args.output_dir, &bundle)?;

    println!(
        "Wrote {}/loss_curve.csv, {}/viscosity_curve.csv, and {}/curve_bundle.toml",
        args.output_dir.display(),
        args.output_dir.display(),
        args.output_dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compute_viscosity_proxy;

    #[test]
    fn viscosity_proxy_at_attractor_equals_nu_base() {
        let nu = compute_viscosity_proxy(0.333, 50.0, 0.375, 0.375);
        assert!((nu - 0.333).abs() < 1e-12);
    }
}
