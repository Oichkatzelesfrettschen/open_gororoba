use anyhow::Result;
use clap::Parser;
use gororoba_cli_physics::anomaly_residual::{
    BestFitSummary, FitGridRecord, FlybyResidualRecord, PioneerBenchmarkContextRecord,
    ensure_parent_dir, read_csv_records, write_csv_records,
};
use gr_core::{Schwarzschild, fractal_metric::{QtensorFractalMetric, fractal_flyby_prediction, pioneer_anomaly_prediction}};
use serde_json::to_string_pretty;
use std::path::PathBuf;

const AU_KM: f64 = 149_597_870.7;

#[derive(Debug, Parser)]
#[command(name = "fractal-metric-fit")]
#[command(about = "Fit plain and Q-tensor fractal metrics against governed anomaly benchmarks")]
struct Args {
    #[arg(
        long,
        default_value = "data/output/anomaly/flyby/flyby_observed_vs_modeled.csv"
    )]
    flyby_input: PathBuf,
    #[arg(
        long,
        default_value = "data/output/anomaly/pioneer/pioneer_benchmark_vs_environment.csv"
    )]
    pioneer_input: PathBuf,
    #[arg(long, default_value = "data/output/anomaly/fractal_metric")]
    out_dir: PathBuf,
}

#[derive(Clone, Copy)]
struct FitParams {
    d_f: f64,
    r_disclination_km: f64,
    s_bulk: f64,
}

fn sign_match(predicted: f64, observed: f64) -> bool {
    if observed.abs() < 1.0e-12 {
        predicted.abs() < 0.1
    } else {
        predicted.signum() == observed.signum()
    }
}

fn relative_error(predicted: f64, observed: f64) -> f64 {
    let denom = observed.abs().max(1.0e-12);
    (predicted - observed).abs() / denom
}

fn qtensor_pioneer_prediction(params: FitParams, r_km: f64, v_km_s: f64) -> f64 {
    let base = pioneer_anomaly_prediction(params.d_f, r_km, v_km_s);
    let qtensor = QtensorFractalMetric::new(
        Schwarzschild::new(1.0),
        params.d_f,
        1.0,
        params.r_disclination_km,
        params.s_bulk,
    );
    (base + qtensor.disclination_acceleration(r_km, v_km_s)) * 1.0e3
}

fn qtensor_flyby_prediction(params: FitParams, perigee_km: f64, v_inf_km_s: f64) -> f64 {
    let base = fractal_flyby_prediction(params.d_f, perigee_km, v_inf_km_s).delta_v_mm_s;
    let qtensor = QtensorFractalMetric::new(
        Schwarzschild::new(1.0),
        params.d_f,
        1.0,
        params.r_disclination_km,
        params.s_bulk,
    );
    let delta_v_q_mm_s =
        qtensor.disclination_acceleration(perigee_km, v_inf_km_s) * (perigee_km / v_inf_km_s) * 1.0e6;
    base + delta_v_q_mm_s
}

fn main() -> Result<()> {
    let args = Args::parse();
    let flybys = read_csv_records::<FlybyResidualRecord>(&args.flyby_input)?;
    let pioneers = read_csv_records::<PioneerBenchmarkContextRecord>(&args.pioneer_input)?;

    let d_f_grid = [2.5, 2.6, 2.7, 2.8, 2.9];
    let s_bulk_grid = [0.0, 0.05, 0.1, 0.3, 0.5];
    let r_d_grid = [1.0e5, 1.0e7, AU_KM, 40.0 * AU_KM];

    let mut fit_rows = Vec::new();
    let mut best: Option<(BestFitSummary, f64)> = None;

    for &d_f in &d_f_grid {
        for &s_bulk in &s_bulk_grid {
            let r_values: &[f64] = if s_bulk == 0.0 { &[0.0] } else { &r_d_grid };
            for &r_disclination_km in r_values {
                let params = FitParams {
                    d_f,
                    r_disclination_km,
                    s_bulk,
                };
                let lane = if s_bulk == 0.0 { "plain" } else { "qtensor" };

                let mut sign_matches = 0usize;
                let mut total = 0usize;
                let mut rel_errors = Vec::new();
                let mut max_relative_error = 0.0f64;

                for row in &flybys {
                    let (perigee_km, v_inf_km_s) = match row.name.as_str() {
                        "Galileo-I (1990-12-08)" => (6371.0 + 960.0, 8.949),
                        "NEAR (1998-01-23)" => (6371.0 + 539.0, 6.851),
                        "Cassini (1999-08-18)" => (6371.0 + 1175.0, 16.01),
                        "Rosetta-I (2005-03-04)" => (6371.0 + 1956.0, 3.863),
                        "MESSENGER (2005-08-02)" => (6371.0 + 2347.0, 4.056),
                        "Juno (2013-10-09)" => (6371.0 + 559.0, 9.897),
                        _ => continue,
                    };
                    let predicted = if s_bulk == 0.0 {
                        fractal_flyby_prediction(params.d_f, perigee_km, v_inf_km_s).delta_v_mm_s
                    } else {
                        qtensor_flyby_prediction(params, perigee_km, v_inf_km_s)
                    };
                    let rel = relative_error(predicted, row.observed_dv_mm_s);
                    let sign_ok = sign_match(predicted, row.observed_dv_mm_s);
                    sign_matches += usize::from(sign_ok);
                    total += 1;
                    rel_errors.push(rel);
                    max_relative_error = max_relative_error.max(rel);
                    fit_rows.push(FitGridRecord {
                        lane: lane.to_string(),
                        subject: row.name.clone(),
                        d_f: params.d_f,
                        r_disclination_km: params.r_disclination_km,
                        s_bulk: params.s_bulk,
                        observed_value: row.observed_dv_mm_s,
                        predicted_value: predicted,
                        absolute_error: (predicted - row.observed_dv_mm_s).abs(),
                        relative_error: rel,
                        sign_match: sign_ok,
                    });
                }

                for row in &pioneers {
                    let (Some(r_au), Some(v_km_s)) = (row.sampled_r_au, row.sampled_speed_km_s) else {
                        continue;
                    };
                    let predicted = if s_bulk == 0.0 {
                        pioneer_anomaly_prediction(params.d_f, r_au * AU_KM, v_km_s) * 1.0e3
                    } else {
                        qtensor_pioneer_prediction(params, r_au * AU_KM, v_km_s)
                    };
                    let rel = relative_error(predicted, row.observed_anomalous_accel_m_s2);
                    let sign_ok = sign_match(predicted, row.observed_anomalous_accel_m_s2);
                    sign_matches += usize::from(sign_ok);
                    total += 1;
                    rel_errors.push(rel);
                    max_relative_error = max_relative_error.max(rel);
                    fit_rows.push(FitGridRecord {
                        lane: lane.to_string(),
                        subject: row.spacecraft.clone(),
                        d_f: params.d_f,
                        r_disclination_km: params.r_disclination_km,
                        s_bulk: params.s_bulk,
                        observed_value: row.observed_anomalous_accel_m_s2,
                        predicted_value: predicted,
                        absolute_error: (predicted - row.observed_anomalous_accel_m_s2).abs(),
                        relative_error: rel,
                        sign_match: sign_ok,
                    });
                }

                if total == 0 {
                    continue;
                }
                let mean_relative_error =
                    rel_errors.iter().sum::<f64>() / rel_errors.len() as f64;
                let verdict = if sign_matches == total && max_relative_error <= 0.5 {
                    "candidate_support"
                } else {
                    "refuted_current_form"
                };
                let summary = BestFitSummary {
                    verdict: verdict.to_string(),
                    best_lane: lane.to_string(),
                    best_d_f: params.d_f,
                    best_r_disclination_km: params.r_disclination_km,
                    best_s_bulk: params.s_bulk,
                    flyby_sign_matches: sign_matches,
                    flyby_total: total,
                    mean_relative_error,
                    max_relative_error,
                    notes: "Batch-1 benchmark fit: no thermal recoil or full engineering-force model".to_string(),
                };
                let score = mean_relative_error + (total - sign_matches) as f64;
                match &best {
                    Some((_, best_score)) if score >= *best_score => {}
                    _ => best = Some((summary, score)),
                }
            }
        }
    }

    let fit_grid_path = args.out_dir.join("fit_grid.csv");
    let best_fit_path = args.out_dir.join("best_fit.json");
    write_csv_records(&fit_grid_path, &fit_rows)?;
    ensure_parent_dir(&best_fit_path)?;
    let (summary, _) = best.expect("non-empty fit grid");
    std::fs::write(&best_fit_path, to_string_pretty(&summary)? + "\n")?;
    println!("artifact: {}", fit_grid_path.display());
    println!("artifact: {}", best_fit_path.display());
    println!("verdict: {}", summary.verdict);
    Ok(())
}
