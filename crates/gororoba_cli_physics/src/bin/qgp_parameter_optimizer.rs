use clap::Parser;
use csv::ReaderBuilder;
use std::{error::Error, fs};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/lhc_2025/alice/pb_pb_5.36tev_polarization.csv"
    )]
    alice_data: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let alice_records = load_alice_data(&args.alice_data)?;

    println!("Optimizing QGP-CD Parameters against ALICE 2025 Data...");

    let mut best_k_vort = 0.0;
    let mut min_error = f64::INFINITY;

    for i in 0..1000 {
        let k_vort = i as f64 * 0.0001;
        let mut total_sq_err = 0.0;
        for rec in &alice_records {
            let centrality_mid = (rec.centrality_min + rec.centrality_max) as f64 / 2.0;
            let omega_mag = 0.05 + 0.002 * centrality_mid;
            let p_pred = (k_vort * omega_mag * 0.5) / 0.15;
            let p_alice = rec.p_h_percent / 100.0;
            total_sq_err += (p_alice - p_pred).powi(2);
        }

        if total_sq_err < min_error {
            min_error = total_sq_err;
            best_k_vort = k_vort;
        }
    }

    println!("Optimization Complete.");
    println!("Best k_vorticity: {}", best_k_vort);
    let rms_err = (min_error / alice_records.len() as f64).sqrt();
    println!("Minimum RMS Error: {}", rms_err);

    let report = format!(
        "model = \"QGPImbalanceBridge\"
target_dataset = \"ALICE 2025 Pb-Pb 5.36 TeV\"
best_k_vorticity = {}
final_rms_error = {}
",
        best_k_vort, rms_err
    );

    fs::write("reports/QGP_Parameter_Optimization.toml", report)?;
    Ok(())
}

struct AliceRecord {
    centrality_min: i32,
    centrality_max: i32,
    p_h_percent: f64,
}
fn load_alice_data(path: &str) -> Result<Vec<AliceRecord>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path(path)?;
    let mut results = Vec::new();
    for result in rdr.records() {
        let record = result?;
        results.push(AliceRecord {
            centrality_min: record[0].parse()?,
            centrality_max: record[1].parse()?,
            p_h_percent: record[2].parse()?,
        });
    }
    Ok(results)
}
