use anyhow::Result;
use clap::Parser;
use std::f64::consts::PI;
use std::fs::File;
use std::path::PathBuf;
use glob::glob;
use hdf5_metno as hdf5;

use spectral_core::wavelet::continuous_wavelet_transform;

const GHOST_FREQ: f64 = 0.786151377757423;
const ALIASED_GHOST_FREQ: f64 = 1.0 - GHOST_FREQ; // 0.2138

#[derive(Parser, Debug)]
#[command(author, version, about = "Multiscale CWT Analysis for Ghost Detection")]
struct Args {
    /// Mode: wow, alice, swarm, or file
    #[arg(short, long, default_value = "wow")]
    mode: String,

    /// Input CSV/H5 path (if mode=file)
    #[arg(short, long)]
    input: Option<PathBuf>,
}

fn analyze_cwt(data: &[f64], fs: f64, title: &str) -> bool {
    println!("\n--- Multiscale CWT Analysis: {} ---", title);
    let n = data.len();
    if n == 0 { return false; }

    // Detrend: Subtract mean
    let mean = data.iter().sum::<f64>() / n as f64;
    let detrended: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Scales: 1 to 50
    let widths: Vec<f64> = (1..=50).map(|i| i as f64).collect();

    // Compute CWT
    let cwt_mat = continuous_wavelet_transform(&detrended, &widths);

    // Global Scale Spectrum: mean squared magnitude across time for each scale
    let mut scale_spectrum = Vec::with_capacity(widths.len());
    for row in &cwt_mat {
        let sum_sq = row.iter().map(|&x| x * x).sum::<f64>();
        scale_spectrum.push(sum_sq / n as f64);
    }

    // Find peak width
    let mut max_p = -1.0;
    let mut peak_idx = 0;
    for (i, &p) in scale_spectrum.iter().enumerate() {
        if p > max_p {
            max_p = p;
            peak_idx = i;
        }
    }
    let peak_width = widths[peak_idx];

    // Convert width to frequency
    // f = 2 / (sqrt(3) * pi * width)
    let peak_freq = 2.0 / (3.0_f64.sqrt() * PI * peak_width) * fs;

    println!("Top Scale Peak:");
    println!("  Width: {}", peak_width);
    println!("  Approx Freq: {:.4} cycles/sample", peak_freq);
    println!("  Ghost Freq Target: {:.4}", ALIASED_GHOST_FREQ);

    let mut detected = false;

    // Check Ghost
    if (peak_freq - ALIASED_GHOST_FREQ).abs() < 0.05 {
        println!("  *** GHOST SCALE MATCH (CWT) ***");
        detected = true;
    }

    // Check Ghost Band Power
    if scale_spectrum.len() >= 3 {
        let ghost_band_power = scale_spectrum[0..3].iter().sum::<f64>() / 3.0;
        let total_power = scale_spectrum.iter().sum::<f64>() / scale_spectrum.len() as f64;
        let ratio = ghost_band_power / total_power;
        println!("  Ghost Band Power Ratio: {:.2}", ratio);
        if ratio > 0.5 {
            println!("  *** SIGNIFICANT GHOST BAND POWER ***");
            detected = true;
        }
    }

    detected
}

fn load_wow_sband() -> Result<Option<Vec<f64>>> {
    let pattern = "data/bl_6equj5_gbt/*.h5";
    let paths: Vec<_> = glob(pattern)?.filter_map(|x| x.ok()).collect();
    if paths.is_empty() { return Ok(None); }

    let mut stacked_ts: Option<Vec<f64>> = None;
    let mut count = 0;

    for path in paths {
        match hdf5::File::open(&path) {
            Ok(file) => {
                if let Ok(dset) = file.dataset("data")
                    && let Ok(data) = dset.read_dyn::<f32>()
                {
                    let shape = data.shape();
                    let nt = shape[0];
                    let mut ts = vec![0.0; nt];

                    // Average across some channels
                    let n_ch_limit = 1000.min(shape[2]);
                    for t in 0..nt {
                        let mut sum = 0.0;
                        for c in 0..n_ch_limit {
                            sum += data[[t, 0, c]] as f64;
                        }
                        ts[t] = sum / n_ch_limit as f64;
                    }

                    if let Some(ref mut s) = stacked_ts {
                        if s.len() == ts.len() {
                            for (i, val) in ts.into_iter().enumerate() {
                                s[i] += val;
                            }
                            count += 1;
                        }
                    } else {
                        stacked_ts = Some(ts);
                        count = 1;
                    }
                }
            }
            Err(_) => continue,
        }
    }

    if let Some(mut s) = stacked_ts {
        if count > 0 {
            for val in &mut s { *val /= count as f64; }
        }
        Ok(Some(s))
    } else {
        Ok(None)
    }
}

fn load_alice() -> Result<Option<Vec<f64>>> {
    let path = "data/external/cms_oo_raa/alice_pbpb_raa_0to5pct_baseline.csv";
    if !std::path::Path::new(path).exists() { return Ok(None); }

    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(file);
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        // R_{AA} is often the last column or named
        // Based on Python df['R_{AA}'].values
        if let Some(val) = record.get(1) { // assuming 2nd column
            if let Ok(v) = val.parse::<f64>() {
                data.push(v);
            }
        }
    }
    Ok(if data.is_empty() { None } else { Some(data) })
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("=== Multidimensional 'Soft' Solver: CWT Analysis (Pure Rust) ===");

    match args.mode.as_str() {
        "wow" => {
            if let Some(data) = load_wow_sband()? {
                analyze_cwt(&data, 1.0, "Wow! S-Band Stacked");
            } else {
                println!("No Wow! data found.");
            }
        }
        "alice" => {
            if let Some(data) = load_alice()? {
                analyze_cwt(&data, 1.0, "ALICE Pb-Pb R_AA");
            } else {
                println!("No ALICE data found.");
            }
        }
        _ => println!("Mode not implemented yet."),
    }

    Ok(())
}
