//! Join associator norms with ESA plasma beta for A(beta) scatter.
//! Reads norms from CSV (stdin or file), ESA density+temp from cached files.

use clap::Args;
use std::{collections::BTreeMap, fs};

const MU0: f64 = 1.2566370614e-6;
const KB: f64 = 1.380649e-23;
const EV_TO_K: f64 = 11604.518;

#[derive(Args)]
pub struct Cli {
    /// Path to associator norms CSV (doy,hour,minute,norm).
    #[arg(long)]
    norms_csv: String,
    /// Path to FGM minute data (for |B|).
    #[arg(long, default_value = "data/external")]
    data_dir: String,
    /// Year of the data.
    #[arg(long, default_value_t = 2008)]
    year: u16,
    /// DOY range start.
    #[arg(long, default_value_t = 301)]
    doy_start: u16,
    /// DOY range end.
    #[arg(long, default_value_t = 307)]
    doy_end: u16,
}

pub fn run(cli: Cli) {
    // Load norms
    let mut norms: BTreeMap<(u16, u8, u8), f64> = BTreeMap::new();
    let content = fs::read_to_string(&cli.norms_csv).expect("read norms");
    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            continue;
        }
        let doy: u16 = parts[0].parse().unwrap_or(0);
        let hour: u8 = parts[1].parse().unwrap_or(0);
        let minute: u8 = parts[2].parse().unwrap_or(0);
        let norm: f64 = parts[3].parse().unwrap_or(0.0);
        norms.insert((doy, hour, minute), norm);
    }
    eprintln!("Loaded {} norms", norms.len());

    // Load FGM for |B|
    let mut bmag_map: BTreeMap<(u16, u8, u8), (f64, usize)> = BTreeMap::new();
    for doy in cli.doy_start..=cli.doy_end {
        let path = format!(
            "{}/themis/tha_fgm_{:04}_{:03}.csv",
            cli.data_dir, cli.year, doy
        );
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 4 {
                continue;
            }
            let ts = parts[0];
            if ts.len() < 19 {
                continue;
            }
            let hour: u8 = ts[11..13].parse().unwrap_or(0);
            let minute: u8 = ts[14..16].parse().unwrap_or(0);
            let bx: f64 = parts[1].parse().unwrap_or(f64::NAN);
            let by: f64 = parts[2].parse().unwrap_or(f64::NAN);
            let bz: f64 = parts[3].parse().unwrap_or(f64::NAN);
            if !bx.is_finite() || bx.abs() > 1e10 {
                continue;
            }
            let bmag = (bx * bx + by * by + bz * bz).sqrt();
            if bmag > 200.0 {
                continue;
            }
            let entry = bmag_map.entry((doy, hour, minute)).or_insert((0.0, 0));
            entry.0 += bmag;
            entry.1 += 1;
        }
    }
    let bmag_avg: BTreeMap<(u16, u8, u8), f64> = bmag_map
        .iter()
        .map(|(&k, &(sum, count))| (k, sum / count as f64))
        .collect();
    eprintln!("Loaded {} FGM minutes", bmag_avg.len());

    // Load ESA (density, temp)
    let mut esa_map: BTreeMap<(u16, u8, u8), (f64, f64)> = BTreeMap::new();
    for doy in cli.doy_start..=cli.doy_end {
        let path = format!(
            "{}/themis_esa/tha_esa_{:04}_{}.csv",
            cli.data_dir, cli.year, doy
        );
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines() {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 3 {
                continue;
            }
            let ts = parts[0];
            if ts.len() < 19 || !ts.starts_with(|c: char| c.is_ascii_digit()) {
                continue;
            }
            let hour: u8 = ts[11..13].parse().unwrap_or(255);
            let minute: u8 = ts[14..16].parse().unwrap_or(255);
            if hour > 23 || minute > 59 {
                continue;
            }
            let den: f64 = parts[1].parse().unwrap_or(f64::NAN);
            let temp: f64 = parts[2].parse().unwrap_or(f64::NAN);
            if !den.is_finite() || den <= 0.0 || !temp.is_finite() || temp <= 0.0 {
                continue;
            }
            // Extract DOY from timestamp
            let ts_doy = if ts.len() >= 10 {
                let month: u32 = ts[5..7].parse().unwrap_or(1);
                let day: u32 = ts[8..10].parse().unwrap_or(1);
                // Approximate DOY
                let days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
                (days_before[month.saturating_sub(1) as usize] + day) as u16
            } else {
                0
            };
            esa_map.insert((ts_doy, hour, minute), (den, temp));
        }
    }
    eprintln!("Loaded {} ESA minutes", esa_map.len());

    // Join: for each minute with all 3, compute beta and output
    let mut a_vals = Vec::new();
    let mut beta_vals = Vec::new();

    for (&key, &norm) in &norms {
        let Some(&bmag_nt) = bmag_avg.get(&key) else {
            continue;
        };
        let Some(&(den_cc, temp_ev)) = esa_map.get(&key) else {
            continue;
        };

        let b_t = bmag_nt * 1e-9;
        let n_m3 = den_cc * 1e6;
        let t_k = temp_ev * EV_TO_K;
        let p_th = n_m3 * KB * t_k;
        let p_mag = b_t * b_t / (2.0 * MU0);
        if p_mag < 1e-20 {
            continue;
        }
        let beta = p_th / p_mag;
        if beta <= 0.0 || beta > 1000.0 {
            continue;
        }

        a_vals.push(norm);
        beta_vals.push(beta);
    }

    eprintln!("Matched A-beta pairs: {}", a_vals.len());

    if a_vals.len() < 10 {
        eprintln!("Too few pairs");
        return;
    }

    // Spearman correlation
    let rank = |vals: &[f64]| -> Vec<f64> {
        let n = vals.len();
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0.0; n];
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            ranks[orig_idx] = rank as f64;
        }
        ranks
    };

    let ra = rank(&a_vals);
    let rb = rank(&beta_vals);
    let n = ra.len() as f64;
    let mean_ra = ra.iter().sum::<f64>() / n;
    let mean_rb = rb.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..ra.len() {
        let da = ra[i] - mean_ra;
        let db = rb[i] - mean_rb;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    let spearman = if denom > 1e-15 { cov / denom } else { 0.0 };

    // Power law fit in log-log
    let log_a: Vec<f64> = a_vals.iter().map(|&a| (a + 1e-10).log10()).collect();
    let log_b: Vec<f64> = beta_vals.iter().map(|&b| b.log10()).collect();
    let sum_x: f64 = log_b.iter().sum();
    let sum_y: f64 = log_a.iter().sum();
    let sum_xy: f64 = log_b.iter().zip(log_a.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_b.iter().map(|x| x * x).sum();
    let alpha = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

    println!("{{");
    println!("  \"normalization\": \"direction\",");
    println!("  \"n_matched\": {},", a_vals.len());
    println!("  \"spearman_r\": {:.4},", spearman);
    println!("  \"power_law_exponent\": {:.4},", alpha);
    println!("  \"current_exponent\": 0.775,");
    println!(
        "  \"interpretation\": \"Direction-only A(beta): Spearman r={:.3}, exponent={:.3}. {}\"",
        spearman,
        alpha,
        if spearman > 0.3 {
            "Power law SURVIVES direction-only embedding. The beta correlation is genuine physics, not normalization artifact."
        } else if spearman > 0.1 {
            "Weak correlation survives. Partial normalization contribution."
        } else {
            "Correlation VANISHES in direction-only. The A(beta) power law is primarily a normalization effect."
        }
    );
    println!("}}");
}
