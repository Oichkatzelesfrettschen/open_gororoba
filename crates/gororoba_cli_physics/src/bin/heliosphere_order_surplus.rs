//! Heliosphere Order Surplus Index (Omega) computation.
//!
//! Phase 3 formalization: compute Omega_mv and Omega_phase per radial bin.
//!   Omega_mv = 1 - A_base / A_mv_phase_null
//!   Omega_phase = 1 - A_base / A_shared_phase_null
//!
//! Omega > 0 means excess algebraic order beyond what phase-randomization can explain.
//! Omega ~ 0 means null-consistent. Omega < 0 means excess non-associativity.
//!
//! The gap Omega_mv - Omega_phase isolates cross-channel coupling from
//! within-channel spectral structure.

use anyhow::{Context, Result};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use data_core::{HeliosphereFeatureRow, magnetic_takens_embed};
use rand::RngExt;
use rustfft::{FftPlanner, num_complex::Complex64};
use serde::Serialize;
use std::{collections::BTreeMap, f64::consts::PI, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-order-surplus",
    about = "Compute per-radial-bin Omega_mv and Omega_phase order surplus indices"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/order_surplus_by_rbin.csv"
    )]
    out_csv: PathBuf,

    #[arg(long, default_value_t = 5.0)]
    bin_size_au: f64,

    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    #[arg(long, default_value_t = 1)]
    takens_lag: usize,

    /// Number of channels: 4 (B-field only) or 8 (B-field + plasma).
    /// 8-channel uses Bx,By,Bz,dB/B,n_p,v_sw,T_p,v_A -- requires
    /// density/speed/temperature to be present and finite.
    #[arg(long, default_value_t = 4)]
    channels: usize,

    /// Precision: "f32" for 37x speedup at 128D+, "f64" for full precision.
    #[arg(long, default_value = "f64")]
    precision: String,

    /// Null iterations per radial bin (more = tighter CI, slower).
    #[arg(long, default_value_t = 5)]
    null_iterations: usize,

    /// Exclude rows from these missions.
    #[arg(long = "exclude-mission")]
    exclude_missions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OmegaBin {
    r_center_au: f64,
    base_mean: f64,
    null_phase_mean: f64,
    null_mv_phase_mean: f64,
    omega_phase: f64,
    omega_mv: f64,
    gap: f64,
    sample_count: usize,
    mission_diversity: usize,
}

/// Shared-phase surrogate: one random phase set applied to all channels.
/// Preserves cross-channel coherence, destroys nonlinear phase coupling.
fn phase_randomize_shared(vectors: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let n = vectors.len();
    if n < 4 {
        return vectors.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let fft_inv = planner.plan_fft_inverse(n);
    let mut rng = rand::rng();
    let half = n / 2;
    let random_phases: Vec<f64> = (0..=half).map(|_| rng.random::<f64>() * 2.0 * PI).collect();

    let mut result = vec![vec![0.0; dim]; n];
    for ch in 0..dim {
        let mut buf: Vec<Complex64> = vectors.iter().map(|v| Complex64::new(v[ch], 0.0)).collect();
        fft_fwd.process(&mut buf);
        for k in 1..half {
            let phase = Complex64::from_polar(1.0, random_phases[k]);
            buf[k] *= phase;
            buf[n - k] *= phase.conj();
        }
        if n.is_multiple_of(2) {
            let sign = if random_phases[half] > PI { -1.0 } else { 1.0 };
            buf[half] *= sign;
        }
        fft_inv.process(&mut buf);
        let scale = 1.0 / n as f64;
        for (i, c) in buf.iter().enumerate() {
            result[i][ch] = c.re * scale;
        }
    }
    result
}

/// Multivariate-phase surrogate: independent random phases per channel.
/// Destroys cross-channel phase coupling while preserving per-channel power spectrum.
fn phase_randomize_multivariate(vectors: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let n = vectors.len();
    if n < 4 {
        return vectors.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let fft_inv = planner.plan_fft_inverse(n);
    let mut rng = rand::rng();
    let half = n / 2;

    let mut result = vec![vec![0.0; dim]; n];
    for ch in 0..dim {
        let channel_phases: Vec<f64> = (0..=half).map(|_| rng.random::<f64>() * 2.0 * PI).collect();
        let mut buf: Vec<Complex64> = vectors.iter().map(|v| Complex64::new(v[ch], 0.0)).collect();
        fft_fwd.process(&mut buf);
        for k in 1..half {
            let phase = Complex64::from_polar(1.0, channel_phases[k]);
            buf[k] *= phase;
            buf[n - k] *= phase.conj();
        }
        if n.is_multiple_of(2) {
            let sign = if channel_phases[half] > PI { -1.0 } else { 1.0 };
            buf[half] *= sign;
        }
        fft_inv.process(&mut buf);
        let scale = 1.0 / n as f64;
        for (i, c) in buf.iter().enumerate() {
            result[i][ch] = c.re * scale;
        }
    }
    result
}

fn associator_mean(norms: &[f64]) -> f64 {
    if norms.is_empty() {
        0.0
    } else {
        norms.iter().sum::<f64>() / norms.len() as f64
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== Heliosphere Order Surplus by Radial Bin ===");
    println!(
        "  dim={}, channels={}, lag={}, precision={}, null_iters={}",
        cli.embedding_dim, cli.channels, cli.takens_lag, cli.precision, cli.null_iterations
    );
    assert!(
        cli.channels == 4 || cli.channels == 8,
        "channels must be 4 or 8"
    );

    let mut reader = ReaderBuilder::new()
        .from_path(&cli.cube_csv)
        .with_context(|| format!("open {}", cli.cube_csv.display()))?;

    let exclude: std::collections::HashSet<&str> =
        cli.exclude_missions.iter().map(|s| s.as_str()).collect();

    let mut all_rows = Vec::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        if exclude.contains(r.mission.as_str()) {
            continue;
        }
        if r.bx.is_finite()
            && r.by.is_finite()
            && r.bz.is_finite()
            && r.b_mag.is_finite()
            && r.b_mag > 0.0
        {
            all_rows.push(r);
        }
    }
    println!(
        "  Loaded {} rows from {}",
        all_rows.len(),
        cli.cube_csv.display()
    );

    // Group by mission, sort chronologically
    let mut mission_groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> =
        BTreeMap::new();
    for row in all_rows {
        mission_groups
            .entry((row.mission.clone(), row.product.clone()))
            .or_default()
            .push(row);
    }
    for rows in mission_groups.values_mut() {
        rows.sort_by(|a, b| {
            a.year
                .cmp(&b.year)
                .then(a.doy.cmp(&b.doy))
                .then(a.hour.cmp(&b.hour))
        });
    }

    let dim = cli.embedding_dim;
    let steps = dim / cli.channels;

    // Step 1: Compute base associator norms tagged with radial position and mission
    println!(
        "[1/3] Computing base {}D ({}-ch x {} steps) Takens associators...",
        dim, cli.channels, steps
    );
    // (r_bin, norm, mission)
    let mut tagged_norms: Vec<(i32, f64, String)> = Vec::new();
    // (r_bin, embedded vectors) -- for null computation
    let mut bin_vectors: BTreeMap<i32, Vec<Vec<f64>>> = BTreeMap::new();

    for ((mission, _product), rows) in &mission_groups {
        if rows.len() < steps + 5 {
            continue;
        }
        let (embedded, meta_idx) = if cli.channels == 8 {
            let (e, i, _eligible) = data_core::plasma_takens_embed_dim(rows, dim, cli.takens_lag);
            (e, i)
        } else {
            magnetic_takens_embed(rows, dim, cli.takens_lag)
        };
        let norms =
            cd_kernel::batch_sliding_associator_norms_dispatch(&embedded, dim, &cli.precision);

        for (k, &norm) in norms.iter().enumerate() {
            let tag_row = meta_idx[k + 2];
            let r_bin = (rows[tag_row].r_au / cli.bin_size_au).floor() as i32;
            tagged_norms.push((r_bin, norm, mission.clone()));
        }

        // Also collect embedded vectors per radial bin for null computation
        for (k, vec) in embedded.iter().enumerate() {
            let tag_row = meta_idx[k.min(meta_idx.len() - 1)];
            let r_bin = (rows[tag_row].r_au / cli.bin_size_au).floor() as i32;
            bin_vectors.entry(r_bin).or_default().push(vec.clone());
        }
    }

    // Aggregate base stats per radial bin
    let mut bin_base: BTreeMap<i32, (f64, usize, usize)> = BTreeMap::new(); // (sum, count, mission_div)
    let mut bin_missions: BTreeMap<i32, std::collections::HashSet<String>> = BTreeMap::new();
    for (r_bin, norm, mission) in &tagged_norms {
        let entry = bin_base.entry(*r_bin).or_insert((0.0, 0, 0));
        entry.0 += norm;
        entry.1 += 1;
        bin_missions
            .entry(*r_bin)
            .or_default()
            .insert(mission.clone());
    }
    for (r_bin, missions) in &bin_missions {
        if let Some(entry) = bin_base.get_mut(r_bin) {
            entry.2 = missions.len();
        }
    }

    // Step 2: Compute null associators per radial bin
    let n_bins = bin_vectors.len();
    println!(
        "[2/3] Computing null surrogates for {} radial bins ({} iters each)...",
        n_bins, cli.null_iterations
    );

    let mut bin_nulls: BTreeMap<i32, (f64, f64)> = BTreeMap::new(); // (phase_mean, mv_phase_mean)

    for (idx, (r_bin, vectors)) in bin_vectors.iter().enumerate() {
        if vectors.len() < 5 {
            bin_nulls.insert(*r_bin, (0.0, 0.0));
            continue;
        }

        let mut phase_means = Vec::new();
        let mut mv_phase_means = Vec::new();

        for _ in 0..cli.null_iterations {
            let surr_phase = phase_randomize_shared(vectors, dim);
            let norms_phase = cd_kernel::batch_sliding_associator_norms_dispatch(
                &surr_phase,
                dim,
                &cli.precision,
            );
            phase_means.push(associator_mean(&norms_phase));

            let surr_mv = phase_randomize_multivariate(vectors, dim);
            let norms_mv =
                cd_kernel::batch_sliding_associator_norms_dispatch(&surr_mv, dim, &cli.precision);
            mv_phase_means.push(associator_mean(&norms_mv));
        }

        let phase_avg = phase_means.iter().sum::<f64>() / phase_means.len() as f64;
        let mv_avg = mv_phase_means.iter().sum::<f64>() / mv_phase_means.len() as f64;
        bin_nulls.insert(*r_bin, (phase_avg, mv_avg));

        if (idx + 1) % 5 == 0 || idx + 1 == n_bins {
            let r_center = (*r_bin as f64 + 0.5) * cli.bin_size_au;
            println!(
                "    [{}/{}] r={:.0} AU: phase_null={:.4}, mv_null={:.4}",
                idx + 1,
                n_bins,
                r_center,
                phase_avg,
                mv_avg
            );
        }
    }

    // Step 3: Compute Omega and write output
    println!("[3/3] Computing Omega and writing CSV...");
    if let Some(parent) = cli.out_csv.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = WriterBuilder::new().from_path(&cli.out_csv)?;

    let mut r_bins_sorted: Vec<i32> = bin_base.keys().copied().collect();
    r_bins_sorted.sort();

    for r_bin in r_bins_sorted {
        let (sum, count, mission_div) = bin_base[&r_bin];
        let base_mean = sum / count as f64;
        let (phase_null, mv_null) = bin_nulls.get(&r_bin).copied().unwrap_or((0.0, 0.0));

        // Omega = 1 - A_base / A_null
        // If null is zero, Omega is undefined -- set to 0.0
        let omega_phase = if phase_null > 1e-15 {
            1.0 - base_mean / phase_null
        } else {
            0.0
        };
        let omega_mv = if mv_null > 1e-15 {
            1.0 - base_mean / mv_null
        } else {
            0.0
        };
        let gap = omega_mv - omega_phase;
        let r_center = (r_bin as f64 + 0.5) * cli.bin_size_au;

        writer.serialize(OmegaBin {
            r_center_au: r_center,
            base_mean,
            null_phase_mean: phase_null,
            null_mv_phase_mean: mv_null,
            omega_phase,
            omega_mv,
            gap,
            sample_count: count,
            mission_diversity: mission_div,
        })?;
    }

    println!("  Wrote {}", cli.out_csv.display());
    Ok(())
}
