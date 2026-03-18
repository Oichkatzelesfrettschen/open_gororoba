//! Cross-algebra template projection correlation (VLBI analogy).
//!
//! Tests whether galaxy-level projections onto different algebraic templates
//! (CD-ZD 7-mode, G2 6-mode, J3(O) 3-mode, sl(2) 2-mode) show excess
//! correlation beyond what template overlap in k-space alone predicts.
//!
//! Analogy: in VLBI, individual antennas see noise, but cross-correlating
//! voltage streams from different antennas recovers a correlated signal.
//! Here, different algebraic templates act as different "antennas" sampling
//! the same underlying dark matter signal through different algebraic windows.
//!
//! Detection: rho_observed - rho_null > 0.03 for any algebra pair (p < 0.001)
//! Rejection: |rho_observed - rho_null| < 0.02 for all 6 pairs
//!
//! Reference: Hypothesis H2, plan imperative-skipping-kahn.md

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, albert_peirce_wavenumbers,
        g2_angular_wavenumbers, sl2_partner_graph_wavenumbers,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::{f64::consts::PI, path::PathBuf};

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "cross-algebra-correlation")]
#[command(about = "Cross-algebra template projection correlation (VLBI analogy)")]
struct Cli {
    /// Path to MaNGA rotation curves CSV.
    #[arg(
        long,
        default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv"
    )]
    rotcurves: PathBuf,

    /// Path to DAPall selection CSV.
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    /// CD dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum rotation curve points per galaxy.
    #[arg(long, default_value_t = 8)]
    min_points: usize,

    /// Minimum x = r/r_s.
    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    /// Maximum x = r/r_s.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Output CSV path.
    #[arg(long, default_value = "cross_algebra_correlation.csv")]
    out_csv: PathBuf,
}

/// Template-weighted projection for a single algebra.
///
/// T = sum_n w_n * c(k_n)  where c(k) = re(k) + i*im(k) is the DFT coeff.
/// Returns (Re(T), Im(T)).
fn template_projection(
    x_values: &[f64],
    delta_values: &[f64],
    wavenumbers: &[f64],
    weights: &[f64],
) -> (f64, f64) {
    let mut t_re = 0.0_f64;
    let mut t_im = 0.0_f64;

    for (mode_idx, (&k, &w)) in wavenumbers.iter().zip(weights.iter()).enumerate() {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let mut count = 0_usize;

        for (&x, &dv) in x_values.iter().zip(delta_values.iter()) {
            re += dv * (k * x).cos();
            im += dv * (k * x).sin();
            count += 1;
        }

        if count > 0 {
            let norm = count as f64;
            re /= norm;
            im /= norm;
        }

        // Weight the contribution from this mode
        t_re += w * re;
        t_im += w * im;
        let _ = mode_idx; // suppress unused variable
    }

    (t_re, t_im)
}

/// Pearson correlation between two f64 slices.
fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    if dx2 == 0.0 || dy2 == 0.0 {
        return 0.0;
    }
    num / (dx2 * dy2).sqrt()
}

/// Fisher z-transform for significance testing.
fn fisher_z(r: f64, n: usize) -> f64 {
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    z * ((n - 3) as f64).sqrt()
}

/// Analytical expected correlation between two templates under noise.
///
/// When galaxy residuals are pure noise, the expected correlation between
/// template projections T_A and T_B is determined by the inner product
/// of their weight vectors in wavenumber space.
fn analytical_null_correlation(
    wavenumbers_a: &[f64],
    weights_a: &[f64],
    wavenumbers_b: &[f64],
    weights_b: &[f64],
    x_min: f64,
    x_max: f64,
) -> f64 {
    // Under white noise, the correlation between two templates is:
    // rho_null = (w_A . w_B)_k / (|w_A|_k * |w_B|_k)
    // where the inner product is over shared wavenumber space.
    //
    // For DFT at different k-values, cross-terms vanish for orthogonal
    // functions over [x_min, x_max]. The correlation comes from
    // wavenumber overlap and the finite-range non-orthogonality.
    //
    // Approximate: numerically integrate cos(k_a * x) * cos(k_b * x) over [x_min, x_max].
    let n_quad = 1000;
    let dx = (x_max - x_min) / n_quad as f64;

    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    let mut inner = 0.0_f64;

    for ia in 0..wavenumbers_a.len() {
        for ib in 0..wavenumbers_b.len() {
            let ka = wavenumbers_a[ia];
            let kb = wavenumbers_b[ib];
            let wa = weights_a[ia];
            let wb = weights_b[ib];

            // <cos(ka*x), cos(kb*x)> + <sin(ka*x), sin(kb*x)>
            // = integral cos((ka-kb)*x) dx over [x_min, x_max]
            // (by product-to-sum identity)
            let mut sum = 0.0_f64;
            for q in 0..n_quad {
                let x = x_min + (q as f64 + 0.5) * dx;
                sum += ((ka - kb) * x).cos() * dx;
            }
            let overlap = sum / (x_max - x_min);
            inner += wa * wb * overlap;
        }
    }

    // Norms
    for ia in 0..wavenumbers_a.len() {
        for ia2 in 0..wavenumbers_a.len() {
            let ka = wavenumbers_a[ia];
            let ka2 = wavenumbers_a[ia2];
            let wa = weights_a[ia];
            let wa2 = weights_a[ia2];
            let mut sum = 0.0_f64;
            for q in 0..n_quad {
                let x = x_min + (q as f64 + 0.5) * dx;
                sum += ((ka - ka2) * x).cos() * dx;
            }
            norm_a += wa * wa2 * sum / (x_max - x_min);
        }
    }

    for ib in 0..wavenumbers_b.len() {
        for ib2 in 0..wavenumbers_b.len() {
            let kb = wavenumbers_b[ib];
            let kb2 = wavenumbers_b[ib2];
            let wb = weights_b[ib];
            let wb2 = weights_b[ib2];
            let mut sum = 0.0_f64;
            for q in 0..n_quad {
                let x = x_min + (q as f64 + 0.5) * dx;
                sum += ((kb - kb2) * x).cos() * dx;
            }
            norm_b += wb * wb2 * sum / (x_max - x_min);
        }
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 0.0;
    }
    inner / (norm_a * norm_b).sqrt()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading MaNGA rotation curves from {:?}...", cli.rotcurves);
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("Loaded {} galaxy rotation curves", rotcurves.len());

    eprintln!("Loading DAPall metadata from {:?}...", cli.dapall);
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;

    let dapall_map: std::collections::HashMap<String, _> = dapall
        .into_iter()
        .map(|e| (e.plateifu.clone(), e))
        .collect();

    // Build per-galaxy residuals
    let mut galaxies: Vec<NormalizedResiduals> = Vec::new();
    let mut skipped = 0usize;

    for galaxy in &rotcurves {
        let meta = match dapall_map.get(&galaxy.plateifu) {
            Some(m) => m,
            None => {
                skipped += 1;
                continue;
            }
        };

        let m200 = 10.0_f64.powf(meta.estimated_log_m200());
        let z = meta.z;
        let nfw = nfw_params_from_mass(m200, z);
        let r_s = nfw.r_s_kpc;

        let points: Vec<NormalizedPoint> = galaxy
            .points
            .iter()
            .filter_map(|pt| {
                if pt.r_kpc <= 0.0 || r_s <= 0.0 {
                    return None;
                }
                let x = pt.r_kpc / r_s;
                let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
                let v_sq = G_KPC_KMS2 * m_enc / pt.r_kpc;
                let v_nfw = v_sq.max(0.0).sqrt();
                if v_nfw < 1.0 {
                    return None;
                }
                let delta_v = (pt.v_obs_km_s - v_nfw) / v_nfw;
                let delta_v_err = pt.v_err_km_s / v_nfw;
                if !delta_v.is_finite() || !delta_v_err.is_finite() {
                    return None;
                }
                Some(NormalizedPoint {
                    x,
                    delta_v,
                    delta_v_err,
                })
            })
            .collect();

        if points.len() >= cli.min_points {
            galaxies.push(NormalizedResiduals {
                name: galaxy.plateifu.clone(),
                r_s_kpc: r_s,
                points,
            });
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "Normalized {} galaxies ({} skipped)",
        galaxies.len(),
        skipped
    );

    // Define algebra wavenumbers and weights
    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let cd_k: Vec<f64> = (1..=cd_params.n_modes)
        .map(|n| 2.0 * PI * n as f64 / cd_params.n_modes as f64)
        .collect();
    let cd_w: Vec<f64> = (1..=cd_params.n_modes)
        .map(|n| cd_params.assessor_fraction / n as f64)
        .collect();

    let g2_k = g2_angular_wavenumbers();
    let g2_w: Vec<f64> = (1..=g2_k.len()).map(|n| 1.0 / (n as f64).sqrt()).collect();

    let j3o_k = albert_peirce_wavenumbers();
    let j3o_w: Vec<f64> = vec![1.0; j3o_k.len()]; // Uniform Peirce weighting

    let sl2_k = sl2_partner_graph_wavenumbers();
    let sl2_w: Vec<f64> = vec![14.0 / 84.0, 7.0 / 84.0]; // Degeneracy-weighted: 14/84, 7/84

    let algebra_names = ["CD-ZD", "G2", "J3(O)", "sl(2)"];
    let all_k = [
        cd_k.as_slice(),
        g2_k.as_slice(),
        j3o_k.as_slice(),
        sl2_k.as_slice(),
    ];
    let all_w = [
        cd_w.as_slice(),
        g2_w.as_slice(),
        j3o_w.as_slice(),
        sl2_w.as_slice(),
    ];

    eprintln!("Computing per-galaxy template projections for 4 algebras...");

    // Per-galaxy template projections: [galaxy][algebra] -> (re, im)
    let n_gal = galaxies.len();
    let mut projections: Vec<[(f64, f64); 4]> = Vec::with_capacity(n_gal);

    for (gi, galaxy) in galaxies.iter().enumerate() {
        if gi % 1000 == 0 {
            eprintln!("  Galaxy {}/{}...", gi, n_gal);
        }

        let x_vals: Vec<f64> = galaxy.points.iter().map(|p| p.x).collect();
        let dv_vals: Vec<f64> = galaxy.points.iter().map(|p| p.delta_v).collect();

        let mut proj = [(0.0_f64, 0.0_f64); 4];
        for alg_idx in 0..4 {
            proj[alg_idx] = template_projection(&x_vals, &dv_vals, all_k[alg_idx], all_w[alg_idx]);
        }
        projections.push(proj);
    }

    eprintln!("Computing cross-correlations...");

    // Extract per-algebra projection vectors
    let mut alg_re: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n_gal)).collect();
    let mut alg_im: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n_gal)).collect();
    for proj in &projections {
        for a in 0..4 {
            alg_re[a].push(proj[a].0);
            alg_im[a].push(proj[a].1);
        }
    }

    // 6 algebra pairs
    let pairs: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    // Analytical null baselines
    eprintln!("\n--- Analytical null baselines ---");
    let mut null_baselines: Vec<f64> = Vec::new();
    for &(a, b) in &pairs {
        let rho_null = analytical_null_correlation(
            all_k[a], all_w[a], all_k[b], all_w[b], cli.x_min, cli.x_max,
        );
        null_baselines.push(rho_null);
        eprintln!(
            "  {}-{}: rho_null = {:.6}",
            algebra_names[a], algebra_names[b], rho_null
        );
    }

    // Observed correlations (Re and Im channels averaged)
    eprintln!("\n--- Observed correlations ---");
    eprintln!(
        "{:>12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "pair", "rho_re", "rho_im", "rho_avg", "rho_null", "excess"
    );

    let mut results: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();

    for (pi, &(a, b)) in pairs.iter().enumerate() {
        let rho_re = pearson_r(&alg_re[a], &alg_re[b]);
        let rho_im = pearson_r(&alg_im[a], &alg_im[b]);
        let rho_avg = (rho_re + rho_im) / 2.0;
        let rho_null = null_baselines[pi];
        let excess = rho_avg - rho_null;

        let pair_name = format!("{}-{}", algebra_names[a], algebra_names[b]);
        eprintln!(
            "{:>12}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
            pair_name, rho_re, rho_im, rho_avg, rho_null, excess
        );

        results.push((pair_name, rho_re, rho_im, rho_avg, rho_null, excess));
    }

    // Random template control (uniform random wavenumbers, 5 modes)
    eprintln!("\n--- Random template control ---");
    let rng_seed: u64 = 42;
    let mut rng_state = rng_seed;
    let rand_k: Vec<f64> = (0..5)
        .map(|_| {
            // Simple LCG for reproducibility (no external crate needed)
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
            0.5 + u * 6.5 // k in [0.5, 7.0]
        })
        .collect();
    let rand_w: Vec<f64> = vec![1.0; 5];

    let mut rand_re: Vec<f64> = Vec::with_capacity(n_gal);
    let mut rand_im: Vec<f64> = Vec::with_capacity(n_gal);
    for galaxy in &galaxies {
        let x_vals: Vec<f64> = galaxy.points.iter().map(|p| p.x).collect();
        let dv_vals: Vec<f64> = galaxy.points.iter().map(|p| p.delta_v).collect();
        let (re, im) = template_projection(&x_vals, &dv_vals, &rand_k, &rand_w);
        rand_re.push(re);
        rand_im.push(im);
    }

    for a in 0..4 {
        let rho_re = pearson_r(&alg_re[a], &rand_re);
        let rho_im = pearson_r(&alg_im[a], &rand_im);
        eprintln!(
            "  {}-RANDOM: rho_re={:.6}, rho_im={:.6}",
            algebra_names[a], rho_re, rho_im
        );
    }

    // Detection/rejection summary
    let max_excess = results.iter().map(|r| r.5.abs()).fold(0.0_f64, f64::max);
    let any_detection = results.iter().any(|r| {
        let z = fisher_z(r.3, n_gal);
        r.5 > 0.03 && z.abs() > 3.3 // p < 0.001 (Bonferroni-adjusted ~ p < 0.000167)
    });

    eprintln!("\n--- Detection criteria ---");
    eprintln!("Max |excess|: {:.6}", max_excess);
    if any_detection {
        eprintln!("RESULT: DETECTION -- excess cross-algebra correlation found");
    } else if max_excess < 0.02 {
        eprintln!("RESULT: REJECTED -- no excess correlation above noise");
    } else {
        eprintln!("RESULT: INCONCLUSIVE -- excess between 0.02 and 0.03");
    }

    // Write output CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "pair", "rho_re", "rho_im", "rho_avg", "rho_null", "excess", "fisher_z",
    ])?;
    for r in &results {
        let z = fisher_z(r.3, n_gal);
        wtr.write_record(&[
            r.0.clone(),
            format!("{:.8}", r.1),
            format!("{:.8}", r.2),
            format!("{:.8}", r.3),
            format!("{:.8}", r.4),
            format!("{:.8}", r.5),
            format!("{:.4}", z),
        ])?;
    }
    wtr.flush()?;

    // Write per-galaxy projections
    let galaxy_csv = cli.out_csv.with_extension("galaxies.csv");
    let mut gwtr = csv::Writer::from_path(&galaxy_csv)?;
    gwtr.write_record([
        "plateifu", "cd_re", "cd_im", "g2_re", "g2_im", "j3o_re", "j3o_im", "sl2_re", "sl2_im",
    ])?;
    for (gi, galaxy) in galaxies.iter().enumerate() {
        let p = &projections[gi];
        gwtr.write_record(&[
            galaxy.name.clone(),
            format!("{:.8e}", p[0].0),
            format!("{:.8e}", p[0].1),
            format!("{:.8e}", p[1].0),
            format!("{:.8e}", p[1].1),
            format!("{:.8e}", p[2].0),
            format!("{:.8e}", p[2].1),
            format!("{:.8e}", p[3].0),
            format!("{:.8e}", p[3].1),
        ])?;
    }
    gwtr.flush()?;

    eprintln!("Pair results written to {:?}", cli.out_csv);
    eprintln!("Galaxy projections written to {:?}", galaxy_csv);

    Ok(())
}
