//! Topological persistence analysis of stacked rotation curve residuals.
//!
//! Computes 0-dimensional sublevel set persistence of the stacked delta_v(x)
//! profile. A 7-mode ZD oscillation creates ~7 persistent H_0 features
//! (born/die at different thresholds). A purely baryonic residual (monotonic
//! cusp + edge effects) creates only 2-3 features.
//!
//! The 1D sublevel persistence algorithm sweeps a threshold from min to max,
//! tracking connected components via union-find. Each local minimum births
//! a component; each local maximum merges two components (younger dies).
//!
//! Reference: Hypothesis H3 from novel hypothesis plan.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "persistence-residual-topology")]
#[command(about = "1D sublevel persistence of stacked rotation curve residuals")]
struct Cli {
    /// Input stacked CSV (columns: x, delta_stack, delta_stack_err, n_contributing).
    #[arg(long)]
    stacked_csv: PathBuf,

    /// Minimum galaxies per x-bin to include.
    #[arg(long, default_value_t = 10)]
    min_per_bin: usize,

    /// Persistence threshold as fraction of max |delta|.
    #[arg(long, default_value_t = 0.1)]
    epsilon_frac: f64,

    /// Optional directory of sweep CSVs for mass-bin stability test.
    #[arg(long)]
    sweep_dir: Option<PathBuf>,
}

fn load_stacked_csv(path: &std::path::Path) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();

    let x_idx = headers
        .iter()
        .position(|h| h == "x")
        .ok_or_else(|| anyhow::anyhow!("No 'x' column in {}", path.display()))?;
    let delta_idx = headers
        .iter()
        .position(|h| h == "delta_stack")
        .ok_or_else(|| anyhow::anyhow!("No 'delta_stack' column in {}", path.display()))?;
    let n_idx = headers
        .iter()
        .position(|h| h == "n_contributing")
        .ok_or_else(|| anyhow::anyhow!("No 'n_contributing' column in {}", path.display()))?;

    let mut x_grid = Vec::new();
    let mut delta = Vec::new();
    let mut n_contrib = Vec::new();

    for result in rdr.records() {
        let rec = result?;
        let x: f64 = rec.get(x_idx).unwrap_or("0").parse().unwrap_or(0.0);
        let d: f64 = rec.get(delta_idx).unwrap_or("0").parse().unwrap_or(0.0);
        let n: usize = rec.get(n_idx).unwrap_or("0").parse().unwrap_or(0);
        x_grid.push(x);
        delta.push(d);
        n_contrib.push(n);
    }

    Ok((x_grid, delta, n_contrib))
}

/// Birth-death pair from 0-dimensional sublevel persistence.
#[derive(Debug, Clone)]
struct PersistencePair {
    birth: f64,
    death: f64,
}

impl PersistencePair {
    fn lifetime(&self) -> f64 {
        self.death - self.birth
    }
}

/// Compute 0-dimensional sublevel set persistence of a 1D function.
///
/// Processes values in ascending order using union-find. Each point that
/// creates a new component births it at its value. When two components merge
/// at a saddle, the younger one (higher birth) dies.
fn sublevel_persistence_1d(values: &[f64]) -> Vec<PersistencePair> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }

    // Sort indices by function value (ascending).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut parent: Vec<usize> = (0..n).collect();
    let birth: Vec<f64> = values.to_vec();
    let mut active = vec![false; n];
    let mut pairs: Vec<PersistencePair> = Vec::new();

    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != r {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }

    for &idx in &order {
        active[idx] = true;
        let mut curr = idx;

        // Check left neighbor.
        if idx > 0 && active[idx - 1] {
            let left = find(&mut parent, idx - 1);
            curr = find(&mut parent, curr);
            if left != curr {
                let (older, younger) = if birth[left] <= birth[curr] {
                    (left, curr)
                } else {
                    (curr, left)
                };
                pairs.push(PersistencePair {
                    birth: birth[younger],
                    death: values[idx],
                });
                parent[younger] = older;
                curr = older;
            }
        }

        // Check right neighbor.
        if idx + 1 < n && active[idx + 1] {
            let right = find(&mut parent, idx + 1);
            curr = find(&mut parent, curr);
            if right != curr {
                let (older, younger) = if birth[right] <= birth[curr] {
                    (right, curr)
                } else {
                    (curr, right)
                };
                pairs.push(PersistencePair {
                    birth: birth[younger],
                    death: values[idx],
                });
                parent[younger] = older;
            }
        }
    }

    // Sort by decreasing lifetime.
    pairs.sort_by(|a, b| {
        b.lifetime()
            .partial_cmp(&a.lifetime())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs
}

fn analyze_profile(
    label: &str,
    x_grid: &[f64],
    delta: &[f64],
    n_contrib: &[usize],
    min_per_bin: usize,
    epsilon_frac: f64,
) -> usize {
    // Filter to valid bins.
    let valid: Vec<(f64, f64)> = x_grid
        .iter()
        .zip(delta.iter())
        .zip(n_contrib.iter())
        .filter(|&((_, _), n)| *n >= min_per_bin)
        .map(|((x, d), _)| (*x, *d))
        .collect();

    let n_bins = valid.len();
    if n_bins < 3 {
        eprintln!("  {}: only {} valid bins, skipping", label, n_bins);
        return 0;
    }

    let values: Vec<f64> = valid.iter().map(|(_, d)| *d).collect();
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let epsilon = epsilon_frac * max_abs;

    let pairs = sublevel_persistence_1d(&values);
    let n_persistent = pairs.iter().filter(|p| p.lifetime() > epsilon).count();

    eprintln!("  {}: {} valid bins, {} persistence pairs", label, n_bins, pairs.len());
    eprintln!("    max |delta| = {:.6}, epsilon = {:.6}", max_abs, epsilon);
    eprintln!("    N_persistent (lifetime > epsilon) = {}", n_persistent);

    // Print persistence diagram.
    println!("# {} persistence diagram", label);
    println!("birth,death,lifetime,persistent");
    for pair in &pairs {
        println!(
            "{:.8},{:.8},{:.8},{}",
            pair.birth,
            pair.death,
            pair.lifetime(),
            if pair.lifetime() > epsilon { 1 } else { 0 }
        );
    }

    // Print stacked profile for reference.
    eprintln!("    Stacked profile (valid bins):");
    for (x, d) in &valid {
        eprintln!("      x={:.4}  delta={:+.6}", x, d);
    }

    n_persistent
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("=== H3: Topological Persistence of Residual Level Set ===");
    eprintln!();

    // Analyze the main stacked profile.
    let (x_grid, delta, n_contrib) = load_stacked_csv(&cli.stacked_csv)?;
    let n_main = analyze_profile(
        "main (D=16)",
        &x_grid,
        &delta,
        &n_contrib,
        cli.min_per_bin,
        cli.epsilon_frac,
    );

    eprintln!();
    eprintln!("=== Verdict ===");
    eprintln!("  N_persistent = {}", n_main);
    if n_main <= 3 {
        eprintln!("  Consistent with baryonic origin (2-3 features expected).");
        eprintln!("  ZD hypothesis gains no support from topological persistence.");
        eprintln!("  FALSIFICATION: N_persistent <= 3 -> REJECT ZD topological signature.");
    } else if n_main >= 5 {
        eprintln!("  Excess topological features detected!");
        eprintln!("  Requires further investigation: possible ZD structure or unmodeled systematics.");
    } else {
        eprintln!("  Borderline (4 features): inconclusive.");
    }

    // Mass-bin stability test if sweep directory provided.
    if let Some(sweep_dir) = &cli.sweep_dir {
        eprintln!();
        eprintln!("=== Mass-bin Stability Test ===");
        let mut counts: Vec<(String, usize)> = Vec::new();

        for entry in std::fs::read_dir(sweep_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("csv") {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let (xg, dl, nc) = load_stacked_csv(&path)?;
            let count = analyze_profile(
                &name,
                &xg,
                &dl,
                &nc,
                cli.min_per_bin,
                cli.epsilon_frac,
            );
            counts.push((name, count));
        }

        if counts.len() >= 2 {
            let mean = counts.iter().map(|(_, c)| *c as f64).sum::<f64>() / counts.len() as f64;
            let var = counts
                .iter()
                .map(|(_, c)| (*c as f64 - mean).powi(2))
                .sum::<f64>()
                / counts.len() as f64;
            let std = var.sqrt();

            eprintln!();
            eprintln!("  Stability summary ({} profiles):", counts.len());
            for (name, c) in &counts {
                eprintln!("    {}: N_persistent = {}", name, c);
            }
            eprintln!("  Mean = {:.2}, Std = {:.2}", mean, std);
            if std < 1.5 {
                eprintln!("  Stable across mass bins (std < 1.5).");
            } else {
                eprintln!("  Unstable across mass bins (std >= 1.5) -- mass-dependent structure.");
            }
        }
    }

    Ok(())
}
