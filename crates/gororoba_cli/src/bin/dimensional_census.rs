//! dimensional-census: APT verification across the full Cayley-Dickson dimensional ladder.
//!
//! Orchestrates Anti-Diagonal Parity Theorem census from dim=4 to dim=4096,
//! classifying every triangle as pure or mixed and verifying the 1:3 ratio,
//! fiber symmetry, and frustration convergence at each dimension.
//!
//! Outputs:
//!   - Per-dimension summary CSV with component/triangle/ratio statistics
//!   - Per-component detail CSV (optional, with --details)
//!   - Frustration convergence table
//!
//! Usage:
//!   dimensional-census --dims 16,32,64              # specific dimensions
//!   dimensional-census --fast                        # TIER 0+1 only (dims 4-32)
//!   dimensional-census --slow                        # include TIER 2 (dims 4-256)
//!   dimensional-census --all                         # all CPU-feasible (dims 4-512)
//!   dimensional-census --dims 128 --monte-carlo 100000  # Monte Carlo at dim 128

use algebra_core::analysis::boxkites::motif_components_for_cross_assessors;
use algebra_core::cd_basis_mul_sign;
use clap::Parser;
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

type Node = (usize, usize);
type EdgeSet = HashSet<(Node, Node)>;

#[derive(Parser)]
#[command(name = "dimensional-census")]
#[command(about = "APT verification across the Cayley-Dickson dimensional ladder")]
struct Args {
    /// Comma-separated dimensions to census (powers of 2, >= 4)
    #[arg(short, long, value_delimiter = ',')]
    dims: Vec<usize>,

    /// TIER 0+1 only: dims 4, 8, 16, 32
    #[arg(long)]
    fast: bool,

    /// Include TIER 2: dims 4-256
    #[arg(long)]
    slow: bool,

    /// All CPU-feasible: dims 4-512
    #[arg(long)]
    all: bool,

    /// Output directory for CSV files
    #[arg(short, long, default_value = "data/csv")]
    output_dir: String,

    /// Write per-component detail CSV
    #[arg(long)]
    details: bool,

    /// Use Monte Carlo sampling instead of exhaustive enumeration
    /// (value = number of samples per dimension)
    #[arg(long, default_value = "0")]
    monte_carlo: usize,

    /// RNG seed for Monte Carlo sampling
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn psi(dim: usize, i: usize, j: usize) -> u8 {
    if cd_basis_mul_sign(dim, i, j) == 1 {
        0
    } else {
        1
    }
}

/// Per-dimension summary row
struct DimResult {
    dim: usize,
    n_components: usize,
    n_nodes: usize,
    n_triangles: usize,
    pure_count: usize,
    mixed_count: usize,
    pure_ratio: f64,
    fiber_00: usize,
    fiber_01: usize,
    fiber_10: usize,
    fiber_11: usize,
    method: String,
    elapsed_ms: u128,
}

/// Per-component detail row
struct ComponentResult {
    dim: usize,
    comp_id: usize,
    n_nodes: usize,
    n_triangles: usize,
    pure_count: usize,
    mixed_count: usize,
    pure_ratio: f64,
}

/// Exhaustive APT census for a single dimension
fn exhaustive_census(dim: usize) -> (DimResult, Vec<ComponentResult>) {
    let start = Instant::now();
    let components = motif_components_for_cross_assessors(dim);
    let n_nodes: usize = components.iter().map(|c| c.nodes.len()).sum();

    let mut total_triangles = 0usize;
    let mut total_pure = 0usize;
    let mut fiber_00 = 0usize;
    let mut fiber_01 = 0usize;
    let mut fiber_10 = 0usize;
    let mut fiber_11 = 0usize;

    let mut comp_results = Vec::new();

    for (comp_id, comp) in components.iter().enumerate() {
        let nodes: Vec<_> = comp.nodes.iter().collect();
        let mut comp_tri = 0usize;
        let mut comp_pure = 0usize;

        // Build edge set for O(1) lookup (APT 1:3 ratio applies to graph triangles only)
        let edge_set: EdgeSet = comp.edges.iter().copied().collect();
        let has_edge = |u: Node, v: Node| -> bool {
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            edge_set.contains(&(a, b))
        };

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                for k in (j + 1)..nodes.len() {
                    let &ni = nodes[i];
                    let &nj = nodes[j];
                    let &nk = nodes[k];

                    // Only count graph triangles (all 3 edges present)
                    if !has_edge(ni, nj) || !has_edge(ni, nk) || !has_edge(nj, nk) {
                        continue;
                    }

                    let (ai, bi) = ni;
                    let (aj, bj) = nj;
                    let (ak, bk) = nk;

                    // Anti-diagonal parity: eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
                    let eta_ij = psi(dim, ai, bj) ^ psi(dim, bi, aj);
                    let eta_ik = psi(dim, ai, bk) ^ psi(dim, bi, ak);
                    let eta_jk = psi(dim, aj, bk) ^ psi(dim, bj, ak);

                    comp_tri += 1;

                    if eta_ij == eta_ik && eta_ik == eta_jk {
                        comp_pure += 1;
                        fiber_00 += 1;
                    } else {
                        let f0 = eta_ij ^ eta_jk;
                        let f1 = eta_jk ^ eta_ik;
                        let fiber_idx = (f0 << 1) | f1;
                        match fiber_idx {
                            1 => fiber_01 += 1,
                            2 => fiber_10 += 1,
                            3 => fiber_11 += 1,
                            _ => {}
                        }
                    }
                }
            }
        }

        let comp_mixed = comp_tri - comp_pure;
        let comp_ratio = if comp_tri > 0 {
            comp_pure as f64 / comp_tri as f64
        } else {
            0.0
        };

        comp_results.push(ComponentResult {
            dim,
            comp_id,
            n_nodes: nodes.len(),
            n_triangles: comp_tri,
            pure_count: comp_pure,
            mixed_count: comp_mixed,
            pure_ratio: comp_ratio,
        });

        total_triangles += comp_tri;
        total_pure += comp_pure;
    }

    let total_mixed = total_triangles - total_pure;
    let pure_ratio = if total_triangles > 0 {
        total_pure as f64 / total_triangles as f64
    } else {
        0.0
    };

    let elapsed_ms = start.elapsed().as_millis();

    let result = DimResult {
        dim,
        n_components: components.len(),
        n_nodes,
        n_triangles: total_triangles,
        pure_count: total_pure,
        mixed_count: total_mixed,
        pure_ratio,
        fiber_00,
        fiber_01,
        fiber_10,
        fiber_11,
        method: "exhaustive".to_string(),
        elapsed_ms,
    };

    (result, comp_results)
}

/// Monte Carlo APT census for a single dimension.
/// Uses rejection sampling: random node triples, keep only graph triangles.
fn monte_carlo_census(
    dim: usize,
    n_samples: usize,
    seed: u64,
) -> (DimResult, Vec<ComponentResult>) {
    let start = Instant::now();
    let components = motif_components_for_cross_assessors(dim);
    let n_nodes_total: usize = components.iter().map(|c| c.nodes.len()).sum();

    if n_nodes_total < 3 {
        let elapsed_ms = start.elapsed().as_millis();
        return (
            DimResult {
                dim,
                n_components: components.len(),
                n_nodes: n_nodes_total,
                n_triangles: 0,
                pure_count: 0,
                mixed_count: 0,
                pure_ratio: 0.0,
                fiber_00: 0,
                fiber_01: 0,
                fiber_10: 0,
                fiber_11: 0,
                method: format!("monte_carlo_{}", n_samples),
                elapsed_ms,
            },
            Vec::new(),
        );
    }

    // Build per-component node lists and edge sets for rejection sampling
    let comp_data: Vec<(Vec<Node>, EdgeSet)> = components
        .iter()
        .filter(|c| c.nodes.len() >= 3)
        .map(|c| {
            let nodes: Vec<Node> = c.nodes.iter().copied().collect();
            let edges: EdgeSet = c.edges.iter().copied().collect();
            (nodes, edges)
        })
        .collect();

    let mut rng_state = seed;
    let next_rng = |state: &mut u64| -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let z = *state ^ (*state >> 30);
        let z_mul = z.wrapping_mul(0xbf58476d1ce4e5b9);
        z_mul ^ (z_mul >> 27)
    };

    let mut pure_count = 0usize;
    let mut mixed_count = 0usize;
    let mut fiber_00 = 0usize;
    let mut fiber_01 = 0usize;
    let mut fiber_10 = 0usize;
    let mut fiber_11 = 0usize;
    let mut accepted = 0usize;

    let n_comps = comp_data.len();
    let max_attempts = n_samples * 100; // avoid infinite loop if triangle density is low
    let mut attempts = 0usize;

    while accepted < n_samples && attempts < max_attempts {
        attempts += 1;
        // Pick a random component (weighted by node count would be better, but uniform is simpler)
        let comp_idx = (next_rng(&mut rng_state) as usize) % n_comps;
        let (ref nodes, ref edge_set) = comp_data[comp_idx];
        let n = nodes.len();
        if n < 3 {
            continue;
        }

        let i = (next_rng(&mut rng_state) as usize) % n;
        let mut j = (next_rng(&mut rng_state) as usize) % n;
        while j == i {
            j = (next_rng(&mut rng_state) as usize) % n;
        }
        let mut k = (next_rng(&mut rng_state) as usize) % n;
        while k == i || k == j {
            k = (next_rng(&mut rng_state) as usize) % n;
        }

        let ni = nodes[i];
        let nj = nodes[j];
        let nk = nodes[k];

        // Check all 3 edges exist (rejection sampling for graph triangles)
        let has_edge = |u: (usize, usize), v: (usize, usize)| -> bool {
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            edge_set.contains(&(a, b))
        };
        if !has_edge(ni, nj) || !has_edge(ni, nk) || !has_edge(nj, nk) {
            continue;
        }

        let (ai, bi) = ni;
        let (aj, bj) = nj;
        let (ak, bk) = nk;

        // Anti-diagonal parity: eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
        let eta_ij = psi(dim, ai, bj) ^ psi(dim, bi, aj);
        let eta_ik = psi(dim, ai, bk) ^ psi(dim, bi, ak);
        let eta_jk = psi(dim, aj, bk) ^ psi(dim, bj, ak);

        accepted += 1;

        if eta_ij == eta_ik && eta_ik == eta_jk {
            pure_count += 1;
            fiber_00 += 1;
        } else {
            mixed_count += 1;
            let f0 = eta_ij ^ eta_jk;
            let f1 = eta_jk ^ eta_ik;
            let fiber_idx = (f0 << 1) | f1;
            match fiber_idx {
                1 => fiber_01 += 1,
                2 => fiber_10 += 1,
                3 => fiber_11 += 1,
                _ => {}
            }
        }
    }

    let pure_ratio = if accepted > 0 {
        pure_count as f64 / accepted as f64
    } else {
        0.0
    };
    let elapsed_ms = start.elapsed().as_millis();

    let result = DimResult {
        dim,
        n_components: components.len(),
        n_nodes: n_nodes_total,
        n_triangles: accepted,
        pure_count,
        mixed_count,
        pure_ratio,
        fiber_00,
        fiber_01,
        fiber_10,
        fiber_11,
        method: format!("monte_carlo_{}_accepted_{}", n_samples, accepted),
        elapsed_ms,
    };

    (result, Vec::new())
}

fn write_summary_csv(path: &Path, results: &[DimResult]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "dim,n_components,n_nodes,n_triangles,pure_count,mixed_count,pure_ratio,fiber_00,fiber_01,fiber_10,fiber_11,method,elapsed_ms"
    )?;
    for r in results {
        writeln!(
            f,
            "{},{},{},{},{},{},{:.6},{},{},{},{},{},{}",
            r.dim,
            r.n_components,
            r.n_nodes,
            r.n_triangles,
            r.pure_count,
            r.mixed_count,
            r.pure_ratio,
            r.fiber_00,
            r.fiber_01,
            r.fiber_10,
            r.fiber_11,
            r.method,
            r.elapsed_ms
        )?;
    }
    Ok(())
}

fn write_detail_csv(path: &Path, results: &[ComponentResult]) -> std::io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "dim,comp_id,n_nodes,n_triangles,pure_count,mixed_count,pure_ratio"
    )?;
    for r in results {
        writeln!(
            f,
            "{},{},{},{},{},{},{:.6}",
            r.dim, r.comp_id, r.n_nodes, r.n_triangles, r.pure_count, r.mixed_count, r.pure_ratio
        )?;
    }
    Ok(())
}

fn main() {
    let args = Args::parse();

    let dims: Vec<usize> = if args.fast {
        vec![4, 8, 16, 32]
    } else if args.slow {
        vec![4, 8, 16, 32, 64, 128, 256]
    } else if args.all {
        vec![4, 8, 16, 32, 64, 128, 256, 512]
    } else if !args.dims.is_empty() {
        args.dims.clone()
    } else {
        eprintln!("No dimensions specified. Use --fast, --slow, --all, or --dims N,M,...");
        std::process::exit(1);
    };

    // Validate dimensions
    for &dim in &dims {
        if !dim.is_power_of_two() || dim < 4 {
            eprintln!("Error: dimension {} must be a power of 2 >= 4", dim);
            std::process::exit(1);
        }
    }

    // Create output directory
    fs::create_dir_all(&args.output_dir).unwrap_or_else(|e| {
        eprintln!("Cannot create output directory {}: {}", args.output_dir, e);
        std::process::exit(1);
    });

    eprintln!("=== Dimensional APT Census ===");
    eprintln!("Dimensions: {:?}", dims);
    if args.monte_carlo > 0 {
        eprintln!(
            "Method: Monte Carlo ({} samples, seed={})",
            args.monte_carlo, args.seed
        );
    } else {
        eprintln!("Method: exhaustive");
    }
    eprintln!();

    let mut summary_results = Vec::new();
    let mut all_details = Vec::new();

    for &dim in &dims {
        eprint!("dim={:<6}", dim);

        let (result, details) = if args.monte_carlo > 0 {
            monte_carlo_census(dim, args.monte_carlo, args.seed)
        } else {
            exhaustive_census(dim)
        };

        eprintln!(
            " comps={:<4} nodes={:<6} tri={:<10} pure={:<10} ratio={:.6} elapsed={}ms",
            result.n_components,
            result.n_nodes,
            result.n_triangles,
            result.pure_count,
            result.pure_ratio,
            result.elapsed_ms
        );

        if args.details && !details.is_empty() {
            all_details.extend(details);
        }

        summary_results.push(result);
    }

    // Write summary CSV
    let summary_path = Path::new(&args.output_dir).join("apt_dimensional_census_summary.csv");
    write_summary_csv(&summary_path, &summary_results).unwrap_or_else(|e| {
        eprintln!("Error writing summary CSV: {}", e);
        std::process::exit(1);
    });
    eprintln!("\nSummary CSV: {}", summary_path.display());

    // Write details CSV if requested
    if args.details && !all_details.is_empty() {
        let detail_path = Path::new(&args.output_dir).join("apt_dimensional_census_details.csv");
        write_detail_csv(&detail_path, &all_details).unwrap_or_else(|e| {
            eprintln!("Error writing detail CSV: {}", e);
            std::process::exit(1);
        });
        eprintln!("Detail CSV:  {}", detail_path.display());
    }

    // Print frustration-like convergence table
    eprintln!("\n=== Pure Ratio Convergence ===");
    eprintln!(
        "{:<8} {:<12} {:<12} {:<12}",
        "dim", "pure_ratio", "deviation", "method"
    );
    for r in &summary_results {
        let deviation = r.pure_ratio - 0.25;
        eprintln!(
            "{:<8} {:<12.6} {:<12.6} {:<12}",
            r.dim, r.pure_ratio, deviation, r.method
        );
    }

    eprintln!("\nDone.");
}
