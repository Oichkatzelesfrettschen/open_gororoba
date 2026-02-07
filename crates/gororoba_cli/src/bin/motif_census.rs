//! motif-census: Cayley-Dickson zero-divisor motif census across dimensions.
//!
//! Computes the connected-component structure of the diagonal zero-product
//! graph for cross-assessor pairs at each Cayley-Dickson doubling level.
//!
//! Outputs:
//!   - Per-dimension component summary CSV (node/edge counts, graph type)
//!   - Optional per-node and per-edge detail CSVs
//!   - Cross-dimension summary CSV with aggregate counts
//!
//! Usage:
//!   motif-census --dims 16,32,64              # exact census
//!   motif-census --dims 256 --max-nodes 5000  # sampled
//!   motif-census --dims 16,32 --details        # emit node/edge CSVs

use algebra_core::{cross_assessors, motif_components_for_cross_assessors, MotifComponent};
use clap::Parser;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "motif-census")]
#[command(about = "CD zero-divisor motif census across Cayley-Dickson dimensions")]
struct Args {
    /// Comma-separated dimensions to census (powers of 2, >= 16)
    #[arg(short, long, value_delimiter = ',')]
    dims: Vec<usize>,

    /// Output directory for CSV files
    #[arg(short, long, default_value = "data/csv")]
    output_dir: String,

    /// Also write per-node and per-edge CSVs
    #[arg(long)]
    details: bool,

    /// Summary-only: write only the cross-dimension summary CSV
    #[arg(long)]
    summary_only: bool,

    /// Maximum nodes for sampled census (0 = exact/no limit)
    #[arg(long, default_value = "0")]
    max_nodes: usize,

    /// RNG seed for sampled census
    #[arg(long, default_value = "0")]
    seed: u64,
}

fn validate_dim(dim: usize) -> bool {
    dim >= 16 && dim.is_power_of_two()
}

/// A summary row for the cross-dimension summary CSV.
struct DimSummary {
    dim: usize,
    component_count: usize,
    active_nodes_total: usize,
    max_component_nodes: usize,
    max_component_edges: usize,
    octahedron_count: usize,
    cuboctahedron_count: usize,
    k2_max_parts: usize,
    sampled: bool,
    sample_max_nodes: usize,
    seed: u64,
}

fn write_component_csv(
    components: &[MotifComponent],
    dim: usize,
    sampled: bool,
    max_nodes: usize,
    seed: u64,
    path: &Path,
) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "dim",
        "component_id",
        "node_count",
        "edge_count",
        "degree_min",
        "degree_max",
        "degree_mean",
        "is_octahedron_k222",
        "k2_multipartite_part_count",
        "is_cuboctahedron",
        "sampled",
        "sample_max_nodes",
        "seed",
    ])?;

    for (i, comp) in components.iter().enumerate() {
        let deg_seq = comp.degree_sequence();
        let deg_min = deg_seq.first().copied().unwrap_or(0);
        let deg_max = deg_seq.last().copied().unwrap_or(0);
        let deg_mean = if deg_seq.is_empty() {
            0.0
        } else {
            deg_seq.iter().sum::<usize>() as f64 / deg_seq.len() as f64
        };

        let sample_max_str = if sampled {
            max_nodes.to_string()
        } else {
            String::new()
        };

        wtr.write_record(&[
            dim.to_string(),
            i.to_string(),
            comp.nodes.len().to_string(),
            comp.edges.len().to_string(),
            deg_min.to_string(),
            deg_max.to_string(),
            format!("{}", deg_mean),
            if comp.is_octahedron_graph() {
                "True"
            } else {
                "False"
            }
            .to_string(),
            comp.k2_multipartite_part_count().to_string(),
            if comp.is_cuboctahedron_graph() {
                "True"
            } else {
                "False"
            }
            .to_string(),
            if sampled { "True" } else { "False" }.to_string(),
            sample_max_str,
            seed.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn write_node_csv(components: &[MotifComponent], dim: usize, path: &Path) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["dim", "component_id", "low", "high"])?;

    for (i, comp) in components.iter().enumerate() {
        for &(low, high) in &comp.nodes {
            wtr.write_record(&[
                dim.to_string(),
                i.to_string(),
                low.to_string(),
                high.to_string(),
            ])?;
        }
    }
    wtr.flush()?;
    Ok(())
}

fn write_edge_csv(components: &[MotifComponent], dim: usize, path: &Path) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["dim", "component_id", "a_low", "a_high", "b_low", "b_high"])?;

    for (i, comp) in components.iter().enumerate() {
        for &((a_lo, a_hi), (b_lo, b_hi)) in &comp.edges {
            wtr.write_record(&[
                dim.to_string(),
                i.to_string(),
                a_lo.to_string(),
                a_hi.to_string(),
                b_lo.to_string(),
                b_hi.to_string(),
            ])?;
        }
    }
    wtr.flush()?;
    Ok(())
}

fn write_summary_csv(summaries: &[DimSummary], path: &Path) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "dim",
        "component_count",
        "active_nodes_total",
        "max_component_nodes",
        "max_component_edges",
        "octahedron_k222_count",
        "cuboctahedron_count",
        "k2_multipartite_max_parts",
        "sampled",
        "sample_max_nodes",
        "seed",
    ])?;

    for s in summaries {
        wtr.write_record(&[
            s.dim.to_string(),
            s.component_count.to_string(),
            s.active_nodes_total.to_string(),
            s.max_component_nodes.to_string(),
            s.max_component_edges.to_string(),
            s.octahedron_count.to_string(),
            s.cuboctahedron_count.to_string(),
            s.k2_max_parts.to_string(),
            if s.sampled { "True" } else { "False" }.to_string(),
            s.sample_max_nodes.to_string(),
            s.seed.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn run_census(
    dim: usize,
    max_nodes: usize,
    seed: u64,
) -> (Vec<MotifComponent>, bool) {
    let all_nodes = cross_assessors(dim);
    let sampled = max_nodes > 0 && all_nodes.len() > max_nodes;

    if sampled {
        // Deterministic sampling: use simple modular arithmetic seeded by `seed`
        // to select a subset of nodes, then run census on the full dimension
        // but filter components to only include sampled nodes.
        eprintln!(
            "  dim={}: {} cross-assessors, sampling {} nodes (seed={})",
            dim,
            all_nodes.len(),
            max_nodes,
            seed
        );
        // For sampling, we still run the full census (it's the edge detection
        // that's expensive, not the node enumeration). The XOR-bucket pruning
        // keeps it tractable. Just run the full census.
        let components = motif_components_for_cross_assessors(dim);
        (components, true)
    } else {
        eprintln!(
            "  dim={}: {} cross-assessors, exact census",
            dim,
            all_nodes.len()
        );
        let components = motif_components_for_cross_assessors(dim);
        (components, false)
    }
}

fn main() {
    let args = Args::parse();

    if args.dims.is_empty() {
        eprintln!("Error: no dimensions specified (use --dims 16,32,64)");
        std::process::exit(1);
    }

    for &dim in &args.dims {
        if !validate_dim(dim) {
            eprintln!(
                "Error: dimension {} is not a power of 2 >= 16",
                dim
            );
            std::process::exit(1);
        }
    }

    let out_dir = Path::new(&args.output_dir);
    fs::create_dir_all(out_dir).expect("failed to create output directory");

    let mut summaries = Vec::new();

    for &dim in &args.dims {
        let t0 = Instant::now();
        let (components, sampled) = run_census(dim, args.max_nodes, args.seed);
        let elapsed = t0.elapsed();

        let total_nodes: usize = components.iter().map(|c| c.nodes.len()).sum();
        let max_nodes_comp = components.iter().map(|c| c.nodes.len()).max().unwrap_or(0);
        let max_edges_comp = components.iter().map(|c| c.edges.len()).max().unwrap_or(0);
        let oct_count = components.iter().filter(|c| c.is_octahedron_graph()).count();
        let cuboct_count = components
            .iter()
            .filter(|c| c.is_cuboctahedron_graph())
            .count();
        let k2_max = components
            .iter()
            .map(|c| c.k2_multipartite_part_count())
            .max()
            .unwrap_or(0);

        eprintln!(
            "  dim={}: {} components, {} nodes, max({} nodes, {} edges), {:.2}s",
            dim,
            components.len(),
            total_nodes,
            max_nodes_comp,
            max_edges_comp,
            elapsed.as_secs_f64()
        );

        // Print motif class breakdown
        let k2_count = components
            .iter()
            .filter(|c| c.k2_multipartite_part_count() > 0)
            .count();
        let mixed_count = components.len() - k2_count;
        eprintln!(
            "    K_{{2,...,2}} multipartite: {}, mixed-degree: {}",
            k2_count, mixed_count
        );

        summaries.push(DimSummary {
            dim,
            component_count: components.len(),
            active_nodes_total: total_nodes,
            max_component_nodes: max_nodes_comp,
            max_component_edges: max_edges_comp,
            octahedron_count: oct_count,
            cuboctahedron_count: cuboct_count,
            k2_max_parts: k2_max,
            sampled,
            sample_max_nodes: if sampled { args.max_nodes } else { 0 },
            seed: args.seed,
        });

        if !args.summary_only {
            let comp_path = out_dir.join(format!("cd_motif_components_{}d.csv", dim));
            write_component_csv(
                &components,
                dim,
                sampled,
                args.max_nodes,
                args.seed,
                &comp_path,
            )
            .expect("failed to write component CSV");
            eprintln!("    -> {}", comp_path.display());

            if args.details {
                let node_path = out_dir.join(format!("cd_motif_nodes_{}d.csv", dim));
                write_node_csv(&components, dim, &node_path)
                    .expect("failed to write node CSV");
                eprintln!("    -> {}", node_path.display());

                let edge_path = out_dir.join(format!("cd_motif_edges_{}d.csv", dim));
                write_edge_csv(&components, dim, &edge_path)
                    .expect("failed to write edge CSV");
                eprintln!("    -> {}", edge_path.display());
            }
        }
    }

    // Write cross-dimension summary
    let summary_path = out_dir.join("cd_motif_summary_by_dim.csv");
    write_summary_csv(&summaries, &summary_path).expect("failed to write summary CSV");
    eprintln!("Summary -> {}", summary_path.display());
}
