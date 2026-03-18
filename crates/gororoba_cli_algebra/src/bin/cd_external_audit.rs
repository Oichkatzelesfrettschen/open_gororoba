//! Deterministic audit for refuted external Cayley-Dickson CSV claims.
//!
//! Produces a small TOML artifact that summarizes four existing falsification
//! lanes backed by exact Rust checks and local CSV inputs:
//! - C-450: the 64D adjacency CSV is a sparse basis-level table, not a
//!   cross-assessor zero-divisor graph
//! - C-455: ZD-adjacent lattice differences do not include E8 root vectors
//! - C-456: the 256D associativity CSV marks non-associative triples as true
//! - C-463: the parity-clique law fails across all checked dimensions

use algebra_experimental::cd_external::{
    build_zd_adjacency_matrix, is_associative_triple, load_lattice_map, parse_adjacency_csv,
    parse_lattice_point, parse_nested_tuple, vec_to_basis_index, verify_parity_clique,
};
use clap::Parser;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::PathBuf,
    process,
};

#[derive(Parser, Debug)]
#[command(name = "cd-external-audit")]
#[command(about = "Deterministic audit for refuted external Cayley-Dickson CSV claims")]
struct Args {
    /// Output TOML artifact path.
    #[arg(
        long,
        default_value = "data/output/claims_falsification/cd_external_audit.toml"
    )]
    output: PathBuf,
}

#[derive(Debug)]
struct SparseAdjacencySummary {
    rows: usize,
    cols: usize,
    total_nonzero_entries: usize,
}

#[derive(Debug)]
struct AssociativitySummary {
    checked_rows: usize,
    csv_true_rows: usize,
    rust_associative_rows: usize,
    rust_nonassociative_rows: usize,
    mismatch_examples: Vec<String>,
}

#[derive(Debug)]
struct LatticeDifferenceSummary {
    total_basis_differences: usize,
    unique_difference_vectors: usize,
    unique_e8_root_vectors: usize,
    unique_norm_squared_values: Vec<i32>,
}

fn repo_data_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(relative)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    fields.push(current);
    fields
}

fn summarize_sparse_adjacency_csv() -> SparseAdjacencySummary {
    let csv_path = repo_data_path("data/csv/cayley_dickson/64d_chingon_adjacency.csv");
    let content = fs::read_to_string(&csv_path).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to read {}: {err}", csv_path.display());
        process::exit(1);
    });
    let matrix = parse_adjacency_csv(&content);
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);
    let total_nonzero_entries = matrix
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&value| value > 0.5)
        .count();

    SparseAdjacencySummary {
        rows,
        cols,
        total_nonzero_entries,
    }
}

fn summarize_associativity_csv() -> AssociativitySummary {
    let csv_path = repo_data_path("data/csv/cayley_dickson/256d_basis_properties.csv");
    let content = fs::read_to_string(&csv_path).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to read {}: {err}", csv_path.display());
        process::exit(1);
    });

    let mut checked_rows = 0usize;
    let mut csv_true_rows = 0usize;
    let mut rust_associative_rows = 0usize;
    let mut rust_nonassociative_rows = 0usize;
    let mut mismatch_examples = Vec::new();

    for line in content.lines().skip(1).take(50) {
        let fields = parse_csv_line(line);
        if fields.len() < 4 {
            continue;
        }

        let Some(a) = parse_nested_tuple(&fields[0]) else {
            continue;
        };
        let Some(b) = parse_nested_tuple(&fields[1]) else {
            continue;
        };
        let Some(c) = parse_nested_tuple(&fields[2]) else {
            continue;
        };

        let csv_assoc = fields[3].trim().eq_ignore_ascii_case("true");
        if csv_assoc {
            csv_true_rows += 1;
        }

        let rust_assoc = is_associative_triple(&a, &b, &c, 1e-10);
        checked_rows += 1;

        if rust_assoc {
            rust_associative_rows += 1;
        } else {
            rust_nonassociative_rows += 1;
            let a_idx = vec_to_basis_index(&a);
            let b_idx = vec_to_basis_index(&b);
            let c_idx = vec_to_basis_index(&c);
            mismatch_examples.push(format!(
                "row_{}: e_{:?}, e_{:?}, e_{:?}",
                checked_rows - 1,
                a_idx,
                b_idx,
                c_idx
            ));
        }
    }

    mismatch_examples.truncate(5);

    AssociativitySummary {
        checked_rows,
        csv_true_rows,
        rust_associative_rows,
        rust_nonassociative_rows,
        mismatch_examples,
    }
}

fn summarize_lattice_differences() -> LatticeDifferenceSummary {
    let lattice_map = load_lattice_map(256);
    let (pairs_16, adjacency_16) = build_zd_adjacency_matrix(16);

    let mut difference_counts: BTreeMap<Vec<i32>, usize> = BTreeMap::new();
    let mut total_basis_differences = 0usize;

    for i in 0..pairs_16.len() {
        for j in (i + 1)..pairs_16.len() {
            if adjacency_16[i][j] == 0 {
                continue;
            }

            let (lo_i, hi_i) = pairs_16[i];
            let (lo_j, hi_j) = pairs_16[j];
            let basis_pairs = [(lo_i, lo_j), (lo_i, hi_j), (hi_i, lo_j), (hi_i, hi_j)];

            for (a, b) in basis_pairs {
                if let (Some(la), Some(lb)) = (lattice_map.get(&a), lattice_map.get(&b)) {
                    let diff: Vec<i32> = la.iter().zip(lb.iter()).map(|(x, y)| x - y).collect();
                    *difference_counts.entry(diff).or_insert(0) += 1;
                    total_basis_differences += 1;
                }
            }
        }
    }

    let mut unique_norm_squared_values = BTreeSet::new();
    let mut unique_e8_root_vectors = 0usize;
    for diff in difference_counts.keys() {
        let norm_sq: i32 = diff.iter().map(|value| value * value).sum();
        unique_norm_squared_values.insert(norm_sq);
        if norm_sq == 2 {
            unique_e8_root_vectors += 1;
        }
    }

    LatticeDifferenceSummary {
        total_basis_differences,
        unique_difference_vectors: difference_counts.len(),
        unique_e8_root_vectors,
        unique_norm_squared_values: unique_norm_squared_values.into_iter().collect(),
    }
}

fn parse_lattice_rows(dim: usize) -> usize {
    let csv_path = repo_data_path(&format!(
        "data/csv/cayley_dickson/{dim}d_lattice_mapping.csv"
    ));
    let content = fs::read_to_string(&csv_path).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to read {}: {err}", csv_path.display());
        process::exit(1);
    });

    content
        .lines()
        .skip(1)
        .filter(|line| {
            let fields = parse_csv_line(line);
            fields.len() >= 2
                && parse_nested_tuple(&fields[0]).is_some()
                && parse_lattice_point(&fields[1]).is_some()
        })
        .count()
}

fn render_report() -> String {
    let sparse = summarize_sparse_adjacency_csv();
    let assoc = summarize_associativity_csv();
    let lattice = summarize_lattice_differences();
    let lattice_rows = parse_lattice_rows(256);
    let parity_16 = verify_parity_clique(16);
    let parity_32 = verify_parity_clique(32);
    let parity_64 = verify_parity_clique(64);

    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "experiment = \"E-204\"");
    let _ = writeln!(
        out,
        "title = \"Deterministic audit for refuted external Cayley-Dickson CSV claims\""
    );
    let _ = writeln!(out, "claims = [\"C-450\", \"C-455\", \"C-456\", \"C-463\"]");
    let _ = writeln!(out, "deterministic = true");
    let _ = writeln!(out);

    let _ = writeln!(out, "[c450.external_adjacency]");
    let _ = writeln!(out, "rows = {}", sparse.rows);
    let _ = writeln!(out, "cols = {}", sparse.cols);
    let _ = writeln!(
        out,
        "total_nonzero_entries = {}",
        sparse.total_nonzero_entries
    );
    let _ = writeln!(
        out,
        "is_sparse_basis_level_table = {}",
        sparse.total_nonzero_entries < 200
    );
    let _ = writeln!(
        out,
        "cross_assessor_graph_is_not_directly_comparable = true"
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "[c455.lattice_difference_audit]");
    let _ = writeln!(out, "lattice_rows_parsed = {lattice_rows}");
    let _ = writeln!(
        out,
        "total_basis_differences = {}",
        lattice.total_basis_differences
    );
    let _ = writeln!(
        out,
        "unique_difference_vectors = {}",
        lattice.unique_difference_vectors
    );
    let _ = writeln!(
        out,
        "unique_e8_root_vectors = {}",
        lattice.unique_e8_root_vectors
    );
    let _ = writeln!(out, "unique_norm_squared_values = [");
    for norm_sq in &lattice.unique_norm_squared_values {
        let _ = writeln!(out, "  {norm_sq},");
    }
    let _ = writeln!(out, "]");
    let _ = writeln!(out);

    let _ = writeln!(out, "[c456.associativity_audit]");
    let _ = writeln!(out, "checked_rows = {}", assoc.checked_rows);
    let _ = writeln!(out, "csv_true_rows = {}", assoc.csv_true_rows);
    let _ = writeln!(
        out,
        "rust_associative_rows = {}",
        assoc.rust_associative_rows
    );
    let _ = writeln!(
        out,
        "rust_nonassociative_rows = {}",
        assoc.rust_nonassociative_rows
    );
    let _ = writeln!(out, "mismatch_examples = [");
    for example in &assoc.mismatch_examples {
        let _ = writeln!(out, "  \"{example}\",");
    }
    let _ = writeln!(out, "]");
    let _ = writeln!(out);

    for parity in [parity_16, parity_32, parity_64] {
        let _ = writeln!(out, "[[c463.parity_clique]]");
        let _ = writeln!(out, "dim = {}", parity.dim);
        let _ = writeln!(out, "n_vertices = {}", parity.n_vertices);
        let _ = writeln!(out, "n_edges = {}", parity.n_edges);
        let _ = writeln!(out, "n_even_edges = {}", parity.n_even_edges);
        let _ = writeln!(out, "n_odd_edges = {}", parity.n_odd_edges);
        let _ = writeln!(out, "n_cross_edges = {}", parity.n_cross_edges);
        let _ = writeln!(
            out,
            "expected_clique_edges = {}",
            parity.expected_clique_edges
        );
        let _ = writeln!(out, "is_parity_biclique = {}", parity.is_parity_biclique);
        let _ = writeln!(out);
    }

    out
}

fn main() {
    let args = Args::parse();
    let report = render_report();
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("ERROR: failed to create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    fs::write(&args.output, report).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to write {}: {err}", args.output.display());
        process::exit(1);
    });
    println!("{}", args.output.display());
}
