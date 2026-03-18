//! Deterministic audit for refuted legacy CSV claims.
//!
//! Produces a small TOML artifact that summarizes three existing falsification
//! lanes:
//! - C-020: legacy "ZD adjacency" matrices are same-parity masks, not algebra
//! - C-021: the 1024D lattice slice needs explicit exceptional exclusions
//! - C-543: the moonshine CSV mixes correct coefficients with trivial/log data

use algebra_analysis::{
    codebook::{
        LatticeVector, enumerate_lattice_by_predicate, is_in_lambda_256, is_in_lambda_512,
        is_in_lambda_1024, is_in_lambda_2048,
    },
    legacy_crossval::{commutativity_matrix, same_parity_adjacency},
};
use clap::Parser;
use std::{fmt::Write as _, fs, path::PathBuf, process};

#[derive(Parser, Debug)]
#[command(name = "legacy-crossval-audit")]
#[command(about = "Deterministic audit for legacy refuted CSV claims")]
struct Args {
    /// Output TOML artifact path.
    #[arg(
        long,
        default_value = "data/output/claims_falsification/legacy_crossval_audit.toml"
    )]
    output: PathBuf,
}

#[derive(Debug)]
struct AdjacencySummary {
    dim: usize,
    comm_edges: usize,
    parity_edges: usize,
    mismatch_edges: usize,
}

fn summarize_legacy_adjacency(dim: usize) -> AdjacencySummary {
    let comm = commutativity_matrix(dim);
    let parity = same_parity_adjacency(dim);
    let mut comm_edges = 0usize;
    let mut parity_edges = 0usize;
    let mut mismatch_edges = 0usize;

    for i in 1..dim {
        for j in 1..dim {
            if i == j {
                continue;
            }
            let comm_edge = comm[i][j];
            let parity_edge = parity[i][j] == 1;
            if comm_edge {
                comm_edges += 1;
            }
            if parity_edge {
                parity_edges += 1;
            }
            if comm_edge != parity_edge {
                mismatch_edges += 1;
            }
        }
    }

    AdjacencySummary {
        dim,
        comm_edges,
        parity_edges,
        mismatch_edges,
    }
}

fn disputed_lambda_1024_points() -> [LatticeVector; 2] {
    [[-1, 1, 1, 0, -1, 1, 0, 1], [-1, 1, 1, 0, -1, 1, 1, 0]]
}

fn moonshine_c10_relative_error() -> f64 {
    let expected = 22_312_779_956_505_600f64 / 1e9;
    let legacy = 23_123_279.479533825f64;
    ((legacy - expected) / expected).abs()
}

fn moonshine_first_eight_match() -> bool {
    let j_coeffs: [f64; 8] = [
        196_884.0,
        21_493_760.0,
        864_299_970.0,
        20_245_856_256.0,
        333_202_640_600.0,
        4_252_023_300_096.0,
        44_656_994_071_935.0,
        401_489_888_665_600.0,
    ];
    let legacy_scaled: [f64; 8] = [
        0.000196884,
        0.02149376,
        0.86429997,
        20.245856256,
        333.2026406,
        4252.023300096001,
        44656.994071935005,
        401490.88665600005,
    ];
    j_coeffs
        .iter()
        .zip(legacy_scaled.iter())
        .all(|(expected, legacy)| (((legacy - expected / 1e9) / (expected / 1e9)).abs()) < 1e-4)
}

fn moonshine_spin_foam_is_trivial_log() -> bool {
    let legacy_amplitudes: [f64; 10] = [
        0.0006931471805599454,
        0.0010986122886681095,
        0.001386294361119891,
        0.0016094379124340999,
        0.0017917594692280557,
        0.0019459101490553127,
        0.002079441541679837,
        0.0021972245773362186,
        0.0023025850929940467,
        0.0023978952727983695,
    ];
    legacy_amplitudes
        .iter()
        .enumerate()
        .all(|(idx, value)| (value - (((idx + 2) as f64).ln() / 1000.0)).abs() < 1e-15)
}

fn render_report() -> String {
    let adjacency_16 = summarize_legacy_adjacency(16);
    let adjacency_32 = summarize_legacy_adjacency(32);
    let adjacency_64 = summarize_legacy_adjacency(64);

    let lambda_256 = enumerate_lattice_by_predicate(is_in_lambda_256).len();
    let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512).len();
    let lambda_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024).len();
    let lambda_2048 = enumerate_lattice_by_predicate(is_in_lambda_2048).len();
    let disputed = disputed_lambda_1024_points();
    let disputed_in_2048 = disputed.iter().filter(|v| is_in_lambda_2048(v)).count();
    let disputed_in_1024 = disputed.iter().filter(|v| is_in_lambda_1024(v)).count();

    let c10_rel_err = moonshine_c10_relative_error();
    let first8_match = moonshine_first_eight_match();
    let trivial_log = moonshine_spin_foam_is_trivial_log();
    let griess_dim = 196_884u64;

    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "experiment = \"E-203\"");
    let _ = writeln!(
        out,
        "title = \"Legacy cross-validation audit for refuted algebraic CSV claims\""
    );
    let _ = writeln!(out, "claims = [\"C-020\", \"C-021\", \"C-543\"]");
    let _ = writeln!(out, "deterministic = true");
    let _ = writeln!(out);

    for summary in [adjacency_16, adjacency_32, adjacency_64] {
        let _ = writeln!(out, "[[c020.legacy_adjacency]]");
        let _ = writeln!(out, "dim = {}", summary.dim);
        let _ = writeln!(out, "comm_edges = {}", summary.comm_edges);
        let _ = writeln!(out, "parity_edges = {}", summary.parity_edges);
        let _ = writeln!(out, "mismatch_edges = {}", summary.mismatch_edges);
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "[c021.lambda_filtration]");
    let _ = writeln!(out, "lambda_256_count = {lambda_256}");
    let _ = writeln!(out, "lambda_512_count = {lambda_512}");
    let _ = writeln!(out, "lambda_1024_count = {lambda_1024}");
    let _ = writeln!(out, "lambda_2048_count = {lambda_2048}");
    let _ = writeln!(out, "disputed_singleton_count = {}", disputed.len());
    let _ = writeln!(
        out,
        "disputed_singletons_in_lambda_2048 = {disputed_in_2048}"
    );
    let _ = writeln!(
        out,
        "disputed_singletons_in_lambda_1024 = {disputed_in_1024}"
    );
    let _ = writeln!(out, "disputed_singletons = [");
    for vector in disputed {
        let rendered = vector
            .iter()
            .map(i8::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(out, "  [{rendered}],");
    }
    let _ = writeln!(out, "]");
    let _ = writeln!(out);

    let _ = writeln!(out, "[c543.moonshine]");
    let _ = writeln!(out, "first_eight_coefficients_match = {first8_match}");
    let _ = writeln!(out, "spin_foam_column_is_trivial_log = {trivial_log}");
    let _ = writeln!(out, "c10_relative_error = {:.12}", c10_rel_err);
    let _ = writeln!(out, "griess_dimension = {griess_dim}");
    let _ = writeln!(
        out,
        "griess_dimension_is_power_of_two = {}",
        griess_dim.is_power_of_two()
    );
    let _ = writeln!(out, "monster_path_summary = \"E8_to_Leech_to_Monster\"");
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
