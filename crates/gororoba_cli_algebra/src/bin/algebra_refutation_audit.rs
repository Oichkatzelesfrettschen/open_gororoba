//! Deterministic audit for four refuted algebraic-structural claims.
//!
//! This binary hardens the following claims with exact Rust checks:
//! - C-445: motif classes are not decided by a GF(2)-linear predicate
//! - C-466: multiplication coupling does not extend generically across bases
//! - C-518: the candidate-B associator sign does not classify pure triangles
//! - C-585: Singh's delta^2 = 3/8 is not reproduced generically for real
//!   trace-free J_3(O) elements

use algebra_analysis::{
    boxkites::{CrossPair, motif_components_for_cross_assessors},
    codebook::{EncodingDictionary, compute_multiplication_coupling, enumerate_lambda_256},
    projective_geometry::{
        PGPoint, component_xor_label, find_affine_class_predicate, find_linear_class_predicate,
        find_minimum_separating_degree,
    },
};
use cd_kernel::{cayley_dickson::cd_basis_mul_sign, mult_table::CdMultTable};
use clap::Parser;
use gororoba_algebra::construction::albert::AlbertElement;
use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write as _,
    fs,
    path::PathBuf,
    process,
};

#[derive(Parser, Debug)]
#[command(name = "algebra-refutation-audit")]
#[command(about = "Deterministic audit for four refuted algebraic-structural claims")]
struct Args {
    /// Output TOML artifact path.
    #[arg(
        long,
        default_value = "data/output/claims_falsification/algebra_refutation_audit.toml"
    )]
    output: PathBuf,
}

#[derive(Debug)]
struct C445Summary {
    labels: Vec<PGPoint>,
    class_sizes: Vec<usize>,
    linear_separator: Option<usize>,
    affine_separator: Option<(usize, u8)>,
    min_degree: Option<usize>,
    degree_results: Vec<(usize, bool)>,
}

#[derive(Debug)]
struct CouplingSummary {
    dim: usize,
    rank: usize,
    unsigned_consistent_count: usize,
    signed_consistent_count: usize,
    unsigned_consistent_bases: Vec<usize>,
    signed_consistent_bases: Vec<usize>,
}

#[derive(Debug)]
struct FiberSummary {
    total_triangles: usize,
    f_pp_pure: usize,
    f_pp_mixed: usize,
    f_other_pure: usize,
    f_other_mixed: usize,
    perfect_equivalence: bool,
    fiber_counts: BTreeMap<(i32, i32), [usize; 2]>,
}

#[derive(Debug)]
struct C585Summary {
    tracefree_diagonal_delta_squared: f64,
    sample_count: usize,
    min_delta_squared: f64,
    max_delta_squared: f64,
    mean_delta_squared: f64,
    closest_to_three_eighths: f64,
    closest_absolute_error: f64,
    exact_match_count: usize,
}

fn write_or_exit(path: &PathBuf, text: &str) {
    if let Some(parent) = path.parent()
        && let Err(err) = fs::create_dir_all(parent)
    {
        eprintln!("ERROR: failed to create {}: {err}", parent.display());
        process::exit(1);
    }
    if let Err(err) = fs::write(path, text) {
        eprintln!("ERROR: failed to write {}: {err}", path.display());
        process::exit(1);
    }
}

fn summarize_c445() -> C445Summary {
    let comps = motif_components_for_cross_assessors(32);
    let labels: Vec<PGPoint> = comps
        .iter()
        .map(|comp| component_xor_label(comp).expect("32D motif component should have PG label"))
        .collect();
    let classes: Vec<usize> = comps
        .iter()
        .map(|comp| if comp.edges.len() == 84 { 0 } else { 1 })
        .collect();
    let max_label = *labels.iter().max().expect("32D labels should be non-empty");
    let n_bits = (usize::BITS - max_label.leading_zeros()) as usize;

    let mut class_sizes = vec![0usize; 2];
    for class in &classes {
        class_sizes[*class] += 1;
    }

    let separating = find_minimum_separating_degree(32);

    C445Summary {
        labels,
        class_sizes,
        linear_separator: find_linear_class_predicate(
            &classes_to_labels(&classes, &comps),
            &classes,
            n_bits,
        ),
        affine_separator: find_affine_class_predicate(
            &classes_to_labels(&classes, &comps),
            &classes,
            n_bits,
        ),
        min_degree: separating.min_degree,
        degree_results: separating.degree_results,
    }
}

fn classes_to_labels(
    classes: &[usize],
    comps: &[algebra_analysis::boxkites::MotifComponent],
) -> Vec<PGPoint> {
    let _ = classes;
    comps
        .iter()
        .map(|comp| component_xor_label(comp).expect("motif component should carry a PG label"))
        .collect()
}

fn summarize_c466(dim: usize) -> CouplingSummary {
    let lambda = enumerate_lambda_256();
    assert!(
        lambda.len() >= dim,
        "Lambda_256 must contain at least {dim} vectors"
    );
    let pairs: Vec<_> = lambda[..dim]
        .iter()
        .enumerate()
        .map(|(index, &vector)| (index, vector))
        .collect();
    let dictionary = EncodingDictionary::try_from_pairs(dim, &pairs)
        .expect("first dim Lambda vectors should form a valid encoding dictionary");
    let table = CdMultTable::generate(dim);
    let coupling = compute_multiplication_coupling(&dictionary, &table);

    let unsigned_consistent_bases: Vec<usize> = coupling
        .results
        .iter()
        .filter(|result| result.unsigned_consistent)
        .map(|result| result.basis_index)
        .collect();
    let signed_consistent_bases: Vec<usize> = coupling
        .results
        .iter()
        .filter(|result| result.signed_consistent)
        .map(|result| result.basis_index)
        .collect();

    CouplingSummary {
        dim,
        rank: coupling.rank,
        unsigned_consistent_count: coupling.unsigned_consistent_count,
        signed_consistent_count: coupling.signed_consistent_count,
        unsigned_consistent_bases,
        signed_consistent_bases,
    }
}

fn edge_present(
    edge_set: &HashSet<(CrossPair, CrossPair)>,
    left: CrossPair,
    right: CrossPair,
) -> bool {
    let edge = if left < right {
        (left, right)
    } else {
        (right, left)
    };
    edge_set.contains(&edge)
}

fn assoc_sign(dim: usize, i: usize, j: usize, k: usize) -> i32 {
    let s_ij = cd_basis_mul_sign(dim, i, j);
    let s_ij_xor_k = cd_basis_mul_sign(dim, i ^ j, k);
    let s_jk = cd_basis_mul_sign(dim, j, k);
    let s_i_jk = cd_basis_mul_sign(dim, i, j ^ k);
    s_ij * s_ij_xor_k * s_jk * s_i_jk
}

fn triangle_is_pure(dim: usize, u: CrossPair, v: CrossPair, w: CrossPair) -> bool {
    let sigma_uv = cd_basis_mul_sign(dim, u.0, v.0) * cd_basis_mul_sign(dim, u.1, v.1);
    let sigma_vw = cd_basis_mul_sign(dim, v.0, w.0) * cd_basis_mul_sign(dim, v.1, w.1);
    let sigma_uw = cd_basis_mul_sign(dim, u.0, w.0) * cd_basis_mul_sign(dim, u.1, w.1);
    let same_count = [sigma_uv, sigma_vw, sigma_uw]
        .iter()
        .filter(|&&sigma| sigma == -1)
        .count();
    let parity_product = sigma_uv * sigma_vw * sigma_uw;
    if parity_product == 1 {
        same_count == 0
    } else {
        same_count == 3
    }
}

fn summarize_c518(dim: usize) -> FiberSummary {
    let components = motif_components_for_cross_assessors(dim);
    let mut total_triangles = 0usize;
    let mut f_pp_pure = 0usize;
    let mut f_pp_mixed = 0usize;
    let mut f_other_pure = 0usize;
    let mut f_other_mixed = 0usize;
    let mut fiber_counts: BTreeMap<(i32, i32), [usize; 2]> = BTreeMap::new();

    for component in &components {
        let nodes: Vec<CrossPair> = component.nodes.iter().copied().collect();
        let edge_set: HashSet<(CrossPair, CrossPair)> = component.edges.iter().copied().collect();

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                for k in (j + 1)..nodes.len() {
                    let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                    if !edge_present(&edge_set, u, v)
                        || !edge_present(&edge_set, v, w)
                        || !edge_present(&edge_set, u, w)
                    {
                        continue;
                    }

                    total_triangles += 1;
                    let is_pure = triangle_is_pure(dim, u, v, w);
                    let fiber = (
                        assoc_sign(dim, u.0, v.0, w.0),
                        assoc_sign(dim, u.1, v.1, w.1),
                    );
                    let counts = fiber_counts.entry(fiber).or_insert([0, 0]);
                    if is_pure {
                        counts[0] += 1;
                    } else {
                        counts[1] += 1;
                    }

                    if fiber == (1, 1) {
                        if is_pure {
                            f_pp_pure += 1;
                        } else {
                            f_pp_mixed += 1;
                        }
                    } else if is_pure {
                        f_other_pure += 1;
                    } else {
                        f_other_mixed += 1;
                    }
                }
            }
        }
    }

    FiberSummary {
        total_triangles,
        f_pp_pure,
        f_pp_mixed,
        f_other_pure,
        f_other_mixed,
        perfect_equivalence: f_pp_mixed == 0 && f_other_pure == 0,
        fiber_counts,
    }
}

fn summarize_c585() -> C585Summary {
    let diagonal = AlbertElement::diagonal(1.0, 0.0, -1.0).delta_squared_x87();
    let target = 3.0 / 8.0;
    let mut values = Vec::new();

    for &a in &[1.0, 0.5, 0.3] {
        let b = -a / 2.0;
        let c = -a - b;
        for oct_idx in 1..=7 {
            let mut z = AlbertElement::zero();
            z.diag = [a, b, c];
            z.off[0][oct_idx] = 1.0;
            z.off[1][(oct_idx + 1) % 7 + 1] = 1.0;
            z.off[2][(oct_idx + 2) % 7 + 1] = 1.0;

            let eigenvalues = z.eigenvalues_x87();
            if eigenvalues[0].is_nan() {
                continue;
            }
            values.push(z.delta_squared_x87());
        }
    }

    let sample_count = values.len();
    let sum = values.iter().sum::<f64>();
    let mean_delta_squared = sum / sample_count as f64;
    let min_delta_squared = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_delta_squared = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut closest_to_three_eighths = values[0];
    let mut closest_absolute_error = (values[0] - target).abs();
    let mut exact_match_count = 0usize;
    for value in &values {
        let error = (*value - target).abs();
        if error < closest_absolute_error {
            closest_absolute_error = error;
            closest_to_three_eighths = *value;
        }
        if error < 1e-9 {
            exact_match_count += 1;
        }
    }

    C585Summary {
        tracefree_diagonal_delta_squared: diagonal,
        sample_count,
        min_delta_squared,
        max_delta_squared,
        mean_delta_squared,
        closest_to_three_eighths,
        closest_absolute_error,
        exact_match_count,
    }
}

fn render_list(values: &[usize]) -> String {
    let inner = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn render_degree_results(values: &[(usize, bool)]) -> String {
    let inner = values
        .iter()
        .map(|(degree, success)| format!("[{}, {}]", degree, success))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn render_fiber_counts(summary: &FiberSummary) -> String {
    let inner = summary
        .fiber_counts
        .iter()
        .map(|((a_lo, a_hi), counts)| {
            format!(
                "{{ a_lo = {}, a_hi = {}, pure = {}, mixed = {} }}",
                a_lo, a_hi, counts[0], counts[1]
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn render_report() -> String {
    let c445 = summarize_c445();
    let c466_dim16 = summarize_c466(16);
    let c466_dim32 = summarize_c466(32);
    let c518_dim16 = summarize_c518(16);
    let c518_dim32 = summarize_c518(32);
    let c518_dim64 = summarize_c518(64);
    let c585 = summarize_c585();

    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "experiment = \"E-205\"");
    let _ = writeln!(
        out,
        "title = \"Deterministic audit for four refuted algebraic-structural claims\""
    );
    let _ = writeln!(out, "claims = [\"C-445\", \"C-466\", \"C-518\", \"C-585\"]");
    let _ = writeln!(out, "deterministic = true");
    let _ = writeln!(out);

    let _ = writeln!(out, "[c445.motif_separator]");
    let _ = writeln!(out, "dim = 32");
    let _ = writeln!(out, "labels = {}", render_list(&c445.labels));
    let _ = writeln!(out, "class_sizes = {}", render_list(&c445.class_sizes));
    let _ = writeln!(
        out,
        "linear_separator = {}",
        c445.linear_separator
            .map(|value| value.to_string())
            .unwrap_or_else(|| "\"none\"".to_string())
    );
    let _ = writeln!(
        out,
        "affine_separator = {}",
        c445.affine_separator
            .map(|(mask, bias)| format!("{{ mask = {}, bias = {} }}", mask, bias))
            .unwrap_or_else(|| "\"none\"".to_string())
    );
    let _ = writeln!(
        out,
        "min_degree = {}",
        c445.min_degree
            .map(|value| value.to_string())
            .unwrap_or_else(|| "\"none\"".to_string())
    );
    let _ = writeln!(
        out,
        "degree_results = {}",
        render_degree_results(&c445.degree_results)
    );
    let _ = writeln!(out);

    for summary in [&c466_dim16, &c466_dim32] {
        let _ = writeln!(out, "[c466.dim_{}]", summary.dim);
        let _ = writeln!(out, "dim = {}", summary.dim);
        let _ = writeln!(out, "rank = {}", summary.rank);
        let _ = writeln!(
            out,
            "unsigned_consistent_count = {}",
            summary.unsigned_consistent_count
        );
        let _ = writeln!(
            out,
            "signed_consistent_count = {}",
            summary.signed_consistent_count
        );
        let _ = writeln!(
            out,
            "unsigned_consistent_bases = {}",
            render_list(&summary.unsigned_consistent_bases)
        );
        let _ = writeln!(
            out,
            "signed_consistent_bases = {}",
            render_list(&summary.signed_consistent_bases)
        );
        let _ = writeln!(out);
    }

    for (dim, summary) in [(16, &c518_dim16), (32, &c518_dim32), (64, &c518_dim64)] {
        let _ = writeln!(out, "[c518.dim_{}]", dim);
        let _ = writeln!(out, "dim = {}", dim);
        let _ = writeln!(out, "total_triangles = {}", summary.total_triangles);
        let _ = writeln!(out, "f_pp_pure = {}", summary.f_pp_pure);
        let _ = writeln!(out, "f_pp_mixed = {}", summary.f_pp_mixed);
        let _ = writeln!(out, "f_other_pure = {}", summary.f_other_pure);
        let _ = writeln!(out, "f_other_mixed = {}", summary.f_other_mixed);
        let _ = writeln!(out, "perfect_equivalence = {}", summary.perfect_equivalence);
        let _ = writeln!(out, "fiber_counts = {}", render_fiber_counts(summary));
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "[c585.tracefree_survey]");
    let _ = writeln!(
        out,
        "tracefree_diagonal_delta_squared = {:.12}",
        c585.tracefree_diagonal_delta_squared
    );
    let _ = writeln!(out, "sample_count = {}", c585.sample_count);
    let _ = writeln!(out, "min_delta_squared = {:.12}", c585.min_delta_squared);
    let _ = writeln!(out, "max_delta_squared = {:.12}", c585.max_delta_squared);
    let _ = writeln!(out, "mean_delta_squared = {:.12}", c585.mean_delta_squared);
    let _ = writeln!(
        out,
        "closest_to_three_eighths = {:.12}",
        c585.closest_to_three_eighths
    );
    let _ = writeln!(
        out,
        "closest_absolute_error = {:.12}",
        c585.closest_absolute_error
    );
    let _ = writeln!(out, "exact_match_count = {}", c585.exact_match_count);
    out
}

fn main() {
    let args = Args::parse();
    let report = render_report();
    write_or_exit(&args.output, &report);
}
