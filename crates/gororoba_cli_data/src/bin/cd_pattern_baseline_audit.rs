//! Deterministic audit for three refuted/closed Cayley-Dickson pattern-match claims.
//!
//! Claims covered:
//! - C-068: the 84-ZD partner spectrum does not uniquely match PDG mass ladders
//! - C-070: a CD-derived associator curve is not exceptionally close to NANOGrav
//! - C-092: generic SO(7) motion leaves the exact ZD lane immediately, so the
//!   surviving orbit structure is discrete/combinatorial rather than continuous

use algebra_analysis::reggiani::{partner_graph_degeneracies, partner_graph_stats};
use algebra_experimental::so7_drift::angle_sweep;
use clap::Parser;
use data_core::{
    catalogs::nanograv::bestfit, parse_nanograv_free_spectrum, parse_pdg_mass_reference_csv,
};
use stats_core::{frechet_distance, frechet_null_test, normalize_spectrum};
use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process,
};

const PERMUTATIONS: usize = 1024;
const RNG_SEED: u64 = 0x0C0D_E207;
const C070_DIM: usize = 16;
const C070_QUANTILES: usize = bestfit::N_BINS;
const C092_ROTATIONS: usize = 64;

#[derive(Parser, Debug)]
#[command(name = "cd-pattern-baseline-audit")]
#[command(
    about = "Deterministic audit for refuted Cayley-Dickson spectrum, shape-match, and SO(7) orbit claims"
)]
struct Args {
    /// Output TOML artifact path.
    #[arg(
        long,
        default_value = "data/output/claims_falsification/cd_pattern_baseline_audit.toml"
    )]
    output: PathBuf,
}

#[derive(Clone)]
struct ParticleMass {
    name: String,
    mass_gev: f64,
}

#[derive(Debug)]
struct C068MatchSummary {
    subset_names: Vec<String>,
    transformed_values: Vec<f64>,
    observed_distance: f64,
    p_value: f64,
    mean_null: f64,
    std_null: f64,
}

#[derive(Debug)]
struct C068Summary {
    pdg_source_path: String,
    unique_eigenvalues: Vec<f64>,
    degeneracies: Vec<usize>,
    unique_eigenvalue_count: usize,
    positive_unique_eigenvalue_count: usize,
    zero_eigenvalue_degeneracy: usize,
    pdg_mass_count: usize,
    best_log_match: C068MatchSummary,
    best_linear_match: C068MatchSummary,
    subset_changed_between_bases: bool,
}

#[derive(Debug)]
struct C070TemplateDistance {
    name: &'static str,
    frechet_distance: f64,
}

#[derive(Debug)]
struct C070Summary {
    nanograv_row_count: usize,
    nanograv_matches_embedded_bestfit: bool,
    quantile_curve_count: usize,
    associator_quantiles: Vec<f64>,
    nanograv_medians: Vec<f64>,
    observed_distance: f64,
    null_p_value: f64,
    null_mean: f64,
    null_std: f64,
    best_simple_template: C070TemplateDistance,
    template_distances: Vec<C070TemplateDistance>,
}

#[derive(Debug)]
struct C092DriftPoint {
    angle_scale: f64,
    mean_product_norm: f64,
    max_product_norm: f64,
    fraction_broken: f64,
}

#[derive(Debug)]
struct C092Summary {
    partner_graph_vertices: usize,
    partner_graph_directed_edges: usize,
    partner_graph_is_symmetric: bool,
    partner_graph_orbit_count: usize,
    partner_graph_orbit_sizes: Vec<usize>,
    drift: Vec<C092DriftPoint>,
    nonzero_scale_breaks_majority: bool,
}

fn write_or_exit(path: &Path, text: &str) {
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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
}

fn repo_path(relative: &str) -> PathBuf {
    repo_root().join(relative)
}

fn parse_nanograv_or_exit(path: &Path) -> Vec<data_core::catalogs::nanograv::FreeSpectrumPoint> {
    parse_nanograv_free_spectrum(path).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to parse {}: {err}", path.display());
        process::exit(1);
    })
}

fn load_pdg_masses_or_exit() -> (String, Vec<ParticleMass>) {
    let source_path = repo_path("data/external/pdg_2025/mass_subset.csv");
    let rows = parse_pdg_mass_reference_csv(&source_path).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to parse {}: {err}", source_path.display());
        process::exit(1);
    });
    let masses = rows
        .into_iter()
        .map(|row| ParticleMass {
            name: row.particle,
            mass_gev: row.mass_gev,
        })
        .collect::<Vec<_>>();
    ("data/external/pdg_2025/mass_subset.csv".to_string(), masses)
}

fn basis_vector(dim: usize, index: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    out[index] = 1.0;
    out
}

fn exact_associator_norms(dim: usize) -> Vec<f64> {
    let mut norms = Vec::new();
    for i in 1..dim {
        let a = basis_vector(dim, i);
        for j in 1..dim {
            if j == i {
                continue;
            }
            let b = basis_vector(dim, j);
            for k in 1..dim {
                if k == i || k == j {
                    continue;
                }
                let c = basis_vector(dim, k);
                let norm = cd_kernel::cayley_dickson::cd_associator_norm(&a, &b, &c);
                norms.push(norm);
            }
        }
    }
    norms.sort_by(|left, right| left.partial_cmp(right).unwrap());
    norms
}

fn quantile_curve(sorted_values: &[f64], n_points: usize) -> Vec<f64> {
    if sorted_values.is_empty() || n_points == 0 {
        return Vec::new();
    }
    if sorted_values.len() == 1 {
        return vec![sorted_values[0]; n_points];
    }
    (0..n_points)
        .map(|idx| {
            let pos = idx * (sorted_values.len() - 1) / (n_points - 1);
            sorted_values[pos]
        })
        .collect()
}

fn frechet_on_normalized(left: &[f64], right: &[f64]) -> f64 {
    frechet_distance(&normalize_spectrum(left), &normalize_spectrum(right))
}

fn generate_templates(n: usize) -> Vec<(&'static str, Vec<f64>)> {
    let denom = (n.saturating_sub(1)).max(1) as f64;
    let grid: Vec<f64> = (0..n).map(|idx| idx as f64 / denom).collect();
    let linear = grid.clone();
    let concave: Vec<f64> = grid.iter().map(|value| value.sqrt()).collect();
    let convex: Vec<f64> = grid.iter().map(|value| value * value).collect();
    let logistic: Vec<f64> = grid
        .iter()
        .map(|value| 1.0 / (1.0 + (-12.0 * (value - 0.5)).exp()))
        .collect();
    vec![
        ("linear_ramp", linear),
        ("concave_sqrt", concave),
        ("convex_square", convex),
        ("logistic_s_curve", logistic),
    ]
}

fn enumerate_fixed_size_subsets(
    n_total: usize,
    subset_size: usize,
    start: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if current.len() == subset_size {
        out.push(current.clone());
        return;
    }
    let remaining_needed = subset_size - current.len();
    let max_start = n_total.saturating_sub(remaining_needed);
    for idx in start..=max_start {
        current.push(idx);
        enumerate_fixed_size_subsets(n_total, subset_size, idx + 1, current, out);
        current.pop();
    }
}

fn all_fixed_size_subsets(n_total: usize, subset_size: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    enumerate_fixed_size_subsets(n_total, subset_size, 0, &mut current, &mut out);
    out
}

fn transform_log10(value: f64) -> f64 {
    value.log10()
}

fn transform_linear(value: f64) -> f64 {
    value
}

fn best_mass_subset_match(
    masses: &[ParticleMass],
    transform: fn(f64) -> f64,
    reference_curve: &[f64],
) -> C068MatchSummary {
    let subsets = all_fixed_size_subsets(masses.len(), reference_curve.len());
    let mut best_names = Vec::new();
    let mut best_values = Vec::new();
    let mut best_distance = f64::INFINITY;

    for subset in subsets {
        let mut names = Vec::with_capacity(reference_curve.len());
        let mut values = Vec::with_capacity(reference_curve.len());
        for &index in &subset {
            names.push(masses[index].name.clone());
            values.push(transform(masses[index].mass_gev));
        }
        values.sort_by(|left, right| left.partial_cmp(right).unwrap());
        let distance = frechet_on_normalized(reference_curve, &values);
        if distance < best_distance {
            best_distance = distance;
            best_names = names;
            best_values = values;
        }
    }

    let null = frechet_null_test(reference_curve, &best_values, PERMUTATIONS, RNG_SEED + 11);
    C068MatchSummary {
        subset_names: best_names,
        transformed_values: best_values,
        observed_distance: null.observed_distance,
        p_value: null.p_value,
        mean_null: null.mean_null,
        std_null: null.std_null,
    }
}

fn summarize_c068() -> C068Summary {
    let (pdg_source_path, pdg_masses) = load_pdg_masses_or_exit();
    let degeneracies = partner_graph_degeneracies(1.0e-9);
    let unique_eigenvalues: Vec<f64> = degeneracies.iter().map(|(value, _)| *value).collect();
    let counts: Vec<usize> = degeneracies.iter().map(|(_, count)| *count).collect();
    let positive_unique_eigenvalue_count = unique_eigenvalues
        .iter()
        .filter(|&&value| value > 1.0e-12)
        .count();
    let zero_eigenvalue_degeneracy = degeneracies
        .iter()
        .find_map(|(value, count)| {
            if value.abs() < 1.0e-12 {
                Some(*count)
            } else {
                None
            }
        })
        .unwrap_or(0);

    let best_log_match = best_mass_subset_match(&pdg_masses, transform_log10, &unique_eigenvalues);
    let best_linear_match =
        best_mass_subset_match(&pdg_masses, transform_linear, &unique_eigenvalues);

    C068Summary {
        pdg_source_path,
        unique_eigenvalues,
        degeneracies: counts,
        unique_eigenvalue_count: degeneracies.len(),
        positive_unique_eigenvalue_count,
        zero_eigenvalue_degeneracy,
        pdg_mass_count: pdg_masses.len(),
        subset_changed_between_bases: best_log_match.subset_names != best_linear_match.subset_names,
        best_log_match,
        best_linear_match,
    }
}

fn summarize_c070() -> C070Summary {
    let nanograv_path = repo_path("data/external/nanograv_15yr_freespectrum.csv");
    let nanograv_rows = parse_nanograv_or_exit(&nanograv_path);
    let nanograv_matches_embedded_bestfit = nanograv_rows.len() == bestfit::N_BINS
        && nanograv_rows
            .iter()
            .zip(bestfit::HD_FREE_SPECTRUM.iter())
            .all(|(row, expected)| {
                (row.frequency - expected.frequency).abs() < 1.0e-18
                    && (row.log10_rho - expected.log10_rho).abs() < 1.0e-6
                    && (row.log10_rho_lo - expected.log10_rho_lo).abs() < 1.0e-6
                    && (row.log10_rho_hi - expected.log10_rho_hi).abs() < 1.0e-6
            });

    let norms = exact_associator_norms(C070_DIM);
    let associator_quantiles = quantile_curve(&norms, C070_QUANTILES);
    let nanograv_medians: Vec<f64> = nanograv_rows.iter().map(|point| point.log10_rho).collect();
    let null = frechet_null_test(
        &associator_quantiles,
        &nanograv_medians,
        PERMUTATIONS,
        RNG_SEED + 70,
    );

    let mut template_distances = Vec::new();
    for (name, template) in generate_templates(bestfit::N_BINS) {
        template_distances.push(C070TemplateDistance {
            name,
            frechet_distance: frechet_on_normalized(&template, &nanograv_medians),
        });
    }
    template_distances.sort_by(|left, right| {
        left.frechet_distance
            .partial_cmp(&right.frechet_distance)
            .unwrap()
    });

    C070Summary {
        nanograv_row_count: nanograv_rows.len(),
        nanograv_matches_embedded_bestfit,
        quantile_curve_count: associator_quantiles.len(),
        associator_quantiles,
        nanograv_medians,
        observed_distance: null.observed_distance,
        null_p_value: null.p_value,
        null_mean: null.mean_null,
        null_std: null.std_null,
        best_simple_template: C070TemplateDistance {
            name: template_distances[0].name,
            frechet_distance: template_distances[0].frechet_distance,
        },
        template_distances,
    }
}

fn summarize_c092() -> C092Summary {
    let stats = partner_graph_stats();
    let scales = [0.0, 0.1, 0.25, 0.5, 1.0];
    let sweep = angle_sweep(&scales, C092_ROTATIONS, RNG_SEED + 92);
    let drift: Vec<C092DriftPoint> = sweep
        .iter()
        .map(|point| C092DriftPoint {
            angle_scale: point.angle_scale,
            mean_product_norm: point.mean_product_norm,
            max_product_norm: point.max_product_norm,
            fraction_broken: point.fraction_broken,
        })
        .collect();
    let nonzero_scale_breaks_majority = drift
        .iter()
        .filter(|point| point.angle_scale > 0.0)
        .all(|point| point.fraction_broken > 0.5);

    C092Summary {
        partner_graph_vertices: stats.n_vertices,
        partner_graph_directed_edges: stats.n_directed_edges,
        partner_graph_is_symmetric: stats.is_symmetric,
        partner_graph_orbit_count: stats.n_orbits,
        partner_graph_orbit_sizes: stats.orbit_sizes,
        drift,
        nonzero_scale_breaks_majority,
    }
}

fn render_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.12}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_name_list(values: &[String]) -> String {
    values
        .iter()
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn main() {
    let args = Args::parse();

    let c068 = summarize_c068();
    let c070 = summarize_c070();
    let c092 = summarize_c092();

    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "experiment = \"E-207\"");
    let _ = writeln!(
        out,
        "title = \"Deterministic audit for Cayley-Dickson spectrum, shape-match, and SO(7) orbit refutations\""
    );
    let _ = writeln!(out, "claims = [\"C-068\", \"C-070\", \"C-092\"]");
    let _ = writeln!(out, "deterministic = true");
    let _ = writeln!(out);

    let _ = writeln!(out, "[c068.partner_spectrum]");
    let _ = writeln!(out, "pdg_source_path = \"{}\"", c068.pdg_source_path);
    let _ = writeln!(
        out,
        "unique_eigenvalues = [{}]",
        render_list(&c068.unique_eigenvalues)
    );
    let _ = writeln!(
        out,
        "degeneracies = [{}]",
        render_usize_list(&c068.degeneracies)
    );
    let _ = writeln!(
        out,
        "unique_eigenvalue_count = {}",
        c068.unique_eigenvalue_count
    );
    let _ = writeln!(
        out,
        "positive_unique_eigenvalue_count = {}",
        c068.positive_unique_eigenvalue_count
    );
    let _ = writeln!(
        out,
        "zero_eigenvalue_degeneracy = {}",
        c068.zero_eigenvalue_degeneracy
    );
    let _ = writeln!(out, "pdg_mass_count = {}", c068.pdg_mass_count);
    let _ = writeln!(
        out,
        "subset_changed_between_bases = {}",
        render_bool(c068.subset_changed_between_bases)
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "[c068.best_log_match]");
    let _ = writeln!(
        out,
        "subset = [{}]",
        render_name_list(&c068.best_log_match.subset_names)
    );
    let _ = writeln!(
        out,
        "transformed_values = [{}]",
        render_list(&c068.best_log_match.transformed_values)
    );
    let _ = writeln!(
        out,
        "observed_distance = {:.12}",
        c068.best_log_match.observed_distance
    );
    let _ = writeln!(out, "p_value = {:.12}", c068.best_log_match.p_value);
    let _ = writeln!(out, "null_mean = {:.12}", c068.best_log_match.mean_null);
    let _ = writeln!(out, "null_std = {:.12}", c068.best_log_match.std_null);
    let _ = writeln!(out);

    let _ = writeln!(out, "[c068.best_linear_match]");
    let _ = writeln!(
        out,
        "subset = [{}]",
        render_name_list(&c068.best_linear_match.subset_names)
    );
    let _ = writeln!(
        out,
        "transformed_values = [{}]",
        render_list(&c068.best_linear_match.transformed_values)
    );
    let _ = writeln!(
        out,
        "observed_distance = {:.12}",
        c068.best_linear_match.observed_distance
    );
    let _ = writeln!(out, "p_value = {:.12}", c068.best_linear_match.p_value);
    let _ = writeln!(out, "null_mean = {:.12}", c068.best_linear_match.mean_null);
    let _ = writeln!(out, "null_std = {:.12}", c068.best_linear_match.std_null);
    let _ = writeln!(out);

    let _ = writeln!(out, "[c070.shape_match]");
    let _ = writeln!(out, "nanograv_row_count = {}", c070.nanograv_row_count);
    let _ = writeln!(
        out,
        "nanograv_matches_embedded_bestfit = {}",
        render_bool(c070.nanograv_matches_embedded_bestfit)
    );
    let _ = writeln!(out, "quantile_curve_count = {}", c070.quantile_curve_count);
    let _ = writeln!(
        out,
        "associator_quantiles = [{}]",
        render_list(&c070.associator_quantiles)
    );
    let _ = writeln!(
        out,
        "nanograv_medians = [{}]",
        render_list(&c070.nanograv_medians)
    );
    let _ = writeln!(out, "observed_distance = {:.12}", c070.observed_distance);
    let _ = writeln!(out, "null_p_value = {:.12}", c070.null_p_value);
    let _ = writeln!(out, "null_mean = {:.12}", c070.null_mean);
    let _ = writeln!(out, "null_std = {:.12}", c070.null_std);
    let _ = writeln!(out);

    let _ = writeln!(out, "[c070.best_simple_template]");
    let _ = writeln!(out, "name = \"{}\"", c070.best_simple_template.name);
    let _ = writeln!(
        out,
        "frechet_distance = {:.12}",
        c070.best_simple_template.frechet_distance
    );
    let _ = writeln!(out);

    for template in &c070.template_distances {
        let _ = writeln!(out, "[[c070.template_distances]]");
        let _ = writeln!(out, "name = \"{}\"", template.name);
        let _ = writeln!(out, "frechet_distance = {:.12}", template.frechet_distance);
        let _ = writeln!(out);
    }

    let _ = writeln!(out, "[c092.orbit_structure]");
    let _ = writeln!(
        out,
        "partner_graph_vertices = {}",
        c092.partner_graph_vertices
    );
    let _ = writeln!(
        out,
        "partner_graph_directed_edges = {}",
        c092.partner_graph_directed_edges
    );
    let _ = writeln!(
        out,
        "partner_graph_is_symmetric = {}",
        render_bool(c092.partner_graph_is_symmetric)
    );
    let _ = writeln!(
        out,
        "partner_graph_orbit_count = {}",
        c092.partner_graph_orbit_count
    );
    let _ = writeln!(
        out,
        "partner_graph_orbit_sizes = [{}]",
        render_usize_list(&c092.partner_graph_orbit_sizes)
    );
    let _ = writeln!(
        out,
        "nonzero_scale_breaks_majority = {}",
        render_bool(c092.nonzero_scale_breaks_majority)
    );
    let _ = writeln!(out);

    for point in &c092.drift {
        let _ = writeln!(out, "[[c092.drift]]");
        let _ = writeln!(out, "angle_scale = {:.6}", point.angle_scale);
        let _ = writeln!(out, "mean_product_norm = {:.12}", point.mean_product_norm);
        let _ = writeln!(out, "max_product_norm = {:.12}", point.max_product_norm);
        let _ = writeln!(out, "fraction_broken = {:.12}", point.fraction_broken);
        let _ = writeln!(out);
    }

    write_or_exit(&args.output, &out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c068_subset_match_uses_full_eigenlevel_count() {
        let summary = summarize_c068();
        assert_eq!(summary.unique_eigenvalue_count, 5);
        assert_eq!(summary.best_log_match.subset_names.len(), 5);
        assert_eq!(summary.best_linear_match.subset_names.len(), 5);
    }

    #[test]
    fn c070_quantile_curve_has_expected_length() {
        let summary = summarize_c070();
        assert_eq!(summary.quantile_curve_count, bestfit::N_BINS);
        assert_eq!(summary.associator_quantiles.len(), bestfit::N_BINS);
        for pair in summary.associator_quantiles.windows(2) {
            assert!(pair[0] <= pair[1] + 1.0e-12);
        }
    }

    #[test]
    fn c092_zero_scale_preserves_but_nonzero_scales_drift() {
        let summary = summarize_c092();
        assert!(summary.drift[0].mean_product_norm < 1.0e-10);
        assert!(summary.drift[1].fraction_broken > 0.0);
    }
}
