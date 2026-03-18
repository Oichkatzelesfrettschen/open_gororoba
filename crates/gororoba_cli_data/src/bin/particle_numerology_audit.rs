//! Deterministic audit for six particle-numerology refutations.
//!
//! Claims covered:
//! - C-069: octonion subalgebra principal angles do not reproduce PMNS angles
//! - C-077: democratic associator mixing stays far from PMNS probabilities
//! - C-081: exact Givens fits are generic, not PMNS-specific
//! - C-084: small Yukawa-like perturbations barely improve democratic mixing
//! - C-587: depth-based associator norms do not generate the lepton hierarchy
//! - C-1355: "1764" is unit-dependent numerology, not an invariant constant

use algebra_analysis::{
    grassmannian::{principal_angles, subspace_from_vectors},
    subalgebra::enumerate_octonion_subalgebras,
};
use clap::Parser;
use gr_core::constants::{E_PLANCK_CGS, M_PLANCK_CGS, g_to_planck_mass};
use materials_core::tang_mass::{depth_based_mass_prediction, mass_ratio_null_test};
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use stats_core::pmns_matrix;
use std::{fmt::Write as _, fs, path::PathBuf, process};

const PMNS_ANGLES_DEG: [f64; 3] = [33.41, 42.2, 8.54];
const PERMUTATIONS: usize = 1024;
const RNG_SEED: u64 = 0x0C0D_E206;
const ERG_PER_GEV: f64 = 1.602_176_634e-3;
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

#[derive(Parser, Debug)]
#[command(name = "particle-numerology-audit")]
#[command(about = "Deterministic audit for refuted PMNS, mass-hierarchy, and numerology claims")]
struct Args {
    /// Output TOML artifact path.
    #[arg(
        long,
        default_value = "data/output/claims_falsification/particle_numerology_audit.toml"
    )]
    output: PathBuf,
}

#[derive(Debug)]
struct NullSummary {
    observed_distance: f64,
    mean_null: f64,
    std_null: f64,
    p_value: f64,
    significant_at_005: bool,
}

#[derive(Debug)]
struct C069Summary {
    subalgebra_count: usize,
    pair_count: usize,
    unique_angle_degrees: Vec<f64>,
    best_triplet_degrees: [f64; 3],
    best_triplet_rms_error_deg: f64,
}

#[derive(Debug)]
struct ProbabilitySummary {
    frobenius_distance: f64,
    mean_diagonal: f64,
    null: NullSummary,
}

#[derive(Debug)]
struct C084Summary {
    democratic: ProbabilitySummary,
    best_distance: f64,
    best_mean_diagonal: f64,
    best_params: [f64; 3],
    improvement: f64,
    null: NullSummary,
}

#[derive(Debug)]
struct C081Summary {
    sample_count: usize,
    max_abs_angle_error_deg: f64,
    mean_abs_angle_error_deg: f64,
}

#[derive(Debug)]
struct C587Summary {
    dim16_norms: [f64; 4],
    dim16_best_ratios: [f64; 3],
    dim16_rms_deviation: f64,
    dim32_rms_deviation: f64,
    null_observed_rms: f64,
    null_mean_rms: f64,
    null_std_rms: f64,
    null_p_value: f64,
    null_significant: bool,
}

#[derive(Debug)]
struct C1355Summary {
    planck_mass_cgs: String,
    planck_mass_si: String,
    planck_mass_gev: String,
    planck_mass_planck_units: String,
    cgs_contains_1764: bool,
    si_contains_1764: bool,
    gev_contains_1764: bool,
    planck_units_contains_1764: bool,
    bcs_formula_value: f64,
    bcs_code_value: f64,
    bcs_relative_error: f64,
    unrelated_window_count: usize,
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

fn basis_vector(dim: usize, index: usize) -> Vec<f64> {
    let mut v = vec![0.0; dim];
    v[index] = 1.0;
    v
}

fn pmns_probability_matrix() -> DMatrix<f64> {
    let pmns = pmns_matrix();
    let mut out = DMatrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = pmns[(i, j)].norm_sqr();
        }
    }
    out
}

fn frobenius_distance_real(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
    (a - b).norm()
}

fn mean_diagonal(matrix: &DMatrix<f64>) -> f64 {
    (0..matrix.nrows()).map(|i| matrix[(i, i)]).sum::<f64>() / matrix.nrows() as f64
}

fn sinkhorn_normalize(mut matrix: DMatrix<f64>, iterations: usize) -> DMatrix<f64> {
    for _ in 0..iterations {
        for i in 0..matrix.nrows() {
            let row_sum: f64 = (0..matrix.ncols()).map(|j| matrix[(i, j)]).sum();
            if row_sum > 1e-15 {
                for j in 0..matrix.ncols() {
                    matrix[(i, j)] /= row_sum;
                }
            }
        }
        for j in 0..matrix.ncols() {
            let col_sum: f64 = (0..matrix.nrows()).map(|i| matrix[(i, j)]).sum();
            if col_sum > 1e-15 {
                for i in 0..matrix.nrows() {
                    matrix[(i, j)] /= col_sum;
                }
            }
        }
    }
    matrix
}

fn random_doubly_stochastic(seed: u64) -> DMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut matrix = DMatrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            matrix[(i, j)] = 0.1 + rng.r#gen::<f64>();
        }
    }
    sinkhorn_normalize(matrix, 24)
}

fn stochastic_null_test(
    observed: &DMatrix<f64>,
    reference: &DMatrix<f64>,
    permutations: usize,
    seed: u64,
) -> NullSummary {
    let observed_distance = frobenius_distance_real(observed, reference);
    let mut distances = Vec::with_capacity(permutations);
    for i in 0..permutations {
        let candidate = random_doubly_stochastic(seed + i as u64);
        distances.push(frobenius_distance_real(&candidate, reference));
    }
    let mean_null = distances.iter().sum::<f64>() / distances.len() as f64;
    let std_null = (distances
        .iter()
        .map(|value| (value - mean_null).powi(2))
        .sum::<f64>()
        / distances.len() as f64)
        .sqrt();
    let n_better_or_equal = distances
        .iter()
        .filter(|&&value| value <= observed_distance)
        .count();
    let p_value = n_better_or_equal as f64 / distances.len() as f64;
    NullSummary {
        observed_distance,
        mean_null,
        std_null,
        p_value,
        significant_at_005: p_value < 0.05,
    }
}

fn democratic_probability_matrix() -> DMatrix<f64> {
    DMatrix::from_element(3, 3, 1.0 / 3.0)
}

fn summarize_probability_matrix(matrix: &DMatrix<f64>, pmns_probs: &DMatrix<f64>) -> ProbabilitySummary {
    ProbabilitySummary {
        frobenius_distance: frobenius_distance_real(matrix, pmns_probs),
        mean_diagonal: mean_diagonal(matrix),
        null: stochastic_null_test(matrix, pmns_probs, PERMUTATIONS, RNG_SEED),
    }
}

fn summarize_c069() -> C069Summary {
    let enumeration = enumerate_octonion_subalgebras(16);
    let mut angle_values = Vec::new();
    let mut pair_count = 0usize;

    for i in 0..enumeration.subalgebras.len() {
        let left_vectors: Vec<Vec<f64>> = enumeration.subalgebras[i]
            .indices
            .iter()
            .map(|&idx| basis_vector(16, idx))
            .collect();
        let left = subspace_from_vectors(&left_vectors);
        for j in (i + 1)..enumeration.subalgebras.len() {
            let right_vectors: Vec<Vec<f64>> = enumeration.subalgebras[j]
                .indices
                .iter()
                .map(|&idx| basis_vector(16, idx))
                .collect();
            let right = subspace_from_vectors(&right_vectors);
            let angles = principal_angles(&left, &right);
            angle_values.extend(angles.into_iter().map(f64::to_degrees));
            pair_count += 1;
        }
    }

    angle_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut unique_angle_degrees = Vec::new();
    for value in angle_values {
        if unique_angle_degrees.is_empty()
            || (value - unique_angle_degrees.last().copied().unwrap_or(f64::NEG_INFINITY)).abs()
                > 1e-9
        {
            unique_angle_degrees.push(value);
        }
    }

    let targets = sorted_angles(PMNS_ANGLES_DEG);
    let mut best_triplet = [0.0; 3];
    let mut best_rms = f64::INFINITY;
    for &a in &unique_angle_degrees {
        for &b in &unique_angle_degrees {
            for &c in &unique_angle_degrees {
                let candidate = sorted_angles([a, b, c]);
                let rms = rms_angle_error(candidate, targets);
                if rms < best_rms {
                    best_rms = rms;
                    best_triplet = candidate;
                }
            }
        }
    }

    C069Summary {
        subalgebra_count: enumeration.subalgebras.len(),
        pair_count,
        unique_angle_degrees,
        best_triplet_degrees: best_triplet,
        best_triplet_rms_error_deg: best_rms,
    }
}

fn sorted_angles(mut angles: [f64; 3]) -> [f64; 3] {
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    angles
}

fn rms_angle_error(candidate: [f64; 3], target: [f64; 3]) -> f64 {
    ((0..3)
        .map(|idx| (candidate[idx] - target[idx]).powi(2))
        .sum::<f64>()
        / 3.0)
        .sqrt()
}

fn summarize_c077(pmns_probs: &DMatrix<f64>) -> ProbabilitySummary {
    summarize_probability_matrix(&democratic_probability_matrix(), pmns_probs)
}

fn summarize_c084(pmns_probs: &DMatrix<f64>) -> C084Summary {
    let democratic = summarize_c077(pmns_probs);
    let mut best_distance = f64::INFINITY;
    let mut best_mean_diagonal = 0.0;
    let mut best_params = [0.0; 3];
    let mut best_matrix = democratic_probability_matrix();

    for ia in -6..=6 {
        for ib in -6..=6 {
            for ic in -6..=6 {
                let a = ia as f64 / 20.0;
                let b = ib as f64 / 20.0;
                let c = ic as f64 / 20.0;
                let mut trial = DMatrix::from_element(3, 3, 1.0);
                trial[(0, 0)] += a;
                trial[(1, 1)] += b;
                trial[(2, 2)] += c;
                if trial.iter().any(|value| *value <= 0.0) {
                    continue;
                }
                let normalized = sinkhorn_normalize(trial, 24);
                let distance = frobenius_distance_real(&normalized, pmns_probs);
                if distance < best_distance {
                    best_distance = distance;
                    best_mean_diagonal = mean_diagonal(&normalized);
                    best_params = [a, b, c];
                    best_matrix = normalized;
                }
            }
        }
    }

    let null = stochastic_null_test(&best_matrix, pmns_probs, PERMUTATIONS, RNG_SEED + 10_000);
    C084Summary {
        democratic,
        best_distance,
        best_mean_diagonal,
        best_params,
        improvement: C084Summary::improvement_from(&best_distance, pmns_probs),
        null,
    }
}

impl C084Summary {
    fn improvement_from(best_distance: &f64, pmns_probs: &DMatrix<f64>) -> f64 {
        frobenius_distance_real(&democratic_probability_matrix(), pmns_probs) - *best_distance
    }
}

fn rotation_x(theta: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        3,
        &[
            1.0,
            0.0,
            0.0,
            0.0,
            theta.cos(),
            theta.sin(),
            0.0,
            -theta.sin(),
            theta.cos(),
        ],
    )
}

fn rotation_y(theta: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        3,
        &[
            theta.cos(),
            0.0,
            theta.sin(),
            0.0,
            1.0,
            0.0,
            -theta.sin(),
            0.0,
            theta.cos(),
        ],
    )
}

fn rotation_z(theta: f64) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        3,
        &[
            theta.cos(),
            theta.sin(),
            0.0,
            -theta.sin(),
            theta.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    )
}

fn givens_rotation_matrix(theta12_deg: f64, theta13_deg: f64, theta23_deg: f64) -> DMatrix<f64> {
    let theta12 = theta12_deg.to_radians();
    let theta13 = theta13_deg.to_radians();
    let theta23 = theta23_deg.to_radians();
    rotation_x(theta23) * rotation_y(theta13) * rotation_z(theta12)
}

fn recover_givens_angles_deg(matrix: &DMatrix<f64>) -> [f64; 3] {
    let theta13 = matrix[(0, 2)].asin().to_degrees();
    let theta12 = matrix[(0, 1)].atan2(matrix[(0, 0)]).to_degrees();
    let theta23 = matrix[(1, 2)].atan2(matrix[(2, 2)]).to_degrees();
    [theta12.abs(), theta13.abs(), theta23.abs()]
}

fn summarize_c081() -> C081Summary {
    let targets = [
        PMNS_ANGLES_DEG,
        [10.0, 20.0, 30.0],
        [15.0, 35.0, 55.0],
        [25.0, 45.0, 65.0],
        [5.0, 80.0, 12.0],
        [60.0, 30.0, 10.0],
    ];
    let mut errors = Vec::new();
    for target in targets {
        let matrix = givens_rotation_matrix(target[0], target[1], target[2]);
        let recovered = recover_givens_angles_deg(&matrix);
        for idx in 0..3 {
            errors.push((target[idx] - recovered[idx]).abs());
        }
    }
    let max_abs_angle_error_deg = errors.iter().copied().fold(0.0, f64::max);
    let mean_abs_angle_error_deg = errors.iter().sum::<f64>() / errors.len() as f64;
    C081Summary {
        sample_count: targets.len(),
        max_abs_angle_error_deg,
        mean_abs_angle_error_deg,
    }
}

fn summarize_c587() -> C587Summary {
    let dim16 = depth_based_mass_prediction(16);
    let dim32 = depth_based_mass_prediction(32);
    let null = mass_ratio_null_test(16, PERMUTATIONS, RNG_SEED + 20_000);
    C587Summary {
        dim16_norms: dim16.depth_norms,
        dim16_best_ratios: [
            dim16.predicted_ratios.0,
            dim16.predicted_ratios.1,
            dim16.predicted_ratios.2,
        ],
        dim16_rms_deviation: dim16.rms_deviation,
        dim32_rms_deviation: dim32.rms_deviation,
        null_observed_rms: null.observed_rms,
        null_mean_rms: null.mean_null_rms,
        null_std_rms: null.std_null_rms,
        null_p_value: null.p_value,
        null_significant: null.significant,
    }
}

fn to_sci_string(value: f64, digits: usize) -> String {
    format!("{value:.digits$e}")
}

fn decimal_windows(text: &str, window: usize) -> Vec<String> {
    let digits: String = text.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.len() < window {
        return Vec::new();
    }
    (0..=digits.len() - window)
        .map(|idx| digits[idx..idx + window].to_string())
        .collect()
}

fn summarize_c1355() -> C1355Summary {
    let cgs = to_sci_string(M_PLANCK_CGS, 6);
    let si = to_sci_string(M_PLANCK_CGS / 1000.0, 6);
    let gev = to_sci_string(E_PLANCK_CGS / ERG_PER_GEV, 6);
    let planck_units = format!("{:.6}", g_to_planck_mass(M_PLANCK_CGS));

    let bcs_formula_value = std::f64::consts::PI / EULER_GAMMA.exp();
    let bcs_code_value = 1.764;
    let bcs_relative_error = (bcs_formula_value - bcs_code_value).abs() / bcs_formula_value;

    let window_pool = [
        cgs.clone(),
        si.clone(),
        gev.clone(),
        planck_units.clone(),
        to_sci_string(gr_core::constants::C_CGS, 6),
        to_sci_string(gr_core::constants::G_CGS, 6),
        to_sci_string(gr_core::constants::HBAR_CGS, 6),
        to_sci_string(gr_core::constants::K_B_CGS, 6),
    ];
    let unrelated_window_count = window_pool
        .iter()
        .flat_map(|text| decimal_windows(text, 4))
        .filter(|window| window != "1764")
        .count();

    C1355Summary {
        planck_mass_cgs: cgs.clone(),
        planck_mass_si: si.clone(),
        planck_mass_gev: gev.clone(),
        planck_mass_planck_units: planck_units.clone(),
        cgs_contains_1764: cgs.contains("1764"),
        si_contains_1764: si.contains("1764"),
        gev_contains_1764: gev.contains("1764"),
        planck_units_contains_1764: planck_units.contains("1764"),
        bcs_formula_value,
        bcs_code_value,
        bcs_relative_error,
        unrelated_window_count,
    }
}

fn render_list(values: &[f64]) -> String {
    let inner = values
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}

fn render_triplet(values: [f64; 3]) -> String {
    format!("[{:.6}, {:.6}, {:.6}]", values[0], values[1], values[2])
}

fn render_null(prefix: &str, null: &NullSummary, out: &mut String) {
    let _ = writeln!(out, "[{prefix}.null]");
    let _ = writeln!(out, "observed_distance = {:.12}", null.observed_distance);
    let _ = writeln!(out, "mean_null = {:.12}", null.mean_null);
    let _ = writeln!(out, "std_null = {:.12}", null.std_null);
    let _ = writeln!(out, "p_value = {:.12}", null.p_value);
    let _ = writeln!(out, "significant_at_005 = {}", null.significant_at_005);
    let _ = writeln!(out);
}

fn render_report() -> String {
    let pmns_probs = pmns_probability_matrix();
    let c069 = summarize_c069();
    let c077 = summarize_c077(&pmns_probs);
    let c081 = summarize_c081();
    let c084 = summarize_c084(&pmns_probs);
    let c587 = summarize_c587();
    let c1355 = summarize_c1355();

    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(out, "experiment = \"E-206\"");
    let _ = writeln!(
        out,
        "title = \"Deterministic audit for PMNS, mass-hierarchy, and numerology refutations\""
    );
    let _ = writeln!(
        out,
        "claims = [\"C-069\", \"C-077\", \"C-081\", \"C-084\", \"C-587\", \"C-1355\"]"
    );
    let _ = writeln!(out, "deterministic = true");
    let _ = writeln!(out);

    let _ = writeln!(out, "[c069.principal_angles]");
    let _ = writeln!(out, "subalgebra_count = {}", c069.subalgebra_count);
    let _ = writeln!(out, "pair_count = {}", c069.pair_count);
    let _ = writeln!(
        out,
        "unique_angle_degrees = {}",
        render_list(&c069.unique_angle_degrees)
    );
    let _ = writeln!(
        out,
        "best_triplet_degrees = {}",
        render_triplet(c069.best_triplet_degrees)
    );
    let _ = writeln!(
        out,
        "best_triplet_rms_error_deg = {:.12}",
        c069.best_triplet_rms_error_deg
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "[c077.democratic_probability]");
    let _ = writeln!(out, "frobenius_distance = {:.12}", c077.frobenius_distance);
    let _ = writeln!(out, "mean_diagonal = {:.12}", c077.mean_diagonal);
    let _ = writeln!(out);
    render_null("c077.democratic_probability", &c077.null, &mut out);

    let _ = writeln!(out, "[c081.givens_triviality]");
    let _ = writeln!(out, "sample_count = {}", c081.sample_count);
    let _ = writeln!(
        out,
        "max_abs_angle_error_deg = {:.12}",
        c081.max_abs_angle_error_deg
    );
    let _ = writeln!(
        out,
        "mean_abs_angle_error_deg = {:.12}",
        c081.mean_abs_angle_error_deg
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "[c084.yukawa_perturbation]");
    let _ = writeln!(
        out,
        "democratic_distance = {:.12}",
        c084.democratic.frobenius_distance
    );
    let _ = writeln!(
        out,
        "democratic_mean_diagonal = {:.12}",
        c084.democratic.mean_diagonal
    );
    let _ = writeln!(out, "best_distance = {:.12}", c084.best_distance);
    let _ = writeln!(out, "best_mean_diagonal = {:.12}", c084.best_mean_diagonal);
    let _ = writeln!(out, "best_params = {}", render_triplet(c084.best_params));
    let _ = writeln!(out, "improvement = {:.12}", c084.improvement);
    let _ = writeln!(out);
    render_null("c084.yukawa_perturbation", &c084.null, &mut out);

    let _ = writeln!(out, "[c587.depth_mass]");
    let _ = writeln!(out, "dim16_norms = {}", render_list(&c587.dim16_norms));
    let _ = writeln!(
        out,
        "dim16_best_ratios = {}",
        render_triplet(c587.dim16_best_ratios)
    );
    let _ = writeln!(out, "dim16_rms_deviation = {:.12}", c587.dim16_rms_deviation);
    let _ = writeln!(out, "dim32_rms_deviation = {:.12}", c587.dim32_rms_deviation);
    let _ = writeln!(out, "null_observed_rms = {:.12}", c587.null_observed_rms);
    let _ = writeln!(out, "null_mean_rms = {:.12}", c587.null_mean_rms);
    let _ = writeln!(out, "null_std_rms = {:.12}", c587.null_std_rms);
    let _ = writeln!(out, "null_p_value = {:.12}", c587.null_p_value);
    let _ = writeln!(out, "null_significant = {}", c587.null_significant);
    let _ = writeln!(out);

    let _ = writeln!(out, "[c1355.numerology]");
    let _ = writeln!(out, "planck_mass_cgs = \"{}\"", c1355.planck_mass_cgs);
    let _ = writeln!(out, "planck_mass_si = \"{}\"", c1355.planck_mass_si);
    let _ = writeln!(out, "planck_mass_gev = \"{}\"", c1355.planck_mass_gev);
    let _ = writeln!(
        out,
        "planck_mass_planck_units = \"{}\"",
        c1355.planck_mass_planck_units
    );
    let _ = writeln!(out, "cgs_contains_1764 = {}", c1355.cgs_contains_1764);
    let _ = writeln!(out, "si_contains_1764 = {}", c1355.si_contains_1764);
    let _ = writeln!(out, "gev_contains_1764 = {}", c1355.gev_contains_1764);
    let _ = writeln!(
        out,
        "planck_units_contains_1764 = {}",
        c1355.planck_units_contains_1764
    );
    let _ = writeln!(out, "bcs_formula_value = {:.12}", c1355.bcs_formula_value);
    let _ = writeln!(out, "bcs_code_value = {:.12}", c1355.bcs_code_value);
    let _ = writeln!(out, "bcs_relative_error = {:.12}", c1355.bcs_relative_error);
    let _ = writeln!(
        out,
        "unrelated_window_count = {}",
        c1355.unrelated_window_count
    );
    out
}

fn main() {
    let args = Args::parse();
    let report = render_report();
    write_or_exit(&args.output, &report);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_preserves_double_stochasticity() {
        let matrix = DMatrix::from_row_slice(3, 3, &[1.2, 0.8, 0.6, 0.9, 1.1, 0.7, 0.5, 0.6, 1.4]);
        let normalized = sinkhorn_normalize(matrix, 24);
        for i in 0..3 {
            let row_sum: f64 = (0..3).map(|j| normalized[(i, j)]).sum();
            let col_sum: f64 = (0..3).map(|j| normalized[(j, i)]).sum();
            assert!((row_sum - 1.0).abs() < 1e-9);
            assert!((col_sum - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn c069_angles_are_boolean() {
        let summary = summarize_c069();
        assert_eq!(summary.unique_angle_degrees.len(), 2);
        assert!(summary.unique_angle_degrees[0].abs() < 1e-9);
        assert!((summary.unique_angle_degrees[1] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn givens_fit_is_exact_for_random_target() {
        let target = [17.0, 31.0, 53.0];
        let matrix = givens_rotation_matrix(target[0], target[1], target[2]);
        let recovered = recover_givens_angles_deg(&matrix);
        for idx in 0..3 {
            assert!((target[idx] - recovered[idx]).abs() < 1e-9);
        }
    }

    #[test]
    fn planck_numerology_is_unit_dependent() {
        let summary = summarize_c1355();
        assert!(summary.cgs_contains_1764);
        assert!(summary.si_contains_1764);
        assert!(!summary.gev_contains_1764);
        assert!(!summary.planck_units_contains_1764);
    }
}
