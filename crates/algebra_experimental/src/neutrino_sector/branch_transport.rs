use super::{construct_pmns_matrices_two_param, extract_pmns_angles};
use crate::quark_sector::extract_ckm_permutation_aware;
use flavor_lifts::{
    TensorElementLift, apply_v6_perturbation, compute_constrained_atmospheric_direction,
    compute_constrained_solar_direction, extract_v6_basis,
};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GradientFrame {
    pub alpha_ch: f64,
    pub alpha_nu: f64,
    pub base_angles: (f64, f64, f64),
    pub perm_u: [usize; 3],
    pub perm_d: [usize; 3],
    pub g_12: [f64; 6],
    pub g_13: [f64; 6],
    pub g_23: [f64; 6],
    pub u_solar: [f64; 6],
    pub u_atmo: [f64; 6],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BranchMapRow {
    pub alpha_ch: f64,
    pub alpha_nu: f64,
    pub perm_match: bool,
    pub perm_u: [usize; 3],
    pub perm_d: [usize; 3],
    pub align_g12: f64,
    pub align_g13: f64,
    pub align_g23: f64,
    pub align_u_solar: f64,
    pub align_u_atmo: f64,
    pub abs_align_g23: f64,
    pub abs_align_u_atmo: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BranchMapReport {
    pub base_branch: String,
    pub same_perm_points: usize,
    pub switched_perm_points: usize,
    pub rows: Vec<BranchMapRow>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BranchWallRow {
    pub kind: String,
    pub alpha_ch_0: f64,
    pub alpha_nu_0: f64,
    pub alpha_ch_1: f64,
    pub alpha_nu_1: f64,
    pub branch_0: String,
    pub branch_1: String,
    pub align_g23: f64,
    pub align_u_atmo: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BranchWallReport {
    pub count: usize,
    pub rows: Vec<BranchWallRow>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PathScanRow {
    pub alpha_ch: f64,
    pub alpha_nu: f64,
    pub branch: String,
    pub perm_match: bool,
    pub align_g12: f64,
    pub align_g13: f64,
    pub align_g23: f64,
    pub align_u_solar: f64,
    pub align_u_atmo: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PathScanReport {
    pub label: String,
    pub rows: Vec<PathScanRow>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LoopStep {
    pub step: usize,
    pub alpha_ch: f64,
    pub alpha_nu: f64,
    pub branch: String,
    pub wall_crossed: bool,
    pub flip_g12: i32,
    pub flip_g13: i32,
    pub flip_g23: i32,
    pub flip_u_solar: i32,
    pub flip_u_atmo: i32,
    pub align_g23: f64,
    pub align_u_atmo: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LoopSummary {
    pub label: String,
    pub wall_crossings: usize,
    pub final_align_g12: f64,
    pub final_align_g13: f64,
    pub final_align_g23: f64,
    pub final_align_u_solar: f64,
    pub final_align_u_atmo: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LoopReport {
    pub steps: Vec<LoopStep>,
    pub summary: LoopSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct V6ProbeArtifacts {
    pub singular_values: Vec<f64>,
    pub base_frame: GradientFrame,
    pub branch_map: BranchMapReport,
    pub branch_walls: BranchWallReport,
    pub fixed_alpha_ch_scan: PathScanReport,
    pub fixed_alpha_nu_scan: PathScanReport,
    pub stable_branch_loop: LoopReport,
    pub wall_crossing_loop: LoopReport,
}

#[derive(Clone, Debug)]
struct OrientedFrame {
    g_12: [f64; 6],
    g_13: [f64; 6],
    g_23: [f64; 6],
    u_solar: [f64; 6],
    u_atmo: [f64; 6],
}

pub fn dot(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f64; 6]) -> f64 {
    dot(a, a).sqrt()
}

pub fn alignment(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let denom = norm(a) * norm(b);
    if denom < 1e-15 {
        0.0
    } else {
        dot(a, b) / denom
    }
}

fn negate(a: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0_f64; 6];
    for i in 0..6 {
        out[i] = -a[i];
    }
    out
}

fn orient_to_reference(reference: &[f64; 6], candidate: &[f64; 6]) -> ([f64; 6], i32, f64) {
    let align = alignment(reference, candidate);
    if align < 0.0 {
        (negate(candidate), -1, -align)
    } else {
        (*candidate, 1, align)
    }
}

fn oriented_from_frame(frame: &GradientFrame) -> OrientedFrame {
    OrientedFrame {
        g_12: frame.g_12,
        g_13: frame.g_13,
        g_23: frame.g_23,
        u_solar: frame.u_solar,
        u_atmo: frame.u_atmo,
    }
}

pub fn perm_label(frame: &GradientFrame) -> String {
    format!("u{:?}_d{:?}", frame.perm_u, frame.perm_d)
}

fn pmns_angles_at(
    m_ch_base: &faer::Mat<f64>,
    m_nu_base: &faer::Mat<f64>,
    perm_u: &[usize; 3],
    perm_d: &[usize; 3],
    v6_basis: &DMatrix<f64>,
    beta: &[f64; 6],
) -> (f64, f64, f64) {
    let lift = TensorElementLift;
    let eig_ch = m_ch_base.selfadjoint_eigendecomposition(faer::Side::Lower);

    let mut m_nu = m_nu_base.clone();
    apply_v6_perturbation(&mut m_nu, v6_basis, beta, &lift);
    let m_nu = (&m_nu + m_nu.transpose()) * faer::scale(0.5);
    let eig_nu = m_nu.selfadjoint_eigendecomposition(faer::Side::Lower);

    let u_raw = eig_ch.u().transpose() * eig_nu.u();
    let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            u_pmns.write(i, j, u_raw.read(perm_u[i], perm_d[j]));
        }
    }
    extract_pmns_angles(&u_pmns)
}

pub fn compute_gradient_frame(
    v6_basis: &DMatrix<f64>,
    alpha_ch: f64,
    alpha_nu: f64,
) -> GradientFrame {
    let charged_pair = (11_usize, 12_usize);
    let neutrino_pair = (7_usize, 8_usize);
    let eps = 0.05_f64;
    let n_basis = v6_basis.nrows().min(6);

    let (m_ch_base_raw, m_nu_base_raw) =
        construct_pmns_matrices_two_param(charged_pair, neutrino_pair, alpha_ch, alpha_nu);
    let m_ch_base = (&m_ch_base_raw + m_ch_base_raw.transpose()) * faer::scale(0.5);
    let m_nu_base = (&m_nu_base_raw + m_nu_base_raw.transpose()) * faer::scale(0.5);

    let eig_ch_0 = m_ch_base.selfadjoint_eigendecomposition(faer::Side::Lower);
    let eig_nu_0 = m_nu_base.selfadjoint_eigendecomposition(faer::Side::Lower);
    let u_raw_0 = eig_ch_0.u().transpose() * eig_nu_0.u();
    let (u_pmns_0, perm_u, perm_d) = extract_ckm_permutation_aware(&u_raw_0);
    let base_angles = extract_pmns_angles(&u_pmns_0);

    let mut g_12 = [0.0_f64; 6];
    let mut g_13 = [0.0_f64; 6];
    let mut g_23 = [0.0_f64; 6];

    for mu in 0..n_basis {
        let mut bp = [0.0_f64; 6];
        let mut bm = [0.0_f64; 6];
        bp[mu] = eps;
        bm[mu] = -eps;

        let (t12_p, t13_p, t23_p) =
            pmns_angles_at(&m_ch_base, &m_nu_base, &perm_u, &perm_d, v6_basis, &bp);
        let (t12_m, t13_m, t23_m) =
            pmns_angles_at(&m_ch_base, &m_nu_base, &perm_u, &perm_d, v6_basis, &bm);

        g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
        g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
        g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
    }

    let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
    let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

    GradientFrame {
        alpha_ch,
        alpha_nu,
        base_angles,
        perm_u,
        perm_d,
        g_12,
        g_13,
        g_23,
        u_solar,
        u_atmo,
    }
}

pub fn compute_branch_map(
    base: &GradientFrame,
    v6_basis: &DMatrix<f64>,
    alpha_ch_values: &[f64],
    alpha_nu_values: &[f64],
) -> BranchMapReport {
    let mut same_perm_points = 0_usize;
    let mut switched_perm_points = 0_usize;
    let mut rows = Vec::new();

    for &alpha_ch in alpha_ch_values {
        for &alpha_nu in alpha_nu_values {
            let frame = compute_gradient_frame(v6_basis, alpha_ch, alpha_nu);
            let perm_match = frame.perm_u == base.perm_u && frame.perm_d == base.perm_d;
            if perm_match {
                same_perm_points += 1;
            } else {
                switched_perm_points += 1;
            }

            let align_g23 = alignment(&base.g_23, &frame.g_23);
            let align_u_atmo = alignment(&base.u_atmo, &frame.u_atmo);
            rows.push(BranchMapRow {
                alpha_ch,
                alpha_nu,
                perm_match,
                perm_u: frame.perm_u,
                perm_d: frame.perm_d,
                align_g12: alignment(&base.g_12, &frame.g_12),
                align_g13: alignment(&base.g_13, &frame.g_13),
                align_g23,
                align_u_solar: alignment(&base.u_solar, &frame.u_solar),
                align_u_atmo,
                abs_align_g23: align_g23.abs(),
                abs_align_u_atmo: align_u_atmo.abs(),
            });
        }
    }

    BranchMapReport {
        base_branch: perm_label(base),
        same_perm_points,
        switched_perm_points,
        rows,
    }
}

pub fn compute_branch_walls(
    base: &GradientFrame,
    v6_basis: &DMatrix<f64>,
    alpha_ch_values: &[f64],
    alpha_nu_values: &[f64],
) -> BranchWallReport {
    let mut frames = Vec::new();
    for &alpha_ch in alpha_ch_values {
        let mut row = Vec::new();
        for &alpha_nu in alpha_nu_values {
            row.push(compute_gradient_frame(v6_basis, alpha_ch, alpha_nu));
        }
        frames.push(row);
    }

    let mut rows = Vec::new();
    for i in 0..frames.len() {
        for j in 0..frames[i].len() {
            if i + 1 < frames.len() {
                let a = &frames[i][j];
                let b = &frames[i + 1][j];
                if a.perm_u != b.perm_u || a.perm_d != b.perm_d {
                    rows.push(BranchWallRow {
                        kind: "vertical".to_string(),
                        alpha_ch_0: a.alpha_ch,
                        alpha_nu_0: a.alpha_nu,
                        alpha_ch_1: b.alpha_ch,
                        alpha_nu_1: b.alpha_nu,
                        branch_0: perm_label(a),
                        branch_1: perm_label(b),
                        align_g23: alignment(&base.g_23, &b.g_23),
                        align_u_atmo: alignment(&base.u_atmo, &b.u_atmo),
                    });
                }
            }
            if j + 1 < frames[i].len() {
                let a = &frames[i][j];
                let b = &frames[i][j + 1];
                if a.perm_u != b.perm_u || a.perm_d != b.perm_d {
                    rows.push(BranchWallRow {
                        kind: "horizontal".to_string(),
                        alpha_ch_0: a.alpha_ch,
                        alpha_nu_0: a.alpha_nu,
                        alpha_ch_1: b.alpha_ch,
                        alpha_nu_1: b.alpha_nu,
                        branch_0: perm_label(a),
                        branch_1: perm_label(b),
                        align_g23: alignment(&base.g_23, &b.g_23),
                        align_u_atmo: alignment(&base.u_atmo, &b.u_atmo),
                    });
                }
            }
        }
    }

    BranchWallReport {
        count: rows.len(),
        rows,
    }
}

pub fn compute_loop_transport(
    label: &str,
    v6_basis: &DMatrix<f64>,
    points: &[(f64, f64)],
) -> LoopReport {
    let frames: Vec<GradientFrame> = points
        .iter()
        .map(|&(alpha_ch, alpha_nu)| compute_gradient_frame(v6_basis, alpha_ch, alpha_nu))
        .collect();

    let start = &frames[0];
    let start_oriented = oriented_from_frame(start);
    let mut prev_oriented = start_oriented.clone();
    let mut prev_frame = start;
    let mut wall_crossings = 0_usize;
    let mut steps = Vec::new();

    steps.push(LoopStep {
        step: 0,
        alpha_ch: start.alpha_ch,
        alpha_nu: start.alpha_nu,
        branch: perm_label(start),
        wall_crossed: false,
        flip_g12: 1,
        flip_g13: 1,
        flip_g23: 1,
        flip_u_solar: 1,
        flip_u_atmo: 1,
        align_g23: 1.0,
        align_u_atmo: 1.0,
    });

    for (step, frame) in frames.iter().enumerate().skip(1) {
        let wall_crossed = frame.perm_u != prev_frame.perm_u || frame.perm_d != prev_frame.perm_d;
        if wall_crossed {
            wall_crossings += 1;
        }

        let (g_12, flip_g12, _) = orient_to_reference(&prev_oriented.g_12, &frame.g_12);
        let (g_13, flip_g13, _) = orient_to_reference(&prev_oriented.g_13, &frame.g_13);
        let (g_23, flip_g23, align_g23) = orient_to_reference(&prev_oriented.g_23, &frame.g_23);
        let (u_solar, flip_u_solar, _) =
            orient_to_reference(&prev_oriented.u_solar, &frame.u_solar);
        let (u_atmo, flip_u_atmo, align_u_atmo) =
            orient_to_reference(&prev_oriented.u_atmo, &frame.u_atmo);

        steps.push(LoopStep {
            step,
            alpha_ch: frame.alpha_ch,
            alpha_nu: frame.alpha_nu,
            branch: perm_label(frame),
            wall_crossed,
            flip_g12,
            flip_g13,
            flip_g23,
            flip_u_solar,
            flip_u_atmo,
            align_g23,
            align_u_atmo,
        });

        prev_oriented = OrientedFrame {
            g_12,
            g_13,
            g_23,
            u_solar,
            u_atmo,
        };
        prev_frame = frame;
    }

    LoopReport {
        steps,
        summary: LoopSummary {
            label: label.to_string(),
            wall_crossings,
            final_align_g12: alignment(&start_oriented.g_12, &prev_oriented.g_12),
            final_align_g13: alignment(&start_oriented.g_13, &prev_oriented.g_13),
            final_align_g23: alignment(&start_oriented.g_23, &prev_oriented.g_23),
            final_align_u_solar: alignment(&start_oriented.u_solar, &prev_oriented.u_solar),
            final_align_u_atmo: alignment(&start_oriented.u_atmo, &prev_oriented.u_atmo),
        },
    }
}

pub fn compute_path_scan(
    label: &str,
    base: &GradientFrame,
    v6_basis: &DMatrix<f64>,
    points: &[(f64, f64)],
) -> PathScanReport {
    let rows = points
        .iter()
        .map(|&(alpha_ch, alpha_nu)| {
            let frame = compute_gradient_frame(v6_basis, alpha_ch, alpha_nu);
            PathScanRow {
                alpha_ch,
                alpha_nu,
                branch: perm_label(&frame),
                perm_match: frame.perm_u == base.perm_u && frame.perm_d == base.perm_d,
                align_g12: alignment(&base.g_12, &frame.g_12),
                align_g13: alignment(&base.g_13, &frame.g_13),
                align_g23: alignment(&base.g_23, &frame.g_23),
                align_u_solar: alignment(&base.u_solar, &frame.u_solar),
                align_u_atmo: alignment(&base.u_atmo, &frame.u_atmo),
            }
        })
        .collect();
    PathScanReport {
        label: label.to_string(),
        rows,
    }
}

pub fn default_alpha_ch_values() -> Vec<f64> {
    (28..=38).map(|x| x as f64 / 10.0).collect()
}

pub fn default_alpha_nu_values() -> Vec<f64> {
    (24..=29).map(|x| x as f64 / 20.0).collect()
}

pub fn stable_branch_loop_points() -> Vec<(f64, f64)> {
    vec![(3.00, 1.30), (3.60, 1.30), (3.60, 1.40), (3.00, 1.40), (3.00, 1.30)]
}

pub fn wall_crossing_loop_points() -> Vec<(f64, f64)> {
    vec![(3.00, 1.25), (3.00, 1.20), (3.10, 1.20), (3.10, 1.25), (3.00, 1.25)]
}

pub fn fixed_alpha_ch_scan_points() -> Vec<(f64, f64)> {
    vec![(3.00, 1.20), (3.00, 1.25), (3.00, 1.30), (3.00, 1.35), (3.00, 1.40), (3.00, 1.45)]
}

pub fn fixed_alpha_nu_scan_points() -> Vec<(f64, f64)> {
    vec![(2.80, 1.35), (3.00, 1.35), (3.20, 1.35), (3.40, 1.35), (3.60, 1.35), (3.80, 1.35)]
}

pub fn default_probe_artifacts() -> V6ProbeArtifacts {
    let (v6_basis, singular_values, _assessors) = extract_v6_basis();
    let base_frame = compute_gradient_frame(&v6_basis, 3.00, 1.35);
    let alpha_ch_values = default_alpha_ch_values();
    let alpha_nu_values = default_alpha_nu_values();
    let branch_map = compute_branch_map(&base_frame, &v6_basis, &alpha_ch_values, &alpha_nu_values);
    let branch_walls =
        compute_branch_walls(&base_frame, &v6_basis, &alpha_ch_values, &alpha_nu_values);
    let fixed_alpha_ch_scan = compute_path_scan(
        "fixed_alpha_ch_3.00",
        &base_frame,
        &v6_basis,
        &fixed_alpha_ch_scan_points(),
    );
    let fixed_alpha_nu_scan = compute_path_scan(
        "fixed_alpha_nu_1.35",
        &base_frame,
        &v6_basis,
        &fixed_alpha_nu_scan_points(),
    );
    let stable_branch_loop =
        compute_loop_transport("stable_branch_loop", &v6_basis, &stable_branch_loop_points());
    let wall_crossing_loop =
        compute_loop_transport("wall_crossing_loop", &v6_basis, &wall_crossing_loop_points());
    V6ProbeArtifacts {
        singular_values,
        base_frame,
        branch_map,
        branch_walls,
        fixed_alpha_ch_scan,
        fixed_alpha_nu_scan,
        stable_branch_loop,
        wall_crossing_loop,
    }
}
