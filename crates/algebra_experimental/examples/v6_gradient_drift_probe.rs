use algebra_experimental::neutrino_sector::{
    TensorElementLift, apply_v6_perturbation, compute_constrained_atmospheric_direction,
    compute_constrained_solar_direction, construct_pmns_matrices_two_param, extract_pmns_angles,
    extract_v6_basis,
};
use algebra_experimental::quark_sector::extract_ckm_permutation_aware;
use nalgebra::DMatrix;

#[derive(Clone, Debug)]
struct GradientFrame {
    alpha_ch: f64,
    alpha_nu: f64,
    base_angles: (f64, f64, f64),
    perm_u: [usize; 3],
    perm_d: [usize; 3],
    g_12: [f64; 6],
    g_13: [f64; 6],
    g_23: [f64; 6],
    u_solar: [f64; 6],
    u_atmo: [f64; 6],
}

fn dot(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64; 6]) -> f64 {
    dot(a, a).sqrt()
}

fn alignment(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let denom = norm(a) * norm(b);
    if denom < 1e-15 {
        0.0
    } else {
        dot(a, b) / denom
    }
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

fn compute_gradient_frame(v6_basis: &DMatrix<f64>, alpha_ch: f64, alpha_nu: f64) -> GradientFrame {
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

fn print_frame(frame: &GradientFrame) {
    println!(
        "frame alpha_ch={:.3} alpha_nu={:.3} base_angles=({:.3},{:.3},{:.3})",
        frame.alpha_ch,
        frame.alpha_nu,
        frame.base_angles.0,
        frame.base_angles.1,
        frame.base_angles.2
    );
    println!("  perm_u={:?} perm_d={:?}", frame.perm_u, frame.perm_d);
    println!(
        "  norms g12={:.6} g13={:.6} g23={:.6}",
        norm(&frame.g_12),
        norm(&frame.g_13),
        norm(&frame.g_23)
    );
    println!(
        "  cosines c12_13={:.6} c12_23={:.6} c13_23={:.6}",
        alignment(&frame.g_12, &frame.g_13),
        alignment(&frame.g_12, &frame.g_23),
        alignment(&frame.g_13, &frame.g_23)
    );
    println!(
        "  solar_selectivity g12.u={:.6} g13.u={:.6} g23.u={:.6}",
        dot(&frame.g_12, &frame.u_solar),
        dot(&frame.g_13, &frame.u_solar),
        dot(&frame.g_23, &frame.u_solar)
    );
}

fn print_comparison(base: &GradientFrame, other: &GradientFrame) {
    println!(
        "compare base=({:.3},{:.3}) other=({:.3},{:.3})",
        base.alpha_ch, base.alpha_nu, other.alpha_ch, other.alpha_nu
    );
    println!(
        "  signed_align g12={:.6} g13={:.6} g23={:.6} u_solar={:.6} u_atmo={:.6}",
        alignment(&base.g_12, &other.g_12),
        alignment(&base.g_13, &other.g_13),
        alignment(&base.g_23, &other.g_23),
        alignment(&base.u_solar, &other.u_solar),
        alignment(&base.u_atmo, &other.u_atmo)
    );
    println!(
        "  abs_align    g12={:.6} g13={:.6} g23={:.6} u_solar={:.6} u_atmo={:.6}",
        alignment(&base.g_12, &other.g_12).abs(),
        alignment(&base.g_13, &other.g_13).abs(),
        alignment(&base.g_23, &other.g_23).abs(),
        alignment(&base.u_solar, &other.u_solar).abs(),
        alignment(&base.u_atmo, &other.u_atmo).abs()
    );
}

fn perm_label(frame: &GradientFrame) -> String {
    format!("u{:?}_d{:?}", frame.perm_u, frame.perm_d)
}

fn sign_label(x: f64) -> &'static str {
    if x >= 0.0 { "+" } else { "-" }
}

fn print_transport_summary(base: &GradientFrame, other: &GradientFrame) {
    let align_g12 = alignment(&base.g_12, &other.g_12);
    let align_g13 = alignment(&base.g_13, &other.g_13);
    let align_g23 = alignment(&base.g_23, &other.g_23);
    let align_u_solar = alignment(&base.u_solar, &other.u_solar);
    let align_u_atmo = alignment(&base.u_atmo, &other.u_atmo);
    println!(
        "transport alpha_ch={:.2} alpha_nu={:.2} branch={} g12={} g13={} g23={} solar={} atmo={}",
        other.alpha_ch,
        other.alpha_nu,
        perm_label(other),
        sign_label(align_g12),
        sign_label(align_g13),
        sign_label(align_g23),
        sign_label(align_u_solar),
        sign_label(align_u_atmo)
    );
}

fn print_branch_map(base: &GradientFrame, v6_basis: &DMatrix<f64>) {
    let mut same_perm_count = 0_usize;
    let mut switched_perm_count = 0_usize;

    println!();
    println!("branch_map_csv");
    println!(
        "alpha_ch,alpha_nu,perm_match,perm_u,perm_d,align_g12,align_g13,align_g23,align_u_solar,align_u_atmo,abs_align_g23,abs_align_u_atmo"
    );

    for alpha_ch_i in 28..=38 {
        let alpha_ch = alpha_ch_i as f64 / 10.0;
        for alpha_nu_i in 24..=29 {
            let alpha_nu = alpha_nu_i as f64 / 20.0;
            let frame = compute_gradient_frame(v6_basis, alpha_ch, alpha_nu);
            let perm_match = frame.perm_u == base.perm_u && frame.perm_d == base.perm_d;
            if perm_match {
                same_perm_count += 1;
            } else {
                switched_perm_count += 1;
            }

            let align_g12 = alignment(&base.g_12, &frame.g_12);
            let align_g13 = alignment(&base.g_13, &frame.g_13);
            let align_g23 = alignment(&base.g_23, &frame.g_23);
            let align_u_solar = alignment(&base.u_solar, &frame.u_solar);
            let align_u_atmo = alignment(&base.u_atmo, &frame.u_atmo);

            println!(
                "{:.2},{:.2},{},{:?},{:?},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                alpha_ch,
                alpha_nu,
                perm_match,
                frame.perm_u,
                frame.perm_d,
                align_g12,
                align_g13,
                align_g23,
                align_u_solar,
                align_u_atmo,
                align_g23.abs(),
                align_u_atmo.abs()
            );
        }
    }

    println!();
    println!(
        "branch_map_summary same_perm_points={} switched_perm_points={}",
        same_perm_count, switched_perm_count
    );
}

fn print_path_scan(
    label: &str,
    base: &GradientFrame,
    v6_basis: &DMatrix<f64>,
    points: &[(f64, f64)],
) {
    println!();
    println!("path_scan label={}", label);
    println!(
        "alpha_ch,alpha_nu,branch,perm_match,align_g12,align_g13,align_g23,align_u_solar,align_u_atmo"
    );
    for &(alpha_ch, alpha_nu) in points {
        let frame = compute_gradient_frame(v6_basis, alpha_ch, alpha_nu);
        println!(
            "{:.2},{:.2},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            alpha_ch,
            alpha_nu,
            perm_label(&frame),
            frame.perm_u == base.perm_u && frame.perm_d == base.perm_d,
            alignment(&base.g_12, &frame.g_12),
            alignment(&base.g_13, &frame.g_13),
            alignment(&base.g_23, &frame.g_23),
            alignment(&base.u_solar, &frame.u_solar),
            alignment(&base.u_atmo, &frame.u_atmo)
        );
        print_transport_summary(base, &frame);
    }
}

fn print_branch_walls(base: &GradientFrame, v6_basis: &DMatrix<f64>) {
    let alpha_ch_values: Vec<f64> = (28..=38).map(|x| x as f64 / 10.0).collect();
    let alpha_nu_values: Vec<f64> = (24..=29).map(|x| x as f64 / 20.0).collect();
    let mut frames = Vec::new();
    for &alpha_ch in &alpha_ch_values {
        let mut row = Vec::new();
        for &alpha_nu in &alpha_nu_values {
            row.push(compute_gradient_frame(v6_basis, alpha_ch, alpha_nu));
        }
        frames.push(row);
    }

    println!();
    println!("branch_walls_csv");
    println!(
        "kind,alpha_ch_0,alpha_nu_0,alpha_ch_1,alpha_nu_1,branch_0,branch_1,align_g23,align_u_atmo"
    );

    let mut wall_count = 0_usize;
    for i in 0..frames.len() {
        for j in 0..frames[i].len() {
            if i + 1 < frames.len() {
                let a = &frames[i][j];
                let b = &frames[i + 1][j];
                if a.perm_u != b.perm_u || a.perm_d != b.perm_d {
                    wall_count += 1;
                    println!(
                        "vertical,{:.2},{:.2},{:.2},{:.2},{},{},{:.6},{:.6}",
                        a.alpha_ch,
                        a.alpha_nu,
                        b.alpha_ch,
                        b.alpha_nu,
                        perm_label(a),
                        perm_label(b),
                        alignment(&base.g_23, &b.g_23),
                        alignment(&base.u_atmo, &b.u_atmo)
                    );
                }
            }
            if j + 1 < frames[i].len() {
                let a = &frames[i][j];
                let b = &frames[i][j + 1];
                if a.perm_u != b.perm_u || a.perm_d != b.perm_d {
                    wall_count += 1;
                    println!(
                        "horizontal,{:.2},{:.2},{:.2},{:.2},{},{},{:.6},{:.6}",
                        a.alpha_ch,
                        a.alpha_nu,
                        b.alpha_ch,
                        b.alpha_nu,
                        perm_label(a),
                        perm_label(b),
                        alignment(&base.g_23, &b.g_23),
                        alignment(&base.u_atmo, &b.u_atmo)
                    );
                }
            }
        }
    }

    println!("branch_wall_summary count={}", wall_count);
}

fn main() {
    let (v6_basis, singular_values, _assessors) = extract_v6_basis();
    println!(
        "v6_probe basis_rows={} basis_cols={} singular_values={:?}",
        v6_basis.nrows(),
        v6_basis.ncols(),
        singular_values
    );
    println!("note structural_v6_subspace_is_fixed=true");
    println!("note drift_object=gradient_selected_frame_inside_v6");

    let points = [
        (3.00_f64, 1.35_f64),
        (3.05_f64, 1.35_f64),
        (3.00_f64, 1.40_f64),
        (3.75_f64, 1.30_f64),
    ];

    let frames: Vec<GradientFrame> = points
        .iter()
        .map(|&(alpha_ch, alpha_nu)| compute_gradient_frame(&v6_basis, alpha_ch, alpha_nu))
        .collect();

    for frame in &frames {
        print_frame(frame);
    }

    let base = &frames[0];
    for frame in frames.iter().skip(1) {
        print_comparison(base, frame);
    }

    println!();
    println!("base_branch={}", perm_label(base));
    print_branch_map(base, &v6_basis);
    print_path_scan(
        "fixed_alpha_ch_3.00",
        base,
        &v6_basis,
        &[
            (3.00, 1.20),
            (3.00, 1.25),
            (3.00, 1.30),
            (3.00, 1.35),
            (3.00, 1.40),
            (3.00, 1.45),
        ],
    );
    print_path_scan(
        "fixed_alpha_nu_1.35",
        base,
        &v6_basis,
        &[
            (2.80, 1.35),
            (3.00, 1.35),
            (3.20, 1.35),
            (3.40, 1.35),
            (3.60, 1.35),
            (3.80, 1.35),
        ],
    );
    print_branch_walls(base, &v6_basis);
}
