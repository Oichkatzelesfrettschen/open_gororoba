use super::super::*;

/// CP violation baseline: real PMNS matrices have J = 0, delta_CP = 0.
///
/// This establishes the interface for future complex-phase extension
/// via the G2 stabilizer's Fano pair signs.
#[test]
fn test_cp_phase_baseline_real_pmns() {
    let result = compute_pmns((11, 12), (7, 8));
    assert!(
        result.jarlskog_invariant.abs() < 1e-15,
        "Real PMNS matrix must have J = 0, got {}",
        result.jarlskog_invariant
    );
    assert!(
        result.cp_phase_deg.abs() < 1e-10,
        "Real PMNS matrix must have delta_CP = 0, got {}",
        result.cp_phase_deg
    );
    println!(
        "  CP phase baseline: J = {:.2e}, delta_CP = {:.2} deg",
        result.jarlskog_invariant, result.cp_phase_deg
    );
    println!("  PDG 2024: delta_CP ~ 195 deg (normal ordering)");
    println!("  Next: complex phase from Fano pair signs (G2 stabilizer J_k)");
}

/// Verify extract_cp_phase correctly recovers delta from known J.
#[test]
fn test_cp_phase_extraction_formula() {
    // PDG values: theta_12=33.41, theta_13=8.54, theta_23=49.0
    // delta_CP=195 deg -> J ~ -0.0334
    let t12 = 33.41_f64;
    let t13 = 8.54_f64;
    let t23 = 49.0_f64;
    let delta = 195.0_f64;
    let s12 = t12.to_radians().sin();
    let c12 = t12.to_radians().cos();
    let s13 = t13.to_radians().sin();
    let c13 = t13.to_radians().cos();
    let s23 = t23.to_radians().sin();
    let c23 = t23.to_radians().cos();
    let j_pdg = s12 * c12 * s23 * c23 * s13 * c13 * c13 * delta.to_radians().sin();

    let recovered = extract_cp_phase((t12, t13, t23), j_pdg);
    // extract_cp_phase returns asin(sin(delta)), which maps 195 -> -15 (mod 360)
    // since sin(195) = sin(-15) = -sin(15).
    let sin_delta = delta.to_radians().sin();
    let sin_recovered = recovered.to_radians().sin();
    assert!(
        (sin_delta - sin_recovered).abs() < 1e-10,
        "sin(delta) mismatch: expected {:.6}, got {:.6}",
        sin_delta,
        sin_recovered
    );
    println!("  J_PDG = {:.6}", j_pdg);
    println!(
        "  Recovered delta_CP = {:.2} deg (sin matches PDG 195 deg)",
        recovered
    );
}

// =======================================================================
// Phase B: Absolute neutrino masses
// =======================================================================

/// Mass-squared ratio r = dm21_sq / dm31_sq is a scale-free prediction.
/// PDG 2024 (normal ordering): r = 7.53e-5 / 2.453e-3 = 0.0307.
#[test]
fn test_mass_squared_ratio() {
    let result = compute_pmns((11, 12), (7, 8));
    let dm21 = result.delta_m21_sq;
    let dm31 = result.delta_m31_sq;

    println!(
        "  Mass eigenvalues (arb. units): {:?}",
        result.neutrino_masses
    );
    println!("  dm21_sq = {:.6e}", dm21);
    println!("  dm31_sq = {:.6e}", dm31);

    if dm31.abs() > 1e-30 {
        let r = dm21 / dm31;
        println!("  r = dm21_sq / dm31_sq = {:.6}", r);
        println!("  PDG 2024: r = 0.0307 (normal ordering)");

        // Register whether the ratio is in the right ballpark
        // (order of magnitude is the first test)
        if r > 0.0 && r < 1.0 {
            println!("  PASS: r is in (0, 1) -- normal ordering");
        } else if r > 1.0 {
            println!("  NOTE: r > 1 -- inverted ordering");
        } else {
            println!("  NOTE: r <= 0 -- degenerate spectrum");
        }
    } else {
        println!("  WARNING: dm31_sq ~ 0, degenerate spectrum");
    }
}

/// Reconstruct absolute masses from the algebraic ratio + m1 input.
/// Scan m1 from 0 to 0.1 eV and check cosmological bound.
#[test]
fn test_absolute_mass_reconstruction() {
    let result = compute_pmns((11, 12), (7, 8));
    let nu = &result.neutrino_masses;

    // Compute ratios relative to the lightest eigenvalue
    // The absolute scale drops out of the PMNS matrix; only ratios matter.
    let m_min = nu[0].max(1e-30);
    let r1 = nu[1] / m_min;
    let r2 = nu[2] / m_min;

    println!("  Mass ratios: m2/m1 = {:.4}, m3/m1 = {:.4}", r1, r2);

    // PDG 2024 mass-squared differences (normal ordering):
    //   dm21_sq = 7.53e-5 eV^2, dm31_sq = 2.453e-3 eV^2
    let pdg_dm21_sq = 7.53e-5_f64;
    let pdg_dm31_sq = 2.453e-3_f64;

    // For several m1 values, compute sum(m_i)
    println!("\n  Absolute mass reconstruction (using PDG dm^2 for scale):");
    println!(
        "  {:>8} | {:>8} {:>8} {:>8} | {:>10} | {:>6}",
        "m1 (eV)", "m1", "m2", "m3", "sum (eV)", "bound"
    );

    for m1_mev in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0] {
        let m1 = m1_mev * 1e-3; // meV -> eV
        let m2 = (m1 * m1 + pdg_dm21_sq).sqrt();
        let m3 = (m1 * m1 + pdg_dm31_sq).sqrt();
        let sum_m = m1 + m2 + m3;
        let bound = if sum_m < 0.12 { "OK" } else { "EXCLUDED" };
        println!(
            "  {:>8.4} | {:>8.5} {:>8.5} {:>8.5} | {:>10.5} | {:>6}",
            m1, m1, m2, m3, sum_m, bound
        );
    }
}

/// Effective electron-neutrino mass m_beta for KATRIN.
/// m_beta^2 = sum |U_ei|^2 m_i^2
#[test]
fn test_effective_electron_neutrino_mass() {
    let result = compute_pmns((11, 12), (7, 8));
    let u = &result.matrix;

    // Using PDG mass-squared diffs with m1 = 0 (lightest case)
    let pdg_dm21_sq = 7.53e-5_f64;
    let pdg_dm31_sq = 2.453e-3_f64;
    let m1 = 0.0_f64;
    let m2 = (m1 * m1 + pdg_dm21_sq).sqrt();
    let m3 = (m1 * m1 + pdg_dm31_sq).sqrt();
    let masses = [m1, m2, m3];

    let mut m_beta_sq = 0.0_f64;
    for i in 0..3 {
        let u_ei = u[(0, i)];
        m_beta_sq += u_ei * u_ei * masses[i] * masses[i];
    }
    let m_beta = m_beta_sq.sqrt();

    println!("  m_beta = {:.4} meV (m1=0 case)", m_beta * 1e3);
    println!("  KATRIN bound: m_beta < 450 meV (90% CL)");
    assert!(m_beta < 0.45, "m_beta = {} eV exceeds KATRIN bound", m_beta);

    // Also compute for m1 = 50 meV (near cosmological bound)
    let m1_heavy = 0.05_f64;
    let m2h = (m1_heavy * m1_heavy + pdg_dm21_sq).sqrt();
    let m3h = (m1_heavy * m1_heavy + pdg_dm31_sq).sqrt();
    let masses_h = [m1_heavy, m2h, m3h];
    let mut m_beta_sq_h = 0.0_f64;
    for i in 0..3 {
        let u_ei = u[(0, i)];
        m_beta_sq_h += u_ei * u_ei * masses_h[i] * masses_h[i];
    }
    let m_beta_h = m_beta_sq_h.sqrt();
    println!("  m_beta = {:.2} meV (m1=50 meV case)", m_beta_h * 1e3);
    println!("  sum(m_i) = {:.4} eV", m1_heavy + m2h + m3h);
}

// =======================================================================
// Phase A: CP violation via Fano pair complex phases
// =======================================================================

/// Explore the CP phase structure from Fano pair signs.
///
/// For each fixed imaginary unit k=1..7, the G2 stabilizer gives 3 Fano
/// lines with signs sigma_j in {+1, -1}. These signs define a complex
/// basis z_j = u_j - i*sigma_j*w_j. When the PMNS matrix is rephased
/// by these signs, it acquires a Dirac CP phase.
///
/// The rephasing: U_PMNS_complex = diag(sigma_1, sigma_2, sigma_3) * U_PMNS_real
/// This is a diagonal phase matrix acting on the flavor (generation) indices.
/// The Jarlskog invariant of the rephased matrix is:
///   J = sigma_1 * sigma_2 * sigma_3 * J_real
/// which is zero since J_real = 0 for a real matrix.
///
/// A more physical approach: the complex structure J_k introduces phases
/// into the OFF-DIAGONAL mass matrix elements. If M_ij for i != j picks up
/// a factor exp(i * phi_ij) where phi_ij comes from the Fano line connecting
/// generations i and j, then the diagonalization produces a genuinely
/// complex unitary matrix with nonzero J.
#[test]
fn test_cp_phase_from_fano_signs() {
    use gororoba_algebra::lie::g2_stabilizer::complex_structure;

    let result = compute_pmns((11, 12), (7, 8));

    println!("  === CP Phase from Fano Pair Signs ===\n");

    for k in 1..=7 {
        let cs = complex_structure(k);
        let signs: Vec<i8> = cs.fano_pairs.iter().map(|&(_, _, s)| s).collect();
        let sign_product: i8 = signs.iter().product();

        println!("  k={}: fano_pairs = {:?}", k, cs.fano_pairs);
        println!("       signs = {:?}, product = {}", signs, sign_product);

        // The Fano signs define a Z_2^3 phase structure.
        // For CP violation, we need a RELATIVE phase between mass matrix
        // elements. Consider the off-diagonal phase:
        // phi_ij = pi * (1 - sigma_line(i,j)) / 2
        // where sigma_line(i,j) is the sign of the Fano line connecting
        // the perp-index pairs for generations i and j.
        //
        // If sigma = +1: phi = 0 (no phase)
        // If sigma = -1: phi = pi (sign flip = exp(i*pi))

        // Count how many signs are -1 (these contribute pi phases)
        let n_negative = signs.iter().filter(|&&s| s == -1).count();
        println!(
            "       negative signs: {} -> discrete phase: {} * pi/3",
            n_negative, n_negative
        );
    }

    // The discrete Z_2^3 structure from Fano signs gives at most
    // exp(i*pi) = -1 phases. For a continuous CP phase like delta_CP = 195 deg,
    // we need a mechanism beyond discrete signs.
    //
    // The continuous mechanism: the psi automorphism (S_3 generator, order 3)
    // introduces exp(2*pi*i/3) = exp(i*120 deg) phases when cycling
    // O_1 -> O_2 -> O_3. This is the natural source of a CP phase near 180 deg.
    //
    // delta_CP ~ 180 + correction from Fano signs
    // PDG: delta_CP ~ 195 deg (= 180 + 15)

    println!("\n  The psi automorphism (order 3) provides the dominant phase:");
    println!("  exp(2*pi*i/3) = exp(i * 120 deg)");
    println!("  Combined with Fano sign corrections:");
    println!("  delta_CP ~ 180 + O(15) deg -- consistent with PDG 195 deg");

    // Compute what delta_CP would be if J comes from the psi automorphism:
    // The psi overlap <sel_i, psi(sel_j)> ~ cos(2*pi/3) = -0.5
    // The imaginary part ~ sin(2*pi/3) = sqrt(3)/2
    // J ~ sin(120 deg) * product_of_sines = sqrt(3)/2 * s12*c12*s23*c23*s13*c13^2
    let (t12, t13, t23) = result.angles_deg;
    let s12 = t12.to_radians().sin();
    let c12 = t12.to_radians().cos();
    let s13 = t13.to_radians().sin();
    let c13 = t13.to_radians().cos();
    let s23 = t23.to_radians().sin();
    let c23 = t23.to_radians().cos();

    let j_max = s12 * c12 * s23 * c23 * s13 * c13 * c13;
    let j_psi = j_max * (2.0 * std::f64::consts::PI / 3.0).sin();
    let delta_psi = extract_cp_phase((t12, t13, t23), j_psi);

    println!("\n  J_max (from angles) = {:.6}", j_max);
    println!("  J_psi (sin(120) * J_max) = {:.6}", j_psi);
    println!("  delta_CP_psi = {:.2} deg", delta_psi);
    println!("  PDG 2024: delta_CP = 195 +/- 25 deg");

    // This is a PREDICTION: the psi automorphism predicts
    // sin(delta_CP) = sin(120 deg) = sqrt(3)/2
    // delta_CP = 60 deg or 120 deg (from asin)
    // But the physical phase could be in the second quadrant: 180 - 60 = 120 or 180 + 60 = 240
    // or equivalently -120 deg = 240 deg.
    // PDG convention: delta_CP in [0, 360), so 195 is close to 180 + 15.
    //
    // The psi prediction gives |sin(delta)| = sqrt(3)/2 ~ 0.866.
    // PDG sin(195 deg) = sin(180+15) = -sin(15) = -0.259.
    // These don't match -- the psi automorphism alone doesn't give the right magnitude.
    //
    // This means the CP phase requires the FULL complex mass matrix construction
    // (task A1-A3), not just a simple rephasing argument.
    println!("\n  NOTE: Full complex mass matrix construction needed for delta_CP.");
    println!(
        "  The psi rephasing gives |sin(delta)| = {:.3}, PDG has |sin(195)| = {:.3}",
        (2.0 * std::f64::consts::PI / 3.0).sin(),
        195.0_f64.to_radians().sin().abs()
    );
}

// =======================================================================
// Phase C: Chi-squared global fit
// =======================================================================

/// Chi-squared evaluation at the current best-fit point.
#[test]
fn test_chi2_best_fit_point() {
    let pdg = Pdg2024::default();
    let result = compute_pmns((11, 12), (7, 8));
    let chi2 = chi_squared_pmns(&result, &pdg);
    let pulls = pmns_pulls(&result, &pdg);

    let (t12, t13, t23) = result.angles_deg;
    println!("  === Chi-squared Global Fit (3 angles) ===\n");
    println!(
        "  {:>10} | {:>8} | {:>8} | {:>6} | {:>6}",
        "Observable", "This", "PDG", "sigma", "Pull"
    );
    println!(
        "  {:-<10}-+-{:-<8}-+-{:-<8}-+-{:-<6}-+-{:-<6}",
        "", "", "", "", ""
    );
    for &(name, pull) in &pulls {
        let (val, pdg_val, err) = match name {
            "theta_12" => (t12, pdg.theta_12_deg, pdg.theta_12_err),
            "theta_13" => (t13, pdg.theta_13_deg, pdg.theta_13_err),
            "theta_23" => (t23, pdg.theta_23_deg, pdg.theta_23_err),
            _ => (0.0, 0.0, 1.0),
        };
        println!(
            "  {:>10} | {:>8.2} | {:>8.2} | {:>6.2} | {:>+6.2}",
            name, val, pdg_val, err, pull
        );
    }
    println!("\n  chi2 = {:.4} (3 dof)", chi2);
    println!("  chi2/dof = {:.4}", chi2 / 3.0);
    println!("  Note: Good fit if chi2/dof < 1. Tension if > 4.");
}

/// Selector pair scan: find the best (charged, neutrino) pair minimizing chi^2.
#[test]
fn test_chi2_selector_scan() {
    use rayon::prelude::*;

    let pdg = Pdg2024::default();

    // All valid selector pairs: (a, b) with 1 <= a < b <= 15
    let pairs: Vec<(usize, usize)> = (1..=15)
        .flat_map(|a| ((a + 1)..=15).map(move |b| (a, b)))
        .collect();

    // Scan all (charged, neutrino) combinations
    // Outer loop parallel via rayon, inner loop sequential.
    let results: Vec<(f64, (usize, usize), (usize, usize), (f64, f64, f64))> = pairs
        .par_iter()
        .flat_map_iter(|&ch| {
            pairs.iter().map(move |&nu| {
                let r = compute_pmns(ch, nu);
                let chi2 = chi_squared_pmns(&r, &pdg);
                (chi2, ch, nu, r.angles_deg)
            })
        })
        .collect();

    // Sort by chi2
    let mut sorted = results;
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("  === Chi-squared Selector Pair Scan ===\n");
    println!(
        "  {:>6} | {:>10} | {:>10} | {:>8} {:>8} {:>8}",
        "chi2", "charged", "neutrino", "t12", "t13", "t23"
    );
    println!(
        "  {:-<6}-+-{:-<10}-+-{:-<10}-+-{:-<8}-{:-<8}-{:-<8}",
        "", "", "", "", "", ""
    );

    for entry in sorted.iter().take(10) {
        let (chi2, ch, nu, (t12, t13, t23)) = entry;
        println!(
            "  {:>6.2} | {:>10?} | {:>10?} | {:>8.2} {:>8.2} {:>8.2}",
            chi2, ch, nu, t12, t13, t23
        );
    }

    let best = &sorted[0];
    println!(
        "\n  Best fit: chi2 = {:.4}, charged = {:?}, neutrino = {:?}",
        best.0, best.1, best.2
    );
    println!("  chi2/dof = {:.4} (3 dof)", best.0 / 3.0);
    println!("  Total pairs scanned: {}", sorted.len());
}

/// Regression test pinning the joint 4D PMNS optimum (C-1491).
///
/// Evaluates at the known-optimal (alpha_ch=3.50, alpha_nu=1.35, t_solar=1.54,
/// t_atmo=2.00) and verifies the angles match with strict tolerances.
/// Also computes the 3x4 Jacobian and reports condition number.
#[test]
fn test_pmns_4d_optimum_regression() {
    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let alpha_ch = 3.50;
    let alpha_nu = 1.35;
    let eps = 0.05_f64;

    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let lift = TensorElementLift;
    let n_basis = v6_basis.nrows().min(6);

    // Build at the known optimum
    let (m_ch_base, m_nu_base) =
        construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
    let eig_ch = m_ch_base.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu_0 = m_nu_base.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_raw_0 = eig_ch.U().transpose() * eig_nu_0.U();
    let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

    let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
        let mut m_nu = m_nu_base.clone();
        apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
        let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_raw = eig_ch.U().transpose() * eig_nu.U();
        let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                u_pmns[(i, j)] = u_raw[(perm_u[i], perm_d[j])];
            }
        }
        extract_pmns_angles(&u_pmns)
    };

    // Compute constrained directions
    let mut g_12 = [0.0_f64; 6];
    let mut g_13 = [0.0_f64; 6];
    let mut g_23 = [0.0_f64; 6];
    for mu in 0..n_basis {
        let mut bp = [0.0_f64; 6];
        bp[mu] = eps;
        let mut bm = [0.0_f64; 6];
        bm[mu] = -eps;
        let (t12_p, t13_p, t23_p) = angles_at(&bp);
        let (t12_m, t13_m, t23_m) = angles_at(&bm);
        g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
        g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
        g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
    }

    let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
    let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

    // Apply at t_solar=1.54, t_atmo=2.00
    let t_solar = 1.54_f64;
    let t_atmo = 2.00_f64;
    let mut beta_opt = [0.0_f64; 6];
    for k in 0..6 {
        beta_opt[k] = t_solar * u_solar[k] + t_atmo * u_atmo[k];
    }

    let (t12, t13, t23) = angles_at(&beta_opt);

    println!("--- 4D OPTIMUM REGRESSION ---");
    println!("  theta_12 = {:.4} deg (expected ~33.84)", t12);
    println!("  theta_13 = {:.4} deg (expected ~8.56)", t13);
    println!("  theta_23 = {:.4} deg (expected ~48.74)", t23);

    // Pin angles -- theta_13 tightest
    assert!(
        (t13 - 8.56).abs() < 0.05,
        "theta_13 regression FAILED: {:.4} (expected ~8.56)",
        t13
    );
    assert!(
        (t12 - 33.84).abs() < 0.5,
        "theta_12 regression FAILED: {:.4} (expected ~33.84)",
        t12
    );
    assert!(
        (t23 - 48.74).abs() < 0.5,
        "theta_23 regression FAILED: {:.4} (expected ~48.74)",
        t23
    );

    println!("  PASS: 4D optimum regression");
}

/// Chi-squared for the full pipeline at all established operating points.
///
/// Compares three levels of the PMNS pipeline against PDG 2024.
#[test]
fn test_chi2_full_pipeline_summary() {
    let pdg = Pdg2024::default();

    // Level 1: Diagonal-only baseline (selectors only)
    let r_diag = compute_pmns((11, 12), (7, 8));
    let chi2_diag = chi_squared_pmns(&r_diag, &pdg);

    // Level 2: Two-param psi coupling (C-1464 result)
    // theta_12 ~ 29.2, theta_13 ~ 8.64, theta_23 ~ 47.1
    let chi2_psi = ((29.2 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
        + ((8.64 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
        + ((47.1 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

    // Level 3: V_6 solar correction (C-1478/C-1490 result)
    let chi2_v6 = ((33.42 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
        + ((8.63 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
        + ((47.08 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

    // Level 4: 4D joint optimum (C-1491 result)
    let chi2_4d = ((33.84 - pdg.theta_12_deg) / pdg.theta_12_err).powi(2)
        + ((8.56 - pdg.theta_13_deg) / pdg.theta_13_err).powi(2)
        + ((48.74 - pdg.theta_23_deg) / pdg.theta_23_err).powi(2);

    println!("  === PMNS Chi-squared Summary (3 angles vs PDG 2024) ===\n");
    println!(
        "  {:>30} | {:>8} | {:>8} | {:>8} {:>8} {:>8}",
        "Pipeline level", "chi2", "chi2/3", "t12", "t13", "t23"
    );
    println!(
        "  {:-<30}-+-{:-<8}-+-{:-<8}-+-{:-<8}-{:-<8}-{:-<8}",
        "", "", "", "", "", ""
    );
    println!(
        "  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
        "Diagonal only",
        chi2_diag,
        chi2_diag / 3.0,
        r_diag.angles_deg.0,
        r_diag.angles_deg.1,
        r_diag.angles_deg.2
    );
    println!(
        "  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
        "Psi coupling (C-1464)",
        chi2_psi,
        chi2_psi / 3.0,
        29.2,
        8.64,
        47.1
    );
    println!(
        "  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
        "V_6 correction (C-1490)",
        chi2_v6,
        chi2_v6 / 3.0,
        33.42,
        8.63,
        47.08
    );
    println!(
        "  {:>30} | {:>8.2} | {:>8.2} | {:>8.2} {:>8.2} {:>8.2}",
        "4D joint optimum (C-1491)",
        chi2_4d,
        chi2_4d / 3.0,
        33.84,
        8.56,
        48.74
    );
    println!(
        "\n  PDG 2024 reference: theta_12={:.2} +/- {:.2}, theta_13={:.2} +/- {:.2}, theta_23={:.2} +/- {:.1}",
        pdg.theta_12_deg,
        pdg.theta_12_err,
        pdg.theta_13_deg,
        pdg.theta_13_err,
        pdg.theta_23_deg,
        pdg.theta_23_err
    );

    // Individual pulls at the 4D optimum
    println!("\n  --- Pulls at 4D optimum ---");
    let pulls_4d = [
        ("theta_12", (33.84 - pdg.theta_12_deg) / pdg.theta_12_err),
        ("theta_13", (8.56 - pdg.theta_13_deg) / pdg.theta_13_err),
        ("theta_23", (48.74 - pdg.theta_23_deg) / pdg.theta_23_err),
    ];
    for (name, pull) in &pulls_4d {
        println!("  {:>10}: {:>+.2} sigma", name, pull);
    }
    println!(
        "\n  Total chi2 at 4D optimum: {:.2} (3 observables, 4 parameters -> 0 effective dof)",
        chi2_4d
    );
}

/// Mass ordering prediction from the algebraic eigenvalue spectrum.
///
/// Normal ordering: m1 < m2 < m3 (dm31 > 0)
/// Inverted ordering: m3 < m1 < m2 (dm31 < 0)
///
/// The algebraic framework predicts the ordering from the sign of
/// dm31_sq = m3^2 - m1^2.
#[test]
fn test_mass_ordering_prediction() {
    // Use the two-param pipeline at the optimized point
    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let (_m_ch, m_nu) = construct_pmns_matrices_two_param(ch_pair, nu_pair, 3.75, 1.30);

    let eig_nu = m_nu.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let mut eigenvalues: Vec<f64> = (0..3).map(|i| eig_nu.S().column_vector()[i]).collect();

    let mut abs_eigenvalues: Vec<f64> = eigenvalues.iter().map(|e| e.abs()).collect();
    abs_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let m1 = abs_eigenvalues[0];
    let m2 = abs_eigenvalues[1];
    let m3 = abs_eigenvalues[2];

    let dm21_sq = m2 * m2 - m1 * m1;
    let dm31_sq = m3 * m3 - m1 * m1;
    let r = dm21_sq / dm31_sq;

    println!("  === Mass Ordering Prediction ===\n");
    println!(
        "  Raw eigenvalues: {:.6e}, {:.6e}, {:.6e}",
        eigenvalues[0], eigenvalues[1], eigenvalues[2]
    );
    println!("  |m_i| sorted:    {:.6e}, {:.6e}, {:.6e}", m1, m2, m3);
    println!("  dm21_sq = {:.6e}", dm21_sq);
    println!("  dm31_sq = {:.6e}", dm31_sq);
    println!("  r = dm21/dm31 = {:.6}", r);
    println!("  PDG: r = 0.0307 (normal ordering)");

    let ordering = if dm31_sq > 0.0 { "NORMAL" } else { "INVERTED" };
    println!("\n  Predicted ordering: {}", ordering);
    println!("  PDG 2024: Normal ordering preferred (>3 sigma)");

    // The hierarchy ratio tells us how separated the mass scales are
    let hierarchy = m3 / m1;
    println!("  m3/m1 = {:.2} (mass hierarchy strength)", hierarchy);

    // Check if the eigenvalue signs are all positive (physical consistency)
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let all_positive = eigenvalues.iter().all(|&e| e > 0.0);
    println!("  All eigenvalues positive: {}", all_positive);
    if !all_positive {
        println!("  (Negative eigenvalues indicate see-saw-like mechanism)");
    }
}

/// Mass ratio r = dm21/dm31 scan over alpha parameters.
///
/// PDG: r = 0.0307. Current baseline gives r = 0.213 (7x too large).
/// Scan alpha_ch x alpha_nu to find values giving r closer to PDG.
#[test]
fn test_mass_ratio_alpha_scan() {
    use rayon::prelude::*;

    let pdg = Pdg2024::default();
    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let pdg_r = 0.0307_f64;

    let grid: Vec<(f64, f64)> = (1..=80)
        .flat_map(|a| (1..=40).map(move |b| (a as f64 * 0.1, b as f64 * 0.1)))
        .collect();

    let results: Vec<_> = grid
        .par_iter()
        .map(|&(a_ch, a_nu)| {
            let (_m_ch, m_nu) = construct_pmns_matrices_two_param(ch_pair, nu_pair, a_ch, a_nu);
            let eig_nu = m_nu.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let mut ev: Vec<f64> = (0..3)
                .map(|i| eig_nu.S().column_vector()[i].abs())
                .collect();
            ev.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let dm21 = ev[1] * ev[1] - ev[0] * ev[0];
            let dm31 = ev[2] * ev[2] - ev[0] * ev[0];
            let r = if dm31.abs() > 1e-30 {
                dm21 / dm31
            } else {
                f64::MAX
            };

            // Also compute angles
            let (m_ch2, _) = construct_pmns_matrices_two_param(ch_pair, nu_pair, a_ch, a_nu);
            let eig_ch = m_ch2.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let u_raw = eig_ch.U().transpose() * eig_nu.U();
            let (u_pmns, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
            let (t12, t13, t23) = extract_pmns_angles(&u_pmns);
            let chi2 = super::pdg_score(t12, t13, t23, &pdg);

            let r_err = (r - pdg_r).abs() / pdg_r;
            (r_err, r, chi2, a_ch, a_nu, t12, t13, t23, ev[2] / ev[0])
        })
        .collect();

    // Sort by mass ratio accuracy
    let mut by_r = results.clone();
    by_r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("  === Mass Ratio r = dm21/dm31 Alpha Scan ===\n");
    println!("  PDG: r = {:.4}, m3/m1 ~ 50\n", pdg_r);
    println!("  Top 10 by ratio accuracy:");
    println!(
        "  {:>6} {:>6} | {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>6}",
        "a_ch", "a_nu", "r", "m3/m1", "t12", "t13", "t23", "chi2"
    );
    for e in by_r.iter().take(10) {
        println!(
            "  {:>6.1} {:>6.1} | {:>8.4} {:>8.1} | {:>8.2} {:>8.2} {:>8.2} | {:>6.1}",
            e.3, e.4, e.1, e.8, e.5, e.6, e.7, e.2
        );
    }

    // Find best combined (good r + good angles)
    let mut combined = results;
    combined.sort_by(|a, b| {
        let sa = a.0 + 0.001 * a.2; // r_error + weight * chi2_angles
        let sb = b.0 + 0.001 * b.2;
        sa.partial_cmp(&sb).unwrap()
    });

    println!("\n  Best combined (r accuracy + angle chi2):");
    for e in combined.iter().take(5) {
        println!(
            "  a_ch={:.1} a_nu={:.1}: r={:.4} (err {:.1}%), chi2={:.1}, t12={:.1} t13={:.1} t23={:.1}",
            e.3,
            e.4,
            e.1,
            e.0 * 100.0,
            e.2,
            e.5,
            e.6,
            e.7
        );
    }
}

/// Regression test pinning the Gauss-Newton 4D optimum (C-1492).
///
/// Evaluates at the known-optimal parameters and verifies angles match
/// with strict tolerances. This is the canonical pinned state of the
/// PMNS angle-sector fit.
#[test]
fn test_pmns_gauss_newton_regression() {
    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let alpha_ch = 3.00;
    let alpha_nu = 1.35;
    let eps = 0.05_f64;

    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let lift = TensorElementLift;
    let n_basis = v6_basis.nrows().min(6);

    let (m_ch_base, m_nu_base) =
        construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);
    let eig_ch = m_ch_base.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu_0 = m_nu_base.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_raw_0 = eig_ch.U().transpose() * eig_nu_0.U();
    let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

    let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
        let mut m_nu = m_nu_base.clone();
        apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
        let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_raw = eig_ch.U().transpose() * eig_nu.U();
        let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                u_pmns[(i, j)] = u_raw[(perm_u[i], perm_d[j])];
            }
        }
        extract_pmns_angles(&u_pmns)
    };

    // Compute constrained directions and apply at optimal t values
    let mut g_12 = [0.0_f64; 6];
    let mut g_13 = [0.0_f64; 6];
    let mut g_23 = [0.0_f64; 6];
    for mu in 0..n_basis {
        let mut bp = [0.0_f64; 6];
        bp[mu] = eps;
        let mut bm = [0.0_f64; 6];
        bm[mu] = -eps;
        let (t12_p, t13_p, t23_p) = angles_at(&bp);
        let (t12_m, t13_m, t23_m) = angles_at(&bm);
        g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
        g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
        g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
    }

    let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
    let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

    // Use Gauss-Newton to find the optimum
    let inner_angles = |t1: f64, t2: f64| -> (f64, f64, f64) {
        let mut beta = [0.0_f64; 6];
        for k in 0..6 {
            beta[k] = t1 * u_solar[k] + t2 * u_atmo[k];
        }
        angles_at(&beta)
    };

    let (t1, t2, (t12, t13, t23), score) = gauss_newton_2d(
        &inner_angles,
        1.5,
        0.0,
        (33.41, 8.54, 49.0),
        (1.0, 2.24, 1.0),
        15,
    );

    println!("--- GAUSS-NEWTON REGRESSION ---");
    println!("  t_solar = {:.4}, t_atmo = {:.4}", t1, t2);
    println!("  theta_12 = {:.4} deg (expected ~33.36)", t12);
    println!("  theta_13 = {:.4} deg (expected ~8.54)", t13);
    println!("  theta_23 = {:.4} deg (expected ~48.99)", t23);
    println!("  score = {:.6e}", score);

    // Strict tolerances -- theta_13 tightest
    assert!(
        (t13 - 8.54).abs() < 0.02,
        "theta_13 regression: {:.4} (tol 0.02)",
        t13
    );
    assert!(
        (t12 - 33.4).abs() < 0.2,
        "theta_12 regression: {:.4} (tol 0.2)",
        t12
    );
    assert!(
        (t23 - 49.0).abs() < 0.2,
        "theta_23 regression: {:.4} (tol 0.2)",
        t23
    );

    println!("  PASS: Gauss-Newton regression");
}

// =======================================================================
// Complex PMNS: CP violation from psi eigenspace decomposition
// =======================================================================

/// Construct complex mass matrices using psi-eigenspace decomposition.
///
/// The psi automorphism (order 3, cycles O_1->O_2->O_3) has eigenvalues
/// {1, omega, omega^2} where omega = exp(2*pi*i/3).
///
/// For a 16D friction profile v, the psi-eigenspace projections are:
///   P_1(v) = (v + psi(v) + psi^2(v)) / 3           (eigenvalue 1)
///   P_w(v) = (v + w^2*psi(v) + w*psi^2(v)) / 3     (eigenvalue omega)
///   P_w2(v) = (v + w*psi(v) + w^2*psi^2(v)) / 3    (eigenvalue omega^2)
///
/// The complex off-diagonal mass matrix element is:
///   M_ij = alpha * (<v_i, P_1(v_j)> + w*<v_i, P_w(v_j)> + w^2*<v_i, P_w2(v_j)>)
///        = alpha * (<v_i, v_j> + w*<v_i, psi_w(v_j)> + w^2*<v_i, psi_w2(v_j)>)
///
/// Simplified: M_ij = alpha * sum_k w^k * <v_i, psi^k(v_j)>
/// where the sum is over the three psi eigenvalues.
///
/// The imaginary part of M_ij carries the CP-violating phase.
#[test]
fn test_complex_pmns_cp_phase() {
    use cd_kernel::{gourlay_psi, gourlay_psi_n};

    let nu_pair = (7_usize, 8);

    // Build friction profiles (same as two-param pipeline)
    use crate::{bell_inequality::rotate_sparse, majorana_braiding::MajoranaMode};

    let (o1, o2, o3, sign_table) = super::psi_setup();
    let subs = [&o1, &o2, &o3];

    let nu_a = MajoranaMode {
        gamma_index: nu_pair.0 - 1,
        cd_basis_index: nu_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: nu_pair.1 - 1,
        cd_basis_index: nu_pair.1,
        cd_dim: 16,
    };

    let build_profile =
        |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j {
                    continue;
                }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

    let nu_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&nu_a, &nu_b, s))
        .collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    // omega = exp(2*pi*i/3) = -1/2 + i*sqrt(3)/2
    let omega_re = -0.5_f64;
    let omega_im = 3.0_f64.sqrt() / 2.0;

    println!("  === Complex PMNS: CP Phase from Psi Eigenspace ===\n");
    println!(
        "  omega = exp(2*pi*i/3) = {:.4} + {:.4}i\n",
        omega_re, omega_im
    );

    // Build complex off-diagonal elements for the neutrino mass matrix
    // M_ij^complex = sum_{k=0,1,2} omega^k * <profile_i, psi^k(profile_j)>
    let mut m_complex = [[(0.0_f64, 0.0_f64); 3]; 3]; // (re, im)

    for i in 0..3 {
        for j in 0..3 {
            // <v_i, psi^0(v_j)> = <v_i, v_j>  (k=0, omega^0 = 1)
            let c0 = dot16(&nu_profiles[i], &nu_profiles[j]);

            // <v_i, psi^1(v_j)>  (k=1, omega^1 = omega)
            let psi1_j = gourlay_psi(&nu_profiles[j]);
            let c1 = dot16(&nu_profiles[i], &psi1_j);

            // <v_i, psi^2(v_j)>  (k=2, omega^2 = omega*)
            let psi2_j = gourlay_psi_n(&nu_profiles[j], 2);
            let c2 = dot16(&nu_profiles[i], &psi2_j);

            // M_ij = c0 * 1 + c1 * omega + c2 * omega^2
            //      = c0 * 1 + c1 * (w_re + i*w_im) + c2 * (w_re - i*w_im)
            // (omega^2 = conjugate of omega for cube root of unity)
            let re = c0 + c1 * omega_re + c2 * omega_re;
            let im = c1 * omega_im - c2 * omega_im;

            m_complex[i][j] = (re, im);
        }
    }

    println!("  Complex neutrino mass matrix M_nu:");
    for i in 0..3 {
        let row: Vec<String> = (0..3)
            .map(|j| {
                let (re, im) = m_complex[i][j];
                if im.abs() < 1e-15 {
                    format!("{:>8.4}", re)
                } else {
                    format!("{:.4}{:+.4}i", re, im)
                }
            })
            .collect();
        println!("    [{}]", row.join("  "));
    }

    // Check Hermiticity: M_ij = M_ji*
    let mut max_herm_err = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let err_re = (m_complex[i][j].0 - m_complex[j][i].0).abs();
            let err_im = (m_complex[i][j].1 + m_complex[j][i].1).abs();
            max_herm_err = max_herm_err.max(err_re).max(err_im);
        }
    }
    println!("\n  Hermiticity error: {:.2e}", max_herm_err);

    // Check if any off-diagonal element has nonzero imaginary part
    let mut max_im = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                max_im = max_im.max(m_complex[i][j].1.abs());
            }
        }
    }
    println!("  Max off-diagonal |Im(M_ij)|: {:.6e}", max_im);

    if max_im > 1e-15 {
        println!("  NONZERO imaginary part detected -- CP violation present!");

        // The Jarlskog invariant is proportional to the imaginary part
        // of the off-diagonal products. For a 3x3 Hermitian matrix,
        // J ~ Im(M_12 * M_23 * M_31) / (mass differences)^3
        let m12 = (m_complex[0][1].0, m_complex[0][1].1);
        let m23 = (m_complex[1][2].0, m_complex[1][2].1);
        let m31 = (m_complex[2][0].0, m_complex[2][0].1);

        // Complex product M_12 * M_23 * M_31
        let p12_23_re = m12.0 * m23.0 - m12.1 * m23.1;
        let p12_23_im = m12.0 * m23.1 + m12.1 * m23.0;
        let _triple_re = p12_23_re * m31.0 - p12_23_im * m31.1;
        let triple_im = p12_23_re * m31.1 + p12_23_im * m31.0;

        println!("  Im(M_12 * M_23 * M_31) = {:.6e}", triple_im);
        println!("  This is proportional to the Jarlskog invariant.");

        // Extract a crude delta_CP from the phase of M_12
        let phase_12 = m_complex[0][1].1.atan2(m_complex[0][1].0);
        println!("  arg(M_12) = {:.2} deg", phase_12.to_degrees());
        println!(
            "  arg(M_23) = {:.2} deg",
            m_complex[1][2].1.atan2(m_complex[1][2].0).to_degrees()
        );
        println!(
            "  arg(M_31) = {:.2} deg",
            m_complex[2][0].1.atan2(m_complex[2][0].0).to_degrees()
        );
    } else {
        println!("  All imaginary parts are zero -- no CP violation in this basis.");
        println!("  This means the psi overlaps <v_i, psi(v_j)> = <v_i, psi^2(v_j)>");
        println!("  (psi is symmetric on the friction profiles).");
    }
}

/// CP violation from cross-sector psi asymmetry.
///
/// The psi-eigenspace decomposition gives Im = 0 within a single sector
/// because <v_i, psi(v_j)> = <v_i, psi^2(v_j)>. But the PMNS matrix is
/// U = U_ch^dagger * U_nu -- a PRODUCT of two diagonalizations. The CP phase
/// emerges from the RELATIVE orientation of the charged and neutrino friction
/// tensors under psi.
///
/// Key idea: build a 6x6 "cross-sector" complex Gram matrix
///   G_ij = sum_k omega^k * <ch_profile_i, psi^k(nu_profile_j)>
/// The off-diagonal elements of G mix charged and neutrino sectors, and
/// the psi asymmetry between sectors (different selectors) can produce
/// nonzero imaginary parts.
#[test]
fn test_cross_sector_cp_phase() {
    use crate::{bell_inequality::rotate_sparse, majorana_braiding::MajoranaMode};
    use cd_kernel::{gourlay_psi, gourlay_psi_n};

    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);

    let (o1, o2, o3, sign_table) = super::psi_setup();
    let subs = [&o1, &o2, &o3];

    let ch_a = MajoranaMode {
        gamma_index: ch_pair.0 - 1,
        cd_basis_index: ch_pair.0,
        cd_dim: 16,
    };
    let ch_b = MajoranaMode {
        gamma_index: ch_pair.1 - 1,
        cd_basis_index: ch_pair.1,
        cd_dim: 16,
    };
    let nu_a = MajoranaMode {
        gamma_index: nu_pair.0 - 1,
        cd_basis_index: nu_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: nu_pair.1 - 1,
        cd_basis_index: nu_pair.1,
        cd_dim: 16,
    };

    let build_profile =
        |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j {
                    continue;
                }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

    let ch_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&ch_a, &ch_b, s))
        .collect();
    let nu_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&nu_a, &nu_b, s))
        .collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    let omega_re = -0.5_f64;
    let omega_im = 3.0_f64.sqrt() / 2.0;

    println!("  === Cross-Sector CP Phase ===\n");

    // Cross-sector Gram matrix: G_ij = sum_k omega^k <ch_i, psi^k(nu_j)>
    let mut g_cross = [[(0.0_f64, 0.0_f64); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let c0 = dot16(&ch_profiles[i], &nu_profiles[j]);
            let psi1_j = gourlay_psi(&nu_profiles[j]);
            let c1 = dot16(&ch_profiles[i], &psi1_j);
            let psi2_j = gourlay_psi_n(&nu_profiles[j], 2);
            let c2 = dot16(&ch_profiles[i], &psi2_j);

            let re = c0 + c1 * omega_re + c2 * omega_re;
            let im = c1 * omega_im - c2 * omega_im;
            g_cross[i][j] = (re, im);
        }
    }

    println!("  Cross-sector Gram matrix G = <ch_i, psi^k(nu_j)>:");
    for i in 0..3 {
        let row: Vec<String> = (0..3)
            .map(|j| {
                let (re, im) = g_cross[i][j];
                if im.abs() < 1e-15 {
                    format!("{:>10.4}", re)
                } else {
                    format!("{:>7.4}{:+.4}i", re, im)
                }
            })
            .collect();
        println!("    [{}]", row.join("  "));
    }

    let mut max_im = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            max_im = max_im.max(g_cross[i][j].1.abs());
        }
    }
    println!("\n  Max |Im(G_ij)|: {:.6e}", max_im);

    if max_im > 1e-15 {
        println!("  NONZERO cross-sector imaginary part -- CP violation from selector asymmetry!");

        // The cross-sector phase is the source of delta_CP.
        // The dominant phase comes from the largest off-diagonal element.
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let (re, im) = g_cross[i][j];
                if (re * re + im * im).sqrt() > 1e-10 {
                    let phase = im.atan2(re).to_degrees();
                    println!(
                        "  arg(G[{},{}]) = {:.2} deg  (|G| = {:.4})",
                        i,
                        j,
                        phase,
                        (re * re + im * im).sqrt()
                    );
                }
            }
        }
    } else {
        println!("  Cross-sector Gram matrix is also real.");
        println!("  psi acts symmetrically on BOTH sectors for these selectors.");

        // Check: are the intra-sector psi overlaps symmetric too?
        // <ch_i, psi(ch_j)> vs <ch_i, psi^2(ch_j)>
        println!("\n  Checking psi symmetry breakdown:");
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let c1_ch = dot16(&ch_profiles[i], &gourlay_psi(&ch_profiles[j]));
                let c2_ch = dot16(&ch_profiles[i], &gourlay_psi_n(&ch_profiles[j], 2));
                let c1_nu = dot16(&nu_profiles[i], &gourlay_psi(&nu_profiles[j]));
                let c2_nu = dot16(&nu_profiles[i], &gourlay_psi_n(&nu_profiles[j], 2));
                let c1_cross = dot16(&ch_profiles[i], &gourlay_psi(&nu_profiles[j]));
                let c2_cross = dot16(&ch_profiles[i], &gourlay_psi_n(&nu_profiles[j], 2));

                println!(
                    "  ({},{}): ch: c1={:.6} c2={:.6} diff={:.2e}  nu: c1={:.6} c2={:.6} diff={:.2e}  cross: c1={:.6} c2={:.6} diff={:.2e}",
                    i,
                    j,
                    c1_ch,
                    c2_ch,
                    (c1_ch - c2_ch).abs(),
                    c1_nu,
                    c2_nu,
                    (c1_nu - c2_nu).abs(),
                    c1_cross,
                    c2_cross,
                    (c1_cross - c2_cross).abs()
                );
            }
        }
    }
}

/// Complex PMNS extension: J_k-based complexification for CP violation.
///
/// The real symmetric mass matrices force J_CP = 0. To get nonzero CP,
/// we use the Fano-derived complex structure J_k from the G2 stabilizer
/// to inject phases into the off-diagonal mass matrix entries.
///
/// For each embedding k=1..7, the 3 Fano pairs define 3 complex lines
/// on e_k^perp. The cross-generational psi-overlaps in this complex
/// basis carry natural phases that become the source of CP violation.
#[test]
fn test_complex_pmns_cp_violation() {
    use cd_kernel::gourlay_psi;
    use gororoba_algebra::lie::g2_stabilizer::complex_structure;

    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);
    let alpha_ch = 3.00;
    let alpha_nu = 1.35;

    // Get the current best real mass matrices
    let (m_ch_real, m_nu_real) =
        construct_pmns_matrices_two_param(ch_pair, nu_pair, alpha_ch, alpha_nu);

    // Apply V_6 correction at the optimal point
    let (v6_basis, _sv, _assessors) = extract_v6_basis();
    let lift = TensorElementLift;
    let eps = 0.05_f64;
    let n_basis = v6_basis.nrows().min(6);

    // Compute constrained directions
    let eig_ch_0 = m_ch_real.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu_0 = m_nu_real.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_raw_0 = eig_ch_0.U().transpose() * eig_nu_0.U();
    let (_, perm_u, perm_d) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw_0);

    let angles_at = |beta: &[f64; 6]| -> (f64, f64, f64) {
        let mut m_nu = m_nu_real.clone();
        apply_v6_perturbation(&mut m_nu, &v6_basis, beta, &lift);
        let m_nu_s = (&m_nu + m_nu.transpose()) * faer::Scale(0.5);
        let eig_nu = m_nu_s.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let u_raw = eig_ch_0.U().transpose() * eig_nu.U();
        let mut u_pmns = faer::Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                u_pmns[(i, j)] = u_raw[(perm_u[i], perm_d[j])];
            }
        }
        extract_pmns_angles(&u_pmns)
    };

    let mut g_12 = [0.0_f64; 6];
    let mut g_13 = [0.0_f64; 6];
    let mut g_23 = [0.0_f64; 6];
    for mu in 0..n_basis {
        let mut bp = [0.0_f64; 6];
        bp[mu] = eps;
        let mut bm = [0.0_f64; 6];
        bm[mu] = -eps;
        let (t12_p, t13_p, t23_p) = angles_at(&bp);
        let (t12_m, t13_m, t23_m) = angles_at(&bm);
        g_12[mu] = (t12_p - t12_m) / (2.0 * eps);
        g_13[mu] = (t13_p - t13_m) / (2.0 * eps);
        g_23[mu] = (t23_p - t23_m) / (2.0 * eps);
    }
    let u_solar = compute_constrained_solar_direction(&g_12, &g_13, &g_23);
    let u_atmo = compute_constrained_atmospheric_direction(&g_23, &g_13, &u_solar);

    // Apply optimal V_6 correction
    let inner_angles = |t1: f64, t2: f64| -> (f64, f64, f64) {
        let mut beta = [0.0_f64; 6];
        for k in 0..6 {
            beta[k] = t1 * u_solar[k] + t2 * u_atmo[k];
        }
        angles_at(&beta)
    };
    let (t_sol, t_atm, _, _) = gauss_newton_2d(
        &inner_angles,
        1.5,
        0.0,
        (33.41, 8.54, 49.0),
        (1.0, 2.24, 1.0),
        15,
    );

    let mut beta_opt = [0.0_f64; 6];
    for k in 0..6 {
        beta_opt[k] = t_sol * u_solar[k] + t_atm * u_atmo[k];
    }
    let mut m_nu_corrected = m_nu_real.clone();
    apply_v6_perturbation(&mut m_nu_corrected, &v6_basis, &beta_opt, &lift);
    let m_nu_corrected = (&m_nu_corrected + m_nu_corrected.transpose()) * faer::Scale(0.5);

    println!("--- COMPLEX PMNS: CP VIOLATION VIA J_k ---");

    // For each k=1..7, build the complex PMNS matrix
    for k in 1..=7 {
        let cs = complex_structure(k);

        // The 3 Fano pairs give 3 complex lines on e_k^perp.
        // Each pair (a_idx, b_idx, sign) defines:
        //   z_j = e_{perp[a_idx]} + i * sign * e_{perp[b_idx]}
        //
        // The cross-generational psi overlap in this complex basis
        // carries a phase. We compute this phase for each generation pair.

        // Build the 3 complex friction profiles
        // For each generation g and each Fano pair j, compute the
        // complex overlap: <profile_g, z_j> = <profile_g, u_j> + i*sign*<profile_g, J_k(u_j)>

        // The psi overlap between generation i and j in the complex basis gives
        // the off-diagonal phase. We use the existing nu_profiles.
        use crate::{bell_inequality::rotate_sparse, majorana_braiding::MajoranaMode};

        let (o1, o2, o3, sign_table) = super::psi_setup();
        let subs = [&o1, &o2, &o3];

        let nu_a = MajoranaMode {
            gamma_index: nu_pair.0 - 1,
            cd_basis_index: nu_pair.0,
            cd_dim: 16,
        };
        let nu_b = MajoranaMode {
            gamma_index: nu_pair.1 - 1,
            cd_basis_index: nu_pair.1,
            cd_dim: 16,
        };

        // Build 16D friction profiles per generation
        let build_profile =
            |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
                let i = mode_i.cd_basis_index;
                let j = mode_j.cd_basis_index;
                let a_sparse = vec![(i, 1.0)];
                let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
                let b_sparse = vec![(j, 1.0)];
                let mut profile = [0.0_f64; 16];
                for &kk in sub {
                    if kk == 0 || kk == i || kk == j {
                        continue;
                    }
                    let x_sparse = [(kk, 1.0)];
                    profile[kk] =
                        sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
                }
                profile
            };

        let nu_profiles: Vec<[f64; 16]> = subs
            .iter()
            .map(|s| build_profile(&nu_a, &nu_b, s))
            .collect();

        // For each generation pair (i,j), compute the psi-overlap PHASE
        // in the J_k complex basis.
        // psi maps nu_profiles[j] -> psi(nu_profiles[j]).
        // The overlap <nu_profiles[i], psi(nu_profiles[j])> is the real part.
        // In the complex basis, the J_k-rotated overlap gives the imaginary part:
        // <nu_profiles[i], J_k(psi(nu_profiles[j]))> is the imaginary component.
        //
        // J_k acts on the lower 8 components (octonion part) via the 6x6 matrix.

        let apply_jk = |v: &[f64; 16]| -> [f64; 16] {
            let mut result = [0.0_f64; 16];
            // J_k acts on perp indices only
            for r in 0..6 {
                for s in 0..6 {
                    result[cs.perp_indices[r]] += cs.matrix[r][s] * v[cs.perp_indices[s]];
                }
            }
            result
        };

        let dot16 = |a: &[f64; 16], b: &[f64; 16]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        };

        // Build complex neutrino mass matrix with CP amplitude scan
        // M_nu^complex[i][j] = M_nu_real[i][j] + i * alpha_CP * <profile_i, J_k(psi(profile_j))>
        // alpha_CP << alpha_nu to preserve the real angle fit

        // Pre-compute the imaginary overlap matrix
        let mut im_template = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    let psi_j = gourlay_psi(&nu_profiles[j]);
                    let jk_psi_j = apply_jk(&psi_j);
                    im_template[i][j] = dot16(&nu_profiles[i], &jk_psi_j);
                }
            }
        }

        // Scan alpha_CP to find the sweet spot: nonzero J_CP with minimal angle distortion
        let mut best_alpha_cp = 0.0_f64;
        let mut best_j_cp = 0.0_f64;
        let mut best_delta = 0.0_f64;
        let mut best_score = f64::MAX;
        let mut best_angles_cp = (0.0_f64, 0.0_f64, 0.0_f64);

        for step in 0..=100_i32 {
            let alpha_cp = step as f64 * 0.01;

            let mut m_nu_re = [[0.0_f64; 3]; 3];
            let mut m_nu_im = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    m_nu_re[i][j] = m_nu_corrected[(i, j)];
                    if i != j {
                        m_nu_im[i][j] = alpha_cp * im_template[i][j];
                    }
                }
            }

            // Hermitize
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let re_avg = (m_nu_re[i][j] + m_nu_re[j][i]) / 2.0;
                    let im_avg = (m_nu_im[i][j] - m_nu_im[j][i]) / 2.0;
                    m_nu_re[i][j] = re_avg;
                    m_nu_re[j][i] = re_avg;
                    m_nu_im[i][j] = im_avg;
                    m_nu_im[j][i] = -im_avg;
                }
                m_nu_im[i][i] = 0.0;
            }

            let mut m_nu_c = faer::Mat::<faer::c64>::zeros(3, 3);
            let mut m_ch_c = faer::Mat::<faer::c64>::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    m_nu_c[(i, j)] = faer::c64::new(m_nu_re[i][j], m_nu_im[i][j]);
                    m_ch_c[(i, j)] = faer::c64::new(m_ch_real[(i, j)], 0.0);
                }
            }

            let eig_ch_c = m_ch_c.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let eig_nu_c = m_nu_c.self_adjoint_eigen(faer::Side::Lower).unwrap();
            let u_pmns_c = eig_ch_c.U().adjoint() * eig_nu_c.U();

            let u_e3 = u_pmns_c[(0, 2)];
            let theta_13 = u_e3.norm().min(1.0).asin().to_degrees();
            let cos_13 = (theta_13.to_radians()).cos();
            let theta_12 = if cos_13 > 1e-15 {
                (u_pmns_c[(0, 1)].norm() / cos_13)
                    .min(1.0)
                    .asin()
                    .to_degrees()
            } else {
                0.0
            };
            let theta_23 = if cos_13 > 1e-15 {
                (u_pmns_c[(1, 2)].norm() / cos_13)
                    .min(1.0)
                    .asin()
                    .to_degrees()
            } else {
                0.0
            };

            let j_cp = (u_pmns_c[(0, 0)]
                * u_pmns_c[(1, 1)]
                * u_pmns_c[(0, 1)].conj()
                * u_pmns_c[(1, 0)].conj())
            .im;
            let delta_cp = (-u_e3).arg().to_degrees();

            // Score: angle preservation + nonzero J_CP
            let angle_cost = ((theta_12 - 33.41) / 33.41).powi(2)
                + ((theta_13 - 8.54) / 8.54).powi(2)
                + ((theta_23 - 49.0) / 49.0).powi(2);
            let score = angle_cost - 0.01 * j_cp.abs(); // reward nonzero J_CP

            if score < best_score && j_cp.abs() > 1e-6 {
                best_score = score;
                best_alpha_cp = alpha_cp;
                best_j_cp = j_cp;
                best_delta = delta_cp;
                best_angles_cp = (theta_12, theta_13, theta_23);
            }
        }

        println!(
            "  k={}: best alpha_CP={:.2}, theta_12={:.2}, theta_13={:.2}, theta_23={:.2}, J_CP={:.4e}, delta={:.1}",
            k,
            best_alpha_cp,
            best_angles_cp.0,
            best_angles_cp.1,
            best_angles_cp.2,
            best_j_cp,
            best_delta
        );
    }

    println!("\n  PDG targets: J_CP ~ 3.0e-2 (leptons), delta_CP ~ -140 to -180 deg (NO best-fit)");
}

/// Fine-grained alpha_CP scan for the best k-embedding.
///
/// At alpha_CP = 1.0 the complex coupling overwhelms the real mass matrix,
/// distorting mixing angles. Scan alpha_CP in [0.001, 0.5] to find the
/// sweet spot where J_CP is nonzero but angles remain close to PDG.
#[test]
fn test_complex_pmns_alpha_scan() {
    use crate::{bell_inequality::rotate_sparse, majorana_braiding::MajoranaMode};
    use cd_kernel::gourlay_psi;
    use gororoba_algebra::lie::g2_stabilizer::complex_structure;
    use nalgebra::SMatrix;
    use num_complex::Complex;

    type Mat3c = SMatrix<Complex<f64>, 3, 3>;

    let pdg = Pdg2024::default();
    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);

    // Build real mass matrices at the optimal point
    let (m_ch_real, m_nu_real) = construct_pmns_matrices_two_param(ch_pair, nu_pair, 3.75, 1.30);

    // Build friction profiles for the imaginary injection
    let (o1, o2, o3, sign_table) = super::psi_setup();
    let subs = [&o1, &o2, &o3];

    let nu_a = MajoranaMode {
        gamma_index: nu_pair.0 - 1,
        cd_basis_index: nu_pair.0,
        cd_dim: 16,
    };
    let nu_b = MajoranaMode {
        gamma_index: nu_pair.1 - 1,
        cd_basis_index: nu_pair.1,
        cd_dim: 16,
    };

    let build_profile =
        |mode_i: &MajoranaMode, mode_j: &MajoranaMode, sub: &[usize]| -> [f64; 16] {
            let i = mode_i.cd_basis_index;
            let j = mode_j.cd_basis_index;
            let a_sparse = vec![(i, 1.0)];
            let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
            let b_sparse = vec![(j, 1.0)];
            let mut profile = [0.0_f64; 16];
            for &k in sub {
                if k == 0 || k == i || k == j {
                    continue;
                }
                let x_sparse = [(k, 1.0)];
                profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
            }
            profile
        };

    let nu_profiles: Vec<[f64; 16]> = subs
        .iter()
        .map(|s| build_profile(&nu_a, &nu_b, s))
        .collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    // Use k=1 (strongest CP signal from earlier scan)
    let k_fixed = 1_usize;
    let _cs = complex_structure(k_fixed);

    // Build the J_k-rotated profiles: J_k(psi(profile_j))
    // J_k acts on the octonion part (indices 1-7) via left-multiplication by e_k
    let apply_jk = |profile: &[f64; 16]| -> [f64; 16] {
        use gororoba_algebra::construction::octonion::Octonion;
        let ek = Octonion::basis(k_fixed);
        let mut result = [0.0_f64; 16];
        // J_k acts on the lower octonion block (indices 0-7)
        let mut oct_part = [0.0_f64; 8];
        for idx in 0..8 {
            oct_part[idx] = profile[idx];
        }
        let oct = Octonion::new(oct_part);
        let jk_oct = ek.multiply(&oct);
        for idx in 0..8 {
            result[idx] = jk_oct.components[idx];
        }
        // Upper block: J_k acts similarly via e_k multiplication on the upper octonion
        let mut upper = [0.0_f64; 8];
        for idx in 0..8 {
            upper[idx] = profile[idx + 8];
        }
        let upper_oct = Octonion::new(upper);
        let jk_upper = ek.multiply(&upper_oct);
        for idx in 0..8 {
            result[idx + 8] = jk_upper.components[idx];
        }
        result
    };

    println!("  === Alpha_CP Fine Scan (k={}) ===\n", k_fixed);
    println!(
        "  {:>8} | {:>8} {:>8} {:>8} | {:>10} | {:>8} | {:>6}",
        "alpha_CP", "t12", "t13", "t23", "J_CP", "delta", "chi2"
    );
    println!(
        "  {:-<8}-+-{:-<8}-{:-<8}-{:-<8}-+-{:-<10}-+-{:-<8}-+-{:-<6}",
        "", "", "", "", "", "", ""
    );

    let mut best_chi2 = f64::MAX;
    let mut best_alpha = 0.0_f64;
    let mut best_result = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    for alpha_step in 0..=50 {
        let alpha_cp = alpha_step as f64 * 0.01; // 0.00 to 0.50

        // Build complex neutrino mass matrix:
        // M_nu_complex = M_nu_real + i * alpha_cp * Im_injection
        let mut m_nu_complex = Mat3c::zeros();
        for i in 0..3 {
            for j in 0..3 {
                m_nu_complex[(i, j)] = Complex::new(m_nu_real[(i, j)], 0.0);
            }
        }

        // Add imaginary off-diagonal: i * alpha_cp * <profile_i, J_k(psi(profile_j))>
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let psi_j = gourlay_psi(&nu_profiles[j]);
                let jk_psi_j = apply_jk(&psi_j);
                let coupling = dot16(&nu_profiles[i], &jk_psi_j);
                m_nu_complex[(i, j)] += Complex::new(0.0, alpha_cp * coupling);
            }
        }

        // Hermitianize: M = (M + M^dagger) / 2
        let m_herm = (m_nu_complex + m_nu_complex.adjoint()) * Complex::new(0.5, 0.0);

        // Eigendecompose
        let eigen = nalgebra::SymmetricEigen::new(m_herm);

        // Charged lepton eigenvectors (real, from faer)
        let eig_ch = m_ch_real.self_adjoint_eigen(faer::Side::Lower).unwrap();

        // Build PMNS: U = U_ch^T * U_nu (U_ch is real, U_nu is complex)
        let u_nu = &eigen.eigenvectors;
        let mut u_pmns = Mat3c::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex::new(0.0, 0.0);
                for m in 0..3 {
                    let u_ch_mi = eig_ch.U()[(m, i)];
                    sum += Complex::new(u_ch_mi, 0.0) * u_nu[(m, j)];
                }
                u_pmns[(i, j)] = sum;
            }
        }

        // Extract angles from |U_ij|
        let u_e3 = u_pmns[(0, 2)].norm();
        let theta_13 = u_e3.min(1.0).asin().to_degrees();
        let cos_13 = theta_13.to_radians().cos();
        let theta_12 = if cos_13 > 1e-15 {
            (u_pmns[(0, 1)].norm() / cos_13)
                .min(1.0)
                .asin()
                .to_degrees()
        } else {
            0.0
        };
        let theta_23 = if cos_13 > 1e-15 {
            (u_pmns[(1, 2)].norm() / cos_13)
                .min(1.0)
                .asin()
                .to_degrees()
        } else {
            0.0
        };

        // Jarlskog: J = Im(U_e2 * U_mu3 * conj(U_e3) * conj(U_mu2))
        let j_cp =
            (u_pmns[(0, 1)] * u_pmns[(1, 2)] * u_pmns[(0, 2)].conj() * u_pmns[(1, 1)].conj()).im;

        let delta = extract_cp_phase((theta_12, theta_13, theta_23), j_cp);

        // Chi^2 over 3 angles only
        let chi2_angles = super::pdg_score(theta_12, theta_13, theta_23, &pdg);

        if alpha_step % 5 == 0 || chi2_angles < best_chi2 {
            println!(
                "  {:>8.3} | {:>8.2} {:>8.2} {:>8.2} | {:>10.4e} | {:>8.1} | {:>6.1}",
                alpha_cp, theta_12, theta_13, theta_23, j_cp, delta, chi2_angles
            );
        }

        if chi2_angles < best_chi2 {
            best_chi2 = chi2_angles;
            best_alpha = alpha_cp;
            best_result = (theta_12, theta_13, theta_23, j_cp, delta);
        }
    }

    println!("\n  === BEST FIT ===");
    println!("  alpha_CP = {:.3}", best_alpha);
    println!(
        "  theta_12 = {:.2} deg (PDG: {:.2})",
        best_result.0, pdg.theta_12_deg
    );
    println!(
        "  theta_13 = {:.2} deg (PDG: {:.2})",
        best_result.1, pdg.theta_13_deg
    );
    println!(
        "  theta_23 = {:.2} deg (PDG: {:.2})",
        best_result.2, pdg.theta_23_deg
    );
    println!("  J_CP = {:.4e} (PDG: ~3e-2)", best_result.3);
    println!("  delta_CP = {:.1} deg (PDG: ~195)", best_result.4);
    println!("  chi2 = {:.2} (3 angles)", best_chi2);
}

/// CP violation via cross-sector rephasing of the EXISTING PMNS matrix.
///
/// Instead of making mass matrices complex (which changes eigenvalues and
/// breaks the permutation alignment), inject the CP phase as a diagonal
/// rephasing of the real PMNS matrix:
///
///   U_CP = diag(1, e^{i*phi_12}, e^{i*phi_13}) * U_real * diag(1, e^{i*psi_2}, e^{i*psi_3})
///
/// where the phases come from the cross-sector Gram matrix:
///   phi_ij = alpha_CP * arg(G_ij)
///   G_ij = sum_k omega^k * <ch_profile_i, psi^k(nu_profile_j)>
///
/// This preserves the mixing angles (|U_ij| unchanged) while introducing
/// a nonzero Jarlskog invariant.
#[test]
fn test_cp_rephasing_pipeline() {
    use crate::{bell_inequality::rotate_sparse, majorana_braiding::MajoranaMode};
    use cd_kernel::gourlay_psi;
    use num_complex::Complex;

    let ch_pair = (11_usize, 12);
    let nu_pair = (7_usize, 8);

    // Step 1: Get the real PMNS matrix from the existing pipeline
    let (m_ch, m_nu) = construct_pmns_matrices_two_param(ch_pair, nu_pair, 3.75, 1.30);
    let eig_ch = m_ch.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let eig_nu = m_nu.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let u_raw = eig_ch.U().transpose() * eig_nu.U();
    let (u_real, _, _) = crate::quark_sector::extract_ckm_permutation_aware(&u_raw);
    let (t12, t13, t23) = extract_pmns_angles(&u_real);

    println!("  === CP Rephasing Pipeline ===\n");
    println!(
        "  Real PMNS angles: t12={:.2}, t13={:.2}, t23={:.2}",
        t12, t13, t23
    );

    // Step 2: Build cross-sector Gram phases
    let (o1, o2, o3, sign_table) = super::psi_setup();
    let subs = [&o1, &o2, &o3];

    let build_profile = |sel: (usize, usize), sub: &[usize]| -> [f64; 16] {
        let mode_i = MajoranaMode {
            gamma_index: sel.0 - 1,
            cd_basis_index: sel.0,
            cd_dim: 16,
        };
        let mode_j = MajoranaMode {
            gamma_index: sel.1 - 1,
            cd_basis_index: sel.1,
            cd_dim: 16,
        };
        let i = mode_i.cd_basis_index;
        let j = mode_j.cd_basis_index;
        let a_sparse = vec![(i, 1.0)];
        let a_rotated = rotate_sparse(&a_sparse, i, j, std::f64::consts::FRAC_PI_4);
        let b_sparse = vec![(j, 1.0)];
        let mut profile = [0.0_f64; 16];
        for &k in sub {
            if k == 0 || k == i || k == j {
                continue;
            }
            let x_sparse = [(k, 1.0)];
            profile[k] = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
        }
        profile
    };

    let ch_profiles: Vec<[f64; 16]> = subs.iter().map(|s| build_profile(ch_pair, s)).collect();
    let nu_profiles: Vec<[f64; 16]> = subs.iter().map(|s| build_profile(nu_pair, s)).collect();

    let dot16 =
        |a: &[f64; 16], b: &[f64; 16]| -> f64 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };

    let omega_re = -0.5_f64;
    let omega_im = 3.0_f64.sqrt() / 2.0;

    // Cross-sector Gram phases
    let mut gram_phases = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let c0 = dot16(&ch_profiles[i], &nu_profiles[j]);
            let psi1_j = gourlay_psi(&nu_profiles[j]);
            let c1 = dot16(&ch_profiles[i], &psi1_j);
            let psi2_j = cd_kernel::gourlay_psi_n(&nu_profiles[j], 2);
            let c2 = dot16(&ch_profiles[i], &psi2_j);
            let re = c0 + c1 * omega_re + c2 * omega_re;
            let im = c1 * omega_im - c2 * omega_im;
            gram_phases[i][j] = im.atan2(re);
        }
    }

    println!("  Cross-sector Gram phases (radians):");
    for i in 0..3 {
        println!(
            "    [{:.4}, {:.4}, {:.4}]",
            gram_phases[i][0], gram_phases[i][1], gram_phases[i][2]
        );
    }

    // Step 3: Apply rephasing to the PMNS matrix.
    // The CP phase enters through the relative phases between rows and columns.
    // For the standard parameterization, delta_CP appears in U_e3:
    //   U_e3 = s13 * e^{-i*delta}
    // So we can inject delta directly into the (0,2) element.
    //
    // The algebraic prediction: delta_CP comes from the off-diagonal
    // Gram phase arg(G_12) = 45 deg = pi/4 (from earlier test).
    // But the physical delta_CP in the PMNS matrix is a combination of
    // all the Gram phases.
    //
    // Rephasing: U_CP[i][j] = U_real[i][j] * exp(i * alpha * gram_phases[i][j])
    println!("\n  Alpha_CP scan with rephasing:");
    println!(
        "  {:>8} | {:>8} {:>8} {:>8} | {:>10} | {:>8}",
        "alpha_CP", "t12", "t13", "t23", "J_CP", "delta"
    );

    for alpha_step in 0..=20 {
        let alpha_cp = alpha_step as f64 * 0.05;

        // Build complex PMNS via rephasing
        let mut u_cp = [[Complex::new(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let phase = alpha_cp * gram_phases[i][j];
                u_cp[i][j] = Complex::from_polar(u_real[(i, j)].abs(), phase);
                // Preserve the sign of the real element
                if u_real[(i, j)] < 0.0 {
                    u_cp[i][j] = -u_cp[i][j];
                }
            }
        }

        // Extract angles from |U_ij| -- these should be unchanged since we
        // only changed phases, not magnitudes
        let u_e3_abs = u_cp[0][2].norm();
        let theta_13_cp = u_e3_abs.min(1.0).asin().to_degrees();
        let cos_13 = theta_13_cp.to_radians().cos();
        let theta_12_cp = if cos_13 > 1e-15 {
            (u_cp[0][1].norm() / cos_13).min(1.0).asin().to_degrees()
        } else {
            0.0
        };
        let theta_23_cp = if cos_13 > 1e-15 {
            (u_cp[1][2].norm() / cos_13).min(1.0).asin().to_degrees()
        } else {
            0.0
        };

        // Jarlskog invariant
        let j_cp = (u_cp[0][1] * u_cp[1][2] * u_cp[0][2].conj() * u_cp[1][1].conj()).im;
        let delta = extract_cp_phase((theta_12_cp, theta_13_cp, theta_23_cp), j_cp);

        if alpha_step % 2 == 0 {
            println!(
                "  {:>8.2} | {:>8.2} {:>8.2} {:>8.2} | {:>10.4e} | {:>8.1}",
                alpha_cp, theta_12_cp, theta_13_cp, theta_23_cp, j_cp, delta
            );
        }
    }

    // The key prediction: at what alpha_CP does |J_CP| ~ 3e-2?
    // And what is the corresponding delta_CP?
    // Since |U_ij| are preserved, the angles stay at their PDG-matched values.
    // The only free parameter is alpha_CP, which sets the magnitude of J.
    let s12 = t12.to_radians().sin();
    let c12 = t12.to_radians().cos();
    let s13 = t13.to_radians().sin();
    let c13 = t13.to_radians().cos();
    let s23 = t23.to_radians().sin();
    let c23 = t23.to_radians().cos();
    let j_max = s12 * c12 * s23 * c23 * s13 * c13 * c13;

    println!("\n  J_max (from angles) = {:.6}", j_max);
    println!("  PDG J_CP ~ 3e-2 -> sin(delta) = {:.4}", 3e-2 / j_max);
    println!(
        "  -> delta_CP ~ {:.1} deg",
        (3e-2 / j_max).clamp(-1.0, 1.0).asin().to_degrees()
    );
    println!(
        "  Gram phase arg(G_12) = {:.1} deg = algebraic prediction",
        gram_phases[0][1].to_degrees()
    );
}
