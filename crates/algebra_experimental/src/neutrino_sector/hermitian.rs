use super::cp_scan::extract_delta_cp_invariant;

// ---------------------------------------------------------------------------
// Stack-allocated 3x3 complex Hermitian eigensolver (Cardano)
// ---------------------------------------------------------------------------

/// A 3x3 complex number: (re, im) pair.
pub type C2 = (f64, f64);

/// Multiply two complex numbers on the stack.
#[inline(always)]
pub fn cmul(a: C2, b: C2) -> C2 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Conjugate of a complex number.
#[inline(always)]
pub fn cconj(a: C2) -> C2 {
    (a.0, -a.1)
}

/// Eigenvalues + PMNS-relevant quantities for two 3x3 Hermitian matrices.
///
/// # Mathematical foundation
///
/// For a 3x3 Hermitian matrix H, the eigenvalues are roots of the
/// **real** characteristic polynomial:
///
/// ```text
/// lambda^3 - tr(H) * lambda^2 + s2(H) * lambda - det(H) = 0
/// ```
///
/// where `s2 = (tr^2 - tr(H^2))/2` is the second symmetric function.
/// This is solved analytically via the depressed cubic (Cardano/Vieta).
///
/// For the PMNS mixing angles, we need the unitary matrix `U` such that
/// `U^dag M_ch U_ch = diag` and `U^dag M_nu U_nu = diag`, then
/// `U_PMNS = U_ch^dag * U_nu`.  The Jarlskog invariant is:
///
/// ```text
/// J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
/// ```
///
/// # Why hand-rolled instead of faer
///
/// `faer::selfadjoint_eigendecomposition` allocates heap memory for the
/// working buffer.  In a tight scan loop (~10,000 calls), the allocation
/// overhead dominates.  This function uses only stack arrays and the
/// analytical Cardano formula, giving ~10x speedup for 3x3 matrices.
///
/// # Callers
///
/// - `test_cp_violation_joint_3d_scan`: inner scan loop
/// - Any future tight-loop PMNS computation
///
/// Returns `(eigenvalues_sorted, eigenvectors_as_columns)` where
/// eigenvalues are in ascending order.
#[allow(clippy::needless_range_loop)]
pub fn hermitian_3x3_eig(h: &[[C2; 3]; 3]) -> ([f64; 3], [[C2; 3]; 3]) {
    // Characteristic polynomial coefficients (all real for Hermitian H):
    // p = -tr(H), q = s2(H), r = -det(H)
    let tr_h = h[0][0].0 + h[1][1].0 + h[2][2].0;

    // tr(H^2) = sum_{i,j} |H[i][j]|^2
    let mut tr_h2 = 0.0_f64;
    for row in h {
        for &(re, im) in row {
            tr_h2 += re * re + im * im;
        }
    }
    let s2 = (tr_h * tr_h - tr_h2) * 0.5;

    // det(H) via Sarrus rule for 3x3 complex matrix (result is real)
    let det = {
        let a = cmul(cmul(h[0][0], h[1][1]), h[2][2]);
        let b = cmul(cmul(h[0][1], h[1][2]), h[2][0]);
        let c = cmul(cmul(h[0][2], h[1][0]), h[2][1]);
        let d = cmul(cmul(h[0][2], h[1][1]), h[2][0]);
        let e = cmul(cmul(h[0][1], h[1][0]), h[2][2]);
        let f = cmul(cmul(h[0][0], h[1][2]), h[2][1]);
        // det = a + b + c - d - e - f (real part only, imaginary cancels)
        a.0 + b.0 + c.0 - d.0 - e.0 - f.0
    };

    // Depressed cubic: t^3 + p*t + q = 0 where lambda = t + tr_h/3
    let shift = tr_h / 3.0;
    let p = s2 - tr_h * tr_h / 3.0;
    let q = tr_h * s2 / 3.0 - 2.0 * tr_h * tr_h * tr_h / 27.0 - det;

    // Vieta trigonometric solution (always 3 real roots for Hermitian)
    let disc = -(4.0 * p * p * p + 27.0 * q * q);
    let mut evals = [0.0_f64; 3];
    if disc.abs() < 1e-30 || p.abs() < 1e-30 {
        // Degenerate case: all eigenvalues equal or nearly so
        evals = [shift; 3];
        if p.abs() > 1e-30 {
            let r = ((-p / 3.0).max(0.0)).sqrt();
            let cos_theta = (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos() / 3.0;
            evals[0] = 2.0 * r * theta.cos() + shift;
            evals[1] = 2.0 * r * (theta - 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
            evals[2] = 2.0 * r * (theta + 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
        }
    } else {
        let r = ((-p / 3.0).max(0.0)).sqrt();
        let cos_theta = (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos() / 3.0;
        evals[0] = 2.0 * r * theta.cos() + shift;
        evals[1] = 2.0 * r * (theta - 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
        evals[2] = 2.0 * r * (theta + 2.0 * std::f64::consts::FRAC_PI_3).cos() + shift;
    }

    // Sort eigenvalues ascending
    if evals[0] > evals[1] {
        evals.swap(0, 1);
    }
    if evals[1] > evals[2] {
        evals.swap(1, 2);
    }
    if evals[0] > evals[1] {
        evals.swap(0, 1);
    }

    // Eigenvectors via (H - lambda*I) null space:
    // For each eigenvalue, find the eigenvector by cross product of two
    // rows of (H - lambda*I).  This is the standard adjugate method.
    let mut evecs = [[(0.0, 0.0); 3]; 3];
    for col in 0..3 {
        let lam = evals[col];
        let mut a = [[(0.0, 0.0); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = h[i][j];
            }
            a[i][i].0 -= lam;
        }
        // Cross product of rows 0 and 1: v = row0 x row1
        let v = [
            (
                cmul(a[0][1], a[1][2]).0 - cmul(a[0][2], a[1][1]).0,
                cmul(a[0][1], a[1][2]).1 - cmul(a[0][2], a[1][1]).1,
            ),
            (
                cmul(a[0][2], a[1][0]).0 - cmul(a[0][0], a[1][2]).0,
                cmul(a[0][2], a[1][0]).1 - cmul(a[0][0], a[1][2]).1,
            ),
            (
                cmul(a[0][0], a[1][1]).0 - cmul(a[0][1], a[1][0]).0,
                cmul(a[0][0], a[1][1]).1 - cmul(a[0][1], a[1][0]).1,
            ),
        ];
        let norm = (v[0].0 * v[0].0
            + v[0].1 * v[0].1
            + v[1].0 * v[1].0
            + v[1].1 * v[1].1
            + v[2].0 * v[2].0
            + v[2].1 * v[2].1)
            .sqrt();
        if norm > 1e-15 {
            for i in 0..3 {
                evecs[i][col] = (v[i].0 / norm, v[i].1 / norm);
            }
        } else {
            // Try rows 0 and 2
            let v2 = [
                (
                    cmul(a[0][1], a[2][2]).0 - cmul(a[0][2], a[2][1]).0,
                    cmul(a[0][1], a[2][2]).1 - cmul(a[0][2], a[2][1]).1,
                ),
                (
                    cmul(a[0][2], a[2][0]).0 - cmul(a[0][0], a[2][2]).0,
                    cmul(a[0][2], a[2][0]).1 - cmul(a[0][0], a[2][2]).1,
                ),
                (
                    cmul(a[0][0], a[2][1]).0 - cmul(a[0][1], a[2][0]).0,
                    cmul(a[0][0], a[2][1]).1 - cmul(a[0][1], a[2][0]).1,
                ),
            ];
            let norm2 = (v2[0].0 * v2[0].0
                + v2[0].1 * v2[0].1
                + v2[1].0 * v2[1].0
                + v2[1].1 * v2[1].1
                + v2[2].0 * v2[2].0
                + v2[2].1 * v2[2].1)
                .sqrt();
            if norm2 > 1e-15 {
                for i in 0..3 {
                    evecs[i][col] = (v2[i].0 / norm2, v2[i].1 / norm2);
                }
            } else {
                // Triple degeneracy -- use identity column
                evecs[col][col] = (1.0, 0.0);
            }
        }

        // U(1) phase canonicalization: make the largest-magnitude component
        // real and nonnegative.  This is the complex analogue of the LAPACK
        // convention for real eigenvectors (largest component positive).
        //
        // Without this, each eigenvector carries an arbitrary e^{i*theta}
        // phase.  Quantities like arg(-U_e3) for delta_CP depend on
        // individual matrix elements and are meaningless without a fixed
        // phase convention.
        let max_idx = {
            let mut best = 0;
            let mut best_mag_sq =
                evecs[0][col].0 * evecs[0][col].0 + evecs[0][col].1 * evecs[0][col].1;
            for idx in 1..3 {
                let mag_sq =
                    evecs[idx][col].0 * evecs[idx][col].0 + evecs[idx][col].1 * evecs[idx][col].1;
                if mag_sq > best_mag_sq {
                    best = idx;
                    best_mag_sq = mag_sq;
                }
            }
            best
        };
        let (re, im) = evecs[max_idx][col];
        let mag = (re * re + im * im).sqrt();
        if mag > 1e-15 {
            // Rotate entire vector by e^{-i*theta} where theta = arg(v_max)
            let cos_t = re / mag;
            let sin_t = im / mag;
            for i in 0..3 {
                let (r, m) = evecs[i][col];
                evecs[i][col] = (r * cos_t + m * sin_t, m * cos_t - r * sin_t);
            }
            // Ensure the reference component is strictly nonneg real
            if evecs[max_idx][col].0 < 0.0 {
                for i in 0..3 {
                    evecs[i][col].0 = -evecs[i][col].0;
                    evecs[i][col].1 = -evecs[i][col].1;
                }
            }
        }
    }

    (evals, evecs)
}

/// Minimum relative eigenvalue gap below which the Cardano cross-product
/// eigenvector method becomes numerically unstable.  When the gap falls
/// below this threshold times the Frobenius norm, we fall back to faer's
/// iterative QR which handles near-degeneracies gracefully.
const EIGGAP_THRESHOLD: f64 = 1e-10;

/// Hybrid 3x3 Hermitian eigensolver: Cardano if well-separated, faer if
/// degenerate.
///
/// Uses [`hermitian_3x3_eig`] (zero-alloc, O(1) Cardano) when eigenvalue
/// gaps are large relative to the matrix norm.  Falls back to faer's
/// `selfadjoint_eigendecomposition` near degeneracies where the cross-product
/// eigenvector method loses accuracy.
///
/// Returns `(eigenvalues_sorted, eigenvectors_as_columns)`.
#[allow(clippy::needless_range_loop)]
pub fn hermitian_3x3_eig_hybrid(h: &[[C2; 3]; 3]) -> ([f64; 3], [[C2; 3]; 3]) {
    let (evals, evecs) = hermitian_3x3_eig(h);

    // Check eigenvalue gap relative to matrix Frobenius norm
    let h_frob_sq: f64 = h
        .iter()
        .flat_map(|row| row.iter())
        .map(|&(r, m)| r * r + m * m)
        .sum();
    let h_norm = h_frob_sq.sqrt();

    let min_gap = (evals[1] - evals[0]).abs().min((evals[2] - evals[1]).abs());

    if min_gap > EIGGAP_THRESHOLD * h_norm {
        (evals, evecs)
    } else {
        // faer fallback for near-degenerate cases
        let mut h_faer = faer::Mat::<faer::c64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                h_faer[(i, j)] = faer::c64::new(h[i][j].0, h[i][j].1);
            }
        }
        let eig = h_faer.self_adjoint_eigen(faer::Side::Lower).unwrap();
        let mut fe = [0.0_f64; 3];
        for i in 0..3 {
            fe[i] = eig.S().column_vector()[i].re;
        }

        // Sort and build index map
        let mut idx = [0_usize, 1, 2];
        idx.sort_by(|&a, &b| fe[a].partial_cmp(&fe[b]).unwrap());
        let sorted_evals = [fe[idx[0]], fe[idx[1]], fe[idx[2]]];

        let mut sorted_evecs = [[(0.0, 0.0); 3]; 3];
        for col in 0..3 {
            let src = idx[col];
            for row in 0..3 {
                let c = eig.U()[(row, src)];
                sorted_evecs[row][col] = (c.re, c.im);
            }
            // Apply same phase canonicalization as Cardano path
            let max_idx = (0..3)
                .max_by(|&a, &b| {
                    let na = sorted_evecs[a][col].0 * sorted_evecs[a][col].0
                        + sorted_evecs[a][col].1 * sorted_evecs[a][col].1;
                    let nb = sorted_evecs[b][col].0 * sorted_evecs[b][col].0
                        + sorted_evecs[b][col].1 * sorted_evecs[b][col].1;
                    na.partial_cmp(&nb).unwrap()
                })
                .unwrap();
            let (re, im) = sorted_evecs[max_idx][col];
            let mag = (re * re + im * im).sqrt();
            if mag > 1e-15 {
                let cos_t = re / mag;
                let sin_t = im / mag;
                for i in 0..3 {
                    let (r, m) = sorted_evecs[i][col];
                    sorted_evecs[i][col] = (r * cos_t + m * sin_t, m * cos_t - r * sin_t);
                }
                if sorted_evecs[max_idx][col].0 < 0.0 {
                    for i in 0..3 {
                        sorted_evecs[i][col].0 = -sorted_evecs[i][col].0;
                        sorted_evecs[i][col].1 = -sorted_evecs[i][col].1;
                    }
                }
            }
        }
        (sorted_evals, sorted_evecs)
    }
}

/// Compute Jarlskog invariant and mixing angles directly from two 3x3
/// Hermitian mass matrices, entirely on the stack.
///
/// # Mathematical foundation
///
/// Given charged-lepton mass matrix `M_ch` and neutrino mass matrix
/// `M_nu`, diagonalise both via [`hermitian_3x3_eig`], form
/// `U_PMNS = U_ch^dag * U_nu`, apply the stored permutation, then
/// extract:
///
/// ```text
/// theta_13 = asin(|U_e3|)
/// theta_12 = asin(|U_e2| / cos(theta_13))
/// theta_23 = asin(|U_mu3| / cos(theta_13))
/// J_CP     = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
/// delta_CP = arg(-U_e3)
/// ```
///
/// # Why this exists
///
/// Eliminates faer heap allocation in tight scan loops.  The eigensolve
/// uses Cardano's formula (O(1) flops, zero allocation) instead of
/// iterative QR (O(n^3) with heap buffer).
///
/// # Returns
///
/// `(theta_12, theta_13, theta_23, j_cp, delta_cp, delta_cp_invariant)` in degrees.
pub fn pmns_from_hermitian_pair(
    m_ch: &[[C2; 3]; 3],
    m_nu: &[[C2; 3]; 3],
    perm_u: &[usize; 3],
    perm_d: &[usize; 3],
) -> (f64, f64, f64, f64, f64, f64) {
    let (_evals_ch, u_ch) = hermitian_3x3_eig(m_ch);
    let (_evals_nu, u_nu) = hermitian_3x3_eig(m_nu);

    // U_PMNS = U_ch^dag * U_nu  (3x3 complex multiply)
    let mut u_pmns = [[(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = (0.0_f64, 0.0_f64);
            for k in 0..3 {
                // U_ch^dag[i][k] = conj(U_ch[k][i])
                let a = cconj(u_ch[k][i]);
                let b = u_nu[k][j];
                s.0 += a.0 * b.0 - a.1 * b.1;
                s.1 += a.0 * b.1 + a.1 * b.0;
            }
            u_pmns[i][j] = s;
        }
    }

    // Apply permutation
    let mut u_perm = [[(0.0, 0.0); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            u_perm[i][j] = u_pmns[perm_u[i]][perm_d[j]];
        }
    }

    // Extract angles
    let u_e3_abs = (u_perm[0][2].0 * u_perm[0][2].0 + u_perm[0][2].1 * u_perm[0][2].1).sqrt();
    let theta_13 = u_e3_abs.min(1.0).asin().to_degrees();
    let cos_13 = theta_13.to_radians().cos();

    let theta_12 = if cos_13 > 1e-15 {
        let u_e2_abs = (u_perm[0][1].0 * u_perm[0][1].0 + u_perm[0][1].1 * u_perm[0][1].1).sqrt();
        (u_e2_abs / cos_13).min(1.0).asin().to_degrees()
    } else {
        0.0
    };

    let theta_23 = if cos_13 > 1e-15 {
        let u_mu3_abs = (u_perm[1][2].0 * u_perm[1][2].0 + u_perm[1][2].1 * u_perm[1][2].1).sqrt();
        (u_mu3_abs / cos_13).min(1.0).asin().to_degrees()
    } else {
        0.0
    };

    // Jarlskog: J = Im(U_e1 * U_mu2 * conj(U_e2) * conj(U_mu1))
    let prod = cmul(
        cmul(u_perm[0][0], u_perm[1][1]),
        cmul(cconj(u_perm[0][1]), cconj(u_perm[1][0])),
    );
    let j_cp = prod.1;

    // delta_CP = arg(-U_e3)
    let neg_ue3 = (-u_perm[0][2].0, -u_perm[0][2].1);
    let delta_cp = neg_ue3.1.atan2(neg_ue3.0).to_degrees();

    // Rephasing-invariant delta_CP via moduli + Jarlskog
    let u_moduli = {
        let mut m = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] =
                    (u_perm[i][j].0 * u_perm[i][j].0 + u_perm[i][j].1 * u_perm[i][j].1).sqrt();
            }
        }
        m
    };
    let delta_cp_invariant = extract_delta_cp_invariant(&u_moduli, j_cp);

    (
        theta_12,
        theta_13,
        theta_23,
        j_cp,
        delta_cp,
        delta_cp_invariant,
    )
}
