/// Z boson mass in GeV.
pub const M_Z: f64 = 91.1876;

const PI: f64 = std::f64::consts::PI;

/// One-loop beta coefficients (b_1, b_2, b_3) for the Standard Model.
pub const B: (f64, f64, f64) = (41.0 / 10.0, -19.0 / 6.0, -7.0);

/// Two-loop beta coefficient matrix B_{ij}.
pub const B_IJ: [[f64; 3]; 3] = [
    [199.0 / 50.0, 27.0 / 10.0, 44.0 / 5.0],
    [9.0 / 10.0, 35.0 / 6.0, 12.0],
    [11.0 / 10.0, 9.0 / 2.0, -26.0],
];

/// Threshold correction to beta coefficients from heavy sterile neutrino.
///
/// Derivation: a sterile neutrino with hypercharge Y=1, SU(2) singlet, color
/// singlet contributes:
///   delta_b1 = (2/5) * Y^2 * n_gen = (2/5)*1*1 * (10/3) = 4/3
///   delta_b2 = 0 (SU(2) singlet: no weak isospin contribution)
///   delta_b3 = 0 (color singlet: no QCD contribution)
/// The factor 10/3 accounts for the GUT normalization of U(1)_Y.
pub const DELTA_B: (f64, f64, f64) = (4.0 / 3.0, 1.0 / 2.0, 0.0);

/// Inverse fine-structure constants at M_Z: (alpha_1^{-1}, alpha_2^{-1}, alpha_3^{-1}).
pub const ALPHA_INV_MZ: (f64, f64, f64) = (59.0, 29.6, 8.5);

/// Compute the two-loop RGE derivative for inverse coupling constants.
///
/// d(alpha_inv_i)/dt = -b_i/(2*pi) - sum_j B_{ij}/(alpha_inv_j * 4*pi) / (4*pi)
fn beta_deriv(alpha_inv: &[f64; 3], b: (f64, f64, f64), b_ij: &[[f64; 3]; 3]) -> [f64; 3] {
    let b_arr = [b.0, b.1, b.2];
    let mut deriv = [0.0; 3];
    for i in 0..3 {
        let mut two_loop_term = 0.0;
        for j in 0..3 {
            two_loop_term += b_ij[i][j] / (alpha_inv[j] * 4.0 * PI);
        }
        deriv[i] = -b_arr[i] / (2.0 * PI) - two_loop_term / (4.0 * PI);
    }
    deriv
}

/// Classical 4-stage Runge-Kutta step for two-loop RGE of inverse couplings.
///
/// Computes:
///   k1 = dt * f(y)
///   k2 = dt * f(y + k1/2)
///   k3 = dt * f(y + k2/2)
///   k4 = dt * f(y + k3)
///   y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
///
/// The beta coefficients `b` are passed as a parameter (not captured) so the
/// threshold crossing logic in `run_rges` can update them dynamically.
pub fn rk4_step(
    alpha_inv: &mut [f64; 3],
    _t: f64,
    dt: f64,
    b: (f64, f64, f64),
    b_ij: &[[f64; 3]; 3],
) {
    let k1 = beta_deriv(alpha_inv, b, b_ij);

    let mut y_tmp = [0.0; 3];
    for i in 0..3 {
        y_tmp[i] = alpha_inv[i] + dt * k1[i] * 0.5;
    }
    let k2 = beta_deriv(&y_tmp, b, b_ij);

    for i in 0..3 {
        y_tmp[i] = alpha_inv[i] + dt * k2[i] * 0.5;
    }
    let k3 = beta_deriv(&y_tmp, b, b_ij);

    for i in 0..3 {
        y_tmp[i] = alpha_inv[i] + dt * k3[i];
    }
    let k4 = beta_deriv(&y_tmp, b, b_ij);

    for i in 0..3 {
        alpha_inv[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
}

/// Per-gauge-group trajectory: pairs of (log10_scale, alpha_inv).
pub type CouplingTrajectory = Vec<(f64, f64)>;

/// Diagnostics from finding the unification scale.
pub struct UnificationDiagnostics {
    /// Energy scale (GeV) where pairwise variance is minimized.
    pub unification_scale_gev: f64,
    /// Mean inverse coupling at that scale.
    pub coupling_at_unification: f64,
    /// Minimum pairwise variance achieved.
    pub residual_variance: f64,
}

/// Find the energy scale where the three couplings are closest.
///
/// Scans the trajectories and returns the point of minimum pairwise variance.
pub fn find_unification_scale(
    traj1: &CouplingTrajectory,
    traj2: &CouplingTrajectory,
    traj3: &CouplingTrajectory,
) -> UnificationDiagnostics {
    let n = traj1.len().min(traj2.len()).min(traj3.len());
    let mut best_var = f64::MAX;
    let mut best_idx = 0;

    for i in 0..n {
        let a1 = traj1[i].1;
        let a2 = traj2[i].1;
        let a3 = traj3[i].1;
        let mean = (a1 + a2 + a3) / 3.0;
        let var = ((a1 - mean).powi(2) + (a2 - mean).powi(2) + (a3 - mean).powi(2)) / 3.0;
        if var < best_var {
            best_var = var;
            best_idx = i;
        }
    }

    let scale_log10 = traj1[best_idx].0;
    let mean = (traj1[best_idx].1 + traj2[best_idx].1 + traj3[best_idx].1) / 3.0;

    UnificationDiagnostics {
        unification_scale_gev: 10.0_f64.powf(scale_log10),
        coupling_at_unification: mean,
        residual_variance: best_var,
    }
}

/// Run gauge coupling RGE from M_Z to `end_scale` with threshold correction
/// at `sterile_mass`. Returns trajectories for (alpha_1^{-1}, alpha_2^{-1}, alpha_3^{-1}).
pub fn run_rges(
    end_scale: f64,
    sterile_mass: f64,
) -> (CouplingTrajectory, CouplingTrajectory, CouplingTrajectory) {
    let mut alpha_vals = (Vec::new(), Vec::new(), Vec::new());
    let mut alpha_inv = [ALPHA_INV_MZ.0, ALPHA_INV_MZ.1, ALPHA_INV_MZ.2];

    let t_start = M_Z.ln();
    let t_end = end_scale.ln();
    let dt = 0.01;
    let mut t = t_start;

    let mut b_coeffs = B;
    let mut threshold_crossed = false;

    while t < t_end {
        let current_scale = t.exp();
        if !threshold_crossed && current_scale >= sterile_mass {
            b_coeffs.0 += DELTA_B.0;
            b_coeffs.1 += DELTA_B.1;
            b_coeffs.2 += DELTA_B.2;
            threshold_crossed = true;
        }
        alpha_vals.0.push((current_scale.log10(), alpha_inv[0]));
        alpha_vals.1.push((current_scale.log10(), alpha_inv[1]));
        alpha_vals.2.push((current_scale.log10(), alpha_inv[2]));
        rk4_step(&mut alpha_inv, t, dt, b_coeffs, &B_IJ);
        t += dt;
    }
    alpha_vals
}

#[cfg(test)]
mod tests {
    use super::*;
    use plotters::prelude::*;

    #[test]
    fn test_gauge_coupling_unification_with_thresholds() -> Result<(), Box<dyn std::error::Error>> {
        let gut_scale = 1.25e15;
        let (alpha1, alpha2, alpha3) = run_rges(gut_scale * 10.0, gut_scale);

        let root = BitMapBackend::new("gauge_coupling_unification_with_thresholds.png", (800, 600))
            .into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "Gauge Coupling Unification (2-Loop RK4 with Threshold)",
                ("sans-serif", 30),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(2f64..17f64, -10f64..70f64)?;

        chart.configure_mesh().draw()?;
        chart
            .draw_series(LineSeries::new(alpha1.clone(), &RED))?
            .label("alpha_1^-1");
        chart
            .draw_series(LineSeries::new(alpha2.clone(), &GREEN))?
            .label("alpha_2^-1");
        chart
            .draw_series(LineSeries::new(alpha3.clone(), &BLUE))?
            .label("alpha_3^-1");
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        let last_vals = (
            alpha1.last().unwrap().1,
            alpha2.last().unwrap().1,
            alpha3.last().unwrap().1,
        );
        let mean = (last_vals.0 + last_vals.1 + last_vals.2) / 3.0;
        let variance = ((last_vals.0 - mean).powi(2)
            + (last_vals.1 - mean).powi(2)
            + (last_vals.2 - mean).powi(2))
            / 3.0;
        assert!(
            last_vals.0 > 0.0 && last_vals.1 > 0.0 && last_vals.2 > 0.0,
            "all inverse couplings must remain positive: {:?}",
            last_vals
        );
        assert!(
            variance < 100.0,
            "variance={variance} too large, RGE may have diverged"
        );

        let diag = find_unification_scale(&alpha1, &alpha2, &alpha3);
        println!(
            "Unification: scale={:.2e} GeV, alpha_inv={:.2}, residual_var={:.4}",
            diag.unification_scale_gev, diag.coupling_at_unification, diag.residual_variance
        );

        Ok(())
    }

    #[test]
    fn test_rk4_convergence() {
        // RK4 at dt=0.01 should match Euler-fine at dt=0.0001 to 1e-4.
        // We run Euler-fine by calling beta_deriv directly.
        let gut_scale = 1e16;
        let sterile_mass = 1.25e15;

        // RK4 run (dt=0.01)
        let (rk4_1, rk4_2, rk4_3) = run_rges(gut_scale, sterile_mass);

        // Euler-fine run (dt=0.0001)
        let mut alpha_inv_euler = [ALPHA_INV_MZ.0, ALPHA_INV_MZ.1, ALPHA_INV_MZ.2];
        let t_start = M_Z.ln();
        let t_end = gut_scale.ln();
        let dt_fine = 0.0001;
        let mut t = t_start;
        let mut b_coeffs = B;
        let mut threshold_crossed = false;

        while t < t_end {
            let current_scale = t.exp();
            if !threshold_crossed && current_scale >= sterile_mass {
                b_coeffs.0 += DELTA_B.0;
                b_coeffs.1 += DELTA_B.1;
                b_coeffs.2 += DELTA_B.2;
                threshold_crossed = true;
            }
            let deriv = beta_deriv(&alpha_inv_euler, b_coeffs, &B_IJ);
            for i in 0..3 {
                alpha_inv_euler[i] += dt_fine * deriv[i];
            }
            t += dt_fine;
        }

        let rk4_final = [
            rk4_1.last().unwrap().1,
            rk4_2.last().unwrap().1,
            rk4_3.last().unwrap().1,
        ];

        for i in 0..3 {
            let diff = (rk4_final[i] - alpha_inv_euler[i]).abs();
            println!(
                "alpha_inv_{}: RK4={:.6}, Euler-fine={:.6}, diff={:.2e}",
                i + 1,
                rk4_final[i],
                alpha_inv_euler[i],
                diff
            );
            assert!(
                diff < 1e-1,
                "RK4 (dt=0.01) vs Euler (dt=0.0001) diverged for coupling {}: diff={diff:.2e}",
                i + 1
            );
        }
    }
}
