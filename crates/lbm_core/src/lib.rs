//! lbm_core: D2Q9 Lattice Boltzmann Method (BGK) solver.
//!
//! This crate provides:
//! - D2Q9 lattice with standard velocity ordering
//! - BGK collision operator
//! - Streaming with periodic boundaries
//! - Bounce-back for no-slip walls
//! - Guo forcing scheme for body forces
//! - Poiseuille flow simulation with analytical comparison
//!
//! # Literature
//! - Succi, "The Lattice Boltzmann Equation" (OUP, 2018), Ch. 10-12
//! - Guo, Zheng & Shi, PRE 65 (2002) 046308 (forcing scheme)

use ndarray::{Array2, Array3};
use std::f64::consts::PI;

pub mod turbulence;

/// D2Q9 lattice weights.
/// i:   0     1     2     3     4     5      6      7      8
/// w: 4/9   1/9   1/9   1/9   1/9   1/36   1/36   1/36   1/36
pub const W: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// D2Q9 velocity x-components.
/// i:   0   1   2  -1   0   1  -1  -1   1
pub const CX: [i32; 9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];

/// D2Q9 velocity y-components.
/// i:   0   0   1   0  -1   1   1  -1  -1
pub const CY: [i32; 9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];

/// Opposite direction indices for bounce-back.
pub const OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// D2Q9 Lattice Boltzmann simulation state.
#[derive(Clone)]
pub struct D2Q9 {
    /// Distribution functions, shape (9, nx, ny)
    pub f: Array3<f64>,
    /// Grid size in x
    pub nx: usize,
    /// Grid size in y
    pub ny: usize,
    /// BGK relaxation time
    pub tau: f64,
}

/// Results from Poiseuille flow simulation.
#[derive(Clone)]
pub struct PoiseuilleResult {
    /// y-coordinates
    pub y: Vec<f64>,
    /// Numerical velocity profile (x-component at mid slice)
    pub ux_numerical: Vec<f64>,
    /// Analytical velocity profile
    pub ux_analytical: Vec<f64>,
    /// Maximum relative error
    pub max_rel_error: f64,
    /// Mass history over time
    pub mass_history: Vec<f64>,
    /// Final density field
    pub rho_final: Array2<f64>,
}

impl D2Q9 {
    /// Create a new D2Q9 simulation with uniform initial density and zero velocity.
    pub fn new(nx: usize, ny: usize, tau: f64) -> Self {
        let rho = Array2::ones((nx, ny));
        let ux = Array2::zeros((nx, ny));
        let uy = Array2::zeros((nx, ny));
        let f = equilibrium(&rho, &ux, &uy);

        Self { f, nx, ny, tau }
    }

    /// Get the kinematic viscosity.
    pub fn viscosity(&self) -> f64 {
        (self.tau - 0.5) / 3.0
    }

    /// Compute macroscopic density and velocity fields.
    pub fn macroscopic(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        macroscopic(&self.f)
    }

    /// Perform one BGK collision step on interior nodes.
    pub fn collide(&mut self, y_start: usize, y_end: usize) {
        let omega = 1.0 / self.tau;
        let (rho, ux, uy) = self.macroscopic();
        let feq = equilibrium(&rho, &ux, &uy);

        for i in 0..9 {
            for x in 0..self.nx {
                for y in y_start..y_end {
                    self.f[[i, x, y]] += omega * (feq[[i, x, y]] - self.f[[i, x, y]]);
                }
            }
        }
    }

    /// Apply body force using Guo forcing scheme (first order).
    pub fn apply_force(&mut self, fx: f64, y_start: usize, y_end: usize) {
        let (rho, _, _) = self.macroscopic();

        for i in 0..9 {
            let cx = CX[i] as f64;
            for x in 0..self.nx {
                for y in y_start..y_end {
                    self.f[[i, x, y]] += 3.0 * W[i] * cx * fx * rho[[x, y]];
                }
            }
        }
    }

    /// Streaming step with periodic boundaries.
    pub fn stream(&mut self) {
        self.f = stream(&self.f);
    }

    /// Apply bounce-back at top and bottom walls.
    pub fn bounce_back_top_bottom(&mut self) {
        bounce_back_top_bottom(&mut self.f);
    }

    /// Get total mass in the system.
    pub fn total_mass(&self) -> f64 {
        self.f.sum()
    }
}

/// Compute D2Q9 equilibrium distributions.
///
/// f_eq_i = w_i * rho * (1 + 3*c_i.u + 9/2*(c_i.u)^2 - 3/2*u^2)
pub fn equilibrium(rho: &Array2<f64>, ux: &Array2<f64>, uy: &Array2<f64>) -> Array3<f64> {
    let (nx, ny) = rho.dim();
    let mut feq = Array3::zeros((9, nx, ny));

    for x in 0..nx {
        for y in 0..ny {
            let r = rho[[x, y]];
            let vx = ux[[x, y]];
            let vy = uy[[x, y]];
            let u_sq = vx * vx + vy * vy;

            for i in 0..9 {
                let cu = CX[i] as f64 * vx + CY[i] as f64 * vy;
                feq[[i, x, y]] = r * W[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
            }
        }
    }

    feq
}

/// Extract macroscopic variables from distribution functions.
pub fn macroscopic(f: &Array3<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let (_, nx, ny) = f.dim();
    let mut rho = Array2::zeros((nx, ny));
    let mut ux = Array2::zeros((nx, ny));
    let mut uy = Array2::zeros((nx, ny));

    for x in 0..nx {
        for y in 0..ny {
            let mut r = 0.0;
            let mut vx = 0.0;
            let mut vy = 0.0;

            for i in 0..9 {
                let fi = f[[i, x, y]];
                r += fi;
                vx += fi * CX[i] as f64;
                vy += fi * CY[i] as f64;
            }

            rho[[x, y]] = r;
            if r > 1e-30 {
                ux[[x, y]] = vx / r;
                uy[[x, y]] = vy / r;
            }
        }
    }

    (rho, ux, uy)
}

/// Streaming step: propagate each population along its lattice velocity.
pub fn stream(f: &Array3<f64>) -> Array3<f64> {
    let (_, nx, ny) = f.dim();
    let mut f_out = Array3::zeros((9, nx, ny));

    for i in 0..9 {
        let dx = CX[i];
        let dy = CY[i];

        for x in 0..nx {
            for y in 0..ny {
                // Source position (with periodic wrapping)
                let src_x = (x as i32 - dx).rem_euclid(nx as i32) as usize;
                let src_y = (y as i32 - dy).rem_euclid(ny as i32) as usize;
                f_out[[i, x, y]] = f[[i, src_x, src_y]];
            }
        }
    }

    f_out
}

/// Apply bounce-back at top and bottom walls (y=0 and y=ny-1).
pub fn bounce_back_top_bottom(f: &mut Array3<f64>) {
    let (_, nx, ny) = f.dim();

    for i in 0..9 {
        let opp = OPP[i];
        if CY[i] == 1 {
            // Population moved into top wall - reflect
            for x in 0..nx {
                f[[opp, x, ny - 1]] = f[[i, x, ny - 1]];
            }
        } else if CY[i] == -1 {
            // Population moved into bottom wall - reflect
            for x in 0..nx {
                f[[opp, x, 0]] = f[[i, x, 0]];
            }
        }
    }
}

/// Simulate 2D Poiseuille flow between parallel walls.
///
/// Walls at y=0 and y=ny-1 (bounce-back), periodic in x.
/// Body force fx drives flow in x-direction.
///
/// Analytical solution (with bounce-back half-way convention):
///   u_x(y) = fx / (2*nu) * (y - 0.5) * (ny - 1.5 - y)
pub fn simulate_poiseuille(
    nx: usize,
    ny: usize,
    tau: f64,
    fx: f64,
    n_steps: usize,
) -> PoiseuilleResult {
    let mut sim = D2Q9::new(nx, ny, tau);
    let nu = sim.viscosity();

    let mut mass_history = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        mass_history.push(sim.total_mass());

        // Collision on interior nodes only
        sim.collide(1, ny - 1);

        // Apply body force on interior nodes
        sim.apply_force(fx, 1, ny - 1);

        // Streaming
        sim.stream();

        // Bounce-back walls
        sim.bounce_back_top_bottom();
    }

    let (rho_final, ux, _uy) = sim.macroscopic();

    // Extract profile at middle x-slice
    let mid_x = nx / 2;
    let ux_numerical: Vec<f64> = (0..ny).map(|y| ux[[mid_x, y]]).collect();

    // Analytical Poiseuille profile with bounce-back half-way convention
    let ux_analytical: Vec<f64> = (0..ny)
        .map(|y| {
            let yf = y as f64;
            let val = fx / (2.0 * nu) * (yf - 0.5) * (ny as f64 - 1.5 - yf);
            val.max(0.0)
        })
        .collect();

    // Compute max relative error on interior points
    let max_ana: f64 = ux_analytical[1..ny - 1].iter().cloned().fold(0.0, f64::max);

    let max_rel_error = if max_ana > 1e-30 {
        ux_numerical[1..ny - 1]
            .iter()
            .zip(ux_analytical[1..ny - 1].iter())
            .map(|(num, ana)| (num - ana).abs())
            .fold(0.0, f64::max)
            / max_ana
    } else {
        0.0
    };

    let y: Vec<f64> = (0..ny).map(|y| y as f64).collect();

    PoiseuilleResult {
        y,
        ux_numerical,
        ux_analytical,
        max_rel_error,
        mass_history,
        rho_final,
    }
}

/// Results from Kolmogorov flow simulation (sinusoidal forcing, fully periodic).
#[derive(Clone)]
pub struct KolmogorovFlowResult {
    /// Final x-velocity field
    pub ux: Array2<f64>,
    /// Final y-velocity field
    pub uy: Array2<f64>,
    /// Final density field
    pub rho: Array2<f64>,
    /// Mass history over time
    pub mass_history: Vec<f64>,
    /// Mean enstrophy (mean square vorticity)
    pub enstrophy: f64,
    /// Kinematic viscosity
    pub viscosity: f64,
}

/// Compute mean enstrophy from velocity fields using finite differences.
///
/// Enstrophy = <|omega|^2> where omega = duy/dx - dux/dy (2D vorticity).
fn compute_enstrophy(ux: &Array2<f64>, uy: &Array2<f64>) -> f64 {
    let (nx, ny) = ux.dim();
    let mut total = 0.0;
    for x in 0..nx {
        for y in 0..ny {
            let xp = (x + 1) % nx;
            let yp = (y + 1) % ny;
            // Central differences with periodic wrapping
            let duy_dx = (uy[[xp, y]] - uy[[(x + nx - 1) % nx, y]]) / 2.0;
            let dux_dy = (ux[[x, yp]] - ux[[x, (y + ny - 1) % ny]]) / 2.0;
            let omega = duy_dx - dux_dy;
            total += omega * omega;
        }
    }
    total / (nx * ny) as f64
}

/// Simulate 2D Kolmogorov flow: sinusoidal forcing in a fully periodic domain.
///
/// The Kolmogorov flow applies a body force `fx(y) = A * sin(2*pi*n*y/ny)`
/// in the x-direction at wavenumber `force_mode`. This is the standard setup
/// for studying 2D turbulence transition:
///
/// - At low Re, the flow is a simple sinusoidal profile (laminar).
/// - At Re > ~40 (for force_mode=1), secondary instabilities appear.
/// - At higher Re, the flow becomes chaotic with broadband energy spectrum.
///
/// # Arguments
/// - `nx`, `ny`: Grid dimensions
/// - `tau`: BGK relaxation time (viscosity = (tau - 0.5) / 3)
/// - `force_amp`: Amplitude of the sinusoidal forcing
/// - `force_mode`: Wavenumber of the forcing (1 = fundamental, 2 = second harmonic)
/// - `n_steps`: Number of LBM timesteps
pub fn simulate_kolmogorov_flow(
    nx: usize,
    ny: usize,
    tau: f64,
    force_amp: f64,
    force_mode: usize,
    n_steps: usize,
) -> KolmogorovFlowResult {
    let mut sim = D2Q9::new(nx, ny, tau);
    let nu = sim.viscosity();
    let mut mass_history = Vec::with_capacity(n_steps / 10 + 1);

    for step in 0..n_steps {
        if step % 10 == 0 {
            mass_history.push(sim.total_mass());
        }

        // BGK collision on all nodes (fully periodic, no walls)
        sim.collide(0, ny);

        // Apply sinusoidal Kolmogorov forcing: fx(y) = A * sin(2*pi*n*y/ny)
        let (rho, _, _) = sim.macroscopic();
        for i in 0..9 {
            let cx = CX[i] as f64;
            for x in 0..nx {
                for y in 0..ny {
                    let fy_force = force_amp
                        * (2.0 * PI * force_mode as f64 * y as f64 / ny as f64).sin();
                    sim.f[[i, x, y]] += 3.0 * W[i] * cx * fy_force * rho[[x, y]];
                }
            }
        }

        // Streaming (periodic boundaries, no bounce-back)
        sim.stream();
    }

    let (rho, ux, uy) = sim.macroscopic();
    let enstrophy = compute_enstrophy(&ux, &uy);

    KolmogorovFlowResult {
        ux,
        uy,
        rho,
        mass_history,
        enstrophy,
        viscosity: nu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weights_sum_to_one() {
        let sum: f64 = W.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_equilibrium_conserves_density() {
        let nx = 10;
        let ny = 10;
        let rho = Array2::from_elem((nx, ny), 2.5);
        let ux = Array2::zeros((nx, ny));
        let uy = Array2::zeros((nx, ny));

        let feq = equilibrium(&rho, &ux, &uy);

        for x in 0..nx {
            for y in 0..ny {
                let rho_sum: f64 = (0..9).map(|i| feq[[i, x, y]]).sum();
                assert_relative_eq!(rho_sum, 2.5, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_equilibrium_conserves_momentum() {
        let nx = 5;
        let ny = 5;
        let rho = Array2::from_elem((nx, ny), 1.0);
        let ux = Array2::from_elem((nx, ny), 0.1);
        let uy = Array2::from_elem((nx, ny), 0.05);

        let feq = equilibrium(&rho, &ux, &uy);

        for x in 0..nx {
            for y in 0..ny {
                let mut px = 0.0;
                let mut py = 0.0;
                for i in 0..9 {
                    px += feq[[i, x, y]] * CX[i] as f64;
                    py += feq[[i, x, y]] * CY[i] as f64;
                }
                assert_relative_eq!(px, 0.1, epsilon = 1e-12);
                assert_relative_eq!(py, 0.05, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_macroscopic_roundtrip() {
        let rho = Array2::from_elem((8, 8), 1.5);
        let ux = Array2::from_elem((8, 8), 0.02);
        let uy = Array2::from_elem((8, 8), -0.01);

        let feq = equilibrium(&rho, &ux, &uy);
        let (rho2, ux2, uy2) = macroscopic(&feq);

        for x in 0..8 {
            for y in 0..8 {
                assert_relative_eq!(rho2[[x, y]], 1.5, epsilon = 1e-12);
                assert_relative_eq!(ux2[[x, y]], 0.02, epsilon = 1e-12);
                assert_relative_eq!(uy2[[x, y]], -0.01, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_streaming_periodic() {
        let mut f = Array3::zeros((9, 4, 4));
        // Put a marker at (1, 1) in direction 1 (cx=1, cy=0)
        f[[1, 1, 1]] = 1.0;

        let f_out = stream(&f);

        // Should have moved to (2, 1)
        assert_relative_eq!(f_out[[1, 2, 1]], 1.0, epsilon = 1e-14);
        assert_relative_eq!(f_out[[1, 1, 1]], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_streaming_wraps() {
        let mut f = Array3::zeros((9, 4, 4));
        // Put marker at edge in direction 1 (cx=1)
        f[[1, 3, 2]] = 1.0;

        let f_out = stream(&f);

        // Should wrap to x=0
        assert_relative_eq!(f_out[[1, 0, 2]], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_bounce_back() {
        let mut f = Array3::zeros((9, 4, 6));
        // Direction 2 moves in +y, put at y=ny-1
        f[[2, 2, 5]] = 1.0;

        bounce_back_top_bottom(&mut f);

        // Should reflect to opposite direction (4)
        assert_relative_eq!(f[[4, 2, 5]], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_mass_conservation() {
        let result = simulate_poiseuille(3, 21, 0.8, 1e-5, 100);

        let initial = result.mass_history[0];
        let final_mass = *result.mass_history.last().unwrap();

        // Mass should be conserved within numerical precision
        assert_relative_eq!(initial, final_mass, epsilon = 1e-10);
    }

    #[test]
    fn test_poiseuille_profile_parabolic() {
        // Run Poiseuille with enough steps to reach steady state
        let result = simulate_poiseuille(3, 41, 0.8, 1e-5, 5000);

        // Profile should be roughly parabolic (max at center)
        let ny = 41;
        let mid_y = ny / 2;

        // Interior velocity should be positive
        assert!(result.ux_numerical[mid_y] > 0.0);

        // Near-wall velocity should be smaller than center
        assert!(result.ux_numerical[2] < result.ux_numerical[mid_y]);
        assert!(result.ux_numerical[ny - 3] < result.ux_numerical[mid_y]);
    }

    #[test]
    fn test_poiseuille_accuracy() {
        // Test convergence to analytical solution
        let result = simulate_poiseuille(3, 41, 0.8, 1e-5, 10000);

        // Should achieve < 5% relative error after 10k steps
        assert!(
            result.max_rel_error < 0.05,
            "Relative error {} exceeds 5%",
            result.max_rel_error
        );
    }

    #[test]
    fn test_viscosity_formula() {
        let sim = D2Q9::new(4, 4, 1.0);
        assert_relative_eq!(sim.viscosity(), 0.5 / 3.0, epsilon = 1e-14);

        let sim2 = D2Q9::new(4, 4, 0.8);
        assert_relative_eq!(sim2.viscosity(), 0.3 / 3.0, epsilon = 1e-14);
    }

    // -- Kolmogorov flow tests -----------------------------------------------

    #[test]
    fn test_kolmogorov_mass_conservation() {
        let result = simulate_kolmogorov_flow(16, 16, 0.8, 1e-5, 1, 200);
        let initial = result.mass_history[0];
        let final_mass = *result.mass_history.last().unwrap();
        // Mass conservation: forcing adds momentum but not mass
        // (Guo forcing preserves mass to machine precision)
        let rel_drift = (final_mass - initial).abs() / initial;
        assert!(
            rel_drift < 1e-6,
            "Mass drift {:.2e} exceeds tolerance",
            rel_drift
        );
    }

    #[test]
    fn test_kolmogorov_develops_flow() {
        let result = simulate_kolmogorov_flow(32, 32, 0.8, 1e-4, 1, 500);
        // After 500 steps with forcing, velocity field should be nonzero
        let max_ux: f64 = result
            .ux
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_ux > 1e-8,
            "Flow should develop: max |ux| = {:.2e}",
            max_ux
        );
    }

    #[test]
    fn test_kolmogorov_sinusoidal_profile() {
        // With low Re forcing, steady state should be approximately sinusoidal
        let ny = 32;
        let result = simulate_kolmogorov_flow(8, ny, 0.9, 1e-5, 1, 2000);
        // At steady state, ux(y) ~ A * sin(2*pi*y/ny)
        // Check: mid-channel should have larger velocity than near boundaries
        let mid = ny / 2; // y = ny/4 is peak of sin
        let quarter = ny / 4;
        let ux_mid: f64 = (0..8).map(|x| result.ux[[x, mid]]).sum::<f64>() / 8.0;
        let ux_quarter: f64 = (0..8).map(|x| result.ux[[x, quarter]]).sum::<f64>() / 8.0;
        // For sin(2*pi*y/ny): peak at y=ny/4, zero crossing at y=ny/2
        // So ux_quarter should be larger in magnitude than ux_mid
        assert!(
            ux_quarter.abs() > ux_mid.abs(),
            "Sinusoidal profile: |ux(ny/4)|={:.6} should exceed |ux(ny/2)|={:.6}",
            ux_quarter.abs(),
            ux_mid.abs()
        );
    }

    #[test]
    fn test_kolmogorov_enstrophy_positive() {
        let result = simulate_kolmogorov_flow(16, 16, 0.8, 1e-4, 1, 300);
        assert!(
            result.enstrophy > 0.0,
            "Enstrophy should be positive after forcing"
        );
        assert!(result.enstrophy.is_finite());
    }

    #[test]
    fn test_kolmogorov_viscosity_stored() {
        let tau = 0.75;
        let result = simulate_kolmogorov_flow(8, 8, tau, 1e-5, 1, 10);
        let expected_nu = (tau - 0.5) / 3.0;
        assert_relative_eq!(result.viscosity, expected_nu, epsilon = 1e-14);
    }

    #[test]
    fn test_compute_enstrophy_zero_field() {
        let ux = Array2::zeros((8, 8));
        let uy = Array2::zeros((8, 8));
        assert_eq!(compute_enstrophy(&ux, &uy), 0.0);
    }

    #[test]
    fn test_compute_enstrophy_sinusoidal_flow() {
        // Periodic sinusoidal flow: ux = A*sin(2*pi*y/N), uy = 0
        // Vorticity = -dux/dy = -A*(2*pi/N)*cos(2*pi*y/N)
        // Enstrophy = A^2*(2*pi/N)^2 * <cos^2> = A^2*(2*pi/N)^2 * 0.5
        let n = 64;
        let amp = 0.1;
        let mut ux = Array2::zeros((n, n));
        let uy = Array2::zeros((n, n));
        for x in 0..n {
            for y in 0..n {
                ux[[x, y]] = amp * (2.0 * PI * y as f64 / n as f64).sin();
            }
        }
        let enst = compute_enstrophy(&ux, &uy);
        let k = 2.0 * PI / n as f64;
        let expected = amp * amp * k * k * 0.5;
        assert!(
            (enst - expected).abs() / expected < 0.01,
            "Enstrophy {:.6e} should be near {:.6e} (1% tolerance)",
            enst,
            expected
        );
    }
}
