//! PDE surrogate models for wavelet compression experiments.
//!
//! Provides `ForcedDiffusion`: a 1D periodic forced-diffusion PDE
//!   du/dt = nu * d^2u/dx^2 + rho * sin(2*pi*k*x) * u^2
//! with periodic boundary conditions, integrated via RK4.

use std::f64::consts::PI;

/// 1D periodic forced-diffusion PDE surrogate.
///
/// Equation:  du/dt = nu * d^2u/dx^2 + rho * sin(2*pi*k*x) * u^2
///
/// Discretized on a uniform periodic grid of `n` points with `dx = 1.0/n`.
/// Second derivative uses centered differences with periodic wrapping.
pub struct ForcedDiffusion {
    /// Number of grid points.
    pub n: usize,
    /// Time step.
    pub dt: f64,
    /// Diffusion coefficient.
    pub nu: f64,
    /// Forcing amplitude.
    pub rho: f64,
    /// Forcing wavenumber.
    pub k: usize,
    dx: f64,
    /// Spatial forcing profile: `sin(2*pi*k*x_i)` precomputed.
    force_profile: Vec<f64>,
}

impl ForcedDiffusion {
    /// Construct a new `ForcedDiffusion` model.
    pub fn new(n: usize, dt: f64, nu: f64, rho: f64, k: usize) -> Self {
        assert!(n >= 2, "ForcedDiffusion: n must be >= 2, got {}", n);
        let dx = 1.0 / n as f64;
        let force_profile: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * k as f64 * i as f64 * dx).sin())
            .collect();
        Self {
            n,
            dt,
            nu,
            rho,
            k,
            dx,
            force_profile,
        }
    }

    /// Evaluate the PDE right-hand side at `u`.
    fn rhs(&self, u: &[f64]) -> Vec<f64> {
        let n = self.n;
        let dx2 = self.dx * self.dx;
        (0..n)
            .map(|i| {
                let left = u[(i + n - 1) % n];
                let right = u[(i + 1) % n];
                let diffusion = self.nu * (left - 2.0 * u[i] + right) / dx2;
                let forcing = self.rho * self.force_profile[i] * u[i] * u[i];
                diffusion + forcing
            })
            .collect()
    }

    /// Advance one RK4 step from `u`, returning the next state.
    pub fn step(&self, u: &[f64]) -> Vec<f64> {
        let dt = self.dt;
        let n = self.n;

        let k1 = self.rhs(u);
        let u2: Vec<f64> = (0..n).map(|i| u[i] + 0.5 * dt * k1[i]).collect();
        let k2 = self.rhs(&u2);
        let u3: Vec<f64> = (0..n).map(|i| u[i] + 0.5 * dt * k2[i]).collect();
        let k3 = self.rhs(&u3);
        let u4: Vec<f64> = (0..n).map(|i| u[i] + dt * k3[i]).collect();
        let k4 = self.rhs(&u4);

        (0..n)
            .map(|i| u[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .collect()
    }

    /// Run `steps` RK4 steps from `u0`, returning the terminal state.
    pub fn run(&self, u0: &[f64], steps: usize) -> Vec<f64> {
        assert_eq!(
            u0.len(),
            self.n,
            "ForcedDiffusion::run: u0 length {} != n {}",
            u0.len(),
            self.n
        );
        let mut u = u0.to_vec();
        for _ in 0..steps {
            u = self.step(&u);
        }
        u
    }

    /// Evaluate RHS with forcing conditionally disabled.
    /// When `active=false`, rho is treated as 0 (pure diffusion, no forcing).
    fn rhs_pulsed(&self, u: &[f64], active: bool) -> Vec<f64> {
        let n = self.n;
        let dx2 = self.dx * self.dx;
        let rho_eff = if active { self.rho } else { 0.0 };
        (0..n)
            .map(|i| {
                let left = u[(i + n - 1) % n];
                let right = u[(i + 1) % n];
                let diffusion = self.nu * (left - 2.0 * u[i] + right) / dx2;
                let forcing = rho_eff * self.force_profile[i] * u[i] * u[i];
                diffusion + forcing
            })
            .collect()
    }

    /// Advance one RK4 step with pulsed forcing.
    /// `active=true`: normal step with forcing.
    /// `active=false`: pure diffusion (rho=0), no energy injection.
    pub fn step_pulsed(&self, u: &[f64], active: bool) -> Vec<f64> {
        let dt = self.dt;
        let n = self.n;

        let k1 = self.rhs_pulsed(u, active);
        let u2: Vec<f64> = (0..n).map(|i| u[i] + 0.5 * dt * k1[i]).collect();
        let k2 = self.rhs_pulsed(&u2, active);
        let u3: Vec<f64> = (0..n).map(|i| u[i] + 0.5 * dt * k2[i]).collect();
        let k3 = self.rhs_pulsed(&u3, active);
        let u4: Vec<f64> = (0..n).map(|i| u[i] + dt * k3[i]).collect();
        let k4 = self.rhs_pulsed(&u4, active);

        (0..n)
            .map(|i| u[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .collect()
    }

    /// Generate a pulsed forcing schedule.
    /// Returns a `Vec<bool>` of length `total_steps` where `true` means forcing is active.
    /// `duty_cycle` in [0,1]: fraction of steps with active forcing.
    /// Pattern alternates in blocks: `on_block` active steps, then `off_block` quench steps.
    pub fn pulsed_schedule(total_steps: usize, duty_cycle: f64) -> Vec<bool> {
        assert!(
            (0.0..=1.0).contains(&duty_cycle),
            "duty_cycle must be in [0,1]"
        );
        if total_steps == 0 {
            return Vec::new();
        }
        // Block sizes: use a period of 20 steps for reasonable alternation
        let period = 20usize;
        let on_count = ((period as f64 * duty_cycle).round() as usize)
            .max(1)
            .min(period);
        let off_count = period.saturating_sub(on_count);
        let mut schedule = Vec::with_capacity(total_steps);
        let mut remaining = total_steps;
        while remaining > 0 {
            let on_steps = on_count.min(remaining);
            schedule.extend(std::iter::repeat_n(true, on_steps));
            remaining = remaining.saturating_sub(on_steps);
            let off_steps = off_count.min(remaining);
            schedule.extend(std::iter::repeat_n(false, off_steps));
            remaining = remaining.saturating_sub(off_steps);
        }
        schedule
    }

    /// Run pulsed simulation according to a schedule.
    /// Returns the terminal state.
    pub fn run_pulsed(&self, u0: &[f64], schedule: &[bool]) -> Vec<f64> {
        assert_eq!(
            u0.len(),
            self.n,
            "ForcedDiffusion::run_pulsed: u0 length mismatch"
        );
        let mut u = u0.to_vec();
        for &active in schedule {
            u = self.step_pulsed(&u, active);
        }
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pure_diffusion_l2_decay() {
        // With rho=0 (no forcing), L2 norm should decrease monotonically.
        let n = 64;
        let pde = ForcedDiffusion::new(n, 1e-4, 0.1, 0.0, 1);
        let u0: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin())
            .collect();

        let mut u = u0.clone();
        let mut prev_norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        for _ in 0..100 {
            u = pde.step(&u);
            let norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                norm <= prev_norm + 1e-10,
                "pure diffusion L2 norm increased: {} > {}",
                norm,
                prev_norm
            );
            prev_norm = norm;
        }
    }

    #[test]
    fn test_zero_forcing_no_blowup() {
        // With rho=0 and nu=0 (no diffusion, no forcing), L1 norm is constant.
        let n = 32;
        let pde = ForcedDiffusion::new(n, 1e-3, 0.0, 0.0, 1);
        let u0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let l1_init: f64 = u0.iter().map(|x| x.abs()).sum();

        let u_final = pde.run(&u0, 50);
        let l1_final: f64 = u_final.iter().map(|x| x.abs()).sum();

        assert_relative_eq!(l1_init, l1_final, epsilon = 1e-10);
    }

    #[test]
    fn test_pulsed_schedule_duty_cycle() {
        let sched = ForcedDiffusion::pulsed_schedule(100, 0.5);
        assert_eq!(sched.len(), 100);
        let on_count = sched.iter().filter(|&&x| x).count();
        // 50% duty cycle -> 50 on steps
        assert_eq!(on_count, 50);
    }

    #[test]
    fn test_pulsed_schedule_quarter_duty() {
        let sched = ForcedDiffusion::pulsed_schedule(100, 0.25);
        assert_eq!(sched.len(), 100);
        let on_count = sched.iter().filter(|&&x| x).count();
        assert_eq!(on_count, 25);
    }

    #[test]
    fn test_step_pulsed_inactive_equals_pure_diffusion() {
        // step_pulsed(false) should give same result as step with rho=0
        let n = 64;
        let pde_forced = ForcedDiffusion::new(n, 1e-4, 0.1, 1.0, 1);
        let pde_pure = ForcedDiffusion::new(n, 1e-4, 0.1, 0.0, 1);
        let u0: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
            .collect();

        let u_pulsed = pde_forced.step_pulsed(&u0, false);
        let u_pure = pde_pure.step(&u0);

        for (a, b) in u_pulsed.iter().zip(u_pure.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_step_pulsed_active_equals_normal() {
        // step_pulsed(true) should give same result as step()
        let n = 64;
        let pde = ForcedDiffusion::new(n, 1e-4, 0.1, 0.5, 1);
        let u0: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / n as f64).sin())
            .collect();

        let u_pulsed = pde.step_pulsed(&u0, true);
        let u_normal = pde.step(&u0);

        for (a, b) in u_pulsed.iter().zip(u_normal.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }
    }
}
