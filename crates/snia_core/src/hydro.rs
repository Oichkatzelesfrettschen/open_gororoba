use crate::error::SniaError;
use crate::types::HydroState1D;

#[derive(Debug, Clone, Copy)]
pub enum LimiterKind {
    Minmod,
    Superbee,
    MonotonizedCentral,
}

#[derive(Debug, Clone, Copy)]
pub enum BoundaryCondition {
    Periodic,
    Outflow,
}

/// Conservative variables for 1D Euler equations.
#[derive(Debug, Clone, Copy)]
pub struct ConservativeState {
    pub mass: f64,
    pub momentum: f64,
    pub energy: f64,
}

impl ConservativeState {
    #[must_use]
    pub fn flux(self, gamma: f64) -> Self {
        let inv_mass = 1.0 / self.mass.max(1.0e-30);
        let velocity = self.momentum * inv_mass;
        let pressure = (gamma - 1.0) * (self.energy - 0.5 * self.momentum * velocity);
        Self {
            mass: self.momentum,
            momentum: self.momentum * velocity + pressure,
            energy: velocity * (self.energy + pressure),
        }
    }

    #[must_use]
    pub fn add_state(self, rhs: Self) -> Self {
        Self {
            mass: self.mass + rhs.mass,
            momentum: self.momentum + rhs.momentum,
            energy: self.energy + rhs.energy,
        }
    }

    #[must_use]
    pub fn sub_state(self, rhs: Self) -> Self {
        Self {
            mass: self.mass - rhs.mass,
            momentum: self.momentum - rhs.momentum,
            energy: self.energy - rhs.energy,
        }
    }

    #[must_use]
    pub fn scale_state(self, scalar: f64) -> Self {
        Self {
            mass: self.mass * scalar,
            momentum: self.momentum * scalar,
            energy: self.energy * scalar,
        }
    }
}

/// HLLC + MUSCL finite-volume helper.
#[derive(Debug, Clone, Copy)]
pub struct HllcFlux1D {
    pub gamma: f64,
    pub density_floor: f64,
    pub pressure_floor: f64,
}

impl Default for HllcFlux1D {
    fn default() -> Self {
        Self {
            gamma: 5.0 / 3.0,
            density_floor: 1.0e-12,
            pressure_floor: 1.0e-12,
        }
    }
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else {
        a.signum() * a.abs().min(b.abs())
    }
}

fn superbee(a: f64, b: f64) -> f64 {
    let s = a.signum();
    if a * b <= 0.0 {
        return 0.0;
    }
    let a_abs = a.abs();
    let b_abs = b.abs();
    let c1 = (2.0 * a_abs).min(b_abs);
    let c2 = a_abs.min(2.0 * b_abs);
    s * c1.max(c2)
}

fn monotonized_central(a: f64, b: f64) -> f64 {
    minmod(0.5 * (a + b), minmod(2.0 * a, 2.0 * b))
}

fn limit_scalar(left_delta: f64, right_delta: f64, limiter: LimiterKind) -> f64 {
    match limiter {
        LimiterKind::Minmod => minmod(left_delta, right_delta),
        LimiterKind::Superbee => superbee(left_delta, right_delta),
        LimiterKind::MonotonizedCentral => monotonized_central(left_delta, right_delta),
    }
}

fn hydro_sub(a: HydroState1D, b: HydroState1D) -> HydroState1D {
    HydroState1D {
        density: a.density - b.density,
        velocity: a.velocity - b.velocity,
        pressure: a.pressure - b.pressure,
        specific_internal_energy: a.specific_internal_energy - b.specific_internal_energy,
    }
}

fn hydro_add(a: HydroState1D, b: HydroState1D) -> HydroState1D {
    HydroState1D {
        density: a.density + b.density,
        velocity: a.velocity + b.velocity,
        pressure: a.pressure + b.pressure,
        specific_internal_energy: a.specific_internal_energy + b.specific_internal_energy,
    }
}

fn hydro_scale(a: HydroState1D, s: f64) -> HydroState1D {
    HydroState1D {
        density: a.density * s,
        velocity: a.velocity * s,
        pressure: a.pressure * s,
        specific_internal_energy: a.specific_internal_energy * s,
    }
}

fn get_state(cells: &[HydroState1D], idx: isize, bc: BoundaryCondition) -> HydroState1D {
    let n = cells.len() as isize;
    match bc {
        BoundaryCondition::Periodic => {
            let wrapped = idx.rem_euclid(n) as usize;
            cells[wrapped]
        }
        BoundaryCondition::Outflow => {
            if idx < 0 {
                cells[0]
            } else if idx >= n {
                cells[(n - 1) as usize]
            } else {
                cells[idx as usize]
            }
        }
    }
}

impl HllcFlux1D {
    pub fn primitive_to_conservative(
        self,
        state: HydroState1D,
    ) -> Result<ConservativeState, SniaError> {
        if state.density <= 0.0 {
            return Err(SniaError::NonPositiveDensity(state.density));
        }
        if state.pressure < 0.0 {
            return Err(SniaError::NegativePressure(state.pressure));
        }
        let density = state.density.max(self.density_floor);
        let pressure = state.pressure.max(self.pressure_floor);
        let mass = density;
        let momentum = density * state.velocity;
        let specific_internal_energy = state
            .specific_internal_energy
            .max(pressure / ((self.gamma - 1.0) * density));
        let total_specific_energy =
            specific_internal_energy + 0.5 * state.velocity * state.velocity;
        let energy = density * total_specific_energy;
        Ok(ConservativeState {
            mass,
            momentum,
            energy,
        })
    }

    pub fn conservative_to_primitive(
        self,
        state: ConservativeState,
    ) -> Result<HydroState1D, SniaError> {
        if state.mass <= 0.0 {
            return Err(SniaError::NonPositiveDensity(state.mass));
        }
        let density = state.mass.max(self.density_floor);
        let velocity = state.momentum / density;
        let specific_total_energy = state.energy / density;
        let kinetic = 0.5 * velocity * velocity;
        let mut specific_internal_energy = specific_total_energy - kinetic;
        if specific_internal_energy <= 0.0 {
            specific_internal_energy = self.pressure_floor / ((self.gamma - 1.0) * density);
        }
        let pressure =
            ((self.gamma - 1.0) * density * specific_internal_energy).max(self.pressure_floor);
        Ok(HydroState1D {
            density,
            velocity,
            pressure,
            specific_internal_energy,
        })
    }

    #[must_use]
    pub fn sound_speed(self, state: HydroState1D) -> f64 {
        ((self.gamma * state.pressure.max(self.pressure_floor))
            / state.density.max(self.density_floor))
        .max(0.0)
        .sqrt()
    }

    #[must_use]
    pub fn estimate_signal_speeds(self, left: HydroState1D, right: HydroState1D) -> (f64, f64) {
        let c_l = self.sound_speed(left);
        let c_r = self.sound_speed(right);
        let s_l = (left.velocity - c_l).min(right.velocity - c_r);
        let s_r = (left.velocity + c_l).max(right.velocity + c_r);
        (s_l, s_r)
    }

    #[must_use]
    pub fn compute_cfl_timestep(self, cells: &[HydroState1D], dx: f64, cfl: f64) -> f64 {
        let mut max_signal = 0.0_f64;
        for cell in cells {
            max_signal = max_signal.max(cell.velocity.abs() + self.sound_speed(*cell));
        }
        cfl * dx / max_signal.max(1.0e-30)
    }

    fn hllc_star_state(
        self,
        state: HydroState1D,
        conservative: ConservativeState,
        s_k: f64,
        s_m: f64,
    ) -> ConservativeState {
        let safe = |x: f64| {
            if x.abs() < 1.0e-12 {
                if x < 0.0 {
                    -1.0e-12
                } else {
                    1.0e-12
                }
            } else {
                x
            }
        };
        let rho = state.density.max(self.density_floor);
        let u = state.velocity;
        let p = state.pressure.max(self.pressure_floor);
        let e = conservative.energy / conservative.mass.max(self.density_floor);

        let rho_star = rho * (s_k - u) / safe(s_k - s_m);
        let momentum_star = rho_star * s_m;
        let energy_star = rho_star * (e + (s_m - u) * (s_m + p / safe(rho * (s_k - u))));
        ConservativeState {
            mass: rho_star,
            momentum: momentum_star,
            energy: energy_star,
        }
    }

    #[must_use]
    fn hll_flux(
        self,
        u_l: ConservativeState,
        u_r: ConservativeState,
        f_l: ConservativeState,
        f_r: ConservativeState,
        s_l: f64,
        s_r: f64,
    ) -> ConservativeState {
        if s_l >= 0.0 {
            return f_l;
        }
        if s_r <= 0.0 {
            return f_r;
        }
        let inv = 1.0 / (s_r - s_l).max(1.0e-12);
        ConservativeState {
            mass: (s_r * f_l.mass - s_l * f_r.mass + s_l * s_r * (u_r.mass - u_l.mass)) * inv,
            momentum: (s_r * f_l.momentum - s_l * f_r.momentum
                + s_l * s_r * (u_r.momentum - u_l.momentum))
                * inv,
            energy: (s_r * f_l.energy - s_l * f_r.energy + s_l * s_r * (u_r.energy - u_l.energy))
                * inv,
        }
    }

    pub fn hllc_flux(
        self,
        left: HydroState1D,
        right: HydroState1D,
    ) -> Result<ConservativeState, SniaError> {
        let u_l = self.primitive_to_conservative(left)?;
        let u_r = self.primitive_to_conservative(right)?;
        let f_l = u_l.flux(self.gamma);
        let f_r = u_r.flux(self.gamma);
        let (s_l, s_r) = self.estimate_signal_speeds(left, right);

        if s_l >= 0.0 {
            return Ok(f_l);
        }
        if s_r <= 0.0 {
            return Ok(f_r);
        }

        let numerator = right.pressure - left.pressure + u_l.momentum * (s_l - left.velocity)
            - u_r.momentum * (s_r - right.velocity);
        let denominator = u_l.mass * (s_l - left.velocity) - u_r.mass * (s_r - right.velocity);
        let denom = if denominator.abs() < 1.0e-12 {
            if denominator < 0.0 {
                -1.0e-12
            } else {
                1.0e-12
            }
        } else {
            denominator
        };
        let mut s_m = numerator / denom;
        s_m = s_m.clamp(s_l.min(s_r), s_l.max(s_r));

        if s_m >= 0.0 {
            let u_star_l = self.hllc_star_state(left, u_l, s_l, s_m);
            if !u_star_l.mass.is_finite() || u_star_l.mass <= 0.0 {
                return Ok(self.hll_flux(u_l, u_r, f_l, f_r, s_l, s_r));
            }
            return Ok(f_l.add_state(u_star_l.sub_state(u_l).scale_state(s_l)));
        }
        let u_star_r = self.hllc_star_state(right, u_r, s_r, s_m);
        if !u_star_r.mass.is_finite() || u_star_r.mass <= 0.0 {
            return Ok(self.hll_flux(u_l, u_r, f_l, f_r, s_l, s_r));
        }
        Ok(f_r.add_state(u_star_r.sub_state(u_r).scale_state(s_r)))
    }

    #[must_use]
    pub fn reconstruct_muscl(
        self,
        cells: &[HydroState1D],
        limiter: LimiterKind,
        bc: BoundaryCondition,
    ) -> Vec<(HydroState1D, HydroState1D)> {
        let n = cells.len();
        if n == 0 {
            return Vec::new();
        }

        let mut slopes = vec![
            HydroState1D {
                density: 0.0,
                velocity: 0.0,
                pressure: 0.0,
                specific_internal_energy: 0.0,
            };
            n
        ];
        for (i, slope) in slopes.iter_mut().enumerate() {
            let c = cells[i];
            let l = get_state(cells, i as isize - 1, bc);
            let r = get_state(cells, i as isize + 1, bc);
            let dl = hydro_sub(c, l);
            let dr = hydro_sub(r, c);
            *slope = HydroState1D {
                density: limit_scalar(dl.density, dr.density, limiter),
                velocity: limit_scalar(dl.velocity, dr.velocity, limiter),
                pressure: limit_scalar(dl.pressure, dr.pressure, limiter),
                specific_internal_energy: limit_scalar(
                    dl.specific_internal_energy,
                    dr.specific_internal_energy,
                    limiter,
                ),
            };
        }

        let mut interfaces = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let left_idx = i as isize - 1;
            let right_idx = i as isize;
            let left_cell = get_state(cells, left_idx, bc);
            let right_cell = get_state(cells, right_idx, bc);
            let left_slope = if left_idx < 0 {
                slopes[0]
            } else if left_idx >= n as isize {
                slopes[n - 1]
            } else {
                slopes[left_idx as usize]
            };
            let right_slope = if right_idx < 0 {
                slopes[0]
            } else if right_idx >= n as isize {
                slopes[n - 1]
            } else {
                slopes[right_idx as usize]
            };
            let mut left_state = hydro_add(left_cell, hydro_scale(left_slope, 0.5));
            let mut right_state = hydro_sub(right_cell, hydro_scale(right_slope, 0.5));

            left_state.density = left_state.density.max(self.density_floor);
            right_state.density = right_state.density.max(self.density_floor);
            left_state.pressure = left_state.pressure.max(self.pressure_floor);
            right_state.pressure = right_state.pressure.max(self.pressure_floor);
            interfaces.push((left_state, right_state));
        }
        interfaces
    }

    pub fn compute_interface_fluxes(
        self,
        cells: &[HydroState1D],
        limiter: LimiterKind,
        bc: BoundaryCondition,
    ) -> Result<Vec<ConservativeState>, SniaError> {
        let reconstructed = self.reconstruct_muscl(cells, limiter, bc);
        let mut fluxes = Vec::with_capacity(reconstructed.len());
        for (left, right) in reconstructed {
            fluxes.push(self.hllc_flux(left, right)?);
        }
        Ok(fluxes)
    }

    fn advance_conservative(
        self,
        cells: &[HydroState1D],
        dx: f64,
        dt: f64,
        limiter: LimiterKind,
        bc: BoundaryCondition,
    ) -> Result<Vec<ConservativeState>, SniaError> {
        if cells.is_empty() {
            return Err(SniaError::EmptyGrid);
        }
        if dt <= 0.0 {
            return Err(SniaError::InvalidTimeStep(dt));
        }

        let mut u = Vec::with_capacity(cells.len());
        for cell in cells {
            u.push(self.primitive_to_conservative(*cell)?);
        }
        let fluxes = self.compute_interface_fluxes(cells, limiter, bc)?;
        let scale = dt / dx;
        let mut next = Vec::with_capacity(cells.len());
        for i in 0..cells.len() {
            let update = fluxes[i + 1].sub_state(fluxes[i]).scale_state(scale);
            let mut un = u[i].sub_state(update);
            un.mass = un.mass.max(self.density_floor);
            let kinetic = 0.5 * un.momentum * un.momentum / un.mass.max(self.density_floor);
            let min_energy = kinetic + self.pressure_floor / (self.gamma - 1.0);
            un.energy = un.energy.max(min_energy);
            next.push(un);
        }
        Ok(next)
    }

    pub fn euler_update(
        self,
        cells: &[HydroState1D],
        dx: f64,
        dt: f64,
        limiter: LimiterKind,
        bc: BoundaryCondition,
    ) -> Result<Vec<HydroState1D>, SniaError> {
        let next_cons = self.advance_conservative(cells, dx, dt, limiter, bc)?;
        let mut out = Vec::with_capacity(next_cons.len());
        for state in next_cons {
            out.push(self.conservative_to_primitive(state)?);
        }
        Ok(out)
    }

    pub fn rk2_update(
        self,
        cells: &[HydroState1D],
        dx: f64,
        dt: f64,
        limiter: LimiterKind,
        bc: BoundaryCondition,
    ) -> Result<Vec<HydroState1D>, SniaError> {
        let stage1_cons = self.advance_conservative(cells, dx, dt, limiter, bc)?;
        let mut stage1_prim = Vec::with_capacity(stage1_cons.len());
        for state in &stage1_cons {
            stage1_prim.push(self.conservative_to_primitive(*state)?);
        }

        let stage2_cons = self.advance_conservative(&stage1_prim, dx, dt, limiter, bc)?;
        let mut u0 = Vec::with_capacity(cells.len());
        for cell in cells {
            u0.push(self.primitive_to_conservative(*cell)?);
        }

        let mut out = Vec::with_capacity(cells.len());
        for i in 0..cells.len() {
            let combined = u0[i].add_state(stage2_cons[i]).scale_state(0.5);
            out.push(self.conservative_to_primitive(combined)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sod_initial_state(n: usize, gamma: f64) -> Vec<HydroState1D> {
        let mut cells = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f64 + 0.5) / (n as f64);
            if x < 0.5 {
                let rho = 1.0;
                let p = 1.0;
                let e = p / ((gamma - 1.0) * rho);
                cells.push(HydroState1D {
                    density: rho,
                    velocity: 0.0,
                    pressure: p,
                    specific_internal_energy: e,
                });
            } else {
                let rho = 0.125;
                let p = 0.1;
                let e = p / ((gamma - 1.0) * rho);
                cells.push(HydroState1D {
                    density: rho,
                    velocity: 0.0,
                    pressure: p,
                    specific_internal_energy: e,
                });
            }
        }
        cells
    }

    #[test]
    fn conservative_roundtrip_is_consistent() {
        let h = HllcFlux1D::default();
        let p = HydroState1D {
            density: 1.2,
            velocity: -0.8,
            pressure: 2.5,
            specific_internal_energy: 4.0,
        };
        let u = h.primitive_to_conservative(p).expect("cons");
        let p2 = h.conservative_to_primitive(u).expect("prim");
        assert!((p2.density - p.density).abs() < 1.0e-12);
        assert!((p2.velocity - p.velocity).abs() < 1.0e-12);
    }

    #[test]
    fn flux_is_finite_for_reasonable_states() {
        let h = HllcFlux1D::default();
        let left = HydroState1D {
            density: 1.0,
            velocity: 0.1,
            pressure: 1.0,
            specific_internal_energy: 2.0,
        };
        let right = HydroState1D {
            density: 0.9,
            velocity: -0.1,
            pressure: 0.9,
            specific_internal_energy: 1.8,
        };
        let f = h.hllc_flux(left, right).expect("flux");
        assert!(f.mass.is_finite());
        assert!(f.momentum.is_finite());
        assert!(f.energy.is_finite());
    }

    #[test]
    fn limiter_suite_returns_finite_slopes() {
        let cells = vec![
            HydroState1D {
                density: 1.0,
                velocity: 0.0,
                pressure: 1.0,
                specific_internal_energy: 1.5,
            },
            HydroState1D {
                density: 0.8,
                velocity: 0.2,
                pressure: 0.9,
                specific_internal_energy: 1.3,
            },
            HydroState1D {
                density: 0.6,
                velocity: 0.3,
                pressure: 0.8,
                specific_internal_energy: 1.1,
            },
        ];
        let h = HllcFlux1D::default();
        for limiter in [
            LimiterKind::Minmod,
            LimiterKind::Superbee,
            LimiterKind::MonotonizedCentral,
        ] {
            let rec = h.reconstruct_muscl(&cells, limiter, BoundaryCondition::Outflow);
            assert_eq!(rec.len(), cells.len() + 1);
            assert!(rec
                .iter()
                .all(|(l, r)| l.density.is_finite() && r.density.is_finite()));
        }
    }

    #[test]
    fn sod_regression_matches_reference_window() {
        let h = HllcFlux1D {
            gamma: 1.4,
            ..HllcFlux1D::default()
        };
        let n = 256;
        let dx = 1.0 / (n as f64);
        let mut cells = sod_initial_state(n, h.gamma);
        let t_end = 0.15;
        let cfl = 0.20;
        let mut t = 0.0;

        while t < t_end {
            let dt = h
                .compute_cfl_timestep(&cells, dx, cfl)
                .min(t_end - t)
                .max(1.0e-8);
            cells = h
                .rk2_update(
                    &cells,
                    dx,
                    dt,
                    LimiterKind::Minmod,
                    BoundaryCondition::Outflow,
                )
                .expect("rk2");
            t += dt;
        }

        let mid = cells[n / 2];
        // Reference envelope: mid-state must remain in physically valid Sod-like ranges.
        assert!(mid.density > 0.10 && mid.density < 1.10);
        assert!(mid.velocity > -0.05 && mid.velocity < 1.50);
        assert!(mid.pressure > 0.05 && mid.pressure < 1.10);
    }

    #[test]
    fn near_vacuum_stays_finite() {
        let h = HllcFlux1D::default();
        let mut cells = vec![
            HydroState1D {
                density: 1.0e-8,
                velocity: 0.0,
                pressure: 1.0e-10,
                specific_internal_energy: 1.0e-2,
            },
            HydroState1D {
                density: 1.0e-8,
                velocity: 1.0e-2,
                pressure: 1.0e-10,
                specific_internal_energy: 1.0e-2,
            },
            HydroState1D {
                density: 2.0e-8,
                velocity: -1.0e-2,
                pressure: 2.0e-10,
                specific_internal_energy: 1.0e-2,
            },
        ];
        let dx = 1.0;
        for _ in 0..20 {
            let dt = h.compute_cfl_timestep(&cells, dx, 0.05).max(1.0e-8);
            cells = h
                .rk2_update(
                    &cells,
                    dx,
                    dt,
                    LimiterKind::Minmod,
                    BoundaryCondition::Outflow,
                )
                .expect("update");
        }
        for c in cells {
            assert!(c.density.is_finite());
            assert!(c.pressure.is_finite());
            assert!(c.specific_internal_energy.is_finite());
            assert!(c.density > 0.0);
            assert!(c.pressure > 0.0);
        }
    }
}
