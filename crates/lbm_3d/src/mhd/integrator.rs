//! Explicit integration of prescribed-velocity periodic magnetic transport.

use super::{MhdError, MhdField};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MhdIntegrator {
    LegacyEuler,
    SspRk3,
}

/// Computed binary64 |R(-d+ia)| squared for R(z)=1+z+z^2/2+z^3/6.
/// A value at most one admits the scalar mode, without a real-arithmetic certificate.
pub fn ssp_rk3_amplification_squared(d: f64, a: f64) -> Result<f64, MhdError> {
    if !d.is_finite() || d < 0.0 || !a.is_finite() {
        return Err(MhdError("invalid RK3 scalar transport numbers"));
    }
    let real = 1.0 - d + (d * d - a * a) / 2.0 - d * d * d / 6.0 + d * a * a / 2.0;
    let imaginary = a * (1.0 - d + d * d / 2.0 - a * a / 6.0);
    let squared = real * real + imaginary * imaginary;
    if !squared.is_finite() {
        return Err(MhdError("RK3 scalar amplification overflow"));
    }
    Ok(squared)
}

fn finite_vectors(values: &[[f64; 3]]) -> Result<(), MhdError> {
    if values.iter().flatten().any(|value| !value.is_finite()) {
        return Err(MhdError("RK3 stage contains nonfinite components"));
    }
    Ok(())
}

impl MhdField {
    fn validate_velocity(&self, velocity: &[[f64; 3]]) -> Result<(), MhdError> {
        self.validate_transport()?;
        if velocity.len() != self.bx.len() {
            return Err(MhdError("MHD velocity length mismatch"));
        }
        finite_vectors(velocity)
    }

    /// Check all finite-grid transverse modes for one spatially uniform velocity.
    /// The result excludes longitudinal coupling and variable-velocity stretching.
    /// Evolution does not invoke this optional diagnostic as a general stability gate.
    pub fn validate_uniform_transverse_ssp_rk3(
        &self,
        velocity: &[[f64; 3]],
    ) -> Result<f64, MhdError> {
        self.validate_velocity(velocity)?;
        let uniform = velocity[0];
        if velocity.iter().any(|value| *value != uniform) {
            return Err(MhdError(
                "transverse Fourier diagnostic requires uniform velocity",
            ));
        }
        let dimensions = [self.nx, self.ny, self.nz];
        let mut maximum = 0.0_f64;
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let mut advection = 0.0;
                    let mut laplacian = 0.0;
                    for axis in 0..3 {
                        let mode = [x, y, z][axis];
                        let angle = std::f64::consts::TAU * mode as f64 / dimensions[axis] as f64;
                        // Centered derivatives annihilate zero and even-grid Nyquist modes exactly.
                        let derivative = if mode == 0
                            || (dimensions[axis].is_multiple_of(2) && mode == dimensions[axis] / 2)
                        {
                            0.0
                        } else {
                            angle.sin()
                        };
                        advection += uniform[axis] * derivative;
                        laplacian += 4.0 * (angle / 2.0).sin().powi(2);
                    }
                    let gain = ssp_rk3_amplification_squared(
                        self.config.eta * laplacian * self.config.dt_mhd,
                        -advection * self.config.dt_mhd,
                    )?;
                    maximum = maximum.max(gain);
                    if gain > 1.0 {
                        return Err(MhdError("unstable uniform transverse RK3 Fourier mode"));
                    }
                }
            }
        }
        Ok(maximum)
    }

    /// Advance prescribed velocity frozen through one selected integration step.
    /// Finite stages establish representability, not variable-flow or coupled stability.
    pub fn try_evolve_b_field_with_integrator(
        &mut self,
        velocity: &[[f64; 3]],
        integrator: MhdIntegrator,
    ) -> Result<(), MhdError> {
        if integrator == MhdIntegrator::LegacyEuler {
            return self.try_evolve_b_field(velocity);
        }
        self.validate_velocity(velocity)?;
        let initial: Vec<[f64; 3]> = (0..self.bx.len())
            .map(|index| [self.bx[index], self.by[index], self.bz[index]])
            .collect();
        let mut stage = initial.clone();
        for (initial_weight, stage_weight) in [(0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0)] {
            let rhs = self.transport_rhs(&stage, velocity)?;
            for index in 0..stage.len() {
                for axis in 0..3 {
                    stage[index][axis] = initial_weight * initial[index][axis]
                        + stage_weight
                            * (stage[index][axis] + self.config.dt_mhd * rhs[index][axis]);
                }
            }
            finite_vectors(&stage)?;
        }
        let potential = self.algebraic_potential(&stage)?;
        for (index, value) in stage.into_iter().enumerate() {
            self.bx[index] = value[0];
            self.by[index] = value[1];
            self.bz[index] = value[2];
        }
        self.psi = potential;
        Ok(())
    }

    fn periodic_neighbors(&self, index: usize) -> [[usize; 2]; 3] {
        let x = index % self.nx;
        let y = (index / self.nx) % self.ny;
        let z = index / (self.nx * self.ny);
        [
            [
                index - x + (x + 1) % self.nx,
                index - x + (x + self.nx - 1) % self.nx,
            ],
            [
                index - y * self.nx + ((y + 1) % self.ny) * self.nx,
                index - y * self.nx + ((y + self.ny - 1) % self.ny) * self.nx,
            ],
            [
                index - z * self.nx * self.ny + ((z + 1) % self.nz) * self.nx * self.ny,
                index - z * self.nx * self.ny + ((z + self.nz - 1) % self.nz) * self.nx * self.ny,
            ],
        ]
    }

    fn algebraic_potential(&self, field: &[[f64; 3]]) -> Result<Vec<f64>, MhdError> {
        if self.config.cleaning_rate == 0.0 {
            return Ok(vec![0.0; field.len()]);
        }
        let mut potential = Vec::with_capacity(field.len());
        for index in 0..field.len() {
            let neighbors = self.periodic_neighbors(index);
            let divergence: f64 = (0..3)
                .map(|axis| {
                    0.5 * (field[neighbors[axis][0]][axis] - field[neighbors[axis][1]][axis])
                })
                .sum();
            let value = -self.config.cleaning_rate * self.config.cleaning_rate * divergence;
            if !value.is_finite() {
                return Err(MhdError("RK3 algebraic potential overflow"));
            }
            potential.push(value);
        }
        Ok(potential)
    }

    fn transport_rhs(
        &self,
        field: &[[f64; 3]],
        velocity: &[[f64; 3]],
    ) -> Result<Vec<[f64; 3]>, MhdError> {
        let cross: Vec<[f64; 3]> = field
            .iter()
            .zip(velocity)
            .map(|(magnetic, flow)| {
                [
                    flow[1] * magnetic[2] - flow[2] * magnetic[1],
                    flow[2] * magnetic[0] - flow[0] * magnetic[2],
                    flow[0] * magnetic[1] - flow[1] * magnetic[0],
                ]
            })
            .collect();
        finite_vectors(&cross)?;
        let potential = self.algebraic_potential(field)?;
        let mut rhs = vec![[0.0; 3]; field.len()];
        for index in 0..field.len() {
            let neighbors = self.periodic_neighbors(index);
            for axis in 0..3 {
                let second = (axis + 1) % 3;
                let third = (axis + 2) % 3;
                let curl = 0.5
                    * (cross[neighbors[second][0]][third] - cross[neighbors[second][1]][third])
                    - 0.5
                        * (cross[neighbors[third][0]][second] - cross[neighbors[third][1]][second]);
                rhs[index][axis] = curl;
                if self.config.eta > 0.0 {
                    let laplacian = field[neighbors[0][0]][axis]
                        + field[neighbors[0][1]][axis]
                        + field[neighbors[1][0]][axis]
                        + field[neighbors[1][1]][axis]
                        + field[neighbors[2][0]][axis]
                        + field[neighbors[2][1]][axis]
                        - 6.0 * field[index][axis];
                    rhs[index][axis] += self.config.eta * laplacian;
                }
                if self.config.cleaning_rate > 0.0 {
                    rhs[index][axis] -=
                        0.5 * (potential[neighbors[axis][0]] - potential[neighbors[axis][1]]);
                }
            }
        }
        finite_vectors(&rhs)?;
        Ok(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mhd::MhdConfig;

    #[test]
    fn mixed_stability_requires_combined_polynomial() {
        assert!(ssp_rk3_amplification_squared(0.0, 1.0).unwrap() < 1.0);
        assert!(ssp_rk3_amplification_squared(0.0, 2.0).unwrap() > 1.0);
        assert!(ssp_rk3_amplification_squared(2.0, 0.0).unwrap() < 1.0);
        assert!(ssp_rk3_amplification_squared(0.0, 1.5).unwrap() < 1.0);
        assert!(ssp_rk3_amplification_squared(2.0, 1.5).unwrap() > 1.0);
        assert!(ssp_rk3_amplification_squared(-1.0, 0.0).is_err());
    }

    #[test]
    fn uniform_diagnostic_rejects_variable_velocity() {
        let field = MhdField::new(4, 1, 1, MhdConfig::default());
        assert!(
            field
                .validate_uniform_transverse_ssp_rk3(&[[1.0, 0.0, 0.0]; 4])
                .is_ok()
        );
        assert!(
            field
                .validate_uniform_transverse_ssp_rk3(&[[2.0, 0.0, 0.0]; 4])
                .is_err()
        );
        let mut velocity = [[1.0, 0.0, 0.0]; 4];
        velocity[1][0] = 0.5;
        assert!(
            field
                .validate_uniform_transverse_ssp_rk3(&velocity)
                .is_err()
        );
    }

    #[test]
    fn stage_overflow_preserves_all_committed_bits() {
        let mut field = MhdField::new(4, 1, 1, MhdConfig::default());
        field.bx = vec![1.0, 0.0, -1.0, 0.0];
        field.psi.fill(-0.0);
        field.config.cleaning_rate = 1e200;
        let snapshot = |state: &MhdField| {
            state
                .bx
                .iter()
                .chain(&state.by)
                .chain(&state.bz)
                .chain(&state.psi)
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let before = snapshot(&field);
        assert!(
            field
                .try_evolve_b_field_with_integrator(&[[0.0; 3]; 4], MhdIntegrator::SspRk3)
                .is_err()
        );
        assert_eq!(before, snapshot(&field));
    }
}
