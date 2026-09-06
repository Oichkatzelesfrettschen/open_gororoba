//! Open-x population transport with measured face inventories.
//!
//! The zero-gradient outflow extrapolates incoming populations from the
//! adjacent streamed cell. The numerical closure requires independent
//! reflection and backflow validation for a physical application.

use crate::{
    boundary::ZouHeBoundary,
    solver::{AOSOA_CHUNK, LbmSolver3D, aosoa_idx},
};

#[derive(Debug, thiserror::Error)]
#[error("open-x boundary: {0}")]
pub struct OpenBoundaryError(&'static str);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XOutflow {
    ZeroGradientPopulations,
}

/// Population mass crossing a face during one lattice timestep.
/// Values are inventories rather than rates; multiply by rho_ref*dx^3
/// for kilograms and divide by dt for a physical mass rate.
#[derive(Clone, Copy, Debug, Default)]
pub struct FaceMassFlux {
    pub min_x_outgoing: f64,
    pub max_x_outgoing: f64,
    pub min_x_incoming: f64,
    pub max_x_incoming: f64,
}
impl FaceMassFlux {
    pub fn net_incoming(self) -> f64 {
        self.min_x_incoming + self.max_x_incoming - self.min_x_outgoing - self.max_x_outgoing
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OpenXMassLedger {
    pub mass_before: f64,
    pub mass_after_streaming: f64,
    pub mass_after_boundary: f64,
    pub face: FaceMassFlux,
}
impl OpenXMassLedger {
    pub fn streaming_residual(self) -> f64 {
        self.mass_after_streaming - self.mass_before
            + self.face.min_x_outgoing
            + self.face.max_x_outgoing
    }
    pub fn boundary_residual(self) -> f64 {
        self.mass_after_boundary
            - self.mass_after_streaming
            - self.face.min_x_incoming
            - self.face.max_x_incoming
    }
    pub fn total_residual(self) -> f64 {
        self.mass_after_boundary - self.mass_before - self.face.net_incoming()
    }
}

pub struct OpenXBoundary {
    dimensions: [usize; 3],
    scratch: Vec<f64>,
    inlet: ZouHeBoundary,
}

impl OpenXBoundary {
    pub fn new(dimensions: [usize; 3]) -> Result<Self, OpenBoundaryError> {
        if dimensions[0] < 2
            || dimensions.contains(&0)
            || dimensions.iter().any(|&n| n > i32::MAX as usize)
        {
            return Err(OpenBoundaryError(
                "nx >= 2 and positive representable dimensions required",
            ));
        }
        let cells = dimensions
            .into_iter()
            .try_fold(1usize, |product, size| product.checked_mul(size))
            .ok_or(OpenBoundaryError("dimension overflow"))?;
        let length = cells
            .div_ceil(AOSOA_CHUNK)
            .checked_mul(AOSOA_CHUNK)
            .and_then(|n| n.checked_mul(19))
            .ok_or(OpenBoundaryError("population size overflow"))?;
        Ok(Self {
            dimensions,
            scratch: vec![0.0; length],
            inlet: ZouHeBoundary::new(),
        })
    }

    /// Transport populations and reconstruct both x faces atomically.
    /// Transverse axes wrap periodically. Outgoing x populations leave the
    /// domain; missing incoming slots are filled by the declared face rule.
    pub fn stream_and_reconstruct(
        &mut self,
        solver: &mut LbmSolver3D,
        velocity: [f64; 3],
        outflow: XOutflow,
    ) -> Result<OpenXMassLedger, OpenBoundaryError> {
        if [solver.nx, solver.ny, solver.nz] != self.dimensions
            || solver.f.len() != self.scratch.len()
        {
            return Err(OpenBoundaryError("solver dimensions or storage mismatch"));
        }
        if !velocity.into_iter().all(f64::is_finite) || velocity[0].abs() >= 1.0 {
            return Err(OpenBoundaryError(
                "finite inlet velocity with abs(ux) < 1 required",
            ));
        }
        let next_timestep = solver
            .timestep
            .checked_add(1)
            .ok_or(OpenBoundaryError("timestep overflow"))?;
        let mass_before = population_mass(solver)?;
        self.scratch.fill(0.0);
        let [nx, ny, nz] = self.dimensions;
        let mut face = FaceMassFlux::default();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let cell = z * nx * ny + y * nx + x;
                    for direction in 0..19 {
                        let population = solver.f[aosoa_idx(cell, direction)];
                        let [cx, cy, cz] = solver.collider.lattice.velocity(direction);
                        let destination_x = x as i64 + i64::from(cx);
                        if destination_x < 0 {
                            face.min_x_outgoing += population;
                        } else if destination_x >= nx as i64 {
                            face.max_x_outgoing += population;
                        } else {
                            let destination_y =
                                (y as i64 + i64::from(cy)).rem_euclid(ny as i64) as usize;
                            let destination_z =
                                (z as i64 + i64::from(cz)).rem_euclid(nz as i64) as usize;
                            let destination = destination_z * nx * ny
                                + destination_y * nx
                                + destination_x as usize;
                            self.scratch[aosoa_idx(destination, direction)] = population;
                        }
                    }
                }
            }
        }
        let mass_after_streaming = active_mass(&self.scratch, nx * ny * nz)?;
        self.inlet
            .apply_velocity_inlet_min_x_aosoa(&mut self.scratch, nx, ny, nz, velocity);
        for z in 0..nz {
            for y in 0..ny {
                let min_cell = z * nx * ny + y * nx;
                let max_cell = min_cell + nx - 1;
                for direction in 0..19 {
                    let cx = solver.collider.lattice.velocity(direction)[0];
                    if cx > 0 {
                        face.min_x_incoming += self.scratch[aosoa_idx(min_cell, direction)];
                    }
                    if cx < 0 {
                        let incoming = match outflow {
                            XOutflow::ZeroGradientPopulations => {
                                self.scratch[aosoa_idx(max_cell - 1, direction)]
                            }
                        };
                        self.scratch[aosoa_idx(max_cell, direction)] = incoming;
                        face.max_x_incoming += incoming;
                    }
                }
            }
        }
        let mass_after_boundary = active_mass(&self.scratch, nx * ny * nz)?;
        let ledger = OpenXMassLedger {
            mass_before,
            mass_after_streaming,
            mass_after_boundary,
            face,
        };
        if ![
            face.min_x_outgoing,
            face.max_x_outgoing,
            face.min_x_incoming,
            face.max_x_incoming,
            ledger.streaming_residual(),
            ledger.boundary_residual(),
            ledger.total_residual(),
        ]
        .into_iter()
        .all(f64::is_finite)
        {
            return Err(OpenBoundaryError("face flux or residual overflow"));
        }
        std::mem::swap(&mut solver.f, &mut self.scratch);
        solver.compute_macroscopic();
        solver.timestep = next_timestep;
        Ok(ledger)
    }
}

fn active_mass(populations: &[f64], cells: usize) -> Result<f64, OpenBoundaryError> {
    let length = cells
        .div_ceil(AOSOA_CHUNK)
        .checked_mul(AOSOA_CHUNK)
        .and_then(|count| count.checked_mul(19))
        .ok_or(OpenBoundaryError("population size overflow"))?;
    if populations.len() != length {
        return Err(OpenBoundaryError("population storage length mismatch"));
    }
    let mut mass = 0.0;
    for cell in 0..cells {
        for direction in 0..19 {
            let population = populations[aosoa_idx(cell, direction)];
            if !population.is_finite() {
                return Err(OpenBoundaryError("nonfinite population"));
            }
            mass += population;
        }
    }
    if !mass.is_finite() {
        return Err(OpenBoundaryError("mass overflow"));
    }
    Ok(mass)
}

/// Sum active populations independently of cached macroscopic density.
pub fn population_mass(solver: &LbmSolver3D) -> Result<f64, OpenBoundaryError> {
    let cells = [solver.nx, solver.ny, solver.nz]
        .into_iter()
        .try_fold(1usize, |product, size| product.checked_mul(size))
        .ok_or(OpenBoundaryError("dimension overflow"))?;
    active_mass(&solver.f, cells)
}
