//! Unified Simulation State.
//!
//! Holds the entire physical and algebraic state of the universe simulation:
//! 1. Fluid Dynamics (LBM D2Q9)
//! 2. Spacetime Metric (GR Kerr/Schwarzschild)
//! 3. Algebraic Field (Sedenion/Octonion)
//!
//! This struct is the primary data transfer object between the engine's
//! update pipeline and the analysis/visualization layers.

pub mod algebra_evolution;
pub mod frustration_e7;
pub mod state_3d;

pub use frustration_e7::E7SpectralFilter;
pub use state_3d::{
    AlgebraicField3D, FrustrationField3D, LbmBackend3D, SimulationConfig3D, SimulationState3D,
};

use algebra_core::physics::octonion_field::{FieldParams, Octonion, Pathion, Sedenion};

/// Pathionic heat sink for absorbing energy from truncated dimensions.
///
/// When the simulation uses Sedenions (dim=16), the full Cayley-Dickson
/// hierarchy continues to Pathions (dim=32). This sink models the energy
/// that would leak into dimensions 16..31 as a scalar damping term,
/// analogous to subgrid-scale dissipation in LES turbulence modeling.
#[derive(Debug, Clone)]
pub struct PathionSink {
    /// Whether the sink is active.
    pub enabled: bool,
    /// How strongly Sedenion associator overflow couples to the sink.
    pub coupling: f64,
    /// Dissipation rate in the virtual Pathion dimensions.
    pub damping: f64,
    /// Total energy absorbed by the sink (diagnostic).
    pub accumulated: f64,
}

impl PathionSink {
    /// Create a new Pathion sink with given coupling and damping parameters.
    pub fn new(coupling: f64, damping: f64) -> Self {
        Self {
            enabled: true,
            coupling,
            damping,
            accumulated: 0.0,
        }
    }

    /// Disabled (no-op) sink.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            coupling: 0.0,
            damping: 0.0,
            accumulated: 0.0,
        }
    }

    /// Absorb energy proportional to the associator norm squared.
    ///
    /// Returns the feedback term (negative = damping) to apply to the
    /// Sedenion field's viscosity modulation.
    pub fn absorb(&mut self, associator_norm: f64, dt: f64) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        // Energy that would go into dimensions 16..31
        let overflow = self.coupling * associator_norm.powi(2);
        self.accumulated += overflow * dt;
        // Return damped feedback to Sedenion space
        -self.damping * overflow * dt
    }

    /// Reset accumulated energy counter.
    pub fn reset(&mut self) {
        self.accumulated = 0.0;
    }
}

impl Default for PathionSink {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Chirality mode for testing the holographic chirality lock hypothesis.
///
/// The hypothesis: macroscopic vortex handedness is locked to the Sedenion
/// multiplication table orientation. Testing requires a 2x2 matrix of
/// (algebra orientation) x (initial vorticity) configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChiralityMode {
    /// Normal CD multiplication, normal vorticity (control baseline).
    Standard,
    /// Normal CD multiplication, INVERTED initial vorticity (u -> -u).
    /// Tests if vorticity can be flipped without changing algebra.
    Flipped,
    /// CONJUGATED CD table (all imaginary signs negated), normal vorticity.
    /// Tests if algebra alone determines handedness.
    Conjugate,
    /// Conjugated CD table + inverted vorticity.
    /// Double-flip: tests if two inversions cancel.
    DoubleFlip,
}
use gr_core::kerr::Kerr;
use gr_core::sedenion_geodesic::sedenion_homotopy_step;
use lbm_core::D2Q9;
use ndarray::Array2;
use stats_core::hypergraph::TriadHypergraph;

/// A test particle in the unified field.
#[derive(Clone, Debug, Default)]
pub struct Particle {
    pub r: f64,
    pub theta: f64,
    pub vr: f64,
    pub vtheta: f64,
}

/// Enum for runtime switching of algebraic species.
#[derive(Clone)]
pub enum AlgebraicField {
    Octonion(Array2<Octonion>),
    Sedenion(Array2<Sedenion>),
    Pathion(Array2<Pathion>),
}

impl AlgebraicField {
    pub fn dim(&self) -> (usize, usize) {
        match self {
            AlgebraicField::Octonion(a) => a.dim(),
            AlgebraicField::Sedenion(a) => a.dim(),
            AlgebraicField::Pathion(a) => a.dim(),
        }
    }
}

/// The Grand Unified Simulation State.
#[derive(Clone)]
pub struct SimulationState {
    /// Fluid dynamics layer (2D Lattice Boltzmann).
    pub fluid: D2Q9,

    /// General Relativity layer (Background Metric).
    /// Currently static, but could evolve (back-reaction).
    pub metric: Kerr,

    /// Algebraic Field layer (Sedenion/Octonion values on the grid).
    /// Maps 1:1 with the fluid grid.
    pub algebra: AlgebraicField,

    /// Test particles (geodesic tracers).
    pub particles: Vec<Particle>,

    /// Simulation time (dimensionless).
    pub time: f64,

    /// Topology history (Betti numbers).
    pub topology_history: Vec<(f64, usize, usize)>,

    /// Configuration parameters.
    pub config: SimulationConfig,
}

/// Configuration for the unified simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,

    /// LBM relaxation time.
    pub tau: f64,

    /// Algebraic field parameters.
    pub algebra_params: FieldParams,

    /// Coupling constant: Fluid -> Algebra (advection strength).
    pub coupling_fluid_algebra: f64,

    /// Coupling constant: Algebra -> Fluid (viscosity modulation).
    pub coupling_algebra_fluid: f64,

    /// Coupling constant: Metric -> Algebra (frustration source).
    pub coupling_metric_algebra: f64,
}

impl SimulationState {
    /// Create a new simulation state.
    pub fn new(config: SimulationConfig) -> Self {
        let fluid = D2Q9::new(config.nx, config.ny, config.tau);

        // Default to a rotating black hole background
        let metric = Kerr::new(1.0, 0.9);

        // Initialize algebraic field to zero based on requested species (defaulting to Octonion for now)
        // In a real config we would pass the species enum.
        let algebra = AlgebraicField::Octonion(Array2::from_elem((config.nx, config.ny), [0.0; 8]));

        Self {
            fluid,
            metric,
            algebra,
            particles: Vec::new(),
            time: 0.0,
            topology_history: Vec::new(),
            config,
        }
    }

    /// Advance the simulation by one time step.
    ///
    /// The update order is critical for causality:
    /// 1. Update Metric (optional back-reaction)
    /// 2. Update Algebra (advected by Fluid, frustrated by Metric)
    /// 3. Update Fluid (viscosity modulated by Algebra)
    pub fn step(&mut self) {
        // 1. Metric Update (Static for now)

        // 2. Algebra Update
        // Advect algebraic field using fluid velocity
        // Apply "Frustration Source" from metric curvature
        self.update_algebra();

        // 3. Fluid Update
        // Compute local viscosity from algebraic associator
        // Collide and Stream
        self.update_fluid();

        // 4. Particle Update (SHI Geodesics)
        self.update_particles();

        // 5. Topology Monitor
        if (self.time as usize).is_multiple_of(10) {
            self.update_topology();
        }

        self.time += 1.0;
    }

    fn update_topology(&mut self) {
        let mut hg = TriadHypergraph::new();
        // Use particle positions as vertices for topology tracking
        for (i, p) in self.particles.iter().enumerate().take(50) {
            // Map (r, theta) to a discrete ID for triad analysis
            let r_id = (p.r * 10.0) as usize;
            let th_id = (p.theta * 10.0) as usize;
            hg.add_triad(i, r_id + 1000, th_id + 2000);
        }
        self.topology_history
            .push((self.time, hg.betti_0(), hg.betti_1()));
    }

    fn update_particles(&mut self) {
        let h = 0.1; // Timestep for geodesics
        for p in &mut self.particles {
            let (r_next, theta_next, vr_next, vtheta_next) =
                sedenion_homotopy_step(&self.metric, p.r, p.theta, p.vr, p.vtheta, h);
            p.r = r_next;
            p.theta = theta_next;
            p.vr = vr_next;
            p.vtheta = vtheta_next;
        }
    }

    fn update_algebra(&mut self) {
        // Simple advection: phi(x) = phi(x - u*dt)
        // For a lattice, we can use a semi-Lagrangian scheme or simple upwinding.
        // For this prototype, we'll just apply a phase rotation based on local fluid vorticity
        // (simulating "stirring" of the internal symmetry space).

        let (_rho, ux, uy) = self.fluid.macroscopic();

        match &mut self.algebra {
            AlgebraicField::Octonion(field) => {
                let (nx, ny) = field.dim();
                let old_field = field.clone();

                for x in 0..nx {
                    for y in 0..ny {
                        let vorticity = (uy[[(x + 1) % nx, y]] - uy[[(x + nx - 1) % nx, y]])
                            - (ux[[x, (y + 1) % ny]] - ux[[x, (y + ny - 1) % ny]]);

                        let mut phi = old_field[[x, y]];
                        let angle = vorticity * self.config.coupling_fluid_algebra;
                        let c = angle.cos();
                        let s = angle.sin();

                        let e1 = phi[1];
                        let e2 = phi[2];
                        phi[1] = e1 * c - e2 * s;
                        phi[2] = e1 * s + e2 * c;

                        field[[x, y]] = phi;
                    }
                }
            }
            AlgebraicField::Sedenion(_field) => {
                // Sedenion update logic (similar but 16 components)
                // TODO: Implement full sedenion advection
            }
            AlgebraicField::Pathion(_field) => {
                // Pathion update logic
            }
        }
    }

    fn update_fluid(&mut self) {
        let (nx, ny) = (self.fluid.nx, self.fluid.ny);

        // Compute viscosity map from self.algebra
        let mut associator_norm = Array2::zeros((nx, ny));

        for x in 0..nx {
            for y in 0..ny {
                let norm_sq: f64 = match &self.algebra {
                    AlgebraicField::Octonion(f) => f[[x, y]].iter().map(|&v| v * v).sum(),
                    AlgebraicField::Sedenion(f) => f[[x, y]].iter().map(|&v| v * v).sum(),
                    AlgebraicField::Pathion(f) => f[[x, y]].iter().map(|&v| v * v).sum(),
                };
                associator_norm[[x, y]] = norm_sq.sqrt() * self.config.coupling_algebra_fluid;
            }
        }

        // Collide with spatially varying viscosity
        self.fluid.collide_with_associator(
            0,
            ny,
            self.config.coupling_algebra_fluid,
            &associator_norm,
        );

        self.fluid.stream();
        self.fluid.bounce_back_top_bottom();
    }
}
