//! Worldline Monte Carlo Casimir energy computation.
//!
//! Implements the v-loop algorithm from White et al. (2021) for computing
//! Casimir vacuum energy densities in custom geometries via the worldline
//! path integral. The method generates closed random loops, scales them
//! to different proper times, and checks boundary intersections to estimate
//! the Casimir energy density at each spatial point.
//!
//! # Modules
//!
//! - `vloop` -- Closed worldline loop generation (v-loop algorithm)
//! - `geometry` -- Casimir boundary definitions (plates, spheres, cylinders)
//! - `energy` -- Monte Carlo integration of the worldline path integral
//! - `ascii_render` -- Terminal cross-section visualizations
//!
//! # Validation target
//!
//! Parallel plates: E/A = -pi^2 / (720 a^3) (Casimir 1948).
//!
//! # References
//! - White et al. (2021): EPJ C 81, 677. DOI:10.1140/epjc/s10052-021-09484-z
//! - Gies, Langfeld, Moyaerts (2003): JHEP 0306, 018
//! - Casimir (1948): Proc. Kon. Ned. Akad. Wet. 51, 793

pub mod ascii_render;
pub mod energy;
pub mod geometry;
pub mod vloop;

// Re-export primary types
pub use ascii_render::{AsciiGrid, energy_cross_section};
pub use energy::{
    CasimirEnergyResult, WorldlineCasimirConfig, casimir_energy_at_point,
    casimir_energy_profile, casimir_parallel_plates_exact,
};
pub use geometry::{
    CasimirGeometry, InfiniteCylinder, ParallelPlates, PillarInCavity, SolidSphere,
    SphereInCylinder,
};
pub use vloop::{WorldlineLoop, generate_unit_loop};
