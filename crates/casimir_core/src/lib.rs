//! Real-scalar Dirichlet worldline interaction energies in 3+1 dimensions.
//!
//! Gaussian centered Brownian polygons sample the free path measure.
//! Distinct-body hit masks implement subtraction of single-body energies.
//! Point and grid evaluations produce loop-center energy density in
//! hbar=c=1 units; the integrated plate estimator produces energy per area.
//! Geometry discretization and integration cutoffs require convergence checks.
//! Single-body self stress and electromagnetic materials require other models.
//!
//! Reference: Gies, Langfeld, Moyaerts, JHEP 0306 (2003) 018,
//! DOI:10.1088/1126-6708/2003/06/018, Eqs. 8, 11, 13 and 44.

pub mod ascii_render;
pub mod energy;
pub mod geometry;
pub mod vloop;

// Re-export primary types
pub use ascii_render::{AsciiGrid, energy_cross_section};
pub use energy::{
    CasimirComputationError, CasimirEnergyField3D, CasimirEnergyResult, WorldlineCasimirConfig,
    casimir_energy_at_point, casimir_energy_field_3d, casimir_energy_profile,
    casimir_parallel_plates_exact, scalar_parallel_plates_exact, scalar_parallel_plates_worldline,
    try_casimir_energy_at_point,
};
pub use geometry::{
    CasimirGeometry, InfiniteCylinder, InteractionGeometryError, ParallelPlates, PillarInCavity,
    SolidSphere, SphereInCylinder,
};
pub use vloop::{WorldlineLoop, generate_unit_loop};
