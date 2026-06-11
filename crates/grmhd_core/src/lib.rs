//! grmhd_core: Minimal General-Relativistic MHD solver for CD analysis.
//!
//! A pure Rust GRMHD code for generating controlled MHD turbulence data
//! to feed into the Cayley-Dickson associator pipeline. NOT intended as
//! a production astrophysical code -- optimized for correctness and
//! simplicity over performance.
//!
//! # Physics
//!
//! - Fixed Kerr (or Schwarzschild) metric in Boyer-Lindquist coordinates
//! - Ideal MHD: conservation of mass, energy-momentum, and magnetic flux
//! - No radiation, no resistivity, no viscosity
//!
//! # Numerical Methods
//!
//! - Finite-volume on logr-theta-phi grid (logarithmic radial spacing)
//! - Piecewise-linear reconstruction (minmod limiter)
//! - HLL Riemann solver (robust, first-order at discontinuities)
//! - Constrained transport for divergence-free B (Toth 2000)
//! - RK2 (Heun) time integration
//!
//! # References
//!
//! - Gammie, McKinney & Toth (2003) "HARM: A Numerical Scheme for GRMHD"
//! - Noble et al. (2006) "Primitive variable solvers for conservative GRMHD"

pub mod con2prim;
pub mod cons;
pub mod ct;
#[cfg(feature = "cubecl")]
pub mod cubecl;
pub mod eos;
pub mod evolve;
pub mod flux;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod grid;
pub mod metric;
pub mod prims;
pub mod recon;
pub mod riemann;
pub mod source;
pub mod torus;
#[cfg(feature = "vulkan")]
pub mod vulkan;
