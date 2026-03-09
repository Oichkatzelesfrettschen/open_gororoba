//! Cosmic ray transport via the Parker Transport Equation (PTE).
//!
//! One-way-coupled to the LBM/MHD heliospheric pipeline:
//!   solar wind velocity -> advection + adiabatic deceleration
//!   B-field -> diffusion tensor K
//!   DM source -> Q_DM injection at fixed rigidity bins

pub mod diffusion;
pub mod grid;
pub mod modulation;
pub mod snapshot;
pub mod solver;
pub mod source;
