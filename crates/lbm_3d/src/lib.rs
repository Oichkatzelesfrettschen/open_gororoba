//! 3D Lattice Boltzmann Method (D3Q19)
//!
//! Solves the BGK collision operator on a D3Q19 lattice with:
//! - Bounce-back boundary conditions
//! - Spatially-varying viscosity fields
//! - ZPE field coupling via relaxation time modulation

pub mod lattice;
pub mod solver;
pub mod boundary;
pub mod zpe_injection;
pub mod viscosity_field;
