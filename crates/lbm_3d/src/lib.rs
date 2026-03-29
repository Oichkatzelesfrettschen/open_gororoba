//! 3D Lattice Boltzmann Method (D3Q19)
//!
//! Solves the BGK collision operator on a D3Q19 lattice with:
//! - Bounce-back boundary conditions
//! - Spatially-varying viscosity fields
//! - ZPE field coupling via relaxation time modulation

pub mod boundary;
pub mod dm_force;
pub mod lattice;
pub mod mhd;
pub mod pulp_compat;
pub mod solver;
pub mod viscosity_field;
pub mod zpe_injection;
