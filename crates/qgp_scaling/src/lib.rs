//! QGP parton energy-loss path-length scaling.
//!
//! Reproduction of Arleo & Falmagne (arXiv:2411.13258, PRD 109 L051503)
//! establishing that parton energy loss in QGP follows epsilon ~ L^beta
//! with beta ~ 1.02, consistent with BDMPS in a longitudinally expanding medium.
//!
//! # Modules
//!
//! - [`nucleus`]: Nuclear density profiles (Pb-208, Au-197, Xe-129)
//! - [`glauber`]: Optical Glauber model (T_AB, Npart, A_perp, L, eccentricity)
//! - [`quenching`]: R_AA model and scaling function f(u,n)
//! - [`epsilon_fit`]: Chi-square epsilon_bar extraction per centrality
//! - [`density_scaling`]: Multi-system fit epsilon_bar = K * (dNch/dy / A_perp) * L^beta
//! - [`v2_relation`]: v2/eccentricity vs d(ln R_AA)/d(ln pT) linear analysis
//! - [`multiplicity`]: Hardcoded dNch/dy tables per collision system and centrality
//! - [`data_tables`]: Published Glauber Npart validation tables

pub mod data_tables;
pub mod density_scaling;
pub mod epsilon_fit;
pub mod glauber;
pub mod multiplicity;
pub mod nucleus;
pub mod quenching;
pub mod v2_relation;
