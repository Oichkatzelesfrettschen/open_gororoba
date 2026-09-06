//! QGP parton energy-loss path-length scaling.
//!
//! Numerical instruments for comparisons with Arleo and Falmagne
//! (arXiv:2212.01324, PRD 109 L051503). Source-model conformance requires
//! identified quenching weights, geometry estimators and reference populations.
//!
//! # Modules
//!
//! - [`nucleus`]: Nuclear density profiles (Pb-208, Au-197, Xe-129)
//! - [`glauber`]: Optical Glauber model (T_AB, Npart, A_perp, L, eccentricity)
//! - [`quenching`]: R_AA model and scaling function f(u,n)
//! - [`epsilon_fit`]: Chi-square epsilon_bar extraction per centrality
//! - [`density_scaling`]: Multi-system fit epsilon_bar = K * (dNch/dy / A_perp) * L^beta
//! - [`directional_path`]: Globally weighted forward-ray path integrals on sampled densities
//! - [`straggling`]: Quantum straggling (Gaussian-smeared R_AA) and precomputed lookup grid
//! - [`v2_relation`]: v2/eccentricity vs d(ln R_AA)/d(ln pT) linear analysis
//! - [`multiplicity`]: Hardcoded dNch/dy tables per collision system and centrality
//! - [`data_tables`]: Published Glauber Npart validation tables
//! - [`competing_models`]: BIC comparison against CUJET3.0 and fractional Langevin
//! - [`critical_point`]: QCD Critical Point (QCP) and Beam Energy Scan (BES) observables
//! - [`flow_cumulants`]: Multi-particle cumulant formulations for extracting flow harmonics (v_n)
//! - [`hydro_wake`]: Hydrodynamic medium response and wake formation
//! - [`nuclear_modification`]: Nuclear modification factor (R_AA) computations

pub mod bdmps_quenching;
pub mod competing_models;
pub mod critical_point;
pub mod data_tables;
pub mod density_scaling;
pub mod directional_path;
pub mod epsilon_fit;
pub mod flow_cumulants;
pub mod fragmentation;
pub mod glauber;
pub mod hydro_wake;
pub mod multiplicity;
pub mod nuclear_modification;
pub mod nucleus;
pub mod quenching;
pub mod straggling;
pub mod v2_relation;
