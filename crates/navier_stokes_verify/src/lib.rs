// SPDX-License-Identifier: MIT

//! Exact rational verification for the original unforced periodic equation.
//!
//! The supported norm is the homogeneous, normalized-volume H^3 norm on the
//! 2*pi torus. Inputs describe complete finite Fourier polynomials. Certificates
//! apply to specified projected data or explicitly named neighborhoods, subject
//! to the analytic and implementation trust boundary in [`beltrami`]. They do
//! not establish arbitrary-data global regularity.

pub mod beltrami;
pub mod certificate;
pub mod interval;
pub mod majorant;
pub mod report;
pub mod residual;
pub mod tail_bounds;
