//! # MaNGA Dark-Matter Hypothesis Experiments
//!
//! Four experiments with ablations testing algebraic dark-matter signatures
//! in MaNGA IFU velocity-residual data.
//!
//! | ID | Hypothesis | Galaxies | Compute |
//! |----|-----------|----------|---------|
//! | H1 | Pre-Fit Injection Recovery (BLOCKING) | 500 x 4\alpha x 5 seeds | ~125 s |
//! | H2 | DC14 Phase-Shift Exclusion Surface | 6992 x 21 \deltax | ~100 s |
//! | H3 | Sign-Split + Mass-Binned Stacking | 2x5 bins + Bonferroni | ~75 s |
//! | H4 | Azimuthal Power Spectrum (conditional H1) | C_l l=2-8, 4 annuli | ~125 s |

pub mod common;
pub mod h1_prefit_injection;
pub mod h2_dc14_exclusion;
pub mod h3_sign_mass_stacking;
pub mod h4_azimuthal_power;
pub mod zd_null_experiment;
