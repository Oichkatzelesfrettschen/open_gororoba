//! Geophysical dataset providers.
//!
//! Covers Earth and planetary gravity, magnetic field models, and related
//! reference datasets. All providers use the same DatasetProvider pattern
//! as the astrophysical catalogs.

pub mod igrf;
pub mod wmm;
pub mod grace;
pub mod grail;
pub mod jpl_ephemeris;
