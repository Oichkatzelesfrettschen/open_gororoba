//! Geophysical dataset providers.
//!
//! Covers Earth and planetary gravity, magnetic field models, and related
//! reference datasets. All providers use the same DatasetProvider pattern
//! as the astrophysical catalogs.

pub mod de_ephemeris;
pub mod egm2008;
pub mod grace;
pub mod grace_fo;
pub mod grail;
pub mod igrf;
pub mod jpl_ephemeris;
pub mod swarm;
pub mod wmm;

pub use swarm::{
    check_timestamp_monotonicity, parse_swarm_csv, SwarmRecord, SWARM_EXPECTED_COLUMNS,
};
